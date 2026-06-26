"""
train_field_mlp.py

Train a simple MLP to predict the rotational vector field from
``rotational_field_sandbox.py`` -- but amortized over a *family* of fields. The
MLP maps ``(coordinates, instance params) -> field value at that coordinate``, so
one network represents every parametrization in the family.

An *instance* is a normalized vector ``p in [0, 1]^k`` selecting (omega,
damp_floor, damp_wall); ``--ranges`` maps it to field arguments. ``curl_nonlin``
and ``well_angle`` are fixed inside ``make_field``. We then compare the learned
model to the analytic field, both as a field (quiver + error) and as a dynamical
system (the same algorithms rolled out on each).

    python scripts/train_field_mlp.py --epochs 60 --hidden 128 128
"""

import argparse

import lightning as L
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import build_dataset
from l2s_games.dynamics import simulate
from l2s_games.envs.toy import make_field
from l2s_games.models import FieldLitModule, FieldMLP, conditioned_field
from l2s_games.viz import overlay_trajectory, plot_field_quiver


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # dataset
    p.add_argument("--n-instances", type=int, default=256, help="training instances")
    p.add_argument("--n-val-instances", type=int, default=64, help="validation instances")
    p.add_argument("--n-test-instances", type=int, default=64, help="held-out test instances")
    p.add_argument("--points-per-instance", type=int, default=256, help="samples per instance")
    p.add_argument("--lim", type=float, default=2.0, help="points sampled in [-lim, lim]^2")
    p.add_argument("--seed", type=int, default=None, help="global seed (None -> random)")
    p.add_argument(
        "--ranges",
        type=float,
        nargs=6,
        default=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        metavar=("OMEGA_LO", "OMEGA_HI", "FLOOR_LO", "FLOOR_HI", "WALL_LO", "WALL_HI"),
        help="(low high) range per varying param: omega, damp_floor, damp_wall",
    )
    # model + training
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128], help="hidden layer widths")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--epochs", type=int, default=60, help="training epochs")
    p.add_argument("--batch", type=int, default=1024, help="minibatch size")
    # dynamics comparison (same knobs as the sandbox)
    p.add_argument("--h", type=float, default=0.1, help="algorithm step size")
    p.add_argument("--n-steps", type=int, default=400, help="iterations per trajectory")
    p.add_argument("--z0", type=float, nargs=2, default=[1.0, 1.0], help="starting iterate")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS),
        choices=list(ALGORITHMS),
        help="algorithms to compare",
    )
    return p


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def relative_error(model, test_ds):
    inputs, targets = test_ds.tensors
    with torch.no_grad():
        preds = model(inputs)
    return (torch.linalg.norm(preds - targets) / torch.linalg.norm(targets)).item()


def sample_test_instance(test_ds):
    """Pick a random held-out instance (the param slice of a random example)."""
    inputs = test_ds.tensors[0]
    idx = torch.randint(len(inputs), (1,)).item()
    return inputs[idx, 2:]  # columns 2: are the instance params


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_field_comparison(true_field, learned_field, instance, lim, grid=21):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    omega, floor, wall = instance.tolist()
    fig.suptitle(f"held-out instance: omega={omega:.2f}, floor={floor:.2f}, wall={wall:.2f}", fontsize=12)
    plot_field_quiver(axes[0], true_field, lim=lim, grid=grid, title="true field")
    plot_field_quiver(axes[1], learned_field, lim=lim, grid=grid, title="learned field")

    xs = torch.linspace(-lim, lim, grid)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    Z = torch.stack([X, Y], dim=-1)
    with torch.no_grad():
        err = torch.linalg.norm(learned_field(Z) - true_field(Z), dim=-1)
    mesh = axes[2].pcolormesh(X, Y, err, cmap="magma", shading="auto")
    axes[2].set_aspect("equal")
    axes[2].set_title(r"$\|\hat v - v\|$", fontsize=11)
    axes[2].set_xlabel(r"$\theta$")
    axes[2].set_ylabel(r"$\psi$")
    fig.colorbar(mesh, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("field_comparison.png", dpi=130)


def plot_dynamics_comparison(true_field, learned_field, algorithms, h, z0, n_steps, lim):
    cols = 3
    rows = -(-len(algorithms) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = axes.ravel()
    print("\nFinal ||z||  (true vs learned field):")
    for ax, name in zip(axes, algorithms):
        true_traj = simulate(true_field, ALGORITHMS[name](h), z0, n_steps)
        learned_traj = simulate(learned_field, ALGORITHMS[name](h), z0, n_steps)
        plot_field_quiver(ax, true_field, lim=lim, title=name)
        overlay_trajectory(ax, true_traj, color="crimson", label="true")
        overlay_trajectory(ax, learned_traj, color="dodgerblue", label="learned")
        ax.legend(fontsize=8, loc="upper right")
        t, l = torch.linalg.norm(true_traj[-1]), torch.linalg.norm(learned_traj[-1])
        print(f"  {name:14s} true={t:.4e}  learned={l:.4e}")
    for ax in axes[len(algorithms) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("field_dynamics_comparison.png", dpi=130)


# --------------------------------------------------------------------------
def main(args):
    L.seed_everything(args.seed)
    ranges = list(zip(args.ranges[::2], args.ranges[1::2]))
    train_ds, val_ds, test_ds = build_dataset(
        args.n_instances,
        args.n_val_instances,
        args.n_test_instances,
        args.points_per_instance,
        args.lim,
        ranges,
    )
    print(f"train examples: {len(train_ds)}   " f"val examples: {len(val_ds)}   test examples: {len(test_ds)}")

    # train the model; train/val loss is shown on the progress bar
    model = FieldMLP(in_dim=2 + len(ranges), out_dim=2, hidden=tuple(args.hidden))
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        FieldLitModule(model, args.lr),
        DataLoader(train_ds, batch_size=args.batch, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch),
    )

    print(f"test relative error = {relative_error(model, test_ds):.4%}")

    p_eval = sample_test_instance(test_ds)
    omega, floor, wall = p_eval.tolist()
    print(f"eval instance (normalized omega, floor, wall) = ({omega:.3f}, {floor:.3f}, {wall:.3f})")

    true_field = make_field(p_eval, ranges)
    learned_field = conditioned_field(model, p_eval)

    plot_field_comparison(true_field, learned_field, p_eval, args.lim)
    plot_dynamics_comparison(true_field, learned_field, args.algorithms, args.h, args.z0, args.n_steps, args.lim)
    plt.show()


if __name__ == "__main__":
    main(build_parser().parse_args())
