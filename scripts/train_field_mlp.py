"""
train_field_mlp.py

Train a simple MLP to predict a game's operator field -- amortized over a *family*
of games chosen with ``--game``. The MLP maps ``(coordinates, instance params) ->
operator value at that coordinate``, so one network represents every parametrization
in the family.

An *instance* is one real-unit ``params`` value drawn from the family. Inputs and targets
are standardized by a normalizer fit on the train split (see ``data.Normalizer``); the
learned field is mapped back to real units for comparison. For a 2D-domain game we compare
the learned model to the analytic operator, both as a field (quiver + error) and as a
dynamical system (the same algorithms rolled out on each).

    python scripts/train_field_mlp.py --game rps --epochs 60 --hidden 128 128
"""

import argparse

import lightning as L
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import build_dataset
from l2s_games.dynamics import simulate
from l2s_games.envs import GAMES, make_game
from torchvision.ops import MLP

from l2s_games.models import FieldLitModule, conditioned_field
from l2s_games.viz import overlay_trajectory, plot_field_quiver


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # game + dataset
    p.add_argument("--game", choices=list(GAMES), default="toy", help="game family to learn")
    p.add_argument("--n-actions", type=int, default=3, help="actions per population (symmetric game)")
    p.add_argument("--n-instances", type=int, default=256, help="training instances")
    p.add_argument("--n-val-instances", type=int, default=64, help="validation instances")
    p.add_argument("--n-test-instances", type=int, default=64, help="held-out test instances")
    p.add_argument("--points-per-instance", type=int, default=256, help="samples per instance")
    p.add_argument("--seed", type=int, default=None, help="global seed (None -> random)")
    # model + training
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128], help="hidden layer widths")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--epochs", type=int, default=60, help="training epochs")
    p.add_argument("--batch", type=int, default=1024, help="minibatch size")
    # dynamics comparison (same knobs as the sandbox)
    p.add_argument("--h", type=float, default=0.1, help="algorithm step size")
    p.add_argument("--n-steps", type=int, default=400, help="iterations per trajectory")
    p.add_argument("--z0", type=float, nargs="+", default=None, help="starting iterate (default: 0.5*lim)")
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
def relative_error(model, test_ds, normalizer):
    """Relative error in real units (predictions and targets de-standardized)."""
    inputs, targets = test_ds.tensors
    with torch.no_grad():
        preds = normalizer.target.inverse_transform(model(inputs))
    targets = normalizer.target.inverse_transform(targets)
    return (torch.linalg.norm(preds - targets) / torch.linalg.norm(targets)).item()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_field_comparison(true_field, learned_field, instance, lim, grid=21):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    summary = ", ".join(f"p{i}={value:.2f}" for i, value in enumerate(instance.tolist()))
    fig.suptitle(f"held-out instance: {summary}", fontsize=12)
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
    game = make_game(args.game, n_actions=args.n_actions) if args.game == "symmetric" else make_game(args.game)
    if not hasattr(game, "domain_dim"):
        print(f"'{args.game}' is not a flat game; the MLP trainer needs the GNN path (follow-up).")
        return
    (train_ds, val_ds, test_ds), normalizer = build_dataset(
        game,
        args.n_instances,
        args.n_val_instances,
        args.n_test_instances,
        args.points_per_instance,
    )
    print(f"train examples: {len(train_ds)}   " f"val examples: {len(val_ds)}   test examples: {len(test_ds)}")

    # train the model; train/val loss is shown on the progress bar
    model = MLP(
        in_channels=game.domain_dim + game.n_params,
        hidden_channels=[*args.hidden, game.domain_dim],
        activation_layer=torch.nn.Tanh,
    )
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

    print(f"test relative error = {relative_error(model, test_ds, normalizer):.4%}")

    params = game.sample_params()
    summary = ", ".join(f"p{i}={value:.3f}" for i, value in enumerate(params.tolist()))
    print(f"eval instance (real units) = ({summary})")

    def true_field(z):
        return game.operator(params, z)

    learned_field = conditioned_field(model, game, params, normalizer)
    z0 = args.z0 if args.z0 is not None else [0.5 * game.lim] * game.domain_dim

    if game.domain_dim == 2:
        plot_field_comparison(true_field, learned_field, params, game.lim)
        plot_dynamics_comparison(true_field, learned_field, args.algorithms, args.h, z0, args.n_steps, game.lim)
        plt.show()
    else:
        print(f"domain_dim={game.domain_dim}; skipping 2D comparison plots")


if __name__ == "__main__":
    main(build_parser().parse_args())
