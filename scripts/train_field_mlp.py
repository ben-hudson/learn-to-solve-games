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
import os

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from l2s_games.algorithms import ALGORITHMS
from l2s_games.callbacks import EquilibriumRolloutCallback, RolloutBufferCallback, RolloutVizCallback
from l2s_games.data import build_dataset, collate_examples
from l2s_games.dynamics import simulate
from l2s_games.envs import make_game
from l2s_games.models import MLPFieldModel, conditioned_field
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
    p.add_argument("--game", choices=["rps", "symmetric"], default="rps", help="matrix game family to learn")
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
    p.add_argument("--val-every-n-epochs", type=int, default=10, help="run validation (+ rollout viz) every N epochs")
    p.add_argument("--batch", type=int, default=1024, help="minibatch size")
    # dynamics comparison (same knobs as the sandbox)
    p.add_argument("--h", type=float, default=0.1, help="algorithm step size")
    p.add_argument("--n-steps", type=int, default=400, help="iterations per trajectory")
    p.add_argument("--z0", type=float, nargs="+", default=None, help="starting iterate (default: 0.5*lim)")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["projection", "extragradient", "optimistic", "momentum", "consensus"],
        choices=list(ALGORITHMS),
        help="algorithms for the validation residual sweep and the single-instance dynamics plots",
    )
    # on-policy rollout sampling (see rollout_sampling / callbacks.RolloutBufferCallback)
    p.add_argument(
        "--sampling",
        choices=["uniform", "rollout"],
        default="uniform",
        help="'uniform' samples the domain uniformly (baseline); 'rollout' trains on points visited "
        "by rolling out the current learned field (on-policy), refreshed every --refresh-every epochs",
    )
    p.add_argument(
        "--rollout-algo",
        choices=list(ALGORITHMS),
        default="extragradient",
        help="algorithm rolled out on the learned field to generate on-policy points (also used for the "
        "rollout viz); it shapes the sampling distribution -- projection spirals on RPS, extragradient/"
        "consensus converge",
    )
    p.add_argument("--refresh-every", type=int, default=5, help="regenerate the on-policy buffer every N epochs")
    p.add_argument(
        "--blend-uniform-frac", type=float, default=0.3, help="fraction of the buffer drawn uniformly (vs on-policy)"
    )
    p.add_argument("--n-rollout-instances", type=int, default=128, help="instances rolled out per buffer refresh")
    p.add_argument("--n-viz-instances", type=int, default=3, help="held-out instances shown in the rollout viz")
    # logging (ported from train_field_gnn.py): wandb logs viz as images, csv saves them as PNGs to disk
    p.add_argument("--logger", choices=["wandb", "csv"], default="csv", help="metrics/viz sink")
    p.add_argument("--exp", type=str, default=None, help="experiment name (wandb group)")
    return p


def build_logger(args, save_dir):
    """A wandb or csv Lightning logger (replaces the old logger-free run so viz has a sink)."""
    if args.logger == "wandb":
        import wandb
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            experiment=wandb.init(project="learn-to-solve-games", group=args.exp, config=vars(args), dir=save_dir),
            save_dir=save_dir,
        )
    return CSVLogger(save_dir=save_dir)


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
    (train_ds, val_ds, test_ds), normalizer = build_dataset(
        game,
        args.n_instances,
        args.n_val_instances,
        args.n_test_instances,
        args.points_per_instance,
    )
    print(f"train examples: {len(train_ds)}   " f"val examples: {len(val_ds)}   test examples: {len(test_ds)}")

    # train the model; train/val loss is shown on the progress bar
    model = MLPFieldModel(
        in_features=game.domain_dim + game.n_params,
        hidden=args.hidden,
        out_features=game.domain_dim,
        lr=args.lr,
        normalizer=normalizer,
    )
    # Each validation epoch sweeps the algorithms, rolling out the learned field batched over the
    # held-out instances and logging the analytic residual at the endpoint per algorithm.
    callbacks = [EquilibriumRolloutCallback(game, name, args.n_steps, args.h) for name in args.algorithms]

    save_dir = os.getenv("SCRATCH", ".")
    logger = build_logger(args, save_dir)

    # On-policy sampling: refresh the training buffer from the current field every --refresh-every
    # epochs, and log the rollout / true-vs-learned field viz through training. Epoch 0 trains on the
    # uniform buffer build_dataset produced (below); the buffer callback takes over from --refresh-every.
    if args.sampling == "rollout":
        rollout_instances = [game.sample_params() for _ in range(args.n_rollout_instances)]
        viz_instances = [game.sample_params() for _ in range(args.n_viz_instances)]
        # The on-policy buffer keeps the same point count as the initial uniform buffer build_dataset
        # made (n_instances * points_per_instance), so every epoch has the same number of batches.
        # If the refreshed epochs were shorter, Lightning -- which fixes val_check_batch from epoch 0 --
        # would never reach the validation batch and would silently skip validation entirely.
        buffer_size = args.n_instances * args.points_per_instance
        callbacks.append(
            RolloutBufferCallback(
                game, train_ds, normalizer, rollout_instances, args.rollout_algo, args.h, args.n_steps,
                buffer_size, args.blend_uniform_frac, args.refresh_every,
            )
        )
        if game.domain_dim == 2:
            callbacks.append(
                RolloutVizCallback(
                    game, normalizer, viz_instances, args.rollout_algo, args.h, args.n_steps, save_dir,
                )
            )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        num_sanity_val_steps=0,
        logger=logger,
        default_root_dir=save_dir,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_every_n_epochs,
        inference_mode=False,  # validation rolls out consensus, whose grad term needs autograd
    )
    collate = collate_examples(game)
    trainer.fit(
        model,
        DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate),
        DataLoader(val_ds, batch_size=args.batch, collate_fn=collate),
    )

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
