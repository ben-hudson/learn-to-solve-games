"""
train_field_simple.py

Amortize a traffic operator from a **cached** dataset root: ordinary supervised regression over a finite,
indexable dataset. One loader, no blending, no operator budget to account for -- everything about *where the
training data comes from* was decided at generation time by
``scripts/generate_traffic_dataset.py``, so this script only says how to fit a model to it.

    python scripts/train_field_simple.py --dataset_root data/sioux_falls/solved

``--operator_dataset`` names the point source, the same word the generation script uses: ``uniform`` (points
spread over the calibrated cost box) or ``expert`` (points along a converging rollout of the true operator,
plus the equilibrium it reaches). A root may hold both, over identical instances, which is what makes the
comparison between them clean -- so this is a genuine choice here rather than a property of the root.

``--amortization`` picks what is learned. ``partial`` (default) learns the operator *field*, which a solver
then rolls out to reach an equilibrium (``--algos`` scores that each validation epoch). ``full`` learns the
equilibrium ``z*`` directly from parameters, with no rollout at inference; it reads only the cached equilibria,
so it ignores ``--operator_dataset`` and never loads an example tensor.

Nothing about the *operator* is passed in. The family comes off the root's ``base_graph.game`` plus whatever
else rides on that graph (the asymmetric coupling matrices), so a run cannot be conditioned on a different
operator than the one its equilibria were solved for. The dataset root therefore identifies the operator, and
``--game`` / ``--noise_scale`` / ``--sample_stds`` / ``--n_cal_instances`` are all generation-time flags.

Contrast ``train_field_gnn.py``, which generates its data at train time from four blendable streaming sources
and needs 53 flags to say so.
"""

import argparse
import functools
import os

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Subset

from l2s_games.algorithms import ALGORITHMS
from l2s_games.callbacks import FieldRolloutCallback, SolutionPredictionCallback
from l2s_games.data import (
    GlobalStandardizer,
    Standardizer,
    collate_normalized_examples,
    fit_normalizer,
    normalize_example,
    solution_examples,
)
from l2s_games.equilibrium_datasets import EquilibriumDataset
from l2s_games.models import FieldModel, GraphormerBackbone, MLPBackbone, SolutionModel
from l2s_games.operator_datasets import OPERATOR_DATASETS, root_family

torch.set_float32_matmul_precision("medium")


def _single_thread_worker(_worker_id):
    """Single-thread each worker: N multi-threaded workers oversubscribe the cores and thrash, causing
    bursty batch delivery. A module-level function (not a local lambda) so it is picklable under the
    ``spawn`` start method (macOS)."""
    torch.set_num_threads(1)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data: everything comes from the root; these only choose which cache to read and how to split it.
    p.add_argument("--dataset_root", type=str, required=True, help="root of the cached dataset to train on")
    p.add_argument(
        "--operator_dataset",
        choices=list(OPERATOR_DATASETS),
        default="uniform",
        help="which point source's cache to read (a root may hold both). Ignored under --amortization full, "
        "which needs only the equilibria",
    )
    p.add_argument("--n_val_instances", type=int, default=128, help="validation split size, in instances")
    p.add_argument("--n_test_instances", type=int, default=128, help="held-out test split size, in instances")
    # The dataset is an in-memory store, so every worker gets a *copy* of it under spawn: resident memory
    # scales with workers x store (~45 MB for 1024x32 uniform, ~470 MB for 1024x501 expert). Per-example work
    # is one line-graph transform (~0.7 ms), so 2 workers already hide most of it; raise it on narrow roots.
    p.add_argument("--n_workers", type=int, default=2, help="dataloader workers (memory scales with workers)")
    p.add_argument("--seed", type=int, default=None, help="global seed")
    p.add_argument(
        "--amortization",
        choices=["partial", "full"],
        default="partial",
        help="'partial' learns the operator field (rolled out to solve); 'full' predicts the equilibrium z* "
        "directly from parameters (no rollout)",
    )
    # model
    p.add_argument(
        "--model",
        choices=["graphormer", "mlp"],
        default="graphormer",
        help="architecture: 'graphormer' (line-graph attention) or 'mlp' (whole-graph flat baseline: the "
        "fixed network's per-edge feats are flattened and every edge predicted jointly, no graph bias)",
    )
    p.add_argument("--dim", type=int, default=128, help="hidden dim")
    p.add_argument("--n_heads", type=int, default=8, help="attention heads (graphormer only)")
    p.add_argument("--n_layers", type=int, default=6, help="layers")
    p.add_argument("--dim_ff", type=int, default=512, help="feed-forward dim (graphormer only)")
    # loss: the model predicts in the normalizer's target space (a global scale, no warp), so "mse" compares
    # there and the others compare in real units *in scale units* -- the norm the rollout-residual bound
    # controls. 'huber' is the stable real-space default; 'rel_l2' is FNO-style per-sample relative L2.
    p.add_argument("--loss", choices=["mse", "huber", "rel_l2"], default="mse", help="training loss norm")
    p.add_argument("--huber_delta_scale", type=float, default=1.0, help="huber knee, in scale units")
    p.add_argument("--rel_eps", type=float, default=1.0, help="rel_l2 denominator floor, in scale units")
    # training (AdamW + linear-warmup -> cosine)
    p.add_argument("--lr", type=float, default=0.0012, help="AdamW learning rate")
    p.add_argument("--start_factor", type=float, default=0.011, help="linear warmup start factor")
    p.add_argument("--warmup_epochs", type=int, default=10, help="linear warmup epochs (must be < --epochs)")
    p.add_argument("--cosine_annealing", type=int, default=0, help="cosine-anneal after warmup (0 disables)")
    p.add_argument("--gradient_clip_val", type=float, default=7, help="gradient-norm clip value")
    # An epoch is a real pass over the whole train split here, not --steps_per_epoch batches of a stream, so
    # these are an order of magnitude smaller than train_field_gnn.py's.
    p.add_argument("--epochs", type=int, default=200, help="training epochs (a full pass each)")
    p.add_argument("--batch_size", type=int, default=128, help="minibatch size")
    p.add_argument("--patience_epochs", type=int, default=40, help="early-stopping patience in epochs")
    p.add_argument("--val_every_n_epochs", type=int, default=10, help="run validation every N epochs")
    # validation equilibrium sweep (ignored under --amortization full, which predicts z* directly)
    p.add_argument(
        "--algos",
        nargs="*",
        choices=list(ALGORITHMS),
        default=["projection", "consensus"],
        help="dynamics algorithms rolled out on the learned field each val epoch, logging the analytic "
        "endpoint residual val/{algo}/residual (pass with no value for fast field-only training)",
    )
    p.add_argument("--h", type=float, default=0.1, help="rollout step size (see scripts/tune_pume_rollout.py)")
    p.add_argument("--n_steps", type=int, default=500, help="rollout iterations")
    # logging
    p.add_argument("--logger", choices=["wandb", "csv"], default="wandb", help="where to log metrics")
    p.add_argument("--exp", type=str, default=None, help="experiment name (wandb group)")
    p.add_argument("--debug", action="store_true", help="run a single train/val batch for quick checking")
    return p


def build_splits(dataset, family, args):
    """``(train, val, targets, target_scaler)`` for the chosen amortization -- the only branch in the script.

    Both modes serve **raw** examples (real units, unfeaturized), so the feats half of the normalizer fit and
    the collate are the same expression either way; what differs is where the examples come from, how the
    split is expressed, and which target scale ``z*`` versus a field wants.

    Under ``full`` the labels are the *cached* equilibria, so a solution model costs no operator evaluations
    at all -- where the old expert solution stream spent ``n_steps`` per label.
    """
    if args.amortization == "full":
        # EquilibriumDataset indexes instances, so plain len() already means instances here.
        instances = list(dataset)
        n_train = len(instances) - args.n_val_instances - args.n_test_instances
        assert n_train > 0, f"{len(instances)} instances is too few for the requested val/test splits"
        train, val = (
            solution_examples(family, part)
            for part in (instances[:n_train], instances[n_train : n_train + args.n_val_instances])
        )
        # z* is a generic per-feature regression target, not a field: no global scale and no warp.
        return train, val, torch.stack([target for _, target in train]), Standardizer.fit

    # An OperatorDataset counts *examples* in len() and instances in len(); both are wanted below.
    ppi = dataset.points_per_instance
    n_train = dataset.len() - args.n_val_instances - args.n_test_instances
    assert n_train > 0, f"{dataset.len()} instances is too few for the requested val/test splits"
    # Contiguous, so each Subset holds whole instances (see OperatorDataset). The test instances are the
    # reserved tail; nothing is built for them.
    train = Subset(dataset, range(0, n_train * ppi))
    # One point per val instance, by the ppi stride: FieldRolloutCallback rolls out --n_steps on *every* val
    # example, and the endpoint residual is a function of the instance alone, so the rest would only pay for
    # repeated rollouts. Index 0 of a block is the uniform draw (uniform) or the rollout start (expert).
    val = Subset(dataset, range(n_train * ppi, (n_train + args.n_val_instances) * ppi, ppi))
    targets = torch.cat([dataset.evaluations(i).targets for i in range(n_train)])
    return train, val, targets, functools.partial(GlobalStandardizer.fit, center=False)


def build_backbone(sample, args):
    """Size the network from one normalized example: feature width plus the line-graph structure."""
    if args.model == "graphormer":
        return GraphormerBackbone(
            n_feats=sample["feats"].shape[-1],
            in_degree=sample["in_degree"],
            out_degree=sample["out_degree"],
            spd=sample["spd"],
            dim=args.dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_ff=args.dim_ff,
            dropout=0.0,
        )
    # The whole-graph flat baseline: flatten the fixed network's per-edge feats [E, k] and predict every edge.
    return MLPBackbone(
        in_features=sample["feats"].numel(),
        hidden=[args.dim] * args.n_layers,
        out_features=sample["feats"].shape[0],
        flatten_start_dim=1,
    )


def main(args):
    if args.seed is None:
        args.seed = torch.randint(0, 2**31 - 1, (1,)).item()
    L.seed_everything(args.seed, workers=True)
    # The root decides the operator: `full` wants only the equilibria, so it reads the raw-only base class
    # (which ignores any example tensors the root happens to carry).
    dataset = (
        EquilibriumDataset(args.dataset_root)
        if args.amortization == "full"
        else OPERATOR_DATASETS[args.operator_dataset](args.dataset_root)
    )
    family = root_family(dataset.base_graph)
    train, val, targets, target_scaler = build_splits(dataset, family, args)
    # Fit on the train split only, over every one of its examples. Feats are streamed rather than stacked:
    # they have to be built (one transform each), and the stacked population would be gigabytes on a wide
    # root. Passing both populations here is what keeps the fit-on-train choice visible.
    normalizer = fit_normalizer((family.transform(item)["feats"] for item, _ in train), targets, target_scaler)
    print(f"train: {len(train)} examples   val: {len(val)}   game: {dataset.base_graph.game}")

    sample, _target = normalize_example(*train[0], family.transform, normalizer)
    net = build_backbone(sample, args)
    train_kwargs = dict(
        lr=args.lr,
        normalizer=normalizer,
        weight_decay=0.0,
        start_factor=args.start_factor,
        warmup_epochs=args.warmup_epochs,
        cosine_annealing=bool(args.cosine_annealing),
    )
    if args.amortization == "full":
        model = SolutionModel(net, **train_kwargs)
        callbacks = [SolutionPredictionCallback(family)]
    else:
        # "mse" is the target-space MSE: with no target warp, FieldModel's mse_target and mse_real coincide,
        # so only one of the two names is worth exposing.
        model = FieldModel(
            net,
            **train_kwargs,
            loss="mse_target" if args.loss == "mse" else args.loss,
            huber_delta_scale=args.huber_delta_scale,
            rel_eps=args.rel_eps,
        )
        callbacks = [FieldRolloutCallback(family, name, args.n_steps, args.h) for name in args.algos]
    callbacks.append(
        EarlyStopping(
            monitor="val/mse",
            mode="min",
            patience=max(1, args.patience_epochs // args.val_every_n_epochs),
            check_finite=False,
            strict=False,
        )
    )

    save_dir = os.getenv("SCRATCH", ".")
    if args.debug:
        logger = None
    elif args.logger == "wandb":
        # The root's own facts are logged beside the flags: nothing in vars(args) says which operator or how
        # many examples a run trained on, and operator_evals is the ground-truth budget -- one evaluation per
        # example by construction, known before training starts, so no counter callback is needed.
        config = {**vars(args), "game": dataset.base_graph.game, "n_train_examples": len(train)}
        if args.amortization == "partial":
            config |= {"points_per_instance": dataset.points_per_instance, "operator_evals": len(train)}
        run = wandb.init(project="learn-to-solve-games", group=args.exp, config=config, dir=save_dir)
        logger = WandbLogger(experiment=run, save_dir=save_dir)
    else:
        logger = CSVLogger(save_dir=save_dir)
    if logger is not None:
        callbacks.append(ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=1, save_last=True, filename="best"))

    collate = collate_normalized_examples(family, normalizer)
    loader = functools.partial(
        DataLoader,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.n_workers,
        persistent_workers=args.n_workers > 0,
        worker_init_fn=_single_thread_worker,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        logger=logger,
        default_root_dir=save_dir,
        fast_dev_run=args.debug,
        enable_checkpointing=logger is not None,
        enable_model_summary=False,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val or None,
        check_val_every_n_epoch=args.val_every_n_epochs,
        # Only the partial-mode consensus rollout's jacrev needs autograd in validation.
        inference_mode=args.amortization == "full" or "consensus" not in args.algos,
    )
    trainer.fit(
        model,
        # Shuffled because consecutive examples share an instance: an unshuffled batch would hold a handful
        # of parametrizations. The named mapping matches training_step's contract (batch = {source: batch}).
        {"train": loader(train, shuffle=True)},
        loader(val),
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
