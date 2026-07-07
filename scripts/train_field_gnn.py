"""
train_field_gnn.py

Amortize the traffic **operator field** with a Graphormer over the line graph. The model maps
each road edge's features ``[cost, free_flow_time, capacity, b, power, +4 demand features]`` plus
the line-graph structure to the per-edge operator value ``costs - bpr(demand_flow(-costs))`` -- so
one network represents the operator across a family of noised SiouxFalls instances.

``--model`` selects the architecture: ``graphormer`` (line-graph attention) or ``mlp`` -- a
whole-graph flat baseline that flattens the fixed network's per-edge feats and predicts every edge
jointly (no graph inductive bias), for benchmarking how much the Graphormer's structure buys.

Training data is **streamed**: every step draws a fresh instance and solves the operator jointly for
``--points_per_instance`` cost points inside ``DataLoader`` workers (one solve amortized over that
many training examples) -- so the model sees unbounded instance diversity rather than a fixed set
(see ``data.build_streaming_dataset`` / ``StreamingFieldDataset``).
The normalizer is fit once on a fixed bootstrap set (``--bootstrap_instances``); val/test stay fixed.

Each validation epoch logs, over the held-out validation set, the field relative error plus -- for
every algorithm in ``--algos`` -- the analytic residual ``||costs - bpr(demand_flow(-costs))||`` at
the endpoint of a projected rollout of that algorithm on the learned field. The whole val batch of
instances is solved at once (see ``FieldModel.batched_field`` / ``MarkovTrafficEquilibrium.operator``).

    python scripts/train_field_gnn.py --n_workers 4
"""

import argparse
import functools
import os

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from l2s_games.algorithms import ALGORITHMS
from l2s_games.callbacks import EquilibriumRolloutCallback
from l2s_games.data import build_streaming_dataset, collate_examples, split_instances
from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs.traffic import MarkovTrafficEquilibrium
from l2s_games.models import GraphormerFieldModel, MLPFieldModel

torch.set_float32_matmul_precision("medium")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dataset: a cached SolvedInstanceDataset (see scripts/generate_traffic_dataset.py) is loaded and
    # split into bootstrap/val/test instances. Training still streams fresh instances on the fly (one
    # point each); the splits fit the normalizer + calibrate the sampling range (bootstrap) and measure
    # generalization (val/test). Epoch length is --steps_per_epoch.
    p.add_argument("--dataset_root", type=str, required=True, help="root of the cached SolvedInstanceDataset to load")
    p.add_argument(
        "--bootstrap_instances", type=int, default=128, help="bootstrap split size (normalizer + calibration)"
    )
    p.add_argument("--n_val_instances", type=int, default=128, help="validation split size")
    p.add_argument("--n_test_instances", type=int, default=128, help="held-out test split size")
    p.add_argument(
        "--points_per_instance",
        type=int,
        default=32,
        help="cost points solved jointly per streamed train instance (also bootstrap density for the "
        "normalizer fit); val/test always solve each instance once",
    )
    p.add_argument(
        "--n_workers", type=int, default=7, help="streaming dataloader workers (0 = serial; changes the stream)"
    )
    p.add_argument("--noise_scale", type=float, default=0.2, help="multiplicative attribute noise")
    p.add_argument("--seed", type=int, default=None, help="global seed")
    # domain coverage (see MarkovTrafficEquilibrium.sample_domain): the range is calibrated from the
    # bootstrap split's equilibria -- per-edge mean (center) and std (spread) -- and sampled within
    # --sample_stds sigma of that mean. --equilibrium_margin/--equilibrium_spread are the uncalibrated
    # fallback only (used when a family is built without a calibrated range, e.g. the sandbox).
    p.add_argument(
        "--sample_stds", type=float, default=3.0, help="sigma radius of the equilibrium ball sample_domain draws from"
    )
    p.add_argument(
        "--equilibrium_margin",
        type=float,
        default=2.5,
        help="uncalibrated fallback: reference-equilibrium ceiling widen",
    )
    p.add_argument("--equilibrium_spread", type=float, default=0.2, help="uncalibrated fallback: multiplicative spread")
    # model
    p.add_argument(
        "--model",
        choices=["graphormer", "mlp"],
        default="graphormer",
        help="architecture: 'graphormer' (line-graph attention) or 'mlp' (whole-graph flat baseline: "
        "the fixed network's per-edge feats are flattened and every edge is predicted jointly, no "
        "graph inductive bias)",
    )
    p.add_argument("--dim", type=int, default=128, help="hidden dim")
    p.add_argument("--n_heads", type=int, default=4, help="attention heads (graphormer only)")
    p.add_argument("--n_layers", type=int, default=6, help="layers")
    p.add_argument("--dim_ff", type=int, default=256, help="feed-forward dim (graphormer only)")
    # No --dropout / --weight_decay: the streaming pipeline sees a fresh instance every step, so it
    # can't overfit -- both are hardcoded to 0 (no regularization) at model construction.
    # training (AdamW + linear-warmup->cosine, ported from markov-traffic-eq)
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--start_factor", type=float, default=0.01, help="linear warmup start factor")
    p.add_argument("--warmup_epochs", type=int, default=50, help="linear warmup epochs (must be < --epochs)")
    p.add_argument("--cosine_annealing", type=int, default=1, help="cosine-anneal after warmup (0 disables)")
    p.add_argument("--gradient_clip_val", type=float, default=1.0, help="gradient-norm clip value")
    p.add_argument("--epochs", type=int, default=400, help="training epochs")
    p.add_argument(
        "--steps_per_epoch", type=int, default=64, help="train batches per epoch (bounds the infinite stream)"
    )
    p.add_argument("--batch", type=int, default=128, help="minibatch size")
    # early stopping on the field relative error (no MAPE here -- we regress the operator field)
    p.add_argument("--patience_epochs", type=int, default=40, help="early-stopping patience in epochs")
    p.add_argument("--val_every_n_epochs", type=int, default=10, help="run validation every N epochs")
    # logging (ported from markov-traffic-eq/scripts/self_supervised.py)
    p.add_argument(
        "--logger",
        choices=["wandb", "csv"],
        default="wandb",
        help="where to log metrics ('csv' writes to {SCRATCH or .}/csvlogs and skips wandb)",
    )
    p.add_argument("--exp", type=str, default=None, help="experiment name (wandb group)")
    p.add_argument("--debug", action="store_true", help="run a single train/val batch for quick sanity checking")
    # validation equilibrium sweep (rollout on the learned field, per algorithm)
    p.add_argument(
        "--algos",
        nargs="*",
        choices=list(ALGORITHMS),
        default=["projection"],
        help="dynamics algorithms rolled out on the learned field each val epoch, logging the analytic "
        "endpoint residual val/{algo}/residual (pass --algos with no value for fast field-only training)",
    )
    # h=0.05 sits above the stability threshold for the stiffer instances -- simGD then oscillates
    # at a ~8e-2 residual floor even on the true operator; h=0.02/1000 converges to ~1e-6 on every
    # val instance, so the learned-field residual is measured against a reachable target.
    p.add_argument("--h", type=float, default=0.02, help="algorithm step size (damped fixed point)")
    p.add_argument("--n_steps", type=int, default=1000, help="iterations for the rollout")
    return p


def main(args):
    # workers=True makes Lightning seed each streaming dataloader worker distinctly & reproducibly.
    if args.seed is None:
        args.seed = torch.randint(0, 2**31 - 1, (1,)).item()
    L.seed_everything(args.seed, workers=True)
    # Load the cached solved instances and split them into bootstrap/val/test. The bootstrap split's
    # equilibria calibrate the streaming sampling range (per-edge mean + std); the calibrated tensors
    # are baked into the picklable factory so every worker shares the same range.
    dataset = SolvedInstanceDataset(args.dataset_root)
    instances = list(dataset)
    bootstrap_inst, val_inst, test_inst = split_instances(
        instances, (args.bootstrap_instances, args.n_val_instances, args.n_test_instances)
    )
    reference_equilibrium, reference_spread = MarkovTrafficEquilibrium.calibrate_range(bootstrap_inst)
    # A picklable factory (base graph + calibrated tensors) the streaming dataset ships to each worker,
    # which builds its own family + route-choice solver lazily -- nothing solver-related is pickled. The
    # main process also needs one live family for collate_fn and the validation rollout callbacks.
    family_factory = functools.partial(
        MarkovTrafficEquilibrium,
        dataset.base_graph,
        noise_scale=args.noise_scale,
        reference_equilibrium=reference_equilibrium,
        reference_spread=reference_spread,
        n_stds=args.sample_stds,
    )
    family = family_factory()
    (train_ds, val_ds, test_ds, bootstrap_ds), normalizer = build_streaming_dataset(
        family_factory,
        bootstrap_inst,
        val_inst,
        test_inst,
        args.points_per_instance,
    )
    print(f"streaming train   bootstrap: {len(bootstrap_ds)}   val: {len(val_ds)}   test: {len(test_ds)}")

    # Size the model from one transformed bootstrap example (line-graph structure + feature width);
    # the train stream is iterable, so it cannot be indexed.
    sample, _ = bootstrap_ds[0]
    if args.model == "graphormer":
        model = GraphormerFieldModel(
            n_feats=sample["feats"].shape[-1],
            in_degree=sample["in_degree"],
            out_degree=sample["out_degree"],
            spd=sample["spd"],
            lr=args.lr,
            normalizer=normalizer,
            weight_decay=0.0,
            start_factor=args.start_factor,
            warmup_epochs=args.warmup_epochs,
            cosine_annealing=bool(args.cosine_annealing),
            dim=args.dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_ff=args.dim_ff,
            dropout=0.0,
        )
    else:
        # Whole-graph flat baseline: flatten the fixed network's per-edge feats [E, k] to one vector
        # and predict every edge jointly (out_features = E), reusing --dim / --n_layers for a uniform
        # hidden stack.
        model = MLPFieldModel(
            in_features=sample["feats"].numel(),  # E * k
            hidden=[args.dim] * args.n_layers,
            out_features=sample["feats"].shape[0],  # E
            flatten_start_dim=1,  # collapse per-edge feats [B, E, k] -> [B, E*k]
            lr=args.lr,
            normalizer=normalizer,
            weight_decay=0.0,
            start_factor=args.start_factor,
            warmup_epochs=args.warmup_epochs,
            cosine_annealing=bool(args.cosine_annealing),
        )
    collate = collate_examples(family)
    # Each validation epoch rolls out the learned field per algorithm on the held-out val batch and
    # logs the analytic residual at the endpoint (plus train/val_mse and train/val_rel_err). Pass no
    # --algos to skip the sweep for fast field-only tuning (rel_err metrics still logged).
    rollouts = [EquilibriumRolloutCallback(family, name, args.n_steps, args.h) for name in args.algos]
    # Stop when the val loss stops improving; tolerant like the source setup (a rollout can log a
    # non-finite residual without aborting the run). cos_err/mag_ratio are reported as diagnostics.
    early_stop = EarlyStopping(
        monitor="val/mse",
        mode="min",
        patience=max(1, args.patience_epochs // args.val_every_n_epochs),
        check_finite=False,
        strict=False,
    )
    save_dir = os.getenv("SCRATCH", ".")
    if args.debug:
        logger = None
    elif args.logger == "wandb":
        logger = WandbLogger(
            experiment=wandb.init(project="learn-to-solve-games", group=args.exp, config=vars(args), dir=save_dir),
            save_dir=save_dir,
        )
    else:
        logger = CSVLogger(save_dir=save_dir)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        logger=logger,
        default_root_dir=save_dir,
        fast_dev_run=args.debug,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=rollouts + [early_stop],
        gradient_clip_val=args.gradient_clip_val or None,
        check_val_every_n_epoch=args.val_every_n_epochs,
        # The train stream is unbounded (no __len__), so cap the epoch at --steps_per_epoch; the
        # epoch-based cosine schedule (over --epochs) counts these bounded epochs.
        limit_train_batches=args.steps_per_epoch,
        inference_mode="consensus" not in args.algos,  # only consensus' rollout jacrev needs autograd in validation
    )
    trainer.fit(
        model,
        DataLoader(
            train_ds,
            batch_size=args.batch,
            num_workers=args.n_workers,
            persistent_workers=args.n_workers > 0,
            collate_fn=collate,
            # Single-thread each worker's route-choice solve: N multi-threaded workers oversubscribe
            # the cores and thrash, causing bursty/stalling batch delivery.
            worker_init_fn=lambda _worker_id: torch.set_num_threads(1),
        ),
        DataLoader(val_ds, batch_size=args.batch, collate_fn=collate),
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
