"""
train_field_gnn.py

Amortize the traffic **operator field** with a Graphormer over the line graph. The model maps
each road edge's features ``[cost, free_flow_time, capacity, b, power, +4 demand features]`` plus
the line-graph structure to the per-edge operator value ``costs - bpr(demand_flow(-costs))`` -- so
one network represents the operator across a family of noised SiouxFalls instances.

``--model`` selects the architecture: ``graphormer`` (line-graph attention) or ``mlp`` -- a
whole-graph flat baseline that flattens the fixed network's per-edge feats and predicts every edge
jointly (no graph inductive bias), for benchmarking how much the Graphormer's structure buys.

Training data is **streamed** from one or more sources selected with ``--sources`` (each its own
stream + dataloader, blended by Lightning's ``CombinedLoader``; the mix is set by the per-source
batch sizes ``--batch_uniform`` / ``--batch_rollout``):

- ``uniform`` (baseline): every step draws a fresh instance and solves the operator jointly for
  ``--points_per_instance`` cost points sampled **uniformly** over the calibrated domain box (see
  ``PUMEMarkovTrafficEquilibrium.sample_domain``) inside ``DataLoader`` workers -- so the model sees
  unbounded instance diversity rather than a fixed set (see ``data.build_streaming_operator_dataset`` /
  ``UniformSampledOperatorStream``).
- ``rollout`` (on-policy): trains on the cost points a solver actually visits when rolling out the
  *current* learned field with ``--train_algo`` from uniform starts, refreshed every
  ``--refresh_every`` epochs (see ``rollout_sampling.OnPolicyOperatorStream``). It holds a live model
  ref, so its loader runs with ``num_workers=0``.

The normalizer is fit once on a fixed bootstrap set (``--bootstrap_instances``); val/test stay fixed.

Each validation epoch logs, over the held-out validation set, the field relative error plus -- for
every algorithm in ``--algos`` -- the analytic operator residual ``||E(c)||`` (PUME excess supply
``z(c) - x(c)``, supply-diagonal preconditioned unless ``--no-precondition``) at the endpoint of a
projected rollout of that algorithm on the learned field. The whole val batch of instances is solved
at once (see ``FieldModel.batched_field`` / ``PUMEMarkovTrafficEquilibrium.operator``).

    python scripts/train_field_gnn.py --n_workers 4
"""

import argparse
import functools
import os

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from l2s_games.algorithms import ALGORITHMS
from l2s_games.callbacks import FieldRolloutCallback, OperatorCountCallback, SolutionPredictionCallback
from l2s_games.data import (
    build_streaming_operator_dataset,
    build_streaming_solution_dataset,
    collate_examples,
    split_instances,
)
from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium
from l2s_games.models import FieldModel, GraphormerBackbone, MLPBackbone, SolutionModel
from l2s_games.operator_count import SharedCounter
from l2s_games.rollout_sampling import ExpertOperatorStream, OnPolicyOperatorStream

torch.set_float32_matmul_precision("medium")


def _single_thread_worker(_worker_id):
    """Single-thread each worker's route-choice solve: N multi-threaded workers oversubscribe the
    cores and thrash, causing bursty/stalling batch delivery. A module-level function (not a local
    lambda) so it is picklable under the ``spawn`` start method (macOS)."""
    torch.set_num_threads(1)


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
    p.add_argument(
        "--precondition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rescale the PUME excess-supply field by the supply-diagonal metric (M^-1 E) so the stiff "
        "flow residual is well-scaled for the rollout algorithms; --no-precondition uses the raw field",
    )
    p.add_argument("--seed", type=int, default=None, help="global seed")
    # domain coverage (see PUMEMarkovTrafficEquilibrium.sample_domain): the range is calibrated from the
    # bootstrap split's equilibria -- per-edge mean (center) and std (spread) -- and sampled within
    # --sample_stds sigma of that mean. --equilibrium_margin/--equilibrium_spread are the uncalibrated
    # fallback only (used when a family is built without a calibrated range, e.g. the sandbox).
    p.add_argument(
        "--sample_stds",
        type=float,
        default=3.0,
        help="sigma reach of the per-edge uniform domain box ceiling (reference_equilibrium + "
        "sample_stds * reference_spread) sample_domain draws uniformly up to",
    )
    p.add_argument(
        "--equilibrium_margin",
        type=float,
        default=2.5,
        help="uncalibrated fallback: reference-equilibrium ceiling widen",
    )
    p.add_argument("--equilibrium_spread", type=float, default=0.2, help="uncalibrated fallback: multiplicative spread")
    # amortization target: 'partial' (default) learns the operator *field* -- a solver still rolls it
    # out (--algos) to reach the equilibrium; 'full' learns the equilibrium *solution* directly,
    # z* = g(params), with no rollout at inference. The same backbone serves both (both predict a
    # per-edge [B, E] vector); 'full' feeds a parameters-only input (the free-flow start fills the
    # query column) and regresses z* via the expert solution stream, ignoring --sources / --algos /
    # --batch_uniform / --batch_rollout.
    p.add_argument(
        "--amortization",
        choices=["partial", "full"],
        default="partial",
        help="'partial' learns the operator field (rolled out to solve); 'full' predicts the "
        "equilibrium solution z* directly from parameters (no rollout)",
    )
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
    p.add_argument("--n_heads", type=int, default=8, help="attention heads (graphormer only)")
    p.add_argument("--n_layers", type=int, default=6, help="layers")
    p.add_argument("--dim_ff", type=int, default=512, help="feed-forward dim (graphormer only)")
    # No --dropout / --weight_decay: the streaming pipeline sees a fresh instance every step, so it
    # can't overfit -- both are hardcoded to 0 (no regularization) at model construction.
    # loss norm: the model always predicts in asinh space; --loss picks the norm the error is measured
    # in. "asinh_mse" (default) compares in asinh space; "l2"/"huber"/"rel_l2" compare in real units
    # (via sinh, in asinh-scale units) -- the norm the rollout-residual bound controls. "huber" is the
    # stable default of the real variants; "rel_l2" is FNO-style per-sample relative L2 with an eps floor.
    p.add_argument(
        "--loss",
        choices=["asinh_mse", "l2", "huber", "rel_l2"],
        default="asinh_mse",
        help="training loss norm (asinh space vs. real-unit L2/Huber/relative-L2)",
    )
    p.add_argument("--huber_delta_scale", type=float, default=1.0, help="huber knee, in asinh-scale units")
    p.add_argument(
        "--rel_eps",
        type=float,
        default=1.0,
        help="rel_l2 denominator floor, in asinh-scale units (caps near-eq up-weighting)",
    )
    # training (AdamW + linear-warmup->cosine, ported from markov-traffic-eq)
    p.add_argument("--lr", type=float, default=0.0012, help="AdamW learning rate")
    p.add_argument("--start_factor", type=float, default=0.011, help="linear warmup start factor")
    p.add_argument("--warmup_epochs", type=int, default=30, help="linear warmup epochs (must be < --epochs)")
    p.add_argument("--cosine_annealing", type=int, default=0, help="cosine-anneal after warmup (0 disables)")
    p.add_argument("--gradient_clip_val", type=float, default=7, help="gradient-norm clip value")
    p.add_argument("--epochs", type=int, default=2000, help="training epochs")
    p.add_argument(
        "--steps_per_epoch", type=int, default=64, help="train batches per epoch (bounds the infinite streams)"
    )
    p.add_argument("--batch_uniform", type=int, default=128, help="minibatch size for the uniform stream")
    p.add_argument("--batch_rollout", type=int, default=128, help="minibatch size for the on-policy rollout stream")
    p.add_argument("--batch_expert", type=int, default=128, help="minibatch size for the expert-demonstration stream")
    # data sources (one stream + dataloader per source; see data.OperatorStream subclasses)
    p.add_argument(
        "--sources",
        nargs="+",
        choices=["uniform", "rollout", "expert"],
        default=["uniform"],
        help="training data sources, each its own stream+dataloader: 'uniform' samples the domain "
        "uniformly (baseline); 'rollout' trains on points visited by rolling out the current learned "
        "field (on-policy), refreshed every --refresh_every epochs; 'expert' trains on the path a "
        "converging algorithm takes on the *true* field plus the equilibrium solutions. Combine them "
        "to blend (the mix is set by --batch_uniform / --batch_rollout / --batch_expert)",
    )
    # Both rollout-based sources generate training points by rolling out the SAME converging algorithm
    # -- 'rollout' on the learned field, 'expert' on the true operator -- so they share --train_algo.
    # consensus is excluded: jacrev does not compose through the analytic operator the expert rolls out.
    p.add_argument(
        "--train_algo",
        choices=[name for name in ALGORITHMS if name != "consensus"],
        default="projection",
        help="converging algorithm rolled out to generate training points, on the learned field (the "
        "on-policy 'rollout' source) and on the *true* field (the 'expert'/solution source); it shapes "
        "the sampling distribution (projection descends the preconditioned excess supply). Reuses --h / --n_steps",
    )
    p.add_argument("--refresh_every", type=int, default=5, help="regenerate the on-policy buffer every N epochs")
    p.add_argument("--n_rollout_instances", type=int, default=128, help="instances rolled out per buffer refresh")
    p.add_argument("--n_expert_instances", type=int, default=128, help="instances rolled out jointly per expert batch")
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
    # validation equilibrium sweep (rollout on the learned field, per algorithm; ignored under --amortization full)
    p.add_argument(
        "--algos",
        nargs="*",
        choices=list(ALGORITHMS),
        default=list(ALGORITHMS),
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
    reference_equilibrium, reference_spread = PUMEMarkovTrafficEquilibrium.calibrate_range(bootstrap_inst)
    # A picklable factory (base graph + calibrated tensors) the streaming dataset ships to each worker,
    # which builds its own family + PUME solver lazily -- nothing solver-related is pickled. The
    # main process also needs one live family for collate_fn and the validation rollout callbacks.
    family_factory = functools.partial(
        PUMEMarkovTrafficEquilibrium,
        dataset.base_graph,
        noise_scale=args.noise_scale,
        reference_equilibrium=reference_equilibrium,
        reference_spread=reference_spread,
        n_stds=args.sample_stds,
        precondition=args.precondition,
    )
    family = family_factory()
    # A process-safe counter of ground-truth operator point-evaluations (the training budget), shared
    # only by the families that generate training data: counting_factory bakes it in, so every
    # streaming worker + the on-policy/expert streams increment the same total, while the main `family`
    # (validation + collate) and the one-time bootstrap/val/test build stay counter-free (family_factory).
    operator_counter = SharedCounter()
    counting_factory = functools.partial(
        PUMEMarkovTrafficEquilibrium,
        dataset.base_graph,
        noise_scale=args.noise_scale,
        reference_equilibrium=reference_equilibrium,
        reference_spread=reference_spread,
        n_stds=args.sample_stds,
        operator_counter=operator_counter,
        precondition=args.precondition,
    )
    # 'full' amortization regresses z* directly: its fixed splits use the cached equilibria and its
    # only train source is the expert solution stream (built below). 'partial' regresses the operator
    # field, with the uniform stream as its always-on train source. train_ds is None under 'full'.
    if args.amortization == "full":
        (val_ds, test_ds, bootstrap_ds), normalizer = build_streaming_solution_dataset(
            family_factory, bootstrap_inst, val_inst, test_inst
        )
        train_ds = None
    else:
        (train_ds, val_ds, test_ds, bootstrap_ds), normalizer = build_streaming_operator_dataset(
            family_factory,
            bootstrap_inst,
            val_inst,
            test_inst,
            args.points_per_instance,
            stream_factory=counting_factory,
        )
    print(f"streaming train   bootstrap: {len(bootstrap_ds)}   val: {len(val_ds)}   test: {len(test_ds)}")

    # Size the backbone from one transformed bootstrap example (line-graph structure + feature width);
    # the train stream is iterable, so it cannot be indexed. The same backbone feeds either task:
    # --amortization picks the Field vs Solution task wrapper (different target + validation), and only
    # the Field task takes the --loss knobs (the solution task is plain MSE on a standardized z*). The
    # mlp is the whole-graph flat baseline: flatten the fixed network's per-edge feats [E, k] to one
    # vector and predict every edge jointly (out_features = E).
    sample, _ = bootstrap_ds[0]
    if args.model == "graphormer":
        net = GraphormerBackbone(
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
    else:
        net = MLPBackbone(
            in_features=sample["feats"].numel(),  # E * k
            hidden=[args.dim] * args.n_layers,
            out_features=sample["feats"].shape[0],  # E
            flatten_start_dim=1,  # collapse per-edge feats [B, E, k] -> [B, E*k]
        )
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
    else:
        loss_kwargs = dict(loss=args.loss, huber_delta_scale=args.huber_delta_scale, rel_eps=args.rel_eps)
        model = FieldModel(net, **train_kwargs, **loss_kwargs)
    collate = collate_examples(family)
    # Per amortization mode, build the training source loaders and the matching validation callback.
    # training_step concatenates every source into one MSE (Lightning's CombinedLoader), and the
    # per-source batch sizes set the mix.
    if args.amortization == "full":
        # Full amortization: the sole train source is the expert *solution* stream -- it rolls out the
        # true operator to z* and regresses z* directly from a parameters-only input (solution_target).
        # Model-free, so it keeps the route-choice-solving workers; include_trajectory off (its
        # operator-value targets cannot mix into a z* regression). Validation scores the model's direct
        # prediction (no rollout).
        solution_stream = ExpertOperatorStream(
            counting_factory,
            normalizer,
            args.train_algo,
            args.h,
            args.n_steps,
            args.n_expert_instances,
            args.points_per_instance,
            args.refresh_every,
            include_trajectory=False,
            solution_target=True,
        )
        train_loaders = {
            "solution": DataLoader(
                solution_stream,
                batch_size=args.batch_expert,
                num_workers=args.n_workers,
                persistent_workers=args.n_workers > 0,
                collate_fn=collate,
                worker_init_fn=_single_thread_worker,
            )
        }
        callbacks = [SolutionPredictionCallback(family)]
    else:
        # One loader per source named in --sources (>=1, argparse-enforced); training_step blends them.
        # Validation rolls out the learned field per --algos (a sweep).
        train_loaders = {}
        if "uniform" in args.sources:
            # Fresh uniform-domain samples -- cold-start coverage while the learned field is near-random,
            # and the picklable stream keeps its route-choice-solving workers.
            train_loaders["uniform"] = DataLoader(
                train_ds,
                batch_size=args.batch_uniform,
                num_workers=args.n_workers,
                persistent_workers=args.n_workers > 0,
                collate_fn=collate,
                worker_init_fn=_single_thread_worker,
            )
        if "rollout" in args.sources:
            # The on-policy stream owns its rollout + buffer, refreshing every --refresh_every epochs:
            # it draws fresh instances and re-rolls out the current field over them (live model ref,
            # hence num_workers=0). Starts are sampled uniformly by sample_domain.
            rollout_stream = OnPolicyOperatorStream(
                counting_factory,
                normalizer,
                model,
                args.train_algo,
                args.h,
                args.n_steps,
                args.n_rollout_instances,
                args.points_per_instance,
                args.refresh_every,
            )
            train_loaders["rollout"] = DataLoader(
                rollout_stream, batch_size=args.batch_rollout, num_workers=0, collate_fn=collate
            )
        if "expert" in args.sources:
            # The expert stream rolls out the *analytic* operator (no model), so it is picklable and
            # keeps the route-choice-solving workers; it yields both the expert trajectory and the
            # equilibrium solutions.
            expert_stream = ExpertOperatorStream(
                counting_factory,
                normalizer,
                args.train_algo,
                args.h,
                args.n_steps,
                args.n_expert_instances,
                args.points_per_instance,
                args.refresh_every,
            )
            train_loaders["expert"] = DataLoader(
                expert_stream,
                batch_size=args.batch_expert,
                num_workers=args.n_workers,
                persistent_workers=args.n_workers > 0,
                collate_fn=collate,
                worker_init_fn=_single_thread_worker,
            )
        # Pass no --algos to skip the sweep for fast field-only tuning (rel_err metrics still logged).
        callbacks = [FieldRolloutCallback(family, name, args.n_steps, args.h) for name in args.algos]
    # Log the cumulative training-data operator point-evaluation budget each step (see OperatorCountCallback).
    callbacks.append(OperatorCountCallback(operator_counter))
    # Stop when the val loss stops improving; tolerant like the source setup (a rollout can log a
    # non-finite residual without aborting the run). cos_err/mag_ratio are reported as diagnostics.
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
        # Tag the run with game=pume_traffic so its config is comparable to train_field_mlp.py's runs
        # (which log --game); this script is traffic-only, so it's a fixed constant.
        run = wandb.init(
            project="learn-to-solve-games",
            group=args.exp,
            config={**vars(args), "game": "pume_traffic"},
            dir=save_dir,
        )
        # Declare the operator budget so it can be *selected* as a custom x-axis in any panel (e.g.
        # plot val/mse against the number of ground-truth operator evaluations spent) -- without
        # forcing it as the default x-axis for any metric.
        run.define_metric("train/operator_evals")
        logger = WandbLogger(experiment=run, save_dir=save_dir)
    else:
        logger = CSVLogger(save_dir=save_dir)
    # Save the best (by val/mse) + last checkpoint so a run's weights survive for downstream analysis.
    # Gated on the logger: debug runs disable checkpointing, and Lightning rejects a ModelCheckpoint
    # when it's off. No dirpath -- with a logger present Lightning places checkpoints under the
    # run-namespaced path (<save_dir>/<project>/<run_id>/checkpoints/), so concurrent sweep runs don't
    # clobber each other. The normalizer rides along in the checkpoint (see FieldModel.on_save_checkpoint),
    # so the loaded model can de-standardize predictions on its own. filename is fixed (no metric
    # interpolation -- the "val/mse" key's slash isn't a valid format field).
    if logger is not None:
        callbacks.append(ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=1, save_last=True, filename="best"))
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
        # The train stream is unbounded (no __len__), so cap the epoch at --steps_per_epoch; the
        # epoch-based cosine schedule (over --epochs) counts these bounded epochs.
        limit_train_batches=args.steps_per_epoch,
        # Only the partial-mode consensus rollout jacrev needs autograd in validation; full mode
        # (direct z* prediction, no rollout) always runs under inference_mode.
        inference_mode=args.amortization == "full" or "consensus" not in args.algos,
    )
    trainer.fit(
        model,
        # A named-source mapping so the batch matches training_step's contract
        # (batch = {source: (inputs, targets)}); Lightning wraps this in a CombinedLoader.
        train_loaders,
        DataLoader(val_ds, batch_size=args.batch_uniform + args.batch_rollout, collate_fn=collate),
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
