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
batch sizes ``--batch_uniform`` / ``--batch_fixed`` / ``--batch_rollout``):

- ``uniform`` (baseline): every step draws a fresh instance and solves the operator jointly for
  ``--points_per_instance`` cost points sampled **uniformly** over the calibrated domain box (see
  ``PUMEMarkovTrafficEquilibrium.sample_domain``) inside ``DataLoader`` workers -- so the model sees
  unbounded instance diversity rather than a fixed set (see ``data.build_streaming_operator_dataset`` /
  ``UniformSampledOperatorStream``).
- ``fixed``: the same uniform point sampling over a **fixed** set of ``--n_train_instances`` instances
  drawn once at startup, so instance diversity is bounded while the points stay fresh -- the knob for
  asking how many parametrizations amortization needs (see
  ``instance_sampling.FixedInstanceOperatorStream``).
- ``rollout`` (on-policy): trains on the cost points a solver actually visits when rolling out the
  *current* learned field with ``--train_algo`` from uniform starts, refreshed every
  ``--refresh_every`` epochs (see ``rollout_sampling.OnPolicyOperatorStream``). It holds a live model
  ref, so its loader runs with ``num_workers=0``.

``--cache`` bounds the two sampling sources' operator budget (see ``caching.CachedOperatorStream``):
they otherwise resample on every visit and so spend evals per epoch forever, capped only by
``--epochs``. Under it, ``--n_train_instances`` instances x ``--points_per_instance`` points are solved
per ``--refresh_every``-epoch window and then reused, so the budget is config-set:

    uniform / fixed + --cache   n_workers * n_train_instances * points_per_instance * windows
    expert                      n_workers * n_expert_instances * (n_steps + 1) * windows
    rollout                     n_rollout_instances * (n_steps + points_per_instance) * windows
    full (--amortization full)  n_workers * n_expert_instances * n_steps * windows

with ``windows = ceil(epochs / refresh_every)`` (one buffer per worker: PyTorch does not shard iterable
datasets). The ``expert`` source is ~1 example per eval -- it keeps the operator values its rollout of
the *true* field consumes instead of re-solving a subsample -- while ``full`` pays ``n_steps`` evals per
``z*`` label. ``train/operator_evals`` logs the running total (see ``OperatorCountCallback``).

The normalizer is fit once on a fixed calibration set (``--n_cal_instances``); val/test stay fixed.

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
from l2s_games.caching import CachedOperatorStream
from l2s_games.callbacks import FieldRolloutCallback, OperatorCountCallback, SolutionPredictionCallback
from l2s_games.data import (
    build_streaming_operator_dataset,
    build_streaming_solution_dataset,
    collate_examples,
    split_instances,
)
from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium
from l2s_games.instance_sampling import FixedInstanceOperatorStream
from l2s_games.models import (
    ConstrainedFieldModel,
    FieldModel,
    GraphormerBackbone,
    MLPBackbone,
    SolutionModel,
)
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
    # split into cal/val/test instances. Training still streams fresh instances on the fly
    # (--points_per_instance points each, one joint solve) unless --sources fixed pins them; the splits
    # fit the normalizer + calibrate the sampling range (cal) and measure generalization (val/test).
    # Epoch length is --steps_per_epoch.
    p.add_argument("--dataset_root", type=str, required=True, help="root of the cached SolvedInstanceDataset to load")
    p.add_argument(
        "--n_cal_instances", type=int, default=128, help="calibration split size (normalizer + range calibration)"
    )
    p.add_argument("--n_val_instances", type=int, default=128, help="validation split size")
    p.add_argument("--n_test_instances", type=int, default=128, help="held-out test split size")
    # The 'fixed' source's instance set is sampled fresh from the family (not split off the cache), so
    # its size is independent of the cache and of the cal/val/test splits.
    p.add_argument(
        "--n_train_instances",
        type=int,
        default=1024,
        help="instances in the fixed training set (the 'fixed' source), sampled once at startup; under "
        "--cache it also sizes the 'uniform' source's per-refresh-window draw",
    )
    p.add_argument(
        "--points_per_instance",
        type=int,
        default=32,
        help="cost points solved jointly per streamed train instance for the 'uniform' / 'fixed' / "
        "'rollout' sources (also calibration density for the normalizer fit); val/test always solve each "
        "instance once, and the 'expert' source keeps every state its rollout visits instead",
    )
    # Bounded operator budget for the two *sampling* sources (see l2s_games/caching.py). They otherwise
    # resample on every visit -- fresh instances ('uniform') or fresh points on fixed instances ('fixed')
    # -- so they spend steps_per_epoch * batch evals per epoch forever and the only cap is --epochs, which
    # ties the data budget to the optimization budget. The rollout sources already bound their spend with
    # a per-window buffer; this gives the sampling sources the same bound.
    p.add_argument(
        "--cache",
        action="store_true",
        help="freeze the sampled (instance, point) pairs per refresh window instead of resampling every "
        "visit, so the train operator budget is n_workers * n_train_instances * points_per_instance * "
        "ceil(epochs / refresh_every) -- set by config rather than growing with --epochs",
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
    # calibration split's equilibria -- per-edge mean (center) and std (spread) -- and sampled within
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
    # loss norm: the model predicts in the normalizer's target space (global scale + --target_warp);
    # --loss picks the norm the error is measured in. "mse_target" (default) compares in that target
    # space; "mse_real"/"huber"/"rel_l2" compare in real units *in scale units* (undo the warp only) --
    # the norm the rollout-residual bound controls. "huber" is the stable default of the real variants;
    # "rel_l2" is FNO-style per-sample relative L2 with an eps floor. (Under --target_warp none,
    # "mse_target" == "mse_real".) "mse" is space-qualified because it is offered in both spaces;
    # "huber"/"rel_l2" are real-space only, hence unqualified.
    p.add_argument(
        "--loss",
        choices=["mse_target", "mse_real", "huber", "rel_l2"],
        default="mse_target",
        help="training loss norm (target space vs. real-unit MSE/Huber/relative-L2)",
    )
    p.add_argument("--huber_delta_scale", type=float, default=1.0, help="huber knee, in scale units")
    p.add_argument(
        "--rel_eps",
        type=float,
        default=1.0,
        help="rel_l2 denominator floor, in scale units (caps near-eq up-weighting)",
    )
    # The warp and --precondition are two tools for the same problem on different axes: the supply
    # diagonal M^-1 is a per-edge, state-dependent rescale (it flattens the per-edge dynamic range and
    # the near-free-flow s' blow-up), while asinh is a global element-wise tail compressor. With
    # preconditioning on -- the default -- the target is already mildly tailed, so the warp has little
    # left to do and costs real geometry: it is a per-coordinate reweighting, so it does not preserve
    # inner products (hence not monotonicity), and undoing it for any real-unit quantity puts a cosh
    # gradient path on large-magnitude samples. Hence 'none' by default; pass 'asinh' to trade the
    # field's direction for tail compression (e.g. under --no-precondition, where the raw flow residual
    # is stiff and heavy-tailed).
    p.add_argument(
        "--target_warp",
        choices=["asinh", "none"],
        default="none",
        help="element-wise warp on the (always-applied) global field-target scale: 'none' (default) is "
        "linear and preserves the field's direction, leaving the tail to --precondition; 'asinh' "
        "compresses the tail. Partial amortization only -- the 'full' path standardizes z* and ignores this",
    )
    # monotonicity constraint (see l2s_games/monotonicity.py): fit the field subject to the model's raw
    # field being monotone, <F(x)-F(y), x-y> >= 0, on the training domain. Pairs are the same-instance
    # points a 'fixed'-source batch already holds, so the constraint adds no solves and no forwards.
    p.add_argument(
        "--monotonicity",
        action="store_true",
        help="constrain the learned field to be monotone (quadratic penalty; requires --sources fixed)",
    )
    p.add_argument(
        "--constraint_space",
        choices=["raw", "preconditioned"],
        default="raw",
        help="which field the constraint is about: 'raw' maps predictions back through the example's "
        "metric diagonal (the *unpreconditioned* operator, which is monotone); 'preconditioned' "
        "constrains what the model literally predicts, which the operator itself violates",
    )
    p.add_argument(
        "--constraint_norm",
        choices=["ratio", "none"],
        default="ratio",
        help="per-pair normalization: 'ratio' divides by ||x-y||^2 (the secant Rayleigh quotient, so the "
        "tolerance is independent of pair separation); 'none' uses the raw inner product",
    )
    p.add_argument(
        "--constraint_tolerance",
        type=float,
        default=1e-3,
        help="violation depth treated as satisfied; also the threshold above which the penalty grows",
    )
    p.add_argument("--penalty_mu", type=float, default=1.0, help="initial penalty coefficient (0 measures only)")
    p.add_argument("--penalty_growth", type=float, default=1.01, help="multiplicative penalty growth per step")
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
    p.add_argument("--batch_fixed", type=int, default=128, help="minibatch size for the fixed-instance stream")
    p.add_argument("--batch_rollout", type=int, default=128, help="minibatch size for the on-policy rollout stream")
    p.add_argument("--batch_expert", type=int, default=128, help="minibatch size for the expert-demonstration stream")
    # data sources (one stream + dataloader per source; see data.OperatorStream subclasses)
    p.add_argument(
        "--sources",
        nargs="+",
        choices=["uniform", "fixed", "rollout", "expert"],
        default=["fixed"],
        help="training data sources, each its own stream+dataloader: 'uniform' samples the domain "
        "uniformly over fresh instances (baseline); 'fixed' samples it the same way but over a fixed set "
        "of --n_train_instances instances; 'rollout' trains on points visited by rolling out the current "
        "learned field (on-policy), refreshed every --refresh_every epochs; 'expert' trains on the path a "
        "converging algorithm takes on the *true* field plus the equilibrium solutions. Combine them "
        "to blend (the mix is set by --batch_uniform / --batch_fixed / --batch_rollout / --batch_expert)",
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
        default=["projection", "consensus"],
        help="dynamics algorithms rolled out on the learned field each val epoch, logging the analytic "
        "endpoint residual val/{algo}/residual (pass --algos with no value for fast field-only training)",
    )
    # Tuned for the PUME preconditioned excess-supply operator (see scripts/tune_pume_rollout.py):
    # projection's stability ceiling is between h=0.2 and h=0.4, and h=0.1 gives a clean monotone
    # decay reaching the equilibrium (rel-dist <0.01 by ~step 200, residual still dropping at 500).
    # The old h=0.02/1000 was sized for the raw cost-space operator and badly under-steps here.
    p.add_argument("--h", type=float, default=0.1, help="algorithm step size (damped fixed point)")
    p.add_argument("--n_steps", type=int, default=500, help="iterations for the rollout")
    return p


def main(args):
    # workers=True makes Lightning seed each streaming dataloader worker distinctly & reproducibly.
    if args.seed is None:
        args.seed = torch.randint(0, 2**31 - 1, (1,)).item()
    L.seed_everything(args.seed, workers=True)
    # Load the cached solved instances and split them into cal/val/test. The calibration split's
    # equilibria calibrate the streaming sampling range (per-edge mean + std); the calibrated tensors
    # are baked into the picklable factory so every worker shares the same range.
    dataset = SolvedInstanceDataset(args.dataset_root)
    instances = list(dataset)
    cal_inst, val_inst, test_inst = split_instances(
        instances, (args.n_cal_instances, args.n_val_instances, args.n_test_instances)
    )
    reference_equilibrium, reference_spread = PUMEMarkovTrafficEquilibrium.calibrate_range(cal_inst)
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
    # (validation + collate) and the one-time cal/val/test build stay counter-free (family_factory).
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
        (val_ds, test_ds, cal_ds), normalizer = build_streaming_solution_dataset(
            family_factory, cal_inst, val_inst, test_inst
        )
        train_ds = None
    else:
        (train_ds, val_ds, test_ds, cal_ds), normalizer = build_streaming_operator_dataset(
            family_factory,
            cal_inst,
            val_inst,
            test_inst,
            args.points_per_instance,
            stream_factory=counting_factory,
            warp=args.target_warp,
            cache_instances=args.n_train_instances if args.cache else 0,
            refresh_every=args.refresh_every,
        )
    print(f"streaming train   cal: {len(cal_ds)}   val: {len(val_ds)}   test: {len(test_ds)}")

    # Size the backbone from one transformed calibration example (line-graph structure + feature width);
    # the train stream is iterable, so it cannot be indexed. The same backbone feeds either task:
    # --amortization picks the Field vs Solution task wrapper (different target + validation), and only
    # the Field task takes the --loss knobs (the solution task is plain MSE on a standardized z*). The
    # mlp is the whole-graph flat baseline: flatten the fixed network's per-edge feats [E, k] to one
    # vector and predict every edge jointly (out_features = E).
    sample, _ = cal_ds[0]
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
        if args.monotonicity:
            # The constraint pairs same-instance points *within* a batch, so it needs a source whose
            # examples carry a stable instance tag and several points per instance: only 'fixed' does
            # (the uniform stream draws a fresh instance each visit, so its points cannot be grouped).
            assert args.sources == ["fixed"], f"--monotonicity requires --sources fixed, got {args.sources}"
            assert args.points_per_instance >= 2, "--monotonicity needs >= 2 points per instance to form a pair"
            model = ConstrainedFieldModel(
                net,
                **train_kwargs,
                **loss_kwargs,
                family=family,
                tolerance=args.constraint_tolerance,
                penalty_mu=args.penalty_mu,
                penalty_growth=args.penalty_growth,
                normalize=args.constraint_norm == "ratio",
                constrain_raw=args.constraint_space == "raw",
            )
        else:
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
            # and the picklable stream keeps its route-choice-solving workers. Under --cache this is the
            # bounded variant (built in build_streaming_operator_dataset): the same fresh-sample
            # distribution while it fills, then reused rather than resampled.
            train_loaders["uniform"] = DataLoader(
                train_ds,
                batch_size=args.batch_uniform,
                num_workers=args.n_workers,
                persistent_workers=args.n_workers > 0,
                collate_fn=collate,
                worker_init_fn=_single_thread_worker,
            )
        if "fixed" in args.sources:
            # Bounded instance diversity: the instance set is drawn once here (reproducible under the
            # global seed) and shipped to every worker, while the domain points stay freshly sampled --
            # so this differs from the uniform source only in pinning the parametrizations. Model-free,
            # so it keeps the route-choice-solving workers. --cache freezes the points too, bounding the
            # budget; the instance set is the same either way.
            fixed_instances = [family.sample_params() for _ in range(args.n_train_instances)]
            if args.cache:
                fixed_stream = CachedOperatorStream(
                    counting_factory,
                    normalizer,
                    args.points_per_instance,
                    instances=fixed_instances,
                    refresh_every=args.refresh_every,
                )
            else:
                fixed_stream = FixedInstanceOperatorStream(
                    counting_factory, normalizer, fixed_instances, args.points_per_instance
                )
            train_loaders["fixed"] = DataLoader(
                fixed_stream,
                batch_size=args.batch_fixed,
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
            # equilibrium solutions. Because the rolled-out field *is* the ground truth, the values the
            # rollout consumes are kept as the targets (RecordedField) rather than re-solved, so its
            # budget is n_expert_instances * (n_steps + 1) per window -- one example per evaluation.
            expert_stream = ExpertOperatorStream(
                counting_factory,
                normalizer,
                args.train_algo,
                args.h,
                args.n_steps,
                args.n_expert_instances,
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
