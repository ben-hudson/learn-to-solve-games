"""
train_field_gnn.py

Amortize the traffic **operator field** with a Graphormer over the line graph. The model maps
each road edge's features ``[cost, free_flow_time, capacity, b, power, +4 demand features]`` plus
the line-graph structure to the per-edge operator value ``costs - bpr(demand_flow(-costs))`` -- so
one network represents the operator across a family of noised SiouxFalls instances.

Training data is **streamed**: every step draws a fresh instance (one cost point each), solving the
operator for its target inside ``DataLoader`` workers -- so the model sees unbounded instance
diversity rather than a fixed set (see ``data.build_streaming_dataset`` / ``StreamingFieldDataset``).
The normalizer is fit once on a fixed bootstrap set (``--n-instances``); validation/test stay fixed.

Each validation epoch logs, over the held-out validation set, the field relative error plus -- for
every algorithm in ``--algos`` -- the analytic residual ``||costs - bpr(demand_flow(-costs))||`` at
the endpoint of a projected rollout of that algorithm on the learned field. The whole val batch of
instances is solved at once (see ``FieldModel.batched_field`` / ``MarkovTrafficEquilibrium.operator``).

    python scripts/train_field_gnn.py --num-workers 4
"""

import argparse
import functools

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from traffic_equilibrium_sandbox import sioux_falls_base_graph

from l2s_games.algorithms import ALGORITHMS
from l2s_games.callbacks import EquilibriumRolloutCallback
from l2s_games.data import build_streaming_dataset, collate_examples
from l2s_games.envs.traffic import MarkovTrafficEquilibrium
from l2s_games.models import GraphormerFieldModel

torch.set_float32_matmul_precision("medium")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dataset (each sample is a route-choice solve, so keep points-per-instance modest)
    # Training streams fresh instances on the fly (one point each); --n-instances only sizes the fixed
    # bootstrap set the normalizer is fit on (and the epoch length, see limit_train_batches below).
    p.add_argument("--n-instances", type=int, default=64, help="bootstrap instances (fit normalizer; size epoch)")
    p.add_argument("--n-val-instances", type=int, default=16, help="validation instances (fixed)")
    p.add_argument("--n-test-instances", type=int, default=16, help="held-out test instances (fixed)")
    p.add_argument("--points-per-instance", type=int, default=32, help="cost samples per fixed (bootstrap/val/test) instance")
    p.add_argument("--num-workers", type=int, default=4, help="streaming dataloader workers (0 = serial; changes the stream)")
    p.add_argument("--noise-scale", type=float, default=0.2, help="multiplicative attribute noise")
    p.add_argument("--seed", type=int, default=0, help="global seed")
    # domain coverage (see MarkovTrafficEquilibrium.sample_domain): samples fill the path from the
    # free-flow-time start up to the base network's reference equilibrium, widened by the margin so
    # the perturbed instances' equilibria stay bracketed -- no per-instance equilibrium solve.
    p.add_argument(
        "--equilibrium-margin",
        type=float,
        default=2.5,
        help="widen the reference-equilibrium ceiling to bracket perturbations",
    )
    p.add_argument(
        "--equilibrium-spread", type=float, default=0.2, help="multiplicative spread off the fft->equilibrium path"
    )
    # the operator is heavy-tailed (BPR blows up at low costs); clip the standardized target to
    # +-this many sigma so a few outliers don't dominate the MSE fit -- the equilibrium F=0 is kept.
    p.add_argument(
        "--target-clip",
        type=float,
        default=300.0,
        help="cap the real-unit field's L2 norm, direction-preserving (0 disables)",
    )
    # model
    p.add_argument("--dim", type=int, default=128, help="Graphormer hidden dim")
    p.add_argument("--n-heads", type=int, default=4, help="attention heads")
    p.add_argument("--n-layers", type=int, default=4, help="encoder layers")
    p.add_argument("--dim-ff", type=int, default=128, help="feed-forward dim")
    p.add_argument("--dropout", type=float, default=0.2, help="dropout")
    # training (AdamW + linear-warmup->cosine, ported from markov-traffic-eq)
    p.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    p.add_argument("--start-factor", type=float, default=0.01, help="linear warmup start factor")
    p.add_argument("--warmup-epochs", type=int, default=50, help="linear warmup epochs (must be < --epochs)")
    p.add_argument("--cosine-annealing", type=int, default=1, help="cosine-anneal after warmup (0 disables)")
    p.add_argument("--gradient-clip-val", type=float, default=1.0, help="gradient-norm clip value")
    p.add_argument("--epochs", type=int, default=400, help="training epochs")
    p.add_argument("--batch", type=int, default=32, help="minibatch size")
    # early stopping on the field relative error (no MAPE here -- we regress the operator field)
    p.add_argument("--patience-epochs", type=int, default=40, help="early-stopping patience in epochs")
    p.add_argument("--val-every-n-epochs", type=int, default=1, help="run validation every N epochs")
    # validation equilibrium sweep (rollout on the learned field, per algorithm)
    p.add_argument(
        "--algos",
        nargs="*",
        choices=list(ALGORITHMS),
        default=[],
        help="dynamics algorithms swept in validation (pass none for fast field-only training)",
    )
    # h=0.05 sits above the stability threshold for the stiffer instances -- simGD then oscillates
    # at a ~8e-2 residual floor even on the true operator; h=0.02/1000 converges to ~1e-6 on every
    # val instance, so the learned-field residual is measured against a reachable target.
    p.add_argument("--h", type=float, default=0.02, help="algorithm step size (damped fixed point)")
    p.add_argument("--n-steps", type=int, default=1000, help="iterations for the rollout")
    return p


def main(args):
    # workers=True makes Lightning seed each streaming dataloader worker distinctly & reproducibly.
    L.seed_everything(args.seed, workers=True)
    # A picklable factory (base graph + floats) the streaming dataset ships to each worker, which
    # builds its own family + route-choice solver lazily -- nothing solver-related is pickled. The
    # main process also needs one live family for collate_fn and the validation rollout callbacks.
    family_factory = functools.partial(
        MarkovTrafficEquilibrium,
        sioux_falls_base_graph(),
        noise_scale=args.noise_scale,
        equilibrium_margin=args.equilibrium_margin,
        equilibrium_spread=args.equilibrium_spread,
    )
    family = family_factory()
    (train_ds, val_ds, test_ds, bootstrap_ds), normalizer = build_streaming_dataset(
        family_factory,
        args.n_instances,
        args.n_val_instances,
        args.n_test_instances,
        args.points_per_instance,
        target_clip=args.target_clip or None,
    )
    print(f"streaming train   bootstrap: {len(bootstrap_ds)}   val: {len(val_ds)}   test: {len(test_ds)}")

    # Size the model from one transformed bootstrap example (line-graph structure + feature width);
    # the train stream is iterable, so it cannot be indexed.
    sample, _ = bootstrap_ds[0]
    model = GraphormerFieldModel(
        n_feats=sample["feats"].shape[-1],
        in_degree=sample["in_degree"],
        out_degree=sample["out_degree"],
        spd=sample["spd"],
        lr=args.lr,
        normalizer=normalizer,
        weight_decay=args.weight_decay,
        start_factor=args.start_factor,
        warmup_epochs=args.warmup_epochs,
        cosine_annealing=bool(args.cosine_annealing),
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    )
    collate = collate_examples(family)
    # Each validation epoch rolls out the learned field per algorithm on the held-out val batch and
    # logs the analytic residual at the endpoint (plus train/val_mse and train/val_rel_err). Pass no
    # --algos to skip the sweep for fast field-only tuning (rel_err metrics still logged).
    rollouts = [EquilibriumRolloutCallback(family, name, args.n_steps, args.h) for name in args.algos]
    # Stop when the field relative error stops improving; tolerant like the source setup (a rollout
    # can log a non-finite residual without aborting the run).
    early_stop = EarlyStopping(
        monitor="val_rel_err",
        mode="min",
        patience=max(1, args.patience_epochs // args.val_every_n_epochs),
        check_finite=False,
        strict=False,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=rollouts + [early_stop],
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.val_every_n_epochs,
        # The train stream is unbounded (no __len__), so cap the epoch; keep it ~the old fixed-set size
        # so the epoch-based cosine schedule stays meaningful.
        limit_train_batches=max(1, (args.n_instances * args.points_per_instance) // args.batch),
        inference_mode="consensus" not in args.algos,  # only consensus' rollout jacrev needs autograd in validation
    )
    trainer.fit(
        model,
        DataLoader(
            train_ds,
            batch_size=args.batch,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        ),
        DataLoader(val_ds, batch_size=args.batch, collate_fn=collate),
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
