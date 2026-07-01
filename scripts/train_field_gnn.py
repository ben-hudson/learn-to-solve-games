"""
train_field_gnn.py

Amortize the traffic **operator field** with a Graphormer over the line graph. The model maps
each road edge's features ``[cost, free_flow_time, capacity, b, power, +4 demand features]`` plus
the line-graph structure to the per-edge operator value ``costs - bpr(demand_flow(-costs))`` -- so
one network represents the operator across a family of noised SiouxFalls instances.

After training it builds the learned field with ``conditioned_field`` and rolls out projected
extragradient on it, checking that the learned-field dynamics converge near the analytic
equilibrium (solved with torchdeq for reference).

    python scripts/train_field_gnn.py --epochs 30
"""

import argparse

import lightning as L
import torch
from torch.utils.data import DataLoader

from traffic_equilibrium_sandbox import sioux_falls_base_graph, solve_equilibrium

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import build_dataset, collate_examples
from l2s_games.dynamics import simulate
from l2s_games.envs import bind
from l2s_games.envs.traffic import MarkovTrafficEquilibrium
from l2s_games.models import GraphormerFieldModel, conditioned_field


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dataset (each sample is a route-choice solve, so keep points-per-instance modest)
    p.add_argument("--n-instances", type=int, default=64, help="training instances")
    p.add_argument("--n-val-instances", type=int, default=16, help="validation instances")
    p.add_argument("--n-test-instances", type=int, default=16, help="held-out test instances")
    p.add_argument("--points-per-instance", type=int, default=16, help="cost samples per instance")
    p.add_argument("--noise-scale", type=float, default=0.2, help="multiplicative attribute noise")
    p.add_argument("--seed", type=int, default=0, help="global seed")
    # model
    p.add_argument("--dim", type=int, default=64, help="Graphormer hidden dim")
    p.add_argument("--n-heads", type=int, default=4, help="attention heads")
    p.add_argument("--n-layers", type=int, default=4, help="encoder layers")
    p.add_argument("--dim-ff", type=int, default=128, help="feed-forward dim")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout")
    # training
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--epochs", type=int, default=30, help="training epochs")
    p.add_argument("--batch", type=int, default=32, help="minibatch size")
    # dynamics check
    p.add_argument("--algo", choices=list(ALGORITHMS), default="simgd", help="rollout algorithm")
    p.add_argument("--h", type=float, default=0.05, help="algorithm step size (damped fixed point)")
    p.add_argument("--n-steps", type=int, default=300, help="iterations for the rollout")
    return p


def relative_error(model, test_ds, normalizer, collate):
    """Relative error in real units (predictions and targets de-standardized)."""
    inputs, targets = collate([test_ds[i] for i in range(len(test_ds))])
    with torch.no_grad():
        preds = normalizer.inverse_target(model(inputs))
    targets = normalizer.inverse_target(targets)
    return (torch.linalg.norm(preds - targets) / torch.linalg.norm(targets)).item()


def main(args):
    L.seed_everything(args.seed)
    family = MarkovTrafficEquilibrium(sioux_falls_base_graph(), noise_scale=args.noise_scale)
    (train_ds, val_ds, test_ds), normalizer = build_dataset(
        family, args.n_instances, args.n_val_instances, args.n_test_instances, args.points_per_instance
    )
    print(f"train examples: {len(train_ds)}   val: {len(val_ds)}   test: {len(test_ds)}")

    # Size the model from one transformed example (line-graph structure + feature width).
    sample, _ = train_ds[0]
    model = GraphormerFieldModel(
        n_feats=sample["feats"].shape[-1],
        in_degree=sample["in_degree"],
        out_degree=sample["out_degree"],
        spd=sample["spd"],
        lr=args.lr,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    )
    collate = collate_examples(family)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate),
        DataLoader(val_ds, batch_size=args.batch, collate_fn=collate),
    )
    print(f"test relative error = {relative_error(model, test_ds, normalizer, collate):.4%}")

    # Learned-vs-analytic dynamics on a held-out instance.
    model.eval()
    graph = family.sample_params()
    vi = bind(family, graph)
    analytic_eq, _ = solve_equilibrium(vi, graph.free_flow_time)
    learned_field = conditioned_field(model, family, graph, normalizer)
    traj = simulate(
        lambda c: -learned_field(c),
        ALGORITHMS[args.algo](args.h),
        graph.free_flow_time.clone(),
        args.n_steps,
        project=lambda c: family.project(graph, c),
    )
    learned_eq = traj[-1]
    print(f"learned-field rollout ({len(traj)} steps):")
    print(f"  analytic residual at endpoint ||r|| = {vi.operator(learned_eq).norm():.4e}")
    print(f"  mean rel. distance to analytic equilibrium = {((learned_eq - analytic_eq).abs() / analytic_eq).mean():.2%}")


if __name__ == "__main__":
    main(build_parser().parse_args())
