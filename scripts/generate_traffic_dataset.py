"""
generate_traffic_dataset.py

Pre-compute and cache a dataset of noised SiouxFalls instances, each solved to user equilibrium by
``PUMESolver``. Solving is the expensive part, so this is run once, offline; training then loads the
cache, splits it into bootstrap / val / test, and calibrates the streaming sampling range from the
bootstrap equilibria (see ``MarkovTrafficEquilibrium.calibrate_range`` and ``train_field_gnn.py``).

    python scripts/generate_traffic_dataset.py 1024 data/sioux_falls/solved

Requires the ``pume`` / ``pumcm`` packages (imported lazily by ``l2s_games.pume_solver``).
"""

import argparse

import lightning as L

from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs.traffic import MarkovTrafficEquilibrium, load_sioux_falls_base_graph
from l2s_games.pume_solver import PUMESolver


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("n_instances", type=int, help="number of noised instances to generate and solve")
    p.add_argument("root", type=str, help="dataset root (holds raw/base_graph.pt and processed/instances.pt)")
    p.add_argument(
        "--data-root",
        type=str,
        default="data/sioux_falls",
        help="root location of the SiouxFalls_*.tntp files (local directory or URL)",
    )
    p.add_argument("--noise-scale", type=float, default=0.2, help="multiplicative attribute noise")
    p.add_argument("--noise-type", choices=["normal", "uniform"], default="normal", help="attribute noise type")
    p.add_argument("--seed", type=int, default=0, help="global seed")
    # PUMESolver tolerances (see l2s_games/pume_solver.py)
    p.add_argument("--inner-max-iter", type=int, default=3000, help="inner modified-policy-iteration max iters")
    p.add_argument("--inner-tol", type=float, default=1e-7, help="inner solver tolerance")
    p.add_argument("--outer-max-iter", type=int, default=500, help="outer equilibrium-iteration max iters")
    p.add_argument("--outer-tol", type=float, default=1e-1, help="outer equilibrium tolerance")
    return p


def main(args):
    L.seed_everything(args.seed)
    base_graph = load_sioux_falls_base_graph(args.data_root)
    # One family gives the canonical base graph and the noising distribution; the solver is built once
    # from that same canonical structure so the equilibrium tensors line up with sample_domain's edges.
    family = MarkovTrafficEquilibrium(base_graph, noise_scale=args.noise_scale, noise_type=args.noise_type)
    solver = PUMESolver(
        family.base_graph,
        inner_max_iter=args.inner_max_iter,
        inner_tol=args.inner_tol,
        outer_max_iter=args.outer_max_iter,
        outer_tol=args.outer_tol,
    )
    dataset = SolvedInstanceDataset(
        args.root,
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=solver.solve,
        n_instances=args.n_instances,
    )
    print(f"generated {len(dataset)} solved instances at {args.root}")


if __name__ == "__main__":
    main(build_parser().parse_args())
