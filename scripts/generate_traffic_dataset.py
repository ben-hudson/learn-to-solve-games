"""
generate_traffic_dataset.py

Pre-compute and cache a dataset of noised SiouxFalls instances, each solved to user equilibrium by the
family's own PUME solver. Solving is the expensive part, so this is run once, offline; training then
loads the cache, splits it into cal / val / test, and calibrates the streaming sampling range from the
calibration equilibria (see ``calibrate_ceiling`` and ``train_field_gnn.py``).

    python scripts/generate_traffic_dataset.py 1024 data/sioux_falls/solved

``--game`` picks which operator the equilibria are solved for, and the choice is **load-bearing**: the
asymmetric family's equilibrium is a different point, and the cached ``equilibrium_cost`` is both the
sampling-range calibration and the ``rel_dist`` reference during training. Generate a separate root per
(game, epsilon):

    python scripts/generate_traffic_dataset.py 1024 data/sioux_falls/solved_asym_eps0.05 \
        --game asym_pume_traffic --epsilon 0.05

This script is also the **only** place the asymmetric interaction matrix ``A`` is built. It is stored on the
dataset's ``base_graph``, so the dataset carries the operator its equilibria were solved for and training
reads it back rather than reconstructing it -- ``train_field_gnn.py`` has no ``--epsilon``. See
``envs/asym_pume_traffic.py`` for why rebuilding it is unsafe, and for how to choose ``--epsilon``.

Requires the ``pume`` / ``pumcm`` packages (imported lazily by ``l2s_games.pume_solver``).
"""

import argparse

import lightning as L

from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs import make_game
from l2s_games.envs.asym_pume_traffic import build_interaction_matrix, build_rotation_matrix
from l2s_games.envs.traffic import load_sioux_falls_base_graph


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
    p.add_argument(
        "--game",
        type=str,
        default="pume_traffic",
        choices=["pume_traffic", "asym_pume_traffic"],
        help="family whose operator the equilibria are solved for; must match training's --game",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="asym_pume_traffic only: *multiplicative* coupling strength of A = (1-eps) I + eps W (neighbour "
        "spillover). Capped by monotonicity: <=0.2 measured monotone on Sioux Falls, 0.5 is not",
    )
    p.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="asym_pume_traffic only: *additive* coupling strength of B = kappa S, S antisymmetric of unit "
        "spectral norm. Monotone at any kappa (<B dc, dc> = 0 exactly) and invisible to the preconditioner, "
        "so this is the knob for a rotation-dominated field; compare against max f' ~ 46. See "
        "envs/asym_pume_traffic.py. Both matrices are built here only and stored with the dataset",
    )
    p.add_argument("--noise-scale", type=float, default=0.2, help="multiplicative attribute noise")
    p.add_argument("--noise-type", choices=["normal", "uniform"], default="normal", help="attribute noise type")
    p.add_argument("--seed", type=int, default=0, help="global seed")
    # PUMESolver tolerances (see l2s_games/pume_solver.py)
    p.add_argument("--inner-max-iter", type=int, default=3000, help="inner modified-policy-iteration max iters")
    p.add_argument("--inner-tol", type=float, default=1e-7, help="inner solver tolerance")
    # 8000 rather than PUMESolver's 500: the outer iteration descends monotonically but slowly, so a hard
    # instance runs out of budget mid-descent (converged=False, "Maximum iterations reached") and caches a
    # cost that is not an equilibrium. Easy instances stop early, so a generous cap costs nothing.
    p.add_argument("--outer-max-iter", type=int, default=8000, help="outer equilibrium-iteration max iters")
    p.add_argument(
        "--initial-stepsize",
        type=float,
        default=2e-1,
        help="aGRAAL initial step size; 2e-1 converges ~3x faster than 5e-2 on both operators (see "
        "l2s_games/pume_solver.py)",
    )
    # PUMESolver's own default (1e-1) is far too loose to *cache*: it stops early, leaving a natural-map
    # residual of ~0.2-1.0 at the returned cost instead of ~1e-3, and equilibrium_cost is both the
    # sampling-range calibration and the rel_dist reference for every training run that reads this cache.
    # Measured on Sioux Falls (per instance, tol -> residual / relative distance to a converged solve):
    # 1e-1 -> 0.22 / 1.1e-2,  1e-2 -> 2.1e-2 / 1.0e-3,  1e-3 -> 9.2e-4 / 3.5e-5,  1e-4 -> 1.4e-4 / 5.3e-6.
    # 1e-3 costs ~2.5 s/instance against ~1.2 s at 1e-1 -- worth it for a one-time offline job. The
    # asymmetric operator converges to the same residuals in the same time, so this is not game-specific.
    p.add_argument("--outer-tol", type=float, default=1e-3, help="outer equilibrium tolerance")
    return p


def main(args):
    L.seed_everything(args.seed)
    base_graph = load_sioux_falls_base_graph(args.data_root)
    # The family owns both halves: it gives the canonical base graph and the noising distribution, and
    # builds its own solver from that same canonical structure (so the equilibrium tensors line up with
    # sample_domain's edges) with the supply operator that matches --game. The asymmetric family also needs
    # its interaction matrix, built here and only here; the family stores it on base_graph so the cache
    # carries it (see the module docstring).
    family = make_game(
        args.game,
        base_graph=base_graph,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type,
        solver_kwargs={
            "inner_max_iter": args.inner_max_iter,
            "inner_tol": args.inner_tol,
            "outer_max_iter": args.outer_max_iter,
            "outer_tol": args.outer_tol,
            "initial_stepsize": args.initial_stepsize,
        },
        **(
            {
                "interaction_matrix": build_interaction_matrix(base_graph, args.epsilon),
                "rotation_matrix": build_rotation_matrix(base_graph, args.kappa),
            }
            if args.game == "asym_pume_traffic"
            else {}
        ),
    )
    dataset = SolvedInstanceDataset(
        args.root,
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solver.solve,
        n_instances=args.n_instances,
    )
    print(f"generated {len(dataset)} solved instances at {args.root}")


if __name__ == "__main__":
    main(build_parser().parse_args())
