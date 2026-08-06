"""
generate_traffic_dataset.py

Pre-compute and cache a dataset of noised SiouxFalls instances, in the two stages
``datasets.EquilibriumDataset`` splits its work into -- because their costs differ by ~1800x:

- **solving** every instance to user equilibrium (~2.5 s each), cached as the *raw* artifact, and
- **evaluating the operator** at cost points per instance (~1.4 ms each), cached as the *processed* one.

    python scripts/generate_traffic_dataset.py 1024 data/sioux_falls/solved \
        --operator-dataset uniform --points_per_instance 32

The staging is why this is one command rather than two. Re-running with different ``process()``-stage
arguments and ``--force-reload`` regenerates the operator examples over the **existing solves** -- seconds
rather than the ~40 minutes a re-solve would cost. Only changing the solve stage (``n_instances``, the
solver tolerances) requires deleting ``raw/``.

``--operator-dataset none`` gives a solve-only root, for the fully-amortized model (which needs the
equilibria and nothing else) or for the tuning and probe scripts.

``--game`` picks which operator the equilibria are solved for, and the choice is **load-bearing**: the
asymmetric family's equilibrium is a different point, and the cached ``equilibrium_cost`` is both the
sampling-ceiling calibration and the ``rel_dist`` reference during training. It is recorded on the root's
``base_graph``, so readers derive the family from the data instead of being told. Generate a separate root
per (game, epsilon, kappa):

    python scripts/generate_traffic_dataset.py 1024 data/sioux_falls/solved_asym_eps0.05 \
        --game asym_pume_traffic --epsilon 0.05 --operator-dataset expert --n_steps 500

This script is also the **only** place the asymmetric interaction matrix ``A`` is built. It is stored on the
dataset's ``base_graph``, so the dataset carries the operator its equilibria were solved for and training
reads it back rather than reconstructing it. See ``envs/asym_pume_traffic.py`` for why rebuilding it is
unsafe, and for how to choose ``--epsilon`` / ``--kappa``.

Requires the ``pume`` / ``pumcm`` packages (imported lazily by ``l2s_games.pume_solver``).
"""

import argparse
import functools

import lightning as L

from l2s_games.datasets import EquilibriumDataset
from l2s_games.envs import make_game
from l2s_games.envs.asym_pume_traffic import build_interaction_matrix, build_rotation_matrix
from l2s_games.envs.traffic import load_sioux_falls_base_graph
from l2s_games.operator_datasets import POINT_SOURCES


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("n_instances", type=int, help="number of noised instances to generate and solve")
    p.add_argument("root", type=str, help="dataset root (holds raw/ and processed/)")
    # --- process() stage: the operator examples, cheap and regenerable over the existing solves ------
    p.add_argument(
        "--operator-dataset",
        choices=["none", *POINT_SOURCES],
        default="uniform",
        help="where each instance's cost points come from: 'uniform' over the calibrated cost box, "
        "'expert' along a converging rollout of the true operator (plus the equilibrium it reaches), or "
        "'none' for a solve-only root",
    )
    p.add_argument(
        "--points_per_instance",
        type=int,
        default=32,
        help="cost points per instance ('uniform' only -- the expert's count follows from --n_steps and "
        "the algorithm's field evaluations per step)",
    )
    p.add_argument(
        "--n_cal_instances",
        type=int,
        default=128,
        help="leading instances whose equilibria calibrate the sampling ceiling. Leading, so that a "
        "consumer splitting train/val/test with train first never calibrates on a held-out equilibrium",
    )
    p.add_argument(
        "--sample_stds",
        type=float,
        default=3.0,
        help="sigma above the mean calibration equilibrium that the per-edge cost box reaches",
    )
    # Expert rollout only. projection is the measured best choice on the potential and multiplicatively
    # coupled operators; switch to optimistic once additive coupling makes the field rotation-dominated
    # (see envs/asym_pume_traffic.py) and re-tune with scripts/tune_pume_rollout.py.
    p.add_argument("--algo", choices=["projection", "extragradient", "optimistic", "momentum"], default="projection")
    p.add_argument("--h", type=float, default=0.1, help="expert rollout step size")
    p.add_argument("--n_steps", type=int, default=500, help="expert rollout iterations")
    p.add_argument(
        "--force-reload",
        action="store_true",
        help="regenerate the operator examples, reusing the cached solves (PyG skips download() whenever "
        "raw/ exists, so this only re-runs the cheap stage)",
    )
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
    couplings = (
        {
            "interaction_matrix": build_interaction_matrix(base_graph, args.epsilon),
            "rotation_matrix": build_rotation_matrix(base_graph, args.kappa),
        }
        if args.game == "asym_pume_traffic"
        else {}
    )
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
        **couplings,
    )
    # Record which family this root belongs to, so a reader derives it from the data instead of being told
    # (a mismatched --game would condition the model on the wrong operator silently). Rides on base_graph
    # like the coupling matrices, and is stripped by model_input -- see traffic._DROPPED_ATTRS.
    family.base_graph.game = args.game
    # The process() stage: attach each instance's operator examples. Everything the point source needs is
    # bound here except the instances themselves, which it receives *solved* so it can calibrate its own
    # sampling ceiling from their equilibria (see operator_datasets.POINT_SOURCES).
    evaluate_fn = None
    if args.operator_dataset != "none":
        point_args = (
            {"algo": args.algo, "h": args.h, "n_steps": args.n_steps}
            if args.operator_dataset == "expert"
            else {"points_per_instance": args.points_per_instance}
        )
        evaluate_fn = functools.partial(
            POINT_SOURCES[args.operator_dataset],
            game=args.game,
            base_graph=family.base_graph,
            n_cal=args.n_cal_instances,
            n_stds=args.sample_stds,
            **point_args,
            **couplings,
        )
    dataset = EquilibriumDataset(
        args.root,
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solver.solve,
        n_instances=args.n_instances,
        evaluate_fn=evaluate_fn,
        force_reload=args.force_reload,
    )
    summary = f"{len(dataset)} solved instances at {args.root}"
    if args.operator_dataset != "none":
        points = dataset[0].points.shape[0]
        summary += f", each with {points} {args.operator_dataset} operator examples ({len(dataset) * points} total)"
    print(f"generated {summary}")


if __name__ == "__main__":
    main(build_parser().parse_args())
