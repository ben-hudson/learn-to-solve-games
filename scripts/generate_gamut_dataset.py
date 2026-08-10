"""
generate_gamut_dataset.py

Pre-compute and cache a dataset of GAMUT two-player games, in the same two stages as the traffic script
(see ``generate_traffic_dataset.py`` for the staging rationale, which carries over unchanged): **solving**
every instance to equilibrium by rolling out the true operator (cached as the raw artifact by
``datasets.EquilibriumDataset``), then **evaluating the operator** at points per instance (the processed
artifact of whichever ``operator_datasets`` subclass ``--operator_dataset`` names).

    python scripts/generate_gamut_dataset.py 256 datasets/gamut_zerosum4 \\
        --gamut_class RandomZeroSum --actions 4 --operator_dataset expert --n_steps 200

``--gamut_class`` picks the GAMUT generator the instances are drawn from, and like the traffic script's
``--game`` it is load-bearing and recorded on the root's ``base_graph`` (as ``game`` = "gamut" plus the
GAMUT provenance), so readers derive the family from the data. Requires a Java runtime and ``gamut.jar``
(``--jar`` or ``$GAMUT_JAR``); see ``l2s_games/gamut.py``.

Convergence caveat: the rollout solver carries a guarantee only for monotone games (zero-sum). The script
therefore reports the worst **natural-map** residual over the solved instances -- the constrained
convergence measure (``||F||`` itself need not vanish at a boundary equilibrium) -- so a bad root is
visible at generation time rather than as a silently wrong ``rel_dist`` reference during training.
"""

import argparse

import lightning as L
import torch

from l2s_games.datasets import EquilibriumDataset
from l2s_games.dynamics import natural_map
from l2s_games.envs import make_game
from l2s_games.envs.gamut import build_gamut_base_graph
from l2s_games.gamut import default_jar
from l2s_games.operator_datasets import OPERATOR_DATASETS, ExpertOperatorDataset, UniformOperatorDataset

# Sizing flags per generator class: RandomZeroSum has no -players; the fixed 2x2 classics take nothing.
_SIZED_CLASSES = ("RandomGame", "CovariantGame")
_ACTION_ONLY_CLASSES = ("RandomZeroSum",)


def gamut_options(args):
    """The class-specific GAMUT flags, assembled once and persisted on the base graph."""
    if args.gamut_class in _SIZED_CLASSES:
        options = ["-players", "2", "-actions", str(args.actions), str(args.actions)]
        return [*options, "-r", str(args.r)] if args.gamut_class == "CovariantGame" else options
    if args.gamut_class in _ACTION_ONLY_CLASSES:
        return ["-actions", str(args.actions), str(args.actions)]
    return []  # fixed 2x2 classics (MatchingPennies, PrisonersDilemma, ...): no parameters


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("n_instances", type=int, help="number of GAMUT instances to generate and solve")
    p.add_argument("root", type=str, help="dataset root (holds raw/ and processed/)")
    # --- the game distribution ------------------------------------------------------------------------
    p.add_argument(
        "--gamut_class",
        default="RandomZeroSum",
        help="GAMUT generator class the instances are drawn from. RandomZeroSum is monotone (the rollout "
        "solver's guarantee holds); CovariantGame/RandomGame are not in general -- watch the residual "
        "report. Fixed 2x2 classics (MatchingPennies, ...) work too and ignore --actions",
    )
    p.add_argument("--actions", type=int, default=4, help="actions per player (shared: square payoffs)")
    p.add_argument("--r", type=float, default=-0.9, help="CovariantGame payoff covariance in [-1, 1]")
    p.add_argument("--jar", default=default_jar(), help="path to gamut.jar (default: $GAMUT_JAR)")
    p.add_argument(
        "--payoff_range",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        help="GAMUT -normalize bounds; a family-level constant the conditioning relies on",
    )
    # --- the solve stage (rollout on the true operator) ------------------------------------------------
    p.add_argument("--solve_algo", default="extragradient", help="rollout algorithm for solve_instance")
    p.add_argument("--solve_h", type=float, default=0.2, help="solve rollout step size (measured: ~1e-7 "
                   "residual at 2000 steps on RandomZeroSum; 0.05 stalls at ~1e-2)")
    p.add_argument("--solve_steps", type=int, default=2000, help="solve rollout length")
    p.add_argument(
        "--n_cal_instances",
        type=int,
        default=8,
        help="extra instances solved into the operator classes' disjoint calibration set. The GAMUT "
        "family's sampling box is the simplex itself, so nothing is calibrated from them "
        "(calibration_kwargs is empty) -- kept small; the structural disjointness is what matters",
    )
    # --- process() stage: the operator examples --------------------------------------------------------
    p.add_argument(
        "--operator_dataset",
        choices=["none", *OPERATOR_DATASETS],
        default="uniform",
        help="where each instance's points come from: 'uniform' over the players' simplices, 'expert' "
        "along a converging rollout (plus the equilibrium it reaches), 'none' for a solve-only root",
    )
    p.add_argument("--points_per_instance", type=int, default=32, help="points per instance ('uniform' only)")
    p.add_argument("--sample_stds", type=float, default=3.0, help="calibration spread (inert for gamut)")
    p.add_argument(
        "--algo",
        default="extragradient",
        help="expert rollout algorithm. Like the solve: the zero-sum field is rotation-dominated, so "
        "'projection' cycles -- keep to the extragradient family",
    )
    p.add_argument("--h", type=float, default=0.05, help="expert rollout step size")
    p.add_argument("--n_steps", type=int, default=200, help="expert rollout length")
    p.add_argument("--seed", type=int, default=0, help="global seed (drives the per-instance GAMUT seeds)")
    p.add_argument("--force_reload", action="store_true", help="regenerate the named source's processed file")
    return p


def worst_natural_map_residual(family, instances):
    """max over instances of ``||natural_map(z*)||`` -- the generation-time convergence report."""
    residuals = [natural_map(family, instance, instance.equilibrium).norm() for instance in instances]
    return torch.stack(residuals).max().item()


def main(args):
    L.seed_everything(args.seed)
    base_graph = build_gamut_base_graph(args.gamut_class, args.actions, options=gamut_options(args))
    base_graph.game = "gamut"  # readers derive the family from the root, as with the traffic script
    family = make_game(
        "gamut",
        base_graph=base_graph,
        jar_path=args.jar,
        payoff_range=tuple(args.payoff_range),
        solve_algo=args.solve_algo,
        solve_h=args.solve_h,
        solve_steps=args.solve_steps,
    )
    solve_kwargs = dict(
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solve_instance,
        n_instances=args.n_instances,
        force_reload=args.force_reload,
    )
    operator_kwargs = dict(n_cal_instances=args.n_cal_instances, n_stds=args.sample_stds, **solve_kwargs)
    if args.operator_dataset == "none":
        dataset = EquilibriumDataset(args.root, **solve_kwargs)
    elif args.operator_dataset == "uniform":
        dataset = UniformOperatorDataset(args.root, points_per_instance=args.points_per_instance, **operator_kwargs)
    else:
        dataset = ExpertOperatorDataset(args.root, algo=args.algo, h=args.h, n_steps=args.n_steps, **operator_kwargs)
    summary = f"{dataset.len()} solved {args.gamut_class} instances at {args.root}"
    if args.operator_dataset != "none":
        summary += f", each with {dataset.points_per_instance} {args.operator_dataset} operator examples"
    print(f"generated {summary}")
    print(f"worst natural-map residual at the cached equilibria: {worst_natural_map_residual(family, dataset.solved_instances()):.3g}")


if __name__ == "__main__":
    main(build_parser().parse_args())
