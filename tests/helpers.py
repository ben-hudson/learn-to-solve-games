"""Shared test helpers: an operator-call counter, a stream-draining shorthand, and a small solved root.

``CountingFamily`` is the in-process stand-in for ``operator_count.SharedCounter``, which only the traffic
family accepts -- every budget assertion in ``test_caching`` / ``test_expert_recording`` is about how many
point-evaluations a source spends, so it needs to count them for the flat families too.

``solved_traffic_root`` gives a test its own freshly solved dataset. Tests used to read a checked-in
``datasets/sioux_falls_512``, which coupled them to an artifact whose on-disk layout could drift out from
under them -- and did, when solving moved from ``process()`` to ``download()``. Solving a handful of
instances at loose tolerance costs a few seconds and keeps each test's data its own.
"""

import itertools
import pathlib

import pytest

from l2s_games.datasets import EquilibriumDataset
from l2s_games.envs import make_game
from l2s_games.envs.traffic import load_sioux_falls_base_graph

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sioux_falls"
# Loose enough to solve in seconds. Tests that use this are about the data pipeline, not about how
# precisely an equilibrium was reached; anything asserting on solution accuracy should tighten it.
_LOOSE_SOLVER = {"outer_tol": 1e-1, "outer_max_iter": 500}


def solved_traffic_root(root, n_instances=4, game="pume_traffic"):
    """A small ``EquilibriumDataset`` solved fresh into ``root``, or a skip if the TNTP files are absent.

    Mirrors what ``scripts/generate_traffic_dataset.py`` does, including recording ``game`` on the base
    graph so a reader derives its family from the data. The base class attaches no operator examples (and
    solves no calibration set), so the instances carry only their equilibria -- callers that want examples
    construct one of the ``operator_datasets`` classes instead.
    """
    if not _DATA_ROOT.exists():
        pytest.skip(f"Sioux Falls TNTP data not found at {_DATA_ROOT}")
    family = make_game(game, base_graph=load_sioux_falls_base_graph(str(_DATA_ROOT)), solver_kwargs=_LOOSE_SOLVER)
    family.base_graph.game = game
    return EquilibriumDataset(
        str(root),
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solve_instance,
        n_instances=n_instances,
        quiet=True,
    )


class CountingFamily:
    """Wraps a family, counting the point-evaluations its operator is asked for.

    Both entry points are counted, and neither double-counts: whichever the caller enters, the delegate
    call lands on the *wrapped* family, whose internal hop between ``operator`` and ``operator_and_preconditioner``
    never re-enters this wrapper. (Families differ in which way that hop goes -- ``pume_traffic``'s
    ``operator`` delegates to ``operator_and_preconditioner``, the flat families' default does the reverse -- so
    counting only one of them would silently miss whole code paths.)
    """

    def __init__(self, family):
        self._family = family
        self.n_evaluations = 0

    def __getattr__(self, name):
        return getattr(self._family, name)

    def operator(self, params, points):
        self.n_evaluations += len(points)
        return self._family.operator(params, points)

    def operator_and_preconditioner(self, params, points):
        self.n_evaluations += len(points)
        return self._family.operator_and_preconditioner(params, points)


def take(stream, n):
    """The first ``n`` examples of an (infinite) stream."""
    return list(itertools.islice(iter(stream), n))
