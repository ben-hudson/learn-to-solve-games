"""Tests for the PUME-based traffic VI family (``PumeMarkovTrafficEquilibrium``).

Covers the two properties the operator swap must guarantee: the PUME excess-supply field has its zero
at the user equilibrium (root correctness), and evaluating a ``[B, E]`` batch equals evaluating each
row on its own (batched consistency), across both the many-distinct-instances rollout path and the
many-points-per-instance dataset path.
"""

import pathlib

import pytest
import torch

from l2s_games.envs import bind, make_game
from l2s_games.envs.pume_traffic import load_sioux_falls_base_graph

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sioux_falls"


@pytest.fixture(scope="module")
def base_graph():
    if not _DATA_ROOT.exists():
        pytest.skip(f"Sioux Falls TNTP data not found at {_DATA_ROOT}")
    return load_sioux_falls_base_graph(str(_DATA_ROOT))


@pytest.fixture(scope="module")
def family(base_graph):
    # Tighten the solver so the fixture's base-graph solve reaches the operator's root closely
    # (the default outer_tol=1e-1 stops well before ||E||~0); root correctness is then unambiguous.
    return make_game("pume_traffic", base_graph=base_graph, solver_kwargs={"outer_tol": 1e-4, "outer_max_iter": 2000})


@pytest.fixture(scope="module")
def base_equilibrium(family):
    """The (canonical) base graph's user-equilibrium cost, solved once via the family's PUME solver."""
    cost, _flow = family.solver.solve(family.base_graph)
    return cost.float()


def test_operator_root_at_equilibrium(family, base_equilibrium):
    """The operator is ~0 at equilibrium and clearly nonzero at a cost perturbed away from it.

    The reference is a uniform +5.0 cost perturbation (still feasible, ``>= free_flow_time``) rather
    than the free-flow start: under supply-diagonal preconditioning the near-free-flow residual is
    deliberately compressed (``s'`` is huge there), so it is not a valid "far from equilibrium" probe.
    """
    graph = family.base_graph
    residual_star = family.operator(graph, base_equilibrium).norm()
    residual_away = family.operator(graph, base_equilibrium + 5.0).norm()
    assert residual_star < 1e-2
    assert residual_away > 1.0


def test_batched_matches_per_instance(family):
    """operator on a stacked [B, E] batch of distinct instances == per-row single-vector evals."""
    torch.manual_seed(0)
    instances = [family.sample_params() for _ in range(3)]
    costs = torch.stack([family.sample_domain(g, 1)[0] for g in instances])  # [B, E]
    batch = family.collate_fn([family.transform(family.model_input(g, c)) for g, c in zip(instances, costs)])
    batched = family.operator(family.params_from_batch(batch), costs)
    per_row = torch.stack([family.operator(g, c) for g, c in zip(instances, costs)])
    assert torch.allclose(batched, per_row, atol=1e-5)


def test_many_points_single_instance(family):
    """operator on one instance at [m, E] points == evaluating each point on its own."""
    torch.manual_seed(1)
    graph = family.sample_params()
    points = family.sample_domain(graph, 4)  # [m, E]
    batched = family.operator(graph, points)
    per_point = torch.stack([family.operator(graph, points[i]) for i in range(points.shape[0])])
    assert torch.allclose(batched, per_point, atol=1e-5)


def test_precondition_is_positive_diagonal_rescale(base_graph, base_equilibrium):
    """Preconditioning applies a positive diagonal metric: same root, same per-coordinate sign.

    The equilibrium is solver-derived (independent of the flag), so it zeroes the raw operator too;
    at an interior cost the preconditioned residual keeps every coordinate's sign (``M^{-1} > 0``) but
    changes scale.
    """
    raw = make_game("pume_traffic", base_graph=base_graph, precondition=False)
    pre = make_game("pume_traffic", base_graph=base_graph, precondition=True)
    assert raw.operator(raw.base_graph, base_equilibrium).norm() < 1e-2
    cost = base_equilibrium + 5.0
    residual_raw = raw.operator(raw.base_graph, cost)
    residual_pre = pre.operator(pre.base_graph, cost)
    nontrivial = residual_raw.abs() > 1e-6
    assert torch.equal(torch.sign(residual_raw[nontrivial]), torch.sign(residual_pre[nontrivial]))
    assert not torch.allclose(residual_raw, residual_pre)


def test_bare_vector_roundtrips_shape(family):
    """A bare [E] cost vector returns a bare [E] residual (the sandbox single-instance path)."""
    torch.manual_seed(2)
    graph = family.sample_params()
    cost = family.sample_domain(graph, 1)[0]
    residual = bind(family, graph).operator(cost)
    assert residual.shape == cost.shape
