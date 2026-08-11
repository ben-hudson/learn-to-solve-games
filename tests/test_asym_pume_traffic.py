"""Tests for the non-potential PUME traffic VI family (``AsymmetricPUMEMarkovTrafficEquilibrium``).

Covers what the supply swap has to guarantee. Two of those are inherited from ``pume_traffic`` and
re-checked here because the new supply is on the path (root correctness, batched consistency), and the
rest are what the family exists for:

- ``epsilon = 0`` reproduces ``pume_traffic`` exactly -- the end-to-end regression test of the whole seam;
- the operator really is **non-potential** (non-symmetric Jacobian), unlike its parent;
- the interaction matrix ``A`` is well formed, is *required* rather than rebuilt, and **round-trips through a
  dataset** -- the invariant that makes a trained model's operator provably the one whose equilibria were
  solved, rather than one that merely ought to match;
- monotonicity is **measured, not assumed**: a PD symmetric part of ``A`` does not imply a monotone
  operator (see the family docstring and ``docs/pume_issues.md``), so the test pins the regime the default
  ``epsilon`` is actually in, and pins the known violation at a larger one.
"""

import pathlib

import pytest
import torch

from l2s_games.equilibrium_datasets import EquilibriumDataset
from l2s_games.envs import bind, make_game
from l2s_games.envs.asym_pume_traffic import (
    build_interaction_matrix,
    build_rotation_matrix,
    coupling_matrices,
)
from l2s_games.envs.pume_traffic import load_sioux_falls_base_graph

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sioux_falls"
_EPSILON = 0.05  # gentle multiplicative coupling, monotone at every cost probed here
_VIOLATING_EPSILON = 0.5  # measurably non-monotone on this network (see docs/pume_issues.md)
_KAPPA = 100.0  # additive coupling well past max f' ~ 46, i.e. rotation-dominated; monotone regardless


@pytest.fixture(scope="module")
def base_graph():
    if not _DATA_ROOT.exists():
        pytest.skip(f"Sioux Falls TNTP data not found at {_DATA_ROOT}")
    return load_sioux_falls_base_graph(str(_DATA_ROOT))


def asym(base_graph, epsilon=_EPSILON, kappa=0.0, **kwargs):
    """A family at ``(epsilon, kappa)``, built the way dataset generation builds one: matrices, then family.

    Both matrices are always constructed explicitly because the family requires them -- so every test below
    exercises the same path ``scripts/generate_traffic_dataset.py`` uses. ``kappa=0`` (no additive coupling)
    is the default, so the multiplicative-coupling tests read unchanged.
    """
    return make_game(
        "asym_pume_traffic",
        base_graph=base_graph,
        interaction_matrix=build_interaction_matrix(base_graph, epsilon),
        rotation_matrix=build_rotation_matrix(base_graph, kappa),
        **kwargs,
    )


@pytest.fixture(scope="module")
def family(base_graph):
    # Tighten the solver so the fixture's base-graph solve reaches the operator's root closely (the
    # default outer_tol=1e-1 stops well before ||E||~0), as in test_pume_traffic.
    return asym(base_graph, solver_kwargs={"outer_tol": 1e-4, "outer_max_iter": 2000})


@pytest.fixture(scope="module")
def base_equilibrium(family):
    """The base graph's *asymmetric* user-equilibrium cost, solved via this family's PUME solver."""
    cost, _flow = family.solver.solve(family.base_graph)
    return cost.float()


def jacobian_by_central_difference(family, params, cost, delta=1e-3):
    """``J[:, i] = dE/dc_i`` of the family's operator, by central differences.

    The PUME operator is not ``jacrev``-transformable (torchdeq/scipy internals), so symmetry of the
    Jacobian -- the property that separates potential from non-potential -- has to be probed numerically.
    """
    columns = []
    for i in range(cost.numel()):
        step = torch.zeros_like(cost)
        step[i] = delta
        columns.append((family.operator(params, cost + step) - family.operator(params, cost - step)) / (2 * delta))
    return torch.stack(columns, dim=1)


def test_zero_coupling_recovers_the_symmetric_family(base_graph):
    """``epsilon = 0, kappa = 0`` gives ``A = I, B = 0``, so the operator must equal ``pume_traffic``'s exactly.

    The cheapest end-to-end check of the whole seam: solver subclass, supply construction, link ordering
    of ``A``, and the family's ``_make_solver`` override. A permuted ``A`` would still be the identity
    here, but any other slip in the path shows up.
    """
    symmetric = make_game("pume_traffic", base_graph=base_graph, precondition=False)
    identity_coupled = asym(base_graph, epsilon=0.0, precondition=False)

    assert torch.equal(identity_coupled.solver.interaction_matrix, torch.eye(base_graph.num_edges, dtype=torch.float64))
    torch.manual_seed(0)
    costs = symmetric.sample_domain(symmetric.base_graph, 3)
    assert torch.allclose(
        identity_coupled.operator(identity_coupled.base_graph, costs),
        symmetric.operator(symmetric.base_graph, costs),
        atol=1e-6,
    )


def test_supply_is_exactly_A_times_inverse_bpr(base_graph):
    """The supply operator is ``A @ f(c)`` with ``A`` in the same link order as the per-edge attrs.

    Ordering is the one silent failure mode of the whole design: ``A`` is built from ``edge_index`` while
    ``f`` is built from the per-edge BPR attrs, and ``_canonicalize`` is what keeps link ``k`` the same
    link in both. A permutation would leave every other test here passing (a permuted ``A`` is still row
    stochastic, still non-symmetric, still PD) while silently solving a different network.
    """
    symmetric = make_game("pume_traffic", base_graph=base_graph)
    asymmetric = asym(base_graph)
    graph = asymmetric.base_graph
    bpr_params = (graph.free_flow_time, graph.capacity, graph.b, graph.power)

    separable_supply = symmetric.solver._make_supply(*bpr_params)
    coupled_supply = asymmetric.solver._make_supply(*bpr_params)
    torch.manual_seed(0)
    cost = symmetric.sample_domain(graph, 1)[0].double()

    expected = asymmetric.solver.interaction_matrix @ separable_supply(cost)
    assert torch.allclose(coupled_supply(cost), expected, atol=0.0)


def test_operator_is_non_potential(base_graph):
    """The Jacobian is non-symmetric -- the property the family exists to provide.

    Measured against the symmetric family at the same cost, which supplies the finite-difference noise
    floor: whatever asymmetry the potential operator shows is numerical, so a real effect has to clear it
    by a wide margin. Both use the raw field, since preconditioning rescales rows and would muddy this.
    """
    graph = make_game("pume_traffic", base_graph=base_graph).base_graph
    cost = graph.free_flow_time.float() * 3.0

    def asymmetry(family):
        jacobian = jacobian_by_central_difference(family, family.base_graph, cost)
        return (jacobian - jacobian.T).abs().max().item()

    noise_floor = asymmetry(make_game("pume_traffic", base_graph=base_graph, precondition=False))
    coupled = asymmetry(asym(base_graph, epsilon=0.2, precondition=False))

    assert coupled > 10.0 * noise_floor, f"asymmetry {coupled:.3g} is within FD noise ({noise_floor:.3g})"


def test_monotonicity_holds_at_the_default_epsilon(base_graph):
    """The raw field is monotone at the default ``epsilon`` -- measured, because it is not guaranteed.

    ``sym(A) > 0`` does *not* imply ``sym(A diag(f'(c))) >= 0`` (see ``docs/pume_issues.md``), so the
    training-time monotonicity prior is only well specified in a regime that has been checked. This pins
    that regime for the default and for the largest coupling documented as safe.
    """
    torch.manual_seed(0)
    for epsilon in (_EPSILON, 0.2):
        family = asym(base_graph, epsilon=epsilon, precondition=False)
        graph = family.base_graph
        products = []
        for _ in range(8):
            first, second = family.sample_domain(graph, 1)[0], family.sample_domain(graph, 1)[0]
            difference = family.operator(graph, first) - family.operator(graph, second)
            products.append((difference * (first - second)).sum().item())
        assert min(products) > 0.0, f"epsilon={epsilon} violated monotonicity (min {min(products):.4g})"


def test_rotation_matrix_is_exactly_antisymmetric(base_graph):
    """``B = -Bᵀ`` and ``⟨B v, v⟩ = 0`` to machine precision.

    This is the whole basis of the additive coupling: an antisymmetric term contributes *exactly* nothing to
    the monotonicity inner product, so it adds rotation at no cost to monotonicity, at any ``kappa``. If the
    construction were only approximately antisymmetric, that guarantee would be approximate too.
    """
    rotation = build_rotation_matrix(base_graph, _KAPPA)
    torch.manual_seed(0)
    vectors = torch.randn(8, base_graph.num_edges, dtype=torch.float64)

    assert torch.equal(rotation, -rotation.T)
    assert torch.allclose(rotation.diagonal(), torch.zeros(base_graph.num_edges, dtype=torch.float64))
    quadratic_forms = ((vectors @ rotation.T) * vectors).sum(dim=-1)
    assert quadratic_forms.abs().max() < 1e-9 * _KAPPA


def test_supply_jacobian_is_analytic_and_its_diagonal_is_separable(base_graph):
    """``jacobian`` matches finite differences, and ``jacobian_diagonal`` is exactly ``A_ii f'_i``.

    The diagonal claim is what makes rotation survive preconditioning: PUME's ``supply_diagonal`` metric is
    built from ``jacobian_diagonal``, and since ``B`` has a zero diagonal the metric sees only the separable
    part. It is also the ``O(m)`` path that avoids materializing a dense Jacobian per evaluation.
    """
    family = asym(base_graph, _EPSILON, _KAPPA)
    graph = family.base_graph
    supply = family.solver._make_supply(graph.free_flow_time, graph.capacity, graph.b, graph.power)
    torch.manual_seed(0)
    cost = family.sample_domain(graph, 1)[0].double()

    delta = 1e-5
    columns = []
    for i in range(cost.numel()):
        step = torch.zeros_like(cost)
        step[i] = delta
        columns.append((supply(cost + step) - supply(cost - step)) / (2 * delta))
    assert torch.allclose(supply.jacobian(cost), torch.stack(columns, dim=1), atol=1e-4)

    diagonal = supply.jacobian_diagonal(cost)
    assert torch.allclose(diagonal, supply.jacobian(cost).diagonal())
    expected = family.solver.interaction_matrix.diagonal() * supply.inner.jacobian_diagonal(cost)
    assert torch.equal(diagonal, expected)


def test_additive_coupling_stays_monotone_where_multiplicative_does_not(base_graph):
    """A rotation-dominated additive coupling is monotone; the multiplicative form at 0.5 is not.

    Both halves matter. The additive claim is exact maths (``⟨BΔc, Δc⟩ = 0``) so this is really an
    implementation check; pairing it with the multiplicative failure at the same ``kappa``-free setting is
    what shows the two couplings are genuinely different instruments rather than reparametrizations.
    """
    rotational = asym(base_graph, epsilon=0.0, kappa=_KAPPA, precondition=False)
    graph = rotational.base_graph
    torch.manual_seed(0)
    products = []
    for _ in range(8):
        first, second = rotational.sample_domain(graph, 1)[0], rotational.sample_domain(graph, 1)[0]
        difference = rotational.operator(graph, first) - rotational.operator(graph, second)
        products.append((difference * (first - second)).sum().item())

    assert min(products) > 0.0, f"kappa={_KAPPA} violated monotonicity (min {min(products):.4g})"


def test_monotonicity_fails_at_large_epsilon(base_graph):
    """A large ``epsilon`` *is* non-monotone, even though ``A`` passes PUME's PD check.

    The counterpart to the test above, and the reason the family docstring caps the recommended range: it
    pins the failure as a known property of the construction rather than a surprise, so that raising the
    default cannot silently invalidate the monotonicity prior. Purely analytic -- no finite differences.
    """
    family = asym(base_graph, epsilon=_VIOLATING_EPSILON, precondition=False)
    graph = family.base_graph
    interaction = family.solver.interaction_matrix
    supply = family.solver._make_supply(graph.free_flow_time, graph.capacity, graph.b, graph.power)
    torch.manual_seed(3)
    cost = family.sample_domain(graph, 1)[0].double()

    symmetric_part = lambda matrix: torch.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min().item()
    supply_jacobian = interaction @ torch.diag(supply.inner.jacobian_diagonal(cost))

    assert symmetric_part(interaction) > 0.0  # A passes PUME's validate_monotone check ...
    assert symmetric_part(supply_jacobian) < 0.0  # ... yet grad z is not positive semi-definite


def test_interaction_matrix_invariants(base_graph):
    """``A = (1 - eps) I + eps W``: row-stochastic, non-symmetric, PD symmetric part, node-adjacent only."""
    family = asym(base_graph)
    interaction = family.solver.interaction_matrix
    n_edges = base_graph.num_edges

    assert interaction.shape == (n_edges, n_edges)
    assert torch.allclose(interaction.sum(dim=1), torch.ones(n_edges, dtype=interaction.dtype))
    assert (interaction - interaction.T).abs().max() > 0.0  # non-symmetric => non-potential
    assert torch.linalg.eigvalsh(0.5 * (interaction + interaction.T)).min() > 0.0

    # Couplings are restricted to links sharing a node, so A is supported on the link-adjacency pattern.
    tails, heads = family.base_graph.edge_index
    endpoints = torch.zeros(base_graph.num_nodes, n_edges, dtype=torch.bool)
    endpoints[tails, torch.arange(n_edges)] = True
    endpoints[heads, torch.arange(n_edges)] = True
    adjacent = (endpoints.T.float() @ endpoints.float()) > 0
    assert not interaction[~adjacent].any(), "A couples links that share no node"


def test_omitted_couplings_come_from_the_graph_or_fail_loudly(base_graph):
    """Omitting the coupling kwargs means "read the stored ones off ``base_graph``", never "build fresh".

    A rebuilt ``A`` only *probably* matches the one a dataset's equilibria were solved with -- its values
    come from an RNG seeded by a PUME default we neither pass nor record -- and a mismatch is silent. So a
    bare graph (no stored couplings) must refuse construction with the regenerate message, and a graph that
    carries them must hand back exactly those.
    """
    with pytest.raises(AssertionError, match="Regenerate the dataset"):
        make_game("asym_pume_traffic", base_graph=base_graph)

    stored = {
        "interaction_matrix": build_interaction_matrix(base_graph, _EPSILON),
        "rotation_matrix": build_rotation_matrix(base_graph, _KAPPA),
    }
    carrying = base_graph.clone()
    for name, matrix in stored.items():
        carrying[name] = matrix
    family = make_game("asym_pume_traffic", base_graph=carrying)
    for name, matrix in stored.items():
        assert torch.equal(getattr(family, name), matrix), name


def test_the_supplied_matrices_are_the_ones_used(base_graph):
    """Changing either coupling changes the operator, so nothing is quietly rebuilt behind them."""
    torch.manual_seed(0)
    base = asym(base_graph, 0.05, precondition=False)
    cost = base.sample_domain(base.base_graph, 1)[0]
    reference = base.operator(base.base_graph, cost)

    for label, family in [
        ("multiplicative", asym(base_graph, 0.4, precondition=False)),
        ("additive", asym(base_graph, 0.05, kappa=1.0, precondition=False)),
    ]:
        assert not torch.allclose(family.operator(family.base_graph, cost), reference, atol=1e-4), label


def test_interaction_matrix_round_trips_through_a_dataset(base_graph, tmp_path):
    """``A`` survives being cached with a dataset and read back -- the invariant this design exists for.

    Generation stores the matrix on ``base_graph``, so the dataset carries the operator its equilibria were
    solved for and training reads it back instead of reconstructing it. Loose solver tolerances here: the
    equilibria's *accuracy* is not what is under test, their operator's identity is.
    """
    expected = {
        "interaction_matrix": build_interaction_matrix(base_graph, _EPSILON),
        "rotation_matrix": build_rotation_matrix(base_graph, _KAPPA),
    }
    family = asym(base_graph, _EPSILON, _KAPPA, solver_kwargs={"outer_tol": 1e-1, "outer_max_iter": 50})
    EquilibriumDataset(
        str(tmp_path),
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solve_instance,
        n_instances=2,
        quiet=True,
    )

    reloaded = EquilibriumDataset(str(tmp_path))
    assert coupling_matrices(reloaded.base_graph).keys() == expected.keys()
    for name, matrix in expected.items():
        assert torch.equal(reloaded.base_graph[name], matrix), name


def test_coupling_matrices_never_reach_the_collated_batch(base_graph):
    """Both couplings ride on the instance graph for persistence but are dropped before collation.

    Otherwise every batch would carry redundant ``[B, E, E]`` tensors -- the operator reads them off the
    solver, and the model never sees them at all.
    """
    family = asym(base_graph, kappa=_KAPPA)
    torch.manual_seed(0)
    instance = family.sample_params()
    batch = family.collate_fn([family.transform(family.model_input(instance, family.sample_domain(instance, 1)[0]))])

    for name in ("interaction_matrix", "rotation_matrix"):
        assert name in instance, name  # cloned from base_graph, so they are there to begin with
        assert name not in batch, name


def test_operator_root_at_equilibrium(family, base_equilibrium):
    """The operator is ~0 at the asymmetric equilibrium and clearly nonzero away from it.

    The reference is a uniform +5.0 cost perturbation rather than the free-flow start: under
    supply-diagonal preconditioning the near-free-flow residual is deliberately compressed (``s'`` is huge
    there), so it is not a valid "far from equilibrium" probe.
    """
    graph = family.base_graph
    assert family.operator(graph, base_equilibrium).norm() < 1e-2
    assert family.operator(graph, base_equilibrium + 5.0).norm() > 1.0


def test_equilibrium_differs_from_the_symmetric_one(base_graph, family, base_equilibrium):
    """The asymmetric equilibrium is a different point, which is why it needs its own solved dataset.

    The cached ``equilibrium`` sets the sampling range (``calibrate_ceiling``) and the ``rel_dist``
    metrics, so reusing a symmetric cache would calibrate around the wrong solution.
    """
    symmetric = make_game(
        "pume_traffic", base_graph=base_graph, solver_kwargs={"outer_tol": 1e-4, "outer_max_iter": 2000}
    )
    symmetric_equilibrium, _flow = symmetric.solver.solve(symmetric.base_graph)

    relative_gap = (base_equilibrium - symmetric_equilibrium.float()).norm() / symmetric_equilibrium.float().norm()
    assert (
        relative_gap > 1e-3
    ), f"asymmetric equilibrium is indistinguishable from the symmetric one ({relative_gap:.3g})"


def test_batched_matches_per_instance(family):
    """operator on a stacked [B, E] batch of distinct instances == per-row single-vector evals."""
    torch.manual_seed(0)
    instances = [family.sample_params() for _ in range(3)]
    costs = torch.stack([family.sample_domain(graph, 1)[0] for graph in instances])  # [B, E]
    batch = family.collate_fn([family.transform(family.model_input(g, c)) for g, c in zip(instances, costs)])

    batched = family.operator(family.params_from_batch(batch), costs)
    per_row = torch.stack([family.operator(g, c) for g, c in zip(instances, costs)])
    assert torch.allclose(batched, per_row, atol=1e-5)


def test_many_points_single_instance(family):
    """operator on one instance at [m, E] points == evaluating each point on its own."""
    torch.manual_seed(1)
    graph = family.sample_params()
    points = family.sample_domain(graph, 4)

    batched = family.operator(graph, points)
    per_point = torch.stack([family.operator(graph, points[i]) for i in range(points.shape[0])])
    assert torch.allclose(batched, per_point, atol=1e-5)


def test_bare_vector_roundtrips_shape(family):
    """A bare [E] cost vector returns a bare [E] residual (the sandbox single-instance path)."""
    torch.manual_seed(2)
    graph = family.sample_params()
    cost = family.sample_domain(graph, 1)[0]

    assert bind(family, graph).operator(cost).shape == cost.shape
