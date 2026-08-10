"""Unit tests for the GAMUT family: the parser, the charted operator, the projection, and the solve.

All offline -- payoff matrices are constructed by hand, so nothing here needs Java or ``gamut.jar``
(the real-invocation tests live in ``test_gamut_datasets.py`` behind a skip). The ground truths are the
closed bimatrix form ``F(x, y) = (-Ay, -B^T x)`` and games whose Nash is known exactly: Matching Pennies
(interior, at the uniform strategies) and a dominance-solvable zero-sum game (pure, on the simplex
boundary -- where ``||F||`` stays away from zero and only the natural map vanishes).
"""

import torch
from torch.func import jacrev

from l2s_games.dynamics import natural_map
from l2s_games.envs.gamut import GamutGame, build_gamut_base_graph
from l2s_games.gamut import parse_simple_output

MATCHING_PENNIES = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
# Row 1 dominates row 2 and column 1 then dominates column 2: the unique (zero-sum) Nash is the pure
# profile (action 1, action 1) -- a boundary point of both simplices.
DOMINANCE_SOLVABLE = torch.tensor([[1.0, 2.0], [0.0, 1.0]])


def bimatrix(a, b):
    """A family + instance for the hand-written bimatrix ``(A, B)``, in the per-edge convention."""
    graph = build_gamut_base_graph("HandWritten", a.shape[0])
    instance = graph.clone()
    instance.payoff = torch.stack([a, b.T])  # payoff[e] = M_ij for edge i -> j
    return GamutGame(graph), instance


def test_parse_simple_output_reads_the_gamut_grammar():
    """Comment lines skipped, 1-indexed action profiles, one payoff per player."""
    text = """
    # Players: 2
    # Actions: 2 2
    [1 1] : [ 0.5 -0.5 ]
    [2 1] : [ -0.25 0.25 ]
    [1 2] : [ -1.0 1.0 ]
    [2 2] : [ 1.0 -1.0 ]
    """
    a, b = parse_simple_output(text, 2)
    assert torch.equal(a, torch.tensor([[0.5, -1.0], [-0.25, 1.0]]))
    assert torch.equal(b, -a)


def test_operator_is_the_charted_bimatrix_field():
    """The per-edge polymatrix aggregation reduces, for two players, to ``(-Ay, -B^T x)`` pulled through
    the Helmert bases -- pinned against the closed form so the edge convention cannot silently flip."""
    torch.manual_seed(0)
    a, b = 2 * torch.rand(3, 3) - 1, 2 * torch.rand(3, 3) - 1
    family, instance = bimatrix(a, b)
    z = torch.randn(family.domain_dim) * 0.1
    x, y = family._strategies(z).unbind(-2)

    expected = torch.cat([(-a @ y) @ family.basis, (-b.T @ x) @ family.basis])
    assert torch.allclose(family.operator(instance, z), expected, atol=1e-6)


def test_matching_pennies_nash_is_the_operator_zero():
    """The chart sits at the uniform strategy, which for Matching Pennies *is* the (interior) Nash."""
    family, instance = bimatrix(MATCHING_PENNIES, -MATCHING_PENNIES)
    assert family.operator(instance, torch.zeros(family.domain_dim)).norm() < 1e-7


def test_zero_sum_field_is_purely_rotational():
    """``<F(z) - F(z'), z - z'> = 0`` for a zero-sum game: the Jacobian is skew, the monotone case the
    rollout solver's guarantee covers."""
    torch.manual_seed(1)
    a = 2 * torch.rand(4, 4) - 1
    family, instance = bimatrix(a, -a)
    z, other = torch.randn(2, family.domain_dim).unbind(0)

    gap = torch.dot(family.operator(instance, z) - family.operator(instance, other), z - other)
    assert gap.abs() < 1e-5

    jacobian = jacrev(lambda point: family.operator(instance, point))(z)  # jacrev composes through it
    assert torch.allclose(jacobian + jacobian.T, torch.zeros_like(jacobian), atol=1e-5)


def test_operator_shapes_match_the_family_contract():
    """``[d]``, ``[N, d]``, and ``[B, d]`` against batched ``[B, E, n, n]`` payoffs (the dense collated
    batch), the batched rows agreeing with per-instance evaluation."""
    torch.manual_seed(2)
    payoffs = [2 * torch.rand(2, 3, 3) - 1 for _ in range(3)]
    family, instance = bimatrix(payoffs[0][0], payoffs[0][1].T)
    points = torch.randn(3, family.domain_dim)

    assert family.operator(instance, points[0]).shape == (family.domain_dim,)
    assert family.operator(instance, points).shape == points.shape

    batch = {"edge_index": instance.edge_index, "payoff": torch.stack(payoffs)}
    batched = family.operator(batch, points)
    for row, payoff in enumerate(payoffs):
        instance.payoff = payoff
        assert torch.allclose(batched[row], family.operator(instance, points[row]), atol=1e-6)


def test_project_is_the_pulled_back_simplex_projection():
    """Feasible points are fixed, infeasible ones land on the simplex (nonnegative, sums to one), and
    projecting twice is projecting once."""
    family, instance = bimatrix(MATCHING_PENNIES, -MATCHING_PENNIES)
    interior = family.sample_domain(instance, 4)
    assert torch.allclose(family.project(instance, interior), interior, atol=1e-6)

    far = torch.tensor([[5.0, -7.0], [-3.0, 2.0]])
    projected = family.project(instance, far)
    strategies = family._strategies(projected)
    assert (strategies >= -1e-6).all()
    assert torch.allclose(strategies.sum(dim=-1), torch.ones(2, 2))
    assert torch.allclose(family.project(instance, projected), projected, atol=1e-6)


def test_sample_domain_draws_feasible_strategies():
    """The domain sampler covers the constrained domain itself: every draw is a valid mixed-strategy
    profile, so rollout starts and uniform operator points never leave the feasible set."""
    torch.manual_seed(3)
    family, instance = bimatrix(MATCHING_PENNIES, -MATCHING_PENNIES)
    strategies = family._strategies(family.sample_domain(instance, 16))
    assert (strategies >= 0).all()
    assert torch.allclose(strategies.sum(dim=-1), torch.ones(16, 2))


def test_solve_reaches_the_interior_nash():
    """Matching Pennies: the extragradient rollout's endpoint is the chart origin."""
    torch.manual_seed(4)
    family, instance = bimatrix(MATCHING_PENNIES, -MATCHING_PENNIES)
    equilibrium = family.solve_instance(instance)
    assert equilibrium.norm() < 1e-3


def test_solve_reaches_a_boundary_nash_where_only_the_natural_map_vanishes():
    """The constrained case ``project`` exists for: the dominance-solvable game's Nash is a pure profile,
    so the raw field never vanishes there and convergence is legible only through the natural map."""
    torch.manual_seed(5)
    family, instance = bimatrix(DOMINANCE_SOLVABLE, -DOMINANCE_SOLVABLE)
    equilibrium = family.solve_instance(instance)

    assert natural_map(family, instance, equilibrium).norm() < 1e-3
    assert family.operator(instance, equilibrium).norm() > 0.5

    pure_profile = torch.eye(2)[0].expand(2, 2)  # both players play action 1
    assert torch.allclose(family._strategies(equilibrium), pure_profile, atol=1e-3)
