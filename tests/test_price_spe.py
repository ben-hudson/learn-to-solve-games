import pytest
import torch

from l2s_games.algorithms import ExtraGradient, SimpleProjection
from l2s_games.envs.spe import PriceSPE


@pytest.fixture
def random_price_spe():
    return PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=5.0, eps=0.1, delta=20.0, scale=0.4)


@pytest.fixture
def random_price_spe_list():
    return [
        PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=5.0, eps=0.1, delta=20.0, scale=0.4) for _ in range(5)
    ]


def intercept_prices(spe: PriceSPE):
    return torch.cat([spe.supply_side_min, spe.demand_side_min])


def operator_jacobian(spe: PriceSPE, prices: torch.Tensor):
    return torch.autograd.functional.jacobian(spe.operator, prices)


def test_operator_jacobian_matches_closed_form(random_price_spe: PriceSPE):
    # [[Gamma + diag(A1/delta), -A/delta], [-A^T/delta, Theta + diag(A^T1/delta)]] -- a bipartite
    # Laplacian weighted by 1/delta over the active routes, plus the block diagonal
    prices = intercept_prices(random_price_spe)
    supply_price, demand_price = prices.split([random_price_spe.n_supply, random_price_spe.n_demand])
    active = demand_price.unsqueeze(-2) - supply_price.unsqueeze(-1) - random_price_spe.route_cost_min > 0
    weights = active / random_price_spe.route_cost_slope

    expected = torch.cat(
        [
            torch.cat([random_price_spe.supply_side_effects + torch.diag(weights.sum(dim=-1)), -weights], dim=-1),
            torch.cat([-weights.T, random_price_spe.demand_side_effects + torch.diag(weights.sum(dim=-2))], dim=-1),
        ],
        dim=-2,
    )

    assert torch.allclose(operator_jacobian(random_price_spe, prices), expected, atol=1e-5)


def test_operator_is_monotone(random_price_spe: PriceSPE):
    J = operator_jacobian(random_price_spe, intercept_prices(random_price_spe))
    symmetric_part_eigenvalues = torch.linalg.eigvalsh((J + J.T) / 2)
    assert (symmetric_part_eigenvalues > 0).all()


def test_zero_kappa_gives_potential_instance():
    spe = PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=0.0, eps=0.1, delta=0.05)
    J = operator_jacobian(spe, intercept_prices(spe))
    assert torch.allclose(J, J.T, atol=1e-5)


def test_positive_kappa_gives_non_potential_instance(random_price_spe: PriceSPE):
    J = operator_jacobian(random_price_spe, intercept_prices(random_price_spe))
    assert not torch.allclose(J, J.T, atol=1e-5)


def test_flows_are_nonnegative_at_negative_prices(random_price_spe: PriceSPE):
    # the clamp makes feasibility structural, which is what licenses iterating on free prices
    prices = -intercept_prices(random_price_spe)
    assert (random_price_spe.flows(prices) >= 0.0).all()


def test_flows_satisfy_route_conditions(random_price_spe: PriceSPE):
    # the closed form solves the route complementarity: open routes have balanced prices, closed
    # ones are unprofitable
    prices = intercept_prices(random_price_spe)
    supply_price, demand_price = prices.split([random_price_spe.n_supply, random_price_spe.n_demand])
    shipments = random_price_spe.flows(prices)

    route_cost = (
        supply_price.unsqueeze(-1) + random_price_spe.route_cost_min + random_price_spe.route_cost_slope * shipments
    )
    balance = route_cost - demand_price.unsqueeze(-2)
    assert torch.allclose(balance[shipments > 0], torch.zeros_like(balance[shipments > 0]), atol=1e-5)
    assert (balance[shipments == 0] >= 0.0).all()


# def test_instance_round_trip_recovers_matrices(random_flow_spe: FlowSPE):
#     restored = random_flow_spe.to_price_spe().to_flow_spe()

#     assert torch.allclose(restored.P, random_flow_spe.P, atol=1e-4)
#     assert torch.allclose(restored.Q, random_flow_spe.Q, atol=1e-4)
#     assert torch.equal(restored.p, random_flow_spe.p)
#     assert torch.equal(restored.q, random_flow_spe.q)
#     assert torch.equal(restored.delta, random_flow_spe.delta)
#     assert torch.equal(restored.c, random_flow_spe.c)


# def test_conversion_preserves_monotonicity(random_flow_spe: FlowSPE):
#     # inverting a matrix with PD symmetric part gives another one, so a converted instance is always
#     # strongly monotone -- no parameter range makes the price formulation ill-posed
#     price_spe = random_flow_spe.to_price_spe()
#     for matrix in (price_spe.Gamma, price_spe.Theta):
#         assert (torch.linalg.eigvalsh((matrix + matrix.T) / 2) > 0).all()


# def rotationality(J):
#     """``||skew(J)|| / ||sym(J)||``: above one the field is rotation-dominated, below one it is
#     close to the gradient of a potential."""
#     sym, skew = (J + J.T) / 2, (J - J.T) / 2
#     return torch.linalg.matrix_norm(skew, ord=2) / torch.linalg.matrix_norm(sym, ord=2)


# def flow_rotationality(flow_spe: FlowSPE):
#     n_routes = flow_spe.n_supply * flow_spe.n_demand
#     jacobian = torch.autograd.functional.jacobian(
#         lambda x: flow_spe.operator(x.view(flow_spe.n_supply, -1)).view(-1), torch.rand(n_routes)
#     )
#     return rotationality(jacobian)


# def test_conversion_damps_rotation(random_flow_spe: FlowSPE):
#     """Converting a rotation-dominant flow instance yields a near-potential price instance.

#     ``sym`` gains the ``1/delta``-weighted bipartite Laplacian in price space while the skew part is
#     only ``skew(Gamma) (+) skew(Theta)``; in flow space the symmetric part is ``delta I`` plus two
#     rank-deficient PSD terms against a full-strength skew part. Raising ``kappa`` does not reverse
#     it either, because conversion inverts: more skew means a larger ``||P||``, hence a smaller
#     ``Gamma = P^-1`` and a smaller ``skew(Gamma)`` against an unchanged Laplacian.

#     So a converted instance is the wrong non-potential benchmark, and
#     ``test_native_sampling_keeps_rotation`` is how to get the right one.
#     """
#     price_spe = random_flow_spe.to_price_spe()
#     price_rotation = rotationality(operator_jacobian(price_spe, intercept_prices(price_spe)))

#     assert price_rotation < 1.0 < flow_rotationality(random_flow_spe)


# def test_native_sampling_keeps_rotation():
#     """Sampling ``Gamma`` directly keeps the rotation in price space -- and exports it to flow space.

#     Rotation lives in whichever space the skew is sampled in, because the two are related by matrix
#     inversion and a skew-dominated matrix has a small inverse. Sampled natively, rotation dominates
#     once ``||skew(Gamma)||`` exceeds the ``(m + n)/delta`` Laplacian, which is why ``delta`` is large
#     here: costly shipping decouples the markets, leaving each one's asymmetric response to the
#     others' prices to dominate, where frictionless arbitrage would average it away.
#     """
#     price_spe = PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=5.0, eps=0.1, delta=20.0, scale=0.4)
#     price_rotation = rotationality(operator_jacobian(price_spe, intercept_prices(price_spe)))

#     assert price_rotation > 1.0 > flow_rotationality(price_spe.to_flow_spe())


# def test_flow_equilibrium_clears_every_market(random_flow_spe: FlowSPE):
#     # the equivalence of the two formulations: a flow-space equilibrium, read as prices, is a zero
#     # of the price-space operator
#     shipments = solve_instance(random_flow_spe)
#     excess_supply = random_flow_spe.to_price_spe().operator(random_flow_spe.prices(shipments))

#     assert torch.allclose(excess_supply, torch.zeros_like(excess_supply), atol=1e-3)


# def test_equilibrium_price_flow_equilibrium_round_trip(random_flow_spe: FlowSPE):
#     # the other direction of the same equivalence: prices and flows invert each other at a point
#     # satisfying the route conditions
#     shipments = solve_instance(random_flow_spe)
#     price_spe = random_flow_spe.to_price_spe()
#     recovered = price_spe.flows(random_flow_spe.prices(shipments))

#     assert torch.allclose(recovered, shipments, atol=5e-3)


# def test_off_equilibrium_price_flow_round_trip(random_flow_spe: FlowSPE):
#     # the round trip characterizes equilibrium, so it must move a non-equilibrium point
#     shipments = torch.rand(random_flow_spe.n_supply, random_flow_spe.n_demand)
#     price_spe = random_flow_spe.to_price_spe()

#     assert not torch.allclose(price_spe.flows(random_flow_spe.prices(shipments)), shipments, atol=1e-3)


def solve_prices(price_spe: PriceSPE, algorithm_cls, step_size, n_steps):
    """Iterate ``algorithm_cls`` on the price-space field. Prices are free, so the projection is the
    identity and ``SimpleProjection`` is plain forward iteration."""
    algorithm = algorithm_cls(step_size, lambda prices: -price_spe.operator(prices), lambda prices: prices)

    prices = torch.zeros(price_spe.n_supply + price_spe.n_demand)
    for _ in range(n_steps):
        prices = algorithm.step(prices)
    return prices


def test_extragradient_converges(random_price_spe: PriceSPE):
    # extragradient's own best pair: a step just over 1/L -- the bound is conservative and nothing
    # destabilizes -- converging in 200 to 400 steps, so 500 is the budget.
    # The prices are free, so the normal cone at every point is {0} and the distance to it is the
    # residual itself -- no clamping to do, unlike the flow-space reading, where the shipments are
    # pinned to the orthant. The two are the same certificate: ``flows`` satisfies the route
    # conditions at any prices, so all that is left to violate is market clearing. Same tolerance
    # test_projection_does_not_converge fails at, which is what makes the two a comparison.
    prices = solve_prices(random_price_spe, ExtraGradient, step_size=1e-1, n_steps=500)
    residual = random_price_spe.operator(prices)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)


def test_projection_does_not_converge(random_price_spe_list):
    """Plain forward iteration does not finish the rotational instances in ten times the budget.

    It does not *diverge*: at this step size, its own best, it converges on all of them eventually,
    the admissible step being ``O(mu/L^2)`` against extragradient's ``O(1/L)``. It just needs 9x to
    82x the iterations -- 2.8k to 20k against 200 to 400 -- so most instances are still short of
    tolerance at 5000. A converted instance shows none of this, the inversion having damped the
    rotation away.
    """
    unconverged = 0
    for price_spe in random_price_spe_list:
        prices = solve_prices(price_spe, SimpleProjection, step_size=1e-2, n_steps=5000)
        residual = price_spe.operator(prices)
        unconverged += not torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)

    assert unconverged >= 1


def test_data_round_trip(random_price_spe: PriceSPE):
    restored = PriceSPE.from_data(random_price_spe.to_data())

    assert torch.equal(restored.supply_side_effects, random_price_spe.supply_side_effects)
    assert torch.equal(restored.demand_side_effects, random_price_spe.demand_side_effects)
    assert torch.equal(restored.supply_side_min, random_price_spe.supply_side_min)
    assert torch.equal(restored.demand_side_min, random_price_spe.demand_side_min)
    assert torch.equal(restored.route_cost_slope, random_price_spe.route_cost_slope)
    assert torch.equal(restored.route_cost_min, random_price_spe.route_cost_min)


def test_batched_operator_matches_per_instance(random_price_spe_list):
    # stacking the instances is what makes the leading dim a batch dim, so this is the same
    # operator a collated batch evaluates
    batched = torch.stack(random_price_spe_list)

    prices = torch.rand(len(random_price_spe_list), batched.n_supply + batched.n_demand)
    expected = torch.stack([spe.operator(point) for spe, point in zip(random_price_spe_list, prices)])
    operators = batched.operator(prices)

    assert operators.shape == prices.shape
    assert torch.allclose(operators, expected)
