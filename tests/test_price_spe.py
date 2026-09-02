import pytest
import torch

from l2s_games.algorithms import SimpleProjection
from l2s_games.envs.spe import dist_to_normal_cone, FlowSPE, PriceSPE
from l2s_games.envs.spe.streams import solve_instance


@pytest.fixture
def random_price_spe():
    return PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=6.0, eps=0.1, delta=0.05, scale=0.4)


@pytest.fixture
def random_price_spe_list():
    return [
        PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=6.0, eps=0.1, delta=0.05, scale=0.4)
        for _ in range(5)
    ]


@pytest.fixture
def random_flow_spe():
    # the same rotation-dominant instances test_spe.py uses, so the two formulations are compared on
    # the hard case; nothing here iterates in price space, where these are badly conditioned
    return FlowSPE.random_bipartite(n_supply=3, n_demand=4, kappa=6.0, eps=0.1, delta=0.05, scale=0.4)


@pytest.fixture
def mild_flow_spe():
    # weak rotation, so plain projection converges within a test's budget. Nothing stronger is
    # called for: price space has no rotational obstacle to overcome (see
    # ``test_price_space_damps_rotation``), and its real difficulty is conditioning -- the
    # monotonicity modulus runs ~1e-3 against a Lipschitz constant of a few units on the kappa=6
    # fixture, which no first-order method fixes and which a semismooth Newton method would, the
    # piecewise-linear operator being ideally suited to one. ``algorithms`` has no such method, so
    # the correctness evidence comes from the cross-formulation tests instead, which run on the hard
    # fixture and iterate only in flow space.
    return FlowSPE.random_bipartite(n_supply=3, n_demand=4, kappa=0.5, eps=0.1, delta=0.05, scale=1.0)


def intercept_prices(spe: PriceSPE):
    """The prices at which every market's own quantity is zero -- a point off equilibrium whose
    surplus ``q_j - p_i - c_ij`` straddles zero, so both open and closed routes are exercised."""
    return torch.cat([spe.p, spe.q])


def operator_jacobian(spe: PriceSPE, prices: torch.Tensor):
    return torch.autograd.functional.jacobian(spe.operator, prices)


def test_operator_jacobian_matches_closed_form(random_price_spe: PriceSPE):
    # [[Gamma + diag(A1/delta), -A/delta], [-A^T/delta, Theta + diag(A^T1/delta)]] -- a bipartite
    # Laplacian weighted by 1/delta over the active routes, plus the block diagonal
    prices = intercept_prices(random_price_spe)
    supply_price, demand_price = prices.split([random_price_spe.n_supply, random_price_spe.n_demand])
    active = demand_price.unsqueeze(-2) - supply_price.unsqueeze(-1) - random_price_spe.c > 0
    weights = active / random_price_spe.delta

    expected = torch.cat(
        [
            torch.cat([random_price_spe.Gamma + torch.diag(weights.sum(dim=-1)), -weights], dim=-1),
            torch.cat([-weights.T, random_price_spe.Theta + torch.diag(weights.sum(dim=-2))], dim=-1),
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

    route_cost = supply_price.unsqueeze(-1) + random_price_spe.c + random_price_spe.delta * shipments
    balance = route_cost - demand_price.unsqueeze(-2)
    assert torch.allclose(balance[shipments > 0], torch.zeros_like(balance[shipments > 0]), atol=1e-5)
    assert (balance[shipments == 0] >= 0.0).all()


def test_instance_round_trip_recovers_matrices(random_flow_spe: FlowSPE):
    restored = random_flow_spe.to_price_spe().to_flow_spe()

    assert torch.allclose(restored.P, random_flow_spe.P, atol=1e-4)
    assert torch.allclose(restored.Q, random_flow_spe.Q, atol=1e-4)
    assert torch.equal(restored.p, random_flow_spe.p)
    assert torch.equal(restored.q, random_flow_spe.q)
    assert torch.equal(restored.delta, random_flow_spe.delta)
    assert torch.equal(restored.c, random_flow_spe.c)


def test_conversion_preserves_monotonicity(random_flow_spe: FlowSPE):
    # inverting a matrix with PD symmetric part gives another one, so a converted instance is always
    # strongly monotone -- no parameter range makes the price formulation ill-posed
    price_spe = random_flow_spe.to_price_spe()
    for matrix in (price_spe.Gamma, price_spe.Theta):
        assert (torch.linalg.eigvalsh((matrix + matrix.T) / 2) > 0).all()


def rotationality(J):
    """``||skew(J)|| / ||sym(J)||``: above one the field is rotation-dominated, below one it is
    close to the gradient of a potential."""
    sym, skew = (J + J.T) / 2, (J - J.T) / 2
    return torch.linalg.matrix_norm(skew, ord=2) / torch.linalg.matrix_norm(sym, ord=2)


def flow_rotationality(flow_spe: FlowSPE):
    n_routes = flow_spe.n_supply * flow_spe.n_demand
    jacobian = torch.autograd.functional.jacobian(
        lambda x: flow_spe.operator(x.view(flow_spe.n_supply, -1)).view(-1), torch.rand(n_routes)
    )
    return rotationality(jacobian)


def test_conversion_damps_rotation(random_flow_spe: FlowSPE):
    """Converting a rotation-dominant flow instance yields a near-potential price instance.

    ``sym`` gains the ``1/delta``-weighted bipartite Laplacian in price space while the skew part is
    only ``skew(Gamma) (+) skew(Theta)``; in flow space the symmetric part is ``delta I`` plus two
    rank-deficient PSD terms against a full-strength skew part. Raising ``kappa`` does not reverse
    it either, because conversion inverts: more skew means a larger ``||P||``, hence a smaller
    ``Gamma = P^-1`` and a smaller ``skew(Gamma)`` against an unchanged Laplacian.

    So a converted instance is the wrong non-potential benchmark, and
    ``test_native_sampling_keeps_rotation`` is how to get the right one.
    """
    price_spe = random_flow_spe.to_price_spe()
    price_rotation = rotationality(operator_jacobian(price_spe, intercept_prices(price_spe)))

    assert price_rotation < 1.0 < flow_rotationality(random_flow_spe)


def test_native_sampling_keeps_rotation():
    """Sampling ``Gamma`` directly keeps the rotation in price space -- and exports it to flow space.

    Rotation lives in whichever space the skew is sampled in, because the two are related by matrix
    inversion and a skew-dominated matrix has a small inverse. Sampled natively, rotation dominates
    once ``||skew(Gamma)||`` exceeds the ``(m + n)/delta`` Laplacian, which is why ``delta`` is large
    here: costly shipping decouples the markets, leaving each one's asymmetric response to the
    others' prices to dominate, where frictionless arbitrage would average it away.
    """
    price_spe = PriceSPE.random_bipartite(n_supply=3, n_demand=4, kappa=5.0, eps=0.1, delta=20.0, scale=0.4)
    price_rotation = rotationality(operator_jacobian(price_spe, intercept_prices(price_spe)))

    assert price_rotation > 1.0 > flow_rotationality(price_spe.to_flow_spe())


def test_flow_equilibrium_clears_every_market(random_flow_spe: FlowSPE):
    # the equivalence of the two formulations: a flow-space equilibrium, read as prices, is a zero
    # of the price-space operator
    shipments = solve_instance(random_flow_spe)
    excess_supply = random_flow_spe.to_price_spe().operator(random_flow_spe.prices(shipments))

    assert torch.allclose(excess_supply, torch.zeros_like(excess_supply), atol=1e-3)


def test_flow_equilibrium_recovered_from_its_prices(random_flow_spe: FlowSPE):
    # the other direction of the same equivalence: prices and flows invert each other at a point
    # satisfying the route conditions
    shipments = solve_instance(random_flow_spe)
    price_spe = random_flow_spe.to_price_spe()
    recovered = price_spe.flows(random_flow_spe.prices(shipments))

    assert torch.allclose(recovered, shipments, atol=5e-3)


def test_price_flow_round_trip_is_not_an_identity_off_equilibrium(random_flow_spe: FlowSPE):
    # the round trip characterizes equilibrium, so it must move a non-equilibrium point
    shipments = torch.rand(random_flow_spe.n_supply, random_flow_spe.n_demand)
    price_spe = random_flow_spe.to_price_spe()

    assert not torch.allclose(price_spe.flows(random_flow_spe.prices(shipments)), shipments, atol=1e-3)


def test_projection_converges_in_price_space(mild_flow_spe: FlowSPE):
    # prices are free, so the projection is the identity and this is plain forward iteration on the
    # field. It suffices on a *converted* instance however large kappa is, because conversion damps
    # the rotation away: tuned its own step size, extragradient has no edge on one and sometimes
    # loses outright (9k steps against 28k). Natively sampled instances are the opposite -- at
    # kappa=5 with delta=20 projection fails on most and extragradient solves them all
    price_spe = mild_flow_spe.to_price_spe()
    algorithm = SimpleProjection(5e-3, lambda prices: -price_spe.operator(prices), lambda prices: prices)

    prices = torch.zeros(price_spe.n_supply + price_spe.n_demand)
    for _ in range(20000):
        prices = algorithm.step(prices)

    shipments = price_spe.flows(prices)
    dist = dist_to_normal_cone(-mild_flow_spe.operator(shipments), shipments)
    assert torch.allclose(dist, torch.tensor(0.0), atol=1e-3)


def test_dict_round_trip(random_price_spe: PriceSPE):
    restored = PriceSPE.from_dict(random_price_spe.to_dict())

    assert torch.equal(restored.Gamma, random_price_spe.Gamma)
    assert torch.equal(restored.Theta, random_price_spe.Theta)
    assert torch.equal(restored.p, random_price_spe.p)
    assert torch.equal(restored.q, random_price_spe.q)
    assert torch.equal(restored.delta, random_price_spe.delta)
    assert torch.equal(restored.c, random_price_spe.c)


def test_batched_operator_matches_per_instance(random_price_spe_list):
    batched = PriceSPE.from_dict(torch.stack([spe.to_dict() for spe in random_price_spe_list]))

    prices = torch.rand(len(random_price_spe_list), batched.n_supply + batched.n_demand)
    expected = torch.stack([spe.operator(point) for spe, point in zip(random_price_spe_list, prices)])
    operators = batched.operator(prices)

    assert operators.shape == prices.shape
    assert torch.allclose(operators, expected)
