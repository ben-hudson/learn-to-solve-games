import pytest
import torch

from torch.utils.data import DataLoader
from l2s_games.algorithms import ExtraGradient, SimpleProjection
from l2s_games.envs.spe import dist_to_normal_cone, SpatialPriceEquilibrium
from l2s_games.envs.spe.streams import EquilibriumStream


@pytest.fixture
def random_spe():
    # weak symmetric coupling (scale) with strong rotation (kappa): plain projection fails
    # on more than half of these instances while extragradient still converges
    return SpatialPriceEquilibrium.random_bipartite(n_supply=3, n_demand=4, kappa=6.0, eps=0.1, delta=0.05, scale=0.4)


@pytest.fixture
def random_spe_list():
    return [
        SpatialPriceEquilibrium.random_bipartite(n_supply=3, n_demand=4, kappa=6.0, eps=0.1, delta=0.05, scale=0.4)
        for _ in range(5)
    ]


def operator_jacobian(spe: SpatialPriceEquilibrium):
    # the operator is affine, so its Jacobian at any point is M
    def flat_operator(x):
        return spe.operator(x.view(spe.n_supply, spe.n_demand)).view(-1)

    return torch.autograd.functional.jacobian(flat_operator, torch.rand(spe.n_supply * spe.n_demand))


def test_operator_matches_closed_form(random_spe: SpatialPriceEquilibrium):
    # F(x) = Mx + b with M = R^T P R + C^T Q C + diag(δ) and b = R^T p + c − C^T q,
    # where R and C are the row-sum and column-sum matrices on flattened x
    row_sum = torch.eye(random_spe.n_supply).repeat_interleave(random_spe.n_demand, dim=1)
    col_sum = torch.eye(random_spe.n_demand).repeat(1, random_spe.n_supply)
    expected_M = (
        row_sum.T @ random_spe.P @ row_sum + col_sum.T @ random_spe.Q @ col_sum + torch.diag(random_spe.delta.view(-1))
    )
    expected_b = row_sum.T @ random_spe.p + random_spe.c.view(-1) - col_sum.T @ random_spe.q

    M = operator_jacobian(random_spe)
    b = random_spe.operator(torch.zeros(random_spe.n_supply, random_spe.n_demand)).view(-1)

    assert torch.allclose(M, expected_M, atol=1e-5)
    assert torch.allclose(b, expected_b, atol=1e-5)


def test_operator_is_monotone(random_spe: SpatialPriceEquilibrium):
    M = operator_jacobian(random_spe)
    symmetric_part_eigenvalues = torch.linalg.eigvalsh((M + M.T) / 2)
    assert (symmetric_part_eigenvalues > 0).all()


def test_zero_kappa_gives_potential_instance():
    spe = SpatialPriceEquilibrium.random_bipartite(n_supply=3, n_demand=4, kappa=0.0, eps=0.1, delta=0.05)
    M = operator_jacobian(spe)
    assert torch.allclose(M, M.T, atol=1e-5)


def test_positive_kappa_gives_non_potential_instance(random_spe: SpatialPriceEquilibrium):
    M = operator_jacobian(random_spe)
    assert not torch.allclose(M, M.T, atol=1e-5)


def test_extragradient_converges_to_normal_cone(random_spe: SpatialPriceEquilibrium):
    # the operator is descent-convention, so the algorithm ascends its negation; relu projects
    # onto the nonnegative orthant. Larger steps exceed 1/L on some instances, and the weak
    # monotonicity (small delta) of the rotation-dominant instances needs the longer run.
    algorithm = ExtraGradient(2e-2, lambda flow: -random_spe.operator(flow), torch.relu)

    shipments = torch.zeros(random_spe.n_supply, random_spe.n_demand)
    for _ in range(5000):
        shipments = algorithm.step(shipments)

    dist = dist_to_normal_cone(-random_spe.operator(shipments), shipments)
    assert torch.allclose(dist, torch.tensor(0.0), atol=1e-3)


def test_projection_fails_on_non_potential_instances(random_spe_list):
    # the rotation-dominant instances are what plain projection cannot handle: it fails on more
    # than half of them, while extragradient solves them all (see the convergence test)
    failures = 0
    for spe in random_spe_list:
        algorithm = SimpleProjection(5e-2, lambda flow: -spe.operator(flow), torch.relu)

        shipments = torch.zeros(spe.n_supply, spe.n_demand)
        for _ in range(2000):
            shipments = algorithm.step(shipments)

        failures += dist_to_normal_cone(-spe.operator(shipments), shipments) > 1e-2

    assert failures >= 1


def test_normal_cone_dist_nonzero_off_equilibrium(random_spe: SpatialPriceEquilibrium):
    shipments = torch.rand(random_spe.n_supply, random_spe.n_demand)
    dist = dist_to_normal_cone(-random_spe.operator(shipments), shipments)
    assert (dist > 0.0).all()


def test_dict_round_trip(random_spe: SpatialPriceEquilibrium):
    restored = SpatialPriceEquilibrium.from_dict(random_spe.to_dict())

    assert torch.equal(restored.P, random_spe.P)
    assert torch.equal(restored.Q, random_spe.Q)
    assert torch.equal(restored.p, random_spe.p)
    assert torch.equal(restored.q, random_spe.q)
    assert torch.equal(restored.delta, random_spe.delta)
    assert torch.equal(restored.c, random_spe.c)


def test_batched_operator_matches_per_instance(random_spe_list):
    batched = SpatialPriceEquilibrium.from_dict(torch.stack([spe.to_dict() for spe in random_spe_list]))

    points = torch.rand(len(random_spe_list), batched.n_supply, batched.n_demand)
    expected = torch.stack([spe.operator(point) for spe, point in zip(random_spe_list, points)])
    operators = batched.operator(points)

    assert operators.shape == points.shape
    assert torch.allclose(operators, expected)


def test_smoke_equilibrium_stream():
    n_supply, n_demand = 3, 4
    batch_size = 5
    dataset = EquilibriumStream(n_supply, n_demand, 6.0, 0.1, 0.05, scale=0.4, n_instances=2 * batch_size, solve=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=torch.stack)
    for batch in dataloader:
        assert batch["P"].shape == (batch_size, n_supply, n_supply)
        assert batch["Q"].shape == (batch_size, n_demand, n_demand)
        assert batch["p"].shape == (batch_size, n_supply)
        assert batch["q"].shape == (batch_size, n_demand)
        assert batch["c"].shape == (batch_size, n_supply, n_demand)
        assert batch["delta"].shape == (batch_size, n_supply, n_demand)
        assert batch["eq"].shape == (batch_size, n_supply, n_demand)
