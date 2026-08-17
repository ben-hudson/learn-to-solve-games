import pytest
import tntp
import torch

from l2s_games.algorithms import SimpleProjection
from l2s_games.envs.traffic import PUMEMapping, PotentialCongestion, dist_to_normal_cone
from pathlib import Path
from torch_geometric.utils import from_networkx


@pytest.fixture
def sioux_falls():
    root = Path("raw_data/sioux_falls")

    network = tntp.convert_to_networkx(
        tntp.read_node_file(root / "SiouxFalls_node.tntp", index_col="Node", x_col="X", y_col="Y", crs="wgs84"),
        tntp.read_net_file(root / "SiouxFalls_net.tntp", crs="wgs84"),
    )
    demand_table = tntp.read_demand_file(root / "SiouxFalls_trips.tntp")

    node_list = list(network.nodes)
    demand_table = demand_table.reindex(index=node_list, columns=node_list)

    return network, demand_table


@pytest.fixture
def perturbed_sioux_falls_pyg_data(sioux_falls):
    network, demand_table = sioux_falls
    data = from_networkx(network)
    free_flow_time = data.free_flow_time.float()
    data.free_flow_time = free_flow_time * (0.9 + 0.2 * torch.rand_like(free_flow_time))
    data.capacity = data.capacity * (0.9 + 0.2 * torch.rand_like(data.capacity))

    # vehicles -> kilovehicles; scaling demand and capacity together keeps the equilibrium
    # identical while shrinking the excess-supply operator 1000x
    data.demand = torch.as_tensor(demand_table.values) / 1000
    data.capacity /= 1000
    return data


@pytest.fixture
def perturbed_sioux_falls_pume_network(perturbed_sioux_falls_pyg_data):
    return PUMEMapping.from_edges_and_demand(
        perturbed_sioux_falls_pyg_data.edge_index,
        perturbed_sioux_falls_pyg_data.demand,
    )


@pytest.fixture
def sioux_falls_congestion_game(perturbed_sioux_falls_pyg_data, perturbed_sioux_falls_pume_network):
    tensors = perturbed_sioux_falls_pyg_data.multi_get_tensor(["free_flow_time", "capacity", "b", "power"])
    game = PotentialCongestion(perturbed_sioux_falls_pume_network, *tensors)
    return game


def test_normal_cone_dist_zero(sioux_falls_congestion_game):
    eq, info = sioux_falls_congestion_game.solve(max_iters=2000, tol=1e-4)
    assert info["converged"]

    lower, upper = (torch.as_tensor(bound) for bound in sioux_falls_congestion_game.cost_bounds)
    # dist_to_normal_cone's cone belongs to the ascent dynamics z <- clamp(z + h op(z));
    # PUME's fixed point is c <- clamp(c - lambda E(c)), so the ascent operator is the
    # excess demand -E = x - z (excess demand pushes costs up).
    excess_demand = -info["excess_supply"]
    dist = dist_to_normal_cone(excess_demand, eq, lower, upper)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-3)


def test_normal_cone_dist_nonzero(sioux_falls_congestion_game):
    free_flow_time = sioux_falls_congestion_game.free_flow_time
    costs = free_flow_time * (1 + 0.5 * torch.rand_like(free_flow_time))
    costs = sioux_falls_congestion_game.project_costs(costs)

    lower, upper = (torch.as_tensor(bound) for bound in sioux_falls_congestion_game.cost_bounds)
    excess_demand = -sioux_falls_congestion_game.compute_excess_supply(costs)
    dist = dist_to_normal_cone(excess_demand, costs, lower, upper)
    assert not torch.allclose(dist, torch.zeros_like(dist), atol=1e-3)


def test_projection_converges(sioux_falls_congestion_game: PotentialCongestion):
    # the ascent operator for z <- project(z + h op(z)) is the excess demand -E (see
    # test_normal_cone_dist_zero), preconditioned with the game's supply-diagonal metric to tame
    # the steep coordinates of the inverse BPR supply curve
    def preconditioned_excess_demand(costs):
        excess_supply, precond = sioux_falls_congestion_game.operator_and_preconditioner(costs)
        return -excess_supply / precond

    # step size tuned over 20 perturbed instances: the dynamics stop converging from h=0.26 up
    # (2/20 instances diverge there, most from h=0.27), and h=0.25 converged on all instances in
    # 135-159 steps, so 200 leaves headroom for the random perturbation
    algorithm = SimpleProjection(0.25, preconditioned_excess_demand, sioux_falls_congestion_game.project_costs)
    costs = sioux_falls_congestion_game.free_flow_time * 1.1
    for _ in range(200):
        costs = algorithm.step(costs)

    lower, upper = (torch.as_tensor(bound) for bound in sioux_falls_congestion_game.cost_bounds)
    excess_demand = -sioux_falls_congestion_game.operator(costs)
    dist = dist_to_normal_cone(excess_demand, costs, lower, upper)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-3)
