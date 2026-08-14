import pytest
import tntp
import torch
from pathlib import Path

from l2s_games.envs.traffic import PUMENetwork, PotentialCongestion
from torch_geometric.utils import from_networkx

from l2s_games.envs.traffic.utils import dist_to_normal_cone


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
def sioux_falls_pyg_data(sioux_falls):
    network, demand_table = sioux_falls
    data = from_networkx(network)
    data.demand = torch.as_tensor(demand_table.values)
    return data


@pytest.fixture
def sioux_falls_pume_network(sioux_falls_pyg_data):
    return PUMENetwork.from_edges_and_demand(
        sioux_falls_pyg_data.edge_index,
        sioux_falls_pyg_data.demand,
    )


@pytest.fixture
def congestion_game(sioux_falls_pyg_data, sioux_falls_pume_network):
    tensors = sioux_falls_pyg_data.multi_get_tensor(["free_flow_time", "capacity", "b", "power"])
    game = PotentialCongestion(sioux_falls_pume_network, *tensors)
    return game


def test_normal_cone_dist_zero(congestion_game):
    eq, info = congestion_game.solve(max_iters=2000, tol=1e-4)
    assert info["converged"]

    lower, upper = (torch.as_tensor(bound) for bound in congestion_game.cost_bounds)
    # dist_to_normal_cone's cone belongs to the ascent dynamics z <- clamp(z + h op(z));
    # PUME's fixed point is c <- clamp(c - lambda E(c)), so the ascent operator is the
    # excess demand -E = x - z (excess demand pushes costs up).
    excess_demand = -info["excess_supply"]
    dist = dist_to_normal_cone(excess_demand, eq, lower, upper)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-3)


def test_normal_cone_dist_nonzero(congestion_game):
    free_flow_time = congestion_game.free_flow_time
    costs = free_flow_time * (1 + 0.5 * torch.rand_like(free_flow_time))
    costs = congestion_game.project_costs(costs)

    lower, upper = (torch.as_tensor(bound) for bound in congestion_game.cost_bounds)
    excess_demand = -congestion_game.compute_excess_supply(costs)
    dist = dist_to_normal_cone(excess_demand, costs, lower, upper)
    assert not torch.allclose(dist, torch.zeros_like(dist), atol=1e-3)
