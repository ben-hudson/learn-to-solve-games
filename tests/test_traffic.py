import pytest
import tntp
import torch
from pathlib import Path

from l2s_games.envs.traffic import PUMENetwork, PotentialCongestion
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


def test_solve(sioux_falls_pyg_data, sioux_falls_pume_network):
    tensors = sioux_falls_pyg_data.multi_get_tensor(["free_flow_time", "capacity", "b", "power"])
    game = PotentialCongestion(sioux_falls_pume_network, *tensors)
    eq, info = game.solve()
    pass
