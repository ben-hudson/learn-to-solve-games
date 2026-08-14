import argparse

from pathlib import Path

import tntp
import torch
from torch_geometric.utils import from_networkx
from l2s_games.envs.traffic import TrafficEquilibriumDataset, PUMEMapping
from l2s_games.envs.traffic.datasets import TrafficOperatorDataset


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--n_instances", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    # config = get_config()

    root = Path("raw_data/sioux_falls")

    network = tntp.convert_to_networkx(
        tntp.read_node_file(root / "SiouxFalls_node.tntp", index_col="Node", x_col="X", y_col="Y", crs="wgs84"),
        tntp.read_net_file(root / "SiouxFalls_net.tntp", crs="wgs84"),
    )
    demand_table = tntp.read_demand_file(root / "SiouxFalls_trips.tntp")

    base_graph = from_networkx(network)
    base_graph.free_flow_time = base_graph.free_flow_time.float()

    node_list = list(network.nodes)
    demand_table = demand_table.reindex(index=node_list, columns=node_list)

    pume_mapping = PUMEMapping.from_edges_and_demand(base_graph.edge_index, torch.as_tensor(demand_table.values))
    eq_dataset = TrafficEquilibriumDataset(
        "datasets/sioux_falls_16_new",
        pume_mapping=pume_mapping,
        base_graph=base_graph,
        n_instances=16,
        quiet=False,
    )
    op_dataset = TrafficOperatorDataset(
        "datasets/sioux_falls_16_new",
        pume_mapping=pume_mapping,
        base_graph=base_graph,
        n_cal_instances=16,
        n_points_per_instance=256,
        force_reload=True,
        quiet=False,
    )
