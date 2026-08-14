import torch
import tqdm
import pickle
from torch_geometric.data import InMemoryDataset

from .game import PotentialCongestion


class TrafficEquilibriumDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        pume_mapping=None,
        base_graph=None,
        n_instances=None,
        quiet=True,
        **kwargs,
    ):
        self.pume_mapping = pume_mapping
        self.base_graph = base_graph
        self.quiet = quiet
        self.n_instances = n_instances

        super().__init__(root, **kwargs)

    @property
    def raw_file_names(self):
        return ["instances.pt"]

    @property
    def processed_file_names(self):
        return ["instances.pt"]

    def download(self):
        required_attrs = ["pume_mapping", "base_graph", "n_instances"]
        assert all(
            getattr(self, attr) is not None for attr in required_attrs
        ), f"No cache at {self.raw_paths[0]}. Pass {required_attrs} to rebuild it."

        instances = self.generate_instances(self.pume_mapping, self.base_graph, self.n_instances)
        torch.save(instances, self.raw_paths[0])

    def generate_instances(self, pume_mapping, base_graph, n_instances: int):
        progress = range(n_instances) if self.quiet else tqdm.trange(n_instances)

        data_list = []
        # we warm start with the prev equilibrium, but the first one gets fft
        # empirically, this is a bit faster than starting from fft every time
        warm_start = base_graph.free_flow_time * 1.1
        for _ in progress:
            free_flow_time = base_graph.free_flow_time * (0.9 + 0.2 * torch.rand_like(base_graph.free_flow_time))
            capacity = base_graph.capacity * (0.9 + 0.2 * torch.rand_like(base_graph.capacity))
            instance = PotentialCongestion(pume_mapping, free_flow_time, capacity, base_graph.b, base_graph.power)

            instance.solve(initial_cost=instance.project_costs(warm_start), max_iters=2000, tol=1e-4)
            assert instance.eq_info["converged"]
            data_list.append(instance.to_data())

            warm_start = instance.eq

        return data_list

    def load_instances(self):
        return torch.load(self.raw_paths[0], weights_only=False)

    def process(self):
        # just copy the instances since they're already solved
        self.save(self.load_instances(), self.processed_paths[0])


class TrafficOperatorDataset(TrafficEquilibriumDataset):
    def __init__(self, root, n_cal_instances=None, n_points_per_instance=None, **kwargs):
        self.n_cal_instances = n_cal_instances
        self.n_points_per_instance = n_points_per_instance
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return ["operators.pt"]

    def process(self):
        required_attrs = ["pume_mapping", "base_graph", "n_cal_instances", "n_points_per_instance"]
        assert all(
            getattr(self, attr) is not None for attr in required_attrs
        ), f"No cache at {self.processed_paths[0]}. Pass {required_attrs} to rebuild it."

        # compute K_U, the subset of the feasible space to sample from
        cal_set = self.generate_instances(self.pume_mapping, self.base_graph, self.n_cal_instances)
        lower, _ = torch.stack([inst.free_flow_time for inst in cal_set], dim=-1).min(dim=-1)
        upper, _ = torch.stack([inst.eq for inst in cal_set], dim=-1).max(dim=-1)
        roi = torch.distributions.Uniform(lower, upper)

        instances = self.load_instances()
        progress = instances if self.quiet else tqdm.tqdm(instances)

        data_list = []
        for data in progress:
            instance = PotentialCongestion.from_data(self.pume_mapping, data)
            data.point = roi.sample((self.n_points_per_instance,))
            data.operator = torch.stack([instance.compute_excess_supply(point) for point in data.point])
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
