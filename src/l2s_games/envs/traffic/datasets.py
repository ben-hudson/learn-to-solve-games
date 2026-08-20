import torch
import tqdm

from tensordict import TensorDict
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import BaseTransform

from .game import PotentialCongestion


class GraphToTensorDict(BaseTransform):
    """Convert a ``Data`` instance to a ``TensorDict``.

    ``torch.stack`` collates the per-instance dicts into a batched ``TensorDict``
    (pass it as the loader's ``collate_fn``), and ``batch[i]``/``batch.unbind(0)``
    slice it back into per-instance views natively.
    """

    def forward(self, data):
        return TensorDict(data.to_dict())


class BuildTrafficFeats(BaseTransform):
    def __init__(self, mode=None):
        super().__init__()

        assert mode in ["full", "partial"], f"Mode must be 'full' or 'partial', got {mode}."
        self.mode = mode

    def forward(self, data):
        feats = torch.stack([data.free_flow_time, data.capacity], dim=-1)

        if self.mode == "partial":
            n_points_per_instance = data.point.size(0)
            feats = feats.expand(n_points_per_instance, -1, -1)
            # here we add the point because it is not constrained to the simplex
            points = data.point.unsqueeze(-1)
            data.feats = torch.cat([feats, points], dim=-1)
        else:
            data.feats = feats

        return data


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
        self.load(self.processed_paths[0])

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

    # TODO: use TrafficEquilibriumStream for this
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
            op, precond = zip(*(instance.operator_and_preconditioner(point) for point in data.point))
            # the operator evaluations run in float64, but the learning stack expects float32
            data.operator = torch.stack(op).float()
            data.preconditioner = torch.stack(precond).float()
            data.preconditioned_operator = data.operator / data.preconditioner
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
