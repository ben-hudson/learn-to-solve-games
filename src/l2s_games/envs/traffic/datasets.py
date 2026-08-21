import torch
import tqdm

from sklearn.preprocessing import MinMaxScaler
from tensordict import TensorDict
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import BaseTransform

from .game import NetworkLoading, NonPotentialCongestion
from .streams import TrafficEquilibriumStream, TrafficOperatorStream


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
        data.feats = torch.stack([data.free_flow_time, data.capacity], dim=-1)

        # if self.mode == "partial":
        #     n_points_per_instance = data.point.size(0)
        #     feats = feats.expand(n_points_per_instance, -1, -1)
        #     # here we add the point because it is not constrained to the simplex
        #     points = data.point.unsqueeze(-1)
        #     data.feats = torch.cat([feats, points], dim=-1)
        # else:
        #     data.feats = feats

        return data


class TrafficEquilibriumDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        network_loading: NetworkLoading = None,
        base_graph=None,
        n_instances=None,
        kappa=0.0,
        quiet=True,
        **kwargs,
    ):
        self.network_loading = network_loading
        self.base_graph = base_graph
        self.quiet = quiet
        self.n_instances = n_instances
        self.kappa = kappa

        super().__init__(root, **kwargs)
        self.base_graph = torch.load(self.raw_paths[1], weights_only=False)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["instances.pt", "base_graph.pt"]

    @property
    def processed_file_names(self):
        return ["instances.pt"]

    def download(self):
        required_attrs = ["network_loading", "base_graph", "n_instances"]
        assert all(
            getattr(self, attr) is not None for attr in required_attrs
        ), f"No cache at {self.raw_paths[0]}. Pass {required_attrs} to rebuild it."

        instances = list(
            TrafficEquilibriumStream(
                self.network_loading,
                self.base_graph,
                self.n_instances,
                quiet=self.quiet,
                solve=True,
            )
        )
        torch.save(instances, self.raw_paths[0])
        torch.save(self.base_graph, self.raw_paths[1])

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
        self.sample_lo, self.sample_hi = torch.load(self.processed_paths[1]).unbind(0)

    @property
    def processed_file_names(self):
        return ["operators.pt", "sample_box.pt"]

    def process(self):
        required_attrs = ["network_loading", "base_graph", "n_cal_instances", "n_points_per_instance"]
        assert all(
            getattr(self, attr) is not None for attr in required_attrs
        ), f"No cache at {self.processed_paths[0]}. Pass {required_attrs} to rebuild it."

        scaler = MinMaxScaler()
        cal_set = TrafficEquilibriumStream(
            self.network_loading,
            self.base_graph,
            self.n_cal_instances,
            quiet=self.quiet,
            solve=True,
        )
        for sample in cal_set:
            scaler.partial_fit(sample.free_flow_time.reshape(1, -1))
            scaler.partial_fit(sample.eq.reshape(1, -1))

        sample_lo = torch.as_tensor(scaler.data_min_)
        sample_hi = torch.as_tensor(scaler.data_max_)
        sample_box = torch.distributions.Uniform(sample_lo, sample_hi)

        instances = self.load_instances()
        progress = instances if self.quiet else tqdm.tqdm(instances)

        # TrafficOperatorStream generates new instances, but we want to attach operators to existing ones
        # TODO: there is a risk here that the generation process must match between TrafficOperatorStream and TrafficOperatorDataset
        operators = []
        for data in progress:
            instance = NonPotentialCongestion.from_data(self.network_loading, data)
            data.point = sample_box.sample((self.n_points_per_instance,)).float()
            op, precond = zip(*(instance.operator_and_preconditioner(point) for point in data.point))
            data.operator = torch.stack(op).float()
            data.preconditioner = torch.stack(precond).float()
            data.preconditioned_operator = data.operator / data.preconditioner
            operators.append(data)

        self.save(operators, self.processed_paths[0])
        torch.save(torch.stack([sample_lo, sample_hi]), self.processed_paths[1])
