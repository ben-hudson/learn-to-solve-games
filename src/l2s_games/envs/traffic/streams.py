import torch
import tqdm

from itertools import count
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch.utils.data import IterableDataset

from .game import NonPotentialCongestion, NetworkLoading


class TrafficEquilibriumStream(IterableDataset):
    """Infinite stream of (optionally) solved traffic instances.

    Samples the same perturbations of ``base_graph`` as
    ``TrafficEquilibriumDataset.generate_instances`` and yields ``Data`` objects
    with the same fields as ``NonPotentialCongestion.to_data`` minus ``eq``, so the
    dataset transforms apply unchanged.

    Wrap in a ``DataLoader`` (with ``GraphToTensorDict`` and
    ``collate_fn=torch.stack``) to get batched ``TensorDict``s. Instances are
    drawn i.i.d., so multi-worker loading needs no sharding: each worker
    iterates its own copy of the stream, and PyTorch derives a distinct RNG
    seed per worker, keeping the streams decorrelated.
    """

    def __init__(
        self,
        network_loading: NetworkLoading,
        base_graph: Data,
        n_instances: int = None,
        quiet: bool = True,
        solve: bool = True,
        transform: BaseTransform = None,
    ):
        super().__init__()
        self.network_loading = network_loading
        self.base_graph = base_graph
        self.n_instances = n_instances
        self.solve = solve
        self.quiet = quiet
        self.transform = transform

    def generate_instances(self):
        counter = range(self.n_instances) if self.n_instances is not None else count()
        progress = counter if self.quiet else tqdm.tqdm(counter)

        # we warm start with the prev equilibrium, but the first one gets fft
        # empirically, this is a bit faster than starting from fft every time
        warm_start = self.base_graph.free_flow_time * 1.1
        for _ in progress:
            free_flow_time = self.base_graph.free_flow_time * (
                0.9 + 0.2 * torch.rand_like(self.base_graph.free_flow_time)
            )
            capacity = self.base_graph.capacity * (0.9 + 0.2 * torch.rand_like(self.base_graph.capacity))
            instance = NonPotentialCongestion(
                self.base_graph.edge_index,
                self.network_loading,
                free_flow_time,
                capacity,
                self.base_graph.b,
                self.base_graph.power,
            )
            if self.solve:
                instance.solve(initial_cost=instance.project_costs(warm_start), max_iters=2000, tol=1e-4)
                assert instance.eq_info["converged"]
                warm_start = instance.eq

            yield instance

    def __iter__(self):
        for instance in self.generate_instances():
            data = instance.to_data()
            yield data if self.transform is None else self.transform(data)


class TrafficOperatorStream(TrafficEquilibriumStream):
    def __init__(self, network_loading, base_graph, sample_lo, sample_hi, n_points_per_instance=1, **kwargs):
        super().__init__(network_loading, base_graph, **kwargs)

        self.sample_box = torch.distributions.Uniform(sample_lo, sample_hi)
        self.n_points_per_instance = n_points_per_instance

    def __iter__(self):
        for instance in self.generate_instances():
            data = instance.to_data()
            data.point = self.sample_box.sample((self.n_points_per_instance,)).float()
            op, precond = zip(*(instance.operator_and_preconditioner(point) for point in data.point))
            data.operator = torch.stack(op).float()
            data.preconditioner = torch.stack(precond).float()
            data.preconditioned_operator = data.operator / data.preconditioner

            yield data if self.transform is None else self.transform(data)
