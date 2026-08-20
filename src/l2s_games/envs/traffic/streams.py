import torch
import tqdm

from itertools import count
from l2s_games.envs.traffic.game import NonPotentialCongestion
from torch.utils.data import IterableDataset


class TrafficEquilibriumStream(IterableDataset):
    """Infinite stream of unsolved traffic instances.

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
        self, pume_mapping, base_graph, n_instances=None, kappa=0.0, quiet=True, solve=True, transform=None
    ):
        super().__init__()
        self.pume_mapping = pume_mapping
        self.base_graph = base_graph
        self.n_instances = n_instances
        self.kappa = kappa
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
                self.pume_mapping,
                free_flow_time,
                capacity,
                self.base_graph.b,
                self.base_graph.power,
                kappa=self.kappa,
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
    def __init__(self, pume_mapping, base_graph, cal_set, n_points_per_instance=1, **kwargs):
        super().__init__(pume_mapping, base_graph, **kwargs)

        lower, _ = torch.stack([inst["free_flow_time"] for inst in cal_set], dim=-1).min(dim=-1)
        upper, _ = torch.stack([inst["eq"] for inst in cal_set], dim=-1).max(dim=-1)
        self.roi = torch.distributions.Uniform(lower, upper)

        self.n_points_per_instance = n_points_per_instance

    def __iter__(self):
        for instance in self.generate_instances():
            data = instance.to_data()
            data.point = self.roi.sample((self.n_points_per_instance,))
            op, precond = zip(*(instance.operator_and_preconditioner(point) for point in data.point))
            data.operator = torch.stack(op).float()
            data.preconditioner = torch.stack(precond).float()
            data.preconditioned_operator = data.operator / data.preconditioner

            yield data if self.transform is None else self.transform(data)
