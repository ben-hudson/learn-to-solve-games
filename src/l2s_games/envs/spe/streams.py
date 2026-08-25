import torch
import tqdm

from itertools import count
from torch_geometric.transforms import BaseTransform
from torch.utils.data import IterableDataset

from l2s_games.algorithms import ExtraGradient
from l2s_games.envs.spe.utils import dist_to_normal_cone

from .game import SpatialPriceEquilibrium


class EquilibriumStream(IterableDataset):
    """Infinite stream of (optionally) solved instances."""

    def __init__(
        self,
        n_supply: int,
        n_demand: int,
        kappa: float,
        eps: float,
        delta: float,
        scale: float = 1.0,
        n_instances: int = None,
        quiet: bool = True,
        solve: bool = True,
        transform: BaseTransform = None,
    ):
        super().__init__()
        self.n_supply = n_supply
        self.n_demand = n_demand
        self.kappa = kappa
        self.eps = eps
        self.delta = delta
        self.scale = scale

        self.n_instances = n_instances
        self.solve = solve
        self.quiet = quiet
        self.transform = transform

    def solve_instance(self, instance):
        algorithm = ExtraGradient(2e-2, lambda flow: -instance.operator(flow), torch.relu)
        flow = torch.zeros(self.n_supply, self.n_demand)
        for _ in range(5000):
            flow = algorithm.step(flow)

        dist = dist_to_normal_cone(-instance.operator(flow), flow)
        assert torch.isclose(dist, torch.zeros_like(dist), atol=1e-3)
        return flow

    def generate_instances(self):
        counter = range(self.n_instances) if self.n_instances is not None else count()
        progress = counter if self.quiet else tqdm.tqdm(counter)

        for _ in progress:
            instance = SpatialPriceEquilibrium.random_bipartite(
                self.n_supply, self.n_demand, self.kappa, self.eps, self.delta, self.scale
            )
            if self.solve:
                instance.eq = self.solve_instance(instance)

            yield instance

    def __iter__(self):
        for instance in self.generate_instances():
            data = instance.to_dict()
            yield data if self.transform is None else self.transform(data)


class OperatorStream(EquilibriumStream):
    def __init__(self, *args, sample_lo, sample_hi, n_points_per_instance=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_box = torch.distributions.Uniform(sample_lo, sample_hi)
        self.n_points_per_instance = n_points_per_instance

    def __iter__(self):
        for instance in self.generate_instances():
            data = instance.to_dict()
            data.point = self.sample_box.sample((self.n_points_per_instance,)).float()
            data.operator = instance.operator(data.point).float()

            yield data if self.transform is None else self.transform(data)
