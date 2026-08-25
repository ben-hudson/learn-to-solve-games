from typing import Callable

import torch
import tqdm

from itertools import count
from torch_geometric.transforms import BaseTransform
from torch.utils.data import IterableDataset

from l2s_games.algorithms import ExtraGradient
from l2s_games.envs.spe.utils import dist_to_normal_cone

from .game import SpatialPriceEquilibrium


def solve_instance(instance):
    algorithm = ExtraGradient(2e-2, lambda flow: -instance.operator(flow), torch.relu)
    flow = torch.zeros(instance.n_supply, instance.n_demand)
    for _ in range(5000):
        flow = algorithm.step(flow)

    dist = dist_to_normal_cone(-instance.operator(flow), flow)
    assert torch.isclose(dist, torch.zeros_like(dist), atol=1e-3)
    return flow


class EquilibriumStream(IterableDataset):
    """Infinite stream of (optionally) solved instances."""

    def __init__(
        self,
        sample_fn: Callable,
        n_instances: int = None,
        solve: bool = False,
        quiet: bool = True,
        transform: BaseTransform = None,
    ):
        super().__init__()
        self.sample_fn = sample_fn
        self.n_instances = n_instances
        self.solve = solve
        self.quiet = quiet
        self.transform = transform

    def generate_instances(self):
        counter = range(self.n_instances) if self.n_instances is not None else count()
        progress = counter if self.quiet else tqdm.tqdm(counter)

        for _ in progress:
            instance = self.sample_fn()
            eq = instance.solve() if self.solve else None
            yield instance, eq

    def __iter__(self):
        for instance, eq in self.generate_instances():
            data = instance.to_data()
            data.eq = eq
            yield data if self.transform is None else self.transform(data)


class OperatorStream(EquilibriumStream):
    def __init__(self, sample_fn, sample_domain_fn, n_points_per_instance=1, **kwargs):
        super().__init__(sample_fn, **kwargs)

        self.sample_domain_fn = sample_domain_fn
        self.n_points_per_instance = n_points_per_instance

    def __iter__(self):
        for instance, eq in self.generate_instances():
            data = instance.to_data()
            data.eq = eq
            data.point = self.sample_domain_fn(instance, self.n_points_per_instance).float()
            data.operator = instance.operator(data.point).float()

            yield data if self.transform is None else self.transform(data)
