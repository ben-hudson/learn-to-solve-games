"""Dataset construction for the amortized field model.

An *instance* is a normalized parametrization ``p in [0, 1]^3`` selecting one
field from the family; a *sample* is a point ``(x, y) in [-lim, lim]^2``. A
training example concatenates the two -> ``[x, y, p0, p1, p2]`` and targets the
field value ``v(x, y)`` for that instance. Train and test sets are drawn from
*disjoint* instance sets so the test split measures generalization to unseen
parametrizations. Reproducibility is handled by ``lightning.seed_everything``
at the call site, so sampling just uses the global RNG.
"""

import torch
from torch.utils.data import TensorDataset

from l2s_games.envs.toy import make_field


def sample_instances(n):
    """``(n, 3)`` instance parameters uniform in ``[0, 1]``."""
    return torch.rand(n, 3)


def sample_points(n, lim):
    """``(n, 2)`` coordinates uniform in ``[-lim, lim]^2``."""
    return (2 * torch.rand(n, 2) - 1) * lim


def _examples_for_instances(instances, points_per_instance, lim, ranges):
    """Stack ``[points | params]`` inputs and field-value targets over instances."""
    inputs, targets = [], []
    for p in instances:
        points = sample_points(points_per_instance, lim)
        with torch.no_grad():
            values = make_field(p, ranges)(points)
        params = p.expand(points_per_instance, -1)
        inputs.append(torch.cat([points, params], dim=-1))
        targets.append(values)
    return torch.cat(inputs), torch.cat(targets)


def build_dataset(n_train, n_val, n_test, points_per_instance, lim, ranges):
    """Train/val/test ``TensorDataset``s drawn from disjoint instance sets."""
    instances = sample_instances(n_train + n_val + n_test)
    splits = instances.split([n_train, n_val, n_test])
    return tuple(
        TensorDataset(*_examples_for_instances(split, points_per_instance, lim, ranges))
        for split in splits
    )
