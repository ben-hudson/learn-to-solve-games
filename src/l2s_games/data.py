"""Dataset construction for the amortized field model.

An *instance* is a normalized vector ``u in [0, 1]^k`` selecting one game from the
family; a *sample* is a point in the game's domain. A training example concatenates
the two -> ``[point | u]`` and targets the game operator's value at that point. The
model conditions on the *normalized* ``u``; the game itself speaks in real units, so
this module owns the ``[0, 1] -> real`` mapping (``denormalize``). Train and test
sets are drawn from *disjoint* instance sets so the test split measures generalization
to unseen parametrizations. Reproducibility is handled by ``lightning.seed_everything``
at the call site, so sampling just uses the global RNG.
"""

import torch
from torch.utils.data import TensorDataset


def denormalize(ranges, normalized):
    """Map a normalized ``[0, 1]^k`` instance to real game parameters via ``ranges``."""
    normalized = torch.as_tensor(normalized, dtype=torch.float32)
    return [low + normalized[i] * (high - low) for i, (low, high) in enumerate(ranges)]


def sample_instances(game, n):
    """``(n, n_params)`` normalized instance vectors uniform in ``[0, 1]``."""
    return torch.rand(n, game.n_params)


def _examples_for_instances(game, instances, points_per_instance):
    """Stack ``[point | u]`` inputs and operator-value targets over instances."""
    inputs, targets = [], []
    for u in instances:
        points = game.sample_points(points_per_instance)
        with torch.no_grad():
            values = game.operator(denormalize(game.ranges, u), points)
        params = u.expand(points_per_instance, -1)
        inputs.append(torch.cat([points, params], dim=-1))
        targets.append(values)
    return torch.cat(inputs), torch.cat(targets)


def build_dataset(game, n_train, n_val, n_test, points_per_instance):
    """Train/val/test ``TensorDataset``s drawn from disjoint instance sets."""
    instances = sample_instances(game, n_train + n_val + n_test)
    splits = instances.split([n_train, n_val, n_test])
    return tuple(
        TensorDataset(*_examples_for_instances(game, split, points_per_instance))
        for split in splits
    )
