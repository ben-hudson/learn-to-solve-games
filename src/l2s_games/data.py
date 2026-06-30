"""Dataset construction for the amortized field model.

An *instance* is one ``params`` value drawn from the family; a *sample* is a point in
that instance's domain. A training example is ``family.model_input(params, point)`` (for
flat games, ``[point | params]``) targeting the operator value there. Train/val/test
instances are drawn independently so the test split measures generalization to unseen
parametrizations. Normalization is **fit on the train split only** -- a per-feature
standardizer, decoupled from how params are sampled, so it adapts to whatever distribution
(uniform, or clamped Gaussian for traffic) actually came out. Reproducibility is handled by
``lightning.seed_everything`` at the call site, so sampling just uses the global RNG.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class Standardizer:
    """Per-feature ``(x - mean) / std`` map, fit from data, with its inverse."""

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, x):
        return cls(x.mean(dim=0), x.std(dim=0))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


@dataclass(frozen=True)
class Normalizer:
    """The fitted input/target standardizers the model trains and predicts through."""

    input: Standardizer
    target: Standardizer


def _examples_for_instances(family, instances, points_per_instance):
    """Stack model inputs and operator-value targets over instances."""
    inputs, targets = [], []
    for params in instances:
        points = family.sample_domain(params, points_per_instance)
        with torch.no_grad():
            values = family.operator(params, points)
        inputs.append(family.model_input(params, points))
        targets.append(values)
    return torch.cat(inputs), torch.cat(targets)


def build_dataset(family, n_train, n_val, n_test, points_per_instance):
    """Train/val/test ``TensorDataset``s (normalized) plus the fitted ``Normalizer``.

    The normalizer is fit on the train split and applied to all three, so val/test see
    no statistics of their own.
    """
    raw = [
        _examples_for_instances(family, [family.sample_params() for _ in range(n)], points_per_instance)
        for n in (n_train, n_val, n_test)
    ]
    train_inputs, train_targets = raw[0]
    normalizer = Normalizer(Standardizer.fit(train_inputs), Standardizer.fit(train_targets))
    datasets = tuple(
        TensorDataset(normalizer.input.transform(inputs), normalizer.target.transform(targets))
        for inputs, targets in raw
    )
    return datasets, normalizer
