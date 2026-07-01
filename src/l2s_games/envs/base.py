"""Variational-inequality abstraction.

A ``VariationalInequalityFamily`` is a parametric family of VIs: drawing one
``params`` value (a flat tensor for normal-form games, a graph for traffic) and
binding it yields a concrete ``VariationalInequality`` -- a fixed operator, a
projection onto the feasible set, and a domain sampler. Operators are plain-torch
functions of ``(params, points)`` so ``torch.func.jacrev`` differentiates them
(``Consensus`` and any Jacobian method depend on this).

The family owns the *conditioning* through three representation seams, so the dataset, training
loop, and dynamics stay blind to whether an instance is a flat vector or a graph:

- ``model_input(params, point)`` -- the **raw input item** carrying one domain point, *before*
  featurization: ``{"point", "params"}`` for a flat game, a PyG ``Data`` with ``.cost`` set for
  traffic.
- ``transform`` -- a per-item callable that builds ``feats`` (and any structure the model needs)
  from the raw item, applied **lazily on every ``__getitem__``**. Default: identity. Featurization
  lives in ``transforms.py`` precisely so it can run lazily and nothing is cached -- see the note
  there. Flat games override it with ``ConcatConditioning``; traffic with a graph ``Compose``.
- ``collate_fn(items)`` -- batches a list of transformed input items into the model's input.
  Default: ``torch.utils.data.default_collate`` (stacks the flat ``{feats}`` dicts); traffic
  overrides it with a dense stack of its single-topology graphs.

Operators are plain-torch functions of ``(params, points)`` so ``torch.func.jacrev`` differentiates
them (``Consensus`` and any Jacobian method depend on this).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import default_collate


def sample_uniform(ranges):
    """One real parameter vector drawn uniformly within ``ranges`` (list of ``(low, high)``)."""
    bounds = torch.tensor(ranges, dtype=torch.float32)
    low, high = bounds[:, 0], bounds[:, 1]
    return low + (high - low) * torch.rand(len(ranges))


def _identity(item):
    return item


class VariationalInequalityFamily(ABC):
    """A parametric family of variational inequalities."""

    @abstractmethod
    def sample_params(self):
        """Draw one instance's parameters in real units."""

    @abstractmethod
    def operator(self, params, points):
        """Evaluate the VI operator at ``points`` (same shape out). Must be plain-torch."""

    @abstractmethod
    def sample_domain(self, params, n):
        """``(n, ...)`` feasible points for the instance ``params``."""

    @abstractmethod
    def model_input(self, params, point):
        """Build the **raw** input item carrying one domain point -- the conditioning seam."""

    @property
    def transform(self):
        """Per-item callable that builds ``feats`` from a raw item (default: identity)."""
        return _identity

    collate_fn = staticmethod(default_collate)

    def project(self, params, points):
        """Project ``points`` onto the feasible set (default: unconstrained)."""
        return points


@dataclass(frozen=True)
class VariationalInequality:
    """A concrete instance: the family's maps with one ``params`` value bound in."""

    operator: Callable
    project: Callable
    sample_domain: Callable


def bind(family, params):
    """Freeze ``params`` into a concrete ``VariationalInequality`` (an ergonomic ``v(z)``)."""
    return VariationalInequality(
        operator=lambda points: family.operator(params, points),
        project=lambda points: family.project(params, points),
        sample_domain=lambda n: family.sample_domain(params, n),
    )
