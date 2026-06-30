"""Variational-inequality abstraction.

A ``VariationalInequalityFamily`` is a parametric family of VIs: drawing one
``params`` value (a flat tensor for normal-form games, a graph for traffic) and
binding it yields a concrete ``VariationalInequality`` -- a fixed operator, a
projection onto the feasible set, and a domain sampler. Operators are plain-torch
functions of ``(params, points)`` so ``torch.func.jacrev`` differentiates them
(``Consensus`` and any Jacobian method depend on this).

The family also owns the *conditioning* seam: ``model_input(params, points)``
builds whatever the field model consumes -- a flat ``[point | params]`` tensor for
the MLP path, a batched graph for the GNN path -- and ``collate_fn`` batches
examples accordingly. Keeping this on the family lets the dataset, training loop,
and dynamics stay blind to the representation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch


def sample_uniform(ranges):
    """One real parameter vector drawn uniformly within ``ranges`` (list of ``(low, high)``)."""
    bounds = torch.tensor(ranges, dtype=torch.float32)
    low, high = bounds[:, 0], bounds[:, 1]
    return low + (high - low) * torch.rand(len(ranges))


def concat_conditioning(points, params):
    """Flat model input: append the instance ``params`` to every point."""
    points = torch.as_tensor(points, dtype=torch.float32)
    params = torch.as_tensor(params, dtype=torch.float32)
    conditioning = params.expand(*points.shape[:-1], params.shape[-1])
    return torch.cat([points, conditioning], dim=-1)


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
    def model_input(self, params, points):
        """Build the field model's input -- the conditioning seam (flat tensor or graph)."""

    def project(self, params, points):
        """Project ``points`` onto the feasible set (default: unconstrained)."""
        return points

    @property
    def collate_fn(self):
        """DataLoader collation for this representation (default: tensor stacking)."""
        return None


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
