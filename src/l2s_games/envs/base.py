"""Abstract game interface.

A *game* is a parametric family of vector fields over a fixed domain. The amortized
model learns the family; this interface is what the dataset and training script
consume so neither one hardcodes a particular game's dimensions.

A game speaks only in *real* parameter units. The ``[0, 1] <-> real`` instance
normalization the model conditions on is a dataset concern (see ``data.py``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    """One real-valued game parameter and the range the family is sampled over."""

    name: str
    low: float
    high: float


class Game(ABC):
    """A parametric family of vector fields over a ``domain_dim``-dimensional domain."""

    lim: float  # domain half-extent: points live in [-lim, lim]^domain_dim

    @property
    @abstractmethod
    def domain_dim(self):
        """Dimension of both the operator's input points and its output values."""

    @property
    @abstractmethod
    def param_specs(self):
        """Tuple of ``ParamSpec`` describing each real game parameter."""

    @property
    def n_params(self):
        return len(self.param_specs)

    @property
    def param_names(self):
        return tuple(spec.name for spec in self.param_specs)

    @property
    def ranges(self):
        return tuple((spec.low, spec.high) for spec in self.param_specs)

    @abstractmethod
    def operator(self, params, points):
        """Evaluate the game operator (real ``params``) at ``points``.

        ``points`` has shape ``(..., domain_dim)``; the return has the same shape.
        Must be plain-torch so ``torch.func.jacrev`` can differentiate it.
        """

    @abstractmethod
    def sample_points(self, n):
        """``(n, domain_dim)`` points drawn over the domain."""
