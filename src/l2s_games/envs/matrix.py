"""Normal-form matrix games as parametric vector fields.

A finite game has one or more populations; population ``p`` has ``n_p`` actions and a
mixed strategy on the ``(n_p - 1)``-simplex. Each simplex is charted with an orthonormal
Helmert basis (columns perpendicular to the all-ones vector), so a domain point splits
into per-population chart coordinates and the operator lives in the tangent space.

``MatrixGame`` holds the shared chart machinery (per-population Helmert bases, domain
dimension, point sampling). Concrete games supply their payoff matrices and assemble
the operator from the bases -- a single-population symmetric game collapses to a linear
rotational field (see ``rps.py``); a two-population game (e.g. Prisoner's Dilemma)
charts each population from its uniform point and couples their payoff gradients.
"""

from abc import abstractmethod

import torch

from l2s_games.envs.base import VariationalInequalityFamily, sample_uniform
from l2s_games.transforms import ConcatConditioning


def helmert_basis(n):
    """``(n, n-1)`` orthonormal columns, each perpendicular to the all-ones vector.

    Column ``k`` (1-indexed) is ``(1, ..., 1, -k, 0, ..., 0) / sqrt(k(k+1))`` with the
    ``-k`` in position ``k`` -- the standard Helmert contrasts. For ``n=3`` this gives
    ``(1,-1,0)/sqrt(2)`` and ``(1,1,-2)/sqrt(6)``.
    """
    basis = torch.zeros(n, n - 1)
    for k in range(1, n):
        column = torch.zeros(n)
        column[:k] = 1.0
        column[k] = -k
        basis[:, k - 1] = column / (k * (k + 1)) ** 0.5
    return basis


class MatrixGame(VariationalInequalityFamily):
    """Abstract normal-form game charted on a product of simplex tangent spaces."""

    def __init__(self, n_actions, lim, weight_range):
        self.n_actions = tuple(n_actions)
        self.lim = lim
        self.weight_range = weight_range
        self.bases = [helmert_basis(n) for n in self.n_actions]
        self.chart_dims = [n - 1 for n in self.n_actions]
        self._domain_dim = sum(self.chart_dims)

    @property
    def domain_dim(self):
        return self._domain_dim

    @property
    def n_params(self):
        return len(self.ranges)

    def sample_params(self):
        return sample_uniform(self.ranges)

    def sample_domain(self, params, n):
        return (2 * torch.rand(n, self.domain_dim) - 1) * self.lim

    def model_input(self, params, point):
        return {"point": point, "params": params}

    @property
    def transform(self):
        return ConcatConditioning()

    @property
    @abstractmethod
    def ranges(self):
        """``(low, high)`` sampling range for each real parameter."""

    @abstractmethod
    def payoff_matrices(self, params):
        """Real ``params`` -> one payoff matrix per population."""
