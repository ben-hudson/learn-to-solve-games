"""Symmetric zero-sum games -- rock-paper-scissors and its larger-matrix cousins.

A single population plays itself under an antisymmetric payoff matrix ``A``. The
payoff to action ``i`` against strategy ``x`` is ``(A x)_i``. Charting the simplex at
the Nash (the null vector of ``A``) makes the constant term vanish, so the operator is
``B^T A (nash + B z) = (B^T A B) z`` in chart coordinates. ``B^T A B`` is antisymmetric
for every antisymmetric ``A``, so the field is purely rotational with its zero at the
chart origin -- the rock-paper-scissors structure, for any number of actions.
"""

import torch

from l2s_games.envs.base import ParamSpec
from l2s_games.envs.matrix import MatrixGame


def _upper_triangle_indices(n):
    """Row/col index pairs of the strict upper triangle of an ``n x n`` matrix."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


class SymmetricZeroSumGame(MatrixGame):
    """One population, antisymmetric payoff, ``n_actions`` actions -> ``n_actions - 1`` domain."""

    def __init__(self, n_actions=3, lim=0.4, weight_range=(0.5, 2.0)):
        super().__init__(n_actions=(n_actions,), lim=lim, weight_range=weight_range)
        self.entries = _upper_triangle_indices(n_actions)

    @property
    def param_specs(self):
        low, high = self.weight_range
        return tuple(ParamSpec(f"w_{i}{j}", low, high) for i, j in self.entries)

    def payoff_matrices(self, params):
        params = torch.as_tensor(params, dtype=torch.float32)
        size = self.n_actions[0]
        matrix = torch.zeros(size, size)
        for weight, (i, j) in zip(params, self.entries):
            matrix[i, j] = weight
            matrix[j, i] = -weight
        return (matrix,)

    def operator(self, params, points):
        """``F(z) = (B^T A B) z`` -- the Nash-centered tangent field, purely rotational.

        Recentering on the Nash makes the constant term vanish, so the operator is the
        antisymmetric linear map ``B^T A B`` for any antisymmetric ``A`` and any number
        of actions: zero at the chart origin, rotational everywhere.
        """
        points = torch.as_tensor(points, dtype=torch.float32)
        (matrix,) = self.payoff_matrices(params)
        basis = self.bases[0]
        generator = basis.T @ matrix @ basis
        return points @ generator.T


class RockPaperScissors2D(SymmetricZeroSumGame):
    """Classic three-action rock-paper-scissors; its strategy chart is 2D."""

    def __init__(self, lim=0.4, weight_range=(0.5, 2.0)):
        super().__init__(n_actions=3, lim=lim, weight_range=weight_range)
