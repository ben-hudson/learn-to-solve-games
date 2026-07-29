"""
asymmetric_pume_solver.py

``PUMESolver`` with a **non-potential** supply operator. It overrides the one seam
``PUMESolver._make_supply`` and inherits everything else (the per-destination PUMCM models, the
reward/flow mappings, the persistent demand loader, ``build_model``, and ``solve``).

The supply generalizes the separable inverse BPR ``f(c)`` with two independent couplings::

    z(c) = A f(c) + B c

* ``A`` -- **multiplicative** coupling, ``A = (1 - eps) I + eps W`` with ``W`` row-stochastic on the
  link-adjacency graph, so link ``a``'s supply is a convex combination of its own inverse-BPR flow and its
  neighbours'. ``A = I`` means none. Built by ``build_interaction_matrix``.
* ``B`` -- **additive** coupling, ``B = kappa S`` with ``S`` antisymmetric. ``B = 0`` means none. Built by
  ``build_rotation_matrix``.

Either one makes ``grad z(c) = A diag(f'(c)) + B`` non-symmetric, so ``E(c) = z(c) - x(c)`` is not the
gradient of any potential. They are *not* interchangeable, and the difference is the reason both exist:

**Multiplicative coupling cannot be unconditionally monotone.** ``sym(A D) >= 0`` for every positive
diagonal ``D`` is linear in ``D``, so it must hold on the extreme rays ``D = diag(e_k)``, where
``x^T A D x = x_k (A^T x)_k``; fixing ``k`` and setting ``x_k = t`` gives ``a_kk t^2 + t sum_{i!=k} a_ik s_i``,
and any nonzero off-diagonal lets a small ``t`` of opposite sign make it negative. So ``A f(c)`` is monotone
for arbitrary positive ``f'`` **iff ``A`` is diagonal** -- coupled-and-monotone is impossible in that form,
and in practice monotonicity survives only while ``eps`` is small relative to the spread of ``f'``.

**Additive coupling is monotone unconditionally.** For antisymmetric ``B``,
``<z(c1) - z(c2), c1 - c2> = <f(c1) - f(c2), c1 - c2> + <B dc, dc>`` and the second term is *identically
zero*, so monotonicity holds for any ``kappa`` with no condition at all. Two consequences follow, and both
are why this form is the one that can be pushed hard:

* ``mu = lambda_min(sym(grad E))`` stays pinned by ``diag(f')`` and ``-grad x`` while ``L`` grows like
  ``||B||``, so the field can be made rotation-*dominated* while remaining monotone.
* ``B`` has a zero diagonal, so ``jacobian_diagonal`` is exactly ``f'(c)`` and PUME's ``supply_diagonal``
  metric is *identical* to the separable case -- the metric is diagonal, the rotation is purely
  off-diagonal, so preconditioning cannot wash it out.

Both matrices are **constants of the pipeline**: built once during dataset generation, stored with the
dataset, and passed in everywhere thereafter. Neither is rebuilt at construction time. That is essential for
``A``, whose values come from an RNG seeded by ``build_asymmetric_interaction_matrix``'s *default*
``seed=42`` that we neither pass nor record; ``B`` is deterministic from the topology, but it follows the
same rule so that the dataset carries the operator in full, under one policy rather than two.
"""

import torch

from l2s_games.envs.traffic import _canonicalize
from l2s_games.pume_solver import PUMESolver
from pume.operators import InverseBPRSupply, SupplyOperator, build_asymmetric_interaction_matrix

__all__ = ["AsymmetricPUMESolver", "CoupledBPRSupply", "build_interaction_matrix", "build_rotation_matrix"]


def build_interaction_matrix(base_graph, epsilon):
    """``A = (1 - epsilon) I + epsilon W`` for a road graph, in **canonical link order**.

    The single place ``A`` is ever constructed; call it once when generating a dataset and store the result.

    Canonicalizing first is load-bearing: ``A``'s row/column ``k`` has to be the same link ``k`` as every
    per-edge BPR attribute, and ``_canonicalize`` is what establishes that ordering (the family canonicalizes
    its own ``base_graph`` identically, so a matrix built here lines up with it). A permuted ``A`` would leave
    every structural property intact -- still row-stochastic, still non-symmetric, still PD symmetric part --
    while silently solving a different network.
    """
    graph = _canonicalize(base_graph)
    edge_list = graph.edge_index.t().tolist()  # (tail, head) per link, in canonical link order
    return build_asymmetric_interaction_matrix(graph.num_edges, edge_list, epsilon=epsilon)


def build_rotation_matrix(base_graph, kappa):
    """``B = kappa S`` with ``S`` antisymmetric and of unit spectral norm, in **canonical link order**.

    ``S = M - M^T`` for ``M[a, b] = 1`` when link ``a`` feeds into link ``b`` (``head(a) == tail(b)``), so it
    reads as downstream coupling minus upstream and is supported on links sharing a node -- the same topology
    the GNN sees. Antisymmetry is structural, not enforced, and ``S_aa = M_aa - M_aa = 0`` gives the zero
    diagonal that keeps ``jacobian_diagonal`` exactly separable (see the module docstring).

    Unlike ``A`` this is a pure function of the topology -- no RNG, hence no seed and nothing that could be
    silently irreproducible. ``kappa`` is folded into the returned matrix rather than kept alongside it, so
    one matrix fully defines the coupling and no scalar can drift out of sync with it; it stays exactly
    recoverable as ``||B||_2`` because ``S`` is normalised.
    """
    graph = _canonicalize(base_graph)
    tails, heads = graph.edge_index
    links = torch.arange(graph.num_edges)
    arrives_at = torch.zeros(graph.num_nodes, graph.num_edges, dtype=torch.float64)
    leaves_from = torch.zeros(graph.num_nodes, graph.num_edges, dtype=torch.float64)
    arrives_at[heads, links] = 1.0
    leaves_from[tails, links] = 1.0
    downstream = arrives_at.T @ leaves_from  # [a, b] = 1 iff head(a) == tail(b), i.e. a feeds into b
    skew = downstream - downstream.T
    return kappa * skew / torch.linalg.matrix_norm(skew, ord=2)


class CoupledBPRSupply(SupplyOperator):
    """Supply ``z(c) = A f(c) + B c`` around a separable ``InverseBPRSupply`` core.

    Replaces PUME's ``AsymmetricBPRSupply``, which only implements the multiplicative half. It also supplies
    the ``O(m)`` ``jacobian_diagonal`` that PUME's version lacks (it inherits the base class's
    ``jacobian(c).diag()``, materializing a dense ``[m, m]`` on every evaluation) -- see
    ``docs/pume_issues.md``.

    There is no ``validate_monotone`` argument. A PD symmetric part of ``A`` does not establish that the
    operator is monotone (see the module docstring), so a per-evaluation check would be an expensive
    ``O(m^3)`` reassurance about the wrong property; what ``A`` can be checked for is asserted once, in
    ``build_interaction_matrix``.
    """

    def __init__(self, free_flow_time, capacity, interaction_matrix, rotation_matrix, alpha, beta, eps=1e-6):
        self.inner = InverseBPRSupply(free_flow_time, capacity, alpha=alpha, beta=beta, eps=eps)
        self.interaction_matrix = interaction_matrix.to(torch.float64)
        self.rotation_matrix = rotation_matrix.to(torch.float64)

    def __call__(self, costs):
        return self.interaction_matrix @ self.inner(costs) + self.rotation_matrix @ costs

    def jacobian(self, costs):
        """``grad z(c) = A diag(f'(c)) + B``."""
        return self.interaction_matrix @ torch.diag(self.inner.jacobian_diagonal(costs)) + self.rotation_matrix

    def jacobian_diagonal(self, costs):
        """``A_ii f'_i`` -- exact because ``B`` has a zero diagonal, and ``O(m)`` rather than a dense Jacobian.

        This is what makes the additive coupling invisible to PUME's ``supply_diagonal`` preconditioner: the
        metric only ever sees the separable part, so rotation survives it.
        """
        return self.interaction_matrix.diagonal() * self.inner.jacobian_diagonal(costs)

    @property
    def is_separable(self):
        return False

    def to(self, device, dtype=None):
        self.inner.to(device, dtype)
        self.interaction_matrix = self.interaction_matrix.to(device=device)
        self.rotation_matrix = self.rotation_matrix.to(device=device)
        return self


class AsymmetricPUMESolver(PUMESolver):
    """``PUMESolver`` whose supply operator is the non-potential ``z(c) = A f(c) + B c``."""

    def __init__(self, graph, interaction_matrix, rotation_matrix, **solver_kwargs):
        super().__init__(graph, **solver_kwargs)
        # Both required, never defaulted: the module docstring covers why these are supplied rather than built.
        self.interaction_matrix = interaction_matrix
        self.rotation_matrix = rotation_matrix

    def _make_supply(self, free_flow_time, capacity, b, power):
        """The coupled supply for one instance's BPR parameters (note ``b -> alpha``, ``power -> beta``)."""
        return CoupledBPRSupply(
            free_flow_time=free_flow_time.double(),
            capacity=torch.clamp(capacity.double(), min=1e-8),
            interaction_matrix=self.interaction_matrix,
            rotation_matrix=self.rotation_matrix,
            alpha=b.double(),
            beta=power.double(),
            eps=1e-6,
        )
