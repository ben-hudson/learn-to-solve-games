"""Non-potential supply ``z(c) = A f(c) + B c`` around a separable inverse-BPR core ``f``.

Two independent couplings generalize the separable inverse BPR:

* ``A`` -- **multiplicative**, ``A = (1 - eps) I + eps W`` with ``W`` row-stochastic on the
  link-adjacency graph, so link ``a``'s supply is a convex combination of its own inverse-BPR flow
  and its neighbours'. ``A = I`` means none. Built by ``build_interaction_matrix``.
* ``B`` -- **additive**, ``B = kappa S`` with ``S`` antisymmetric. ``B = 0`` means none. Built by
  ``build_rotation_matrix``.

Either one makes ``grad z(c) = A diag(f'(c)) + B`` non-symmetric, so the excess supply
``E(c) = z(c) - x(c)`` is not the gradient of any potential. They are *not* interchangeable:

**Multiplicative coupling cannot be unconditionally monotone.** ``sym(A D) >= 0`` for every
positive diagonal ``D`` forces ``A`` diagonal (check the extreme rays ``D = diag(e_k)``), so
coupled-and-monotone is impossible in that form; in practice monotonicity survives only while
``eps`` is small relative to the spread of ``f'``.

**Additive coupling is monotone unconditionally.** For antisymmetric ``B``, ``<B dc, dc>`` is
identically zero, so monotonicity holds for any ``kappa``. The field can be made
rotation-*dominated* while remaining monotone, and ``B``'s zero diagonal keeps
``jacobian_diagonal`` exactly ``f'(c)`` -- PUME's ``supply_diagonal`` metric is diagonal, the
rotation purely off-diagonal, so preconditioning cannot wash it out.

Both matrices are **constants of the pipeline**: built once during dataset generation, stored with
the dataset, and passed in everywhere thereafter. That is essential for ``A``, whose values come
from an RNG inside ``build_asymmetric_interaction_matrix``; ``B`` is deterministic from the
topology, but it follows the same rule so the dataset carries the operator in full.
"""

import torch

from pume.operators import InverseBPRSupply, SupplyOperator

from .utils import sparse_incidence_matrix


def build_rotation_matrix(edge_index: torch.Tensor, kappa: float) -> torch.Tensor:
    """``B = kappa S`` with ``S`` antisymmetric and of unit spectral norm, in ``edge_index`` order.

    ``S = M - M^T`` for ``M[a, b] = 1`` when link ``a`` feeds into link ``b``
    (``head(a) == tail(b)``), so it reads as downstream coupling minus upstream and is supported on
    links sharing a node -- the same topology the GNN sees. Antisymmetry is structural, and
    ``S_aa = 0`` gives the zero diagonal that keeps ``jacobian_diagonal`` exactly separable.

    A pure function of the topology -- no RNG, nothing silently irreproducible. ``kappa`` is folded
    into the returned matrix, so one matrix fully defines the coupling and no scalar can drift out
    of sync with it; it stays recoverable as ``||B||_2`` because ``S`` is normalised.
    """
    incidence = sparse_incidence_matrix(edge_index, dtype=torch.float64).to_dense()
    arrives_at = incidence.clamp(min=0)  # [node, link] = 1 iff head(link) == node
    leaves_from = (-incidence).clamp(min=0)  # [node, link] = 1 iff tail(link) == node
    downstream = arrives_at.T @ leaves_from  # [a, b] = 1 iff a feeds into b
    skew = downstream - downstream.T
    return kappa * skew / torch.linalg.matrix_norm(skew, ord=2)


class CoupledBPRSupply(SupplyOperator):
    """Supply ``z(c) = A f(c) + B c`` around a separable ``InverseBPRSupply`` core.

    Replaces PUME's ``AsymmetricBPRSupply``, which only implements the multiplicative half and
    lacks the ``O(m)`` ``jacobian_diagonal`` (it materializes a dense Jacobian per evaluation).

    There is no ``validate_monotone`` argument. A PD symmetric part of ``A`` does not establish
    that the operator is monotone (see the module docstring), so a per-evaluation check would be
    an expensive ``O(m^3)`` reassurance about the wrong property.
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
        """``A_ii f'_i`` -- exact because ``B`` has a zero diagonal, and ``O(m)``.

        This is what makes the additive coupling invisible to PUME's ``supply_diagonal``
        preconditioner: the metric only ever sees the separable part, so rotation survives it.
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
