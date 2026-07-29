"""
asymmetric_pume_solver.py

``PUMESolver`` with a **non-potential** supply operator: ``z(c) = A f(c)`` instead of the separable
inverse BPR ``f(c)``. This is the whole asymmetric extension -- it overrides the one seam
``PUMESolver._make_supply`` and inherits everything else (the per-destination PUMCM models, the
reward/flow mappings, the persistent demand loader, ``build_model``, and ``solve``).

The interaction matrix ``A = (1 - eps) I + eps W`` is built **once per graph structure**: ``W`` is
row-stochastic and supported on the link-adjacency graph (links sharing a node), so link ``a``'s supply
is a convex combination of its own inverse-BPR flow and its neighbours'. ``A`` is non-symmetric, hence
``grad z(c) = A diag(f'(c))`` is non-symmetric and the excess supply ``E(c) = z(c) - x(c)`` is not the
gradient of any potential -- while ``sym(A) = (A + A^T)/2`` stays positive definite, so the VI remains
monotone. ``eps = 0`` gives ``A = I`` and recovers the separable operator exactly.
"""

import torch

from l2s_games.pume_solver import PUMESolver
from pume.operators import AsymmetricBPRSupply, build_asymmetric_interaction_matrix


class AsymmetricPUMESolver(PUMESolver):
    """``PUMESolver`` whose supply operator is the non-potential ``z(c) = A f(c)``."""

    def __init__(self, graph, epsilon=0.05, **solver_kwargs):
        super().__init__(graph, **solver_kwargs)
        self.epsilon = epsilon
        # Built once per structure: A depends only on the topology and epsilon. Deliberately *not*
        # seeded from the global RNG -- families are rebuilt inside dataloader workers, which
        # `seed_everything(..., workers=True)` seeds distinctly, so a global draw here would hand every
        # worker a different operator. PUME's builder default keeps A a pure function of its arguments.
        # `graph` is the canonicalized base graph, so link k in A is the same link k as every per-edge
        # attr (the ordering invariant `traffic._canonicalize` establishes).
        edge_list = graph.edge_index.t().tolist()  # (tail, head) per link, in canonical link order
        self.interaction_matrix = build_asymmetric_interaction_matrix(
            graph.num_edges, edge_list, epsilon=epsilon
        )

    def _make_supply(self, free_flow_time, capacity, b, power):
        """The non-potential supply ``z(c) = A f(c)`` for one instance's BPR parameters.

        ``validate_monotone=False`` is load-bearing rather than a shortcut: the check is an ``O(m^3)``
        eigendecomposition, this runs once per *operator evaluation* (via ``build_model``), and
        ``build_asymmetric_interaction_matrix`` already asserted the PD symmetric part in ``__init__``
        for an ``A`` that never changes afterwards.
        """
        return AsymmetricBPRSupply(
            free_flow_time=free_flow_time.double(),
            capacity=torch.clamp(capacity.double(), min=1e-8),
            interaction_matrix=self.interaction_matrix,
            alpha=b.double(),
            beta=power.double(),
            eps=1e-6,
            validate_monotone=False,
        )
