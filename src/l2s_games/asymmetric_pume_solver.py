"""
asymmetric_pume_solver.py

``PUMESolver`` with a **non-potential** supply operator: ``z(c) = A f(c)`` instead of the separable
inverse BPR ``f(c)``. This is the whole asymmetric extension -- it overrides the one seam
``PUMESolver._make_supply`` and inherits everything else (the per-destination PUMCM models, the
reward/flow mappings, the persistent demand loader, ``build_model``, and ``solve``).

The interaction matrix ``A = (1 - eps) I + eps W`` (``W`` row-stochastic and supported on the
link-adjacency graph, so link ``a``'s supply is a convex combination of its own inverse-BPR flow and its
neighbours') is **supplied, never built here**. ``A`` is non-symmetric, hence ``grad z(c) = A diag(f'(c))``
is non-symmetric and the excess supply ``E(c) = z(c) - x(c)`` is not the gradient of any potential.
``eps = 0`` gives ``A = I`` and recovers the separable operator exactly.

``A`` is a **constant of the pipeline**, so it is built once by ``build_interaction_matrix`` during dataset
generation, stored with the dataset, and passed in everywhere thereafter. It is deliberately not rebuilt at
construction time: its values come from an RNG seeded by ``build_asymmetric_interaction_matrix``'s *default*
``seed=42``, which we neither pass nor record, so a rebuild silently depends on that default (and on NumPy's
stream for it) staying fixed. Requiring the matrix means the operator a model trains against is provably the
one whose equilibria were solved, rather than one that merely ought to match.
"""

import torch

from l2s_games.envs.traffic import _canonicalize
from l2s_games.pume_solver import PUMESolver
from pume.operators import AsymmetricBPRSupply, build_asymmetric_interaction_matrix

__all__ = ["AsymmetricPUMESolver", "build_interaction_matrix"]


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


class AsymmetricPUMESolver(PUMESolver):
    """``PUMESolver`` whose supply operator is the non-potential ``z(c) = A f(c)``."""

    def __init__(self, graph, interaction_matrix, **solver_kwargs):
        super().__init__(graph, **solver_kwargs)
        # Required, never defaulted: see the module docstring on why rebuilding from epsilon is unsafe.
        self.interaction_matrix = interaction_matrix

    def _make_supply(self, free_flow_time, capacity, b, power):
        """The non-potential supply ``z(c) = A f(c)`` for one instance's BPR parameters.

        ``validate_monotone=False`` is load-bearing rather than a shortcut: the check is an ``O(m^3)``
        eigendecomposition and this runs once per *operator evaluation* (via ``build_model``), while ``A``
        never changes -- ``build_interaction_matrix`` already asserted its PD symmetric part when the matrix
        was constructed. (That assertion is also weaker than it looks; see the module docstring of
        ``envs/asym_pume_traffic.py`` on why it does not imply a monotone operator.)
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
