"""Non-potential (asymmetric) PUME Markov traffic equilibrium as a variational inequality family.

Identical to ``PUMEMarkovTrafficEquilibrium`` (see ``pume_traffic.py``) in every respect -- same road-graph
instances, same cost-space domain, same excess-supply residual ``E(c) = z(c) - x(c)``, same conditioning
seam, same preconditioning -- except that the supply is **non-separable**:

    z(c) = A f(c),    A = (1 - eps) I + eps W

with ``f`` the element-wise inverse BPR and ``W`` row-stochastic on the link-adjacency graph, so link
``a``'s supply is a convex combination of its own inverse-BPR flow (weight ``1 - eps``) and its
neighbours' (total weight ``eps``). Everything is inherited through the ``_make_solver`` hook; only the
supply operator differs (see ``asymmetric_pume_solver.AsymmetricPUMESolver``).

**Why this family exists.** ``grad z(c) = A diag(f'(c))`` is non-symmetric whenever ``A`` is, so
``E(c)`` is not the gradient of any potential and the VI is genuinely a *game* rather than a convex
minimization in disguise. The repo's other non-potential fields (``toy``'s rotational well, ``rps``) are
flat 2-D families on the MLP path; this is the first one at traffic scale on the graph path, so the
game-dynamics comparison has something rotational to bite on there.

**Monotonicity is conditional, not guaranteed -- check it for the ``eps`` you use.**
``build_asymmetric_interaction_matrix`` enforces ``sym(A) = (A + A^T)/2 > 0``, and PUME's
``AsymmetricBPRSupply`` docstring reads that as implying a monotone VI. It does not. Monotonicity needs
``sym(grad z) = sym(A diag(f'(c))) >= 0``, and a PD symmetric part of ``A`` does not survive scaling by an
arbitrary positive diagonal: for ``A = [[1, 2], [-2, 1]]``, ``sym(A) = I > 0`` while
``D = diag(1, 100)`` gives ``sym(A D) = [[1, 99], [99, 100]]``, whose determinant is negative. Nor is this
about asymmetry as such -- any non-diagonal coupling, symmetric included, fails once ``diag(f')`` is
spread enough. Measured here (analytically, at a feasible cost with ``f'`` spanning ~770x across links):
``lambda_min sym(A) = +0.30``, ``lambda_min diag(f') = +0.06``, ``lambda_min sym(A diag(f')) = -0.09``.

Practically, monotonicity holds while the spread of ``f'`` over *coupled* links stays below roughly
``4 / (eps / degree)^2``, so it is an ``eps`` question, and an empirical one -- see the table below and
``tests/test_asym_pume_traffic.py``, which *measures* it rather than assuming it. Two consequences:

- The training-time monotonicity prior (``--monotonicity --constraint_space raw``) is only well specified
  where the true operator really is monotone. At an ``eps`` that violates it, the penalty fights the field
  it is regressing.
- ``optimistic``'s convergence guarantee is for monotone Lipschitz operators, so it is on firm ground only
  in the same regime. It remains the right driver over ``projection`` either way (see ``algorithms.py``),
  since projection has no guarantee even in the monotone case.

Separately, the *preconditioned* field ``M^{-1} E`` is not monotone at all -- ``M`` is state-dependent, and
``lambda_min sym(grad)`` is negative for the symmetric family too. That is pre-existing and is why
``--constraint_space raw`` is the default.

``eps = 0`` gives ``A = I`` and reproduces ``pume_traffic`` exactly, which is the cheapest regression
test of the whole path.

**Choosing eps** -- a *generation-time* decision, since it only enters through ``build_interaction_matrix``
(there is no ``epsilon`` on this class, and none on the training scripts). The builder keeps ``A`` close to
the identity, so the asymmetry it induces is milder than ``eps`` suggests -- ``eps`` is split across each
link's neighbours, and what the dynamics actually see is ``max|J - J^T| / max|J|`` for ``J = grad E``.
Measured on Sioux Falls at ``c = 3 t0``:

    eps          0.0    0.05    0.2     0.5              >= ~0.72
    rel. asym.   0      0.8%    3.4%    8.8%             builder rejects
    monotone?    yes    yes     yes     NO (-0.037)      (sym(A) PD bound)

So ``0.05`` and ``0.2`` are gentle-to-moderate perturbations that stayed monotone at every cost probed,
while ``0.5`` is measurably non-monotone. Stronger rotation and a valid monotonicity prior
pull in opposite directions here; pick ``eps`` for whichever the experiment needs, and re-measure if the
network or the cost box changes. The builder's own upper limit (``eps < 1/(1 - lambda_min(sym(W)))``,
topology-dependent) is a *weaker* constraint than monotonicity of the operator -- it raises only well
after the field has stopped being monotone, so it is not the guard to rely on.

**``A`` is supplied, not rebuilt.** It is a *constant of the pipeline*: built once by
``build_interaction_matrix(base_graph, epsilon)`` when a dataset is generated, persisted on that dataset's
``base_graph``, and passed back in by everything downstream. Nothing else may construct it -- in particular
the training script must not, because a rebuild is only *probably* the same matrix. Its sparsity pattern is
the deterministic link-adjacency graph, but its values come from an RNG seeded by
``build_asymmetric_interaction_matrix``'s default ``seed=42``, which we neither pass nor record; a dataset
paired with a rebuilt ``A`` therefore depends on that default, and NumPy's stream for it, never changing.
Getting it wrong is silent and severe: the residual at a cached equilibrium under a mismatched ``A`` is
~5.0 rather than ~1e-3.

Two deliberate limits:

- ``A`` is a **fixed property of the operator**, one matrix per dataset -- like ``edge_index``, not like the
  noised BPR attrs. Sampling it per instance is not possible under the current conditioning seam: ``A`` is
  pairwise ``[m, m]`` and the per-edge feature vector cannot express it, so the model would be asked to
  predict a field it cannot see. The natural harder variant is to keep ``W`` fixed, sample ``eps`` per
  instance, and expose it as an extra per-edge feature.
- ``AsymmetricBPRSupply`` does not override ``jacobian_diagonal``, so the supply-diagonal preconditioner
  reaches it through ``SupplyOperator``'s default ``jacobian(c).diag()``, materializing a dense
  ``[m, m]`` per operator evaluation. Numerically correct (``diag(A diag(f')) = A_ii f'_i``) and
  negligible against a 24-destination PUMCM solve at Sioux Falls' 76 links, but it is ``O(m^2)`` where
  ``A.diagonal() * f'(c)`` would be ``O(m)`` -- revisit if the network grows.
"""

from l2s_games.asymmetric_pume_solver import AsymmetricPUMESolver, build_interaction_matrix
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium

__all__ = ["AsymmetricPUMEMarkovTrafficEquilibrium", "build_interaction_matrix"]


class AsymmetricPUMEMarkovTrafficEquilibrium(PUMEMarkovTrafficEquilibrium):
    """``PUMEMarkovTrafficEquilibrium`` with a non-potential supply ``z(c) = A f(c)``.

    ``interaction_matrix`` is required: build it once with ``build_interaction_matrix`` when generating a
    dataset, and pass the *stored* one thereafter (see the module docstring).
    """

    def __init__(self, base_graph, interaction_matrix, **kwargs):
        # Set before super().__init__, which is what calls _make_solver.
        self.interaction_matrix = interaction_matrix
        super().__init__(base_graph, **kwargs)
        # Ride along on the graph so dataset generation persists A inside base_graph.pt, making the dataset
        # carry the operator its equilibria were solved for. Stripped again in `model_input` (_DROPPED_ATTRS)
        # so it never reaches a collated batch -- the model has no use for it, the solver holds the copy the
        # operator actually reads.
        self.base_graph.interaction_matrix = self.solver.interaction_matrix

    def _make_solver(self, solver_kwargs):
        return AsymmetricPUMESolver(self.base_graph, self.interaction_matrix, **solver_kwargs)
