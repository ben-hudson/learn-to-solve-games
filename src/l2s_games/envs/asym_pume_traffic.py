"""Non-potential (asymmetric) PUME Markov traffic equilibrium as a variational inequality family.

Identical to ``PUMEMarkovTrafficEquilibrium`` (see ``pume_traffic.py``) in every respect -- same road-graph
instances, same cost-space domain, same excess-supply residual ``E(c) = z(c) - x(c)``, same conditioning
seam, same preconditioning -- except that the supply is **non-separable**:

    z(c) = A f(c) + B c

with ``f`` the element-wise inverse BPR, ``A = (1 - eps) I + eps W`` a *multiplicative* coupling (``W``
row-stochastic on the link-adjacency graph, so link ``a``'s supply is a convex combination of its own flow
and its neighbours'), and ``B = kappa S`` an *additive* one (``S`` antisymmetric). ``A = I`` and ``B = 0``
recover the separable operator exactly. Everything is inherited through the ``_make_solver`` hook; only the
supply operator differs (see ``asymmetric_pume_solver``, which explains why the two couplings behave very
differently and when to reach for each).

**Why this family exists.** ``grad z(c) = A diag(f'(c)) + B`` is non-symmetric whenever either coupling is
present, so ``E(c)`` is not the gradient of any potential and the VI is genuinely a *game* rather than a
convex minimization in disguise. The repo's other non-potential fields (``toy``'s rotational well, ``rps``)
are flat 2-D families on the MLP path; this is the first one at traffic scale on the graph path.

**Which coupling to use.** ``eps`` is capped by monotonicity (see below) at a few percent relative Jacobian
asymmetry, and at those levels the rollout dynamics are indistinguishable from the potential case --
measured: ``projection`` beats ``optimistic`` and ``extragradient`` at every step size, with and without
preconditioning. ``kappa`` has no such cap: additive antisymmetric coupling is monotone unconditionally
(``<B dc, dc> = 0`` exactly), and because ``B`` has a zero diagonal it also survives the ``supply_diagonal``
preconditioner untouched. So ``kappa`` is the knob for a genuinely rotation-dominated field and ``eps`` is
the one that stays physically interpretable as neighbour spillover.

**``eps`` monotonicity is conditional, not guaranteed -- check it for the value you use.**
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
- ``optimistic``'s and ``extragradient``'s convergence guarantees are for monotone Lipschitz operators, so
  they are on firm ground only in the same regime. They are nevertheless *measured worse* than
  ``projection`` here (see ``algorithms.py``); the guarantee is not the binding consideration.

The additive coupling ``B`` has no such condition -- ``<B dc, dc> = 0`` for antisymmetric ``B`` makes its
contribution to monotonicity exactly zero, at any ``kappa``.

Separately, the *preconditioned* field ``M^{-1} E`` is not monotone at all -- ``M`` is state-dependent, and
``lambda_min sym(grad)`` is negative for the symmetric family too. That is pre-existing and is why
``--constraint_space raw`` is the default.

``eps = 0, kappa = 0`` reproduces ``pume_traffic`` exactly, which is the cheapest regression test of the
whole path.

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

**Choosing kappa.** It scales a unit-spectral-norm ``S``, so ``kappa`` is directly the rotational magnitude
and is comparable against ``max f' ~ 46`` on Sioux Falls: rotation only *dominates* the field once
``kappa`` is of that order or larger. Large ``kappa`` can drive individual supply components negative
(``f(c) >= 0``, so this is the rotation term overpowering BPR flow); PUME tolerates it and the root still has
``z = x >= 0``, but the operator stops being interpretable as a traffic supply -- ``scripts/
probe_rotational_supply.py`` reports where that starts.

**Both matrices are supplied, not rebuilt.** They are *constants of the pipeline*: built once by
``build_interaction_matrix(base_graph, epsilon)`` / ``build_rotation_matrix(base_graph, kappa)`` when a
dataset is generated, persisted on that dataset's ``base_graph``, and read back off the stored graph by
everything downstream (``__init__`` does this itself when the kwargs are omitted). Nothing else may
construct them -- in particular the training script must not.

For ``A`` that is a correctness requirement: its sparsity pattern is the deterministic link-adjacency graph,
but its values come from an RNG seeded by ``build_asymmetric_interaction_matrix``'s default ``seed=42``,
which we neither pass nor record, so a rebuild is only *probably* the same matrix. ``B`` is fully
deterministic from the topology and could safely be rebuilt -- it follows the same rule anyway, because one
policy ("the dataset carries the operator, in full, or it is not usable") is easier to hold in your head than
two, and because it makes swapping in a random ``S`` later a one-line change rather than a dataset migration.
Getting it wrong is silent and severe: the residual at a cached equilibrium under a mismatched ``A`` is
~5.0 rather than ~1e-3.

One deliberate limit: both couplings are a **fixed property of the operator**, one matrix per dataset -- like
``edge_index``, not like the noised BPR attrs. Sampling per instance is not possible under the current
conditioning seam, since a coupling is pairwise ``[m, m]`` and the per-edge feature vector cannot express it,
so the model would be asked to predict a field it cannot see. The natural harder variant is to keep the
pattern fixed, sample ``eps``/``kappa`` per instance, and expose the scalar as an extra per-edge feature.
"""

from l2s_games.asymmetric_pume_solver import (
    AsymmetricPUMESolver,
    build_interaction_matrix,
    build_rotation_matrix,
)
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium

__all__ = [
    "AsymmetricPUMEMarkovTrafficEquilibrium",
    "build_interaction_matrix",
    "build_rotation_matrix",
    "coupling_matrices",
]

COUPLING_MATRICES = ("interaction_matrix", "rotation_matrix")


def coupling_matrices(base_graph):
    """The couplings persisted on a base graph, as constructor kwargs for this family.

    The read side of the "dataset carries its operator" rule: generation stores both matrices on
    ``base_graph``, and every consumer passes them back rather than rebuilding (see the module docstring for
    why, and why the rule covers the deterministic matrix too). A graph predating either has to be
    regenerated, so the absence is an error rather than a default.
    """
    missing = [name for name in COUPLING_MATRICES if name not in base_graph]
    assert not missing, (
        f"this graph is missing the coupling matrices its equilibria were solved with: {', '.join(missing)}. "
        "Regenerate the dataset:\n"
        "  python scripts/generate_traffic_dataset.py <n> <root> --game asym_pume_traffic "
        "--epsilon <eps> --kappa <kappa>"
    )
    return {name: base_graph[name] for name in COUPLING_MATRICES}


class AsymmetricPUMEMarkovTrafficEquilibrium(PUMEMarkovTrafficEquilibrium):
    """``PUMEMarkovTrafficEquilibrium`` with a non-potential supply ``z(c) = A f(c) + B c``.

    Both matrices are built once, with ``build_interaction_matrix`` / ``build_rotation_matrix``, when a
    dataset is generated -- that caller passes them in. Every other consumer omits them and gets the
    *stored* ones back off ``base_graph``, where generation persisted them, so a dataset root is
    self-describing and nothing downstream can rebuild a mismatched coupling (see the module docstring).
    """

    def __init__(self, base_graph, interaction_matrix=None, rotation_matrix=None, **kwargs):
        if interaction_matrix is None:  # a reader: the root's graph carries the stored couplings
            stored = coupling_matrices(base_graph)
            interaction_matrix, rotation_matrix = (stored[name] for name in COUPLING_MATRICES)
        # Set before super().__init__, which is what calls _make_solver.
        self.interaction_matrix = interaction_matrix
        self.rotation_matrix = rotation_matrix
        super().__init__(base_graph, **kwargs)
        # Ride along on the graph so dataset generation persists both couplings inside base_graph.pt, making
        # the dataset carry the operator its equilibria were solved for. Stripped again in `model_input`
        # (_DROPPED_ATTRS) so they never reach a collated batch -- the model has no use for them, the solver
        # holds the copies the operator actually reads.
        self.base_graph.interaction_matrix = self.solver.interaction_matrix
        self.base_graph.rotation_matrix = self.solver.rotation_matrix

    def _make_solver(self, solver_kwargs):
        return AsymmetricPUMESolver(
            self.base_graph, self.interaction_matrix, self.rotation_matrix, **solver_kwargs
        )
