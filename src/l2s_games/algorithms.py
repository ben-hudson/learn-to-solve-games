"""
Game-optimization dynamics as a small class hierarchy.

Each algorithm is an ``Algorithm`` subclass carrying its own state, constructed from a step size, the
vector field ``operator_fn``, and a projection ``project_fn`` onto the feasible set, and exposing
``step(z) -> z_next``, where ``z`` is the current iterate (any shape). Every constructor takes
``step_size`` first (subclasses add their own extra hyperparameters), so the ``ALGORITHMS`` registry
can build any of them uniformly as ``ALGORITHMS[name](step_size, operator_fn, project_fn, **kwargs)``.
To add your own, subclass ``Algorithm``, implement ``step``, and register the class in ``ALGORITHMS``.

Methods that evaluate the field at an **intermediate** point must project that point too -- the
textbook constrained forms do (Korpelevich for extragradient, Popov for optimistic), and it is what
keeps every state an algorithm queries feasible, which the expert data stream relies on (see
``rollout_sampling``).
"""

from abc import ABC, abstractmethod

import torch


class Algorithm(ABC):
    """Base game-dynamics update: holds the step size, vector field ``operator_fn``, and projection
    ``project_fn``, and exposes ``step(z) -> z_next``.

    The shared contract behind the ``ALGORITHMS`` registry: ``step_size`` is always the first
    constructor argument, so every algorithm builds uniformly as
    ``ALGORITHMS[name](step_size, operator_fn, project_fn, **kwargs)``.
    """

    def __init__(self, step_size, operator_fn, project_fn):
        self.step_size = step_size
        self.operator_fn = operator_fn
        self.project_fn = project_fn

    @abstractmethod
    def step(self, z):
        """One update from iterate ``z`` under the field ``operator_fn``, projected onto the
        feasible set by ``project_fn``, returning the next iterate."""


class SimpleProjection(Algorithm):
    """Basic projection method for a variational inequality: a forward step along the field,
    then a projection back onto the feasible set -- ``z <- project(z + h v(z))``.

    Under an identity ``project`` (unconstrained) this is exactly plain forward Euler on ``v`` -- the
    "simultaneous gradient descent" step in the two-player framing, with no simultaneity to speak of
    for a single-operator VI. As a VI solver it converges on *strongly* monotone or cocoercive
    operators, but it has no guarantee on a merely monotone one, and it fails outright on purely
    rotational fields such as RPS. It queries the field only at the (feasible) current iterate, so it
    is the one method for which projecting the intermediate point is vacuous."""

    def step(self, z, v, project):
        return project(z + self.h * v(z))


class ExtraGradient(Algorithm):
    """Projected extragradient (Korpelevich 1976): a lookahead step, then a step from the current
    iterate along the field *at* the lookahead point.

    Both stages project, which is the textbook constrained form and not merely a safety net: the
    lookahead point is where the field is evaluated, so leaving it unprojected queries the operator
    outside the feasible set. Converges on monotone Lipschitz operators for ``h < 1/L`` -- including
    the rotational fields where plain projection stalls or spirals -- at the cost of **two** field
    evaluations per step."""

    def step(self, z, v, project):
        z_half = project(z + self.h * v(z))
        return project(z + self.h * v(z_half))


class Optimistic(Algorithm):
    """Optimistic / past-extragradient method (Popov 1980), the constrained form of optimistic
    gradient descent.

    Same two-projection shape as ``ExtraGradient``, but the lookahead reuses the field value cached at
    the *previous* extrapolated point, so it costs **one** field evaluation per step instead of two --
    which is what makes it the cheaper choice when field evaluations are the budget (the traffic
    operator's ground-truth evaluations, see ``operator_count``).

    Note this is Popov's iteration, not the unconstrained OGDA algebraic form ``z + 2h g - h g_prev``:
    that form has no feasible-set generalization, which is why the constrained literature uses this
    one. The two are closely related and both converge on monotone fields, but they are distinct
    iterations, so unconstrained results are not numerically comparable across the two."""

    def __init__(self, step_size, operator_fn, project_fn):
        super().__init__(step_size, operator_fn, project_fn)
        self.g_bar = None  # field value at the previous extrapolated point

    def step(self, z):
        if self.g_bar is None:  # first step: no past extrapolation to lean on
            self.g_bar = self.operator_fn(z)
        z_bar = self.project_fn(z + self.step_size * self.g_bar)  # free: g_bar was evaluated last step
        self.g_bar = self.operator_fn(z_bar)  # the one field evaluation per step
        return self.project_fn(z + self.step_size * self.g_bar)


class Momentum(Algorithm):
    """Heavy-ball momentum."""

    def __init__(self, h, beta=0.9):
        super().__init__(h)
        self.beta, self.m = beta, None

    def step(self, z, v, project):
        g = v(z)
        self.m = g if self.m is None else self.beta * self.m + g
        return project(z + self.h * self.m)


class Consensus(Algorithm):
    """Consensus optimization (Mescheder et al. 2017): follow the modified
    field  v - gamma * J^T v = v - gamma * grad(0.5 * ||v||^2),
    which adds a contractive component and damps the rotation."""

    def __init__(self, h, gamma=1.0):
        super().__init__(h)
        self.gamma = gamma

    def step(self, z, v, project):
        g = v(z)
        # J^T v = grad(0.5 * ||v||^2); computing it as a gradient (not the full Jacobian) is O(n)
        # and shape-agnostic, so it also works on a batched iterate z [B, E] -- the per-instance
        # Jacobians stay decoupled because the batched field has no cross-instance coupling.
        consensus_term = torch.func.grad(lambda x: 0.5 * (v(x) ** 2).sum())(z)
        return project(z + self.h * (g - self.gamma * consensus_term))


# Names map straight to the classes: every constructor takes ``step_size``, ``operator_fn``, and
# ``project_fn`` first plus optional extra hyperparameters, so
# ``ALGORITHMS[name](step_size, operator_fn, project_fn)`` uses the class-level defaults (beta=0.9,
# gamma=1.0) and passing ``beta=0.5`` overrides them.
ALGORITHMS = {
    "projection": SimpleProjection,
    "extragradient": ExtraGradient,
    "optimistic": Optimistic,
    "momentum": Momentum,
    "consensus": Consensus,
}
