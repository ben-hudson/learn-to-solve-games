"""
Game-optimization dynamics as a small class hierarchy.

Each algorithm is an ``Algorithm`` subclass carrying its own state and step size ``h``, and exposing
``step(z, v) -> z_next``, where ``z`` is the current iterate (any shape) and ``v`` is the vector
field. Every constructor takes ``h`` first (subclasses add their own extra hyperparameters), so the
``ALGORITHMS`` registry can build any of them uniformly as ``ALGORITHMS[name](h, **kwargs)``. To add
your own, subclass ``Algorithm``, implement ``step``, and register the class in ``ALGORITHMS``.
"""

from abc import ABC, abstractmethod

import torch


def _identity(z):
    return z


class Algorithm(ABC):
    """Base game-dynamics update: holds the step size ``h`` and exposes ``step(z, v) -> z_next``.

    The shared contract behind the ``ALGORITHMS`` registry: ``h`` is always the first constructor
    argument, so every algorithm builds uniformly as ``ALGORITHMS[name](h, **kwargs)``.
    """

    def __init__(self, h):
        self.h = h

    @abstractmethod
    def step(self, z, v):
        """One update from iterate ``z`` under field ``v``, returning the next iterate."""


class SimpleProjection(Algorithm):
    """Basic projection method for a variational inequality: a forward step along the field,
    then a projection back onto the feasible set -- ``z <- project(z + h v(z))``.

    ``project`` defaults to the identity (unconstrained), where this is exactly plain forward
    Euler on ``v`` -- the "simultaneous gradient descent" step in the two-player framing, with
    no simultaneity to speak of for a single-operator VI. As a VI solver it converges on
    (co)monotone operators like the traffic equilibrium residual, but not on purely rotational
    fields such as RPS. When rolled out through ``simulate``, the feasible-set projection can be
    supplied there instead and ``project`` left at its default."""

    def __init__(self, h, project=None):
        super().__init__(h)
        self.project = project if project is not None else _identity

    def step(self, z, v):
        return self.project(z + self.h * v(z))


class ExtraGradient(Algorithm):
    """Extragradient: a lookahead step, then a step taken from the lookahead
    point. Converges on rotational fields where plain GD diverges."""

    def step(self, z, v):
        z_half = z + self.h * v(z)
        return z + self.h * v(z_half)


class Optimistic(Algorithm):
    """Optimistic gradient descent (extrapolation from the past gradient)."""

    def __init__(self, h):
        super().__init__(h)
        self.prev = None

    def step(self, z, v):
        g = v(z)
        if self.prev is None:
            self.prev = g
        z_next = z + 2.0 * self.h * g - self.h * self.prev
        self.prev = g
        return z_next


class Momentum(Algorithm):
    """Heavy-ball momentum."""

    def __init__(self, h, beta=0.9):
        super().__init__(h)
        self.beta, self.m = beta, None

    def step(self, z, v):
        g = v(z)
        self.m = g if self.m is None else self.beta * self.m + g
        return z + self.h * self.m


class Consensus(Algorithm):
    """Consensus optimization (Mescheder et al. 2017): follow the modified
    field  v - gamma * J^T v = v - gamma * grad(0.5 * ||v||^2),
    which adds a contractive component and damps the rotation."""

    def __init__(self, h, gamma=1.0):
        super().__init__(h)
        self.gamma = gamma

    def step(self, z, v):
        g = v(z)
        # J^T v = grad(0.5 * ||v||^2); computing it as a gradient (not the full Jacobian) is O(n)
        # and shape-agnostic, so it also works on a batched iterate z [B, E] -- the per-instance
        # Jacobians stay decoupled because the batched field has no cross-instance coupling.
        consensus_term = torch.func.grad(lambda x: 0.5 * (v(x) ** 2).sum())(z)
        return z + self.h * (g - self.gamma * consensus_term)


# Names map straight to the classes: every constructor takes ``h`` first plus optional extra
# hyperparameters, so ``ALGORITHMS[name](h)`` uses the class-level defaults (beta=0.9, gamma=1.0)
# and ``ALGORITHMS[name](h, beta=0.5)`` overrides them.
ALGORITHMS = {
    "projection": SimpleProjection,
    "extragradient": ExtraGradient,
    "optimistic": Optimistic,
    "momentum": Momentum,
    "consensus": Consensus,
}
