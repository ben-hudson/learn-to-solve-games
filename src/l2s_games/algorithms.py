"""
Each algorithm carries its own state and exposes ``.step(z, v) -> z_next``,
where ``z`` is the current iterate (any shape) and ``v`` is the vector field.
To add your own, copy a class, implement ``.step``, and register it in
``ALGORITHMS``.
"""

import torch


def _identity(z):
    return z


class SimpleProjection:
    """Basic projection method for a variational inequality: a forward step along the field,
    then a projection back onto the feasible set -- ``z <- project(z + h v(z))``.

    ``project`` defaults to the identity (unconstrained), where this is exactly plain forward
    Euler on ``v`` -- the "simultaneous gradient descent" step in the two-player framing, with
    no simultaneity to speak of for a single-operator VI. As a VI solver it converges on
    (co)monotone operators like the traffic equilibrium residual, but not on purely rotational
    fields such as RPS. When rolled out through ``simulate``, the feasible-set projection can be
    supplied there instead and ``project`` left at its default."""

    def __init__(self, h, project=None):
        self.h = h
        self.project = project if project is not None else _identity

    def step(self, z, v):
        return self.project(z + self.h * v(z))


class ExtraGradient:
    """Extragradient: a lookahead step, then a step taken from the lookahead
    point. Converges on rotational fields where plain GD diverges."""

    def __init__(self, h):
        self.h = h

    def step(self, z, v):
        z_half = z + self.h * v(z)
        return z + self.h * v(z_half)


class Optimistic:
    """Optimistic gradient descent (extrapolation from the past gradient)."""

    def __init__(self, h):
        self.h, self.prev = h, None

    def step(self, z, v):
        g = v(z)
        if self.prev is None:
            self.prev = g
        z_next = z + 2.0 * self.h * g - self.h * self.prev
        self.prev = g
        return z_next


class Momentum:
    """Heavy-ball momentum."""

    def __init__(self, h, beta=0.9):
        self.h, self.beta, self.m = h, beta, None

    def step(self, z, v):
        g = v(z)
        self.m = g if self.m is None else self.beta * self.m + g
        return z + self.h * self.m


class Consensus:
    """Consensus optimization (Mescheder et al. 2017): follow the modified
    field  v - gamma * J^T v = v - gamma * grad(0.5 * ||v||^2),
    which adds a contractive component and damps the rotation."""

    def __init__(self, h, gamma=1.0):
        self.h, self.gamma = h, gamma

    def step(self, z, v):
        g = v(z)
        # J^T v = grad(0.5 * ||v||^2); computing it as a gradient (not the full Jacobian) is O(n)
        # and shape-agnostic, so it also works on a batched iterate z [B, E] -- the per-instance
        # Jacobians stay decoupled because the batched field has no cross-instance coupling.
        consensus_term = torch.func.grad(lambda x: 0.5 * (v(x) ** 2).sum())(z)
        return z + self.h * (g - self.gamma * consensus_term)


# Each constructor forwards **kwargs to its algorithm class, so callers can override the extra
# hyperparameters (momentum beta, consensus gamma) while the class-level defaults still apply when
# none are passed -- e.g. ALGORITHMS["momentum"](h) keeps beta=0.9, ALGORITHMS["momentum"](h, beta=0.5)
# overrides it.
ALGORITHMS = {
    "projection": lambda h, **kwargs: SimpleProjection(h, **kwargs),
    "extragradient": lambda h, **kwargs: ExtraGradient(h, **kwargs),
    "optimistic": lambda h, **kwargs: Optimistic(h, **kwargs),
    "momentum": lambda h, **kwargs: Momentum(h, **kwargs),
    "consensus": lambda h, **kwargs: Consensus(h, **kwargs),
}
