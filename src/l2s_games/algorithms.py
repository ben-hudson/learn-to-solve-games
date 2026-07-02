"""
Each algorithm carries its own state and exposes ``.step(z, v) -> z_next``,
where ``z`` is the current iterate (shape ``(2,)``) and ``v`` is the vector
field. To add your own, copy a class, implement ``.step``, and register it in
``ALGORITHMS``.
"""

import torch


class SimGD:
    """Simultaneous gradient descent (explicit Euler on v)."""

    def __init__(self, h):
        self.h = h

    def step(self, z, v):
        return z + self.h * v(z)


class AltGD:
    """Alternating gradient descent: update theta, then psi with the new theta."""

    def __init__(self, h, n_disc=1):
        self.h, self.n_disc = h, n_disc

    def step(self, z, v):
        th, ps = z
        th = th + self.h * v(torch.stack([th, ps]))[0]  # generator update
        for _ in range(self.n_disc):  # discriminator update(s)
            ps = ps + self.h * v(torch.stack([th, ps]))[1]
        return torch.stack([th, ps])


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


ALGORITHMS = {
    "simgd": lambda h: SimGD(h),
    "altgd": lambda h: AltGD(h, n_disc=1),
    "extragradient": lambda h: ExtraGradient(h),
    "optimistic": lambda h: Optimistic(h),
    "momentum": lambda h: Momentum(h, beta=0.9),
    "consensus": lambda h: Consensus(h, gamma=1.0),
}
