"""Roll out an optimization algorithm on a vector field, and score where it ended up."""

import torch


def _identity(z):
    return z


def simulate(field, algo, z0, n_steps, project=_identity):
    """Iterate ``algo`` on ``field`` from ``z0``, returning the trajectory.

    ``project`` maps points onto the feasible set (default: unconstrained). It is threaded *into*
    ``algo.step``, so lookahead methods project their intermediate points as their textbook constrained
    forms require, and applied again to each returned iterate -- idempotent for a box, and the guard for
    an algorithm that projects nothing.
    """
    z = project(torch.as_tensor(z0, dtype=torch.float32))
    traj = [z.clone()]
    for _ in range(n_steps):
        z = torch.as_tensor(algo.step(z, field, project), dtype=torch.float32).detach()
        if not torch.isfinite(z).all():  # blew up
            break
        z = project(torch.clamp(z, -1e6, 1e6))
        traj.append(z.clone())
    return torch.stack(traj)


def natural_map(family, params, z):
    """The VI natural map ``z - project(z - operator(z))`` at ``z``.

    The convergence measure for a *constrained* VI: its norm is zero exactly at a solution, whereas
    ``||operator(z)||`` stays positive at a solution where the feasible set is active. Reduces to the
    operator itself when the domain is unconstrained.
    """
    return z - family.project(params, z - family.operator(params, z))
