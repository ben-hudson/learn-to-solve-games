"""Roll out an optimization algorithm on a vector field."""

import torch


def _identity(z):
    return z


def simulate(field, algo, z0, n_steps, project=_identity):
    """Iterate ``algo`` on ``field`` from ``z0``, returning the trajectory.

    ``project`` maps each iterate back onto the feasible set after the step (default:
    unconstrained), so a constrained VI rolls out with the existing algorithms unchanged.
    """
    z = project(torch.as_tensor(z0, dtype=torch.float32))
    traj = [z.clone()]
    for _ in range(n_steps):
        z = torch.as_tensor(algo.step(z, field), dtype=torch.float32).detach()
        if not torch.isfinite(z).all():  # blew up
            break
        z = project(torch.clamp(z, -1e6, 1e6))
        traj.append(z.clone())
    return torch.stack(traj)
