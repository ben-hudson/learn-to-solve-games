"""Roll out an optimization algorithm on a vector field."""

import torch


def simulate(field, algo, z0, n_steps):
    """Iterate ``algo`` on ``field`` from ``z0``, returning the trajectory."""
    z = torch.as_tensor(z0, dtype=torch.float32)
    traj = [z.clone()]
    for _ in range(n_steps):
        z = torch.as_tensor(algo.step(z, field), dtype=torch.float32).detach()
        if not torch.isfinite(z).all():  # blew up
            break
        z = torch.clamp(z, -1e6, 1e6)
        traj.append(z.clone())
    return torch.stack(traj)
