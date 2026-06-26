"""
rotational_field_sandbox.py

A sandbox for experimenting with optimization / game dynamics on parametric
2D rotational vector fields -- the kind that show up in the Dirac-GAN analysis
of Mescheder et al. (2018), "Which Training Methods for GANs do actually
Converge?".

State is z = (theta, psi). The "game" is a vector field v(z); following v with
simultaneous gradient descent is vanilla GAN training. Near a Nash equilibrium
the field is dominated by a rotational (antisymmetric) component -- that is what
makes naive GD spiral outward instead of converging.

The field lives in ``l2s_games.envs.toy`` and the algorithms in
``l2s_games.algorithms``. Edit the CONFIG block and run:

    python scripts/rotational_field_sandbox.py
"""

import matplotlib.pyplot as plt
import torch

from l2s_games.algorithms import ALGORITHMS, jacobian
from l2s_games.envs.toy import rotational_field

# --------------------------------------------------------------------------
# CONFIG -- edit me
# --------------------------------------------------------------------------
OMEGA = 0.1  # rotation strength (antisymmetric part) -> imaginary eigenvalues
# The symmetric ("potential") part is an oriented, possibly anisotropic well.
# DAMP_FLOOR is the contraction ALONG the trough, DAMP_WALL the contraction
# PERPENDICULAR to it (the steep walls). WELL_ANGLE orients the trough.
#   DAMP_FLOOR == DAMP_WALL                  -> round bowl (isotropic spiral)
#   DAMP_FLOOR == 0, DAMP_WALL == g, angle 0 -> paper's gradient penalty (damp psi only)
#   DAMP_FLOOR == 0, DAMP_WALL  > 0, angle=-45 -> trough top-left to bottom-right
DAMP_FLOOR = 0.0  # contraction along the trough  (0 = flat valley floor)
DAMP_WALL = 0.0  # contraction across the trough (0 together with FLOOR = pure rotation)
WELL_ANGLE = -45.0  # orientation of the trough floor, in degrees
CURL_NONLIN = 0.0  # Dirac-GAN-style curvature: rotation speed grows with theta*psi

H = 0.1  # learning rate / step size
N_STEPS = 400  # iterations per trajectory
Z0 = (1.0, 1.0)  # starting iterate (red dot)

# which algorithms to compare (keys into ALGORITHMS)
COMPARE = ["simgd", "altgd", "extragradient", "optimistic", "momentum", "consensus"]


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate(field, algo, z0, n_steps):
    z = torch.as_tensor(z0, dtype=torch.float32)
    traj = [z.clone()]
    for _ in range(n_steps):
        z = torch.as_tensor(algo.step(z, field), dtype=torch.float32)
        if not torch.isfinite(z).all():  # blew up
            break
        z = torch.clamp(z, -1e6, 1e6)
        traj.append(z.clone())
    return torch.stack(traj)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_phase(ax, field, traj, title, lim=2.0, grid=21):
    xs = torch.linspace(-lim, lim, grid)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    V = field(torch.stack([X, Y], dim=-1))
    U, W = V[..., 0], V[..., 1]
    mag = torch.hypot(U, W) + 1e-9
    ax.quiver(
        X, Y, U / mag, W / mag, mag, cmap="viridis", alpha=0.6, scale=30, width=0.004, pivot="mid"
    )
    traj = traj.numpy()
    ax.plot(traj[:, 0], traj[:, 1], "-", color="crimson", lw=1.2, alpha=0.9)
    ax.plot(traj[0, 0], traj[0, 1], "o", color="crimson", ms=6)
    ax.plot(0, 0, "k*", ms=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\psi$")


def main():
    field = rotational_field(OMEGA, DAMP_FLOOR, DAMP_WALL, WELL_ANGLE, CURL_NONLIN)

    eig = torch.linalg.eigvals(jacobian(field, torch.zeros(2)))
    eig = torch.round(eig.real, decimals=4) + 1j * torch.round(eig.imag, decimals=4)
    print("Field Jacobian eigenvalues at origin:", eig)

    names = COMPARE
    trajs = {name: simulate(field, ALGORITHMS[name](H), Z0, N_STEPS) for name in names}

    for name in names:
        final = torch.linalg.norm(trajs[name][-1])
        print(f"  {name:14s} final ||z|| = {final:.4e}  ({len(trajs[name])} steps)")

    # ---- phase portraits, one per algorithm ----
    cols = 3
    rows = -(-len(names) // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        plot_phase(ax, field, trajs[name], name)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(
        rf"Field: $\omega$={OMEGA}, floor={DAMP_FLOOR}, wall={DAMP_WALL}, "
        rf"angle={WELL_ANGLE}$^\circ$, nonlin={CURL_NONLIN},  h={H}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("phase_portraits.png", dpi=130)

    # ---- convergence: distance to equilibrium vs iteration ----
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for name in names:
        d = torch.linalg.norm(trajs[name], dim=1)
        ax2.semilogy(d.numpy(), label=name, lw=1.6)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel(r"$\|z_k\|$  (distance to equilibrium)")
    ax2.set_title("Convergence")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("convergence.png", dpi=130)

    plt.show()


if __name__ == "__main__":
    main()
