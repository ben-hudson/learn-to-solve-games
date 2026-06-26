"""Shared phase-portrait plotting helpers."""

import torch


def plot_field_quiver(ax, field, lim=2.0, grid=21, title=None):
    """Draw a normalized, magnitude-colored quiver of ``field`` on ``ax``."""
    xs = torch.linspace(-lim, lim, grid)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    with torch.no_grad():
        V = field(torch.stack([X, Y], dim=-1))
    U, W = V[..., 0], V[..., 1]
    mag = torch.hypot(U, W) + 1e-9
    ax.quiver(
        X, Y, U / mag, W / mag, mag, cmap="viridis", alpha=0.6, scale=30, width=0.004, pivot="mid"
    )
    ax.plot(0, 0, "k*", ms=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\psi$")
    if title is not None:
        ax.set_title(title, fontsize=11)


def overlay_trajectory(ax, traj, color="crimson", label=None):
    """Plot a trajectory line and its starting point on ``ax``."""
    traj = traj.numpy()
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=1.2, alpha=0.9, label=label)
    ax.plot(traj[0, 0], traj[0, 1], "o", color=color, ms=6)
