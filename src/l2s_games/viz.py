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


def plot_trajectory_arrows(ax, traj, true_field, learned_field, lim, n_arrows=40, max_arrow=0.08, title=None):
    """Draw the learned-field rollout as a blue line, with the true and learned operators arrowed along it.

    On a white background over the full domain ``[-lim, lim]^2`` (equilibrium star at the center), the
    rollout ``traj`` is a single blue line and, at ``n_arrows`` points sampled evenly along it, both
    operators are drawn as arrows: the true operator (crimson) and the learned field (dodgerblue, like
    the trajectory). The two share one scale so their lengths are directly comparable, and the arrow
    *length encodes the operator magnitude* (a high quantile of the sampled magnitudes maps to
    ``max_arrow`` data units, so a lone blow-up arrow doesn't shrink the rest into invisibility).
    Showing both matters because a lookahead/momentum algorithm (e.g. extragradient) does not step
    straight along the learned field, so the trajectory tangent is not the learned direction. ``traj``
    is ``[T+1, 2]``.
    """
    traj = torch.as_tensor(traj, dtype=torch.float32)
    ax.plot(traj[:, 0].numpy(), traj[:, 1].numpy(), "-", color="dodgerblue", lw=1.4, zorder=2, label="trajectory")
    ax.plot(traj[0, 0].item(), traj[0, 1].item(), "o", color="dodgerblue", ms=6, zorder=3)

    idx = torch.linspace(0, len(traj) - 1, min(n_arrows, len(traj))).round().long()
    pts = traj[idx]
    with torch.no_grad():
        true_v = true_field(pts)
        learned_v = learned_field(pts)
    # One shared scale for both fields so lengths are comparable; a high quantile (not the max) sets it,
    # so a lone blow-up arrow -- common early when the learned field is untrained -- doesn't shrink all
    # the rest into invisibility.
    mags = torch.cat([torch.linalg.norm(true_v, dim=-1), torch.linalg.norm(learned_v, dim=-1)])
    ref = torch.quantile(mags, 0.95).clamp(min=1e-12)
    scale = float(ref / max_arrow)  # scale_units="xy": arrow length in data units = |vector| / scale
    x, y = pts[:, 0].numpy(), pts[:, 1].numpy()
    # Enlarged arrowheads (headwidth/headlength are in shaft-width units) so the two fields' directions
    # stay readable even when the arrows are short.
    quiver_kw = dict(
        angles="xy", scale_units="xy", scale=scale, width=0.005,
        headwidth=6, headlength=8, headaxislength=7, zorder=3,
    )
    ax.quiver(x, y, true_v[:, 0].numpy(), true_v[:, 1].numpy(), color="crimson", label="true field", **quiver_kw)
    ax.quiver(x, y, learned_v[:, 0].numpy(), learned_v[:, 1].numpy(), color="dodgerblue", label="learned field", **quiver_kw)

    ax.plot(0, 0, "k*", ms=11, zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\psi$")
    if title is not None:
        ax.set_title(title, fontsize=11)
