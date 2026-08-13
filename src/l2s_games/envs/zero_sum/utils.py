import torch


def project_onto_simplex(strategies):
    """Exact Euclidean projection of each strategy vector (last dim) onto the probability simplex.

    The sort-and-threshold algorithm (Held et al. 1974; Duchi et al. 2008): shift every coordinate
    down by the level ``tau`` at which the positive part sums to one. ``tau`` is found by sorting:
    the support is the largest ``k`` whose ``k``-th largest coordinate still exceeds the mean excess
    ``(sum of top-k - 1) / k`` of the coordinates above it.
    """
    actions = torch.arange(1, strategies.size(-1) + 1, device=strategies.device)
    sorted_strategies = strategies.sort(dim=-1, descending=True).values
    excess = sorted_strategies.cumsum(dim=-1) - 1
    support_size = (sorted_strategies * actions > excess).sum(dim=-1, keepdim=True)
    tau = excess.gather(-1, support_size - 1) / support_size
    return (strategies - tau).clamp(min=0)


def dist_to_normal_cone(operator, strategies):
    """Euclidean distance from ``operator`` to the normal cone of the probability simplex at
    ``strategies`` (actions on the last dim): the VI stationarity residual, zero exactly where
    ``strategies`` is a fixed point of the ascent dynamics ``z <- project(z + h operator(z))``.

    The cone at ``z`` is ``{u : u_i = lambda on the support, u_i <= lambda off it}``, so the
    squared distance is ``min`` over ``lambda`` of the support's squared deviations from ``lambda``
    plus the off-support upward deviations -- the same sort-and-threshold solve as
    ``project_onto_simplex``: with the coordinates ordered support-first then by operator value,
    the minimizing ``lambda`` is the mean of the largest self-consistent active prefix.
    """
    support = strategies > 0
    order = operator.masked_fill(support, torch.inf).argsort(dim=-1, descending=True)
    sorted_operator = operator.gather(-1, order)

    positions = torch.arange(1, operator.size(-1) + 1, device=operator.device)
    prefix_mean = sorted_operator.cumsum(dim=-1) / positions
    active = (positions <= support.sum(dim=-1, keepdim=True)) | (sorted_operator > prefix_mean)
    multiplier = prefix_mean.gather(-1, active.sum(dim=-1, keepdim=True) - 1)

    deviation = torch.where(support, operator - multiplier, (operator - multiplier).clamp(min=0))
    return deviation.square().sum(dim=-1).sqrt()
