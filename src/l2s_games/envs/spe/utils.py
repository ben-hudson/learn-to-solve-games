import torch


def dist_to_normal_cone(operator, shipments):
    """Euclidean distance from ``operator`` to the normal cone of the nonnegative orthant at
    ``shipments`` (routes on the last two dims): the VI stationarity residual, zero exactly where
    ``shipments`` is a fixed point of the ascent dynamics ``z <- relu(z + h operator(z))``.

    The cone is separable across routes: ``{u : u_ij <= 0 on unused routes, u_ij = 0 on used
    ones}``, so the nearest cone point is reached by clamping away the operator components that
    point into the feasible directions. Zero exactly at the complementarity conditions of the
    spatial price equilibrium: used routes have balanced prices, unused routes are unprofitable.
    """
    deviation = torch.where(shipments == 0, operator.clamp(min=0), operator)
    return deviation.square().sum(dim=(-2, -1)).sqrt()
