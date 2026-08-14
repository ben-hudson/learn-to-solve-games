import torch


def dist_to_normal_cone(operator, costs, lower, upper):
    """Euclidean distance from ``operator`` to the normal cone of the box ``[lower, upper]`` at
    ``costs`` (coordinates on the last dim): the VI stationarity residual, zero exactly where
    ``costs`` is a fixed point of the ascent dynamics ``z <- clamp(z + h operator(z), lower, upper)``.

    The cone is separable across coordinates: ``{u : u_i <= 0 at the lower bound, u_i >= 0 at the
    upper bound, u_i = 0 in the interior}``, so the nearest cone point is reached by clamping away
    the operator components that point into the feasible directions at each active bound.
    """
    deviation = torch.where(costs == lower, operator.clamp(min=0), operator)
    deviation = torch.where(costs == upper, deviation.clamp(max=0), deviation)
    return deviation.square().sum(dim=-1).sqrt()


def sparse_incidence_matrix(
    edge_index: torch.Tensor,
    n_nodes: int = None,
    n_edges: int = None,
    dtype: torch.dtype = torch.float32,
):
    if n_nodes is None:
        n_nodes = torch.unique(edge_index).size(0)
    if n_edges is None:
        n_edges = edge_index.size(1)

    edge_number = torch.arange(n_edges, device=edge_index.device)

    tails = -torch.ones_like(edge_index[0])
    heads = torch.ones_like(edge_index[1])

    tail_coords = torch.stack((edge_index[0], edge_number))
    head_coords = torch.stack((edge_index[1], edge_number))

    indices = torch.cat((tail_coords, head_coords), dim=-1)
    values = torch.cat((tails, heads), dim=-1).to(dtype)

    return torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_edges))
