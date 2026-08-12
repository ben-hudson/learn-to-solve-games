import pygambit as gambit
import torch
import torch_geometric

from torch_geometric.data import Data


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


def profile_to_tensor(profile, dtype=torch.float32):
    return torch.tensor([[prob for _, prob in strategy] for _, strategy in profile.mixed_strategies()], dtype=dtype)


def tensor_to_profile(game, tensor):
    return game.mixed_strategy_profile(data=tensor.tolist())


class RandomZeroSum:
    def __init__(self, n_actions=3, solve=False, gamut_jar="./gamut.jar"):
        self.game = gambit.catalog.generate_gamut(
            "RandomZeroSum",
            params={"actions": [n_actions, n_actions], "normalize": 0},
            gamut_jar=gamut_jar,
        )
        if solve:
            self.solve()

    def solve(self):
        self.eq = gambit.nash.lp_solve(self.game, rational=False).equilibria[0]

    def to_data(self, dtype=torch.float32):
        # player interaction graph is just a complete graph
        n_players = 2
        adjacency = torch.ones(n_players, n_players, dtype=torch.long) - torch.eye(n_players, dtype=torch.long)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)

        A, B = self.game.to_arrays(dtype=float)
        return Data(
            edge_index=edge_index,
            num_nodes=n_players,
            # messy because to_arrays returns object even though we ask for floats
            A=torch.as_tensor(A.astype(float), dtype=dtype),
            B=torch.as_tensor(B.astype(float), dtype=dtype),
            eq=profile_to_tensor(self.eq, dtype=dtype),
        )

    @classmethod
    def from_data(cls, data: Data):
        instance = cls.__new__(cls)
        instance.game = gambit.Game.from_arrays(data.A.numpy(), data.B.numpy())
        instance.eq = tensor_to_profile(instance.game, data.eq)
        return instance
