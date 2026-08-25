import pygambit as gambit
import torch

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def profile_to_tensor(profile, dtype=torch.float32):
    return torch.tensor([[prob for _, prob in strategy] for _, strategy in profile.mixed_strategies()], dtype=dtype)


def tensor_to_profile(game, tensor):
    return game.mixed_strategy_profile(data=tensor.tolist())


def solve(instance):
    profile = gambit.nash.lp_solve(instance.game, rational=False).equilibria[0]
    return profile_to_tensor(profile)


def operator(A, B, points):
    x, y = points.unbind(dim=-2)
    F_x = -(A @ y.unsqueeze(-1)).squeeze(-1)
    F_y = -(B.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)
    return torch.stack([F_x, F_y], dim=-2)


class RandomZeroSum:
    def __init__(self, n_actions=3, gamut_jar="./gamut.jar"):
        self.game = gambit.catalog.generate_gamut(
            "RandomZeroSum",
            params={"actions": [n_actions, n_actions], "normalize": 0},
            gamut_jar=gamut_jar,
        )

    def to_data(self, dtype=torch.float32):
        # player interaction graph is just a complete graph
        n_players = 2
        adjacency = torch.ones(n_players, n_players, dtype=torch.long) - torch.eye(n_players, dtype=torch.long)
        edge_index, _ = dense_to_sparse(adjacency)

        A, B = self.game.to_arrays(dtype=float)
        return Data(
            edge_index=edge_index,
            num_nodes=n_players,
            # messy because to_arrays returns object even though we ask for floats
            A=torch.as_tensor(A.astype(float), dtype=dtype),
            B=torch.as_tensor(B.astype(float), dtype=dtype),
        )

    @classmethod
    def from_data(cls, data: Data):
        instance = cls.__new__(cls)
        instance.game = gambit.Game.from_arrays(data.A.numpy(), data.B.numpy())
        return instance
