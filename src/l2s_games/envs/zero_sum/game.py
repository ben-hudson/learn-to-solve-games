import pygambit as gambit
import torch

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


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
        edge_index, _ = dense_to_sparse(adjacency)

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
