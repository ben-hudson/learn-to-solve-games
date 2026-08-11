import pygambit as gambit
import torch
import torch_geometric

from torch_geometric.data import Data


def profile_to_tensor(profile):
    return torch.tensor(
        [[prob for _, prob in strategy] for _, strategy in profile.mixed_strategies()], dtype=torch.float64
    )


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

    def to_data(self):
        # player interaction graph is just a complete graph
        adjacency = torch.ones(2, 2) - torch.eye(2)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)

        A, B = self.game.to_arrays(dtype=float)
        return Data(
            edge_index=edge_index,
            A=torch.as_tensor(A.astype(float)),
            B=torch.as_tensor(B.astype(float)),
            eq=profile_to_tensor(self.eq),
        )

    @classmethod
    def from_data(cls, data: Data):
        instance = cls.__new__(cls)
        instance.game = gambit.Game.from_arrays(data.A.numpy(), data.B.numpy())
        instance.eq = tensor_to_profile(instance.game, data.eq)
        return instance
