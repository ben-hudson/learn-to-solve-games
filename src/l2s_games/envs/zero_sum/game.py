import pygambit as gambit
import torch
from tensordict import tensorclass

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def profile_to_tensor(profile, dtype=torch.float32):
    return torch.tensor([[prob for _, prob in strategy] for _, strategy in profile.mixed_strategies()], dtype=dtype)


def tensor_to_profile(game, tensor):
    return game.mixed_strategy_profile(data=tensor.tolist())


def solve(instance, dtype=torch.float32):
    game = instance.to_gambit()
    profile = gambit.nash.lp_solve(game, rational=False).equilibria[0]
    return profile_to_tensor(profile, dtype=dtype)


def operator(A, B, strategies):
    """Each player's payoff gradient (``Ay``, ``B^T x``): the ascent field. The projected ascent
    dynamics ``z <- project(z + h operator(z))`` are stationary exactly at the equilibria
    ``solve`` finds, which is the sign convention everything downstream assumes
    (``algorithms``, ``dist_to_normal_cone``, the losses).
    """
    x, y = strategies.unbind(dim=-2)
    F_x = (A @ y.unsqueeze(-1)).squeeze(-1)
    F_y = (B.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)
    return torch.stack([F_x, F_y], dim=-2)


@tensorclass
class RandomZeroSum:
    A: torch.Tensor
    B: torch.Tensor
    n_players: int
    n_actions: int

    def to_gambit(self):
        return gambit.Game.from_arrays(self.A.numpy(), self.B.numpy())

    @classmethod
    def sample_normalized(cls, n_actions=3, dtype=torch.float32):
        """
        Sample zero-sum games in the style of nfg_transformer.
        """
        payoff = torch.rand(n_actions, n_actions, dtype=dtype)
        centered = payoff - payoff.mean()
        A = centered / centered.std(correction=0)
        return cls(A, -A, 2, n_actions)

    @classmethod
    def from_gambit(cls, n_actions=3, gamut_jar="./gamut.jar", dtype=torch.float32):
        game = gambit.catalog.generate_gamut(
            "RandomZeroSum",
            params={"actions": [n_actions, n_actions], "normalize": 0},
            gamut_jar=gamut_jar,
        )
        A, B = game.to_arrays(dtype=float)
        return cls(
            torch.as_tensor(A.astype(float), dtype=dtype),
            torch.as_tensor(B.astype(float), dtype=dtype),
            2,
            n_actions,
        )

    def to_data(self, dtype=torch.float32):
        # player interaction graph is just a complete graph
        adjacency = torch.ones(self.n_players, self.n_players, dtype=torch.long) - torch.eye(
            self.n_players, dtype=torch.long
        )
        edge_index, _ = dense_to_sparse(adjacency)

        return Data(
            edge_index=edge_index,
            num_nodes=self.n_players,
            A=self.A.to(dtype),
            B=self.B.to(dtype),
            n_players=self.n_players,
            n_actions=self.n_actions,
        )

    @classmethod
    def from_data(cls, data: Data):
        return cls(
            data.A,
            data.B,
            data.n_players,
            data.n_actions,
        )

    def operator(self, strategies):
        return operator(self.A, self.B, strategies)

    def solve(self):
        return solve(self)
