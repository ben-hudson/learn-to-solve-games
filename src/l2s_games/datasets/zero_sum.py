import torch
import tqdm

from l2s_games.envs.zero_sum import RandomZeroSum
from torch_geometric.data import Data, InMemoryDataset


class RandomZeroSumEquilibriumDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        n_instances=None,
        n_actions=3,
        gamut_jar="./gamut.jar",
        quiet=True,
        **kwargs,
    ):
        self.n_instances = n_instances
        self.n_actions = n_actions
        self.gamut_jar = gamut_jar
        self.quiet = quiet

        super().__init__(root, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["instances.pt"]

    @property
    def processed_file_names(self):
        return ["equilibria.pt"]

    def download(self):
        required_attrs = ["n_instances", "n_actions", "gamut_jar"]
        assert all(
            getattr(self, attr) for attr in required_attrs
        ), f"No cache at {self.raw_paths[0]}. Pass {required_attrs} to rebuild it."

        progress = range(self.n_instances) if self.quiet else tqdm.trange(self.n_instances)

        data_list = []
        for i in progress:
            instance = RandomZeroSum(n_actions=self.n_actions, solve=True, gamut_jar=self.gamut_jar)
            data = instance.to_data().apply(lambda tensor: tensor.to(torch.float32))
            data_list.append(data)

        torch.save(data_list, self.raw_paths[0])

    def load_instances(self):
        return torch.load(self.raw_paths[0], weights_only=False)

    def process(self):
        # just copy the instances since they're already solved
        self.save(self.load_instances(), self.processed_paths[0])


class RandomZeroSumOperatorDataset(RandomZeroSumEquilibriumDataset):
    def __init__(self, root, n_points_per_instance=None, **kwargs):
        self.n_points_per_instance = n_points_per_instance
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return ["operators.pt"]

    @classmethod
    def eval_operator(cls, instance: Data, points: torch.Tensor):
        x, y = points.unbind(dim=-2)
        return torch.stack([-y @ instance.A.T, -x @ instance.B], dim=-2)

    def process(self):
        required_attrs = ["n_actions", "n_points_per_instance"]
        assert all(
            getattr(self, attr) for attr in required_attrs
        ), f"No cache at {self.processed_paths[0]}. Pass {required_attrs} to rebuild it."

        progress = iter(self.load_instances()) if self.quiet else tqdm.tqdm(self.load_instances())
        player_simplex = torch.distributions.Dirichlet(torch.ones(self.n_actions))

        data_list = []
        for instance in progress:
            points = player_simplex.sample((self.n_points_per_instance, 2))  # 2 players
            operators = self.eval_operator(instance, points)
            for point, operator in zip(points, operators):
                data_list.append(Data(A=instance.A, B=instance.B, point=point, operator=operator))

        self.save(data_list, self.processed_paths[0])
