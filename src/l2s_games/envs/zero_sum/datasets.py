import torch
import tqdm

from torch_geometric.data import InMemoryDataset

from .game import RandomZeroSum


class RandomZeroSumEquilibriumDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        n_instances=None,
        n_actions=None,
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
        return ["instances.pt"]

    def download(self):
        required_attrs = ["n_instances", "n_actions"]
        assert all(
            getattr(self, attr) for attr in required_attrs
        ), f"No cache at {self.raw_paths[0]}. Pass {required_attrs} to rebuild it."

        progress = range(self.n_instances) if self.quiet else tqdm.trange(self.n_instances)

        data_list = []
        for i in progress:
            instance = RandomZeroSum(n_actions=self.n_actions, solve=True, gamut_jar=self.gamut_jar)
            data_list.append(instance.to_data())

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
    def eval_operator(cls, instance, points):
        # batch-agnostic: A/B may be one game's [n, n] (points [P, 2, n]) or a collated
        # batch's [B, n, n] (points [B, 2, n])
        x, y = points.unbind(dim=-2)
        F_x = -(instance.A @ y.unsqueeze(-1)).squeeze(-1)
        F_y = -(instance.B.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)
        return torch.stack([F_x, F_y], dim=-2)

    def process(self):
        required_attrs = ["n_points_per_instance"]
        assert all(
            getattr(self, attr) for attr in required_attrs
        ), f"No cache at {self.processed_paths[0]}. Pass {required_attrs} to rebuild it."

        instances = self.load_instances()
        n_actions = instances[0].A.size(0)
        player_simplex = torch.distributions.Dirichlet(torch.ones(n_actions))

        progress = instances if self.quiet else tqdm.tqdm(instances)

        data_list = []
        for instance in progress:
            instance.point = player_simplex.sample((self.n_points_per_instance, 2))  # 2 players
            instance.operator = self.eval_operator(instance, instance.point)
            data_list.append(instance)

        self.save(data_list, self.processed_paths[0])
