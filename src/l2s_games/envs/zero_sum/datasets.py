import torch
import tqdm

from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import BaseTransform

from .game import RandomZeroSum, operator


class BuildZeroSumFeats(BaseTransform):
    def __init__(self, mode=None):
        super().__init__()

        assert mode in ["full", "partial"], f"Mode must be 'full' or 'partial', got {mode}."
        self.mode = mode

    def forward(self, data):
        # node 0 is player 1 (payoffs A), node 1 is player 2 (payoffs B)
        payoffs = torch.stack([data.A, data.B]).flatten(start_dim=1)

        if self.mode == "partial":
            n_points_per_instance = data.point.size(0)
            data.payoffs = payoffs.expand(n_points_per_instance, -1, -1)
        else:
            data.payoffs = payoffs

        return data


# TODO: this is simple, but has some disadvantages
# mainly that we can't add elements to the batch after converting it
# the main advantage is we can collate it with default_collate
# we could just switch to collate_as_dense_graphs or to a dict
class GraphToTuple(BaseTransform):
    def forward(self, data):
        return data.to_namedtuple()


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
            getattr(self, attr) is not None for attr in required_attrs
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
        return operator(instance.A, instance.B, points)

    def process(self):
        required_attrs = ["n_points_per_instance"]
        assert all(
            getattr(self, attr) is not None for attr in required_attrs
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
