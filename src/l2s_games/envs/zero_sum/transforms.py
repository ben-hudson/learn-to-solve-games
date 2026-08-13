import torch

from torch_geometric.transforms import BaseTransform


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


class GraphToTuple(BaseTransform):
    def forward(self, data):
        return data.to_namedtuple()
