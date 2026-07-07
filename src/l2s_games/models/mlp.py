"""MLP field model for flat (vector-domain) games: ``[point | params] -> field value``."""

import torch
from torchvision.ops import MLP

from .base import FieldModel


class MLPFieldModel(FieldModel):
    """Tanh MLP over the flat ``[point | params]`` feature vector."""

    def __init__(self, in_features, hidden, out_features, lr, flatten_start_dim=-1, **kwargs):
        super().__init__(lr, **kwargs)
        self.flatten_start_dim = flatten_start_dim
        self.mlp = MLP(
            in_channels=in_features,
            hidden_channels=[*hidden, out_features],
            activation_layer=torch.nn.Tanh,
        )

    def forward(self, inputs):
        # ``flatten_start_dim`` selects how much of the feats tensor feeds one MLP call. The default
        # ``-1`` is a no-op (last axis only): flat games apply the MLP pointwise over the feature axis,
        # so arbitrary leading axes -- a batch, or a grid of eval points -- pass through unchanged.
        # Setting it to ``1`` collapses one fixed graph's per-node feats ``[B, N, k]`` into a single
        # vector ``[B, N*k]`` -- a whole-graph MLP that predicts every node at once (``out_features =
        # N``), with no graph inductive bias.
        return self.mlp(inputs["feats"].flatten(start_dim=self.flatten_start_dim))
