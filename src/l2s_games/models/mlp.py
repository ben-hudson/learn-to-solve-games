"""MLP field model for flat (vector-domain) games: ``[point | params] -> field value``."""

import torch
from torchvision.ops import MLP

from .base import FieldModel


class MLPFieldModel(FieldModel):
    """Tanh MLP over the flat ``[point | params]`` feature vector."""

    def __init__(self, in_features, hidden, out_features, lr, **kwargs):
        super().__init__(lr, **kwargs)
        self.mlp = MLP(
            in_channels=in_features,
            hidden_channels=[*hidden, out_features],
            activation_layer=torch.nn.Tanh,
        )

    def forward(self, inputs):
        return self.mlp(inputs["feats"])
