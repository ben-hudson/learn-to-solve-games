"""Shared Lightning base for amortized field models."""

import lightning as L
import torch
from torch import nn


class FieldModel(L.LightningModule):
    """MSE-regression training for a field model; subclasses implement ``forward``.

    A batch is ``(inputs, targets)`` where ``inputs`` is whatever the family's
    ``model_input`` produces (a dict with at least a ``feats`` entry) and ``self(inputs)``
    is the predicted field. Subclasses build their own network and define ``forward``.
    """

    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def _mse(self, batch):
        inputs, targets = batch
        return self.loss_fn(self(inputs), targets)

    def training_step(self, batch, _):
        loss = self._mse(batch)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        self.log("val_mse", self._mse(batch), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
