"""Amortized field model: ``(coordinates, instance params) -> field value``."""

import lightning as L
import torch
from torch import nn


class FieldLitModule(L.LightningModule):
    """Wraps a field model in an MSE-regression Lightning training step."""

    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def _mse(self, batch):
        inputs, targets = batch
        return self.loss_fn(self.model(inputs), targets)

    def training_step(self, batch, _):
        loss = self._mse(batch)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        self.log("val_mse", self._mse(batch), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def conditioned_field(model, family, params, normalizer):
    """Return a plain field callable ``v(z)`` for a fixed instance, in real units.

    The family builds the model input from ``(params, z)`` (the conditioning seam, so
    this works for both the flat and graph representations); ``v`` normalizes that input,
    runs ``model``, and maps the prediction back to real units -- making a trained field
    model a drop-in field for ``simulate`` / quiver plotting. It is grad-transparent, so
    ``torch.func.jacrev`` can differentiate it; value-only callers handle their own
    ``no_grad`` / ``detach``.
    """

    def v(z):
        z = torch.as_tensor(z, dtype=torch.float32)
        normalized_input = normalizer.input.transform(family.model_input(params, z))
        return normalizer.target.inverse_transform(model(normalized_input))

    return v
