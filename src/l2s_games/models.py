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


def conditioned_field(model, params):
    """Return a plain field callable ``v(z)`` for a fixed instance.

    The returned function accepts a point ``(2,)`` or grid ``(..., 2)``,
    appends the instance ``params``, and runs ``model`` -- so a trained
    field ``MLP`` becomes a drop-in field for ``simulate`` / quiver plotting.
    It is grad-transparent, so ``torch.func.jacrev`` can differentiate it;
    callers that only need values are responsible for their own ``no_grad`` /
    ``detach``.
    """
    params = torch.as_tensor(params, dtype=torch.float32)

    def v(z):
        z = torch.as_tensor(z, dtype=torch.float32)
        broadcast_params = params.expand(*z.shape[:-1], params.shape[-1])
        features = torch.cat([z, broadcast_params], dim=-1)
        return model(features)

    return v
