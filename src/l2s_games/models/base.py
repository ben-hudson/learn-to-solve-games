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

    def __init__(self, lr, normalizer=None):
        super().__init__()
        self.lr = lr
        self.normalizer = normalizer
        self.loss_fn = nn.MSELoss()

    def batched_field(self, family, batch):
        """Learned field over a whole batch of instances: ``v(Z)``, ``Z [B, E] -> [B, E]``, real units.

        Splices the domain point ``Z`` into the batch's precomputed inputs via the family seam
        (``batched_field_input``) and de-standardizes the prediction. Reuses the batch's line-graph
        structure -- no re-featurization -- and stays grad-/jacrev-transparent (no in-place ops).
        """

        def v(z):
            inputs = family.batched_field_input(batch, z, self.normalizer)
            return self.normalizer.inverse_target(self(inputs))

        return v

    def _mse(self, batch):
        inputs, targets = batch
        return self.loss_fn(self(inputs), targets)

    def training_step(self, batch, _):
        loss = self._mse(batch)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        inputs, targets = batch
        batch_size = targets.shape[0]
        prediction = self(inputs)
        self.log("val_mse", self.loss_fn(prediction, targets), on_epoch=True, prog_bar=True, batch_size=batch_size)
        preds = self.normalizer.inverse_target(prediction)
        reals = self.normalizer.inverse_target(targets)
        rel_err = torch.linalg.norm(preds - reals) / torch.linalg.norm(reals)
        self.log("val_rel_err", rel_err, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
