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

    def __init__(
        self, lr, normalizer=None, weight_decay=1e-2, start_factor=0.01, warmup_epochs=50, cosine_annealing=True
    ):
        super().__init__()
        self.lr = lr
        self.normalizer = normalizer
        self.weight_decay = weight_decay
        self.start_factor = start_factor
        self.warmup_epochs = warmup_epochs
        self.cosine_annealing = cosine_annealing
        self.loss_fn = nn.MSELoss()

    def batched_field(self, family, batch):
        """Learned field over a whole batch of instances: ``v(Z)``, ``Z [B, E] -> [B, E]``, real units.

        Splices the domain point ``Z`` into the batch's precomputed inputs via the family seam
        (``batched_field_input``) and de-standardizes the prediction. Reuses the batch's line-graph
        structure -- no re-featurization -- and stays grad-/jacrev-transparent (no in-place ops).
        """

        def v(z):
            inputs = family.batched_field_input(batch, z, self.normalizer)
            # De-standardize into real units; the asinh target scaler's inverse (sinh) is smooth, so
            # this stays jacrev-transparent for Consensus.
            return self.normalizer.inverse_target(self(inputs))

        return v

    def _field_metrics(self, prediction, targets):
        """Direction + magnitude fidelity of the predicted field, in real units.

        Both are computed per point over the edge axis (the R^E field vector the dynamics step along),
        averaged over points whose true field is non-degenerate (``‖true‖`` above a floor -- the
        direction of a ~zero vector, right at equilibrium, is undefined). Returns ``(cos_err,
        mag_err)``, which isolate the two failure modes: ``cos_err = mean(1 - cos(pred, true))``
        (0 = aligned) is the pure angle error, and ``mag_err = mean(|‖pred‖ - ‖true‖|)`` (0 = right
        scale, in real field units) is the pure magnitude error -- the absolute gap in field length,
        which cosine (magnitude-blind) does not capture.
        """
        pred = self.normalizer.inverse_target(prediction)
        true = self.normalizer.inverse_target(targets)
        pred_norm = torch.linalg.norm(pred, dim=-1)
        true_norm = torch.linalg.norm(true, dim=-1)
        keep = true_norm > 1e-6 * true_norm.mean()
        cos = (pred * true).sum(dim=-1) / (pred_norm * true_norm).clamp(min=1e-12)
        cos_err = (1.0 - cos)[keep].mean()
        mag_err = (pred_norm - true_norm).abs()[keep].mean()
        return cos_err, mag_err

    def training_step(self, batch, _):
        inputs, targets = batch
        prediction = self(inputs)
        loss = self.loss_fn(prediction, targets)
        # Only the loss is logged on train: under the streaming pipeline every batch is a fresh unseen
        # instance, so a train relative error is not a fit signal (it just re-estimates val_rel_err).
        self.log("train/mse", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.shape[0])
        return loss

    def validation_step(self, batch, _):
        inputs, targets = batch
        batch_size = targets.shape[0]
        prediction = self(inputs)
        cos_err, mag_err = self._field_metrics(prediction, targets)
        self.log("val/mse", self.loss_fn(prediction, targets), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/cos_err", cos_err, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/mag_err", mag_err, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        # AdamW with linear warmup then cosine annealing (ported from markov-traffic-eq): the flat lr
        # bounced near convergence, and warmup stabilizes the transformer's early steps. Warmup ramps
        # from lr*start_factor up to lr over warmup_epochs, then cosine decays to ~0 over the rest.
        # Requires warmup_epochs < trainer.max_epochs (else the cosine T_max is non-positive).
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=self.start_factor, total_iters=self.warmup_epochs
        )
        if self.cosine_annealing:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.trainer.max_epochs - self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup, cosine], milestones=[self.warmup_epochs])
        else:
            scheduler = warmup
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
