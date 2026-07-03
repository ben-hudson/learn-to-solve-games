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

    def __init__(self, lr, normalizer=None, weight_decay=1e-2, start_factor=0.01, warmup_epochs=50, cosine_annealing=True):
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
            # No norm clip here: the field must stay jacrev-transparent for Consensus, but
            # NormClip (a BaseTransform) copy-copies its input and breaks jacrev. Clipping the
            # training target already teaches the model the reduced-range, correctly-directed field.
            return self.normalizer.inverse_target(self(inputs))

        return v

    def _rel_err(self, prediction, targets):
        """Clipped relative error in real units. Targets are already clipped by the dataset, so the
        prediction is norm-clipped the same way (after de-standardizing both) -- this measures error
        against the clipped field rather than being dominated by the operator's blow-up outliers.
        """
        preds = self.normalizer.clip_field(self.normalizer.inverse_target(prediction))
        reals = self.normalizer.inverse_target(targets)  # dataset already clipped these
        return torch.linalg.norm(preds - reals) / torch.linalg.norm(reals)

    def training_step(self, batch, _):
        inputs, targets = batch
        prediction = self(inputs)
        loss = self.loss_fn(prediction, targets)
        batch_size = targets.shape[0]
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        # Same clipped metric as val, on the train set -- the train/val gap reveals over/underfitting.
        self.log("train_rel_err", self._rel_err(prediction, targets), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, _):
        inputs, targets = batch
        batch_size = targets.shape[0]
        prediction = self(inputs)
        self.log("val_mse", self.loss_fn(prediction, targets), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_rel_err", self._rel_err(prediction, targets), on_epoch=True, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        # AdamW with linear warmup then cosine annealing (ported from markov-traffic-eq): the flat lr
        # bounced near convergence, and warmup stabilizes the transformer's early steps. Warmup ramps
        # from lr*start_factor up to lr over warmup_epochs, then cosine decays to ~0 over the rest.
        # Requires warmup_epochs < trainer.max_epochs (else the cosine T_max is non-positive).
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=self.start_factor, total_iters=self.warmup_epochs)
        if self.cosine_annealing:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.trainer.max_epochs - self.warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup, cosine], milestones=[self.warmup_epochs])
        else:
            scheduler = warmup
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
