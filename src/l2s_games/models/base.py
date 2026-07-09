"""Shared Lightning base for amortized field models."""

import copy

import lightning as L
import torch
from torch import nn

from l2s_games.data import normalize_input


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
        # Own a private copy of the normalizer as a submodule: its stats are buffers, so Lightning
        # moves them to the model's device with the rest of the module (needed by inverse_target /
        # batched_field_input, which combine them with on-device predictions) and serializes them
        # into state_dict. It must be a *copy* -- the data pipeline shares one normalizer across the
        # CPU-side datasets/streams (some in forked workers, where CUDA tensors are unsafe), so the
        # model cannot move the shared instance onto the GPU.
        self.normalizer = copy.deepcopy(normalizer)
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

    def conditioned_field(self, family, params):
        """Return a plain field callable ``v(z)`` for a fixed instance, in real units.

        The single-instance mirror of ``batched_field``: runs the family's conditioning seams
        (``model_input`` -> ``transform`` -> standardize ``feats`` via ``normalize_input`` ->
        ``collate_fn`` a batch of one), then de-standardizes the prediction. Works for any
        representation (flat dict or graph ``Data``), so a trained field model is a drop-in field for
        ``simulate`` / plotting. Unlike ``batched_field`` (which splices ``z`` into a precomputed batch
        and so needs ``z`` shaped ``[B, d]``), this re-featurizes per call, so ``v`` accepts any ``z``
        shape -- a bare ``[d]`` iterate, ``[n, d]``, or a ``[grid, grid, d]`` quiver grid. It is
        grad-transparent (``torch.func.jacrev`` works -- ``feats`` and normalization are differentiable
        in ``z``, structure is not a function of ``z``); value-only callers handle their own
        ``no_grad`` / ``detach``.
        """

        def v(z):
            z = torch.as_tensor(z, dtype=torch.float32)
            item = normalize_input(family.model_input(params, z), family.transform, self.normalizer)
            # De-standardize into real units; the asinh target scaler's inverse (sinh) is smooth, so
            # this stays jacrev-transparent for Consensus / quiver.
            return self.normalizer.inverse_target(self(family.collate_fn([item])).squeeze(0))

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
        # A train batch is a mapping of named data sources (e.g. {"uniform": ..., "rollout": ...})
        # to each source's (inputs, targets) -- Lightning's CombinedLoader over one loader per source.
        # Concatenate every source's prediction + target into one MSE (family-agnostic: this operates
        # on the model's output tensors, not on the family-specific collated inputs).
        prediction = torch.cat([self(inputs) for inputs, _ in batch.values()])
        targets = torch.cat([targets for _, targets in batch.values()])
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
