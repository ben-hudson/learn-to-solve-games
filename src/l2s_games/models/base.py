"""Shared Lightning bases for amortized models.

An *amortized model* wraps a backbone network (a plain ``nn.Module`` mapping the family's
``model_input`` to a per-element ``[B, E]`` prediction) in MSE-regression training. Two tasks
subclass the generic base, differing in *what* the prediction means:

- ``FieldModel`` predicts the operator **field** ``operator(params, point)`` -- a heavy-tailed vector
  field with an equilibrium at ``F=0``. It owns the field-specific pieces: the target-space loss
  variants, the direction/magnitude field metrics, and the ``batched_field`` / ``conditioned_field``
  seams a solver rolls out.
- ``SolutionModel`` predicts the **solution** ``z*`` directly (full amortization). ``z*`` is a
  generic regression target, so it uses the base's plain MSE and standardized target; it exposes a
  ``solve`` seam (the direct prediction) instead of a rollable field.
"""

import copy

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from l2s_games.data import normalize_input


class AmortizedModel(L.LightningModule):
    """MSE-regression training for a backbone ``net``; architecture- and task-agnostic.

    A batch is ``(inputs, targets)`` where ``inputs`` is whatever the family's ``model_input``
    produces (a dict with at least a ``feats`` entry) and ``self(inputs) = self.net(inputs)`` is the
    prediction, in the normalizer's target space. The loss is plain MSE there; ``FieldModel``
    overrides ``_compute_loss`` with its field-specific variants.
    """

    def __init__(
        self,
        net,
        lr,
        normalizer=None,
        weight_decay=1e-2,
        start_factor=0.01,
        warmup_epochs=50,
        cosine_annealing=True,
    ):
        super().__init__()
        self.net = net
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

    def forward(self, inputs):
        return self.net(inputs)

    def inverse_target(self, y):
        """De-standardize a target-space tensor (a prediction or a target) into real units.

        The model owns the normalizer, so it owns this inverse -- callers (the model's own field/solve
        seams and the validation callbacks) go through here rather than reaching into ``normalizer``.
        """
        return self.normalizer.inverse_target(y)

    def _compute_loss(self, prediction, targets):
        """Generic regression loss: MSE in the normalizer's (standardized) target space."""
        return F.mse_loss(prediction, targets)

    def _extra_val_metrics(self, prediction, targets, batch_size):
        """Hook for task-specific validation metrics (no-op here; see ``FieldModel``)."""

    def training_step(self, batch, _):
        # A train batch is a mapping of named data sources (e.g. {"uniform": ..., "rollout": ...})
        # to each source's (inputs, targets) -- Lightning's CombinedLoader over one loader per source.
        # Concatenate every source's prediction + target into one loss (family-agnostic: this operates
        # on the model's output tensors, not on the family-specific collated inputs).
        prediction = torch.cat([self(inputs) for inputs, _ in batch.values()])
        targets = torch.cat([targets for _, targets in batch.values()])
        loss = self._compute_loss(prediction, targets)
        # Log the optimized loss plus the plain MSE under a fixed name (comparable across loss modes).
        # Only these are logged on train: under the streaming pipeline every batch is a fresh unseen
        # instance, so a train relative error is not a fit signal (it just re-estimates val_rel_err).
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.shape[0])
        self.log(
            "train/mse", self.loss_fn(prediction, targets), on_step=False, on_epoch=True, batch_size=targets.shape[0]
        )
        return loss

    def validation_step(self, batch, _):
        inputs, targets = batch
        batch_size = targets.shape[0]
        prediction = self(inputs)
        # val/mse is always the plain MSE (fixed across loss modes -> comparable + a stable monitor);
        # val/loss is the optimized loss (equals val/mse when the loss is plain MSE).
        self.log("val/mse", self.loss_fn(prediction, targets), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "val/loss", self._compute_loss(prediction, targets), on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self._extra_val_metrics(prediction, targets, batch_size)

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


class FieldModel(AmortizedModel):
    """Predicts the operator field ``operator(params, point)`` -- rolled out by a solver to reach ``z*``.

    Adds the field-specific pieces on top of the generic regressor: the loss variants (the model
    predicts in the normalizer's target space -- global-standardized, warped per ``--target_warp``),
    the direction/magnitude field metrics, and the ``batched_field`` / ``conditioned_field`` seams that
    make a trained model a drop-in field for ``simulate`` / plotting.
    """

    def __init__(self, net, lr, loss="mse_target", huber_delta_scale=1.0, rel_eps=1.0, **kwargs):
        super().__init__(net, lr, **kwargs)
        # Which norm the training loss measures error in. The model predicts in the normalizer's target
        # space (a global scale + optional warp). "mse_target" (default) compares there directly; the
        # real-space variants ("mse_real"/"huber"/"rel_l2") compare in *scaled* real units via
        # _scaled_real (undo the warp only, not the scale) -- the norm the rollout-residual bound
        # controls. Staying in scale units keeps the loss O(1) so lr transfers. huber is the stable
        # default of the real variants. (Under --target_warp none the warp is identity, so "mse_target"
        # and "mse_real" coincide.)
        # Naming: "mse" is the only norm offered in both spaces, so each variant is space-qualified
        # ("_target" = warped target space, "_real" = real/unwarped scaled space); "huber"/"rel_l2" are
        # real-space only, hence unqualified. "rel_l2"'s "l2" is the FNO relative-L2 *norm*, not the
        # squared-error "mse_real".
        assert loss in ("mse_target", "mse_real", "huber", "rel_l2"), loss
        self.loss = loss
        self.huber_delta_scale = huber_delta_scale
        self.rel_eps = rel_eps

    def batched_field(self, family, batch):
        """Learned field over a whole batch of instances: ``v(Z)``, ``Z [B, E] -> [B, E]``, real units.

        Splices the domain point ``Z`` into the batch's precomputed inputs via the family seam
        (``batched_field_input``) and de-standardizes the prediction. Reuses the batch's line-graph
        structure -- no re-featurization -- and stays grad-/jacrev-transparent (no in-place ops).
        """

        def v(z):
            inputs = family.batched_field_input(batch, z, self.normalizer)
            # De-standardize into real units; the target scaler's inverse (scale, and sinh if the warp
            # is asinh) is smooth, so this stays jacrev-transparent for Consensus.
            return self.inverse_target(self(inputs))

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
            return self.inverse_target(self(family.collate_fn([item])).squeeze(0))

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
        pred = self.inverse_target(prediction)
        true = self.inverse_target(targets)
        pred_norm = torch.linalg.norm(pred, dim=-1)
        true_norm = torch.linalg.norm(true, dim=-1)
        keep = true_norm > 1e-6 * true_norm.mean()
        cos = (pred * true).sum(dim=-1) / (pred_norm * true_norm).clamp(min=1e-12)
        cos_err = (1.0 - cos)[keep].mean()
        mag_err = (pred_norm - true_norm).abs()[keep].mean()
        return cos_err, mag_err

    def _scaled_real(self, z):
        """Map the model's target-space value to *scaled* real units by undoing only the warp.

        The normalizer's target round-trip is scale-then-warp; this inverts just the warp, leaving the
        value in scale units (real / scale) -- O(1), so the huber/rel knobs and lr transfer. For
        ``--target_warp asinh`` this is ``sinh(z)``; for ``none`` (no warp) it is the identity.
        """
        warp = self.normalizer.target_warp
        return warp.inverse_transform(z) if warp is not None else z

    def _compute_loss(self, prediction, targets):
        """The optimized training loss under the configured norm (see ``__init__``).

        ``prediction``/``targets`` are in the normalizer's target space. ``_scaled_real`` undoes the
        warp (``sinh`` for asinh, identity for none), giving ``real_units / scale`` -- comparing there
        measures the real-unit error the rollout-residual bound controls, kept in scale units so it
        stays O(1) (lr transfers). ``rel_l2`` is the operator-learning-standard relative L2 (FNO), per
        sample, with a ``rel_eps`` floor so the vanishing target at the equilibrium doesn't detonate the
        ratio (and which also down-weights large-field samples where an asinh warp's gradients would
        otherwise blow up).
        """
        if self.loss == "mse_target":
            return F.mse_loss(prediction, targets)
        pred_r, true_r = self._scaled_real(prediction), self._scaled_real(targets)
        if self.loss == "mse_real":
            return F.mse_loss(pred_r, true_r)
        if self.loss == "huber":
            return F.huber_loss(pred_r, true_r, delta=self.huber_delta_scale)
        # rel_l2: per-sample ||pred_r - true_r|| / sqrt(||true_r||^2 + eps^2), mean over batch.
        num = (pred_r - true_r).norm(dim=-1)
        den = (true_r.norm(dim=-1).square() + self.rel_eps**2).sqrt()
        return (num / den).mean()

    def _extra_val_metrics(self, prediction, targets, batch_size):
        cos_err, mag_err = self._field_metrics(prediction, targets)
        self.log("val/cos_err", cos_err, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/mag_err", mag_err, on_epoch=True, prog_bar=True, batch_size=batch_size)


class SolutionModel(AmortizedModel):
    """Predicts the equilibrium solution ``z*`` directly from parameters (full amortization).

    ``z*`` is a generic regression target (a per-edge cost vector, standardized like the inputs), so
    training uses the base's plain MSE -- none of the field-specific loss or field metrics apply.
    Instead of a rollable field it exposes ``solve``: the direct, de-standardized, feasible prediction.
    """

    def solve(self, family, inputs):
        """The predicted equilibrium ``z*`` for a batch of instances, in real units and feasible.

        De-standardizes the network output and projects onto the feasible set (the solution analogue
        of ``FieldModel.conditioned_field`` / ``batched_field``): used by ``SolutionPredictionCallback``
        to score the analytic residual, and a drop-in equilibrium estimator for downstream analysis.
        """
        return family.project(inputs, self.inverse_target(self(inputs)))
