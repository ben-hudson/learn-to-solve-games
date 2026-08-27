import lightning as L
import torch

from l2s_games.algorithms import Optimistic
from l2s_games.envs.zero_sum.game import operator

from .losses import NashAprLoss, NormHuberLoss, NormLoss
from .utils import dist_to_normal_cone, simplex_projection


class AmortizedModel(L.LightningModule):
    def __init__(
        self,
        backbone,
        dim,
        n_actions,
        feat_mean,
        feat_scale,
        readout=None,
        lr=1e-3,
        start_factor=0.01,
        warmup_epochs=10,
        cosine_annealing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        # the readout has to match what the backbone embeds: a graph backbone gives one embedding
        # per player, off which every action's logit is read at once, while an action-token backbone
        # gives one per action and wants an `ActionReadout` scoring them one at a time
        self.readout = torch.nn.Linear(dim, n_actions) if readout is None else readout
        # fitted normalization stats as buffers: they move to the model's device with the
        # module and serialize into checkpoints
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_scale", torch.as_tensor(feat_scale, dtype=torch.float32))
        # scored at validation for every model, whether or not it is the training objective:
        # nfg_transformer's NE objective (its `equilibria.nash_approx`) is this same quantity, so
        # logging it always makes our runs comparable to theirs and to each other
        self.nash_apr = NashAprLoss()
        self.lr = lr
        self.start_factor = start_factor
        self.warmup_epochs = warmup_epochs
        self.cosine_annealing = cosine_annealing

    def normalize_feats(self, feats: torch.Tensor):
        return (feats - self.feat_mean) / self.feat_scale

    def configure_optimizers(self):
        # Adam with linear warmup then optional cosine annealing (ported from train_field_gnn.py):
        # warmup ramps from lr*start_factor up to lr over warmup_epochs, then cosine decays to ~0
        # over the rest. Requires warmup_epochs < trainer.max_epochs (else the cosine T_max is
        # non-positive).
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
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

    def predict_strategies(self, batch):
        """The model's strategy profile per instance, ``[B, n_players, n_actions]`` on the simplex.

        The one thing the two amortizations disagree on: ``FieldModel`` rolls its learned field out
        to a fixed point, ``SolutionModel`` reads a profile straight off the payoffs.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        # stationarity of the predicted profile under the true operator: zero exactly at a Nash
        # equilibrium, so unlike a distance to the LP solution it is robust to equilibrium
        # non-uniqueness. A and B are the raw payoffs -- nothing rescales them, in either
        # subclass -- so this is in ground-truth payoff units, and the two amortizations are
        # directly comparable on it. Subclasses add their own val/loss on top via super().
        strategies = self.predict_strategies(batch)
        op = operator(batch["A"], batch["B"], strategies)
        residual = dist_to_normal_cone(op, strategies)
        self.log("val/residual", residual.norm(dim=-1).mean())
        # the deviation gain of the predicted profile: also zero exactly at a Nash equilibrium, but
        # in payoff units rather than the residual's operator-distance units, and the metric
        # nfg_transformer reports. Redundant with val/loss only when NashAprLoss is the objective.
        self.log("val/nash_apr", self.nash_apr(strategies, batch["A"], batch["B"]))
        return strategies


class FieldModel(AmortizedModel):
    def __init__(
        self,
        backbone,
        dim,
        n_actions,
        feat_mean,
        feat_scale,
        target_scale,
        step_size=2e-3,
        steps=2000,
        huber_delta=1.0,
        **kwargs,
    ):
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        # the knee, in normalized operator units: samples inside it get an MSE-like gradient that
        # anneals as they fit, samples beyond it all push with the same bounded magnitude
        self.loss = NormHuberLoss(delta=huber_delta)
        self.step_size = step_size
        self.steps = steps
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def predict_field(self, batch):
        """Field prediction and target at every sampled point, both ``[B * P, n_players, n_actions]``.

        Folds the sampled points into the batch dimension: every point is an independent evaluation
        on the same graph. Reads the batch without mutating it, so the caller can go on to use the
        per-instance keys (``validation_step`` scores the rollout off the same batch).
        """
        n_points_per_instance = batch["point"].size(1)
        point = batch["point"].flatten(0, 1)
        # normalize: one global mean/scale for the payoffs, one global scale for the
        # operator (isotropic, so its direction is untouched)
        payoffs = self.normalize_feats(batch["payoffs"].flatten(0, 1))
        target = batch["operator"].flatten(0, 1) / self.target_scale
        in_degree = batch["in_degree"].repeat_interleave(n_points_per_instance, dim=0)
        out_degree = batch["out_degree"].repeat_interleave(n_points_per_instance, dim=0)
        spd = batch["spd"].repeat_interleave(n_points_per_instance, dim=0)
        # feats: the raw point (simplex coordinates need no normalization) alongside
        # the normalized payoffs
        feats = torch.cat([point, payoffs], dim=-1)
        return self.readout(self.backbone(feats, in_degree, out_degree, spd)), target

    def training_step(self, batch, batch_idx):
        loss = self.loss(*self.predict_field(batch))
        # epoch-level, matching SolutionModel and the val metrics: an epoch is only a handful of
        # steps, so the per-step default is noisy and CSVLogger's 100-step flush would drop it
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_strategies(self, batch):
        # the learned field stands in for the true operator: one Optimistic rollout per game,
        # from the uniform profile. The rollout follows the normalized field, an isotropic
        # rescale of the true one, so its equilibria are unchanged and the step size is
        # scale-free.
        # every sampled point shares the instance's payoffs, so one copy per game suffices
        payoffs = self.normalize_feats(batch["payoffs"][:, 0])

        def operator(strategies):
            feats = torch.cat([strategies, payoffs], dim=-1)
            return self.readout(self.backbone(feats, batch["in_degree"], batch["out_degree"], batch["spd"]))

        algorithm = Optimistic(self.step_size, operator, simplex_projection)
        # start from the uniform profile: one [2, n] profile per instance, shaped like a
        # single sampled point
        strategies = torch.full_like(batch["point"][:, 0], 1 / batch["point"].size(-1))
        for _ in range(self.steps):
            strategies = algorithm.step(strategies)
        return strategies

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        # the training objective on held-out points: how well the field itself is fit, as opposed
        # to val/residual's verdict on where rolling it out lands. Stays in the normalized units
        # train/loss lives in, so the two curves are comparable.
        self.log("val/loss", self.loss(*self.predict_field(batch)))


class SolutionModel(AmortizedModel):
    """Predicts the equilibrium strategy profile directly from the payoffs (full amortization).

    Trained self-supervised on the Nash approximation loss (Duan et al. 2023, Algorithm 1): the
    predicted profile is scored by how much any player gains by deviating, so no solver labels
    are needed and equilibrium non-uniqueness is a non-issue.
    """

    def __init__(
        self,
        backbone,
        dim,
        n_actions,
        feat_mean,
        feat_scale,
        loss,
        projection=simplex_projection,
        normalize_logits=False,
        **kwargs,
    ):
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        self.loss = loss
        # LayerNorm over the actions pins the logit scale, ending the norm race that saturates the
        # softmax under vertex-seeking loss gradients: the LN Jacobian is orthogonal to its input,
        # so the gradient component that grows the logits is discarded and only direction changes
        # pass. No learnable affine -- a trainable gain would restart the race one parameter
        # deeper. The pinned scale still reaches vertices through sparsemax (the top-1 logit gap
        # can exceed sparsemax's support threshold of 1) but caps softmax at ~0.8 mass on an
        # action, so pair it with the sparsemax or ste projections.
        self.logit_norm = (
            torch.nn.LayerNorm(n_actions, elementwise_affine=False) if normalize_logits else torch.nn.Identity()
        )
        # maps the readout onto the simplex: simplex_projection (sparsemax -- exact, reaches the
        # boundary, but zero gradient off the active support), softmax_projection (smooth
        # everywhere, but interior-only), or straight_through_projection (the exact projection's
        # value with the softmax's gradient). Only the readout is affected; FieldModel's rollout
        # always needs the exact projection, since Optimistic's convergence assumes a true
        # projection.
        self.projection = projection

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # normalize only the payoff features fed to the network. A and B are left at their raw
        # scale, so everything scored against them -- the loss and the val/residual -- is
        # reported in ground-truth payoff units, matching FieldModel.validation_step.
        batch["payoffs"] = self.normalize_feats(batch["payoffs"])
        return batch

    def predict_strategies(self, batch):
        # one embedding per player node; the projection puts each player's readout on the simplex,
        # so the prediction is a valid mixed-strategy profile
        embedding = self.backbone(batch["payoffs"], batch["in_degree"], batch["out_degree"], batch["spd"])
        return self.projection(self.logit_norm(self.readout(embedding)))

    def training_step(self, batch, batch_idx):
        loss = self.loss(self.predict_strategies(batch), batch["A"], batch["B"])
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # the base step already predicted the profile; reuse it rather than paying a second forward
        strategies = super().validation_step(batch, batch_idx)
        self.log("val/loss", self.loss(strategies, batch["A"], batch["B"]))
