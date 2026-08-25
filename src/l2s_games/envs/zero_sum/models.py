import lightning as L
import torch

from l2s_games.algorithms import Optimistic
from l2s_games.envs.zero_sum.game import operator

from .losses import NashAprLoss, NormHuberLoss, NormLoss
from .utils import dist_to_normal_cone, project_onto_simplex


class AmortizedModel(L.LightningModule):
    def __init__(
        self,
        backbone,
        dim,
        n_actions,
        feat_mean,
        feat_scale,
        lr=1e-3,
        start_factor=0.01,
        warmup_epochs=10,
        cosine_annealing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.readout = torch.nn.Linear(dim, n_actions)
        # fitted normalization stats as buffers: they move to the model's device with the
        # module and serialize into checkpoints
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_scale", torch.as_tensor(feat_scale, dtype=torch.float32))
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


class FieldModel(AmortizedModel):
    def __init__(
        self, backbone, dim, n_actions, feat_mean, feat_scale, target_scale, step_size=2e-3, steps=2000, **kwargs
    ):
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        self.loss = NormHuberLoss(delta=1.0)
        self.step_size = step_size
        self.steps = steps
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def training_step(self, batch, batch_idx):
        # fold the sampled points into the batch dimension: every point is an
        # independent evaluation on the same graph
        n_points_per_instance = batch["point"].size(1)
        # release the batch dims so per-point keys ([B * P, ...]) can coexist with
        # per-instance keys ([B, ...])
        batch.batch_size = []
        batch["point"] = batch["point"].flatten(0, 1)
        batch["payoffs"] = batch["payoffs"].flatten(0, 1)
        batch["operator"] = batch["operator"].flatten(0, 1)
        batch["in_degree"] = batch["in_degree"].repeat_interleave(n_points_per_instance, dim=0)
        batch["out_degree"] = batch["out_degree"].repeat_interleave(n_points_per_instance, dim=0)
        batch["spd"] = batch["spd"].repeat_interleave(n_points_per_instance, dim=0)
        # normalize: one global mean/scale for the payoffs, one global scale for the
        # operator (isotropic, so its direction is untouched)
        batch["payoffs"] = self.normalize_feats(batch["payoffs"])
        batch["operator"] = batch["operator"] / self.target_scale
        # feats: the raw point (simplex coordinates need no normalization) alongside
        # the normalized payoffs
        feats = torch.cat([batch["point"], batch["payoffs"]], dim=-1)
        prediction = self.readout(self.backbone(feats, batch["in_degree"], batch["out_degree"], batch["spd"]))
        loss = self.loss(prediction, batch["operator"])
        self.log("train/loss", loss)
        return loss

    def solve(self, batch):
        # the learned field stands in for the true operator: one Optimistic rollout per game,
        # from the uniform profile. The rollout follows the normalized field, an isotropic
        # rescale of the true one, so its equilibria are unchanged and the step size is
        # scale-free.
        # every sampled point shares the instance's payoffs, so one copy per game suffices
        payoffs = self.normalize_feats(batch["payoffs"][:, 0])

        def operator(strategies):
            feats = torch.cat([strategies, payoffs], dim=-1)
            return self.readout(self.backbone(feats, batch["in_degree"], batch["out_degree"], batch["spd"]))

        algorithm = Optimistic(self.step_size, operator, project_onto_simplex)
        # start from the uniform profile: one [2, n] profile per instance, shaped like a
        # single sampled point
        strategies = torch.full_like(batch["point"][:, 0], 1 / batch["point"].size(-1))
        for _ in range(self.steps):
            strategies = algorithm.step(strategies)
        return strategies

    def validation_step(self, batch, batch_idx):
        # stationarity of the rollout endpoint under the true operator: zero exactly at a Nash
        # equilibrium, so unlike a distance to the LP solution it is robust to equilibrium
        # non-uniqueness. target_scale puts it in the same normalized units as train/loss.
        strategies = self.solve(batch)
        op = operator(batch["A"], batch["B"], strategies)
        residual = dist_to_normal_cone(op / self.target_scale, strategies)
        self.log("val/residual", residual.norm(dim=-1).mean())


class SolutionModel(AmortizedModel):
    """Predicts the equilibrium strategy profile directly from the payoffs (full amortization).

    Trained self-supervised on the Nash approximation loss (Duan et al. 2023, Algorithm 1): the
    predicted profile is scored by how much any player gains by deviating, so no solver labels
    are needed and equilibrium non-uniqueness is a non-issue.
    """

    def __init__(self, backbone, dim, n_actions, feat_mean, feat_scale, loss, **kwargs):
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        self.loss = loss

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # normalize the payoff features fed to the network. A and B, which the loss scores
        # deviations against, are isotropically rescaled (deviation gains are invariant to
        # utility shifts and linear in scale, so the minimizers are unchanged): the paper's
        # utilities live in [0, 1], and raw GAMUT payoffs (~1e2) blow up the softmax gradients
        batch["payoffs"] = self.normalize_feats(batch["payoffs"])
        return batch

    def predict_strategies(self, batch):
        # one embedding per player node; softmax puts each player's readout on the simplex,
        # so the prediction is a valid mixed-strategy profile
        embedding = self.backbone(batch["payoffs"], batch["in_degree"], batch["out_degree"], batch["spd"])
        # return self.readout(embedding).softmax(dim=-1)
        return project_onto_simplex(self.readout(embedding))

    def training_step(self, batch, batch_idx):
        loss = self.loss(self.predict_strategies(batch), batch["A"], batch["B"])
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        strategies = self.predict_strategies(batch)
        self.log("val/loss", self.loss(strategies, batch["A"], batch["B"]))
        # stationarity of the prediction under the operator: zero exactly at a Nash equilibrium,
        # so unlike a distance to the LP solution it is robust to equilibrium non-uniqueness.
        op = operator(batch["A"], batch["B"], strategies)
        residual = dist_to_normal_cone(op, strategies)
        self.log("val/residual", residual.norm(dim=-1).mean())
