import argparse
import os

import lightning as L
import torch
import wandb

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import random_split, DataLoader
from l2s_games.datasets.zero_sum import RandomZeroSumOperatorDataset
from torch_geometric.transforms import BaseTransform, Compose
from sklearn.preprocessing import StandardScaler

from l2s_games.algorithms import Optimistic
from l2s_games.envs.zero_sum import dist_to_normal_cone, project_onto_simplex
from l2s_games.losses import NashAprLoss, NormLoss
from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.models.nash_mlp import NashMLPBackbone
from l2s_games.transforms import DegreeEmbedding, SPDEmbedding


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--amortization", type=str, choices=["full", "partial"], default="partial")
    parser.add_argument("--fully_amortized_loss", type=str, choices=["mse", "ni"], default="mse")
    parser.add_argument("--partially_amortized_loss", type=str, choices=["mse", "norm", "huber"], default="norm")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--logger",
        choices=["wandb", "csv"],
        default="wandb",
        help="where to log metrics ('csv' writes to {SCRATCH or .} and skips wandb)",
    )
    parser.add_argument("--debug", action="store_true", help="run a single train/val batch for quick sanity checking")
    parser.add_argument("--patience_epochs", type=int, default=40, help="early-stopping patience in epochs")
    parser.add_argument("--val_every_n_epochs", type=int, default=10, help="run validation every N epochs")
    # optimizer (Adam + linear-warmup->cosine, ported from train_field_gnn.py)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--start_factor", type=float, default=0.01, help="linear warmup start factor")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="linear warmup epochs (must be < --epochs)")
    parser.add_argument("--cosine_annealing", type=int, default=0, help="cosine-anneal after warmup (0 disables)")
    parser.add_argument("--gradient_clip_val", type=float, default=0, help="gradient-norm clip value (0 disables)")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")

    config = parser.parse_args()
    if config.seed is None:
        config.seed = torch.randint(0, 2**31 - 1, (1,)).item()
    return config


class BuildZeroSumFeats(BaseTransform):
    def __init__(self, mode=None):
        super().__init__()

        assert mode in ["full", "partial"], f"Mode must be 'full' or 'partial', got {mode}."
        self.mode = mode

    def forward(self, data):
        # node 0 is player 1 (payoffs A), node 1 is player 2 (payoffs B)
        payoffs = torch.stack([data.A, data.B]).flatten(start_dim=1)

        if self.mode == "partial":
            n_points_per_instance = data.point.size(0)
            data.payoffs = payoffs.expand(n_points_per_instance, -1, -1)
        else:
            data.payoffs = payoffs

        return data


class GraphToTuple(BaseTransform):
    def forward(self, data):
        return data.to_namedtuple()


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
        self, backbone, dim, n_actions, feat_mean, feat_scale, target_scale, step_size=0.1, steps=100, **kwargs
    ):
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        self.loss = NormLoss()
        self.step_size = step_size
        self.steps = steps
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # fold the sampled points into the batch dimension: every point is an
        # independent evaluation on the same graph
        n_points = batch.point.size(1)
        batch = batch._replace(
            point=batch.point.flatten(0, 1),
            payoffs=batch.payoffs.flatten(0, 1),
            operator=batch.operator.flatten(0, 1),
            in_degree=batch.in_degree.repeat_interleave(n_points, dim=0),
            out_degree=batch.out_degree.repeat_interleave(n_points, dim=0),
            spd=batch.spd.repeat_interleave(n_points, dim=0),
        )
        # normalize: one global mean/scale for the payoffs, one global scale for the
        # operator (isotropic, so its direction is untouched)
        return batch._replace(
            payoffs=(batch.payoffs - self.feat_mean) / self.feat_scale,
            operator=batch.operator / self.target_scale,
        )

    def training_step(self, batch, batch_idx):
        # feats: the raw point (simplex coordinates need no normalization) alongside
        # the normalized payoffs
        feats = torch.cat([batch.point, batch.payoffs], dim=-1)
        prediction = self.readout(self.backbone(feats, batch.in_degree, batch.out_degree, batch.spd))
        loss = self.loss(prediction, batch.operator)
        self.log("train/loss", loss)
        return loss

    def solve(self, batch):
        # the learned field stands in for the true operator: one Optimistic rollout per game,
        # from the uniform profile. The rollout follows the normalized field, an isotropic
        # rescale of the true one, so its equilibria are unchanged and the step size is
        # scale-free.

        # this recovers one instance per game, but its ugly
        n_points_per_game = batch.point.size(0) // batch.A.size(0)
        payoffs = batch.payoffs[::n_points_per_game]
        in_degree = batch.in_degree[::n_points_per_game]
        out_degree = batch.out_degree[::n_points_per_game]
        spd = batch.spd[::n_points_per_game]

        def operator(strategies):
            feats = torch.cat([strategies, payoffs], dim=-1)
            return self.readout(self.backbone(feats, in_degree, out_degree, spd))

        algorithm = Optimistic(h=self.step_size)
        strategies = torch.full_like(batch.eq, 1 / batch.eq.size(-1))
        for _ in range(self.steps):
            strategies = algorithm.step(strategies, operator, project_onto_simplex)
        return strategies

    def validation_step(self, batch, batch_idx):
        feats = torch.cat([batch.point, batch.payoffs], dim=-1)
        prediction = self.readout(self.backbone(feats, batch.in_degree, batch.out_degree, batch.spd))
        self.log("val/loss", self.loss(prediction, batch.operator))
        # stationarity of the rollout endpoint under the true operator: zero exactly at a Nash
        # equilibrium, so unlike a distance to the LP solution it is robust to equilibrium
        # non-uniqueness. target_scale puts it in the same normalized units as val_loss.
        strategies = self.solve(batch)
        operator = RandomZeroSumOperatorDataset.eval_operator(batch, strategies)
        residual = dist_to_normal_cone(operator / self.target_scale, strategies)
        self.log("val/residual", residual.norm(dim=-1).mean())


class SolutionModel(AmortizedModel):
    """Predicts the equilibrium strategy profile directly from the payoffs (full amortization).

    Trained self-supervised on the Nash approximation loss (Duan et al. 2023, Algorithm 1): the
    predicted profile is scored by how much any player gains by deviating, so no solver labels
    are needed and equilibrium non-uniqueness is a non-issue.
    """

    def __init__(self, backbone, dim, n_actions, feat_mean, feat_scale, **kwargs):
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        self.loss = NashAprLoss()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # normalize the payoff features fed to the network. A and B, which the loss scores
        # deviations against, are isotropically rescaled (deviation gains are invariant to
        # utility shifts and linear in scale, so the minimizers are unchanged): the paper's
        # utilities live in [0, 1], and raw GAMUT payoffs (~1e2) blow up the softmax gradients
        return batch._replace(
            payoffs=(batch.payoffs - self.feat_mean) / self.feat_scale,
            A=batch.A / self.feat_scale,
            B=batch.B / self.feat_scale,
        )

    def predict_strategies(self, batch):
        # one embedding per player node; softmax puts each player's readout on the simplex,
        # so the prediction is a valid mixed-strategy profile
        embedding = self.backbone(batch.payoffs, batch.in_degree, batch.out_degree, batch.spd)
        return self.readout(embedding).softmax(dim=-1)

    def training_step(self, batch, batch_idx):
        loss = self.loss(self.predict_strategies(batch), batch.A, batch.B)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        strategies = self.predict_strategies(batch)
        self.log("val/loss", self.loss(strategies, batch.A, batch.B))
        # stationarity of the prediction under the operator: zero exactly at a Nash equilibrium,
        # so unlike a distance to the LP solution it is robust to equilibrium non-uniqueness.
        # A/B are already feat_scale-normalized here, so the residual is in the loss's units.
        operator = RandomZeroSumOperatorDataset.eval_operator(batch, strategies)
        residual = dist_to_normal_cone(operator, strategies)
        self.log("val/residual", residual.norm(dim=-1).mean())


if __name__ == "__main__":
    config = get_config()
    L.seed_everything(config.seed, workers=True)

    transforms = Compose(
        [BuildZeroSumFeats(mode=config.amortization), SPDEmbedding(), DegreeEmbedding(), GraphToTuple()]
    )
    # the operator dataset contains the equilibrium solutions too, so it works for the fully amortized model
    dataset = RandomZeroSumOperatorDataset(
        config.dataset, n_points_per_instance=256, force_reload=True, transform=transforms
    )

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    sample = dataset[0]

    feat_scaler = StandardScaler()
    operator_scaler = StandardScaler(with_mean=False)
    for batch in train_loader:
        # payoff entries are mutually comparable, so they share one global mean/scale
        # (a single column) rather than per-entry stats that would distort the game
        feat_scaler.partial_fit(batch.payoffs.reshape(-1, 1))
        # a single column, so the fit yields one global scale: an isotropic rescale of the
        # operator field that preserves its direction
        operator_scaler.partial_fit(batch.operator.reshape(-1, 1))

    dim = 128
    optimizer_kwargs = dict(
        lr=config.lr,
        start_factor=config.start_factor,
        warmup_epochs=config.warmup_epochs,
        cosine_annealing=bool(config.cosine_annealing),
    )
    if config.amortization == "full":
        # the paper's NE-approximator MLP: its [0, 1]-projected parameters bound the embedding
        # scale, so the softmax readout cannot saturate under the vertex-seeking NashApr gradient
        backbone = NashMLPBackbone(
            n_feats=sample.payoffs.size(-1),
            n_players=sample.payoffs.size(0),
            dim=dim,
        )
        model = SolutionModel(
            backbone,
            dim=dim,
            n_actions=sample.A.size(-1),
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            **optimizer_kwargs,
        )
    else:
        backbone = GraphormerBackbone(
            n_feats=sample.payoffs.size(-1) + sample.point.size(-1),
            in_degree=sample.in_degree,
            out_degree=sample.out_degree,
            spd=sample.spd,
            dim=dim,
            n_heads=8,
            n_layers=6,
            dim_ff=dim * 2,
            dropout=0.0,
        )
        model = FieldModel(
            backbone,
            dim=dim,
            n_actions=sample.A.size(-1),
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            target_scale=operator_scaler.scale_,
            **optimizer_kwargs,
        )
    save_dir = os.getenv("SCRATCH", ".")
    if config.logger == "wandb" and not config.debug:
        run = wandb.init(project="learn-to-solve-games", config=vars(config), dir=save_dir)
        logger = WandbLogger(experiment=run, save_dir=save_dir)
    else:
        logger = CSVLogger(save_dir=save_dir)
    # Debug runs disable checkpointing, and Lightning rejects a ModelCheckpoint when it's off.
    callbacks = [
        EarlyStopping(
            monitor="val/residual",
            mode="min",
            patience=max(1, config.patience_epochs // config.val_every_n_epochs),
            check_finite=False,
            strict=False,
        )
    ]
    if not config.debug:
        callbacks.append(
            ModelCheckpoint(monitor="val/residual", mode="min", save_top_k=1, save_last=True, filename="best")
        )
    trainer = L.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        default_root_dir=save_dir,
        fast_dev_run=config.debug,
        enable_checkpointing=logger is not None,
        callbacks=callbacks,
        gradient_clip_val=config.gradient_clip_val or None,
        check_val_every_n_epoch=config.val_every_n_epochs,
    )
    trainer.fit(model, train_loader, val_loader)
