import argparse
import lightning as L
import torch
import wandb

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import random_split, DataLoader
from l2s_games.datasets.zero_sum import RandomZeroSumEquilibriumDataset, RandomZeroSumOperatorDataset
from torch_geometric.transforms import BaseTransform, Compose
from sklearn.preprocessing import StandardScaler

from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.transforms import DegreeEmbedding, SPDEmbedding


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--amortization", type=str, choices=["full", "partial"], default="partial")
    parser.add_argument("--fully_amortized_loss", type=str, choices=["mse", "ni"], default="mse")
    parser.add_argument("--partially_amortized_loss", type=str, choices=["mse", "norm", "huber"], default="norm")
    parser.add_argument("--seed", type=int, default=None)

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


class NormLoss(torch.nn.Module):
    """``||prediction - target||`` over each sample's flattened field, averaged over samples.

    Zero-residual samples are masked out of the mean: the norm's gradient at exactly zero is NaN.
    """

    def forward(self, prediction, target):
        # flatten start_dim=1 because the operator is a vector, so we have n_players*n_actions
        residual_norm = (prediction - target).flatten(start_dim=1).norm(dim=-1)
        return residual_norm[residual_norm > 0].mean()


class FieldModel(L.LightningModule):
    def __init__(self, backbone, dim, n_actions, feat_mean, feat_scale, target_scale, **kwargs):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.readout = torch.nn.Linear(dim, n_actions)
        self.loss = NormLoss()
        # fitted normalization stats as buffers: they move to the model's device with the
        # module and serialize into checkpoints
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_scale", torch.as_tensor(feat_scale, dtype=torch.float32))
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feats = torch.cat([batch.point, batch.payoffs], dim=-1)
        prediction = self.readout(self.backbone(feats, batch.in_degree, batch.out_degree, batch.spd))
        loss = self.loss(prediction, batch.operator)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SolutionModel(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = 0
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    config = get_config()
    L.seed_everything(config.seed, workers=True)

    transforms = Compose(
        [BuildZeroSumFeats(mode=config.amortization), SPDEmbedding(), DegreeEmbedding(), GraphToTuple()]
    )
    dataset = (
        RandomZeroSumEquilibriumDataset(config.dataset, transform=transforms)
        if config.amortization == "full"
        else RandomZeroSumOperatorDataset(config.dataset, n_points_per_instance=128, transform=transforms)
    )

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    sample = dataset[0]

    feat_scaler = StandardScaler()
    target_scaler = StandardScaler(with_mean=False)
    for batch in train_loader:
        # payoff entries are mutually comparable, so they share one global mean/scale
        # (a single column) rather than per-entry stats that would distort the game
        feat_scaler.partial_fit(batch.payoffs.reshape(-1, 1))
        # a single column, so the fit yields one global scale: an isotropic rescale of the
        # operator field that preserves its direction
        target_scaler.partial_fit(batch.operator.reshape(-1, 1))

    dim = 128
    model = FieldModel(
        GraphormerBackbone(
            n_feats=sample.payoffs.size(-1) + sample.point.size(-1),
            in_degree=sample.in_degree,
            out_degree=sample.out_degree,
            spd=sample.spd,
            dim=dim,
            n_heads=8,
            n_layers=6,
            dim_ff=dim * 2,
            dropout=0.0,
        ),
        dim=dim,
        n_actions=sample.A.size(-1),
        feat_mean=feat_scaler.mean_,
        feat_scale=feat_scaler.scale_,
        target_scale=target_scaler.scale_,
    )
    trainer = L.Trainer(
        max_epochs=100,
        logger=[CSVLogger("logs")],
        callbacks=[EarlyStopping(monitor="val_loss"), ModelCheckpoint(monitor="val_loss")],
    )
    trainer.fit(model, train_loader, val_loader)
