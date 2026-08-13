import argparse
import lightning as L
import os
import torch
import wandb

from l2s_games.envs.zero_sum import (
    BuildZeroSumFeats,
    FieldModel,
    GraphToTuple,
    RandomZeroSumOperatorDataset,
    SolutionModel,
)
from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.models.nash_mlp import NashMLPBackbone
from l2s_games.transforms import DegreeEmbedding, SPDEmbedding
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import Compose
from torch.utils.data import random_split, DataLoader


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--amortization", type=str, choices=["full", "partial"], default="partial")
    parser.add_argument("--cosine_annealing", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fully_amortized_loss", type=str, choices=["mse", "ni"], default="mse")
    parser.add_argument("--gradient_clip_val", type=float, default=0)
    parser.add_argument("--logger", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--partially_amortized_loss", type=str, choices=["mse", "norm", "huber"], default="norm")
    parser.add_argument("--patience_epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start_factor", type=float, default=0.01)
    parser.add_argument("--val_every_n_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    config = parser.parse_args()
    if config.seed is None:
        config.seed = torch.randint(0, 2**31 - 1, (1,)).item()
    return config


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
