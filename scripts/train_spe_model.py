"""Train the fully amortized spatial price equilibrium solver.

The model reads an instance and predicts its equilibrium prices in one shot, scored by the
market-clearing field those prices induce -- no solver labels anywhere, so instances are *sampled*
rather than loaded and every epoch trains on fresh ones. The validation instances are drawn once and
kept, so the metric moves with the model rather than with the draw; they are held out only in the
sense that the model never trains on them, instances being i.i.d. draws from one sampler.

    python scripts/train_spe_model.py --debug
"""

import argparse
import lightning as L
import os
import torch
import wandb

from functools import partial
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose

from l2s_games.envs.spe import PriceSPE
from l2s_games.envs.spe.models import SpeSolutionModel, SpeTransformerBackbone
from l2s_games.envs.spe.streams import EquilibriumStream
from l2s_games.envs.spe.transforms import BuildSPEFeats
from l2s_games.envs.traffic.datasets import GraphToTensorDict
from l2s_games.losses import EGLoss, PotentialLoss


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cosine_annealing", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--delta", type=float, default=20.0, help="route cost per unit flow")
    parser.add_argument(
        "--eg_step_size",
        type=float,
        default=2e-2,
        help="lookahead distance of the eg loss, in raw price units (the same units the rollout "
        "algorithms step in). Independent of --lr, which only scales how far the update follows the "
        "lookahead field. Ignored unless --loss=eg.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.1, help="floor on the price effects' PD part")
    parser.add_argument("--gradient_clip_val", type=float, default=0)
    parser.add_argument(
        "--kappa",
        type=float,
        default=5.0,
        help="asymmetry of the price effect matrices: 0 makes the field a potential gradient, and at "
        "the default it is rotation-dominated, which is where --loss=eg earns its second operator "
        "evaluation",
    )
    parser.add_argument("--logger", choices=["wandb", "csv"], default="wandb")
    parser.add_argument(
        "--loss",
        type=str,
        choices=["potential", "eg"],
        default="eg",
        help="field surrogate the prices are scored by: 'potential' takes one plain step along the "
        "market-clearing field, 'eg' an extragradient step through a lookahead point",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_demand", type=int, default=4)
    parser.add_argument("--n_instances_per_epoch", type=int, default=512)
    parser.add_argument("--n_supply", type=int, default=3)
    parser.add_argument("--n_val_instances", type=int, default=128)
    parser.add_argument("--patience_epochs", type=int, default=40)
    parser.add_argument("--scale", type=float, default=0.4, help="spread of the price effect matrices")
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

    transforms = Compose([BuildSPEFeats(mode="full"), GraphToTensorDict()])
    sample = partial(
        PriceSPE.random_bipartite,
        n_supply=config.n_supply,
        n_demand=config.n_demand,
        kappa=config.kappa,
        eps=config.eps,
        delta=config.delta,
        scale=config.scale,
    )
    val_dataset = list(EquilibriumStream(sample, n_instances=config.n_val_instances, transform=transforms))
    train_dataset = EquilibriumStream(sample, n_instances=config.n_instances_per_epoch, transform=transforms)
    # torch.stack collates the per-instance TensorDicts into a batched TensorDict
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=torch.stack)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=torch.stack)

    instance = val_dataset[0]
    dim = 128
    # nothing is normalized: the sampled data is already O(1)-O(10) in comparable units and the loss
    # evaluates the true operator, so every logged number stays in ground-truth price units
    backbone = SpeTransformerBackbone(
        dim=dim,
        n_heads=8,
        n_layers=6,
        dim_ff=dim * 2,
        dropout=0.0,
        n_market_feats=instance["supply_feats"].size(-1),
        n_route_feats=instance["route_feats"].size(-1),
        n_pair_feats=instance["supply_pairs"].size(-1),
    )
    model = SpeSolutionModel(
        backbone,
        dim=dim,
        loss=PotentialLoss() if config.loss == "potential" else EGLoss(step_size=config.eg_step_size),
        lr=config.lr,
        start_factor=config.start_factor,
        warmup_epochs=config.warmup_epochs,
        cosine_annealing=bool(config.cosine_annealing),
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
