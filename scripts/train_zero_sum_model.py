import argparse
import lightning as L
import os
import torch
import wandb

from l2s_games.envs.spe.streams import OperatorStream
from l2s_games.envs.traffic.datasets import GraphToTensorDict
from l2s_games.envs.zero_sum import (
    BuildZeroSumFeats,
    FieldModel,
    SolutionModel,
)
from l2s_games.envs.zero_sum.game import RandomZeroSum
from l2s_games.envs.zero_sum.losses import EGLoss, NashAprLoss, PotentialLoss
from l2s_games.envs.zero_sum.utils import simplex_projection, softmax_projection, straight_through_projection
from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.models.nash_mlp import NashMLPBackbone
from l2s_games.transforms import DegreeEmbedding, SPDEmbedding
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import Compose
from torch.utils.data import random_split, DataLoader
from functools import partial


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--amortization", type=str, choices=["full", "partial"], default="partial")
    parser.add_argument("--cosine_annealing", type=int, default=0)
    # parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--eg_step_size",
        type=float,
        default=2e-3,
        help="fixed lookahead distance of EGLoss, in raw payoff units (the same units the rollout "
        "algorithms step in). Ignored unless --amortization=full and --fully_amortized_loss=eg.",
    )
    parser.add_argument("--eg_projection", type=int, default=1, help="Apply projection in EGLoss forward.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fully_amortized_loss", type=str, choices=["potential", "eg", "ni"], default="ni")
    parser.add_argument("--gradient_clip_val", type=float, default=0)
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="knee of the partially amortized model's NormHuberLoss, in normalized operator units: "
        "residual norms below it get an MSE-like gradient that anneals as they fit, ones above it "
        "a bounded gradient of fixed magnitude. Ignored when --amortization=full.",
    )
    parser.add_argument("--logger", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--normalize_logits",
        type=int,
        default=0,
        help="apply a non-affine LayerNorm to the fully amortized model's readout logits before "
        "the simplex projection: pins the logit scale so vertex-seeking loss gradients cannot "
        "saturate the softmax backward. Caps softmax at ~0.8 mass per action, so pair with "
        "--projection=sparsemax or ste. Ignored when --amortization=partial.",
    )
    parser.add_argument("--partially_amortized_loss", type=str, choices=["mse", "norm", "huber"], default="norm")
    parser.add_argument("--patience_epochs", type=int, default=40)
    parser.add_argument(
        "--projection",
        type=str,
        choices=["softmax", "sparsemax", "ste"],
        default="sparsemax",
        help="how the fully amortized model's readout is mapped onto the simplex: 'sparsemax' is "
        "the exact Euclidean projection, which reaches the boundary and so can output pure "
        "strategies, but passes no gradient off the active support; 'softmax' is smooth everywhere "
        "but confined to the interior; 'ste' is a straight-through estimator combining the two -- "
        "the exact projection's value with the softmax's gradient. Ignored when "
        "--amortization=partial (the field rollout always needs the exact projection).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start_factor", type=float, default=0.01)
    parser.add_argument("--val_every_n_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--n_instances", type=int, default=1024)
    parser.add_argument("--n_points_per_instance", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_actions", type=int, default=3)

    config = parser.parse_args()
    if config.seed is None:
        config.seed = torch.randint(0, 2**31 - 1, (1,)).item()
    return config


if __name__ == "__main__":
    config = get_config()
    L.seed_everything(config.seed, workers=True)

    transforms = Compose(
        [BuildZeroSumFeats(mode=config.amortization), SPDEmbedding(), DegreeEmbedding(), GraphToTensorDict()]
    )
    # the operator dataset contains the equilibrium solutions too, so it works for the fully amortized model
    # dataset = RandomZeroSumOperatorDataset(config.dataset, n_points_per_instance=256, transform=transforms)
    sample = partial(RandomZeroSum.sample, n_actions=config.n_actions)
    sample_domain = lambda instance, n: torch.distributions.Dirichlet(torch.ones(instance.n_actions)).sample(
        (n, instance.n_players)
    )
    dataset = list(
        OperatorStream(
            sample,
            sample_domain,
            n_instances=config.n_instances,
            n_points_per_instance=config.n_points_per_instance,
            quiet=False,
            transform=transforms,
        )
    )

    cal_dataset, val_dataset, test_dataset, _ = random_split(dataset, [128, 128, 128, len(dataset) - 3 * 128])
    cal_loader = DataLoader(cal_dataset, batch_size=config.batch_size, collate_fn=torch.stack)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=torch.stack)

    train_dataset = OperatorStream(
        sample, sample_domain, n_instances=512, n_points_per_instance=16, quiet=True, transform=transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=torch.stack)

    feat_scaler = StandardScaler()
    operator_scaler = StandardScaler(with_mean=False)
    for batch in cal_loader:
        # payoff entries are mutually comparable, so they share one global mean/scale
        # (a single column) rather than per-entry stats that would distort the game
        feat_scaler.partial_fit(batch["payoffs"].reshape(-1, 1))
        # a single column, so the fit yields one global scale: an isotropic rescale of the
        # operator field that preserves its direction
        operator_scaler.partial_fit(batch["operator"].reshape(-1, 1))

    sample = dataset[0]
    dim = 128
    optimizer_kwargs = dict(
        lr=config.lr,
        start_factor=config.start_factor,
        warmup_epochs=config.warmup_epochs,
        cosine_annealing=bool(config.cosine_annealing),
    )
    if config.amortization == "full":
        if config.fully_amortized_loss == "ni":
            # the paper's NE-approximator MLP: its [0, 1]-projected parameters bound the embedding
            # scale, so the softmax readout cannot saturate under the vertex-seeking NashApr gradient
            backbone = NashMLPBackbone(
                n_feats=sample["payoffs"].size(-1),
                n_players=sample["payoffs"].size(0),
                dim=dim,
            )
            loss = NashAprLoss()
        else:
            backbone = GraphormerBackbone(
                n_feats=sample["payoffs"].size(-1),
                in_degree=sample["in_degree"],
                out_degree=sample["out_degree"],
                spd=sample["spd"],
                dim=dim,
                n_heads=8,
                n_layers=6,
                dim_ff=dim * 2,
                dropout=0.0,
            )
            if config.fully_amortized_loss == "potential":
                loss = PotentialLoss()
            elif config.fully_amortized_loss == "eg":
                loss = EGLoss(step_size=config.eg_step_size, project=bool(config.eg_projection))
        model = SolutionModel(
            backbone,
            dim=dim,
            n_actions=sample["A"].size(-1),
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            loss=loss,
            projection={
                "softmax": softmax_projection,
                "sparsemax": simplex_projection,
                "ste": straight_through_projection,
            }[config.projection],
            normalize_logits=bool(config.normalize_logits),
            **optimizer_kwargs,
        )
    else:
        backbone = GraphormerBackbone(
            n_feats=sample["point"].size(-1) + sample["payoffs"].size(-1),
            in_degree=sample["in_degree"],
            out_degree=sample["out_degree"],
            spd=sample["spd"],
            dim=dim,
            n_heads=8,
            n_layers=6,
            dim_ff=dim * 2,
            dropout=0.0,
        )
        model = FieldModel(
            backbone,
            dim=dim,
            n_actions=sample["n_actions"],
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            target_scale=operator_scaler.scale_,
            huber_delta=config.huber_delta,
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
