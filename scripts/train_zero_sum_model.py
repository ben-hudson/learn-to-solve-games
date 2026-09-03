import argparse
import lightning as L
import os
import torch
import wandb

from l2s_games.envs.spe.streams import OperatorStream
from l2s_games.envs.traffic.datasets import GraphToTensorDict
from l2s_games.envs.zero_sum import (
    ActionReadout,
    ActionTokenBackbone,
    BuildZeroSumFeats,
    FieldModel,
    SolutionModel,
)
from l2s_games.envs.zero_sum.game import RandomZeroSum
from l2s_games.envs.zero_sum.losses import EGLoss, NashAprLoss, PotentialLoss
from l2s_games.envs.zero_sum.utils import simplex_projection, softmax_projection, straight_through_projection
from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.models.nash_mlp import NashMLPBackbone
from l2s_games.models.nfg_transformer import NfgTransformerBackbone
from l2s_games.models.zero_sum_transformer import ZeroSumTransformerBackbone
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
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["graphormer", "nash_mlp", "payoff_bias", "axial", "zero_sum_transformer", "nfg_transformer"],
        default="graphormer",
        help="what embeds the game. 'graphormer' and 'nash_mlp' read the payoff matrix as one flat "
        "vector per player, so the action count is baked into their input and readout widths and "
        "the action-relabelling symmetry has to be learned from data. The other four tokenize by "
        "action and are equivariant to that relabelling by construction, at a cost that rises in "
        "that order: 'payoff_bias' keeps P*T tokens and lets the payoffs in only as an attention "
        "bias, 'axial' carries them in T**2 joint-action tokens, and the two NfgTransformers rebuild "
        "the joint-action grid every layer -- 'zero_sum_transformer' over one shared grid, "
        "'nfg_transformer' over the reference's per-player grid, whose second copy holds nothing but "
        "the negation of the first once the game is zero-sum.",
    )
    parser.add_argument("--cosine_annealing", type=int, default=0)
    # parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dim", type=int, default=128, help="hidden dimension of the backbone.")
    parser.add_argument(
        "--dim_ff_mult",
        type=int,
        default=2,
        help="feed-forward dimension of each attention layer, as a multiple of --dim. Ignored when "
        "--backbone=nash_mlp.",
    )
    parser.add_argument(
        "--eg_step_size",
        type=float,
        default=0.1,
        help="fixed lookahead distance of EGLoss, in raw payoff units (the same units the rollout "
        "algorithms step in). RandomZeroSum.sample now normalizes each instance to unit variance, "
        "which shrank the field by the ~57.7 standard deviation of the old [-100, 100] draw, so "
        "this is the rescaled counterpart of the 2e-3 that was tuned against unnormalized payoffs. "
        "Ignored unless --amortization=full and --fully_amortized_loss=eg.",
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
        "--natural_map_step",
        type=float,
        default=1.0,
        help="lookahead of the val/natural_map metric: the distance the predicted profile is "
        "moved by one projected ascent step on the true operator, which is zero exactly where "
        "val/residual is. Runs are only comparable on it at a shared value.",
    )
    parser.add_argument(
        "--dim_qkv",
        type=int,
        default=None,
        help="hidden dimension of each attention layer's query/key/value projections, independent of "
        "--dim. Defaults to --dim; the paper's models use 128 at every --dim. Ignored unless "
        "--backbone=nfg_transformer.",
    )
    parser.add_argument("--n_heads", type=int, default=8, help="attention heads in each attention layer.")
    parser.add_argument("--n_layers", type=int, default=6, help="layers in the backbone.")
    parser.add_argument(
        "--n_self_attend_per_block",
        type=int,
        default=1,
        help="action-to-action self-attention layers in each NfgTransformer block (the paper's A). "
        "Ignored unless --backbone is nfg_transformer or zero_sum_transformer.",
    )
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
        default="ste",
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


def build_backbone(config, sample):
    """The backbone named by --backbone, and the readout that matches what it embeds.

    A graph backbone embeds each player from the flat feature vector ``BuildZeroSumFeats`` builds and
    reads every action's logit off that single embedding, which fixes the action count in the
    readout's output width. An action-token backbone embeds each action and scores them one at a
    time, so nothing in the model is sized by the action count.
    """
    dim = config.dim
    n_players, n_actions = int(sample["n_players"]), int(sample["n_actions"])
    # the partially amortized models prepend the query point to the payoffs
    n_feats = n_actions**2 + (n_actions if config.amortization == "partial" else 0)

    if config.backbone == "nash_mlp":
        # the paper's NE-approximator MLP: its [0, 1]-projected parameters bound the embedding
        # scale, so the softmax readout cannot saturate under the vertex-seeking NashApr gradient
        backbone = NashMLPBackbone(n_feats=n_feats, n_players=n_players, dim=dim)
        return backbone, torch.nn.Linear(dim, n_actions)

    transformer_kwargs = dict(
        dim=dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dim_ff=dim * config.dim_ff_mult,
        dropout=0.0,
    )
    if config.backbone == "graphormer":
        backbone = GraphormerBackbone(
            n_feats=n_feats,
            in_degree=sample["in_degree"],
            out_degree=sample["out_degree"],
            spd=sample["spd"],
            **transformer_kwargs,
        )
        return backbone, torch.nn.Linear(dim, n_actions)

    # one token per action, with the query point -- one coordinate per action -- as a token feature
    action_token_backbones = {
        "zero_sum_transformer": partial(
            ZeroSumTransformerBackbone, n_self_attend_per_block=config.n_self_attend_per_block
        ),
        "nfg_transformer": partial(
            NfgTransformerBackbone,
            n_self_attend_per_block=config.n_self_attend_per_block,
            dim_qkv=config.dim_qkv,
            # the reference only ever seeds from zeros, so the fully amortized backbone must own no
            # feature projection at all -- not the identity one a zero-width Linear would give
            partially_amortized=config.amortization == "partial",
        ),
    }
    backbone = action_token_backbones[config.backbone](
        n_feats=1 if config.amortization == "partial" else 0, **transformer_kwargs
    )
    return ActionTokenBackbone(backbone, n_actions), ActionReadout(dim)


if __name__ == "__main__":
    config = get_config()
    L.seed_everything(config.seed, workers=True)

    transforms = Compose(
        [BuildZeroSumFeats(mode=config.amortization), SPDEmbedding(), DegreeEmbedding(), GraphToTensorDict()]
    )
    # the operator dataset contains the equilibrium solutions too, so it works for the fully amortized model
    # dataset = RandomZeroSumOperatorDataset(config.dataset, n_points_per_instance=256, transform=transforms)
    sample = partial(RandomZeroSum.sample_normalized, n_actions=config.n_actions)
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
        sample,
        sample_domain,
        n_instances=config.n_instances,
        n_points_per_instance=16,
        quiet=True,
        transform=transforms,
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
    dim = config.dim
    optimizer_kwargs = dict(
        lr=config.lr,
        start_factor=config.start_factor,
        warmup_epochs=config.warmup_epochs,
        cosine_annealing=bool(config.cosine_annealing),
        natural_map_step=config.natural_map_step,
    )
    backbone, readout = build_backbone(config, sample)
    if config.amortization == "full":
        loss = {
            "ni": NashAprLoss,
            "potential": PotentialLoss,
            "eg": partial(EGLoss, step_size=config.eg_step_size, project=bool(config.eg_projection)),
        }[config.fully_amortized_loss]()
        model = SolutionModel(
            backbone,
            dim=dim,
            n_actions=sample["A"].size(-1),
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            readout=readout,
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
        model = FieldModel(
            backbone,
            dim=dim,
            n_actions=sample["n_actions"],
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            readout=readout,
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
