import argparse
import lightning as L
import os
import torch
import wandb


from l2s_games.algorithms import SimpleProjection
from l2s_games.envs.traffic import COST_UPPER_BOUND, PotentialCongestion, PUMEMapping, dist_to_normal_cone
from l2s_games.envs.traffic.datasets import TrafficOperatorDataset
from l2s_games.envs.zero_sum import GraphToTuple, AmortizedModel
from l2s_games.envs.zero_sum.losses import NormLoss
from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.transforms import DegreeEmbedding, SPDEmbedding
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import Compose, BaseTransform, LineGraph
from torch.utils.data import random_split, DataLoader


class BuildTrafficFeats(BaseTransform):
    def __init__(self, mode=None):
        super().__init__()

        assert mode in ["full", "partial"], f"Mode must be 'full' or 'partial', got {mode}."
        self.mode = mode

    def forward(self, data):
        feats = torch.stack([data.free_flow_time, data.capacity], dim=-1)

        if self.mode == "partial":
            n_points_per_instance = data.point.size(0)
            feats = feats.expand(n_points_per_instance, -1, -1)
            # here we add the point because it is not constrained to the simplex
            points = data.point.unsqueeze(-1)
            data.feats = torch.cat([feats, points], dim=-1)
        else:
            data.feats = feats

        return data


class TrafficFieldModel(AmortizedModel):
    # step_size and steps are the rollout parameters validated in
    # tests/test_traffic.py::test_projection_converges
    def __init__(
        self, backbone, dim, feat_mean, feat_scale, target_scale, pume_mapping, step_size=0.25, steps=200, **kwargs
    ):
        n_actions = 1
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        self.loss = NormLoss()
        self.pume_mapping = pume_mapping
        self.step_size = step_size
        self.steps = steps
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def training_step(self, batch, batch_idx):
        batch_size, n_points_per_instance, n_edges = batch.point.shape
        batch = batch._replace(
            feats=batch.feats.flatten(0, 1),
            in_degree=batch.in_degree.repeat_interleave(n_points_per_instance, dim=0),
            out_degree=batch.out_degree.repeat_interleave(n_points_per_instance, dim=0),
            point=batch.point.flatten(0, 1),
            preconditioned_operator=batch.preconditioned_operator.flatten(0, 1),
            spd=batch.spd.repeat_interleave(n_points_per_instance, dim=0),
        )
        # normalize: per-column mean/scale for the feats, one global scale for the
        # preconditioned operator (isotropic, so its direction is untouched)
        batch = batch._replace(
            feats=self.normalize_feats(batch.feats),
            preconditioned_operator=batch.preconditioned_operator / self.target_scale,
        )
        prediction = self.readout(self.backbone(batch.feats, batch.in_degree, batch.out_degree, batch.spd)).squeeze(-1)
        loss = self.loss(prediction, batch.preconditioned_operator)
        self.log("train/loss", loss)
        return loss

    def solve(self, batch):
        # the learned field stands in for the true operator: one SimpleProjection rollout per game
        # with the test's parameters, from the same initial point. The network predicts the
        # *normalized* preconditioned excess supply, so target_scale restores the raw cost units
        # the step size was tuned for, and the ascent field is its negation (excess demand pushes
        # costs up).
        def preconditioned_excess_demand(costs):
            feats = self.normalize_feats(torch.stack([batch.free_flow_time, batch.capacity, costs], dim=-1))
            prediction = self.readout(self.backbone(feats, batch.in_degree, batch.out_degree, batch.spd)).squeeze(-1)
            return -prediction * self.target_scale

        def project_costs(costs):
            return costs.clamp(min=batch.free_flow_time).clamp(max=COST_UPPER_BOUND)

        algorithm = SimpleProjection(self.step_size, preconditioned_excess_demand, project_costs)
        costs = batch.free_flow_time * 1.1
        for _ in range(self.steps):
            costs = algorithm.step(costs)
        return costs

    def validation_step(self, batch, batch_idx):
        # stationarity of the rollout endpoint under the true operator: zero exactly at the
        # equilibrium, and the same raw-units residual the test thresholds at 1e-3, so the logged
        # value is directly comparable
        # PUME operates in float64, which MPS does not support, so the games live on the CPU
        solved_costs = self.solve(batch).cpu()
        residuals = []
        for free_flow_time, capacity, alpha, beta, costs in zip(
            batch.free_flow_time.cpu(), batch.capacity.cpu(), batch.alpha.cpu(), batch.beta.cpu(), solved_costs
        ):
            game = PotentialCongestion(self.pume_mapping, free_flow_time, capacity, alpha, beta)
            lower, upper = (torch.as_tensor(bound) for bound in game.cost_bounds)
            excess_demand = -game.operator(costs)
            residuals.append(dist_to_normal_cone(excess_demand, costs, lower, upper))
        self.log("val/residual", torch.stack(residuals).mean())


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--amortization", type=str, choices=["full", "partial"], default="partial")
    parser.add_argument("--batch_size", type=int, default=4)
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
        [
            BuildTrafficFeats(mode=config.amortization),
            LineGraph(force_directed=True),
            SPDEmbedding(),
            DegreeEmbedding(),
            GraphToTuple(),
        ]
    )
    # the operator dataset contains the equilibrium solutions too, so it works for the fully amortized model
    dataset = TrafficOperatorDataset(config.dataset, transform=transforms)
    # every instance shares the network and OD demand, and only the untransformed instances keep
    # the road network's edge_index (LineGraph rewrites it), so the mapping is rebuilt from raw
    instance = dataset.load_instances()[0]
    pume_mapping = PUMEMapping.from_edges_and_demand(instance.edge_index, instance.demand_matrix)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    sample = dataset[0]

    feat_scaler = StandardScaler()
    operator_scaler = StandardScaler(with_mean=False)
    for batch in train_loader:
        # TODO: now the issue is we have different feats that mean different things but we still want to capture the relative costs of the edges within a network
        feat_scaler.partial_fit(batch.feats.reshape(-1, batch.feats.size(-1)))
        # a single column, so the fit yields one global scale: an isotropic rescale of the
        # operator field that preserves its direction
        operator_scaler.partial_fit(batch.preconditioned_operator.reshape(-1, 1))

    dim = 128
    optimizer_kwargs = dict(
        lr=config.lr,
        start_factor=config.start_factor,
        warmup_epochs=config.warmup_epochs,
        cosine_annealing=bool(config.cosine_annealing),
    )
    if config.amortization == "full":
        raise Exception()
    else:
        backbone = GraphormerBackbone(
            n_feats=sample.feats.size(-1),
            in_degree=sample.in_degree,
            out_degree=sample.out_degree,
            spd=sample.spd,
            dim=dim,
            n_heads=8,
            n_layers=6,
            dim_ff=dim * 2,
            dropout=0.0,
        )
        model = TrafficFieldModel(
            backbone,
            dim=dim,
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            target_scale=operator_scaler.scale_,
            pume_mapping=pume_mapping,
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
