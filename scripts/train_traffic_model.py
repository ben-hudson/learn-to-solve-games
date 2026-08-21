import argparse
from pathlib import Path
import lightning as L
import os
import tntp
import torch
from torch_geometric.utils import from_networkx
import wandb

from l2s_games.envs.traffic import (
    BuildTrafficFeats,
    GraphToTensorDict,
    PUMEMapping,
    TrafficFieldModel,
    TrafficOperatorDataset,
    TrafficSolutionModel,
)
from l2s_games.envs.traffic.streams import TrafficOperatorStream
from l2s_games.models.graphormer import GraphormerBackbone
from l2s_games.transforms import DegreeEmbedding, SPDEmbedding
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import Compose, LineGraph
from torch.utils.data import random_split, DataLoader


def load_base_graph(root: Path):
    network = tntp.convert_to_networkx(
        tntp.read_node_file(root / "SiouxFalls_node.tntp", index_col="Node", x_col="X", y_col="Y", crs="wgs84"),
        tntp.read_net_file(root / "SiouxFalls_net.tntp", crs="wgs84"),
    )
    demand_table = tntp.read_demand_file(root / "SiouxFalls_trips.tntp")

    base_graph = from_networkx(network)
    base_graph.free_flow_time = base_graph.free_flow_time.float()

    node_list = list(network.nodes)
    demand_table = demand_table.reindex(index=node_list, columns=node_list)

    demand_scale = 1000
    base_graph.demand_matrix = torch.as_tensor(demand_table.values) / demand_scale
    base_graph.capacity = base_graph.capacity / demand_scale
    return base_graph


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--amortization", type=str, choices=["full", "partial"], default="partial")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cosine_annealing", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fully_amortized_loss", type=str, choices=["potential", "wardrop"], default="potential")
    parser.add_argument("--gradient_clip_val", type=float, default=0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--logger", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_instances_per_epoch", type=int, default=512)
    parser.add_argument("--n_points_per_instance", type=int, default=128)
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
            GraphToTensorDict(),
        ]
    )
    # the operator dataset contains the equilibrium solutions too, so it works for the fully amortized model
    dataset = TrafficOperatorDataset(config.dataset, transform=transforms)

    cal_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    # torch.stack collates the per-instance TensorDicts into a batched TensorDict
    cal_loader = DataLoader(cal_dataset, batch_size=config.batch_size, collate_fn=torch.stack)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=torch.stack)

    # every instance shares the network and OD demand, and only the untransformed instances keep
    # the road network's edge_index (LineGraph rewrites it), so the mapping is rebuilt from raw
    base_graph = load_base_graph(Path("raw_data/sioux_falls"))
    pume_mapping = PUMEMapping.from_edges_and_demand(base_graph.edge_index, base_graph.demand_matrix)
    train_dataset = TrafficOperatorStream(
        pume_mapping,
        base_graph,
        cal_dataset,
        n_points_per_instance=config.n_points_per_instance,
        n_instances=config.n_instances_per_epoch,
        kappa=cal_dataset[0]["kappa"],
        solve=False,
        quiet=True,
        transform=transforms,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=torch.stack)

    sample = dataset[0]

    feat_scaler = StandardScaler()
    operator_scaler = StandardScaler(with_mean=False)
    solution_scaler = StandardScaler()
    for batch in cal_loader:
        # TODO: how is the point itself normalized?
        feat_scaler.partial_fit(batch["feats"].reshape(-1, batch["feats"].size(-1)))
        # a single column, so the fit yields one global scale: an isotropic rescale of the
        # operator field that preserves its direction
        operator_scaler.partial_fit(batch["preconditioned_operator"].reshape(-1, 1))
        solution_scaler.partial_fit(batch["eq"].reshape(-1, 1))

    dim = 128
    optimizer_kwargs = dict(
        lr=config.lr,
        start_factor=config.start_factor,
        warmup_epochs=config.warmup_epochs,
        cosine_annealing=bool(config.cosine_annealing),
    )
    if config.amortization == "full":
        backbone = GraphormerBackbone(
            n_feats=sample["feats"].size(-1),
            in_degree=sample["in_degree"],
            out_degree=sample["out_degree"],
            spd=sample["spd"],
            dim=dim,
            n_heads=8,
            n_layers=6,
            dim_ff=dim * 2,
            dropout=0.0,
        )
        model = TrafficSolutionModel(
            backbone,
            dim=dim,
            feat_mean=feat_scaler.mean_,
            feat_scale=feat_scaler.scale_,
            target_mean=solution_scaler.mean_,
            target_scale=solution_scaler.scale_,
            pume_mapping=pume_mapping,
            loss=config.fully_amortized_loss,
            **optimizer_kwargs,
        )
    else:
        backbone = GraphormerBackbone(
            n_feats=sample["feats"].size(-1),
            in_degree=sample["in_degree"],
            out_degree=sample["out_degree"],
            spd=sample["spd"],
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
            loss=config.partially_amortized_loss,
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
            # monitor="train/loss",
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
