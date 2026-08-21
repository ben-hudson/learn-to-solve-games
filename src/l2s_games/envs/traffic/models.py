import torch

from torch.nn import MSELoss
from l2s_games.algorithms import SimpleProjection
from l2s_games.envs.traffic.losses import PotentialLoss, WardropAprLoss
from l2s_games.envs.zero_sum import AmortizedModel, NormHuberLoss, NormLoss

from .game import NonPotentialCongestion, COST_UPPER_BOUND
from .utils import dist_to_normal_cone


class TrafficSolutionModel(AmortizedModel):
    def __init__(
        self, backbone, dim, feat_mean, feat_scale, target_mean, target_scale, pume_mapping, loss="potential", **kwargs
    ):
        n_actions = 1
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        # both losses are self-supervised in raw cost units: potential descends the dual potential's
        # exact gradient (the excess supply), wardrop the hardest one-step deviation gain
        self.loss = {"potential": PotentialLoss(), "wardrop": WardropAprLoss()}[loss]
        self.pume_mapping = pume_mapping
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def unnormalize_targets(self, targets: torch.Tensor):
        return targets * self.target_scale + self.target_mean

    def predict_sol(self, batch):
        feats = self.normalize_feats(batch["feats"])
        return self.readout(self.backbone(feats, batch["in_degree"], batch["out_degree"], batch["spd"])).squeeze(-1)

    def training_step(self, batch, batch_idx):
        # the network predicts in normalized target space; the loss evaluates the games'
        # operators, which live in raw cost units
        pred_costs = self.unnormalize_targets(self.predict_sol(batch))
        # PUME operates in float64, which MPS does not support, so the games live on the CPU
        games = [NonPotentialCongestion.from_data(self.pume_mapping, sample) for sample in batch.cpu().unbind(0)]
        loss = self.loss(games, pred_costs)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_costs = self.unnormalize_targets(self.predict_sol(batch)).cpu()
        games = [NonPotentialCongestion.from_data(self.pume_mapping, sample) for sample in batch.cpu().unbind(0)]
        residuals = []
        for game, costs in zip(games, pred_costs):
            lower, upper = (torch.as_tensor(bound) for bound in game.cost_bounds)
            excess_demand = -game.operator(costs)
            residuals.append(dist_to_normal_cone(excess_demand, costs, lower, upper))
        self.log("val/residual", torch.stack(residuals).mean())


class TrafficFieldModel(AmortizedModel):
    # step_size and steps are the rollout parameters validated in
    # tests/test_traffic.py::test_projection_converges
    def __init__(
        self,
        backbone,
        dim,
        feat_mean,
        feat_scale,
        point_min,
        point_max,
        target_scale,
        pume_mapping,
        loss="norm",
        huber_delta=1.0,
        step_size=0.25,
        steps=200,
        **kwargs,
    ):
        n_actions = 1
        super().__init__(backbone, dim, n_actions, feat_mean, feat_scale, **kwargs)

        # the model predicts in normalized target space (one global scale), so every loss measures
        # the error in target-std units. huber_delta caps the per-sample gradient norm at
        # min(||r||, delta), so it is only a knob between the floor and where training starts.
        self.loss = {"mse": MSELoss(), "norm": NormLoss(), "huber": NormHuberLoss(huber_delta)}[loss]
        self.pume_mapping = pume_mapping
        self.step_size = step_size
        self.steps = steps

        self.register_buffer("point_min", torch.as_tensor(point_min, dtype=torch.float32))
        self.register_buffer("point_max", torch.as_tensor(point_max, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def normalize_point(self, point):
        return (point - self.point_min) / (self.point_max - self.point_min)

    def training_step(self, batch, batch_idx):
        # fold the sampled points into the batch dimension: every point is an
        # independent evaluation on the same graph
        n_points_per_instance = batch["point"].size(1)
        feats = self.normalize_feats(batch["feats"].repeat_interleave(n_points_per_instance, dim=0))
        points = self.normalize_point(batch["point"].flatten(0, 1))
        feats_and_point = torch.cat([feats, points.unsqueeze(-1)], dim=-1)
        in_degree = batch["in_degree"].repeat_interleave(n_points_per_instance, dim=0)
        out_degree = batch["out_degree"].repeat_interleave(n_points_per_instance, dim=0)
        spd = batch["spd"].repeat_interleave(n_points_per_instance, dim=0)
        # one global scale for the preconditioned operator (isotropic, so its direction is untouched)
        target = batch["preconditioned_operator"].flatten(0, 1) / self.target_scale

        prediction = self.readout(self.backbone(feats_and_point, in_degree, out_degree, spd)).squeeze(-1)
        loss = self.loss(prediction, target)
        self.log("train/loss", loss)
        return loss

    def solve(self, batch):
        # the learned field stands in for the true operator: one SimpleProjection rollout per game
        # with the test's parameters, from the same initial point. The network predicts the
        # *normalized* preconditioned excess supply, so target_scale restores the raw cost units
        # the step size was tuned for, and the ascent field is its negation (excess demand pushes
        # costs up).
        def preconditioned_excess_demand(costs):
            feats = self.normalize_feats(batch["feats"])
            points = self.normalize_point(costs)
            feats_and_point = torch.cat([feats, points.unsqueeze(-1)], dim=-1)
            prediction = self.readout(
                self.backbone(feats_and_point, batch["in_degree"], batch["out_degree"], batch["spd"])
            )
            return -prediction.squeeze(-1) * self.target_scale

        def project_costs(costs):
            return costs.clamp(min=batch["free_flow_time"]).clamp(max=COST_UPPER_BOUND)

        algorithm = SimpleProjection(self.step_size, preconditioned_excess_demand, project_costs)
        costs = batch["free_flow_time"] * 1.1
        for _ in range(self.steps):
            costs = algorithm.step(costs)
        return costs

    def validation_step(self, batch, batch_idx):
        # stationarity of the rollout endpoint under the true operator: zero exactly at the
        # equilibrium, and the same raw-units residual the test thresholds at 1e-3, so the logged
        # value is directly comparable
        # PUME operates in float64, which MPS does not support, so the games live on the CPU
        solved_costs = self.solve(batch).cpu()
        games = [NonPotentialCongestion.from_data(self.pume_mapping, sample) for sample in batch.cpu().unbind(0)]
        residuals = []
        for game, costs in zip(games, solved_costs):
            lower, upper = (torch.as_tensor(bound) for bound in game.cost_bounds)
            excess_demand = -game.operator(costs)
            residuals.append(dist_to_normal_cone(excess_demand, costs, lower, upper))
        self.log("val/residual", torch.stack(residuals).mean())
