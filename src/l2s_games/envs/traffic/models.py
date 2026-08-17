import torch

from l2s_games.algorithms import SimpleProjection
from l2s_games.envs.zero_sum import AmortizedModel, NormLoss

from .game import PotentialCongestion, COST_UPPER_BOUND
from .utils import dist_to_normal_cone


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
