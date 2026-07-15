"""PUME-based Markov traffic equilibrium as a variational inequality family.

Same road-graph instance and cost-space domain as ``MarkovTrafficEquilibrium`` (see ``traffic.py``),
but the operator is PUME's **excess-supply** field ``E(c) = z(c) - x(c)`` -- the supply flow ``z(c)``
(inverse BPR) minus the recursive-logit demand flow ``x(c)`` -- whose zero is the user equilibrium.
This is a flow-space residual (contrast the cost-space ``costs - bpr(...)`` of the sibling family);
it shares the same equilibrium root but is evaluated straight from the PUME package (``PUMEModel``),
so the asymmetry of a non-potential supply plugs in at a single seam. That is the point of this
family: ``AsymmetricMarkovTrafficEquilibrium`` will subclass it and swap ``InverseBPRSupply`` for
``AsymmetricBPRSupply`` via ``PUMESolver._make_supply``, with everything else unchanged.

This is a standalone ``VariationalInequalityFamily`` (not a subclass of ``MarkovTrafficEquilibrium``),
so it is not tied to that family's cost-space formulation. The pure graph helpers and the conditioning
seam are shared by import; only ``operator`` differs materially.

The raw excess-supply field is a *flow* residual while the domain point is a *cost*, so it is stiff:
its per-link scale (and BPR curvature near free-flow time) span orders of magnitude, forcing tiny
rollout steps. ``precondition=True`` (the default) rescales it by PUME's ``supply_diagonal`` metric
``M = diag(max(s'(c), |x(c)|/|c|, 1))`` and returns ``M^{-1} E`` -- the same diagonal aGRAAL uses --
which pulls the flow residual back toward cost units and bounds the per-link Lipschitz constant. This
reuses PUME's public primitives (``supply_operator.jacobian_diagonal``, ``compute_demand``, and the
``MetricPreconditioner`` container's ``apply_inverse``); only the element-wise floor is assembled here.
The zero is unchanged (``M > 0``), so equilibria, calibration, and projection are all preserved (box
projection under a diagonal metric is plain coordinate clamping). ``precondition=False`` recovers the
PUME-native excess supply. The supply diagonal comes straight from the supply operator, so it composes
with the future asymmetric supply (``diag(A diag(f')) = A_ii f'_i``) with no extra code.
"""

import torch

from l2s_games.envs.base import VariationalInequalityFamily
from l2s_games.envs.traffic import (
    _EDGE_ATTRS,
    _DROPPED_NODE_ATTRS,
    _NOISED_ATTRS,
    _canonicalize,
    load_sioux_falls_base_graph,  # re-exported for callers/tests that build the base graph
)
from l2s_games.operator_count import LocalCounter
from l2s_games.pume_solver import PUMESolver
from l2s_games.transforms import traffic_field_transform
from pume.preconditioning import MetricPreconditioner

__all__ = ["PUMEMarkovTrafficEquilibrium", "load_sioux_falls_base_graph"]


class PUMEMarkovTrafficEquilibrium(VariationalInequalityFamily):
    """Family of single-graph traffic equilibria whose operator is PUME's excess supply."""

    def __init__(
        self,
        base_graph,
        noise_scale=0.2,
        noise_type="normal",
        equilibrium_margin=2.5,
        equilibrium_spread=0.2,
        reference_equilibrium=None,
        reference_spread=None,
        n_stds=3.0,
        operator_counter=None,
        solver_kwargs=None,
        precondition=True,
    ):
        self.base_graph = _canonicalize(base_graph)
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        # Rescale the stiff flow residual by PUME's supply-diagonal metric (see module docstring);
        # False recovers the raw PUME-native excess supply.
        self.precondition = precondition
        # Cumulative point-evaluation counter; the no-op LocalCounter default keeps `operator`
        # branch-free (see traffic.MarkovTrafficEquilibrium for the shared-counter streaming setup).
        self.operator_counter = operator_counter or LocalCounter()
        # Domain-sampling range: reference_equilibrium/spread (calibrated from bootstrap equilibria)
        # center and scale the cost box, else fall back to the shipped reference cost. See sample_domain.
        if reference_equilibrium is not None and reference_spread is not None:
            self.reference_equilibrium = torch.as_tensor(reference_equilibrium, dtype=torch.float32)
            self.reference_spread = torch.as_tensor(reference_spread, dtype=torch.float32)
        else:
            self.reference_equilibrium = self.base_graph.Cost * equilibrium_margin
            self.reference_spread = equilibrium_spread * self.reference_equilibrium
        self.n_stds = n_stds
        # The PUME operator backend: builds the per-destination PUMCM demand models once from the
        # (shared) topology; per-instance supply + demand are rebuilt per operator call in `build_model`.
        self.solver = PUMESolver(self.base_graph, **(solver_kwargs or {}))

    def sample_params(self):
        graph = self.base_graph.clone()
        for attr in _NOISED_ATTRS:
            value = getattr(graph, attr)
            if self.noise_type == "normal":
                factor = 1.0 + torch.randn_like(value) * self.noise_scale
            else:  # uniform in [1 - scale, 1 + scale)
                factor = 1.0 + (2.0 * torch.rand_like(value) - 1.0) * self.noise_scale
            setattr(graph, attr, value * factor.clamp(min=1e-2))
        return graph

    def _excess_supply(self, params, index, cost):
        """Excess supply ``z(c) - x(c)`` (optionally supply-diagonal preconditioned) at one cost vector.

        ``index`` selects the per-instance BPR attrs + demand from a batched ``params`` (rank-2), or is
        ``None`` for a single-instance ``params`` (rank-1). ``build_model`` reuses the shared PUMCM
        structure, so only the light per-instance supply/demand wrapper is assembled here. PUME solves
        on CPU float64 (it cannot run on MPS, which rejects float64), so every input is moved to CPU --
        a no-op for the CPU tensors of data generation, and the move the on-device validation sweep needs.
        """
        row = (lambda name: params[name][index]) if index is not None else (lambda name: params[name])
        pick = lambda name: row(name).cpu()
        free_flow_time, capacity, b, power = (pick(name) for name in _EDGE_ATTRS)
        model = self.solver.build_model(free_flow_time, capacity, b, power, pick("demand"))
        c = cost.cpu().double()
        if not self.precondition:
            return model.compute_excess_supply(c)
        # Split the excess supply so the demand solve is reused by the metric floor (one demand solve,
        # not two): E = z(c) - x(c), then apply PUME's supply-diagonal metric M^{-1} to E.
        supply, demand = model.compute_supply(c), model.compute_demand(c)
        return self._supply_diagonal_metric(model.supply_operator, c, demand).apply_inverse(supply - demand)

    @staticmethod
    def _supply_diagonal_metric(supply_operator, cost, demand):
        """PUME's ``supply_diagonal`` metric ``M = diag(max(s'(c), |x|/|c|, 1))`` as a ``MetricPreconditioner``.

        Mirrors the element-wise floor PUME's aGRAAL solver uses (``pume/solvers/base/agraal.py``),
        built from the public supply-Jacobian diagonal; ``apply_inverse`` supplies the ``M^{-1}`` action.
        """
        s_diag = supply_operator.jacobian_diagonal(cost)
        demand_floor = demand.abs() / cost.abs().clamp(min=1e-8)
        diag = torch.maximum(s_diag, torch.maximum(torch.ones_like(s_diag), demand_floor))
        return MetricPreconditioner(
            mode="supply_diagonal",
            effective_mode="supply_diagonal",
            matrix=None,
            diag=diag,
            inv_diag=1.0 / diag,
            chol=None,
            info={},
        )

    def operator(self, params, costs):
        """PUME excess-supply residual ``z(c) - x(c)`` (zero at equilibrium), one row per cost vector.

        ``costs`` is ``[B, E]`` (one cost vector per instance) or a bare ``[E]`` vector (a batch of
        one). When ``params``' per-edge attrs are rank-2 (``[B, E]``), each cost row is a distinct
        instance (the expert-rollout path); when rank-1 they share one instance (the many-points-per-
        instance dataset path), so one ``PUMEModel`` is reused across the rows. PUME evaluates one cost
        vector at a time, so the batch is a Python loop -- the heavy PUMCM structure is built once in
        ``__init__``; only the supply/demand wrapper is per row.
        """
        costs = torch.as_tensor(costs, dtype=torch.float32)
        device = costs.device  # PUME solves on CPU; hand the result back on the caller's device
        single = costs.dim() == 1
        costs = costs.unsqueeze(0) if single else costs
        batch_size = costs.shape[0]
        self.operator_counter.add(batch_size)  # one point-evaluation per cost vector solved
        per_instance = params["free_flow_time"].dim() == 2
        residuals = torch.stack(
            [self._excess_supply(params, row if per_instance else None, costs[row]) for row in range(batch_size)]
        ).float().to(device)
        return residuals.squeeze(0) if single else residuals

    def project(self, params, costs):
        # Subscript access works for both a single graph and the dense batch dict (validation sweep).
        return torch.clamp(torch.as_tensor(costs, dtype=torch.float32), min=params["free_flow_time"])

    def params_from_batch(self, batch):
        """The operator's params from the dense model batch: point ``edge_index`` at the physical
        topology (the batch's own ``edge_index`` is the Graphormer line graph). Per-edge attrs pass
        through as-is; ``build_model`` reads only the BPR attrs + demand, so the topology field is
        carried for interface parity with a raw instance graph.
        """
        return {**batch, "edge_index": batch["physical_edge_index"]}

    def initial_point(self, batch):
        """Rollout start for the validation sweep: the example's uniformly sampled cost point ``[B, E]``
        (feasible by construction; ``project`` still clamps)."""
        return batch["cost"]

    @staticmethod
    def calibrate_range(instances):
        """Per-edge ``(reference_equilibrium, reference_spread)`` from a set of solved instances -- the
        per-edge mean and std of their cached ``equilibrium_cost``. Feed the pair into ``__init__`` so
        ``sample_domain`` draws around where the equilibria actually are."""
        eq = torch.stack([instance.equilibrium_cost.float() for instance in instances])  # [N, E]
        return eq.mean(dim=0), eq.std(dim=0)

    def sample_domain(self, graph, n):
        """Feasible cost points drawn uniformly per edge over the calibrated box.

        Each edge's cost is drawn uniformly in ``[free_flow_time, ceiling]`` with per-edge
        ``ceiling = reference_equilibrium + n_stds * reference_spread``, spanning the segment the
        rollout traverses from the free-flow-time start up to (a few sigma above) the equilibrium.
        ``hi`` is floored at ``free_flow_time`` so every sample is feasible (``>= free_flow_time``).
        """
        fft = graph.free_flow_time
        ceiling = self.reference_equilibrium + self.n_stds * self.reference_spread
        hi = torch.maximum(ceiling, fft)
        return fft + torch.rand(n, graph.num_edges) * (hi - fft)

    def model_input(self, graph, cost):
        """Raw input item: the instance graph with the domain point attached as ``.cost``.

        Featurization is deferred to ``transform`` (lazy per access; see ``transforms.py``). Unused
        float64 node-coordinate attrs are dropped so they never reach the collated batch.
        """
        item = graph.clone()
        for attr in _DROPPED_NODE_ATTRS:
            if attr in item:
                del item[attr]
        item.cost = torch.as_tensor(cost, dtype=torch.float32)
        return item

    @property
    def transform(self):
        return traffic_field_transform()

    @staticmethod
    def batched_field_input(batch, costs, normalizer):
        """Splice a batch of costs ``[B, E]`` into the dense batch's inputs for the learned field.

        The domain point is the cost, in ``feats`` column 0 (see ``BuildTrafficEdgeData``).
        Standardization is per-column affine, so overwriting that column with the standardized cost is
        exact; concatenation (not in-place assignment) keeps it jacrev-transparent.
        """
        costs = torch.as_tensor(costs, dtype=torch.float32)
        standardized_cost = (costs - normalizer.input.mean[0]) / normalizer.input.std[0]
        feats = torch.cat([standardized_cost.unsqueeze(-1), batch["feats"][..., 1:]], dim=-1)
        return {**batch, "feats": feats}

    @staticmethod
    def collate_fn(items):
        """Dense-batch same-topology line graphs: stack every per-item tensor, share the topologies.

        The Graphormer uses dense attention over one fixed topology, so a batch is stacked tensors plus
        the shared (line-graph) ``edge_index`` and physical ``physical_edge_index`` -- both identical
        across the batch, so stored once un-stacked. The real-unit BPR/demand params survive the stack,
        which the operator's per-instance ``build_model`` needs.
        """
        shared = ("edge_index", "physical_edge_index")  # one topology across the batch -- store once
        batch = {key: items[0][key] for key in shared}
        for key in items[0].keys():
            value = items[0][key]
            if key not in shared and isinstance(value, torch.Tensor):
                batch[key] = torch.stack([item[key] for item in items])
        return batch
