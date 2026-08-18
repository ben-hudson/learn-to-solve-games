from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from scipy import sparse
from torch_geometric.data import Data

from pumcm import PUMCM, ModifiedPolicyIteration, RelativeEntropy
from pumcm.utils.structure_builder import build_structures_batch
from pume import PUMEModel, StackedPUMCMDemandLoader
from pume.operators import InverseBPRSupply
from utils.mapping import FlowMapping, RewardMapping

from .utils import sparse_incidence_matrix

COST_UPPER_BOUND = 700.0  # exp(-700) underflows

DEFAULT_OUTER_SOLVER_OPTIONS = {
    "oracle_type": "aa1",
    "base_method": "agraal",
    "meta": {
        "safeguard_factor": 0.9,
        "safeguard_tau": 0.999,
        "restart_period": 20,
        "adaptive_eta": True,
    },
    "oracle": {
        "memory": 10,
        "regularization": 1e-6,
        "attempt_period": 5,
        "stall_ratio": 0.0,
    },
    "base_options": {
        "metric_mode": "supply_diagonal",
        "initial_stepsize": 5e-2,
    },
}

DEFAULT_INNER_SOLVER_OPTIONS = {
    "m": 1,
    "max_iter": 3000,
    "tol": 1e-7,
}


@dataclass
class PUMEMapping:
    edge_index: torch.Tensor
    mdps: List[PUMCM]
    reward_mapping: RewardMapping
    flow_mapping: FlowMapping
    demand_loader: StackedPUMCMDemandLoader
    demand_matrix: torch.Tensor

    @classmethod
    def from_edges_and_demand(cls, edge_index: torch.Tensor, demand_matrix: torch.Tensor, **inner_solver_kwargs):
        n_nodes = demand_matrix.size(0)
        n_edges = edge_index.size(1)

        incidence = sparse_incidence_matrix(edge_index, n_nodes=n_nodes, n_edges=n_edges).coalesce()
        incidence_scipy = sparse.coo_matrix(
            (incidence.values().numpy(), incidence.indices().numpy()),
            shape=incidence.shape,
        ).tocsr()
        states_to_actions = (incidence_scipy.copy() == -1).astype(float)
        actions_to_states = (incidence_scipy.copy() == 1).astype(float).T.tocsr()

        is_dest = demand_matrix.sum(dim=0) > 0
        dests = torch.nonzero(is_dest).flatten().tolist()
        structs = build_structures_batch(
            lambda_full=states_to_actions,
            P_full=actions_to_states,
            termination_states=dests,
            gamma=1.0,  # discount factor
        )
        choice_model = RelativeEntropy(normalization="cardinality_normalization")
        solver_kwargs = inner_solver_kwargs if len(inner_solver_kwargs) > 0 else DEFAULT_INNER_SOLVER_OPTIONS
        mdps = [
            PUMCM(
                structure=struct,
                choice_model=choice_model,
                solver=ModifiedPolicyIteration(**solver_kwargs),
            )
            for struct in structs
        ]

        reward_mapping = RewardMapping(
            A_sa_l=sparse.identity(n_edges, format="csr"),
            base_u_sa=torch.zeros(n_edges, dtype=torch.float64),
        )
        flow_mapping = FlowMapping(B_sa_l=sparse.identity(n_edges, format="csr"))

        initial_states_list = [demand_matrix[:, dest].to(torch.float64) for dest in dests]
        demand_loader = StackedPUMCMDemandLoader(
            pumcm_models=mdps,
            flow_mappings=flow_mapping.B_sa_l,
            reward_provider=lambda costs, _dest_idx: reward_mapping.rewards(costs),
            initial_states_list=initial_states_list,
            reward_invariant=True,
        )

        return cls(edge_index, mdps, reward_mapping, flow_mapping, demand_loader, demand_matrix)


class PotentialCongestion(PUMEModel):
    def __init__(
        self,
        network: PUMEMapping,
        free_flow_time: torch.Tensor,
        capacity: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ):
        self.edge_index = network.edge_index
        self.demand_matrix = network.demand_matrix

        supply = InverseBPRSupply(
            free_flow_time=free_flow_time,
            capacity=capacity,
            alpha=alpha,
            beta=beta,
            eps=1e-6,
        )

        cost_lower = free_flow_time.numpy()
        super().__init__(
            pumcm_models=network.mdps,
            supply_func=None,
            supply=supply,
            reward_mapping=network.reward_mapping,
            flow_mapping=network.flow_mapping,
            cost_bounds=(cost_lower, np.full_like(cost_lower, COST_UPPER_BOUND)),
            demand_loader=network.demand_loader,
        )

    def _solve(self, initial_cost=None, solver=None, method="meta", max_iters=1000, tol=1e-3):
        if initial_cost is None:
            initial_cost = self.free_flow_time * 1.1
        options = dict(**DEFAULT_OUTER_SOLVER_OPTIONS, max_iterations=max_iters, convergence_tolerance=tol)
        solve_info = super().solve(c_initial=initial_cost, solver=solver, method=method, options=options)
        self.eq, self.eq_info = solve_info["cost"], solve_info

    def solve(self, **kwargs):
        self._solve(**kwargs)
        return self.eq, self.eq_info

    def operator(self, costs):
        return self.compute_excess_supply(costs)

    def operator_and_preconditioner(self, costs: torch.Tensor, eps: float = 1e-8):
        """``(excess_supply, preconditioner_diagonal)``: the raw excess supply
        ``E(c) = z(c) - x(c)`` and the supply-diagonal metric
        ``M = diag(max(s'(c), |x(c)|/|c|, 1))`` -- the same element-wise floor PUME's aGRAAL
        solver builds.

        Neither tensor is rescaled here: the caller applies ``excess_supply / diagonal`` to get
        the preconditioned field ``M^{-1} E``, and keeps the raw excess supply for the
        monotonicity constraint, which must be stated about it -- the preconditioned field is not
        monotone. The raw excess supply is a flow residual at a cost point, so it is stiff: the
        steep coordinates of the inverse BPR supply curve span orders of magnitude, and the demand
        floor catches the edges where demand, not supply sensitivity, dominates (making the metric
        invariant to the demand's units). ``M^{-1} E`` pulls the residual back toward cost units,
        and ``M > 0``, so the zero (the equilibrium) is unchanged. The demand the floor needs
        comes out of the same solve the residual already needs, so it is free.
        """
        costs = costs.double()  # PUME operates in float64
        demand = self.compute_demand(costs)
        excess_supply = self.compute_supply(costs) - demand

        supply_diagonal = self.supply_operator.jacobian_diagonal(costs)
        demand_floor = demand.abs() / costs.abs().clamp(min=eps)
        precond = torch.maximum(supply_diagonal, torch.maximum(torch.ones_like(supply_diagonal), demand_floor))
        return excess_supply, precond

    @property
    def free_flow_time(self):
        return self.supply_operator.t0

    @property
    def n_nodes(self):
        return self.pumcm_models[0].structure.num_states_full

    @property
    def n_edges(self):
        return self.edge_index.size(1)

    def to_data(self, dtype=torch.float32) -> Data:
        # the solver works in float64, but the learning stack expects float32
        return Data(
            edge_index=self.edge_index,
            num_nodes=self.n_nodes,
            num_edges=self.n_edges,
            free_flow_time=self.free_flow_time.to(dtype),
            capacity=self.supply_operator.cap.to(dtype),
            alpha=self.supply_operator.alpha.to(dtype),
            beta=self.supply_operator.beta.to(dtype),
            demand_matrix=self.demand_matrix.to(dtype),
            eq=self.eq.to(dtype),
        )

    @classmethod
    def from_data(cls, network, data):
        return cls(network, data["free_flow_time"], data["capacity"], data["alpha"], data["beta"])


class NonPotentialCongestion(PotentialCongestion):
    pass
