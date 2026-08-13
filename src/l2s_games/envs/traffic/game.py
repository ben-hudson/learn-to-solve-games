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


def sparse_incidence_matrix(
    edge_index: torch.Tensor,
    n_nodes: int = None,
    n_edges: int = None,
    dtype: torch.dtype = torch.float32,
):
    if n_nodes is None:
        n_nodes = torch.unique(edge_index).size(0)
    if n_edges is None:
        n_edges = edge_index.size(1)

    edge_number = torch.arange(n_edges, device=edge_index.device)

    tails = -torch.ones_like(edge_index[0])
    heads = torch.ones_like(edge_index[1])

    tail_coords = torch.stack((edge_index[0], edge_number))
    head_coords = torch.stack((edge_index[1], edge_number))

    indices = torch.cat((tail_coords, head_coords), dim=-1)
    values = torch.cat((tails, heads), dim=-1).to(dtype)

    return torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_edges))


DEFAULT_OUTER_SOLVER_OPTIONS = {
    "oracle_type": "aa1",
    "base_method": "agraal",
    "max_iterations": 2000,
    "convergence_tolerance": 1e-3,
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
class PUMENetwork:
    edge_index: torch.Tensor
    mdps: List[PUMCM]
    reward_mapping: RewardMapping
    flow_mapping: FlowMapping
    demand: StackedPUMCMDemandLoader

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
        demand = StackedPUMCMDemandLoader(
            pumcm_models=mdps,
            flow_mappings=flow_mapping.B_sa_l,
            reward_provider=lambda costs, _dest_idx: reward_mapping.rewards(costs),
            initial_states_list=initial_states_list,
            reward_invariant=True,
        )

        return cls(edge_index, mdps, reward_mapping, flow_mapping, demand)


class PotentialCongestion(PUMEModel):
    def __init__(
        self,
        network: PUMENetwork,
        free_flow_time: torch.Tensor,
        capacity: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ):
        # these are the tensors we need for to_data
        self.edge_index = network.edge_index
        self.free_flow_time = free_flow_time
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        supply = InverseBPRSupply(
            free_flow_time=self.free_flow_time,
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta,
            eps=1e-6,
        )

        cost_lower = self.free_flow_time.numpy()
        super().__init__(
            pumcm_models=network.mdps,
            supply_func=None,
            supply=supply,
            reward_mapping=network.reward_mapping,
            flow_mapping=network.flow_mapping,
            cost_bounds=(cost_lower, np.full_like(cost_lower, 700.0)),  # exp(-700) underflows
            demand_loader=network.demand,
        )

    def solve(self, initial_costs=None, solver=None, solver_options=None, method="meta"):
        if initial_costs is None:
            initial_costs = self.free_flow_time * 1.1
        if solver_options is None:
            solver_options = DEFAULT_OUTER_SOLVER_OPTIONS
        solve_info = super().solve(c_initial=initial_costs, solver=solver, method=method, options=solver_options)
        return solve_info["cost"], solve_info

    def to_data(self) -> Data:
        return Data(
            edge_index=self.edge_index,
            free_flow_time=self.free_flow_time,
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta,
        )

    @classmethod
    def from_data(cls, network, data: Data):
        tensors = data.multi_get_tensor(["free_flow_time", "capacity", "alpha", "beta"])
        return cls.__init__(network, *tensors)


class NonPotentialCongestion:
    pass
