"""
pume_solver.py

Reference user-equilibrium solver for the Markov traffic VI, built on the PUME (Perturbed Utility
Markovian Equilibrium) package. It solves the cost-space residual ``costs - bpr(demand_flow(-costs))``
to zero on a PyG road graph with PUME's aGRAAL base under Anderson (type-1) meta-acceleration and
supply-diagonal preconditioning -- more robust on noised instances than the damped-fixed-point
torchdeq iteration it replaces.

Construct one solver per graph *structure* (edge_index + sink nodes); the per-destination PUMCM
models are the expensive part and are built once here. Then call ``solve(instance)`` for any
instance sharing that structure, returning equilibrium ``(costs, flows)`` as float tensors in real
units. Follows PUME's own SiouxFalls equilibrium example rather than reconstructing incidence via a
separate route-choice dependency: the state->action / action->state matrices come straight off
``edge_index`` and the destination structures are built with the batched ``build_structures_batch``.
"""

import numpy as np
import scipy.sparse
import torch
import torch_geometric.data

from pumcm import (
    PUMCM,
    FlowMapping,
    ModifiedPolicyIteration,
    RecursiveLogit,
    RewardMapping,
    build_structures_batch,
)
from pume import PUMEModel, StackedPUMCMDemandLoader
from pume.operators import InverseBPRSupply


class PUMESolver:
    """PUME-based reference equilibrium solver for a fixed traffic-graph structure."""

    def __init__(
        self,
        graph: torch_geometric.data.Data,
        choice_model=None,
        inner_max_iter=3000,
        inner_tol=1e-7,
        outer_max_iter=500,
        outer_tol=1e-1,
    ):
        choice_model = choice_model if choice_model is not None else RecursiveLogit()
        self._outer_max_iter = outer_max_iter
        self._outer_tol = outer_tol

        num_links = graph.num_edges
        source_nodes, target_nodes = graph.edge_index.numpy()
        links = np.arange(num_links)
        ones = np.ones(num_links)
        states_to_actions = scipy.sparse.coo_matrix(
            (ones, (source_nodes, links)), shape=(graph.num_nodes, num_links)
        ).tocsr()
        actions_to_states = scipy.sparse.coo_matrix(
            (ones, (links, target_nodes)), shape=(num_links, graph.num_nodes)
        ).tocsr()

        destination_nodes = graph.sink_node_mask.argmax(dim=-1).tolist()
        structures = build_structures_batch(
            lambda_full=states_to_actions,
            P_full=actions_to_states,
            termination_states=destination_nodes,
            gamma=1.0,
        )
        self._pumcm_models = [
            PUMCM(
                structure=structure,
                choice_model=choice_model,
                solver=ModifiedPolicyIteration(m=1, max_iter=inner_max_iter, tol=inner_tol, verbose=False),
            )
            for structure in structures
        ]

        # TNTP identity mappings: cost -> reward is u = -c, occupancy -> flow is the identity.
        self._reward_mapping = RewardMapping(
            A_sa_l=scipy.sparse.identity(num_links, format="csr"),
            base_u_sa=torch.zeros(num_links, dtype=torch.float64),
        )
        self._flow_mapping = FlowMapping(B_sa_l=scipy.sparse.identity(num_links, format="csr"))

        # One persistent demand loader for the operator (excess-supply field) path, built once from
        # the base-graph OD. It is reused across every operator evaluation; the per-instance OD is
        # threaded in per call via `compute_demand(initial_states_list=...)` (see the traffic family's
        # `_excess_supply`), so this loader's baked OD only fixes the per-destination shapes the
        # override validates against. Reusing one loader is what fixes the leak: PUME caches the
        # torch-CSR of each demand loader's stacked flow mapping in a module-global dict keyed by the
        # mapping's `id()` with no eviction, so a fresh loader per eval leaked unboundedly.
        self._demand_loader = self._make_demand_loader(graph.demand)

    def _reward_provider(self, costs, _destination_idx):
        """cost -> full state-action reward (destination-invariant TNTP mapping ``u = -c``)."""
        return self._reward_mapping.rewards(costs)

    def _make_demand_loader(self, demand) -> StackedPUMCMDemandLoader:
        """A stacked demand loader for one OD table (per-destination initial states)."""
        return StackedPUMCMDemandLoader(
            pumcm_models=self._pumcm_models,
            flow_mappings=self._flow_mapping.B_sa_l,
            reward_provider=self._reward_provider,
            initial_states_list=list(demand.double().unbind(dim=0)),
            reward_invariant=True,
        )

    def _make_supply(self, free_flow_time, capacity, b, power):
        """Build the supply operator ``z(c)`` for one instance's BPR parameters.

        The single seam an asymmetric subclass overrides: base supply is separable ``InverseBPRSupply``
        (diagonal Jacobian, potential VI); an ``AsymmetricBPRSupply`` swaps in here to make the
        excess-supply operator non-potential.
        """
        return InverseBPRSupply(
            free_flow_time=free_flow_time.double(),
            capacity=torch.clamp(capacity.double(), min=1e-8),
            alpha=b.double(),
            beta=power.double(),
            eps=1e-6,
        )

    def _build_model(self, free_flow_time, capacity, b, power, demand_loader) -> PUMEModel:
        """Assemble a ``PUMEModel`` from a per-instance supply + cost box and a given demand loader.

        The topology-dependent pieces (PUMCM models, reward/flow mappings) are shared from ``__init__``;
        only the per-instance supply and cost box are built here. The caller supplies the demand loader:
        the operator reuses the persistent ``self._demand_loader`` (see ``build_model``); the offline
        ``solve`` passes a fresh per-instance loader (its internal iteration bakes in that instance's OD).
        """
        supply = self._make_supply(free_flow_time, capacity, b, power)

        cost_lower = free_flow_time.double().detach().numpy()
        # Cap cost below the exp(-cost) float64 underflow cliff (~709): an unbounded upper lets the
        # iterate run away, after which demand snaps to zero and the solve stalls without recovering.
        cost_upper = np.full_like(cost_lower, 700.0, dtype=np.float64)

        return PUMEModel(
            pumcm_models=self._pumcm_models,
            supply_func=supply,
            reward_mapping=self._reward_mapping,
            flow_mapping=self._flow_mapping,
            cost_bounds=(cost_lower, cost_upper),
            demand_loader=demand_loader,
        )

    def build_model(self, free_flow_time, capacity, b, power) -> PUMEModel:
        """Assemble the operator ``PUMEModel`` for one instance's BPR params, reusing the shared loader.

        Only the per-instance supply + cost box are built; the demand loader is the persistent
        ``self._demand_loader`` shared across every operator evaluation (the caller threads this
        instance's OD in per call via ``compute_demand(initial_states_list=...)``). Reusing one loader
        keeps PUME's ``id``-keyed torch-CSR cache pinned to a single entry, eliminating the per-eval leak.
        """
        return self._build_model(free_flow_time, capacity, b, power, self._demand_loader)

    def solve(self, instance: torch_geometric.data.Data):
        """Solve ``instance`` to user equilibrium; returns ``(costs, flows)`` in real units.

        Offline (dataset generation) path: builds a fresh per-instance demand loader baking in this
        instance's OD, because the equilibrium iteration calls ``compute_demand(costs)`` with no
        per-call OD override. Run once offline, so its bounded cache growth is irrelevant.
        """
        loader = self._make_demand_loader(instance.demand)
        model = self._build_model(instance.free_flow_time, instance.capacity, instance.b, instance.power, loader)
        cost_lower = instance.free_flow_time.double().detach().numpy()

        result = model.solve(
            c_initial=torch.as_tensor(cost_lower, dtype=torch.float64) * 1.1,
            method="meta",
            options={
                "max_iterations": self._outer_max_iter,
                "convergence_tolerance": self._outer_tol,
                "oracle_type": "aa1",
                "base_method": "agraal",
                "base_options": {"metric_mode": "supply_diagonal", "initial_stepsize": 5e-2},
            },
        )
        return result["cost"].float(), result["demand"].float()
