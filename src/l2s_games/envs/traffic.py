"""Markov traffic equilibrium as a variational inequality family.

An instance is a road graph (a PyG ``Data`` with per-edge ``free_flow_time``,
``capacity``, ``b``, ``power`` and an OD ``demand`` matrix); the domain point is a
per-edge cost vector. The operator is the cost-space equilibrium residual
``costs - bpr(demand_flow(-costs))`` whose zero is the user equilibrium: current costs
minus the BPR cost of the flow the recursive-logit route-choice model routes at those
costs. This residual is smooth (``bpr`` has bounded slope), so simultaneous-GD on it is
exactly the damped fixed-point iteration that converges to equilibrium, and its Jacobian
is finite everywhere. The feasible set is ``costs >= free_flow_time``, so ``project``
clamps to that floor.

Parameters are multiplicative noise on a base graph; the clamped noise does not fill any
analytic box, which is why normalization is fit on the realized training distribution in
``data.py`` rather than from declared ranges. The conditioning seam (``model_input`` /
``collate_fn``) packs costs onto the graph and batches with PyG, ready for the amortized
GNN field model (follow-up).
"""

import torch
from route_choice import MarkovRouteChoice
from torch_geometric.data import Batch

from l2s_games.envs.base import VariationalInequalityFamily

_NOISED_ATTRS = ("free_flow_time", "capacity", "demand")


def bpr(free_flow_time, flow, capacity, b, power):
    """Bureau of Public Roads link cost: free-flow time inflated by congestion ``flow / capacity``."""
    return free_flow_time * (1.0 + b * torch.pow(flow / capacity, power))


class MarkovTrafficEquilibrium(VariationalInequalityFamily):
    """Family of single-graph traffic equilibria, varied by noising a base graph."""

    def __init__(self, base_graph, noise_scale=0.2, noise_type="normal"):
        self.base_graph = base_graph
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.dest_dim = -2
        # ift=True gives exact implicit-function-theorem gradients through the value/flow linear
        # solves (those differentiate cleanly). The full operator Jacobian is currently blocked by a
        # NaN in route_choice's EdgeProb softmax backward at sink nodes (see the note there), so
        # Jacobian-based dynamics target the learned field; non-Jacobian methods drive this operator.
        self.route_choice = MarkovRouteChoice(None, node_dim=-1, ift=True, f_max_iter=500, f_tol=1e-8)

    def sample_params(self):
        graph = self.base_graph.clone()
        draw = torch.randn_like if self.noise_type == "normal" else torch.rand_like
        for attr in _NOISED_ATTRS:
            value = getattr(graph, attr)
            factor = (draw(value) * self.noise_scale + 0.9).clamp(min=1e-2)
            setattr(graph, attr, value * factor)
        return graph

    def _demand_flow(self, edge_index, rewards, sink_node_mask, demand):
        """Aggregate route-choice edge flows over destinations for given link rewards ``(B, E)``.

        Rewards are expanded to a real per-destination axis ``(B, N, E)`` rather than broadcast,
        so the implicit gradient flows through ``expand`` (a sum) and stays finite.
        """
        n_dest = sink_node_mask.size(self.dest_dim)
        rewards = rewards.unsqueeze(-1).expand(*rewards.shape, n_dest).movedim(-1, self.dest_dim)
        _, probs, _ = self.route_choice.get_values_and_probs(edge_index, rewards, sink_node_mask)
        _, edge_flows, _ = self.route_choice.get_flows(edge_index, probs, demand)
        return edge_flows.sum(dim=self.dest_dim)

    def operator(self, graph, costs):
        """Cost-space equilibrium residual ``costs - bpr(demand_flow(-costs))`` (zero at equilibrium)."""
        costs = torch.as_tensor(costs, dtype=torch.float32)
        flat = costs.reshape(1, -1)  # (1, E): the route-choice solver needs a leading batch dim
        demand_flow = self._demand_flow(graph.edge_index, -flat, graph.sink_node_mask.unsqueeze(0), graph.demand.unsqueeze(0))
        implied_cost = bpr(graph.free_flow_time, demand_flow, graph.capacity, graph.b, graph.power)
        return (flat - implied_cost).reshape(costs.shape)

    def project(self, graph, costs):
        return torch.clamp(torch.as_tensor(costs, dtype=torch.float32), min=graph.free_flow_time)

    def sample_domain(self, graph, n):
        return graph.free_flow_time * (1.0 + torch.rand(n, graph.num_edges))

    def model_input(self, graph, costs):
        instance = graph.clone()
        instance.cost = torch.as_tensor(costs, dtype=torch.float32)
        return instance

    @property
    def collate_fn(self):
        return Batch.from_data_list
