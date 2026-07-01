"""Markov traffic equilibrium as a variational inequality family.

An instance is a road graph (a PyG ``Data`` with per-edge ``free_flow_time``,
``capacity``, ``b``, ``power`` and an OD ``demand`` matrix); the domain point is a
per-edge cost vector. The operator is the cost-space equilibrium residual
``costs - bpr(demand_flow(-costs))`` whose zero is the user equilibrium: current costs
minus the BPR cost of the flow the recursive-logit route-choice model routes at those
costs. This residual is smooth (``bpr`` has bounded slope), so simultaneous-GD on it is
exactly the damped fixed-point iteration that converges to equilibrium. The feasible set
is ``costs >= free_flow_time``, so ``project`` clamps to that floor.

Parameters are multiplicative noise on a base graph; the clamped noise does not fill any
analytic box, which is why normalization is fit on the realized training distribution in
``data.py``. The conditioning seam (``model_input`` -> ``transform`` -> ``collate_fn``) charts the
road graph onto its line graph (edges -> nodes): ``model_input`` attaches the domain point as
``.cost``, ``transform`` (see ``transforms.py``) builds per-edge features ``[cost, free_flow_time,
capacity, b, power, +4 demand features]`` plus the line-graph degree/shortest-path structure the
Graphormer consumes -- **fresh on every access, nothing cached** -- and ``collate_fn`` stacks the
single-topology graphs into a dense batch.
"""

import torch
from route_choice import MarkovRouteChoice
from torch_geometric.utils import coalesce

from l2s_games.envs.base import VariationalInequalityFamily
from l2s_games.transforms import traffic_field_transform

_NOISED_ATTRS = ("free_flow_time", "capacity", "demand")
_EDGE_ATTRS = ("free_flow_time", "capacity", "b", "power")
_DENSE_KEYS = ("feats", "in_degree", "out_degree", "spd")


def bpr(free_flow_time, flow, capacity, b, power):
    """Bureau of Public Roads link cost: free-flow time inflated by congestion ``flow / capacity``."""
    return free_flow_time * (1.0 + b * torch.pow(flow / capacity, power))


def _canonicalize(graph):
    """Coalesce ``edge_index`` (so line-graph node ``k`` == edge ``k``) and reorder per-edge attrs."""
    graph = graph.clone()
    edge_index, order = coalesce(graph.edge_index, torch.arange(graph.num_edges), num_nodes=graph.num_nodes)
    graph.edge_index = edge_index
    for attr in _EDGE_ATTRS:
        setattr(graph, attr, getattr(graph, attr)[order])
    return graph


class MarkovTrafficEquilibrium(VariationalInequalityFamily):
    """Family of single-graph traffic equilibria, varied by noising a base graph."""

    def __init__(self, base_graph, noise_scale=0.2, noise_type="normal"):
        self.base_graph = _canonicalize(base_graph)
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.dest_dim = -2
        # ift=True gives exact implicit-function-theorem gradients through the value/flow linear
        # solves. (The analytic operator's full Jacobian is blocked by a NaN in route_choice's
        # EdgeProb softmax backward at sink nodes -- see the note there. Jacobian-based dynamics
        # target the learned field; non-Jacobian methods drive this operator.)
        self.route_choice = MarkovRouteChoice(None, node_dim=-1, ift=True, f_max_iter=500, f_tol=1e-8)

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

    def _demand_flow(self, edge_index, rewards, sink_node_mask, demand):
        """Aggregate route-choice edge flows over destinations for given link rewards ``(1, E)``.

        Rewards are expanded to a real per-destination axis ``(1, N, E)`` rather than broadcast,
        so the implicit gradient flows through ``expand`` (a sum) and stays finite.
        """
        n_dest = sink_node_mask.size(self.dest_dim)
        rewards = rewards.unsqueeze(-1).expand(*rewards.shape, n_dest).movedim(-1, self.dest_dim)
        _, probs, _ = self.route_choice.get_values_and_probs(edge_index, rewards, sink_node_mask)
        _, edge_flows, _ = self.route_choice.get_flows(edge_index, probs, demand)
        return edge_flows.sum(dim=self.dest_dim)

    def _single_operator(self, graph, costs):
        """Residual for one cost vector ``(E,)`` -> ``(E,)``."""
        flat = costs.reshape(1, -1)  # (1, E): the route-choice solver needs a leading batch dim
        demand_flow = self._demand_flow(
            graph.edge_index, -flat, graph.sink_node_mask.unsqueeze(0), graph.demand.unsqueeze(0)
        )
        implied_cost = bpr(graph.free_flow_time, demand_flow, graph.capacity, graph.b, graph.power)
        return (flat - implied_cost).reshape(costs.shape)

    def operator(self, graph, costs):
        """Cost-space equilibrium residual ``costs - bpr(demand_flow(-costs))`` (zero at equilibrium).

        ``costs`` is one cost vector ``(E,)`` (dynamics) or a batch ``(k, E)`` (data generation);
        the batch is looped, since each is an independent route-choice solve.
        """
        costs = torch.as_tensor(costs, dtype=torch.float32)
        if costs.dim() == 1:
            return self._single_operator(graph, costs)
        return torch.stack([self._single_operator(graph, c) for c in costs])

    def project(self, graph, costs):
        return torch.clamp(torch.as_tensor(costs, dtype=torch.float32), min=graph.free_flow_time)

    def sample_domain(self, graph, n):
        return graph.free_flow_time * (1.0 + torch.rand(n, graph.num_edges))

    def model_input(self, graph, cost):
        """Raw input item: the instance graph with the domain point attached as ``.cost``.

        Featurization (per-edge features + line-graph structure) is deferred to ``transform`` so it
        runs lazily per access -- see the note in ``transforms.py``.
        """
        item = graph.clone()
        item.cost = torch.as_tensor(cost, dtype=torch.float32)
        return item

    @property
    def transform(self):
        return traffic_field_transform()

    @staticmethod
    def collate_fn(items):
        """Dense-batch same-topology line graphs: stack per-node tensors, share ``edge_index``.

        The Graphormer uses dense attention over one fixed topology, so a batch is stacked tensors
        ``{feats [B,E,k], in_degree [B,E], out_degree [B,E], spd [B,E,E]}`` plus the shared
        ``edge_index`` -- not a PyG sparse ``Batch``.
        """
        batch = {key: torch.stack([item[key] for item in items]) for key in _DENSE_KEYS}
        batch["edge_index"] = items[0].edge_index
        return batch
