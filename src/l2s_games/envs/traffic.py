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

import pathlib
from urllib.parse import urljoin, urlparse

import tntp
import torch
from route_choice import MarkovRouteChoice
from torch_geometric.utils import coalesce, from_networkx

from l2s_games.envs.base import VariationalInequalityFamily
from l2s_games.transforms import traffic_field_transform


def _join(root, name):
    """Join ``name`` onto a ``base`` that is either a URL or a local directory path."""
    if urlparse(str(root)).scheme in ("http", "https", "ftp"):
        return urljoin(str(root), name)
    return str(pathlib.Path(root) / name)


_NOISED_ATTRS = ("free_flow_time", "capacity")  # TODO: add demand back in once everything is working
_EDGE_ATTRS = ("free_flow_time", "capacity", "b", "power")
_REFERENCE_ATTRS = ("Cost", "Volume")  # TNTP-shipped reference equilibrium cost/flow, when present


def load_sioux_falls_base_graph(root, scaling=1000.0):
    """Sioux Falls as a PyG graph: BPR params, OD demand, sink mask, and reference costs (``Cost``).

    ``base`` is the base location of the ``SiouxFalls_*.tntp`` files -- either a local directory
    or a URL (e.g. the upstream TransportationNetworks raw path).
    """
    node_df = tntp.read_node_file(
        _join(root, "SiouxFalls_node.tntp"), index_col="Node", x_col="X", y_col="Y", crs="wgs84"
    )
    net_df = tntp.read_net_file(_join(root, "SiouxFalls_net.tntp"), crs="wgs84")
    flow_df = tntp.read_flow_file(_join(root, "SiouxFalls_flow.tntp"), u_col="From", v_col="To")
    flow_df = flow_df.rename(columns={"From": "init_node", "To": "term_node"})
    net_df = net_df.merge(flow_df, on=["init_node", "term_node"])  # adds the reference Cost column
    network = tntp.convert_to_networkx(node_df, net_df)
    node_list = list(network.nodes)
    demand_table = tntp.read_demand_file(_join(root, "SiouxFalls_trips.tntp")).reindex(
        index=node_list, columns=node_list
    )

    graph = from_networkx(network)
    graph.free_flow_time = graph.free_flow_time.float()
    graph.capacity = graph.capacity.float() / scaling  # flow/capacity ratio is scale-invariant, so costs are unchanged
    graph.b = graph.b.float()
    graph.power = graph.power.float()
    graph.demand = torch.as_tensor(demand_table.values.T, dtype=torch.float32).clone() / scaling  # demand[dest, origin]
    graph.sink_node_mask = torch.diag_embed(torch.ones(len(node_list), dtype=torch.long))
    return graph


def bpr(free_flow_time, flow, capacity, b, power):
    """Bureau of Public Roads link cost: free-flow time inflated by congestion ``flow / capacity``."""
    return free_flow_time * (1.0 + b * torch.pow(flow / capacity, power))


def _canonicalize(graph):
    """Coalesce ``edge_index`` (so line-graph node ``k`` == edge ``k``) and reorder per-edge attrs."""
    graph = graph.clone()
    edge_index, order = coalesce(graph.edge_index, torch.arange(graph.num_edges), num_nodes=graph.num_nodes)
    graph.edge_index = edge_index
    reference_attrs = tuple(attr for attr in _REFERENCE_ATTRS if attr in graph)
    for attr in _EDGE_ATTRS + reference_attrs:
        setattr(graph, attr, getattr(graph, attr)[order])
    return graph


class MarkovTrafficEquilibrium(VariationalInequalityFamily):
    """Family of single-graph traffic equilibria, varied by noising a base graph."""

    def __init__(
        self,
        base_graph,
        noise_scale=0.2,
        noise_type="normal",
        equilibrium_margin=2.5,
        equilibrium_spread=0.2,
    ):
        self.base_graph = _canonicalize(base_graph)
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        # Domain sampling must span the whole path the rollout traverses -- from the free-flow-time
        # start up to the equilibrium. Rather than solve each instance's equilibrium, anchor the top
        # of that range at the base network's shipped reference equilibrium cost (``Cost``), widened
        # by ``equilibrium_margin`` so the bounded parameter perturbations' equilibria stay bracketed
        # by one fixed ceiling shared across every instance. See sample_domain.
        self.reference_equilibrium = self.base_graph.Cost * equilibrium_margin
        self.equilibrium_spread = equilibrium_spread
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
        """Aggregate route-choice edge flows over destinations for given link rewards ``(B, E)``.

        Rewards are expanded to a real per-destination axis ``(B, N, E)`` rather than broadcast,
        so the implicit gradient flows through ``expand`` (a sum) and stays finite. The leading
        ``B`` axis lets the solver handle a whole batch of instances jointly.
        """
        n_dest = sink_node_mask.size(self.dest_dim)
        rewards = rewards.unsqueeze(-1).expand(*rewards.shape, n_dest).movedim(-1, self.dest_dim)
        _, probs, _ = self.route_choice.get_values_and_probs(edge_index, rewards, sink_node_mask)
        _, edge_flows, _ = self.route_choice.get_flows(edge_index, probs, demand)
        return edge_flows.sum(dim=self.dest_dim)

    @staticmethod
    def _to_batch(value, batch_size, base_rank):
        """Broadcast a per-instance attribute of rank ``base_rank`` to a leading batch dim if it lacks one."""
        if value.dim() == base_rank:
            value = value.unsqueeze(0).expand(batch_size, *([-1] * base_rank))
        return value

    def operator(self, params, costs):
        """Cost-space equilibrium residual ``costs - bpr(demand_flow(-costs))`` (zero at equilibrium).

        ``params`` exposes the physical topology under ``edge_index`` (``[2, E]``) plus the per-edge
        BPR attrs and OD ``demand`` -- satisfied natively by a raw instance graph, and by the dense
        model batch once ``params_from_batch`` has pointed ``edge_index`` at the stashed physical
        topology. Per-edge attrs of a single instance (rank 1 / 2) are broadcast over the cost batch
        so the route-choice solver solves every row jointly. ``costs`` is ``[B, E]`` (one cost vector
        per instance) or a bare ``[E]`` vector (a batch of one, squeezed back on return).
        """
        costs = torch.as_tensor(costs, dtype=torch.float32)
        # TODO(remove): the bare-[E] path (this `single` flag and the `squeeze(0)` on return) is only
        # hit by the sandbox's single-instance dynamics. The training pipeline (dataset gen + the
        # validation sweep) always passes batched costs, so this can go once the sandbox is dropped.
        single = costs.dim() == 1
        costs = costs.unsqueeze(0) if single else costs
        batch_size = costs.shape[0]
        free_flow_time, capacity, b, power = (self._to_batch(params[name], batch_size, 1) for name in _EDGE_ATTRS)
        demand = self._to_batch(params["demand"], batch_size, 2)
        sink_node_mask = self._to_batch(params["sink_node_mask"], batch_size, 2)
        demand_flow = self._demand_flow(params["edge_index"], -costs, sink_node_mask, demand)
        residual = costs - bpr(free_flow_time, demand_flow, capacity, b, power)
        return residual.squeeze(0) if single else residual

    def project(self, params, costs):
        # Subscript access works for both a single graph and the dense batch dict (validation sweep).
        return torch.clamp(torch.as_tensor(costs, dtype=torch.float32), min=params["free_flow_time"])

    def params_from_batch(self, batch):
        """The operator's params, extracted from the dense model batch.

        The batch's own ``edge_index`` is the line graph (the Graphormer's topology); point
        ``edge_index`` at the stashed physical topology so the batch satisfies the operator's
        contract, matching a raw instance graph. Per-edge attrs are read through as-is.
        """
        return {**batch, "edge_index": batch["physical_edge_index"]}

    def initial_point(self, batch):
        return batch["free_flow_time"]

    def sample_domain(self, graph, n):
        """Feasible cost points spanning the whole path from the free-flow-time start to equilibrium.

        The rollout starts at ``free_flow_time`` and converges to the equilibrium, so training must
        cover that entire segment -- not just a ball around either end. Each point interpolates from
        the instance's ``free_flow_time`` toward the fixed ``reference_equilibrium`` ceiling by a
        scalar reach in ``[0, 1]`` (so the samples fill the path rather than a high-dimensional box),
        with multiplicative noise spreading points off the line. Costs are clamped to the feasible
        floor ``free_flow_time``.
        """
        fft = graph.free_flow_time
        reach = torch.rand(n, 1)
        line = fft + reach * (self.reference_equilibrium - fft)
        spread = 1.0 + self.equilibrium_spread * torch.randn(n, graph.num_edges)
        return torch.clamp(line * spread, min=fft)

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
    def batched_field_input(batch, costs, normalizer):
        """Splice a batch of costs ``[B, E]`` into the dense batch's inputs for the learned field.

        The domain point is the cost, which lives in ``feats`` column 0 (see ``BuildTrafficEdgeData``).
        Standardization is per-column affine, so overwriting that column with the standardized cost is
        exact; concatenation (not in-place assignment) keeps it jacrev-transparent. The remaining
        (static) feature columns and the line-graph structure are reused as-is.
        """
        costs = torch.as_tensor(costs, dtype=torch.float32)
        standardized_cost = (costs - normalizer.input.mean[0]) / normalizer.input.std[0]
        feats = torch.cat([standardized_cost.unsqueeze(-1), batch["feats"][..., 1:]], dim=-1)
        return {**batch, "feats": feats}

    @staticmethod
    def collate_fn(items):
        """Dense-batch same-topology line graphs: stack every per-item tensor, share the topologies.

        The Graphormer uses dense attention over one fixed topology, so a batch is stacked tensors
        ``{feats [B,E,k], in_degree [B,E], out_degree [B,E], spd [B,E,E], ...}`` plus the shared
        (line-graph) ``edge_index`` -- not a PyG sparse ``Batch``. Every other tensor attribute is
        stacked, so the real-unit BPR/demand params survive -- the batched analytic operator needs
        them. Both topologies (``edge_index`` and the physical ``physical_edge_index``) are identical
        across the batch, so they are shared un-stacked rather than copied ``B`` times.
        """
        shared = ("edge_index", "physical_edge_index")  # one topology across the batch -- store once
        batch = {key: items[0][key] for key in shared}
        for key in items[0].keys():
            value = items[0][key]
            if key not in shared and isinstance(value, torch.Tensor):
                batch[key] = torch.stack([item[key] for item in items])
        return batch
