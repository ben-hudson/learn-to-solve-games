"""Featurization transforms for the field-model pipeline (flat and graph games alike).

A *transform* is the per-family ``transform`` seam (see ``envs/base.py``): a pure ``item -> item``
callable that builds the ``feats`` the field model consumes, applied **lazily on every
``__getitem__``**. Keeping featurization here (rather than in ``model_input``, whose result would be
stored) is what lets the graph path recompute its line-graph structure and static edge features
fresh per access -- so nothing is cached. Flat games gain nothing from laziness but follow the same
shape for uniformity; their ``feats`` are cheap to rebuild.

Normalization is deliberately *not* a transform: it fits on the train split and must invert at
inference, so it lives on ``data.Normalizer`` and is applied by the agnostic dataset layer. The
*stateless* target clip (``NormClip``), by contrast, has nothing to fit and applies identically
everywhere, so it *is* a transform and lives here.
"""

import scipy.sparse.csgraph
import torch
import torch_geometric.utils
from torch_geometric.transforms import BaseTransform, Compose, LineGraph
from torch_geometric.utils import degree

_EDGE_ATTRS = ("free_flow_time", "capacity", "b", "power")


class NormClip(BaseTransform):
    """Direction-preserving L2-norm clip of a real-unit field.

    Scales a field ``f`` by ``min(1, max_norm / ‖f‖)`` over the last (field) axis, capping its
    magnitude at ``max_norm`` while leaving its direction -- and the equilibrium ``f = 0`` --
    untouched. Unlike a per-component clamp it never rotates the field, so a model trained on the
    clipped target learns the operator's true direction even in the heavy-tailed blow-up regions.

    Stateless (nothing to fit), so it is a plain transform. Applied only to grad-free tensors (the
    training target and the ``rel_err`` metric): ``BaseTransform.__call__`` copy-copies its input,
    which breaks ``jacrev``, so it must not sit in the inference field path.
    """

    def __init__(self, max_norm):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, f):
        norm = torch.linalg.norm(f, dim=-1, keepdim=True).clamp(min=1e-12)  # safe denom at f = 0
        return f * (self.max_norm / norm).clamp(max=1.0)


class ConcatConditioning:
    """Flat-game ``feats`` builder: ``feats = [point | params]``.

    The raw item (from a flat family's ``model_input``) carries ``point`` and ``params``; this
    appends the instance ``params`` to the domain ``point`` so one field model can represent the
    whole family. Returns the item with ``feats`` added, everything it already carried untouched.
    """

    def __call__(self, item):
        point, params = item["point"], item["params"]
        conditioning = params.expand(*point.shape[:-1], params.shape[-1])
        # Only ``feats`` is added -- every key the raw item carries rides along. The raw params are what
        # let the collated batch supply per-instance params to the analytic operator during the
        # validation equilibrium sweep; the raw (un-standardized) point also sits inside feats (the
        # coordinate the field is evaluated at), but the batch needs it raw as the validation rollout's
        # start iterate (mirrors how traffic keeps a raw ``cost`` alongside the standardized cost in
        # ``feats[0]``). An ``instance_index`` diagnostic tag rides along the same way, matching the
        # graph transform, which mutates its ``Data`` in place.
        return dict(item, feats=torch.cat([point, conditioning], dim=-1))


def demand_edge_features(graph):
    """``[E, 4]`` per-edge demand features, in this codebase's ``demand[dest, origin]`` convention.

    For edge ``i -> j``: total demand originating at ``i``, total demand destined for ``j``, the
    direct ``i -> j`` demand, and the reverse ``j -> i`` demand.
    """
    # TODO: I think we are probably duplicating some features
    # for example when we have edges i->j and j->i we duplicate the demand[i, j] and demand [j, i] feats
    i, j = graph.edge_index
    demand = graph.demand  # demand[dest, origin]
    return torch.stack(
        [demand[:, i].sum(dim=0), demand[j, :].sum(dim=1), demand[j, i], demand[i, j]],
        dim=-1,
    )


class BuildTrafficEdgeData(BaseTransform):
    """Build ``data.feats = [cost | free_flow_time | capacity | b | power | 4 demand feats]``.

    Reads the domain point off ``data.cost`` and the BPR/demand attributes off the (physical) graph,
    so it must run **before** ``LineGraph`` replaces ``edge_index``. Recomputed on every call -- the
    static edge features are never cached.

    NOTE (well-posedness, revisit): the 4 demand features (``demand_edge_features``) are a *lossy*
    per-edge summary of the full OD matrix, but the route-choice flow -- and hence the operator target
    -- depends on the full per-destination demand. Since ``sample_params`` noises every OD entry
    independently, instances with the same 4-feature summary can have different targets, so the
    regression may be under-determined (an irreducible error floor). A diagnostic that held demand
    fixed did NOT lower train_rel_err (~0.60 either way), but that is inconclusive: train error is
    currently floored by a *different* (optimization/capacity) bottleneck that would mask any
    demand-induced floor. Revisit once the model can fit a well-posed problem to low train error --
    then re-run fixed-vs-noised demand to see whether this encoding imposes its own ceiling. If so,
    the fix is a richer demand conditioning (e.g. full per-node origin/destination demand vectors),
    not more training.
    """

    def forward(self, data):
        cost = torch.as_tensor(data.cost, dtype=torch.float32)
        bpr_attrs = torch.stack([getattr(data, attr) for attr in _EDGE_ATTRS], dim=-1)
        static = torch.cat([bpr_attrs, demand_edge_features(data)], dim=-1)
        data.feats = torch.cat([cost.unsqueeze(-1), static], dim=-1)
        # LineGraph (next in the pipeline) overwrites edge_index with the line-graph adjacency, so
        # stash the physical edge_index -- the batched analytic operator needs it to solve route choice.
        data.physical_edge_index = data.edge_index.clone()
        return data


class BuildGameNodeData(BaseTransform):
    """Build per-player ``data.feats = [chart point | own payoff matrix]`` for a matrix game.

    The player-graph analogue of ``BuildTrafficEdgeData``, without the ``LineGraph`` step: the domain
    point lives per player *node* (each player's simplex-chart coordinates), so the physical graph is
    already the Graphormer's topology. Node ``i``'s conditioning is the payoff matrix on its outgoing
    edge, flattened -- which pins the current shape to one outgoing edge per player (2-player games);
    a polymatrix extension has to aggregate a node's edge payoffs instead. ``feats`` is
    ``[num_nodes, chart_dim + n_actions**2]``, recomputed on every call like the traffic transform.
    """

    def forward(self, data):
        chart = torch.as_tensor(data.point, dtype=torch.float32).reshape(data.num_nodes, -1)
        own_edge = data.edge_index[0].argsort()  # one outgoing edge per node: edge k conditions node src[k]
        payoffs = data.payoff.reshape(data.payoff.shape[0], -1).float()[own_edge]
        data.feats = torch.cat([chart, payoffs], dim=-1)
        return data


class SPDEmbedding(BaseTransform):
    """Computes all-pairs shortest-path distances and stores them as ``data.spd``.

    Uses unweighted, directed BFS via ``scipy.sparse.csgraph.shortest_path``. The result is an
    ``[N, N]`` float tensor where entry ``[u, v]`` is the minimum number of hops from node ``u`` to
    node ``v`` (``inf`` if unreachable).
    """

    def forward(self, data):
        # .tocsr(): shortest_path's auto method picks Floyd-Warshall on dense graphs (e.g. the 2-node
        # player graph), and that path rejects the COO matrix to_scipy_sparse_matrix returns.
        adj_sp = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        spd_sp = scipy.sparse.csgraph.shortest_path(adj_sp, directed=True, unweighted=True)
        data.spd = torch.as_tensor(spd_sp).float()
        return data


class DegreeEmbedding(BaseTransform):
    """Computes per-node in- and out-degree as ``data.in_degree`` / ``data.out_degree`` (``[N]``)."""

    def forward(self, data):
        data.in_degree = degree(data.edge_index[1], num_nodes=data.num_nodes)
        data.out_degree = degree(data.edge_index[0], num_nodes=data.num_nodes)
        return data


def traffic_field_transform():
    """The traffic ``feats`` + line-graph-structure pipeline.

    Charts the road graph onto its line graph (edges -> nodes): builds per-edge features, then the
    degree/shortest-path structure the Graphormer consumes. ``force_directed=True`` keeps the line
    graph directed; ``BuildTrafficEdgeData`` runs first so it sees the physical ``edge_index``.
    Because the input ``edge_index`` is coalesced (see ``traffic._canonicalize``), line-graph node
    ``k`` is edge ``k``, so ``feats`` stays aligned with the structure.
    """
    return Compose([BuildTrafficEdgeData(), LineGraph(force_directed=True), SPDEmbedding(), DegreeEmbedding()])


def game_field_transform():
    """The matrix-game ``feats`` + structure pipeline: like traffic's, minus ``LineGraph`` -- the domain
    point lives per player node, so the player graph itself is the Graphormer's topology."""
    return Compose([BuildGameNodeData(), SPDEmbedding(), DegreeEmbedding()])
