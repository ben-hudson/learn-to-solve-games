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
    whole family. Returns a fresh ``{"feats": ...}`` item.
    """

    def __call__(self, item):
        point = torch.as_tensor(item["point"], dtype=torch.float32)
        params = torch.as_tensor(item["params"], dtype=torch.float32)
        conditioning = params.expand(*point.shape[:-1], params.shape[-1])
        # Keep the raw params alongside feats so the collated batch can supply per-instance params
        # to the analytic operator during the validation equilibrium sweep.
        return {"feats": torch.cat([point, conditioning], dim=-1), "params": params}


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


class SPDEmbedding(BaseTransform):
    """Computes all-pairs shortest-path distances and stores them as ``data.spd``.

    Uses unweighted, directed BFS via ``scipy.sparse.csgraph.shortest_path``. The result is an
    ``[N, N]`` float tensor where entry ``[u, v]`` is the minimum number of hops from node ``u`` to
    node ``v`` (``inf`` if unreachable).
    """

    def forward(self, data):
        adj_sp = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
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
