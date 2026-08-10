"""GAMUT normal-form games as a graph-structured VI family.

Two-player bimatrix games drawn from a GAMUT generator class (RandomZeroSum, CovariantGame, RandomGame,
the classic 2x2s -- see ``l2s_games/gamut.py``), represented **polymatrix-ready** on a player graph: an
instance is a fully-connected PyG graph with one node per player and one payoff matrix per directed edge
(``payoff[e] = M_ij`` for edge ``i -> j``, so player ``i``'s payoff is ``sum_e x_i^T M_e x_j``). A bimatrix
``(A, B)`` is stored as ``payoff = [A, B^T]``; the operator ``F_i = -chart_i(sum_e M_e x_j)`` is already
the general polymatrix field, which is the point of keeping the graph -- extending to GAMUT's
``PolymatrixGame`` later changes the *generation*, not the representation. One constraint follows:
``n_actions`` is shared across players, so the edge payoffs stack into one ``[E, n, n]`` tensor.

The chart reuses ``matrix.helmert_basis`` -- not for centering but as the simplex tangent-space chart: it
removes the redundant sum-to-one direction, so the domain is ``(n-1)``-dimensional per player and the
operator is the tangential payoff gradient. Unlike ``rps`` the chart sits at the **uniform strategy**, not
the Nash, so each instance's equilibrium is a nonzero chart point: it is solved at generation time by
rolling out the true operator (``solve_instance``) and cached by ``EquilibriumDataset``, and
``reference_equilibrium`` reads it back off the batch. ``project`` is the Euclidean simplex projection
pulled back through the (orthonormal) chart, which makes the VI *constrained*: games whose Nash sits on
the simplex boundary (zero-probability actions) are still well-posed, with ``dynamics.natural_map`` as
the convergence measure -- the raw ``||F||`` need not vanish there.

Conditioning is the graph path (Graphormer), the traffic idiom minus ``LineGraph``: the domain point lives
per player *node*, so ``model_input`` clones the instance graph and attaches it as ``.point``, and
``transforms.game_field_transform`` builds per-node ``feats = [chart coords | own payoff matrix]``.

Which rollout algorithm converges is game-class-dependent: the bilinear field of a zero-sum game is
rotation-dominated, so ``projection`` cycles and the extragradient-family methods are the right default
(``solve_algo`` and the expert source's ``--algo`` alike). Non-monotone classes (CovariantGame with
``r > 0``, RandomGame) carry no guarantee at all -- the rollout endpoint is a candidate, checked by the
generation script's natural-map residual report.
"""

import torch
import torch_geometric.data
import torch_geometric.utils

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate
from l2s_games.envs.base import VariationalInequalityFamily, collate_dense_graphs
from l2s_games.envs.matrix import helmert_basis
from l2s_games.gamut import default_jar, generate_payoffs
from l2s_games.transforms import game_field_transform

# Instance-graph attrs stripped in ``model_input`` so they never reach a collated batch: the stored
# per-instance evaluation blocks (spent by the time ``model_input`` runs -- see ``traffic._DROPPED_ATTRS``
# for the shared rationale) and the provenance strings/ints the model has no use for.
_DROPPED_ATTRS = (
    "points",
    "targets",
    "preconditioner_diagonal",
    "game",
    "gamut_class",
    "gamut_options",
    "n_actions",
)


def build_gamut_base_graph(gamut_class, n_actions, n_players=2, options=()):
    """The canonical player graph a GAMUT dataset is generated around: fully connected, no payoffs.

    Carries the full generation provenance (``gamut_class`` + ``gamut_options``), so the dataset records
    which distribution its instances were drawn from; ``sample_params`` clones this graph and attaches
    fresh payoffs. Like the traffic base graph it deliberately carries no equilibrium -- a clone would
    inherit it as though it were its own.
    """
    adjacency = torch.ones(n_players, n_players) - torch.eye(n_players)
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
    graph = torch_geometric.data.Data(edge_index=edge_index, num_nodes=n_players)
    graph.gamut_class = gamut_class
    graph.gamut_options = list(options)
    graph.n_actions = int(n_actions)
    return graph


def project_simplex(x):
    """Euclidean projection of the last axis onto the probability simplex (sort-based, batched)."""
    sorted_x = torch.sort(x, dim=-1, descending=True).values
    cumulative = sorted_x.cumsum(dim=-1) - 1.0
    ranks = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype)
    support = (sorted_x * ranks > cumulative).sum(dim=-1, keepdim=True)  # active-coordinate count
    threshold = cumulative.gather(-1, support - 1) / support.to(x.dtype)
    return (x - threshold).clamp(min=0.0)


class GamutGame(VariationalInequalityFamily):
    """GAMUT two-player games on the player graph, charted per player on simplex tangent spaces."""

    def __init__(
        self,
        base_graph,
        jar_path=None,
        payoff_range=(-1.0, 1.0),
        solve_algo="extragradient",
        solve_h=0.2,
        solve_steps=2000,
    ):
        # solve_h: payoffs are normalized into payoff_range, so the field's Lipschitz constant is ~2 and
        # 0.2 sits well inside extragradient's stable range -- measured on real RandomZeroSum instances,
        # it reaches ~1e-7 natural-map residual in 2000 steps where 0.05 stalls at ~1e-2.
        self.base_graph = base_graph
        self.n_players = base_graph.num_nodes
        self.n_actions = base_graph.n_actions
        # jar_path stays unresolved until sample_params: serving a cached root never needs Java.
        self.jar_path = jar_path
        self.payoff_range = payoff_range
        self.solve_algo = solve_algo
        self.solve_h = solve_h
        self.solve_steps = solve_steps
        self.basis = helmert_basis(self.n_actions)  # one shared basis: n_actions is shared across players
        self.chart_dim = self.n_actions - 1
        self.uniform = torch.full((self.n_actions,), 1.0 / self.n_actions)

    @property
    def domain_dim(self):
        return self.n_players * self.chart_dim

    def sample_params(self):
        """One fresh GAMUT instance: the base graph cloned, with new payoffs in the per-edge convention."""
        graph = self.base_graph.clone()
        a, b = generate_payoffs(
            graph.gamut_class,
            self.n_actions,
            self.jar_path or default_jar(),
            *self.payoff_range,
            options=graph.gamut_options,
        )
        graph.payoff = torch.stack([a, b.T])  # payoff[e] = M_ij for edge i -> j
        return graph

    def _strategies(self, points):
        """Chart points ``[..., d]`` -> mixed strategies ``[..., n_players, n_actions]``."""
        points = torch.as_tensor(points, dtype=torch.float32)
        chart = points.reshape(*points.shape[:-1], self.n_players, self.chart_dim)
        return self.uniform + chart @ self.basis.T

    def operator(self, params, points):
        """The tangential payoff-gradient field ``F_i = -B^T sum_{e: i -> j} M_e x_j``, zero at the Nash.

        ``params`` is an instance graph or the dense collated batch (``params_from_batch``); its
        ``payoff`` is ``[E, n, n]`` or ``[B, E, n, n]``, broadcast against ``points`` ``[d]`` / ``[N, d]``
        / ``[B, d]``. For two players this is exactly the charted ``(-Ay, -B^T x)``. Plain torch
        (einsum + indexing), so ``jacrev`` composes through it.
        """
        strategies = self._strategies(points)
        source, destination = params["edge_index"]
        opponents = strategies[..., destination, :]  # [..., E, n]
        gradients = -torch.einsum("...eab,...eb->...ea", params["payoff"].float(), opponents)
        incidence = torch.eye(self.n_players)[:, source]  # [P, E]: edge e contributes to node source[e]
        field = torch.einsum("pe,...ea->...pa", incidence, gradients) @ self.basis  # [..., P, n-1]
        return field.reshape(*strategies.shape[:-2], self.domain_dim)

    def sample_domain(self, params, n):
        """Feasible chart points: strategies drawn uniformly on each player's simplex (Dirichlet(1)),
        pulled back through the chart -- covering exactly the constrained domain a rollout can visit."""
        simplex_uniform = torch.distributions.Dirichlet(torch.ones(self.n_players, self.n_actions))
        return ((simplex_uniform.sample((n,)) - self.uniform) @ self.basis).reshape(n, -1)

    def project(self, params, points):
        """Per-player simplex projection, pulled back through the chart.

        Exact because the basis is orthonormal: projecting ``x = u + Bz`` onto the simplex and mapping
        back is the Euclidean projection of ``z`` onto the simplex's chart image. This is what keeps
        rollout iterates feasible and makes boundary-Nash games well-posed (see the module docstring).
        """
        points = torch.as_tensor(points, dtype=torch.float32)
        projected = (project_simplex(self._strategies(points)) - self.uniform) @ self.basis
        return projected.reshape(*points.shape)

    def solve_instance(self, instance):
        """The equilibrium chart point, by rolling out the true operator (``solve_algo``) to its
        constrained zero -- the ``solve_fn`` a ``EquilibriumDataset`` caches."""
        start = self.sample_domain(instance, 1)[0]
        trajectory = simulate(
            lambda z: -self.operator(instance, z),
            ALGORITHMS[self.solve_algo](self.solve_h),
            start,
            self.solve_steps,
            project=lambda z: self.project(instance, z),
        )
        return trajectory[-1]

    # --- conditioning seams (the graph path, traffic idiom minus LineGraph) --------------------------

    def model_input(self, graph, point):
        """Raw input item: the instance graph with the domain point attached as ``.point``.

        Featurization is deferred to ``transform`` (lazy per access; see ``transforms.py``); the stored
        evaluation blocks and provenance are dropped so they never reach a collated batch, while
        ``payoff`` and ``equilibrium`` survive -- the batched operator and ``reference_equilibrium``
        read them off the batch.
        """
        item = graph.clone()
        for attr in _DROPPED_ATTRS:
            if attr in item:
                del item[attr]
        item.point = torch.as_tensor(point, dtype=torch.float32)
        return item

    @property
    def transform(self):
        return game_field_transform()

    collate_fn = staticmethod(collate_dense_graphs)

    def params_from_batch(self, batch):
        # The batch's edge_index IS the physical player graph (no LineGraph re-chart to undo).
        return batch

    def batched_field_input(self, batch, points, normalizer):
        """Splice a batch of chart points ``[B, d]`` into the dense batch's inputs for the learned field.

        The point occupies the leading ``chart_dim`` columns of each player node's ``feats`` (see
        ``transforms.BuildGameNodeData``); standardization is per-column affine, so overwriting those
        columns with the standardized coords is exact, and concatenation keeps it jacrev-transparent.
        """
        chart = torch.as_tensor(points, dtype=torch.float32).reshape(-1, self.n_players, self.chart_dim)
        standardized = (chart - normalizer.input.mean[: self.chart_dim]) / normalizer.input.std[: self.chart_dim]
        feats = torch.cat([standardized, batch["feats"][..., self.chart_dim :]], dim=-1)
        return {**batch, "feats": feats}

    def initial_point(self, batch):
        """Rollout start for the validation sweep: the example's uniformly sampled domain point,
        ``[B, d]`` (kept raw on the item as ``.point``, surviving transform and collate)."""
        return batch["point"]

    def reference_equilibrium(self, batch):
        """Each instance's own solved ``equilibrium``, ``[B, d]`` -- the chart is not Nash-centered, so
        the base default ``0.0`` would silently measure ``||z_end||``."""
        return batch["equilibrium"].float()
