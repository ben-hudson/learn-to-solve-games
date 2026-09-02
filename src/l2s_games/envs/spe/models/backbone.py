"""Market-token backbone for the spatial price equilibrium.

The residual stream is one token per market -- ``m`` supply plus ``n`` demand. The instance has
exactly three kinds of data, and the backbone gives each one slot:

- per-market (``p``, ``-q``) seeds the market tokens;
- cross-side pairs (``c``, ``delta``) become *route* tokens, one per route;
- same-side pairs (``P``, ``Q``) become *pair* tokens, one per ordered pair of same-side markets.

Route and pair tokens are transient: rebuilt from the current market tokens every layer, consumed,
and discarded. The instance data is therefore re-injected at full strength at every depth, while the
sequence carried forward stays at ``m + n``.

Both steps have the same shape. Build the neighbour tokens, prepend the market token to them, and run
one ``TransformerEncoderLayer`` over that sequence; position zero is the refined market token. The
market token is both the query and the residual -- it reads its neighbours and carries itself forward
through the layer's own skip connection -- while the neighbours also see each other, so a market's
routes compete inside the same attention.

Edge data enters as **token content**, never as an attention bias. That is what the pair tokens buy.
A bias can only reweight values that depend on the key alone, so the output stays in the convex hull
of those values; but the route cost enters the physics *additively*,

    x_ij = relu( -(pi_i + rho~_j + c_ij) / delta_ij )

and no reweighting can produce a per-edge additive term. With ``c_ij`` in the route token the bracket
is formable. The same argument, softer, applies to ``P``: ``(P s)_i = sum_k P_ik s_k`` has freely
signed coefficients (``random_psd_plus_skew`` leaves the skew part unconstrained) and a magnitude
that scales with ``||P||``, neither of which survives a softmax over bias-derived weights.

Sign convention (assumed of the caller, and load-bearing). The operator is

    F_ij = pi_i + c_ij + delta_ij x_ij - rho_j,   pi = P s + p,   rho = q - Q d

which is not symmetric between the sides. Feeding the demand intercept as ``-q`` makes it so: with
``rho~ = -rho = Q d - q``, both sides read "own matrix times own marginal, plus intercept" and
``F_ij = pi_i + rho~_j + c_ij + delta_ij x_ij`` is symmetric in the two roles. Every module is then
shared between the sides, so the backbone needs no side-type embedding and is equivariant to swapping
the sides wholesale as well as to permuting either side's markets. Nothing is sized by ``m`` or ``n``.

One known limitation, deliberate at this scale. Attention returns a convex combination of the values,
so an aggregate that genuinely grows with the number of markets -- the marginals ``s_i = sum_j x_ij``
do -- has to be recovered from the count rather than read off directly. Harmless while training and
testing at one instance size, since the model just learns the constant. It is the first thing to
revisit for size extrapolation, where the fix is to replace the two attention steps with unnormalized
sum aggregation rather than to add anything.
"""

import torch


def _encoder_layer(dim, n_heads, dim_ff, dropout):
    return torch.nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=n_heads,
        dim_feedforward=dim_ff,
        dropout=dropout,
        activation="gelu",
        norm_first=True,
        batch_first=True,
    )


def _attend_over_neighbours(layer, markets, neighbours):
    """Refined market tokens ``[B, k, dim]`` from neighbour tokens ``[B, k, n_neighbours, dim]``.

    Row ``i`` of ``neighbours`` holds the tokens market ``i`` reads. Prepending the market token makes
    it position zero of a ``1 + n_neighbours`` sequence, and folding the market axis into the batch
    runs every market's sequence through one shared layer at once.
    """
    batch_size, n_markets, _, _ = neighbours.shape
    sequence = torch.cat([markets.unsqueeze(2), neighbours], dim=2)
    refined = layer(sequence.flatten(0, 1))[:, 0]
    return refined.unflatten(0, (batch_size, n_markets))


class SpeTransformerBlock(torch.nn.Module):
    """One refinement of the market embeddings, in two attention steps.

    - *across sides*: each market reads the routes it takes part in -- supply market ``i`` its ``n``
      outbound routes, demand market ``j`` its ``m`` inbound ones. This is the only step the route
      data enters, and the two directions run off one route grid built from the block's input
      embeddings, so neither side sees the other's update within the block.
    - *within side*: each market reads the same-side markets, through pair tokens carrying the price
      effects. This is the only step ``P`` and ``Q`` enter.

    The block owns its own embeddings as well as its own attention, so each depth builds the route
    and pair tokens its own way rather than re-running one shared projection -- the same choice
    ``NfgTransformerBlock`` makes for its ``joint_embedding``.

    Both steps share their attention layer between the sides, and the route grid is symmetric in the
    two roles, so nothing in the block distinguishes supply from demand. That is the side symmetry
    the ``-q`` convention buys.
    """

    def __init__(self, dim, n_heads, dim_ff, dropout, n_route_feats, n_pair_feats):
        super().__init__()

        self.route_embedding = torch.nn.Linear(dim + n_route_feats, dim)
        self.pair_embedding = torch.nn.Linear(dim + n_pair_feats, dim)
        self.across_sides = _encoder_layer(dim, n_heads, dim_ff, dropout)
        self.within_side = _encoder_layer(dim, n_heads, dim_ff, dropout)

    def embed_routes(self, supply, demand, route_feats):
        """Route tokens ``[B, m, n, dim]``, one per route.

        The two endpoints enter as a *sum* rather than a concatenation: the physics depends on
        ``pi_i + rho~_j`` and never on the two separately, so summing makes the role symmetry
        structural instead of something the layer has to learn. It also lets one grid serve both
        sides -- a route belongs to both its endpoints and reads the same way from either.
        """
        endpoints = supply.unsqueeze(2) + demand.unsqueeze(1)
        return torch.nn.functional.gelu(self.route_embedding(torch.cat([endpoints, route_feats], dim=-1)))

    def embed_pairs(self, markets, pair_feats):
        """Pair tokens ``[B, k, k, dim]``: entry ``(i, j)`` is the message market ``j`` sends ``i``.

        Unlike a route token, a pair token sits in exactly one market's sequence, so it is a directed
        message and carries only the *neighbour's* embedding -- market ``i`` is already position zero
        of its own sequence. The diagonal is kept: ``(P s)_i`` includes ``k = i``, the own-price
        effect.
        """
        neighbours = markets.unsqueeze(1).expand(-1, markets.size(1), -1, -1)
        return torch.nn.functional.gelu(self.pair_embedding(torch.cat([neighbours, pair_feats], dim=-1)))

    def forward(self, supply, demand, route_feats, supply_pairs, demand_pairs):
        """Refined market embeddings ``([B, m, dim], [B, n, dim])``."""
        # across sides first, so the block ends on the step carrying P and Q: the readout predicts a
        # price, and a price is a function of the own-side matrix and the own-side marginal
        routes = self.embed_routes(supply, demand, route_feats)
        supply, demand = (
            _attend_over_neighbours(self.across_sides, supply, routes),
            _attend_over_neighbours(self.across_sides, demand, routes.transpose(1, 2)),
        )
        supply = _attend_over_neighbours(self.within_side, supply, self.embed_pairs(supply, supply_pairs))
        demand = _attend_over_neighbours(self.within_side, demand, self.embed_pairs(demand, demand_pairs))
        return supply, demand


class SpeTransformerBackbone(torch.nn.Module):
    """Market tokens refined by alternating cross-side (route) and within-side (pair) attention.

    Args:
        dim: hidden dimension of the market, route and pair tokens.
        n_heads: number of attention heads in both steps.
        n_layers: number of refinement blocks.
        dim_ff: feed-forward dimension in each step.
        dropout: dropout rate.
        n_market_feats: per-market feature width -- the intercept, ``p_i`` or ``-q_j``.
        n_route_feats: per-route feature width. Two, for ``(c_ij, delta_ij)``.
        n_pair_feats: per-pair feature width. Two, for ``(P_ik, P_ki)``: the matrix is not symmetric,
            so a market's effect on another and the reverse are different numbers.
    """

    def __init__(
        self, dim, n_heads, n_layers, dim_ff, dropout, n_market_feats=1, n_route_feats=2, n_pair_feats=2
    ):
        super().__init__()

        # the only thing distinguishing one market token from another before attention is its
        # intercept. One projection for both sides, which is the side symmetry the -q convention buys
        self.market_embedding = torch.nn.Linear(n_market_feats, dim)
        self.blocks = torch.nn.ModuleList(
            SpeTransformerBlock(dim, n_heads, dim_ff, dropout, n_route_feats, n_pair_feats)
            for _ in range(n_layers)
        )
        # norm_first leaves the residual stream unnormalized at the end, so the backbone owns a final
        # norm rather than handing the readout an unbounded embedding
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, supply_feats, demand_feats, route_feats, supply_pairs, demand_pairs):
        """Per-market embeddings ``([B, m, dim], [B, n, dim])``.

        ``supply_feats`` is ``[B, m, n_market_feats]`` and ``demand_feats`` ``[B, n, n_market_feats]``
        (the intercepts, demand negated); ``route_feats`` is ``[B, m, n, n_route_feats]``;
        ``supply_pairs`` is ``[B, m, m, n_pair_feats]`` and ``demand_pairs``
        ``[B, n, n, n_pair_feats]``.
        """
        supply = self.market_embedding(supply_feats)
        demand = self.market_embedding(demand_feats)

        for block in self.blocks:
            supply, demand = block(supply, demand, route_feats, supply_pairs, demand_pairs)

        return self.norm(supply), self.norm(demand)


class StraightThroughRelu(torch.autograd.Function):
    """``relu`` forward; backward passes gradient on closed routes only when it would open them.

    The true derivative is zero wherever the route is closed, which under a self-supervised field
    loss is fatal rather than merely slow: at initialization the prices are near zero and the route
    costs are strictly positive, so *every* route is closed, the whole Jacobian is zero, and no
    signal reaches the network at all.

    Passing gradient unconditionally overcorrects. A route that is *correctly* closed carries slack
    -- shipping there would lose money -- and the field pushes its flow further negative, which shows
    up as pressure on the prices of the two markets it touches. Prices are shared down whole rows and
    columns and most routes are inactive, so that is a large force with no counterpart in the
    variational inequality, where slack constrains nothing.

    Masking to the one-sided case leaves exactly the violated complementarity conditions, which is
    the same rule ``dist_to_normal_cone`` scores: on a closed route it keeps only the part of the
    ascent field that wants flow to appear. Training gradient and validation residual then measure
    the same violation. The forward is untouched, so nonnegativity, hard zeros and the active set are
    exact.
    """

    @staticmethod
    def forward(ctx, surplus):
        ctx.save_for_backward(surplus)
        return torch.relu(surplus)

    @staticmethod
    def backward(ctx, grad_output):
        (surplus,) = ctx.saved_tensors
        # the field loss hands back dL/dx = -(ascent field), so "wants this route to open" is
        # grad_output < 0
        return grad_output * ((surplus > 0) | (grad_output < 0))


class FlowReadout(torch.nn.Module):
    """Market embeddings -> shipments ``[B, m, n]``, through one price per market.

    Complementarity plus ``F_ij = 0`` on the support gives the shipments in closed form from the two
    equilibrium price vectors, so the head predicts ``m + n`` scalars and reconstructs the ``m * n``
    flows rather than regressing them. Nonnegativity and the active set -- which routes carry flow at
    all, the combinatorial content of the problem -- come out of the ``relu`` instead of being
    learned.

    ``route_cost_min`` and ``route_cost_slope`` must be in **real units**: the reconstruction is a
    dimensional identity, and standardizing ``c`` and ``delta`` with independent scalers silently
    rescales the flows. Requires ``delta > 0``.

    A constant added to one price vector and subtracted from the other leaves the flows unchanged, so
    the prices are pinned only up to that one gauge direction. Harmless with the loss downstream of
    the reconstruction, where it is a flat direction in the head; it does mean the prices are not a
    well-posed target on their own.
    """

    def __init__(self, dim):
        super().__init__()
        self.price = torch.nn.Linear(dim, 1)

    def forward(self, supply, demand, route_cost_min, route_cost_slope):
        supply_price = self.price(supply)  # [B, m, 1]
        demand_price = self.price(demand).transpose(-2, -1)  # [B, 1, n]
        surplus = -(supply_price + demand_price + route_cost_min)
        return StraightThroughRelu.apply(surplus / route_cost_slope)
