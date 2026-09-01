"""NfgTransformer (Liu et al., ICLR 2024) as a payoff backbone.

A PyTorch port of the encoder in ``google-deepmind/nfg_transformer`` (JAX/Haiku), adapted to this
repo's backbone contract: ``payoffs [B, P, T, T] -> per-action embedding [B, P, T, dim]``. The
reference implementation operates on a single game with per-player action counts and leaves batching
to ``jax.vmap``; here the batch is explicit and both players share one action count ``T``, which is
what the rest of the pipeline assumes.

The attention sublayers are the reference's, transcribed in ``nfg_attention`` rather than taken from
``torch.nn`` -- see that module for the choices the stock layers cannot express. What remains of the
reference is here: the three-stage block, the weight sharing that makes it equivariant, and the
tensor layout that stands in for its ``jax.vmap``.

Three deviations from the reference remain, all deliberate. Two are scope reductions: two players
rather than n, and no joint-action mask (nothing upstream produces one -- every joint action of a
dense payoff tensor is observed). The third is an addition: ``partially_amortized`` seeds the action
embeddings from per-action features, through the ``initial_action_embeddings`` argument the reference
exposes and never passes.
"""

import torch

from l2s_games.models.nfg_attention import CrossAttention, SelfAttention, gelu, rms_norm


class NfgTransformerBlock(torch.nn.Module):
    """One refinement of the per-action embeddings, in the reference's three attention stages.

    Named after the Haiku scopes they carry in the reference implementation:

    - ``a2j`` (action-to-joint): each joint action is embedded from the acting action's own embedding
      and the payoff *that player* receives there, then the players at a joint action attend to each
      other. This is the only stage the payoffs enter, and they enter as token content -- a scalar per
      player per joint action -- so a pass can read a linear functional of them.
    - ``a2p`` (action-to-play): each action cross-attends, as a single query, over the ``T`` joint
      actions it takes part in. This is the reduction from the ``T ** 2`` grid back to ``P * T``
      tokens, and it is where the block's residual on its input embedding lives.
    - ``a2a`` (action-to-action): self-attention over all ``P * T`` action tokens at once, so an
      action can compare itself to its own alternatives and to the co-player's.

    Every stage's parameters are shared across players and across actions -- one set of weights
    applied to sequences that differ only in which slice of the grid they hold. That sharing is the
    equivariance: permuting a player's actions permutes the sequences handed to each stage, never the
    weights, and nothing in the module is sized by ``T`` or ``P``.

    Args:
        dim: hidden dimension carried by the action embeddings.
        dim_qkv: hidden dimension of every stage's attention projections.
        n_heads: number of attention heads in every stage.
        dim_ff: feed-forward dimension in every stage.
        dropout: dropout rate.
        n_self_attend: number of ``a2a`` layers per block.
    """

    def __init__(self, dim, dim_qkv, n_heads, dim_ff, dropout, n_self_attend):
        super().__init__()

        # the payoff arrives as a single channel appended to the acting player's own embedding, so
        # this input width is independent of the player count
        self.joint_embedding = torch.nn.Linear(dim + 1, dim)
        self.action_to_joint = SelfAttention(dim, dim_qkv, n_heads, dim_ff, dropout)
        self.action_to_play = CrossAttention(dim, dim_qkv, n_heads, dim_ff, dropout)
        # the reference builds these one at a time, so unlike the two stages above they do not share
        # weights across the repeats
        self.action_to_action = torch.nn.ModuleList(
            SelfAttention(dim, dim_qkv, n_heads, dim_ff, dropout) for _ in range(n_self_attend)
        )

    def embed_joint_actions(self, payoffs, embedding):
        """Joint-action tokens ``[B, P, T, T, dim]``, one per player per joint action.

        Entry ``[b, p, i, j]`` pairs *player p's own* action embedding at ``(i, j)`` -- action ``i``
        for the row player, ``j`` for the column player -- with the payoff player ``p`` receives
        there. Each player therefore reads the grid from its own side, which is what makes the ``a2j``
        attention that follows an exchange between the players at a joint action rather than a pooling
        of one shared token.
        """
        n_actions = payoffs.size(-1)
        row, col = embedding.unbind(dim=1)
        acting = torch.stack(
            [
                row.unsqueeze(2).expand(-1, -1, n_actions, -1),
                col.unsqueeze(1).expand(-1, n_actions, -1, -1),
            ],
            dim=1,
        )  # [B, P, T_row, T_col, dim]
        return gelu(self.joint_embedding(torch.cat([acting, payoffs.unsqueeze(-1)], dim=-1)))

    def attend_across_players(self, joint):
        """Refined joint-action tokens ``[B, P, T, T, dim]``: the players at a joint action attend.

        The player axis becomes the sequence -- ``P`` tokens per joint action, every joint action of
        every instance folded into the batch -- so what each player learns here is what the outcome
        looks like to the other.
        """
        batch_size, _, n_actions, _, _ = joint.shape
        tokens = joint.movedim(1, -2).flatten(0, 2)  # [B * T * T, P, dim]
        refined = self.action_to_joint(tokens)
        return refined.unflatten(0, (batch_size, n_actions, n_actions)).movedim(-2, 1)

    def pool_joint_actions(self, joint, embedding):
        """Per-action embedding ``[B, P, T, dim]``: each action pools the joint actions it takes part in.

        Moving the acting player's own axis in front of the co-player's puts an action's joint actions
        on the sequence axis, uniformly for both players; the query is the *block input* embedding, so
        the cross-attention's residual carries the input forward the way the reference's
        ``CrossAttention`` does.
        """
        acting_axis_first = torch.stack(
            [own.movedim(1 + player, 1) for player, own in enumerate(joint.unbind(dim=1))], dim=1
        )  # [B, P, T_acting, T_coplayer, dim]
        keys = acting_axis_first.flatten(0, 2)  # [B * P * T, T_coplayer, dim]
        queries = embedding.flatten(0, 2).unsqueeze(1)  # [B * P * T, 1, dim]
        pooled = self.action_to_play(queries, keys).squeeze(1)
        return pooled.unflatten(0, embedding.shape[:3])

    def forward(self, payoffs, embedding):
        """Refined per-action embedding ``[B, P, T, dim]`` from ``payoffs [B, P, T, T]``."""
        joint = self.attend_across_players(self.embed_joint_actions(payoffs, embedding))
        pooled = self.pool_joint_actions(joint, embedding)

        tokens = pooled.flatten(1, 2)  # [B, P * T, dim]
        for layer in self.action_to_action:
            tokens = layer(tokens)
        return tokens.unflatten(1, pooled.shape[1:3])


class NfgTransformerBackbone(torch.nn.Module):
    """Action tokens refined against a joint-action grid rebuilt every block -> ``[B, P, T, dim]``.

    The third point in the design space the two sibling backbones bracket. ``PayoffBiasBackbone``
    keeps ``P * T`` action tokens and lets the game in only through an attention bias;
    ``AxialBackbone`` pays ``T ** 2`` tokens to carry the payoffs as token content and reads the
    actions back out by averaging the grid's lines. This does both, per block: it *materializes* the
    ``T ** 2`` grid from the current action embeddings and the payoffs, refines it, and reduces it
    back to ``P * T`` action tokens by learned cross-attention rather than by a mean. The action
    embeddings, not the grid, are the residual stream -- so the grid is re-derived from scratch each
    block against embeddings that have already seen the payoffs once.

    Costlier than either sibling (the grid is rebuilt ``n_layers`` times, and each block runs three
    attention stages rather than one), and the reference's answer to why: the reduction is learned and
    the payoffs stay in the tokens, so neither the soft-argmax bottleneck of a bias nor the
    permutation-invariant mean of a line readout limits what an action embedding can encode.

    Equivariant to permutations of either player's actions: initial embeddings are zeros, there is no
    positional embedding anywhere, and the payoffs (with the query point, when partially amortized --
    itself one coordinate per action, so it permutes with them) are the only thing that distinguishes
    one action from another. Nothing in the blocks is sized by the action count, so a trained module
    runs on games of any size. Not equivariant to *swapping the players*, for one reason: the
    reference's final norm is per player, so the player index reaches the output there and nowhere
    else.

    Two players only -- ``embed_joint_actions`` builds the grid from the two axes of the payoff
    tensor. (The reference is n-player; generalizing means broadcasting each player's embedding along
    its own axis of an n-dimensional grid.)

    Args:
        dim: hidden dimension of the action embeddings, the paper's ``D``.
        n_heads: number of attention heads, the paper's ``H``.
        n_layers: number of refinement blocks, the paper's ``K``.
        dim_ff: feed-forward dimension in each attention stage. The reference fixes it at ``4 * dim``.
        dropout: dropout rate. The reference has none.
        n_feats: per-action feature width. Read only when ``partially_amortized``.
        n_self_attend_per_block: number of ``a2a`` layers per block, the paper's ``A``.
        partially_amortized: add a projection of the per-action features to the seed embedding. The
            reference has no counterpart for it, so it is off by default and the fully amortized
            backbone is exactly the reference's.
        dim_qkv: hidden dimension of every attention's projections. Independent of ``dim``: the paper
            runs 128 against action embeddings of 32, 64 and 128. Defaults to ``dim``.
        n_players: number of players, needed up front only because the final norms are per player.
    """

    def __init__(
        self,
        dim,
        n_heads,
        n_layers,
        dim_ff,
        dropout,
        n_feats=0,
        n_self_attend_per_block=1,
        partially_amortized=False,
        dim_qkv=None,
        n_players=2,
    ):
        super().__init__()

        self.dim = dim
        # The reference seeds the blocks with zeros, and with nothing else: it exposes the seed as
        # `initial_action_embeddings`, but never passes it. Fully amortized, we do the same and own no
        # module that could perturb those zeros -- not even a bias. Partially amortized, the query
        # point has to reach the tokens somehow, and the seed is the only place it fits.
        self.feature_embedding = torch.nn.Linear(n_feats, dim) if partially_amortized else None
        self.blocks = torch.nn.ModuleList(
            NfgTransformerBlock(dim, dim_qkv or dim, n_heads, dim_ff, dropout, n_self_attend_per_block)
            for _ in range(n_layers)
        )
        # Pre-norm blocks leave the residual stream unnormalized at the end, so the backbone owns a
        # final norm rather than handing the readout an unbounded embedding. One per player, as in the
        # reference -- the only parameter in the encoder that a player index selects.
        self.norms = torch.nn.ModuleList(rms_norm(dim) for _ in range(n_players))

    def forward(self, payoffs, feats=None):
        """Per-action embedding ``[B, P, T, dim]``.

        ``payoffs`` is ``[B, P, T, T]``; ``feats`` is per-action features ``[B, P, T, n_feats]``, such
        as the query point a field model is evaluated at, and is read only when partially amortized.
        """
        batch_size, n_players, n_actions, _ = payoffs.shape

        embedding = payoffs.new_zeros(batch_size, n_players, n_actions, self.dim)
        if self.feature_embedding is not None:
            embedding = embedding + self.feature_embedding(feats)

        for block in self.blocks:
            embedding = block(payoffs, embedding)
        return torch.stack([norm(embedding[:, player]) for player, norm in enumerate(self.norms)], dim=1)
