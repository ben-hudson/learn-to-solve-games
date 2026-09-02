"""The NfgTransformer with everything the zero-sum case makes redundant taken out.

``NfgTransformerBackbone`` is a faithful port of an architecture built for n-player general-sum
games. Two of its features exist only to serve that generality, and in a zero-sum game
(``B = -A``, two players) they buy nothing:

- **The per-player joint-action grid.** The reference builds one token per player per joint action,
  each carrying the payoff *that player* receives. With ``B = -A`` the second copy is the first one
  negated -- the same numbers, no new information -- for twice the tokens.
- **The ``a2j`` stage.** Its job is to let the players at a joint action attend to each other, which
  is how a general-sum game's two payoff channels get reconciled into a shared view of the outcome.
  There is nothing to reconcile when one number describes the outcome for both players.

What is left is one shared token per joint action, pooled by the reference's ``a2p`` cross-attention
and mixed by its ``a2a`` self-attention. See ``ZeroSumTransformerBackbone`` for the one thing that
has to be *added* to keep it correct.
"""

import torch


class ZeroSumTransformerBlock(torch.nn.Module):
    """One refinement of the per-action embeddings against a single shared joint-action grid.

    Two stages, keeping the reference's names:

    - ``a2p`` (action-to-play): each action cross-attends, as a single query, over the ``T`` joint
      actions it takes part in -- the row player's action ``i`` over row ``i``, the column player's
      action ``j`` over column ``j``. This is the reduction from the ``T ** 2`` grid back to ``P * T``
      action tokens, and where the block's residual on its input embedding lives.
    - ``a2a`` (action-to-action): self-attention over all ``P * T`` action tokens at once, so an
      action can compare itself to its own player's alternatives -- which the grid never shows it,
      since a joint action holds one action of each player -- and to the co-player's.

    Both stages' weights are shared across players and actions, applied to sequences that differ only
    in which slice of the grid they hold. That sharing is the equivariance to relabelling either
    player's actions, and it is why the two players have to arrive already distinguishable (see
    ``ZeroSumTransformerBackbone``).

    Args:
        dim: hidden dimension carried by the action embeddings.
        n_heads: number of attention heads in both stages.
        dim_ff: feed-forward dimension in both stages.
        dropout: dropout rate.
        n_self_attend: number of ``a2a`` layers.
    """

    def __init__(self, dim, n_heads, dim_ff, dropout, n_self_attend):
        super().__init__()

        # both players' embeddings meet one payoff scalar, against the reference's one embedding and
        # one payoff per player-copy of the joint action
        self.joint_embedding = torch.nn.Linear(2 * dim + 1, dim)
        self.action_to_play = torch.nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.action_to_action = torch.nn.ModuleList(
            torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
            )
            for _ in range(n_self_attend)
        )

    def embed_joint_actions(self, payoffs, embedding):
        """Joint-action tokens ``[B, T, T, dim]``, one per cell of the grid.

        Entry ``(i, j)`` holds the two embeddings that meet there and the single payoff the row player
        receives. The two embeddings enter in fixed slots -- row first, column second -- so the token
        records which player played which action, not merely that the two met.
        """
        n_actions = payoffs.size(-1)
        row, col = embedding.unbind(dim=1)
        meeting = torch.cat(
            [
                row.unsqueeze(2).expand(-1, -1, n_actions, -1),
                col.unsqueeze(1).expand(-1, n_actions, -1, -1),
                payoffs.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.nn.functional.gelu(self.joint_embedding(meeting))

    def pool_joint_actions(self, joint, embedding):
        """Per-action embedding ``[B, P, T, dim]``: each action pools its own line of the grid.

        Transposing the grid for the column player puts the acting player's own axis in front of the
        co-player's for both, so one cross-attention handles both roles. The query is the *block
        input* embedding, so this stage's residual carries the input forward the way the reference's
        ``CrossAttention`` does.
        """
        lines = torch.stack([joint, joint.transpose(1, 2)], dim=1)  # [B, P, T_acting, T_coplayer, dim]
        keys = lines.flatten(0, 2)  # [B * P * T, T_coplayer, dim]
        queries = embedding.flatten(0, 2).unsqueeze(1)  # [B * P * T, 1, dim]
        pooled = self.action_to_play(queries, keys).squeeze(1)
        return pooled.unflatten(0, embedding.shape[:3])

    def forward(self, payoffs, embedding):
        """Refined per-action embedding ``[B, P, T, dim]`` from the row player's ``payoffs [B, T, T]``."""
        pooled = self.pool_joint_actions(self.embed_joint_actions(payoffs, embedding), embedding)

        tokens = pooled.flatten(1, 2)  # [B, P * T, dim]
        for layer in self.action_to_action:
            tokens = layer(tokens)
        return tokens.unflatten(1, pooled.shape[1:3])


class ZeroSumTransformerBackbone(torch.nn.Module):
    """NfgTransformer over a single shared joint-action grid -> per-action ``[B, P, T, dim]``.

    Reads one payoff matrix, since the column player's is its negation, and rebuilds one grid per
    block instead of one per player per block. That halves the grid and drops a whole attention stage
    against ``NfgTransformerBackbone``, while keeping what makes the architecture what it is: the
    action embeddings are the residual stream, the grid is re-derived from them and the payoffs every
    block, and the reduction back to action tokens is learned cross-attention rather than a mean.

    One thing has to be *added* back. In the reference the two players are told apart by their payoff
    channels -- the column player's tokens carry ``-A``, which is how it knows it minimizes what the
    row player maximizes. Collapse the grid and that signal is gone: with zero-initialized embeddings
    and weights shared across players, pooling row ``i`` and pooling column ``j`` are the same
    function of their line, so the module would score the minimizer's actions by the maximizer's
    preferences. So the roles are seeded instead: each player's actions start from a learned
    role vector, through the ``initial_action_embeddings`` seam the reference exposes for exactly
    this. One vector per role is constant across that player's actions, so equivariance to
    relabelling actions survives; equivariance to *swapping* the players does not, which is correct
    here -- a maximizer and a minimizer are not interchangeable.

    Args:
        dim: hidden dimension of the action embeddings.
        n_heads: number of attention heads.
        n_layers: number of refinement blocks.
        dim_ff: feed-forward dimension in each attention stage.
        dropout: dropout rate.
        n_feats: per-action feature width, ``0`` when ``forward`` is called without ``feats``.
        n_self_attend_per_block: number of ``a2a`` layers per block.
    """

    def __init__(self, dim, n_heads, n_layers, dim_ff, dropout, n_feats=0, n_self_attend_per_block=1):
        super().__init__()

        self.role_embedding = torch.nn.Embedding(2, dim)
        self.feature_embedding = torch.nn.Linear(n_feats, dim)
        self.blocks = torch.nn.ModuleList(
            ZeroSumTransformerBlock(dim, n_heads, dim_ff, dropout, n_self_attend_per_block) for _ in range(n_layers)
        )
        # norm_first leaves the residual stream unnormalized at the end, so the backbone owns a final
        # norm rather than handing the readout an unbounded embedding
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, payoffs, feats=None):
        """Per-action embedding ``[B, P, T, dim]``.

        ``payoffs`` is ``[B, P, T, T]`` for interchangeability with the other action-token backbones,
        but only the row player's matrix is read -- the column player's is its negation. ``feats`` is
        optional per-action features ``[B, P, T, n_feats]``, such as the query point a field model is
        evaluated at; they refine the role seed rather than replacing it.
        """
        batch_size, _, n_actions, _ = payoffs.shape

        embedding = self.role_embedding.weight.unsqueeze(1).expand(-1, n_actions, -1)  # [P, T, dim]
        embedding = embedding.expand(batch_size, -1, -1, -1)
        if feats is not None:
            embedding = embedding + self.feature_embedding(feats)

        for block in self.blocks:
            embedding = block(payoffs[:, 0], embedding)
        return self.norm(embedding)
