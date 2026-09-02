"""The pre-norm attention primitives of ``google-deepmind/nfg_transformer``, in PyTorch.

A transcription of the ``Dense`` / ``Attention`` / ``SelfAttention`` / ``CrossAttention`` modules of
the reference's ``network.py``. They are the generic Transformer parts of the architecture -- nothing
here knows about games -- but they are not ``torch.nn.TransformerEncoderLayer``: the reference makes
several choices that layer does not expose, and they are the whole reason this file exists.

- RMSNorm rather than LayerNorm, everywhere.
- Query and key are RMSNormed *after* their projection and *before* the head split, so the learned
  scale is per channel across all heads. This and the unbiased q/k/v projections below are the
  ViT-22B recipe (https://arxiv.org/pdf/2302.05442.pdf) the reference cites.
- The q/k/v projections carry no bias; only the attention's output projection does.
- The attention width is decoupled from the embedding width. The paper's models run 128 attention
  channels against action embeddings of 32, 64 or 128, which is not expressible when the two are
  tied.
- ``CrossAttention`` normalizes its keys and values as well as its query, and is cross-attention plus
  feed-forward only -- no self-attention sublayer, unlike ``torch.nn.TransformerDecoderLayer``.

The reference has no dropout. The ``dropout`` argument is this repo's, at the standard residual
position; at ``0.0`` -- what every caller passes -- these modules are the reference exactly.
"""

import torch


def gelu(x):
    """The reference's activation. ``jax.nn.gelu`` defaults to the tanh approximation; torch's does not."""
    return torch.nn.functional.gelu(x, approximate="tanh")


def rms_norm(dim):
    """RMSNorm over the last axis, at Haiku's ``eps``. Torch's default eps is the dtype's, not 1e-5."""
    return torch.nn.RMSNorm(dim, eps=1e-5)


class Dense(torch.nn.Module):
    """The feed-forward sublayer: widen, gelu, project back.

    The reference fixes the widening at ``4 * dim``; taking ``dim_ff`` as an argument is the same
    module with the factor left to the caller, which is what the sibling backbones' signatures use.

    Args:
        dim: input and output width.
        dim_ff: hidden width.
    """

    def __init__(self, dim, dim_ff):
        super().__init__()

        self.widen = torch.nn.Linear(dim, dim_ff)
        self.project = torch.nn.Linear(dim_ff, dim)

    def forward(self, x):
        return self.project(gelu(self.widen(x)))


class Attention(torch.nn.Module):
    """Multi-headed {cross, self}-attention with normalized, unbiased queries and keys.

    Args:
        dim: width of the query input, and of the output.
        dim_qkv: width of the query, key and value projections, split across the heads. Independent
            of ``dim``.
        n_heads: number of attention heads.
        dim_kv: width of the key/value input. Defaults to ``dim``, which is the self-attention case
            and, in this architecture, the cross-attention case too.
    """

    def __init__(self, dim, dim_qkv, n_heads, dim_kv=None):
        super().__init__()

        self.n_heads = n_heads
        # https://arxiv.org/pdf/2302.05442.pdf: unbiased projections, and the query and key
        # normalized before the softmax to keep the logits from growing during training
        self.query = torch.nn.Linear(dim, dim_qkv, bias=False)
        self.key = torch.nn.Linear(dim_kv or dim, dim_qkv, bias=False)
        self.value = torch.nn.Linear(dim_kv or dim, dim_qkv, bias=False)
        self.query_norm = rms_norm(dim_qkv)
        self.key_norm = rms_norm(dim_qkv)
        self.output = torch.nn.Linear(dim_qkv, dim)

    def split_heads(self, projected):
        """``[B, T, dim_qkv] -> [B, n_heads, T, dim_qkv // n_heads]``, the layout attention wants."""
        return projected.unflatten(-1, (self.n_heads, -1)).transpose(-3, -2)

    def forward(self, inputs_q, inputs_kv):
        """Attended queries ``[B, T_q, dim]`` from ``inputs_q [B, T_q, dim]``, ``inputs_kv [B, T_kv, dim]``."""
        # the norms run over the full projection, before the split, so one scale spans all heads
        queries = self.split_heads(self.query_norm(self.query(inputs_q)))
        keys = self.split_heads(self.key_norm(self.key(inputs_kv)))
        values = self.split_heads(self.value(inputs_kv))

        # scaled_dot_product_attention's default scale is 1 / sqrt(dim_qkv // n_heads), the
        # reference's; with no mask to apply there is nothing else in `attend` to reproduce
        attended = torch.nn.functional.scaled_dot_product_attention(queries, keys, values)
        return self.output(attended.transpose(-3, -2).flatten(-2))


class SelfAttention(torch.nn.Module):
    """Pre-norm self-attention followed by a feed-forward, each residual.

    Args:
        dim: width of the residual stream.
        dim_qkv: width of the attention's projections.
        n_heads: number of attention heads.
        dim_ff: hidden width of the feed-forward.
        dropout: dropout rate on each sublayer's output. The reference has none.
    """

    def __init__(self, dim, dim_qkv, n_heads, dim_ff, dropout):
        super().__init__()

        self.qkv_norm = rms_norm(dim)
        self.attention = Attention(dim, dim_qkv, n_heads)
        self.dense_norm = rms_norm(dim)
        self.dense = Dense(dim, dim_ff)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.qkv_norm(x)
        x = x + self.dropout(self.attention(normalized, normalized))
        return x + self.dropout(self.dense(self.dense_norm(x)))


class CrossAttention(torch.nn.Module):
    """Pre-norm cross-attention followed by a feed-forward, each residual on the query.

    Both sides of the attention are normalized, by separate norms, and the residual carries the query
    forward -- so the module reads the key-value sequence into whatever the query already holds.

    Args:
        dim: width of the query, which is the residual stream, and of the key-value input.
        dim_qkv: width of the attention's projections.
        n_heads: number of attention heads.
        dim_ff: hidden width of the feed-forward.
        dropout: dropout rate on each sublayer's output. The reference has none.
    """

    def __init__(self, dim, dim_qkv, n_heads, dim_ff, dropout):
        super().__init__()

        self.query_norm = rms_norm(dim)
        self.key_value_norm = rms_norm(dim)
        self.attention = Attention(dim, dim_qkv, n_heads)
        self.dense_norm = rms_norm(dim)
        self.dense = Dense(dim, dim_ff)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, queries, key_values):
        attended = self.attention(self.query_norm(queries), self.key_value_norm(key_values))
        x = queries + self.dropout(attended)
        return x + self.dropout(self.dense(self.dense_norm(x)))
