import torch


class GraphormerBackbone(torch.nn.Module):
    """Graphormer over the line graph: per-edge features + structure -> per-edge ``[B, E]`` prediction.

    The complete backbone network: embeds the per-edge ``feats`` to the hidden dim, adds degree
    embeddings and shortest-path spatial biases, applies a Transformer encoder, and reads out a scalar
    per edge. ``forward`` reads the line-graph structure from its input (``in_degree`` / ``out_degree``
    / ``spd``) rather than from buffers, so topology may vary across instances; only the
    embedding-table sizes are fixed at construction, inferred from a representative structure. A plain
    ``nn.Module`` (no training logic) so both ``FieldModel`` and ``SolutionModel`` can wrap it.

    Args:
        n_feats: per-edge feature width fed to the input embedding.
        in_degree / out_degree / spd: a representative instance's structure tensors; their maxima size
            the degree / spatial-encoding embedding tables.
        dim: hidden dimension of the Transformer.
        n_heads: number of attention heads.
        n_layers: number of Transformer encoder layers.
        dim_ff: feed-forward dimension in each Transformer layer.
        dropout: dropout rate.
    """

    def __init__(self, n_feats, in_degree, out_degree, spd, dim, n_heads, n_layers, dim_ff, dropout):
        super().__init__()

        self.n_heads = n_heads
        self.edge_embedding = torch.nn.Linear(n_feats, dim)
        self.in_degree_embedding = torch.nn.Embedding(int(in_degree.max()) + 1, dim)
        self.out_degree_embedding = torch.nn.Embedding(int(out_degree.max()) + 1, dim)
        self.spatial_encoding = torch.nn.Embedding(int(spd[spd.isfinite()].max()) + 1, n_heads)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.readout = torch.nn.Linear(dim, 1)

        # The MHA fast path on CPU mishandles the 3D per-head attention bias,
        # producing NaN in eval mode. A registered hook disables the fast path.
        for layer in self.encoder.layers:
            layer.self_attn.register_forward_hook(lambda m, i, o: None)

    def forward(self, inputs):
        """Per-edge prediction ``[B, E]`` from the batched inputs dict.

        ``inputs`` carries ``feats`` ``[B, E, n_feats]`` plus the line-graph structure ``in_degree`` /
        ``out_degree`` ``[B, E]`` and ``spd`` ``[B, E, E]`` (may contain ``inf``).
        """
        node_embedding = self.edge_embedding(inputs["feats"])
        in_degree, out_degree, spd = inputs["in_degree"], inputs["out_degree"], inputs["spd"]
        B, N, _ = node_embedding.shape

        in_deg = in_degree.clamp(0, self.in_degree_embedding.num_embeddings - 1).long()
        out_deg = out_degree.clamp(0, self.out_degree_embedding.num_embeddings - 1).long()
        embedding = node_embedding + self.in_degree_embedding(in_deg) + self.out_degree_embedding(out_deg)

        spd_clamped = spd.clamp(0, self.spatial_encoding.num_embeddings - 1).long()
        attn_bias = self.spatial_encoding(spd_clamped)  # [B, N, N, n_heads]
        attn_bias = attn_bias.permute(0, 3, 1, 2)  # [B, n_heads, N, N]
        attn_bias = attn_bias.masked_fill(spd.isinf().unsqueeze(1), -torch.inf)
        attn_bias = attn_bias.reshape(B * self.n_heads, N, N)

        encoded = self.encoder(embedding, mask=attn_bias)  # [B, N, dim]
        return self.readout(encoded).squeeze(-1)  # [B, E]
