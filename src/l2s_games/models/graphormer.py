import torch

from .base import FieldModel


class GraphormerBackbone(torch.nn.Module):
    """Graphormer model for node-property predictions.

    Encodes features, degree embeddings, and shortest-path spatial
    biases, then applies a standard Transformer encoder followed by an
    MLP readout to produce per-node predictions.

    Args:
        dim: Hidden dimension of the Transformer.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        max_in_degree: Maximum in-degree for the degree embedding table.
        max_out_degree: Maximum out-degree for the degree embedding table.
        max_spd: Maximum shortest-path distance for the spatial encoding.
        dim_ff: Feed-forward dimension in each Transformer layer.
        dropout: Dropout rate.
    """

    def __init__(self, dim, n_heads, n_layers, max_in_degree, max_out_degree, max_spd, dim_ff, dropout):
        super().__init__()

        self.n_heads = n_heads
        self.in_degree_embedding = torch.nn.Embedding(max_in_degree + 1, dim)
        self.out_degree_embedding = torch.nn.Embedding(max_out_degree + 1, dim)
        self.spatial_encoding = torch.nn.Embedding(max_spd + 1, n_heads)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # The MHA fast path on CPU mishandles the 3D per-head attention bias,
        # producing NaN in eval mode. A registered hook disables the fast path.
        for layer in self.encoder.layers:
            layer.self_attn.register_forward_hook(lambda m, i, o: None)

    def forward(self, node_embedding, in_degree, out_degree, spd):
        """Produce per-node encodings from batched inputs.

        Args:
            node_embedding: Pre-computed node embeddings ``[B, N, dim]``.
            in_degree: Per-node in-degrees ``[B, N]``.
            out_degree: Per-node out-degrees ``[B, N]``.
            spd: Shortest-path distances ``[B, N, N]`` (may contain ``inf``).

        Returns:
            Encoded tensor of shape ``[B, N, dim]``.
        """
        B, N, _ = node_embedding.shape

        in_deg = in_degree.clamp(0, self.in_degree_embedding.num_embeddings - 1).long()
        out_deg = out_degree.clamp(0, self.out_degree_embedding.num_embeddings - 1).long()
        embedding = node_embedding + self.in_degree_embedding(in_deg) + self.out_degree_embedding(out_deg)

        spd_clamped = spd.clamp(0, self.spatial_encoding.num_embeddings - 1).long()
        attn_bias = self.spatial_encoding(spd_clamped)  # [B, N, N, n_heads]
        attn_bias = attn_bias.permute(0, 3, 1, 2)  # [B, n_heads, N, N]
        attn_bias = attn_bias.masked_fill(spd.isinf().unsqueeze(1), -torch.inf)
        attn_bias = attn_bias.reshape(B * self.n_heads, N, N)

        return self.encoder(embedding, mask=attn_bias)  # [B, N, dim]


class GraphormerFieldModel(FieldModel):
    """Graphormer over the line graph: per-edge features + structure -> per-edge field value.

    ``forward`` reads the line-graph structure from its input (``in_degree`` / ``out_degree`` /
    ``spd``) rather than from buffers, so topology may vary across instances. Only the
    embedding-table sizes are fixed at construction, inferred from a representative structure.
    """

    def __init__(
        self, n_feats, in_degree, out_degree, spd, lr, dim=64, n_heads=4, n_layers=4, dim_ff=128, dropout=0.0, **kwargs
    ):
        super().__init__(lr, **kwargs)
        self.edge_embedding = torch.nn.Linear(n_feats, dim)
        self.backbone = GraphormerBackbone(
            dim,
            n_heads,
            n_layers,
            max_in_degree=int(in_degree.max()),
            max_out_degree=int(out_degree.max()),
            max_spd=int(spd[spd.isfinite()].max()),
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.readout = torch.nn.Linear(dim, 1)

    def forward(self, inputs):
        node_embedding = self.edge_embedding(inputs["feats"])
        encoded = self.backbone(node_embedding, inputs["in_degree"], inputs["out_degree"], inputs["spd"])
        return self.readout(encoded).squeeze(-1)
