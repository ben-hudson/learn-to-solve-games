import torch


class NashMLPBackbone(torch.nn.Module):
    """The NE-approximator network of Duan et al. 2023 (Section 6.1), as a drop-in backbone.

    A fully connected network over the whole flattened game utility (no graph inductive bias):
    ``n_layers`` hidden layers of ``hidden_dim`` nodes, batch normalization without learnable
    parameters before each ReLU, and every parameter projected into [0, 1] (see ``forward``).
    The projection makes the network a Lipschitz hypothesis class (the paper's Definition 5.9),
    which satisfies the covering-number assumption behind its PAC-learnability result -- and in
    practice bounds the output scale, so downstream softmax logits cannot saturate.

    Matches ``GraphormerBackbone``'s calling convention -- per-player ``feats`` in, per-player
    ``[B, N, dim]`` embedding out, with the wrapping model owning the readout -- so the two are
    interchangeable; the graph-structure arguments are accepted and ignored.

    Args:
        n_feats: per-player feature width (the flattened payoff matrix).
        n_players: number of player nodes; input and output are joint over all of them.
        dim: per-player embedding dimension handed to the wrapping model's readout.
        hidden_dim: hidden-layer width (1024 in the paper).
        n_layers: number of hidden layers (4 in the paper).
    """

    def __init__(self, n_feats, n_players, dim, hidden_dim=1024, n_layers=4):
        super().__init__()

        self.n_players = n_players
        self.dim = dim

        widths = [n_players * n_feats] + [hidden_dim] * n_layers
        hidden_layers = [
            torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.BatchNorm1d(out_dim, affine=False),
                torch.nn.ReLU(),
            )
            for in_dim, out_dim in zip(widths[:-1], widths[1:])
        ]
        self.mlp = torch.nn.Sequential(*hidden_layers, torch.nn.Linear(hidden_dim, n_players * dim))

    def forward(self, feats, in_degree, out_degree, spd):
        """Per-player embedding ``[B, N, dim]`` from per-player ``feats`` ``[B, N, n_feats]``.

        Parameters are only ever read here, so clamping them on entry is projected SGD: it is
        equivalent to clamping after each optimizer step (the paper's parameter clipping).
        """
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.clamp_(0, 1)

        embedding = self.mlp(feats.flatten(start_dim=1))
        return embedding.view(-1, self.n_players, self.dim)
