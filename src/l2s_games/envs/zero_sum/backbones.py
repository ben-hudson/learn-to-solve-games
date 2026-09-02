import torch


class ActionTokenBackbone(torch.nn.Module):
    """Adapts an action-token backbone to the calling convention of the zero-sum models.

    The action-token backbones (``PayoffBiasBackbone``, ``AxialBackbone``, ``NfgTransformerBackbone``)
    read a payoff matrix and per-action features, while the models hand their backbone the flat
    per-player vector ``BuildZeroSumFeats`` builds: the flattened payoff matrix, preceded by the query
    point when the model is partially amortized. This splits that vector back apart.

    Matches ``GraphormerBackbone``'s calling convention so the two are interchangeable, and like
    ``NashMLPBackbone`` it accepts and ignores the graph-structure arguments -- an action token knows
    which player it belongs to and which payoffs it meets, which is all the structure there is.

    Returns a per-action ``[B, P, T, dim]`` embedding where a graph backbone returns a per-player
    ``[B, P, dim]``, so pair it with ``ActionReadout``.

    Args:
        backbone: the action-token backbone to wrap.
        n_actions: actions per player, which is what splits the payoffs from the query point.
    """

    def __init__(self, backbone, n_actions):
        super().__init__()

        self.backbone = backbone
        self.n_actions = n_actions

    def forward(self, feats, in_degree, out_degree, spd):
        """Per-action embedding ``[B, P, T, dim]`` from flat per-player ``feats`` ``[B, P, n_feats]``."""
        n_payoffs = self.n_actions**2
        point, payoffs = feats.split([feats.size(-1) - n_payoffs, n_payoffs], dim=-1)
        # the point is one coordinate per action, and is absent exactly when the model is fully
        # amortized -- then the backbones seed their tokens from the payoffs alone
        point_feats = point.unsqueeze(-1) if point.size(-1) else None
        return self.backbone(payoffs.unflatten(-1, (self.n_actions, self.n_actions)), point_feats)


class ActionReadout(torch.nn.Module):
    """Scores each action token independently -> ``[B, P, T]``.

    The per-player counterpart, ``Linear(dim, n_actions)``, fixes the action count in its output
    width; scoring one token at a time leaves the model as size-agnostic as the action-token
    backbones themselves are.
    """

    def __init__(self, dim):
        super().__init__()

        self.score = torch.nn.Linear(dim, 1)

    def forward(self, embedding):
        return self.score(embedding).squeeze(-1)
