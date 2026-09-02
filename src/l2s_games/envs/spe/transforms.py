"""Instance data -> the feature tensors the market-token backbone reads.

``SpeTransformerBackbone`` slots the instance data by *arity*: per-market scalars seed the market
tokens, cross-side pairs become route tokens and same-side pairs become pair tokens. This transform
is the only place that mapping is written down, so the backbone never touches raw instance fields.
"""

import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


def directed_pair_feats(effects):
    """Same-side pair features ``[k, k, 2]``: entry ``(i, j)`` is ``(effects_ij, effects_ji)``.

    A pair token sits in exactly one market's sequence, so it has to carry both directions itself:
    the price-effect matrices are asymmetric (``random_psd_plus_skew`` leaves the skew part free), so
    what market ``j`` does to market ``i`` and the reverse are different numbers, and market ``i``
    never gets to read row ``j`` from anywhere else.
    """
    return torch.stack([effects, effects.transpose(-2, -1)], dim=-1)


class BuildSPEFeats(BaseTransform):
    """Adds ``supply_feats``, ``demand_feats``, ``route_feats``, ``supply_pairs``, ``demand_pairs``.

    Shapes are per-instance and unbatched -- ``[m, 1]``, ``[n, 1]``, ``[m, n, 2]``, ``[m, m, 2]``,
    ``[n, n, 2]`` -- so ``torch.stack`` over a ``GraphToTensorDict``-ed dataset produces exactly the
    leading batch dim ``SpeTransformerBackbone.forward`` expects.

    The demand intercept is negated on the way in. That is the whole of the ``-q`` convention the
    backbone's side symmetry rests on: with ``rho~ = -rho`` both sides read "own matrix times own
    marginal, plus intercept", every module is shared between them, and the readout's price head
    predicts ``-v`` on demand tokens, which is the sign ``FlowReadout`` reconstructs flows in.

    Route features stay in **real units** (no scaling here): ``FlowReadout`` divides by the cost
    slope, so the reconstruction is a dimensional identity that independent scalers would break.
    """

    def __init__(self, mode=None):
        super().__init__()

        assert mode in ["full"], f"Mode must be 'full', got {mode}."
        self.mode = mode

    def forward(self, data: Data):
        data.supply_feats = data.p.unsqueeze(-1)
        data.demand_feats = -data.q.unsqueeze(-1)
        data.route_feats = torch.stack([data.c, data.delta], dim=-1)
        data.supply_pairs = directed_pair_feats(data.Gamma)
        data.demand_pairs = directed_pair_feats(data.Theta)
        return data
