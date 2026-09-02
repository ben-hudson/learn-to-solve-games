"""Readouts from market embeddings to an SPE iterate."""

import torch


class PriceReadout(torch.nn.Module):
    """Market embeddings -> prices ``[B, m + n]``, supply markets first.

    The price-space unknowns, so this is the head ``PriceSPE.operator`` scores directly. Unlike the
    flow-space readout it needs no reconstruction and no route data: the prices *are* the iterate,
    they are free (``PriceSPE`` has no constraint set), and the shipments they imply are feasible for
    any sign, so nothing here has to be projected or clamped.

    One head shared between the sides, like ``FlowReadout``'s. The backbone is fed the symmetrized
    convention -- a demand market's intercept enters as ``-q``, so its token behaves like a supply
    market's -- and the head therefore predicts in that convention too: ``u_i`` on a supply token and
    ``rho~_j = -v_j`` on a demand token. Negating the demand block on the way out puts the vector back
    in the units the operator reads, and keeps the whole model free of any side-type parameter.
    """

    def __init__(self, dim):
        super().__init__()
        self.price = torch.nn.Linear(dim, 1)

    def forward(self, supply, demand):
        supply_price = self.price(supply).squeeze(-1)  # u [B, m]
        demand_price = -self.price(demand).squeeze(-1)  # v [B, n]
        return torch.cat([supply_price, demand_price], dim=-1)
