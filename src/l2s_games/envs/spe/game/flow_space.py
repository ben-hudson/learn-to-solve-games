"""The spatial price equilibrium in its two formulations.

Both classes carry blocks of one canonical variational inequality (Dafermos & Nagurney 1984),

    <c(x*) + u* - v*, x - x*> + <s(u*) - R x*, u - u*> + <C x* - d(v*), v - v*>  >=  0

whose variables are shipments ``x``, supply prices ``u`` and demand prices ``v``. ``FlowSPE`` keeps
the shipments and eliminates the prices, so its operator is the route block; ``PriceSPE`` keeps the
prices and eliminates the shipments, so its operator is the two market blocks. Same problem, same
solution set, related by ``to_price_spe`` / ``to_flow_spe`` on instances and ``prices`` / ``flows``
on points.

Sign convention (assumed of every caller, and load-bearing). ``operator`` returns the monotone map
in the standard variational-inequality sign, so the convergent dynamics **ascend its negation**:
``z <- project(z - h operator(z))``. That is what ``dist_to_normal_cone`` scores and what the
``algorithms`` registry expects, hence the ``-instance.operator(z)`` at every call site.
"""

import torch

from tensordict import tensorclass
from torch_geometric.data import Data

from .utils import random_psd_plus_skew, sample_market_data


@tensorclass
class FlowSPE:
    supply_side_effects: torch.Tensor
    demand_side_effects: torch.Tensor
    supply_side_min: torch.Tensor
    demand_side_min: torch.Tensor
    route_cost_slope: torch.Tensor
    route_cost_min: torch.Tensor
    # F(x) = Mx + b, x>=0
    # M = R^T P R + C^T Q C + δI
    # b = R^T p + c − C^T q
    # x_ij = quantity shipped from supply market i to demand market j
    # -> sum over consumers j is total supply shipped from supply market i (s_i)
    # -> sum over suppliers i is total demand received at demand market j (d_j)
    # P = supply-side price effects
    # Q = demand-side price effects
    # π_i(s) = Σ_k P_ik s_k + p_i (supply price)
    # ρ_j(d) = q_j − Σ_l Q_jl d_l (demand price)
    # p and q are intercepts - minimum prices
    # Shipping cost:
    # T_ij(x) = π_i(s) + c_ij + δ x_ij
    # c is a fixed cost and delta is a linear flow component
    # x is a dense (..., n_supply, n_demand) shipment matrix, so R and C are just
    # row and column sums; leading dims are classical batch dims.
    # eq-condition is F(x) = pi_on_routes + c + δx − rho_on_routes

    @classmethod
    def random_bipartite(cls, n_supply, n_demand, kappa, eps, delta, scale=1.0):
        P = random_psd_plus_skew(1, n_supply, kappa, eps, scale=scale).squeeze(0)
        Q = random_psd_plus_skew(1, n_demand, kappa, eps, scale=scale).squeeze(0)
        p, q, delta, c = sample_market_data(n_supply, n_demand, delta)

        return cls(P, Q, p, q, delta, c)

    def prices(self, x):
        """Market prices ``(..., m + n)`` induced by shipments ``x``: ``π(s)`` then ``ρ(d)``.

        Defined at any ``x``, not just an equilibrium one -- the inverse price functions are what
        this formulation is written in, so the prices are always available as a by-product. Inverse
        of ``PriceSPE.flows`` only at a point satisfying the route complementarity conditions; see
        that method.
        """
        supply = x.sum(dim=-1)  # s_i, total supplied from each supply market (..., m)
        demand = x.sum(dim=-2)  # d_j, total received at each demand market (..., n)
        supply_side_price = (self.P @ supply.unsqueeze(-1)).squeeze(-1) + self.p  # π(s) (..., m)
        demand_side_price = self.q - (self.Q @ demand.unsqueeze(-1)).squeeze(-1)  # ρ(d) (..., n)
        return torch.cat([supply_side_price, demand_side_price], dim=-1)

    def operator(self, x):
        supply_side_price, demand_side_price = self.prices(x).split([self.n_supply, self.n_demand], dim=-1)
        route_cost = self.delta * x + self.c  # cost slope * flow + min price (..., m, n)
        return supply_side_price.unsqueeze(-1) + route_cost - demand_side_price.unsqueeze(-2)

    # def to_price_spe(self):
    #     """The same instance in price space. Inverting a matrix with PD symmetric part gives another
    #     one (``x^T M^-1 x = y^T sym(M) y > 0`` for ``y = M^-1 x``), so the result is always a
    #     strongly monotone instance.
    #     """
    #     return PriceSPE(
    #         torch.linalg.inv(self.P),
    #         torch.linalg.inv(self.Q),
    #         self.p,
    #         self.q,
    #         self.delta,
    #         self.c,
    #     )

    @property
    def n_supply(self):
        return self.p.size(-1)

    @property
    def n_demand(self):
        return self.q.size(-1)

    def to_data(self):
        return Data(
            P=self.supply_side_effects,
            Q=self.demand_side_effects,
            p=self.supply_side_min,
            q=self.demand_side_min,
            delta=self.route_cost_slope,
            c=self.route_cost_min,
        )

    @classmethod
    def from_data(cls, data: Data):
        return cls(
            data["P"],
            data["Q"],
            data["p"],
            data["q"],
            data["delta"],
            data["c"],
        )
