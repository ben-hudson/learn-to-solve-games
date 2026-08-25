import torch

from tensordict import TensorDict


def random_psd_plus_skew(batch: int, k: int, kappa: float, eps: float, scale: float = 1.0, dtype=torch.float32):
    """J = (A A^T)/k + eps I + kappa (B - B^T).  sym(J) is PD; skew part free.

    kappa = 0.0  -> symmetric Jacobian (potential instance).
    kappa > 0.0  -> non-potential instance (monotone, asymmetric).
    """
    A = torch.randn(batch, k, k, dtype=dtype) * scale
    B = torch.randn(batch, k, k, dtype=dtype) * scale
    sym = A @ A.transpose(-1, -2) / k + eps * torch.eye(k, dtype=dtype)
    skew = kappa * (B - B.transpose(-1, -2))
    return sym + skew


class SpatialPriceEquilibrium:
    def __init__(
        self,
        supply_side_effects: torch.Tensor,
        demand_side_effects: torch.Tensor,
        supply_side_min: torch.Tensor,
        demand_side_min: torch.Tensor,
        route_cost_slope: torch.Tensor,
        route_cost_min: torch.Tensor,
    ):
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
        self.P = supply_side_effects
        self.Q = demand_side_effects
        self.p = supply_side_min
        self.q = demand_side_min
        self.delta = route_cost_slope
        self.c = route_cost_min

    @classmethod
    def random_bipartite(cls, n_supply, n_demand, kappa, eps, delta):
        P = random_psd_plus_skew(1, n_supply, kappa, eps).squeeze(0)
        Q = random_psd_plus_skew(1, n_demand, kappa, eps).squeeze(0)

        p = torch.empty(n_supply).uniform_(1.0, 3.0)
        q = torch.empty(n_demand).uniform_(4.0, 8.0)
        c = torch.empty(n_supply, n_demand).uniform_(0.5, 3.0)
        delta = torch.full((n_supply, n_demand), delta)

        return cls(P, Q, p, q, delta, c)

    def operator(self, x):
        supply = x.sum(dim=-1)  # s_i, total supplied from each supply market (..., m)
        demand = x.sum(dim=-2)  # d_j, total received at each demand market (..., n)
        supply_side_price = (self.P @ supply.unsqueeze(-1)).squeeze(-1) + self.p  # π(s) (..., m)
        demand_side_price = self.q - (self.Q @ demand.unsqueeze(-1)).squeeze(-1)  # ρ(d) (..., n)
        route_cost = self.delta * x + self.c  # cost slope * flow + min price (..., m, n)
        return supply_side_price.unsqueeze(-1) + route_cost - demand_side_price.unsqueeze(-2)

    @property
    def n_supply(self):
        return self.p.size(-1)

    @property
    def n_demand(self):
        return self.q.size(-1)

    def to_dict(self):
        return TensorDict(
            P=self.P,
            Q=self.Q,
            p=self.p,
            q=self.q,
            delta=self.delta,
            c=self.c,
        )

    @classmethod
    def from_dict(cls, data: TensorDict):
        return cls(
            data["P"],
            data["Q"],
            data["p"],
            data["q"],
            data["delta"],
            data["c"],
        )


if __name__ == "__main__":
    eq = SpatialPriceEquilibrium.random_bipartite(3, 4, 0.5, 0.1, 0.05)
    eq.operator(torch.rand(3, 4))

    instances = [SpatialPriceEquilibrium.random_bipartite(3, 4, 0.5, 0.1, 0.05) for _ in range(2)]
    batch = torch.stack([instance.to_dict() for instance in instances])
    SpatialPriceEquilibrium.from_dict(batch).operator(torch.rand(2, 3, 4))
