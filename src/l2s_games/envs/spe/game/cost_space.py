import torch

from tensordict import tensorclass
from torch_geometric.data import Data

from .utils import random_psd_plus_skew, sample_market_data


@tensorclass
class PriceSPE:
    supply_side_effects: torch.Tensor  # (Gamma) interactions between supply markets
    demand_side_effects: torch.Tensor  # (Theta) interactions between demand markets
    supply_side_min: torch.Tensor  # min supply side price
    demand_side_min: torch.Tensor  # min demand side price
    route_cost_slope: torch.Tensor  # route cost per unit flow
    route_cost_min: torch.Tensor  # min route cost

    # Ψ(u, v) = 0, u and v free (prices may be negative)
    # Ψ_u = Γ(u − p) − R x(u, v)     market clearing at each supply market
    # Ψ_v = C x(u, v) − Θ(q − v)     market clearing at each demand market
    # Γ = P^{-1} = supply-side quantity effects, Θ = Q^{-1} = demand-side quantity effects
    # u_i = price at supply market i, v_j = price at demand market j
    # -> unknowns live on markets, not routes: m + n of them instead of m·n
    # s_i(u) = Σ_k Γ_ik (u_k − p_k) (supply function, what market i ships out at price u)
    # d_j(v) = Σ_l Θ_jl (q_l − v_l) (demand function, what market j takes in at price v)
    # these invert π and ρ: s = P^{-1}(π − p) and d = Q^{-1}(q − ρ)
    # Implied shipments, from solving route complementarity in closed form (needs δ > 0):
    # x_ij(u, v) = max(0, v_j − u_i − c_ij) / δ_ij
    # -> x_ij > 0 exactly when route ij is profitable, and then u_i + c_ij + δ_ij x_ij = v_j
    # -> x >= 0 holds automatically, whatever sign the prices take
    # R and C are still row and column sums, so R x = s(u) and C x = d(v) at equilibrium
    # eq-condition is Ψ = supply_function − shipped_out on supply markets,
    #                     received_in − demand_function on demand markets
    # Ψ is affine on each active set A_ij = 1[v_j > u_i + c_ij], with Jacobian
    # [[Γ + diag(A1/δ), −A/δ], [−A^T/δ, Θ + diag(A^T1/δ)]]
    # -> bipartite Laplacian weighted by 1/δ plus PD block-diagonal, so sym part is PD
    # prices are a dense (..., m + n) vector, supply markets first; leading dims are classical
    # batch dims.

    @classmethod
    def random_bipartite(cls, n_supply, n_demand, kappa, eps, delta, scale=1.0):
        """Sample ``Γ`` and ``Θ`` directly, so no instance ever needs an inverse at operator time.

        Same signature as ``FlowSPE.random_bipartite``, so either drops into a ``sample_fn`` slot,
        but this is *not* that distribution pushed through inversion: the two generators produce
        different instance ensembles and their datasets are not interchangeable. Matched pairs have
        to come from ``to_price_spe`` / ``to_flow_spe``.
        """
        Gamma = random_psd_plus_skew(1, n_supply, kappa, eps, scale=scale).squeeze(0)
        Theta = random_psd_plus_skew(1, n_demand, kappa, eps, scale=scale).squeeze(0)
        p, q, delta, c = sample_market_data(n_supply, n_demand, delta)

        return cls(Gamma, Theta, p, q, delta, c)

    def flows(self, prices):
        """Shipments ``(..., m, n)`` implied by ``prices``, from route complementarity in closed form.

        The clamp is what carries the combinatorial content of the problem -- which market pairs
        trade at all -- and it also makes nonnegativity structural: the shipments are feasible for
        any prices, negative ones included, which is what licenses iterating on the prices
        unconstrained. Inverse of ``FlowSPE.prices`` only where the route conditions hold, so the
        round trip through the two is a fixed-point characterization of equilibrium rather than an
        identity.
        """
        supply_price, demand_price = prices.split([self.n_supply, self.n_demand], dim=-1)
        surplus = demand_price.unsqueeze(-2) - supply_price.unsqueeze(-1) - self.route_cost_min  # (..., m, n)
        return surplus.clamp(min=0.0) / self.route_cost_slope

    def operator(self, prices):
        supply_price, demand_price = prices.split([self.n_supply, self.n_demand], dim=-1)
        x = self.flows(prices)
        supply = (self.supply_side_effects @ (supply_price - self.supply_side_min).unsqueeze(-1)).squeeze(-1)
        demand = (self.demand_side_effects @ (self.demand_side_min - demand_price).unsqueeze(-1)).squeeze(-1)
        return torch.cat([supply - x.sum(dim=-1), x.sum(dim=-2) - demand], dim=-1)

    # def to_flow_spe(self):
    #     """The same instance in flow space; see ``FlowSPE.to_price_spe`` for why it stays monotone."""
    #     return FlowSPE(
    #         torch.linalg.inv(self.Gamma),
    #         torch.linalg.inv(self.Theta),
    #         self.p,
    #         self.q,
    #         self.delta,
    #         self.c,
    #     )

    @property
    def n_supply(self):
        return self.supply_side_min.size(-1)

    @property
    def n_demand(self):
        return self.demand_side_min.size(-1)

    def to_data(self):
        return Data(
            Gamma=self.supply_side_effects,
            Theta=self.demand_side_effects,
            p=self.supply_side_min,
            q=self.demand_side_min,
            delta=self.route_cost_slope,
            c=self.route_cost_min,
        )

    @classmethod
    def from_data(cls, data: Data):
        return cls(
            data["Gamma"],
            data["Theta"],
            data["p"],
            data["q"],
            data["delta"],
            data["c"],
        )
