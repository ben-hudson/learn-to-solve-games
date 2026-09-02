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


def sample_market_data(n_supply, n_demand, delta):
    """The data both formulations share: intercepts ``(p, q)`` and route costs ``(delta, c)``.

    Sampled identically for either formulation, so the only thing that distinguishes a ``FlowSPE``
    instance from a ``PriceSPE`` one is whether its matrices act on quantities or on prices.
    """
    p = torch.empty(n_supply).uniform_(1.0, 3.0)
    q = torch.empty(n_demand).uniform_(4.0, 8.0)
    c = torch.empty(n_supply, n_demand).uniform_(0.5, 3.0)
    return p, q, torch.full((n_supply, n_demand), delta), c


class FlowSPE:
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
        self.eq = None

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

    def to_price_spe(self):
        """The same instance in price space. Inverting a matrix with PD symmetric part gives another
        one (``x^T M^-1 x = y^T sym(M) y > 0`` for ``y = M^-1 x``), so the result is always a
        strongly monotone instance.
        """
        return PriceSPE(
            torch.linalg.inv(self.P),
            torch.linalg.inv(self.Q),
            self.p,
            self.q,
            self.delta,
            self.c,
        )

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
            eq=self.eq,
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


class PriceSPE:
    def __init__(
        self,
        supply_quantity_effects: torch.Tensor,
        demand_quantity_effects: torch.Tensor,
        supply_side_min: torch.Tensor,
        demand_side_min: torch.Tensor,
        route_cost_slope: torch.Tensor,
        route_cost_min: torch.Tensor,
    ):
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
        self.Gamma = supply_quantity_effects
        self.Theta = demand_quantity_effects
        self.p = supply_side_min
        self.q = demand_side_min
        self.delta = route_cost_slope
        self.c = route_cost_min
        self.eq = None

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
        surplus = demand_price.unsqueeze(-2) - supply_price.unsqueeze(-1) - self.c  # (..., m, n)
        return surplus.clamp(min=0.0) / self.delta

    def operator(self, prices):
        supply_price, demand_price = prices.split([self.n_supply, self.n_demand], dim=-1)
        x = self.flows(prices)
        supply = (self.Gamma @ (supply_price - self.p).unsqueeze(-1)).squeeze(-1)  # s(u) (..., m)
        demand = (self.Theta @ (self.q - demand_price).unsqueeze(-1)).squeeze(-1)  # d(v) (..., n)
        return torch.cat([supply - x.sum(dim=-1), x.sum(dim=-2) - demand], dim=-1)

    def to_flow_spe(self):
        """The same instance in flow space; see ``FlowSPE.to_price_spe`` for why it stays monotone."""
        return FlowSPE(
            torch.linalg.inv(self.Gamma),
            torch.linalg.inv(self.Theta),
            self.p,
            self.q,
            self.delta,
            self.c,
        )

    @property
    def n_supply(self):
        return self.p.size(-1)

    @property
    def n_demand(self):
        return self.q.size(-1)

    def to_dict(self):
        return TensorDict(
            Gamma=self.Gamma,
            Theta=self.Theta,
            p=self.p,
            q=self.q,
            delta=self.delta,
            c=self.c,
            eq=self.eq,
        )

    @classmethod
    def from_dict(cls, data: TensorDict):
        return cls(
            data["Gamma"],
            data["Theta"],
            data["p"],
            data["q"],
            data["delta"],
            data["c"],
        )
