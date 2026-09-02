import torch


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
