import torch

from l2s_games.envs.zero_sum.game import operator

from .utils import simplex_projection


class NormLoss(torch.nn.Module):
    """``||prediction - target||`` over each sample's flattened field, averaged over samples.

    Zero-residual samples are masked out of the mean: the norm's gradient at exactly zero is NaN.
    """

    def forward(self, prediction, target):
        # flatten start_dim=1 because the operator is a vector, so we have n_players*n_actions
        residual_norm = (prediction - target).flatten(start_dim=1).norm(dim=-1)
        return residual_norm[residual_norm > 0].mean()


class NormHuberLoss(torch.nn.Module):
    """Huber on each sample's residual norm, averaged over samples: MSE-like inside the knee,
    ``NormLoss``-like beyond it.

    The per-sample gradient is the residual itself for ``||r|| <= delta`` (so fitted samples
    self-anneal instead of bouncing at a noise floor) and ``delta * r/||r||`` beyond (so
    badly-fit samples all push with the same bounded magnitude, like ``NormLoss``'s unit
    vector). Huber on the scalar norm against a zero target realizes exactly this piecewise
    loss, so the composition reuses ``huber_loss`` rather than restating it. Zero-residual
    samples are masked out of the mean: the norm's gradient at exactly zero is NaN.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, prediction, target):
        residual_norm = (prediction - target).flatten(start_dim=1).norm(dim=-1)
        residual_norm = residual_norm[residual_norm > 0]
        return torch.nn.functional.huber_loss(residual_norm, torch.zeros_like(residual_norm), delta=self.delta)


class NashAprLoss(torch.nn.Module):
    """Nash approximation loss for bimatrix games (Duan et al. 2023, Eq. 1), averaged over the batch.

    ``NashApr(σ, u) = max_i max_{a_i} [u_i(a_i, σ_{-i}) - u_i(σ)]`` is the largest utility any player
    gains by deviating to a pure best response, so it is zero exactly at a Nash equilibrium. It is
    computed from the strategies and payoffs alone -- no equilibrium labels -- which sidesteps the
    equilibrium-selection problem that breaks supervised regression onto a solver's solution.
    """

    def forward(self, strategies, A, B):
        x, y = strategies.unbind(dim=-2)
        # each player's expected payoff per pure action against the opponent's mixed strategy:
        # (Ay)_a = u_1(a, y) and (x^T B)_a = u_2(x, a)
        pure_payoffs_1 = torch.einsum("bij,bj->bi", A, y)
        pure_payoffs_2 = torch.einsum("bi,bij->bj", x, B)
        # deviation gain = best pure payoff - current expected payoff u_i(x, y)
        gain_1 = pure_payoffs_1.max(dim=-1).values - torch.einsum("bi,bi->b", x, pure_payoffs_1)
        gain_2 = pure_payoffs_2.max(dim=-1).values - torch.einsum("bj,bj->b", y, pure_payoffs_2)
        return torch.maximum(gain_1, gain_2).mean()


class PotentialLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, strategies: torch.Tensor, A, B):
        operator = PotentialGradient.apply(strategies, A, B)
        return operator.sum(dim=-1).mean()


class PotentialGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, strategies, A, B):
        op = operator(A, B, strategies)
        ctx.save_for_backward(op)
        return op.square()

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (op,) = ctx.saved_tensors
        # one gradient per forward input, positionally: (strategies, A, B). Returning ``-op``
        # for the profile makes the optimizer's descent step move it *along* the field --
        # ``z <- z + lr * op(z)``, the ascent convention every dynamic in ``algorithms`` follows.
        # The payoffs are data, so they get no gradient.
        return -op * grad_loss, None, None


class EGLoss(torch.nn.Module):
    """Extragradient surrogate: reports the squared operator norm at the *lookahead* point, and
    hands back a gradient that makes one optimizer step reproduce one extragradient update.

    The backward substitutes the field itself for the true Jacobian-vector product, which turns
    ``z <- z - lr * dL/dz`` into Korpelevich's ``z <- z + lr * op(z_half)``
    (``algorithms.ExtraGradient``), pushed back through the network. The forward value is only
    what that surrogate is reported as: like ``PotentialLoss`` it is the squared field norm, which
    a zero-sum game does *not* drive to zero (at an interior equilibrium the field is a nonzero
    constant vector, normal to the simplex), so it is not monotone over training and
    ``val/residual`` is the metric to read.

    That equivalence only holds while the lookahead ``h`` matches the step the optimizer is about
    to take, so ``step_size`` accepts a zero-argument callable as well as a float: pass
    ``lambda: model.current_lr`` to track the live learning rate through warmup and annealing.

    ``project`` controls whether the lookahead point is projected back onto the simplex, as the
    textbook constrained form does; turning it off queries the field off the feasible set.
    """

    def __init__(self, step_size, project: bool = True):
        super().__init__()
        self.step_size = step_size
        self.project = project

    def forward(self, strategies: torch.Tensor, A, B):
        step_size = self.step_size() if callable(self.step_size) else self.step_size
        operator = EGGradient.apply(strategies, A, B, step_size, self.project)
        return operator.sum(dim=-1).mean()


class EGGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, strategies, A, B, step_size, project):
        # Korpelevich's iteration projects the lookahead too, so the field is never queried off the
        # simplex -- the contract ``algorithms.ExtraGradient`` keeps. ``project=False`` drops it,
        # leaving the half-step wherever plain extrapolation puts it.
        strategies_half = strategies + step_size * operator(A, B, strategies)
        if project:
            strategies_half = simplex_projection(strategies_half)
        op = operator(A, B, strategies_half)
        ctx.save_for_backward(op)
        return op.square()

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (op,) = ctx.saved_tensors
        # one gradient per forward input, positionally: (strategies, A, B, step_size, project). See
        # ``PotentialGradient.backward`` for the sign; everything else here is a constant.
        return -op * grad_loss, None, None, None, None
