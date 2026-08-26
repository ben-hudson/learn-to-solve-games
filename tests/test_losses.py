import pytest
import torch

from l2s_games.envs.zero_sum.game import RandomZeroSum, operator, profile_to_tensor, solve
from l2s_games.envs.zero_sum.losses import EGLoss, NashAprLoss, PotentialLoss
from l2s_games.envs.zero_sum.utils import simplex_projection


@pytest.fixture
def random_zero_sum():
    return RandomZeroSum.from_gambit(n_actions=3)


@pytest.fixture
def random_point(random_zero_sum: RandomZeroSum):
    return profile_to_tensor(random_zero_sum.to_gambit().random_strategy_profile())


def test_nash_apr_zero_at_equilibrium(random_zero_sum: RandomZeroSum):
    eq = solve(random_zero_sum)
    loss = NashAprLoss()(eq.unsqueeze(0), random_zero_sum.A.unsqueeze(0), random_zero_sum.B.unsqueeze(0))
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-4)


def test_nash_apr_positive_off_equilibrium(random_zero_sum: RandomZeroSum, random_point: torch.Tensor):
    loss = NashAprLoss()(random_point.unsqueeze(0), random_zero_sum.A.unsqueeze(0), random_zero_sum.B.unsqueeze(0))
    assert loss > 0.0


def test_potential_loss_gradient_descends_along_field(random_zero_sum: RandomZeroSum, random_point: torch.Tensor):
    # the sign contract: gradient *descent* on the loss moves the profile *along* the ascent
    # field, so the update matches the stationarity convention solve/dist_to_normal_cone use
    strategies = random_point.unsqueeze(0).requires_grad_()
    loss = PotentialLoss()(strategies, random_zero_sum.A.unsqueeze(0), random_zero_sum.B.unsqueeze(0))
    (gradient,) = torch.autograd.grad(loss, strategies)

    field = operator(random_zero_sum.A, random_zero_sum.B, strategies.detach())
    # the loss means over every player vector, so the per-vector gradient carries that factor
    n_player_vectors = strategies.shape[:-1].numel()
    assert torch.allclose(gradient, -field / n_player_vectors)


def test_eg_loss_gradient_descends_along_field_at_lookahead(
    random_zero_sum: RandomZeroSum, random_point: torch.Tensor
):
    # same sign contract as PotentialLoss, but the field is evaluated at the projected
    # lookahead point, so one descent step reproduces one extragradient update
    step_size = 2e-3
    strategies = random_point.unsqueeze(0).requires_grad_()
    loss = EGLoss(step_size)(strategies, random_zero_sum.A.unsqueeze(0), random_zero_sum.B.unsqueeze(0))
    (gradient,) = torch.autograd.grad(loss, strategies)

    lookahead = simplex_projection(
        strategies.detach() + step_size * operator(random_zero_sum.A, random_zero_sum.B, strategies.detach())
    )
    field = operator(random_zero_sum.A, random_zero_sum.B, lookahead)
    n_player_vectors = strategies.shape[:-1].numel()
    assert torch.allclose(gradient, -field / n_player_vectors)
