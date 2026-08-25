import pytest
import torch

from l2s_games.algorithms import Optimistic, SimpleProjection
from l2s_games.envs.spe.streams import EquilibriumStream, OperatorStream
from l2s_games.envs.traffic.datasets import GraphToTensorDict
from l2s_games.envs.zero_sum import (
    dist_to_normal_cone,
    profile_to_tensor,
    simplex_projection,
    RandomZeroSum,
    RandomZeroSumEquilibriumDataset,
    RandomZeroSumOperatorDataset,
)
from torch.utils.data import DataLoader, default_collate
from functools import partial
from l2s_games.envs.zero_sum.game import operator, solve, profile_to_tensor, tensor_to_profile


@pytest.fixture
def random_zero_sum():
    return RandomZeroSum.from_gambit(n_actions=3)


@pytest.fixture
def random_zero_sums():
    return [RandomZeroSum.from_gambit(n_actions=3) for _ in range(5)]


def test_normal_cone_dist_zero(random_zero_sum: RandomZeroSum):
    eq = solve(random_zero_sum)
    # eval_operator returns the descent-convention VI field; the ascent field is its negation
    dist = dist_to_normal_cone(-random_zero_sum.operator(eq), eq)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-5)


def test_normal_cone_dist_nonzero(random_zero_sum: RandomZeroSum):
    profile = random_zero_sum.to_gambit().random_strategy_profile()
    point = profile_to_tensor(profile)
    dist = dist_to_normal_cone(-random_zero_sum.operator(point), point)
    assert (dist > 0.0).all()


def test_optimistic_converges(random_zero_sum: RandomZeroSum):
    algorithm = Optimistic(2e-3, random_zero_sum.operator, simplex_projection)
    strategy = torch.full((random_zero_sum.n_players, random_zero_sum.n_actions), 1 / random_zero_sum.n_actions)
    for _ in range(2000):
        strategy = algorithm.step(strategy)

    dist = dist_to_normal_cone(random_zero_sum.operator(strategy), strategy)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-3)


def test_projection_fails_on_rotational_instances(random_zero_sums):
    # the zero-sum field is purely rotational, which plain projection cannot handle: it cycles
    # around mixed equilibria instead of converging (it can still land on pure-strategy
    # equilibria, which sit on simplex vertices), while optimistic solves these instances
    failures = 0
    for random_zero_sum in random_zero_sums:
        algorithm = SimpleProjection(2e-3, random_zero_sum.operator, simplex_projection)
        strategy = torch.full((random_zero_sum.n_players, random_zero_sum.n_actions), 1 / random_zero_sum.n_actions)
        for _ in range(2000):
            strategy = algorithm.step(strategy)

        dist = dist_to_normal_cone(random_zero_sum.operator(strategy), strategy)
        failures += (dist > 1e-2).any()

    assert failures >= 1


def test_data_round_trip(random_zero_sum: RandomZeroSum):
    eq = solve(random_zero_sum, dtype=torch.float64)
    restored = RandomZeroSum.from_data(random_zero_sum.to_data(dtype=torch.float64))
    restored_eq = solve(restored, dtype=torch.float64)

    assert (random_zero_sum.A == restored.A).all()
    assert (random_zero_sum.B == restored.B).all()
    # pygambit builds slightly different internal payoffs from float32 vs float64 arrays,
    # so the two LP solutions agree only to ~1e-8, not exactly
    assert torch.allclose(eq, restored_eq)


def test_operator(random_zero_sum: RandomZeroSum):
    game = random_zero_sum.to_gambit()
    profile = game.random_strategy_profile()
    strategy_values = [[profile.strategy_value(strategy) for strategy in player.strategies] for player in game.players]
    expected = -torch.tensor(strategy_values, dtype=torch.float32)

    point = profile_to_tensor(profile)
    operator_values = operator(random_zero_sum.A, random_zero_sum.B, point)

    assert operator_values.shape == point.shape
    assert torch.allclose(operator_values, expected)


def test_operator_batched_points(random_zero_sum: RandomZeroSum):
    game = random_zero_sum.to_gambit()
    profiles = [game.random_strategy_profile() for _ in range(5)]
    strategy_values = [
        [[profile.strategy_value(strategy) for strategy in player.strategies] for player in game.players]
        for profile in profiles
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float32)

    points = torch.stack([profile_to_tensor(profile) for profile in profiles])
    operator_values = operator(random_zero_sum.A, random_zero_sum.B, points)

    assert operator_values.shape == points.shape
    assert torch.allclose(operator_values, expected)


def test_operator_batched_games(random_zero_sums):
    games = [random_zero_sum.to_gambit() for random_zero_sum in random_zero_sums]
    profiles = [game.random_strategy_profile() for game in games]
    strategy_values = [
        [[profile.strategy_value(strategy) for strategy in player.strategies] for player in game.players]
        for game, profile in zip(games, profiles)
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float32)

    batch = torch.stack(random_zero_sums)
    points = torch.stack([profile_to_tensor(profile) for profile in profiles])
    operator_values = operator(batch.A, batch.B, points)

    assert operator_values.shape == points.shape
    assert torch.allclose(operator_values, expected)


def test_smoke_equilibrium_stream():
    batch_size = 5
    n_actions = 3
    n_players = 2

    sample = partial(RandomZeroSum.from_gambit, n_actions=n_actions)
    dataset = EquilibriumStream(
        sample, n_instances=2 * batch_size, solve=True, quiet=True, transform=GraphToTensorDict()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=torch.stack)
    for batch in dataloader:
        assert batch["A"].shape == (batch_size, n_actions, n_actions)
        assert batch["B"].shape == (batch_size, n_actions, n_actions)
        assert batch["eq"].shape == (batch_size, n_players, n_actions)


def test_smoke_operator_stream():
    batch_size = 5
    n_points = 4
    n_actions = 3
    n_players = 2

    sample = partial(RandomZeroSum.from_gambit, n_actions=n_actions)
    sample_domain = lambda instance, n: torch.distributions.Dirichlet(torch.ones(instance.n_actions)).sample(
        (n, instance.n_players)
    )
    dataset = OperatorStream(
        sample,
        sample_domain,
        n_instances=2 * batch_size,
        n_points_per_instance=n_points,
        solve=False,
        quiet=True,
        transform=GraphToTensorDict(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=torch.stack)
    for batch in dataloader:
        assert batch["A"].shape == (batch_size, n_actions, n_actions)
        assert batch["B"].shape == (batch_size, n_actions, n_actions)
        assert batch["operator"].shape == (batch_size, n_points, n_players, n_actions)
