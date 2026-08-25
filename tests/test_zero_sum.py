import pytest
import torch

from l2s_games.algorithms import Optimistic, SimpleProjection
from l2s_games.envs.zero_sum import (
    dist_to_normal_cone,
    profile_to_tensor,
    project_onto_simplex,
    RandomZeroSum,
    RandomZeroSumEquilibriumDataset,
    RandomZeroSumOperatorDataset,
)
from torch.utils.data import default_collate


@pytest.fixture
def random_zero_sum():
    return RandomZeroSum(n_actions=3, solve=True)


@pytest.fixture
def random_zero_sums():
    return [RandomZeroSum(n_actions=3, solve=True) for _ in range(5)]


def test_normal_cone_dist_zero(random_zero_sum: RandomZeroSum):
    instance = random_zero_sum.to_data()
    operator = RandomZeroSumOperatorDataset.eval_operator(instance, instance.eq)

    # eval_operator returns the descent-convention VI field; the ascent field is its negation
    dist = dist_to_normal_cone(-operator, instance.eq)
    assert torch.allclose(dist, torch.tensor(0.0), atol=1e-5)


def test_normal_cone_dist_nonzero(random_zero_sum: RandomZeroSum):
    profile = random_zero_sum.game.random_strategy_profile()

    instance = random_zero_sum.to_data()
    point = profile_to_tensor(profile)
    operator = RandomZeroSumOperatorDataset.eval_operator(instance, point)

    dist = dist_to_normal_cone(-operator, point)
    assert (dist > 0.0).all()


def test_optimistic_converges(random_zero_sum: RandomZeroSum):
    instance = random_zero_sum.to_data()

    # eval_operator returns the descent-convention VI field; the algorithm ascends its negation
    def operator(point):
        return -RandomZeroSumOperatorDataset.eval_operator(instance, point)

    algorithm = Optimistic(2e-3, operator, project_onto_simplex)
    strategies = torch.full_like(instance.eq, 1 / instance.eq.size(-1))  # uniform strategy
    for _ in range(2000):
        strategies = algorithm.step(strategies)

    dist = dist_to_normal_cone(operator(strategies), strategies)
    assert torch.allclose(dist, torch.tensor(0.0), atol=1e-3)


def test_projection_fails_on_rotational_instances(random_zero_sums):
    # the zero-sum field is purely rotational, which plain projection cannot handle: it cycles
    # around mixed equilibria instead of converging (it can still land on pure-strategy
    # equilibria, which sit on simplex vertices), while optimistic solves these instances
    failures = 0
    for random_zero_sum in random_zero_sums:
        instance = random_zero_sum.to_data()

        def operator(point):
            return -RandomZeroSumOperatorDataset.eval_operator(instance, point)

        algorithm = SimpleProjection(2e-3, operator, project_onto_simplex)
        strategies = torch.full_like(instance.eq, 1 / instance.eq.size(-1))  # uniform strategy
        for _ in range(2000):
            strategies = algorithm.step(strategies)

        dist = dist_to_normal_cone(operator(strategies), strategies)
        failures += (dist > 1e-2).any()

    assert failures >= 1


def test_data_round_trip(random_zero_sum: RandomZeroSum):
    restored = RandomZeroSum.from_data(random_zero_sum.to_data(dtype=torch.float64))

    for expected_payoffs, restored_payoffs in zip(
        random_zero_sum.game.to_arrays(dtype=float), restored.game.to_arrays(dtype=float)
    ):
        assert (expected_payoffs == restored_payoffs).all()

    assert torch.equal(profile_to_tensor(restored.eq), profile_to_tensor(random_zero_sum.eq))
    assert torch.allclose(torch.tensor(restored.eq.max_regret()), torch.tensor(0.0))


def test_dataset_operator(random_zero_sum: RandomZeroSum):
    profile = random_zero_sum.game.random_strategy_profile()
    strategy_values = [
        [profile.strategy_value(strategy) for strategy in player.strategies] for player in random_zero_sum.game.players
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float32)

    instance = random_zero_sum.to_data()
    point = profile_to_tensor(profile)
    operator = RandomZeroSumOperatorDataset.eval_operator(instance, point)

    assert operator.shape == point.shape
    assert torch.allclose(operator, expected)


def test_dataset_operator_batched_points(random_zero_sum: RandomZeroSum):
    profiles = [random_zero_sum.game.random_strategy_profile() for _ in range(5)]
    strategy_values = [
        [
            [profile.strategy_value(strategy) for strategy in player.strategies]
            for player in random_zero_sum.game.players
        ]
        for profile in profiles
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float32)

    instance = random_zero_sum.to_data()
    points = torch.stack([profile_to_tensor(profile) for profile in profiles])
    operators = RandomZeroSumOperatorDataset.eval_operator(instance, points)

    assert operators.shape == points.shape
    assert torch.allclose(operators, expected)


def test_dataset_operator_batched_games(random_zero_sums):
    profiles = [random_zero_sum.game.random_strategy_profile() for random_zero_sum in random_zero_sums]
    strategy_values = [
        [[profile.strategy_value(strategy) for strategy in player.strategies] for player in game.game.players]
        for game, profile in zip(random_zero_sums, profiles)
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float32)

    instances = [game.to_data() for game in random_zero_sums]
    batch = default_collate([instance.to_namedtuple() for instance in instances])
    points = torch.stack([profile_to_tensor(profile) for profile in profiles])
    operators = RandomZeroSumOperatorDataset.eval_operator(batch, points)

    assert operators.shape == points.shape
    assert torch.allclose(operators, expected)


def test_smoke_equilibrium_dataset(tmp_path):
    dataset = RandomZeroSumEquilibriumDataset(tmp_path, n_instances=10, n_actions=3)

    assert len(dataset) == 10
    assert dataset[0].A.shape == (3, 3)
    assert dataset[0].B.shape == (3, 3)
    assert dataset[0].eq.shape == (2, 3)


def test_smoke_operator_dataset(tmp_path):
    dataset = RandomZeroSumOperatorDataset(tmp_path, n_instances=10, n_points_per_instance=4, n_actions=3)

    assert len(dataset) == 10
    assert dataset[0].A.shape == (3, 3)
    assert dataset[0].B.shape == (3, 3)
    assert dataset[0].point.shape == (4, 2, 3)
    assert dataset[0].operator.shape == (4, 2, 3)
