import pytest
import torch
from l2s_games.envs.gamut import (
    RandomZeroSum,
    RandomZeroSumEquilibriumDataset,
    RandomZeroSumOperatorDataset,
    profile_to_tensor,
)


@pytest.fixture
def random_zero_sum():
    return RandomZeroSum(n_actions=3, solve=True)


def test_data_round_trip(random_zero_sum: RandomZeroSum):
    restored = RandomZeroSum.from_data(random_zero_sum.to_data())

    for expected_payoffs, restored_payoffs in zip(
        random_zero_sum.game.to_arrays(dtype=float), restored.game.to_arrays(dtype=float)
    ):
        assert (expected_payoffs == restored_payoffs).all()

    assert torch.equal(profile_to_tensor(restored.eq), profile_to_tensor(random_zero_sum.eq))
    assert restored.eq.max_regret() == pytest.approx(0.0)


def test_dataset_operator(random_zero_sum: RandomZeroSum):
    profile = random_zero_sum.game.random_strategy_profile()
    strategy_values = [
        [profile.strategy_value(strategy) for strategy in player.strategies] for player in random_zero_sum.game.players
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float64)

    instance = random_zero_sum.to_data()
    point = profile_to_tensor(profile)
    operator = RandomZeroSumOperatorDataset.eval_operator(instance, point)

    assert operator.shape == point.shape
    assert torch.allclose(operator, expected)


def test_dataset_operator_batched(random_zero_sum: RandomZeroSum):
    profiles = [random_zero_sum.game.random_strategy_profile() for _ in range(5)]
    strategy_values = [
        [
            [profile.strategy_value(strategy) for strategy in player.strategies]
            for player in random_zero_sum.game.players
        ]
        for profile in profiles
    ]
    expected = -torch.tensor(strategy_values, dtype=torch.float64)

    instance = random_zero_sum.to_data()
    points = torch.stack([profile_to_tensor(profile) for profile in profiles])
    operators = RandomZeroSumOperatorDataset.eval_operator(instance, points)

    assert operators.shape == points.shape
    assert torch.allclose(operators, expected)


def test_smoke_equilibrium_dataset(tmp_path):
    dataset = RandomZeroSumEquilibriumDataset(tmp_path, n_instances=10, n_actions=3)

    assert len(dataset) == 10
    assert dataset[0].A.shape == (3, 3)
    assert dataset[0].B.shape == (3, 3)
    assert dataset[0].eq.shape == (2, 3)


def test_smoke_operator_dataset(tmp_path):
    dataset = RandomZeroSumOperatorDataset(tmp_path, n_instances=5, n_points_per_instance=2, n_actions=3)

    assert len(dataset) == 10
    assert dataset[0].A.shape == (3, 3)
    assert dataset[0].B.shape == (3, 3)
    assert dataset[0].point.shape == (2, 3)
    assert dataset[0].operator.shape == (2, 3)
