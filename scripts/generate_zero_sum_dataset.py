import argparse

from pathlib import Path
from l2s_games.envs.zero_sum import RandomZeroSumEquilibriumDataset


def get_config():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--n_instances", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = get_config()
    dataset = RandomZeroSumEquilibriumDataset(
        config.dataset, n_instances=config.n_instances, n_actions=config.n_actions, quiet=config.quiet
    )
