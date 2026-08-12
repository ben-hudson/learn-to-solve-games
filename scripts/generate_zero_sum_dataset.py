from l2s_games.datasets.zero_sum import RandomZeroSumEquilibriumDataset

if __name__ == "__main__":
    dataset = RandomZeroSumEquilibriumDataset("datasets/random_zero_sum_512", n_instances=512, n_actions=3, quiet=False)
