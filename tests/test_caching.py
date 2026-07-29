"""Tests for the bounded-budget training source (``caching.CachedOperatorStream``).

The source exists to make the ground-truth operator spend a *config* number rather than something that
grows with the epoch count, so what is worth pinning is where the spend stops and what is reused:

1. The **budget is capped** -- draining far past the cache leaves the operator-call count where the fill
   left it. This is the whole point of the class and the one property no other source has.
2. The **reuse is exact** -- a later pass re-emits the same ``(point, target)`` pairs, still correctly
   paired. Contrast ``test_instance_sampling.test_points_are_resampled_on_a_later_visit``: the fixed
   source deliberately redraws its points, which is precisely what makes its budget unbounded.
3. The cache holds **groups, not examples** -- ``model_input`` clones the instance per point, so a buffer
   of examples would cost ``points_per_instance`` times the memory (13 GB rather than 1.3 GB for a
   million cached traffic points). A refactor could reintroduce that layout without any test noticing.
4. The **per-worker** semantics: each dataloader replica fills its own cache, which is where the
   ``n_workers`` factor in the budget formula comes from.

Uses the flat ``rps`` family: its operator is a cheap matrix product, so the file runs without a
route-choice solve. A counting wrapper around ``operator_and_metric`` stands in for the ``SharedCounter``
the traffic family carries, since counting is what every budget assertion here is about.
"""

import itertools

import pytest
import torch
from torch.utils.data import DataLoader

from l2s_games.caching import CachedOperatorStream
from l2s_games.data import INSTANCE_INDEX, build_dataset
from l2s_games.envs import make_game

from helpers import CountingFamily, take

N_INSTANCES = 4
POINTS_PER_INSTANCE = 3
CACHED_EXAMPLES = N_INSTANCES * POINTS_PER_INSTANCE


def rps_family():
    """A picklable family factory -- a local lambda cannot cross the worker (spawn) boundary."""
    return make_game("rps")


@pytest.fixture
def family():
    torch.manual_seed(0)
    return CountingFamily(make_game("rps"))


@pytest.fixture
def normalizer(family):
    """A fitted normalizer, built the ordinary way (fit on a train split of the same family)."""
    _datasets, normalizer = build_dataset(family, n_train=4, n_val=1, n_test=1, points_per_instance=2)
    family.n_evaluations = 0  # the fit is not part of any budget under test
    return normalizer


@pytest.fixture
def stream(family, normalizer):
    """The fresh-instance variant: n_instances drawn by the stream, never refreshed."""
    return CachedOperatorStream(
        lambda: family, normalizer, POINTS_PER_INSTANCE, n_instances=N_INSTANCES, refresh_every=10**6
    )


def pairs(examples):
    """The ``(point, target)`` content of each example, as hashable tuples."""
    return [(tuple(item["point"].tolist()), tuple(target.tolist())) for item, target in examples]


def test_budget_stops_at_the_cache_size(stream, family):
    """Draining well past the cache costs exactly one operator evaluation per cached point."""
    take(stream, 10 * CACHED_EXAMPLES)
    assert family.n_evaluations == CACHED_EXAMPLES


def test_later_passes_reuse_the_same_points(stream):
    """Pass 2 re-emits pass 1's ``(point, target)`` pairs -- reuse, not resampling."""
    examples = take(stream, 2 * CACHED_EXAMPLES)
    first, second = pairs(examples[:CACHED_EXAMPLES]), pairs(examples[CACHED_EXAMPLES:])
    assert sorted(second) == sorted(first)


def test_targets_stay_paired_with_their_points(stream, family, normalizer):
    """Every re-emitted example's target is still the operator at its own point (in real units)."""
    for item, target in take(stream, 2 * CACHED_EXAMPLES):
        expected, _metric = family.operator_and_metric(item["params"], item["point"].unsqueeze(0))
        assert torch.allclose(normalizer.inverse_target(target), expected[0], atol=1e-6)


def test_instance_tag_groups_the_points(stream):
    """Each cached instance contributes ``points_per_instance`` examples under its own index."""
    indices = [int(item[INSTANCE_INDEX]) for item, _ in take(stream, CACHED_EXAMPLES)]
    counts = torch.bincount(torch.tensor(indices), minlength=N_INSTANCES)
    assert torch.equal(counts, torch.full((N_INSTANCES,), POINTS_PER_INSTANCE))


def test_cache_holds_groups_not_examples(stream):
    """The retained cache is one group per instance -- not one cloned item per point."""
    take(stream, CACHED_EXAMPLES)
    assert len(stream._groups) == N_INSTANCES
    assert all(len(group.points) == POINTS_PER_INSTANCE for group in stream._groups)


def test_refresh_resolves_the_window(family, normalizer):
    """``refresh_every=1`` re-solves per epoch (a fresh window), so the budget steps up once per pass."""
    stream = CachedOperatorStream(
        lambda: family, normalizer, POINTS_PER_INSTANCE, n_instances=N_INSTANCES, refresh_every=1
    )
    take(stream, CACHED_EXAMPLES)
    take(stream, CACHED_EXAMPLES)  # a second __iter__ == the next epoch
    assert family.n_evaluations == 2 * CACHED_EXAMPLES


def test_pinned_instances_are_the_only_ones_used(family, normalizer):
    """The instances-list variant never samples: every example comes from the passed set."""
    instances = [family.sample_params() for _ in range(N_INSTANCES)]
    stream = CachedOperatorStream(
        lambda: family, normalizer, POINTS_PER_INSTANCE, instances=instances, refresh_every=10**6
    )
    assert stream.n_instances == N_INSTANCES
    for item, _ in take(stream, 2 * CACHED_EXAMPLES):
        assert torch.equal(item["params"], instances[int(item[INSTANCE_INDEX])])


def test_every_worker_fills_its_own_cache(family, normalizer):
    """Two workers -> two caches, hence the n_workers factor in the budget formula.

    Pins the semantics inherited from the expert stream (PyTorch does not shard iterable datasets); the
    assertion to flip if the sources are ever made to shard a single budget across workers.
    """
    stream = CachedOperatorStream(
        rps_family, normalizer, POINTS_PER_INSTANCE, n_instances=N_INSTANCES, refresh_every=10**6
    )
    loader = DataLoader(stream, batch_size=1, num_workers=2, collate_fn=list)
    examples = [batch[0] for batch in itertools.islice(iter(loader), 4 * CACHED_EXAMPLES)]
    assert len(set(pairs(examples))) == 2 * CACHED_EXAMPLES
