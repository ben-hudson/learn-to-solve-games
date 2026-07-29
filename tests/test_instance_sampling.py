"""Tests for the fixed-instance training source (``FixedInstanceOperatorStream``).

Covers the three properties the source must guarantee: the instance set is **fixed** (every example
comes from the passed instances, and all of them are visited), the domain points are **not** (a second
visit to the same instance carries a different point), and the diagnostic ``instance_index`` tag
survives the conditioning seam -- family ``transform`` then ``collate_fn`` -- onto the batch without
entering ``feats``.

The stream tests use the flat ``rps`` family: its operator is a cheap matrix product, so the whole file
runs without a route-choice solve. The graph representation is covered separately by pushing a tagged
traffic item through its transform + collate (no operator call).
"""

import itertools
import pathlib

import pytest
import torch

from l2s_games.data import INSTANCE_INDEX, build_dataset, collate_examples
from l2s_games.envs import make_game
from l2s_games.envs.pume_traffic import load_sioux_falls_base_graph
from l2s_games.instance_sampling import FixedInstanceOperatorStream

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sioux_falls"
N_INSTANCES = 4
POINTS_PER_INSTANCE = 3


@pytest.fixture
def family():
    torch.manual_seed(0)
    return make_game("rps")


@pytest.fixture
def normalizer(family):
    """A fitted normalizer, built the ordinary way (fit on a train split of the same family)."""
    _datasets, normalizer = build_dataset(family, n_train=4, n_val=1, n_test=1, points_per_instance=2)
    return normalizer


@pytest.fixture
def stream(family, normalizer):
    instances = [family.sample_params() for _ in range(N_INSTANCES)]
    return FixedInstanceOperatorStream(lambda: family, normalizer, instances, POINTS_PER_INSTANCE)


def take(stream, n):
    return list(itertools.islice(iter(stream), n))


def test_visits_every_instance_exactly_once_per_pass(stream):
    """One pass yields ``points_per_instance`` examples for each instance, and nothing else."""
    examples = take(stream, N_INSTANCES * POINTS_PER_INSTANCE)
    indices = [int(item[INSTANCE_INDEX]) for item, _ in examples]
    counts = torch.bincount(torch.tensor(indices), minlength=N_INSTANCES)
    assert torch.equal(counts, torch.full((N_INSTANCES,), POINTS_PER_INSTANCE))


def test_points_are_resampled_on_a_later_visit(stream):
    """The instance set is fixed but the domain points are not: two passes give distinct points."""
    examples = take(stream, 2 * N_INSTANCES * POINTS_PER_INSTANCE)
    points_by_index = {}
    for item, _ in examples:
        points_by_index.setdefault(int(item[INSTANCE_INDEX]), []).append(item["point"])
    for index, points in points_by_index.items():
        assert len(points) == 2 * POINTS_PER_INSTANCE, index
        stacked = torch.stack(points)
        assert stacked.unique(dim=0).shape[0] == stacked.shape[0], f"repeated point for instance {index}"


def test_instances_come_from_the_fixed_set(stream, family):
    """Every example's conditioning params match the instance its index points at."""
    for item, _ in take(stream, 2 * N_INSTANCES * POINTS_PER_INSTANCE):
        assert torch.equal(item["params"], stream.instances[int(item[INSTANCE_INDEX])])


def test_index_survives_collation_without_entering_feats(stream, family, normalizer):
    """The tag reaches the batch as ``[B]`` and leaves the model input width untouched."""
    examples = take(stream, N_INSTANCES)
    inputs, _targets = collate_examples(family)(examples)
    indices = torch.stack([item[INSTANCE_INDEX] for item, _ in examples])
    assert torch.equal(inputs[INSTANCE_INDEX], indices)
    assert inputs[INSTANCE_INDEX].shape == (len(examples),)
    # feats is still [point | params] -- the tag rides alongside, never inside the model input.
    assert inputs["feats"].shape == (len(examples), family.domain_dim + family.n_params)


def test_index_survives_the_graph_conditioning_seam():
    """The graph path carries the tag too: traffic's transform + dense collate keep it (no solve)."""
    if not _DATA_ROOT.exists():
        pytest.skip(f"Sioux Falls TNTP data not found at {_DATA_ROOT}")
    torch.manual_seed(0)
    traffic = make_game("pume_traffic", base_graph=load_sioux_falls_base_graph(str(_DATA_ROOT)))
    graph = traffic.sample_params()
    items = []
    for index in range(2):
        item = traffic.model_input(graph, traffic.sample_domain(graph, 1)[0])
        item[INSTANCE_INDEX] = torch.tensor(index)
        items.append(traffic.transform(item))
    batch = traffic.collate_fn(items)
    assert torch.equal(batch[INSTANCE_INDEX], torch.tensor([0, 1]))
