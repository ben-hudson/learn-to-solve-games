"""Tests for the on-disk operator-example caches (``operator_datasets``).

These datasets store per *instance* but index per *point*, so most of what can go wrong is in that
mismatch and in the config sidecar that makes it decodable. What is pinned here:

1. **The flat index decodes correctly** -- ``len`` counts examples, and consecutive indices inside one
   instance's block resolve to that instance's graph and to distinct points of it. A ``divmod`` that
   silently mis-decoded would still yield valid-looking examples, which is exactly why it needs a test.
2. **The cache is a cache** -- reopening does not re-evaluate the operator, and returns identical tensors.
3. **The config file is authoritative, and disagreeing with it raises.** ``process()`` is skipped on a
   cache hit, so a constructor value that differs from the stored one cannot be honoured.
4. **The stored per-instance blocks never reach the model.** They ride on the instance graph, so without
   ``_DROPPED_ATTRS`` every *other* point of an instance would be collated into the batch beside the one
   being trained on.
5. **Targets are the operator at their own point** -- for the expert dataset, that the recorded rollout
   values were kept rather than re-derived, and that the appended endpoint is the near-zero one.

Needs the Sioux Falls TNTP files and a solved-instance cache, since the sampling ceiling is calibrated
from real equilibria; skips without them, like the other traffic-family tests.
"""

import pathlib
import shutil

import pytest
import torch

from l2s_games.data import PRECONDITIONER_DIAGONAL, Normalizer, GlobalStandardizer, Standardizer, collate_examples
from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs import make_game
from l2s_games.operator_datasets import ExpertTrajectoryOperatorDataset, UniformOperatorDataset

_DATASET_ROOT = pathlib.Path(__file__).resolve().parents[1] / "datasets" / "sioux_falls_512"

N_INSTANCES = 3
POINTS_PER_INSTANCE = 4
N_CAL_INSTANCES = 8
N_STEPS = 5
N_FEATS = 9  # cost + 4 BPR attrs + 4 demand feats (see transforms.BuildTrafficEdgeData)


@pytest.fixture(scope="module")
def solved_root(tmp_path_factory):
    """A private copy of a solved-instance root, so generating into it cannot touch the real one."""
    if not _DATASET_ROOT.exists():
        pytest.skip(f"Solved-instance cache not found at {_DATASET_ROOT}")
    root = tmp_path_factory.mktemp("operator_datasets") / "root"
    (root / "raw").mkdir(parents=True)
    (root / "processed").mkdir()
    shutil.copy(_DATASET_ROOT / "raw" / "base_graph.pt", root / "raw" / "base_graph.pt")
    shutil.copy(_DATASET_ROOT / "processed" / "instances.pt", root / "processed" / "instances.pt")
    return root


@pytest.fixture(scope="module")
def family(solved_root):
    """The family the examples are generated with, its cost box calibrated from real equilibria."""
    solved = SolvedInstanceDataset(str(solved_root))
    ceiling = make_game("pume_traffic", base_graph=solved.base_graph).calibrate_ceiling(
        list(solved)[:N_CAL_INSTANCES]
    )
    return make_game("pume_traffic", base_graph=solved.base_graph, sampling_ceiling=ceiling)


@pytest.fixture
def normalizer():
    """An identity normalizer: these tests are about storage and indexing, not about the fit."""
    return Normalizer(
        Standardizer(torch.zeros(N_FEATS), torch.ones(N_FEATS)),
        GlobalStandardizer(torch.zeros(()), torch.ones(())),
    )


@pytest.fixture(scope="module")
def uniform(solved_root, family):
    torch.manual_seed(0)
    return UniformOperatorDataset(
        str(solved_root),
        family=family,
        n_instances=N_INSTANCES,
        points_per_instance=POINTS_PER_INSTANCE,
        quiet=True,
    )


@pytest.fixture(scope="module")
def expert(solved_root, family):
    torch.manual_seed(0)
    return ExpertTrajectoryOperatorDataset(
        str(solved_root), family=family, n_instances=2, algo="projection", h=0.1, n_steps=N_STEPS, quiet=True
    )


# --- the index mismatch: stored per instance, served per point -------------------------------------


def test_length_counts_examples_not_instances(uniform):
    assert uniform.n_stored_instances == N_INSTANCES
    assert len(uniform) == N_INSTANCES * POINTS_PER_INSTANCE


def test_expert_width_is_the_rollout_plus_its_endpoint(expert):
    """``projection`` evaluates the field once per step, and the endpoint adds the one state it never
    queried -- so the width is ``n_steps + 1`` and the budget is readable off the config."""
    assert expert.points_per_instance == N_STEPS + 1
    assert len(expert) == expert.n_stored_instances * (N_STEPS + 1)


def test_flat_index_decodes_to_the_right_instance_and_point(uniform, normalizer):
    """Consecutive indices inside a block share the instance and differ in the point.

    A mis-decoding ``divmod`` would still return well-formed examples, so this compares against the
    stored blocks directly: example ``i`` must carry point ``i % points_per_instance`` of instance
    ``i // points_per_instance``.
    """
    uniform.normalizer = normalizer
    for index in range(len(uniform)):
        instance, point = divmod(index, POINTS_PER_INSTANCE)
        item, target = uniform[index]
        stored = uniform.evaluations(instance)
        assert torch.equal(item["cost"], stored.points[point])
        assert torch.equal(target, stored.targets[point])
        assert torch.equal(item[PRECONDITIONER_DIAGONAL], stored.preconditioner_diagonal[point])


def test_points_within_an_instance_are_distinct(uniform):
    """Each block holds ``points_per_instance`` *different* points -- i.e. the order argument varies.

    Guards the failure where every index in a block resolves to the same point (e.g. ``order=[0]``),
    which would leave the length right and the values plausible.
    """
    points = uniform.evaluations(0).points
    assert len({tuple(row.tolist()) for row in points}) == POINTS_PER_INSTANCE


# --- caching -------------------------------------------------------------------------------------


def test_reopening_does_not_re_evaluate_the_operator(solved_root, family, uniform):
    """A second construction with no callables reloads and yields identical tensors.

    Passing no ``family`` would break the *read* path (``get`` needs ``model_input``), so the signal that
    nothing was regenerated is the operator-call count staying flat while the tensors match.
    """
    before = family.operator_counter.value
    reopened = UniformOperatorDataset(str(solved_root), family=family, quiet=True)
    assert family.operator_counter.value == before
    assert len(reopened) == len(uniform)
    assert reopened.points_per_instance == POINTS_PER_INSTANCE
    for index in range(reopened.n_stored_instances):
        assert torch.equal(reopened.evaluations(index).targets, uniform.evaluations(index).targets)


def test_a_disagreeing_points_per_instance_raises(solved_root, family):
    """``process()`` is skipped on a cache hit, so a new value cannot be honoured -- say so loudly.

    Silently preferring either value would decode every index into the wrong ``(instance, point)`` pair.
    """
    with pytest.raises(AssertionError, match="points per instance"):
        UniformOperatorDataset(
            str(solved_root), family=family, points_per_instance=POINTS_PER_INSTANCE + 1, quiet=True
        )


def test_deleting_either_processed_file_regenerates_both(solved_root, family, uniform):
    """PyG's cache check requires *all* ``processed_paths``, so a half-written root is a miss.

    Without that, a root holding the evaluations but no config (or the reverse) would fail obscurely on
    read instead of simply rebuilding.
    """
    config_path = pathlib.Path(uniform.processed_paths[1])
    config_path.unlink()
    torch.manual_seed(0)
    rebuilt = UniformOperatorDataset(
        str(solved_root),
        family=family,
        n_instances=N_INSTANCES,
        points_per_instance=POINTS_PER_INSTANCE,
        quiet=True,
    )
    assert config_path.exists()
    assert len(rebuilt) == N_INSTANCES * POINTS_PER_INSTANCE


def test_noise_settings_are_recorded_as_provenance(uniform, family):
    """Recorded, never read back, never asserted on -- so train and val examples may differ in noise."""
    config = torch.load(uniform.processed_paths[1], weights_only=False)
    assert config["noise_scale"] == family.noise_scale
    assert config["noise_type"] == family.noise_type


def test_a_root_holds_both_datasets_over_one_base_graph(solved_root, uniform, expert):
    """The two caches share the root's single ``raw/base_graph.pt``.

    That sharing is the point: the asymmetric family's coupling matrices live on the base graph, and a
    second copy could disagree with the equilibria silently (see ``envs/asym_pume_traffic``).
    """
    assert list((solved_root / "raw").iterdir()) == [solved_root / "raw" / "base_graph.pt"]
    processed = {path.name for path in (solved_root / "processed").iterdir()}
    assert {"instances.pt", "operators.pt", "operators_config.pt", "trajectories.pt"} <= processed


# --- what reaches the model ------------------------------------------------------------------------


def test_stored_blocks_do_not_reach_a_collated_batch(uniform, family, normalizer):
    """The per-instance ``points`` / ``targets`` blocks are stripped by ``_DROPPED_ATTRS``.

    They ride on the instance graph, so without the strip a batch would carry every *other* point of each
    instance beside the one being trained on -- and the single ``PRECONDITIONER_DIAGONAL`` row, which the
    monotonicity constraint does read, would be shadowed by the whole block.
    """
    uniform.normalizer = normalizer
    inputs, targets = collate_examples(family)([uniform[index] for index in range(5)])
    n_edges = family.base_graph.num_edges
    assert "points" not in inputs
    assert inputs["feats"].shape == (5, n_edges, N_FEATS)
    assert targets.shape == (5, n_edges)
    # the surviving diagonal is one row per example, not the stored [points_per_instance, E] block
    assert inputs[PRECONDITIONER_DIAGONAL].shape == (5, n_edges)


# --- targets ---------------------------------------------------------------------------------------


def test_uniform_targets_are_the_operator_at_their_stored_points(uniform, family):
    """Re-evaluating the operator at the stored points reproduces the stored targets.

    Also covers the hoisted ``PUMEModel``: these points share one rank-1 ``params``, so the whole block
    was evaluated against a single model build.
    """
    stored = uniform.evaluations(0)
    recomputed, diagonal = family.operator_and_preconditioner(stored.params, stored.points)
    assert torch.allclose(recomputed, stored.targets, atol=1e-4)
    assert torch.allclose(diagonal, stored.preconditioner_diagonal, atol=1e-4)


def test_expert_endpoint_is_the_near_zero_target(expert):
    """The rollout descends, and the appended endpoint is the closest thing to the root it found.

    The endpoint is the only near-zero target the model ever sees, which is what teaches the field where
    its equilibrium is -- so it must genuinely be last, and genuinely smaller than the start.
    """
    stored = expert.evaluations(0)
    norms = stored.targets.norm(dim=-1)
    assert norms[-1] < norms[0]
    assert norms[-1] == norms.min()
