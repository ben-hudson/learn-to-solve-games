"""Tests for the expert stream's recorded rollout (``rollout_sampling.RecordedField``).

Rolling out the *true* operator already solves it at every visited state, and those solves are exactly
the regression targets -- but ``simulate`` returns only the iterates, so they used to be discarded and a
subsample of the trajectory re-solved. Keeping them turns the rollout into its own data collection. Four
things are worth pinning:

1. **Correctness of the recording** -- a recorded value really is the operator at its recorded state, and
   its metric really maps back to the raw field. A mis-slice here would train the model on targets
   belonging to other states or other instances, which no accuracy metric would flag as a data bug.
2. **The budget** -- a window costs ``n_instances * (n_steps + 1)`` evaluations and yields that many
   examples, i.e. ~1 example per evaluation. This is the change's whole point.
3. **What was recorded** -- under ``projection`` the recorded states are the trajectory minus its
   endpoint, so nothing is double-counted and the endpoint solve (the ``+ 1``) is the only extra one.
4. **``full`` amortization is untouched** -- with a ``z*`` target nothing is recorded and the budget stays
   ``n_instances * n_steps``, so the solution baseline it feeds is unaffected.

Uses the flat ``rps`` family (a cheap matrix-product operator, and ``matrix.py`` implements the batched
seams the expert path needs: ``params_from_batch`` / ``collate_fn`` / ``project``), so the file runs
without a route-choice solve. The one exception is the ``z*``-target path, whose parameters-only input is
traffic-specific (it reads ``free_flow_time`` off the instance): that test rolls out two tiny traffic
instances instead, and skips when the solved-instance cache is absent.
"""

import functools
import pathlib

import pytest
import torch

from l2s_games.data import PRECONDITIONER_DIAGONAL, build_dataset, build_streaming_operator_dataset, split_instances
from l2s_games.datasets import SolvedInstanceDataset
from l2s_games.envs import make_game
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium
from l2s_games.rollout_sampling import ExpertOperatorStream

from helpers import CountingFamily, take

_DATASET_ROOT = pathlib.Path(__file__).resolve().parents[1] / "datasets" / "sioux_falls_512"

N_INSTANCES = 3
N_STEPS = 5
# One rollout evaluation per step (projection calls the field once), plus the endpoint residual solve.
EVALS_PER_INSTANCE = N_STEPS + 1
WINDOW = N_INSTANCES * EVALS_PER_INSTANCE


def rps_family():
    return make_game("rps")


@pytest.fixture
def family():
    torch.manual_seed(0)
    return CountingFamily(rps_family())


@pytest.fixture
def normalizer(family):
    _datasets, normalizer = build_dataset(family, n_train=4, n_val=1, n_test=1, points_per_instance=2)
    family.n_evaluations = 0  # the normalizer fit is not part of any budget under test
    return normalizer


@pytest.fixture(scope="module")
def traffic():
    """A counting traffic family + fitted normalizer, for the traffic-only ``z*``-target path."""
    if not _DATASET_ROOT.exists():
        pytest.skip(f"Solved-instance cache not found at {_DATASET_ROOT}")
    torch.manual_seed(0)
    dataset = SolvedInstanceDataset(str(_DATASET_ROOT))
    cal, val, test = split_instances(list(dataset), (2, 1, 1))
    factory = functools.partial(
        PUMEMarkovTrafficEquilibrium,
        dataset.base_graph,
        sampling_ceiling=PUMEMarkovTrafficEquilibrium.calibrate_ceiling(cal),
    )
    _splits, normalizer = build_streaming_operator_dataset(factory, cal, val, test, 1)
    return CountingFamily(factory()), normalizer


def expert_stream(family, normalizer, n_steps=N_STEPS, **kwargs):
    return ExpertOperatorStream(
        lambda: family,
        normalizer,
        "projection",
        h=0.05,
        n_steps=n_steps,
        n_instances=N_INSTANCES,
        refresh_every=10**6,
        **kwargs,
    )


def test_one_window_costs_one_evaluation_per_example(family, normalizer):
    """``n_instances * (n_steps + 1)`` evaluations, and that many distinct examples out of them."""
    stream = expert_stream(family, normalizer)
    examples = take(stream, WINDOW)
    assert family.n_evaluations == WINDOW
    assert sum(len(group.points) for group in stream._buffer[0]) == WINDOW
    points = {tuple(item["point"].tolist()) for item, _ in examples}
    assert len(points) == WINDOW


def test_recorded_values_are_the_operator_at_their_own_state(family, normalizer):
    """Every example's target, in real units, is the operator at that example's point and params."""
    stream = expert_stream(family, normalizer)
    for item, target in take(stream, 2 * WINDOW):
        expected, _metric = family.operator_and_preconditioner(item["params"], item["point"].unsqueeze(0))
        assert torch.allclose(normalizer.inverse_target(target), expected[0], atol=1e-5)


def test_metric_maps_the_recorded_target_back_to_the_raw_field(family, normalizer):
    """``metric_diagonal * target`` is the raw field -- for rps the operator is already raw, so ones."""
    stream = expert_stream(family, normalizer)
    for item, _target in take(stream, WINDOW):
        assert torch.equal(item[PRECONDITIONER_DIAGONAL], torch.ones_like(item[PRECONDITIONER_DIAGONAL]))


def test_recorded_states_are_the_trajectory_without_its_endpoint(family, normalizer):
    """Under projection the field sees each iterate once, so the recording is ``traj[:-1]`` per instance.

    Pins that the ``+ 1`` in the budget is the endpoint solve alone: each group is the ``n_steps`` visited
    states plus ``z*``, with no state counted twice.
    """
    stream = expert_stream(family, normalizer)
    take(stream, WINDOW)
    groups, _examples = stream._buffer
    assert len(groups) == N_INSTANCES
    for group in groups:
        assert len(group.points) == EVALS_PER_INSTANCE
        # The endpoint is one projection step beyond the last recorded state, so it is not a repeat.
        assert not torch.equal(group.points[-1], group.points[-2])


def test_solution_target_records_nothing(traffic):
    """``full`` amortization: ``n_instances * n_steps`` evaluations, ``n_instances`` z* examples, no groups.

    Traffic-only: the ``z*`` target's parameters-only input reads ``free_flow_time`` off the instance. Kept
    tiny (3 instances x 3 steps) since the point is the accounting, not convergence.
    """
    family, normalizer = traffic
    family.n_evaluations = 0
    stream = expert_stream(family, normalizer, n_steps=3, include_trajectory=False, solution_target=True)
    take(stream, N_INSTANCES)
    groups, examples = stream._buffer
    assert family.n_evaluations == N_INSTANCES * 3
    assert groups == []
    assert len(examples) == N_INSTANCES


def test_solutions_only_keeps_just_the_endpoints(family, normalizer):
    """The operator-target solutions-only baseline: one endpoint group per instance, path discarded."""
    stream = expert_stream(family, normalizer, include_trajectory=False)
    take(stream, N_INSTANCES)
    groups, _examples = stream._buffer
    assert [len(group.points) for group in groups] == [1] * N_INSTANCES
