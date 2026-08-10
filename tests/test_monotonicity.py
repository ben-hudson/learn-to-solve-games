"""Tests for the monotonicity constraint (``l2s_games.monotonicity``).

Three things are worth pinning, in decreasing order of how quietly they could break:

1. The **metric round-trip** ``M * (M^-1 E) == E``. The whole design rests on constraining the *raw*
   field rather than the preconditioned one the model predicts, so if the metric diagonal is wrong the
   constraint silently enforces monotonicity on a field that does not have it.
2. The **sign convention** of the ratio. An inverted sign turns the constraint into a push *away* from
   monotonicity, and nothing else in the pipeline would notice.
3. The **batch grouping**, i.e. that pairs are formed within an instance and never across instances --
   monotonicity is a property of one operator, so a cross-instance pair is meaningless.
"""

import functools
import pathlib

import pytest
import torch

from l2s_games.data import INSTANCE_INDEX, PRECONDITIONER_DIAGONAL, split_instances
from l2s_games.streaming import build_streaming_operator_dataset, collate_examples
from l2s_games.envs import make_game
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium, load_sioux_falls_base_graph
from l2s_games.instance_sampling import FixedInstanceOperatorStream
from l2s_games.monotonicity import monotonicity_ratios, monotonicity_violations

from helpers import solved_traffic_root

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sioux_falls"
_POINTS_PER_INSTANCE = 4


@pytest.fixture(scope="module")
def base_graph():
    if not _DATA_ROOT.exists():
        pytest.skip(f"Sioux Falls TNTP data not found at {_DATA_ROOT}")
    return load_sioux_falls_base_graph(str(_DATA_ROOT))


@pytest.fixture(scope="module")
def solved_batch(tmp_path_factory):
    """A real collated batch from the fixed-instance stream, plus the family that built it."""
    torch.manual_seed(0)
    dataset = solved_traffic_root(tmp_path_factory.mktemp("monotonicity") / "root", n_instances=8)
    cal, val, test = split_instances(list(dataset), (4, 2, 2))
    factory = functools.partial(
        PUMEMarkovTrafficEquilibrium,
        dataset.base_graph,
        sampling_ceiling=PUMEMarkovTrafficEquilibrium.calibrate_ceiling(cal),
    )
    _splits, normalizer = build_streaming_operator_dataset(factory, cal, val, test, _POINTS_PER_INSTANCE)
    family = factory()
    instances = [family.sample_params() for _ in range(3)]
    stream = FixedInstanceOperatorStream(factory, normalizer, instances, _POINTS_PER_INSTANCE)
    items = [example for example, _ in zip(iter(stream), range(2 * _POINTS_PER_INSTANCE))]
    return family, normalizer, collate_examples(family)(items)


# --- the metric round-trip: the assumption the raw-field constraint rests on ----------------------


def test_preconditioner_diagonal_maps_operator_back_to_raw(base_graph):
    """``M * (M^-1 E) == E``, and the diagonal is all ones when the operator is already raw.

    Several points share one instance, so this also pins that the hoisted ``PUMEModel`` (built once per
    rank-1 ``params``, see ``operator_and_preconditioner``) gives each row what a per-row build did.
    """
    torch.manual_seed(0)
    preconditioned = make_game("pume_traffic", base_graph=base_graph, precondition=True)
    raw = make_game("pume_traffic", base_graph=base_graph, precondition=False)
    graph = preconditioned.sample_params()
    points = preconditioned.sample_domain(graph, 3)

    values, diagonal = preconditioned.operator_and_preconditioner(graph, points)
    raw_values, raw_diagonal = raw.operator_and_preconditioner(graph, points)

    assert torch.allclose(diagonal * values, raw_values, rtol=1e-4, atol=1e-4)
    assert torch.equal(raw_diagonal, torch.ones_like(raw_diagonal))
    assert (diagonal >= 1.0).all()  # PUME's supply diagonal is floored at 1


def test_operator_still_matches_operator_and_preconditioner(base_graph):
    """``operator`` delegates without changing its result, and costs one evaluation per row, not two."""
    torch.manual_seed(0)
    family = make_game("pume_traffic", base_graph=base_graph)
    graph = family.sample_params()
    points = family.sample_domain(graph, 3)

    before = family.operator_counter.value
    values, _metric = family.operator_and_preconditioner(graph, points)
    assert torch.equal(family.operator(graph, points), values)
    assert family.operator_counter.value - before == 2 * len(points)  # one call each, not doubled


# --- sign convention and the divisor ---------------------------------------------------------------


def test_matching_is_one_to_one_by_halves():
    """``P`` points give ``P // 2`` disjoint pairs, matching ``i`` with ``i + P // 2`` (odd ``P`` drops one)."""
    torch.manual_seed(0)
    points, field = torch.randn(6, 3), torch.randn(6, 3)
    ratios = monotonicity_ratios(field, points, normalize=False)

    assert ratios.shape == (3,)
    expected = torch.stack([(field[i] - field[i + 3]) @ (points[i] - points[i + 3]) for i in range(3)])
    assert torch.allclose(ratios, expected, atol=1e-5)
    assert monotonicity_ratios(field[:5], points[:5], normalize=False).shape == (2,)


def test_ratio_sign_convention():
    """Identity field -> ``+1`` (monotone), negated -> ``-1``, skew -> ``0`` (rotational, not monotone-violating)."""
    torch.manual_seed(0)
    points = torch.randn(6, 4)
    skew = torch.randn(4, 4)
    skew = skew - skew.T

    assert torch.allclose(monotonicity_ratios(points, points), torch.ones(3), atol=1e-5)
    assert torch.allclose(monotonicity_ratios(-points, points), -torch.ones(3), atol=1e-5)
    assert torch.allclose(monotonicity_ratios(points @ skew.T, points), torch.zeros(3), atol=1e-5)


def test_unnormalized_ratio_is_the_raw_inner_product():
    """With ``normalize=False`` the same cases scale by ``||dx||^2`` -- which checks the divisor itself."""
    torch.manual_seed(0)
    points = torch.randn(4, 3)
    squared_distance = (points[:2] - points[2:]).pow(2).sum(dim=-1)

    assert torch.allclose(monotonicity_ratios(points, points, normalize=False), squared_distance, atol=1e-5)
    assert torch.allclose(monotonicity_ratios(-points, points, normalize=False), -squared_distance, atol=1e-5)


def test_differencing_survives_a_stiff_field():
    """A huge field with a tiny pair separation keeps the right sign -- the cancellation trap.

    The raw operator reaches ~1e12 near free-flow time, and a gram-expanded inner product
    (``d_i + d_j - K_ij - K_ji``) computes the answer as a difference of numbers that large, which in
    float32 loses the sign entirely. This pins the stable ordering: a monotone field at that scale must
    report exactly zero violation.
    """
    torch.manual_seed(0)
    points = 100.0 + torch.randn(8, 40)
    close = points + 1e-4 * torch.randn(8, 40)
    stacked_points = torch.cat([points, close])
    field = 1e10 * stacked_points  # monotone (a positive multiple of the identity), and enormous

    assert monotonicity_violations(field, stacked_points).max() == 0.0


def test_violations_are_one_sided():
    """A monotone field yields exactly zero violation; an anti-monotone one yields the depth."""
    torch.manual_seed(0)
    points = torch.randn(6, 3)
    assert torch.all(monotonicity_violations(points, points) == 0.0)
    assert torch.allclose(monotonicity_violations(-points, points), torch.ones(3), atol=1e-5)


# --- the constraint on real batches ----------------------------------------------------------------


def test_batch_carries_the_constraint_inputs(solved_batch):
    """The collated batch has everything the constraint reads: metric, instance tag, raw points."""
    family, _normalizer, (inputs, targets) = solved_batch
    batch_size = targets.shape[0]

    assert inputs[PRECONDITIONER_DIAGONAL].shape == (batch_size, family.base_graph.num_edges)
    assert inputs[PRECONDITIONER_DIAGONAL].dtype == torch.float32
    assert inputs[INSTANCE_INDEX].dtype == torch.long  # IndexedMultiplier (stage 2) requires long
    assert family.initial_point(inputs).shape == inputs[PRECONDITIONER_DIAGONAL].shape
    # The stream emits each instance's points contiguously, so a batch holds whole runs.
    _values, counts = inputs[INSTANCE_INDEX].unique(return_counts=True)
    assert (counts == _POINTS_PER_INSTANCE).all()


def test_analytic_raw_field_is_monotone_on_a_real_batch(solved_batch):
    """The **ground-truth** raw field violates nothing -- the gate on the whole raw-space argument.

    This is the batch-level counterpart of the offline sweep that found 0 / 2560 violating pairs. If it
    fails, the metric diagonal is being attached or applied incorrectly, and any constraint built on it
    would be enforcing monotonicity on a field that does not possess it.
    """
    family, normalizer, (inputs, targets) = solved_batch
    raw_field = normalizer.inverse_target(targets) * inputs[PRECONDITIONER_DIAGONAL]
    points = family.initial_point(inputs)
    instance_index = inputs[INSTANCE_INDEX]

    for index in instance_index.unique():
        rows = instance_index == index
        assert monotonicity_violations(raw_field[rows], points[rows]).max() == 0.0


def test_metric_is_what_makes_the_field_monotone(base_graph):
    """The raw field is monotone on close pairs where the **preconditioned** one is not.

    This is the premise the whole raw-space design rests on, and it only shows up at *small* pair
    separations: an offline sweep found the preconditioned operator violating 12.3% of pairs at
    separation ~1e-4 but only 0.2% at separation ~1, because ``M(x) ~ M(y)`` once the points are close in
    *fraction* but not in the reweighting they induce. Independent draws (the separation the training
    batches use today) therefore almost never expose it -- hence the deliberately close pairs here, and
    enough of them that seeing zero violations would be a real failure rather than sampling luck.

    Guards against a vacuous raw-space check: if the metric were dropped, mis-scaled, or applied to the
    wrong endpoint, the two spaces would stop differing and this would fail.
    """
    torch.manual_seed(0)
    n_pairs, separation = 60, 1e-4
    family = make_game("pume_traffic", base_graph=base_graph, precondition=True)
    graph = family.sample_params()
    x = family.sample_domain(graph, n_pairs)
    y = x + separation * (family.sample_domain(graph, n_pairs) - x)  # feasible: the box is convex

    # One batched call for both legs; per-pair ratios come from 2-point stacks so each pair keeps its own
    # separation (an all-pairs call would mix these close pairs with far ones).
    # One batched call for both legs. Stacking x then y makes the halves-matching pair each x with its
    # own y, so every pair keeps the intended separation.
    points = torch.cat([x, y])
    preconditioned, metric = family.operator_and_preconditioner(graph, points)
    raw = preconditioned * metric

    assert monotonicity_violations(raw, points).max() == 0.0
    assert (monotonicity_violations(preconditioned, points) > 0).any()
