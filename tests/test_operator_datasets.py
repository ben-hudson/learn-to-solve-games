"""Tests for the equilibrium/operator datasets: stored per instance, served per point.

The classes stage their work across PyG's raw/processed split because solving is ~1800x more expensive than
evaluating the operator, and they store per *instance* while the field model consumes per *point*. Nearly
everything that can go wrong lives in those two facts, plus the invariants the class hierarchy exists for:

1. **Both point sources share one set of solves.** Building the second source must not re-solve, and both must
   describe identical instances -- that is the entire reason each source is a subclass over shared ``raw/``
   rather than its own root. Were solving repeated, the result would still be correct, just ~1800x slower per
   instance, so nothing but this test would notice.
2. **The base class writes nothing.** It defines no ``process()``, so PyG must not create a ``processed/``
   directory at all: ``raw/instances.pt`` already *is* the solved instances, and a pass-through processed file
   would be a byte-identical second copy. It solves no calibration set either -- that is
   ``OperatorDataset``'s raw file.
3. **The calibration set never enters the dataset.** It is solved into its own raw file, so no train/val/test
   split can contain one and the sampling box cannot depend on a held-out equilibrium.
4. **The flat index decodes correctly.** ``len(dataset)`` counts examples (PyG's ``len()`` keeps counting
   instances), and consecutive indices within a block resolve to that instance's graph and to distinct points
   of it. A mis-decoding ``divmod`` would still yield well-formed examples, which is exactly why it needs
   asserting.
5. **A contiguous slice is instance-disjoint.** How train/val/test are taken, so it is load-bearing.
6. **A live family never crosses a worker boundary.** It holds the PUME solver, which holds a thread lock
   and cannot be pickled, so the dataset builds its family lazily and drops it from ``__getstate__``. Every
   other test builds in-process, so only an explicit ``num_workers > 0`` test covers this.
7. **The stored blocks never reach the model.** They ride on the instance graph, so without
   ``_DROPPED_ATTRS`` a batch would carry every *other* point of each instance beside the one being
   trained on.

Needs the Sioux Falls TNTP files (instances are solved for real); skips without them.
"""

import pathlib

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from l2s_games.data import (
    PRECONDITIONER_DIAGONAL,
    GlobalStandardizer,
    Normalizer,
    Standardizer,
    collate_normalized_examples,
    fit_normalizer,
)
from l2s_games.datasets import EquilibriumDataset
from l2s_games.envs import make_game
from l2s_games.envs.traffic import load_sioux_falls_base_graph
from l2s_games.operator_datasets import ExpertOperatorDataset, OperatorDataset, UniformOperatorDataset

_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "raw_data" / "sioux_falls"

GAME = "pume_traffic"
N_INSTANCES = 3
POINTS_PER_INSTANCE = 3
N_CAL = 2
N_STEPS = 4
N_FEATS = 9  # cost + 4 BPR attrs + 4 demand feats (see transforms.BuildTrafficEdgeData)
# Loose tolerances so the fixtures solve in seconds rather than minutes; these tests are about storage and
# indexing, not about how exactly the equilibria were reached.
SOLVER_KWARGS = {"outer_tol": 1e-1, "outer_max_iter": 500}


@pytest.fixture(scope="module")
def base_graph():
    if not _DATA_ROOT.exists():
        pytest.skip(f"Sioux Falls TNTP data not found at {_DATA_ROOT}")
    graph = load_sioux_falls_base_graph(str(_DATA_ROOT))
    graph.game = GAME  # the generation script's job; readers derive their family from it
    return graph


def build_root(root, base_graph, cls=EquilibriumDataset, **point_kwargs):
    """Generate a dataset the way ``generate_traffic_dataset.py`` does, at test scale.

    ``cls`` is what decides which processed file (if any) gets written -- the class *is* the point source.
    """
    torch.manual_seed(0)
    family = make_game(GAME, base_graph=base_graph, solver_kwargs=SOLVER_KWARGS)
    family.base_graph.game = GAME
    if cls is not EquilibriumDataset:  # the calibration set is the operator classes' raw file, not the base's
        point_kwargs["n_cal_instances"] = N_CAL
    return cls(
        str(root),
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solve_instance,
        n_instances=N_INSTANCES,
        quiet=True,
        **point_kwargs,
    )


def identity_collate(family):
    """A ``collate_normalized_examples`` with an identity normalizer: the collate tests here are about what
    reaches a batch, not about the fit."""
    normalizer = Normalizer(
        Standardizer(torch.zeros(N_FEATS), torch.ones(N_FEATS)),
        GlobalStandardizer(torch.zeros(()), torch.ones(())),
    )
    return collate_normalized_examples(family, normalizer)


@pytest.fixture(scope="module")
def uniform_root(tmp_path_factory, base_graph):
    root = tmp_path_factory.mktemp("uniform") / "root"
    dataset = build_root(root, base_graph, UniformOperatorDataset, points_per_instance=POINTS_PER_INSTANCE)
    return dataset, root


# --- the class split: one expensive stage, one processed file per source ----------------------------


def test_both_sources_share_one_set_of_solves(tmp_path, base_graph):
    """The property the whole class split exists for.

    Adding the expert source to a root that already has the uniform one must reuse the solves -- no
    ``force_reload``, since it is simply a different processed file -- and both must describe *identical*
    instances, so uniform-vs-expert varies only in where the points came from.
    """
    root = tmp_path / "root"
    uniform = build_root(root, base_graph, UniformOperatorDataset, points_per_instance=POINTS_PER_INSTANCE)
    equilibria = torch.stack([instance.equilibrium for instance in uniform.instances()])
    raw_mtime = (root / "raw" / "instances.pt").stat().st_mtime

    expert = build_root(root, base_graph, ExpertOperatorDataset, n_steps=N_STEPS)

    assert (root / "processed" / "uniform_operators.pt").exists(), "the first source's file was clobbered"
    assert (root / "processed" / "expert_operators.pt").exists()
    assert (root / "raw" / "instances.pt").stat().st_mtime == raw_mtime, "the solves were re-run"
    assert torch.equal(torch.stack([instance.equilibrium for instance in expert.instances()]), equilibria)
    assert uniform.points_per_instance == POINTS_PER_INSTANCE
    assert expert.points_per_instance == N_STEPS + 1


def test_a_solve_only_root_has_no_processed_directory_and_no_cal_set(tmp_path, base_graph):
    """``EquilibriumDataset`` defines no ``process()``, so PyG's ``has_process`` is false and nothing is
    written -- not even an empty directory. A pass-through processed file would duplicate
    ``raw/instances.pt`` byte for byte. Nor does it solve a calibration set: nothing here calibrates, so
    those solves belong to (and are paid for by) the operator classes."""
    root = tmp_path / "root"
    dataset = build_root(root, base_graph)

    assert not (root / "processed").exists()
    assert not (root / "raw" / "cal_instances.pt").exists()
    assert len(dataset) == N_INSTANCES
    assert "equilibrium" in dataset[0]
    assert "points" not in dataset[0]


def test_the_base_class_reads_an_operator_root_without_touching_it(tmp_path, base_graph):
    """Anything that wants only equilibria -- the solution model, the tuning scripts -- constructs the base
    class on *any* root and gets the raw instances: no example tensors loaded, and nothing written."""
    root = tmp_path / "root"
    build_root(root, base_graph, UniformOperatorDataset, points_per_instance=POINTS_PER_INSTANCE)
    before = sorted(path.name for path in (root / "processed").iterdir())

    dataset = build_root(root, base_graph)

    assert sorted(path.name for path in (root / "processed").iterdir()) == before
    assert "points" not in dataset[0], "the base class loaded the processed examples"


# --- the calibration set ---------------------------------------------------------------------------


def test_the_calibration_set_is_disjoint_from_the_dataset(uniform_root):
    """Solved into its own raw file and never added to the dataset, so no split can contain one and the
    sampling box is independent of every instance a consumer might hold out."""
    dataset, _root = uniform_root
    cal = dataset.cal_instances()

    assert len(cal) == N_CAL
    assert dataset.len() == N_INSTANCES, "the calibration instances leaked into the dataset"
    dataset_equilibria = {tuple(instance.equilibrium.tolist()) for instance in dataset.instances()}
    assert not dataset_equilibria & {tuple(instance.equilibrium.tolist()) for instance in cal}


def test_the_box_is_calibrated_from_the_cal_set_not_the_dataset(uniform_root):
    """``_calibrated_family`` reads the dedicated calibration file, all of it, and nothing else -- the
    whole point of the disjoint set. The dataset's own equilibria must leave the ceiling untouched."""
    dataset, _root = uniform_root
    family = dataset._calibrated_family()

    assert torch.equal(family.sampling_ceiling, type(family).calibrate_ceiling(dataset.cal_instances(), 3.0))
    assert not torch.allclose(
        family.sampling_ceiling, type(family).calibrate_ceiling(dataset.instances(), 3.0)
    )


def test_process_builds_its_family_through_the_calibration(tmp_path, base_graph, monkeypatch):
    """``process()`` must draw its points from the *calibrated* family.

    The test above pins what ``_calibrated_family`` computes; this one pins that generation actually goes
    through it. Verified by capturing the call, because the family ``process()`` builds is discarded -- and
    because a bounds check on the sampled points would only discriminate when the calibrated ceiling happens
    to be the tighter one, which is luck rather than a property.
    """
    calls = []
    real = OperatorDataset._calibrated_family

    def spy(self):
        calls.append(torch.stack([instance.equilibrium for instance in self.cal_instances()]))
        return real(self)

    monkeypatch.setattr(OperatorDataset, "_calibrated_family", spy)
    root = tmp_path / "root"
    dataset = build_root(root, base_graph, UniformOperatorDataset, points_per_instance=POINTS_PER_INSTANCE)

    assert len(calls) == 1, "process() did not build its family through _calibrated_family"
    assert torch.equal(
        calls[0], torch.stack([instance.equilibrium for instance in dataset.cal_instances()])
    )


def test_the_root_records_its_own_family_and_no_stale_equilibrium(uniform_root):
    """``game`` rides on the base graph so a reader is not told which family to use.

    The base graph itself carries no equilibrium: ``sample_params`` clones it, so one would be inherited by
    every noised instance as its own -- right shape, plausible magnitude, and silently wrong for
    ``calibrate_ceiling`` and the ``rel_dist`` reference.
    """
    dataset, _root = uniform_root
    assert dataset.base_graph.game == GAME
    assert "equilibrium" not in dataset.base_graph
    equilibria = {tuple(instance.equilibrium.tolist()) for instance in dataset.instances()}
    assert len(equilibria) == N_INSTANCES, "instances share an equilibrium -- one was inherited, not solved"


# --- stored per instance, served per point ---------------------------------------------------------


def test_length_counts_examples_and_pyg_len_counts_instances(uniform_root):
    """The one subtle bit of serving from the PyG class itself: ``__len__``/``__getitem__`` speak examples,
    while PyG's ``len()``/``get()`` keep speaking instances underneath (generation and the per-instance
    accessors are built on them)."""
    dataset, _root = uniform_root
    assert dataset.points_per_instance == POINTS_PER_INSTANCE
    assert len(dataset) == N_INSTANCES * POINTS_PER_INSTANCE
    assert dataset.len() == N_INSTANCES
    assert len(dataset.instances()) == N_INSTANCES


def test_flat_index_decodes_to_the_right_instance_and_point(uniform_root):
    """Example ``i`` carries point ``i % ppi`` of instance ``i // ppi``.

    Compared against the stored blocks directly, because a mis-decoding ``divmod`` would still return
    well-formed examples with plausible values.
    """
    dataset, _root = uniform_root
    for index in range(len(dataset)):
        instance, point = divmod(index, POINTS_PER_INSTANCE)
        item, target = dataset[index]
        stored = dataset.evaluations(instance)
        assert torch.equal(item["cost"], stored.points[point])
        assert torch.equal(target, stored.targets[point])
        assert torch.equal(item[PRECONDITIONER_DIAGONAL], stored.preconditioner_diagonal[point])


def test_points_within_an_instance_are_distinct(uniform_root):
    """Guards the failure where every index in a block resolves to the same point (e.g. ``order=[0]``),
    which would leave the length right and every value plausible."""
    dataset, _root = uniform_root
    points = dataset.evaluations(0).points
    assert len({tuple(row.tolist()) for row in points}) == POINTS_PER_INSTANCE


def test_a_contiguous_subset_is_instance_disjoint(uniform_root):
    """How train/val/test are taken: a contiguous ``Subset`` of examples splits on block boundaries, so it
    holds whole instances."""
    dataset, _root = uniform_root
    held_out = Subset(dataset, range(2 * POINTS_PER_INSTANCE, 3 * POINTS_PER_INSTANCE))

    assert len(held_out) == POINTS_PER_INSTANCE
    own_points = dataset.evaluations(2).points
    for item, _target in held_out:
        assert any(torch.equal(item["cost"], point) for point in own_points)


# --- what reaches the model ------------------------------------------------------------------------


def test_a_dataloader_with_workers_yields_batches(uniform_root):
    """A live family cannot cross a worker boundary: it holds the PUME solver, which holds a thread lock
    and cannot be pickled, so ``__getstate__`` must drop it for each worker to rebuild lazily. Indexing in
    the main process *first* makes the family live, so this covers the drop, not just the laziness -- and
    the failure mode is a dataloader that dies at startup, not a wrong number."""
    _dataset, root = uniform_root
    dataset = UniformOperatorDataset(str(root))
    dataset[0]
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=identity_collate(dataset.family))
    inputs, targets = next(iter(loader))
    n_edges = dataset.get(0).num_edges
    assert inputs["feats"].shape == (2, n_edges, N_FEATS)
    assert targets.shape == (2, n_edges)


def test_stored_blocks_do_not_reach_a_collated_batch(uniform_root):
    """The blocks ride on the instance graph, so ``_DROPPED_ATTRS`` must strip them -- otherwise a batch
    carries every *other* point of each instance beside the one being trained on, and the single
    ``PRECONDITIONER_DIAGONAL`` row the monotonicity constraint reads would be shadowed by the whole block."""
    dataset, _root = uniform_root
    n_edges = dataset.get(0).num_edges
    inputs, targets = identity_collate(dataset.family)([dataset[i] for i in range(4)])

    assert "points" not in inputs and "targets" not in inputs
    assert inputs["feats"].shape == (4, n_edges, N_FEATS)
    assert targets.shape == (4, n_edges)
    assert inputs[PRECONDITIONER_DIAGONAL].shape == (4, n_edges)  # one row, not the [ppi, E] block


def test_reference_equilibrium_is_each_instances_own_solved_cost(uniform_root):
    """The ``z*`` that ``val/{algo}/eq_dist`` measures against, read off the batch rather than threaded in.

    Per-instance is the whole point: a constant (the base family's ``0.0``, which the traffic families used to
    inherit) or one shared row would leave the shape right and the metric meaningless. ``float32`` because
    ``PUMESolver`` returns float64 and the batch goes to the model's device.
    """
    dataset, _root = uniform_root
    batch_size = POINTS_PER_INSTANCE + 1  # spans a block boundary, so two different instances
    inputs, _targets = identity_collate(dataset.family)([dataset[i] for i in range(batch_size)])
    equilibrium = dataset.family.reference_equilibrium(inputs)

    assert equilibrium.dtype == torch.float32
    assert equilibrium.shape == (batch_size, dataset.get(0).num_edges)
    assert torch.allclose(equilibrium[0], dataset.get(0).equilibrium.float())
    assert not torch.allclose(equilibrium[0], equilibrium[-1]), "every row got the same instance's z*"


# --- the normalizer fit ----------------------------------------------------------------------------


def test_raw_examples_agrees_with_flat_indexing(uniform_root):
    """The two handles onto the same data -- per-instance and flat -- must not drift.

    ``raw_examples`` exists so a caller can group by instance without redoing ``__getitem__``'s ``divmod``;
    if the two disagreed, a normalizer would be fit on different examples than training sees.
    """
    dataset, _root = uniform_root
    grouped = list(dataset.raw_examples(range(dataset.len())))

    assert len(grouped) == len(dataset)
    for (grouped_item, grouped_target), (flat_item, flat_target) in zip(grouped, dataset):
        assert torch.equal(grouped_item["cost"], flat_item["cost"])
        assert torch.equal(grouped_target, flat_target)


def test_raw_examples_order_selects_points_within_each_instance(uniform_root):
    """``order`` is per *instance*, so every instance contributes whatever it selects.

    The property that matters for a subsampled fit: a stride over the flat index would skip whole instances
    once it exceeded ``ppi``, while ``order`` cannot.
    """
    dataset, _root = uniform_root
    firsts = list(dataset.raw_examples(range(dataset.len()), order=[0]))

    assert len(firsts) == dataset.len()
    for index, (item, _target) in enumerate(firsts):
        assert torch.equal(item["cost"], dataset.evaluations(index).points[0])


def test_streamed_feats_fit_matches_a_stacked_one(uniform_root):
    """``Standardizer.fit_iter`` is an implementation detail of *how* the population is read, not a
    different statistic -- so it must agree with ``fit`` over the same stacked population.

    A tolerance rather than equality: sklearn accumulates the population std (ddof=0) where torch's default
    is unbiased, a ``sqrt(n/(n-1))`` factor. Also pins the zero-variance convention on traffic's constant
    ``b`` / ``power`` columns, which is the one place the two implementations could silently diverge.
    """
    dataset, _root = uniform_root
    feats = [dataset.family.transform(item)["feats"] for item, _ in dataset.raw_examples(range(dataset.len()))]

    stacked = Standardizer.fit(torch.stack(feats))
    streamed = Standardizer.fit_iter(iter(feats))

    assert torch.allclose(stacked.mean, streamed.mean, atol=1e-4)
    assert torch.allclose(stacked.std, streamed.std, rtol=1e-3)
    assert (streamed.std > 0).all(), "a constant feature divided by zero instead of mapping to 1"


def test_fit_normalizer_uses_only_the_instances_it_is_given(uniform_root):
    """The fit-on-train invariant, and the reason the caller passes instance indices.

    Fitting over the whole root must give a *different* normalizer than fitting over a strict subset --
    otherwise held-out instances would be contributing statistics with nothing to reveal it.
    """
    dataset, _root = uniform_root
    train = range(dataset.len() - 1)

    def fit(instances):
        return fit_normalizer(
            (dataset.family.transform(item)["feats"] for item, _ in dataset.raw_examples(instances)),
            torch.cat([dataset.evaluations(i).targets for i in instances]),
        )

    partial, whole = fit(train), fit(range(dataset.len()))

    assert not torch.allclose(partial.input.mean, whole.input.mean)
    assert not torch.allclose(partial.target.std, whole.target.std)


def test_fit_normalizer_sees_every_point_not_just_the_first(tmp_path, base_graph):
    """Fitting over every example must differ from fitting over one point per instance.

    On an expert root the trajectory descends, so point 0 is the rollout *start* and the rest sit nearer the
    equilibrium: a fit that quietly sampled only the starts would be calibrated to the wrong distribution.
    This is the assertion that keeps the fit exhaustive.
    """
    dataset = build_root(tmp_path / "root", base_graph, ExpertOperatorDataset, n_steps=N_STEPS)
    instances = range(dataset.len())

    def feats(order):
        return (dataset.family.transform(item)["feats"] for item, _ in dataset.raw_examples(instances, order))

    every = Standardizer.fit_iter(feats(None))
    starts_only = Standardizer.fit_iter(feats([0]))

    assert not torch.allclose(every.mean, starts_only.mean)


# --- targets ---------------------------------------------------------------------------------------


def test_uniform_targets_are_the_operator_at_their_stored_points(uniform_root):
    """Re-evaluating the operator at the stored points reproduces the stored targets.

    Also covers the hoisted ``PUMEModel``: these points share one rank-1 ``params``, so the whole block was
    evaluated against a single model build.
    """
    dataset, _root = uniform_root
    stored = dataset.evaluations(0)
    values, diagonal = dataset.family.operator_and_preconditioner(stored.params, stored.points)

    assert torch.allclose(values, stored.targets, atol=1e-4)
    assert torch.allclose(diagonal, stored.preconditioner_diagonal, atol=1e-4)


def test_expert_points_are_a_descending_rollout_plus_its_endpoint(tmp_path, base_graph):
    """``projection`` evaluates the field once per step, and the endpoint adds the one state the algorithm
    never queried -- the only near-zero target in the set, which is what teaches the field where its root
    is. So the width is ``n_steps + 1`` and the last target is the smallest."""
    dataset = build_root(tmp_path / "root", base_graph, ExpertOperatorDataset, algo="projection", n_steps=N_STEPS)
    stored = dataset.evaluations(0)

    assert stored.points.shape[0] == N_STEPS + 1
    norms = stored.targets.norm(dim=-1)
    assert norms[-1] < norms[0]
    assert norms[-1] == norms.min()
