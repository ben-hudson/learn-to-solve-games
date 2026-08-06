"""Tests for the two-stage dataset (``datasets.EquilibriumDataset``) and its example view.

The dataset stages its work across PyG's raw/processed split because solving is ~1800x more expensive than
evaluating the operator, and it stores per *instance* while the field model consumes per *point*. Nearly
everything that can go wrong lives in those two facts:

1. **The staging holds.** Regenerating the operator examples must reuse the cached solves. This is the
   whole reason solving lives in ``download()``, and it is the property a refactor is most likely to break
   silently -- the result would still be correct, just 1800x slower.
2. **The flat index decodes correctly.** ``len`` counts examples, and consecutive indices within a block
   resolve to that instance's graph and to distinct points of it. A mis-decoding ``divmod`` would still
   yield well-formed examples, which is exactly why it needs asserting.
3. **A contiguous slice is instance-disjoint.** How train/val/test are taken, so it is load-bearing.
4. **The second stage is optional.** No ``evaluate_fn`` leaves instances carrying only equilibria.
5. **A live family never crosses a worker boundary.** It holds the PUME solver, which holds a thread lock
   and cannot be pickled, so ``OperatorExamples`` takes a factory. Every other test builds in-process, so
   only an explicit ``num_workers > 0`` test covers this.
6. **The stored blocks never reach the model.** They ride on the instance graph, so without
   ``_DROPPED_ATTRS`` a batch would carry every *other* point of each instance beside the one being
   trained on.

Needs the Sioux Falls TNTP files (instances are solved for real); skips without them.
"""

import functools
import pathlib

import pytest
import torch
from torch.utils.data import DataLoader

from l2s_games.data import (
    PRECONDITIONER_DIAGONAL,
    GlobalStandardizer,
    Normalizer,
    Standardizer,
    collate_examples,
)
from l2s_games.datasets import EquilibriumDataset
from l2s_games.envs import make_game
from l2s_games.envs.traffic import load_sioux_falls_base_graph
from l2s_games.operator_datasets import POINT_SOURCES, OperatorExamples

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


def build_root(root, base_graph, point_source=None, **point_kwargs):
    """Generate a dataset the way ``generate_traffic_dataset.py`` does, at test scale."""
    torch.manual_seed(0)
    family = make_game(GAME, base_graph=base_graph, solver_kwargs=SOLVER_KWARGS)
    family.base_graph.game = GAME
    evaluate_fn = None
    if point_source is not None:
        evaluate_fn = functools.partial(
            POINT_SOURCES[point_source],
            game=GAME,
            base_graph=family.base_graph,
            n_cal=N_CAL,
            quiet=True,
            **point_kwargs,
        )
    return EquilibriumDataset(
        str(root),
        base_graph=family.base_graph,
        sample_fn=family.sample_params,
        solve_fn=family.solver.solve,
        n_instances=N_INSTANCES,
        evaluate_fn=evaluate_fn,
        quiet=True,
    )


def read_family(dataset):
    """The factory a consumer derives from the root -- the game comes off the data, not from a flag."""
    return functools.partial(make_game, dataset.base_graph.game, base_graph=dataset.base_graph)


def identity_normalizer():
    """These tests are about storage and indexing, not about the fit."""
    return Normalizer(
        Standardizer(torch.zeros(N_FEATS), torch.ones(N_FEATS)),
        GlobalStandardizer(torch.zeros(()), torch.ones(())),
    )


@pytest.fixture(scope="module")
def uniform_root(tmp_path_factory, base_graph):
    root = tmp_path_factory.mktemp("uniform") / "root"
    return build_root(root, base_graph, "uniform", points_per_instance=POINTS_PER_INSTANCE), root


@pytest.fixture
def examples(uniform_root):
    dataset, _root = uniform_root
    return OperatorExamples(list(dataset), read_family(dataset), identity_normalizer())


# --- the staging: the expensive half must be cached separately -------------------------------------


def test_regenerating_examples_reuses_the_cached_solves(tmp_path, base_graph):
    """The point of putting solving in ``download()``.

    Deleting the processed stage and rebuilding with a *different* point source must not re-solve: the
    equilibria come back bit-identical and the raw file is untouched. Were solving in ``process()`` this
    would still be correct, just ~1800x slower per instance -- so nothing but this test would notice.
    """
    root = tmp_path / "root"
    dataset = build_root(root, base_graph, "uniform", points_per_instance=POINTS_PER_INSTANCE)
    first = torch.stack([instance.equilibrium_cost for instance in dataset])
    raw_mtime = (root / "raw" / "instances.pt").stat().st_mtime

    (root / "processed" / "operator_examples.pt").unlink()
    rebuilt = build_root(root, base_graph, "expert", algo="projection", h=0.1, n_steps=N_STEPS)

    assert (root / "raw" / "instances.pt").stat().st_mtime == raw_mtime, "the solves were re-run"
    assert torch.equal(torch.stack([instance.equilibrium_cost for instance in rebuilt]), first)
    assert rebuilt[0].points.shape[0] == N_STEPS + 1, "the examples were not regenerated"


def test_the_second_stage_is_optional(tmp_path, base_graph):
    """No ``evaluate_fn`` -> instances carry their equilibria and no operator examples."""
    dataset = build_root(tmp_path / "root", base_graph)
    assert len(dataset) == N_INSTANCES
    assert "equilibrium_cost" in dataset[0]
    assert "points" not in dataset[0]


def test_the_root_records_its_own_family_and_no_stale_equilibrium(uniform_root):
    """``game`` rides on the base graph so a reader is not told which family to use.

    The base graph itself carries no equilibrium: ``sample_params`` clones it, so one would be inherited by
    every noised instance as its own -- right shape, plausible magnitude, and silently wrong for
    ``calibrate_ceiling`` and the ``rel_dist`` reference.
    """
    dataset, _root = uniform_root
    assert dataset.base_graph.game == GAME
    assert "equilibrium_cost" not in dataset.base_graph
    equilibria = {tuple(instance.equilibrium_cost.tolist()) for instance in dataset}
    assert len(equilibria) == N_INSTANCES, "instances share an equilibrium -- one was inherited, not solved"


# --- stored per instance, served per point ---------------------------------------------------------


def test_length_counts_examples_not_instances(examples):
    assert examples.points_per_instance == POINTS_PER_INSTANCE
    assert len(examples) == N_INSTANCES * POINTS_PER_INSTANCE


def test_flat_index_decodes_to_the_right_instance_and_point(examples):
    """Example ``i`` carries point ``i % ppi`` of instance ``i // ppi``.

    Compared against the stored blocks directly, because a mis-decoding ``divmod`` would still return
    well-formed examples with plausible values.
    """
    for index in range(len(examples)):
        instance, point = divmod(index, POINTS_PER_INSTANCE)
        item, target = examples[index]
        stored = examples.evaluations(instance)
        assert torch.equal(item["cost"], stored.points[point])
        assert torch.equal(target, stored.targets[point])
        assert torch.equal(item[PRECONDITIONER_DIAGONAL], stored.preconditioner_diagonal[point])


def test_points_within_an_instance_are_distinct(examples):
    """Guards the failure where every index in a block resolves to the same point (e.g. ``order=[0]``),
    which would leave the length right and every value plausible."""
    points = examples.evaluations(0).points
    assert len({tuple(row.tolist()) for row in points}) == POINTS_PER_INSTANCE


def test_a_contiguous_slice_is_instance_disjoint(uniform_root):
    """How train/val/test are taken: slicing the instances splits the examples on block boundaries."""
    dataset, _root = uniform_root
    instances = list(dataset)
    held_out = OperatorExamples(instances[2:], read_family(dataset), identity_normalizer())

    assert len(held_out) == POINTS_PER_INSTANCE
    own_points = held_out.evaluations(0).points
    for index in range(len(held_out)):
        item, _target = held_out[index]
        assert any(torch.equal(item["cost"], point) for point in own_points)


# --- what reaches the model ------------------------------------------------------------------------


def test_a_dataloader_with_workers_yields_batches(uniform_root):
    """``OperatorExamples`` must hold a *factory*: a live family holds the PUME solver, which holds a
    thread lock and cannot be pickled to a worker. Every other test builds in-process, so this is the only
    coverage -- and the failure mode is a dataloader that dies at startup, not a wrong number."""
    dataset, _root = uniform_root
    examples = OperatorExamples(list(dataset), read_family(dataset), identity_normalizer())
    loader = DataLoader(
        examples, batch_size=2, num_workers=2, collate_fn=collate_examples(read_family(dataset)())
    )
    inputs, targets = next(iter(loader))
    assert inputs["feats"].shape == (2, dataset[0].num_edges, N_FEATS)
    assert targets.shape == (2, dataset[0].num_edges)


def test_stored_blocks_do_not_reach_a_collated_batch(examples, uniform_root):
    """The blocks ride on the instance graph, so ``_DROPPED_ATTRS`` must strip them -- otherwise a batch
    carries every *other* point of each instance beside the one being trained on, and the single
    ``PRECONDITIONER_DIAGONAL`` row the monotonicity constraint reads would be shadowed by the whole block."""
    dataset, _root = uniform_root
    n_edges = dataset[0].num_edges
    inputs, targets = collate_examples(read_family(dataset)())([examples[i] for i in range(4)])

    assert "points" not in inputs and "targets" not in inputs
    assert inputs["feats"].shape == (4, n_edges, N_FEATS)
    assert targets.shape == (4, n_edges)
    assert inputs[PRECONDITIONER_DIAGONAL].shape == (4, n_edges)  # one row, not the [ppi, E] block


# --- targets ---------------------------------------------------------------------------------------


def test_uniform_targets_are_the_operator_at_their_stored_points(uniform_root):
    """Re-evaluating the operator at the stored points reproduces the stored targets.

    Also covers the hoisted ``PUMEModel``: these points share one rank-1 ``params``, so the whole block was
    evaluated against a single model build.
    """
    dataset, _root = uniform_root
    family = read_family(dataset)()
    stored = OperatorExamples(list(dataset), read_family(dataset)).evaluations(0)
    values, diagonal = family.operator_and_preconditioner(stored.params, stored.points)

    assert torch.allclose(values, stored.targets, atol=1e-4)
    assert torch.allclose(diagonal, stored.preconditioner_diagonal, atol=1e-4)


def test_expert_points_are_a_descending_rollout_plus_its_endpoint(tmp_path, base_graph):
    """``projection`` evaluates the field once per step, and the endpoint adds the one state the algorithm
    never queried -- the only near-zero target in the set, which is what teaches the field where its root
    is. So the width is ``n_steps + 1`` and the last target is the smallest."""
    dataset = build_root(tmp_path / "root", base_graph, "expert", algo="projection", h=0.1, n_steps=N_STEPS)
    stored = OperatorExamples(list(dataset), read_family(dataset)).evaluations(0)

    assert stored.points.shape[0] == N_STEPS + 1
    norms = stored.targets.norm(dim=-1)
    assert norms[-1] < norms[0]
    assert norms[-1] == norms.min()
