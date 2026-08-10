"""The equilibrium/operator dataset contract on a GAMUT root -- the invariants of
``test_operator_datasets.py``, exercised on the flat-game graph path.

Two layers, split by what they need:

- **Offline (the main coverage).** ``generate_payoffs`` is monkeypatched with a zero-sum draw from the
  global RNG, so generation is monotone (the rollout solver converges) and needs no Java. The patch is
  active only *while roots are built*: everything the serving tests do afterwards -- indexing, collation,
  workers -- runs unpatched, which is itself the claim that serving a cached root never invokes GAMUT.
- **Java-gated.** Real ``java -jar gamut.jar`` invocations, skipped unless a runtime and the jar are
  present (mirroring the Sioux Falls data skip): the output really is zero-sum and normalized, the
  ``-random_seed`` drawn from the global RNG makes instances reproducible, and a tiny root generates
  end to end.
"""

import pathlib
import subprocess

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from l2s_games.data import (
    PRECONDITIONER_DIAGONAL,
    GlobalStandardizer,
    Normalizer,
    Standardizer,
    collate_normalized_examples,
)
from l2s_games.datasets import EquilibriumDataset
from l2s_games.envs import make_game
from l2s_games.envs.gamut import build_gamut_base_graph
from l2s_games.gamut import default_jar, generate_payoffs
from l2s_games.operator_datasets import ExpertOperatorDataset, UniformOperatorDataset

N_ACTIONS = 3
N_PLAYERS = 2
CHART_DIM = N_ACTIONS - 1
DOMAIN_DIM = N_PLAYERS * CHART_DIM
N_FEATS = CHART_DIM + N_ACTIONS**2  # per-node: chart coords | own flattened payoff matrix
N_INSTANCES = 3
POINTS_PER_INSTANCE = 3
N_CAL = 2
N_STEPS = 30


def fake_generate_payoffs(gamut_class, n_actions, jar_path, min_payoff=-1.0, max_payoff=1.0, options=()):
    """Zero-sum payoffs from the global RNG, double-centered around a per-instance **interior** Nash.

    Zero-sum makes the field monotone (the offline rollouts converge); the centering (``A y* = alpha 1``
    and ``A^T x* = beta 1``, so the tangential field vanishes exactly at a Dirichlet-drawn ``(x*, y*)``)
    gives every instance its own interior equilibrium -- so the expert endpoint really is the smallest
    target and the per-instance ``reference_equilibrium`` rows really differ. Like the real GAMUT call,
    reproducible only through the global seed.
    """
    raw = min_payoff + (max_payoff - min_payoff) * torch.rand(n_actions, n_actions)
    x, y = torch.distributions.Dirichlet(torch.full((2, n_actions), 5.0)).sample()
    ones = torch.ones(n_actions)
    a = raw - torch.outer(raw @ y, ones) - torch.outer(ones, raw.T @ x) + (x @ raw @ y) * torch.ones(n_actions, n_actions)
    return a, -a


def build_root(root, cls=EquilibriumDataset, **point_kwargs):
    """Generate a dataset the way ``generate_gamut_dataset.py`` does, at test scale."""
    torch.manual_seed(0)
    options = ("-actions", str(N_ACTIONS), str(N_ACTIONS))  # the fake ignores these; the real jar needs them
    graph = build_gamut_base_graph("RandomZeroSum", N_ACTIONS, options=options)
    graph.game = "gamut"
    family = make_game("gamut", base_graph=graph, solve_steps=300)
    if cls is not EquilibriumDataset:
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


@pytest.fixture(autouse=True)
def offline_gamut(request, monkeypatch):
    """Patch the GAMUT call away for every offline test (the java-gated ones opt out)."""
    if "gamut_jar" in request.keywords:
        return
    monkeypatch.setattr("l2s_games.envs.gamut.generate_payoffs", fake_generate_payoffs)


@pytest.fixture(scope="module")
def uniform_root(tmp_path_factory):
    """A shared uniform root, built under the patch and *served* outside it: after this fixture returns,
    any code path that reached for GAMUT again would run the real (absent) jar and fail loudly."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("l2s_games.envs.gamut.generate_payoffs", fake_generate_payoffs)
        root = tmp_path_factory.mktemp("uniform") / "root"
        dataset = build_root(root, UniformOperatorDataset, points_per_instance=POINTS_PER_INSTANCE)
    return dataset, root


def identity_collate(family):
    normalizer = Normalizer(
        Standardizer(torch.zeros(N_FEATS), torch.ones(N_FEATS)),
        GlobalStandardizer(torch.zeros(()), torch.ones(())),
    )
    return collate_normalized_examples(family, normalizer)


# --- the class split, on a GAMUT root ----------------------------------------------------------------


def test_both_sources_share_one_set_of_solves(tmp_path):
    root = tmp_path / "root"
    uniform = build_root(root, UniformOperatorDataset, points_per_instance=POINTS_PER_INSTANCE)
    equilibria = torch.stack([instance.equilibrium for instance in uniform.instances()])
    raw_mtime = (root / "raw" / "instances.pt").stat().st_mtime

    expert = build_root(root, ExpertOperatorDataset, algo="optimistic", h=0.2, n_steps=N_STEPS)

    assert (root / "raw" / "instances.pt").stat().st_mtime == raw_mtime, "the solves were re-run"
    assert torch.equal(torch.stack([instance.equilibrium for instance in expert.instances()]), equilibria)
    assert uniform.points_per_instance == POINTS_PER_INSTANCE
    # optimistic evaluates once per step plus a first-step bootstrap; the endpoint adds one more example
    assert expert.points_per_instance == N_STEPS + 2


def test_a_solve_only_root_has_no_processed_directory_and_no_cal_set(tmp_path):
    root = tmp_path / "root"
    dataset = build_root(root)

    assert not (root / "processed").exists()
    assert not (root / "raw" / "cal_instances.pt").exists()
    assert len(dataset) == N_INSTANCES
    assert "equilibrium" in dataset[0]
    assert "points" not in dataset[0]


def test_the_calibration_set_is_disjoint_from_the_dataset(uniform_root):
    dataset, _root = uniform_root
    cal = dataset.cal_instances()

    assert len(cal) == N_CAL
    assert dataset.len() == N_INSTANCES, "the calibration instances leaked into the dataset"
    dataset_equilibria = {tuple(instance.equilibrium.tolist()) for instance in dataset.instances()}
    assert not dataset_equilibria & {tuple(instance.equilibrium.tolist()) for instance in cal}


def test_the_root_records_its_family_and_gamut_provenance(uniform_root):
    """``game`` names the family and the GAMUT class/options pin the distribution the instances were
    drawn from; the base graph carries no equilibrium of its own (clones would inherit it)."""
    dataset, _root = uniform_root
    assert dataset.base_graph.game == "gamut"
    assert dataset.base_graph.gamut_class == "RandomZeroSum"
    assert "equilibrium" not in dataset.base_graph
    assert "payoff" not in dataset.base_graph


# --- stored per instance, served per point -----------------------------------------------------------


def test_flat_index_decodes_to_the_right_instance_and_point(uniform_root):
    dataset, _root = uniform_root
    assert len(dataset) == N_INSTANCES * POINTS_PER_INSTANCE
    for index in range(len(dataset)):
        instance, point = divmod(index, POINTS_PER_INSTANCE)
        item, target = dataset[index]
        stored = dataset.evaluations(instance)
        assert torch.equal(item["point"], stored.points[point])
        assert torch.equal(target, stored.targets[point])
        assert torch.equal(item[PRECONDITIONER_DIAGONAL], stored.preconditioner_diagonal[point])


def test_a_contiguous_subset_is_instance_disjoint(uniform_root):
    dataset, _root = uniform_root
    held_out = Subset(dataset, range(2 * POINTS_PER_INSTANCE, 3 * POINTS_PER_INSTANCE))

    assert len(held_out) == POINTS_PER_INSTANCE
    own_points = dataset.evaluations(2).points
    for item, _target in held_out:
        assert any(torch.equal(item["point"], point) for point in own_points)


# --- what reaches the model --------------------------------------------------------------------------


def test_a_dataloader_with_workers_yields_batches(uniform_root):
    """The worker-boundary path: the family is rebuilt per worker (dropped by ``__getstate__``), and the
    collated inputs have the player-graph shapes -- node feats, not the traffic line graph."""
    _dataset, root = uniform_root
    dataset = UniformOperatorDataset(str(root))
    dataset[0]  # make the family live in the main process first, so the drop is what is covered
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=identity_collate(dataset.family))
    inputs, targets = next(iter(loader))

    assert inputs["feats"].shape == (2, N_PLAYERS, N_FEATS)
    assert targets.shape == (2, DOMAIN_DIM)


def test_stored_blocks_and_provenance_do_not_reach_a_collated_batch(uniform_root):
    """``model_input`` strips the per-instance blocks and provenance, while ``payoff``, ``equilibrium``
    and the single preconditioner row survive -- the batched operator and ``reference_equilibrium``
    read them off the batch."""
    dataset, _root = uniform_root
    inputs, targets = identity_collate(dataset.family)([dataset[i] for i in range(4)])

    assert "points" not in inputs and "targets" not in inputs
    assert "gamut_class" not in inputs and "gamut_options" not in inputs and "game" not in inputs
    assert inputs["feats"].shape == (4, N_PLAYERS, N_FEATS)
    assert inputs["payoff"].shape == (4, N_PLAYERS, N_ACTIONS, N_ACTIONS)
    assert inputs["equilibrium"].shape == (4, DOMAIN_DIM)
    assert inputs[PRECONDITIONER_DIAGONAL].shape == (4, DOMAIN_DIM)  # one row, not the block
    assert targets.shape == (4, DOMAIN_DIM)


def test_reference_equilibrium_is_each_instances_own(uniform_root):
    dataset, _root = uniform_root
    batch_size = POINTS_PER_INSTANCE + 1  # spans a block boundary, so two different instances
    inputs, _targets = identity_collate(dataset.family)([dataset[i] for i in range(batch_size)])
    equilibrium = dataset.family.reference_equilibrium(inputs)

    assert equilibrium.shape == (batch_size, DOMAIN_DIM)
    assert torch.allclose(equilibrium[0], dataset.get(0).equilibrium.float())
    assert not torch.allclose(equilibrium[0], equilibrium[-1]), "every row got the same instance's z*"


# --- targets -----------------------------------------------------------------------------------------


def test_uniform_targets_are_the_operator_at_their_stored_points(uniform_root):
    dataset, _root = uniform_root
    stored = dataset.evaluations(0)
    values, diagonal = dataset.family.operator_and_preconditioner(stored.params, stored.points)

    assert torch.allclose(values, stored.targets, atol=1e-5)
    assert torch.allclose(diagonal, stored.preconditioner_diagonal)


def test_expert_points_are_a_converging_rollout_plus_its_endpoint(tmp_path):
    """``optimistic`` (the rotation-safe driver) queries once per step after a one-off first-step
    bootstrap eval; the appended endpoint is the one near-zero target that teaches the field where its
    root is."""
    dataset = build_root(tmp_path / "root", ExpertOperatorDataset, algo="optimistic", h=0.2, n_steps=N_STEPS)
    stored = dataset.evaluations(0)

    assert stored.points.shape == (N_STEPS + 2, DOMAIN_DIM)
    norms = stored.targets.norm(dim=-1)
    assert norms[-1] < norms[0]
    assert norms[-1] == norms.min()


# --- the real thing, when a JVM and the jar are around -----------------------------------------------


def _gamut_available():
    if not pathlib.Path(default_jar()).exists():
        return False
    try:  # macOS ships a /usr/bin/java stub that fails without a runtime, so probe it for real
        return subprocess.run(["java", "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


gamut_jar = pytest.mark.skipif(not _gamut_available(), reason="needs a Java runtime and $GAMUT_JAR")


@pytest.mark.gamut_jar
@gamut_jar
def test_real_random_zero_sum_is_zero_sum_and_normalized():
    torch.manual_seed(0)
    a, b = generate_payoffs("RandomZeroSum", N_ACTIONS, default_jar(), options=["-actions", "3", "3"])
    assert torch.allclose(a + b, torch.zeros_like(a), atol=1e-6)
    assert a.abs().max() <= 1.0 + 1e-6


@pytest.mark.gamut_jar
@gamut_jar
def test_the_global_seed_reproduces_an_instance():
    """The repo's seeding rule holds through the subprocess: the GAMUT -random_seed is drawn from the
    global torch RNG, so re-seeding reproduces the instance exactly."""
    options = ["-actions", "3", "3"]
    torch.manual_seed(7)
    first, _ = generate_payoffs("RandomZeroSum", N_ACTIONS, default_jar(), options=options)
    torch.manual_seed(7)
    again, _ = generate_payoffs("RandomZeroSum", N_ACTIONS, default_jar(), options=options)
    assert torch.equal(first, again)


@pytest.mark.gamut_jar
@gamut_jar
def test_a_real_root_generates_end_to_end(tmp_path):
    dataset = build_root(tmp_path / "root", UniformOperatorDataset, points_per_instance=2)
    assert dataset.len() == N_INSTANCES
    assert all("equilibrium" in instance for instance in dataset.instances())
