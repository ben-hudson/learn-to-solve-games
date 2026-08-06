"""Operator examples: datasets that generate an instance's cost points and serve them as training examples.

``OperatorDataset`` extends ``datasets.EquilibriumDataset`` (the solves) with everything the point sources
share, on both sides of the raw/processed split:

- **Generation.** ``download()`` additionally solves a *disjoint* calibration set (``raw/cal_instances.pt``),
  drawn from the same ``sample_fn``, whose equilibria calibrate the sampling box points are drawn over
  (``calibrate_ceiling``). Keeping them out of the dataset is what makes the box's independence from any
  val/test split **structural** rather than a convention about prefixes. ``process()`` then picks every
  instance's cost points, evaluates the operator there, and attaches the result; the subclasses differ *only*
  in where the points come from (``_evaluate``): ``UniformOperatorDataset`` spreads them over the calibrated
  cost box, ``ExpertOperatorDataset`` puts them on the path a converging algorithm actually walks on the true
  operator. Each owns its own processed file, so the class you name *is* the provenance -- there is nothing to
  record, and nothing that can disagree -- and both read the **same** ``raw/``, so one set of solves carries
  both sources over identical instances, equilibria and calibration box.
- **Serving.** The stored dataset holds *instances*; a field model wants *(model input, operator value)*
  examples. ``__len__``/``__getitem__`` therefore count and serve **examples** (PyG's ``len()``/``get()``
  stay per-instance underneath): instance ``i`` owns the contiguous range ``[i*ppi, (i+1)*ppi)``, so a flat
  index decodes with ``divmod`` and a **contiguous ``torch.utils.data.Subset`` is an instance-disjoint
  split**, which is how train/val/test are taken. The serving family is derived from the root itself
  (``base_graph.game`` plus the coupling matrices riding on the graph), so a reader is never told which
  operator the examples came from -- the silent-mismatch hazard ``base_graph.game`` exists for. Examples
  come out **raw** (real units, unfeaturized): the dataset carries only state derived from the root, like
  its parent, and normalization -- experiment state, fit on the train split after the dataset exists --
  enters at the DataLoader boundary via ``data.collate_normalized_examples``.

Because ``cal_instances.pt`` is this class's raw file, a solve-only root (a plain ``EquilibriumDataset``)
has no calibration set, and constructing an operator dataset on one re-runs the **whole** ``download()`` --
a full re-solve, reproducible under the same seed. Deliberately no per-file skip logic: a partial re-run
seeded identically would replay the instance RNG stream into the cal set, silently duplicating the first
``n_cal_instances`` dataset instances and violating the disjointness the set exists for. Generate operator
roots by default; reading one as ``EquilibriumDataset`` ignores the examples, so they are strictly more
capable.

Why the operator examples ride on the instances rather than living in their own dataset: they describe the
same instances, so two files would have to be kept describing the same thing -- which is what previously
forced a pinned-instance argument, a coverage cap, a config sidecar recording it, and cross-file matching at
load time. All of that was bookkeeping created by the split.
"""

import functools

import torch
import tqdm

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import (
    OperatorEvaluations,
    operator_examples,
    sample_and_eval_operator,
)
from l2s_games.datasets import EquilibriumDataset
from l2s_games.dynamics import simulate
from l2s_games.envs import GAMES, make_game
from l2s_games.envs.asym_pume_traffic import coupling_matrices
from l2s_games.rollout_sampling import RecordedField, with_endpoint

# The three per-instance tensors a point source produces, in `OperatorEvaluations` field order. Listed in
# `traffic._DROPPED_ATTRS` too, so `model_input` strips them from the example it builds: they ride on the
# instance graph, so without that a batch would carry every *other* point of each instance beside the one
# being trained on.
EVALUATION_ATTRS = ("points", "targets", "preconditioner_diagonal")


def _root_family(base_graph, **kwargs):
    """The family a root's ``base_graph`` names, carrying the couplings its equilibria were solved with.

    Everything rides on the root: ``base_graph.game`` picks the family, and for the asymmetric one the
    coupling matrices are on the graph too. Deriving them rather than being told removes the silent-mismatch
    hazard ``base_graph.game`` exists for.
    """
    couplings = coupling_matrices(base_graph) if base_graph.game == "asym_pume_traffic" else {}
    return make_game(base_graph.game, base_graph=base_graph, **couplings, **kwargs)


def _attach(instance, evaluations):
    """Attach one instance's evaluated points to it, in place."""
    for attr in EVALUATION_ATTRS:
        instance[attr] = getattr(evaluations, attr)


class OperatorDataset(EquilibriumDataset):
    """Everything the point sources share: the calibration set, the ``process()`` skeleton, and serving.

    A subclass states only its processed file name and where one instance's points come from
    (``_evaluate`` / ``_desc``). See the module docstring for the design.

    Args:
        root: as ``EquilibriumDataset``, plus ``raw/cal_instances.pt`` and this source's processed file.
        n_cal_instances: how many *additional* instances to solve for the calibration set. No default, so a
            generating caller states it (tests want two, generation wants ~128).
        n_stds: how many sigma above the calibration equilibria's mean the sampling ceiling sits.
        **kwargs: forwarded to ``EquilibriumDataset``.
    """

    def __init__(self, root, n_cal_instances=None, n_stds=3.0, **kwargs):
        self.n_cal_instances = n_cal_instances
        self.n_stds = n_stds
        self._family = None
        super().__init__(root, **kwargs)

    # --- generation ---------------------------------------------------------------------------------

    @property
    def raw_file_names(self):
        return [*super().raw_file_names, "cal_instances.pt"]

    def download(self):
        """The base solves plus the disjoint calibration set."""
        assert self.n_cal_instances is not None, "n_cal_instances is required to solve a fresh dataset"
        super().download()
        torch.save(self._solve(self.n_cal_instances, "solving cal set"), self.raw_paths[2])

    def cal_instances(self):
        """The disjoint calibration instances -- they set the sampling box and never enter the dataset."""
        return torch.load(self.raw_paths[2], weights_only=False)

    def _calibrated_family(self):
        """A family whose ``sample_domain`` box is calibrated from the dedicated calibration instances.

        Calibrates from *all* of ``cal_instances()`` -- they are already the dedicated disjoint set, so
        there is no prefix to slice and no held-out equilibrium to avoid.
        """
        base_graph = self.raw_base_graph()
        ceiling = GAMES[base_graph.game].calibrate_ceiling(self.cal_instances(), self.n_stds)
        return _root_family(base_graph, sampling_ceiling=ceiling)

    def process(self):
        """Attach each instance's operator examples -- the cheap stage, one processed file per source."""
        family = self._calibrated_family()
        instances = self.solved_instances()
        progress = instances if self.quiet else tqdm.tqdm(instances, desc=self._desc())
        for instance in progress:
            _attach(instance, self._evaluate(family, instance))
        self.save(instances, self.processed_paths[0])

    def load_instances(self):
        self.load(self.processed_paths[0])

    # --- serving ------------------------------------------------------------------------------------

    @property
    def family(self):
        """The root's own family: ``__getitem__`` uses its ``model_input``, and the DataLoader boundary its
        ``transform`` / ``collate_fn`` -- the points and targets are already stored, so nothing here touches
        the operator, the solver, or ``sample_domain``. Built lazily and dropped by ``__getstate__``, so
        each ``DataLoader`` worker constructs its own."""
        if self._family is None:
            self._family = _root_family(self.base_graph)
        return self._family

    def __getstate__(self):
        # A live family cannot cross a worker boundary: it holds the PUME solver, which holds a thread
        # lock and so cannot be pickled. Construction costs ~0.02 s, so rebuilding per worker is free.
        return {**self.__dict__, "_family": None}

    @functools.cached_property
    def points_per_instance(self):
        """Read off the stored data (lazily -- the store is empty during ``super().__init__``), so it is
        the truth about the file whatever this source was generated with."""
        return self.get(0).points.shape[0]

    def evaluations(self, index):
        """The ``index``-th instance's stored ``OperatorEvaluations``, unexpanded.

        The cheap read: no ``model_input`` clone, so a caller wanting the raw tensors in bulk -- fitting a
        normalizer over every target, say -- does not pay a ~13 KB graph clone per *point*, which indexing
        through ``__getitem__`` would cost.
        """
        instance = self.get(index)
        return OperatorEvaluations(instance, *(instance[attr] for attr in EVALUATION_ATTRS))

    def instances(self):
        """Per-instance access, now that indexing means examples: the solved instances with their
        examples attached."""
        return [self.get(i) for i in range(self.len())]

    def __len__(self):
        """Examples, not instances -- PyG's ``len()`` keeps counting instances underneath."""
        return self.len() * self.points_per_instance

    def __getitem__(self, index):
        """Rebuild one **raw** example on access -- real units, nothing featurized, nothing cached.
        Featurization and normalization happen at the DataLoader boundary
        (``data.collate_normalized_examples``).

        Overrides the dunder rather than PyG's ``get`` so the per-point index never meets PyG's
        per-instance machinery (whose ``transform`` would also be applied to the tuple this returns).
        """
        instance, point = divmod(index, self.points_per_instance)
        return next(operator_examples(self.family, self.evaluations(instance), order=[point]))


class UniformOperatorDataset(OperatorDataset):
    """Cost points drawn **uniformly** over the calibrated box, one block per instance.

    Coverage of the whole segment a rollout traverses -- from the free-flow-time start up to a few sigma past
    the calibration equilibria -- rather than of the path any one algorithm takes. One
    ``operator_and_preconditioner`` call covers an instance's whole block, and since those points share a
    rank-1 ``params`` that call builds a single ``PUMEModel`` for the lot.
    """

    def __init__(self, root, points_per_instance=32, **kwargs):
        # Stored as n_points: it is the generation knob, while the points_per_instance *property* reads the
        # stored blocks -- so a read of an existing root serves whatever width it was generated with.
        self.n_points = points_per_instance
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return ["uniform_operators.pt"]

    def _desc(self):
        return "uniform points"

    def _evaluate(self, family, instance):
        return sample_and_eval_operator(family, instance, self.n_points)


class ExpertOperatorDataset(OperatorDataset):
    """Cost points visited by rolling out the **true** operator, plus the equilibrium the rollout reaches.

    The expert demonstration source: it covers the distribution a learned field is actually rolled out on
    rather than the whole domain. ``RecordedField`` keeps every ``(state, value, diagonal)`` triple the rollout
    asks for, so the evaluations the algorithm already paid for *are* the training targets -- one example per
    evaluation, nothing re-solved. Recording at the field level keeps it algorithm-agnostic: ``extragradient``
    queries twice per step and ``projection`` once, and both are legitimate pairs.

    The converged endpoint is appended as one extra example. It is the one state the algorithm never queried
    (``simulate`` returns it without evaluating there) and the only near-zero target in the set, so it is what
    teaches the field where its root is. Hence ``n_steps * evals_per_step + 1`` points per instance, uniform
    across instances since every rollout runs the full ``n_steps``.

    ``algo`` must be chosen per operator: ``projection`` for the potential and multiplicatively-coupled
    families, ``optimistic`` once additive coupling makes the field rotation-dominated (see
    ``envs/asym_pume_traffic``).
    """

    def __init__(self, root, algo="projection", h=0.1, n_steps=500, **kwargs):
        self.algo = algo
        self.h = h
        self.n_steps = n_steps
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return ["expert_operators.pt"]

    def _desc(self):
        return f"{self.algo} rollouts"

    def _evaluate(self, family, params):
        """One instance's rollout on the true operator from a uniform start, keeping every evaluation.

        Single-instance (rank-1 ``params``), so ``simulate`` is driven directly rather than through
        ``rollout_sampling.batched_rollout``, which is shaped for a collated batch. The field is negated for
        descent -- toward the operator's zero -- so the *recorded* values stay the unnegated operator, which is
        the target convention the whole pipeline regresses.
        """
        start = family.sample_domain(params, 1)[0]
        recorder = RecordedField(family, params)
        trajectory = simulate(
            lambda z: -recorder(z),
            ALGORITHMS[self.algo](self.h),
            start,
            self.n_steps,
            project=lambda z: family.project(params, z),
        )
        endpoint = trajectory[-1]
        target, diagonal = family.operator_and_preconditioner(params, endpoint)
        return with_endpoint(recorder.evaluations(params), endpoint, target, diagonal)


OPERATOR_DATASETS = {"uniform": UniformOperatorDataset, "expert": ExpertOperatorDataset}
