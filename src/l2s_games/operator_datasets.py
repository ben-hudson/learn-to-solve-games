"""On-disk caches of ``(instance, point) -> operator value`` training examples.

Evaluating the operator is the expensive part (a route-choice demand solve per point for traffic), so
these pay it once, offline, and persist the result as PyG ``InMemoryDataset``s -- the same move
``datasets.SolvedInstanceDataset`` makes for equilibria. That replaces the streaming sources (see
``data.OperatorStream`` and its subclasses), whose budget was
``n_workers * n_instances * points_per_instance * ceil(epochs / refresh_every)``: a formula tied to the
optimization budget and to ``persistent_workers``. Here the budget is ``n_instances *
points_per_instance``, fixed when the cache is generated and independent of how long you train on it.

Two sources, differing *only* in where an instance's points come from -- the axis
``OperatorStream._raw_stream`` factors on -- so they share a base class and one hook:

- ``UniformOperatorDataset`` draws them uniformly over the calibrated cost box (``sample_domain``).
- ``ExpertTrajectoryOperatorDataset`` records the states a converging algorithm actually visits while
  rolling out the **true** operator, plus the equilibrium it lands on.

Both live in the **same root** as the ``SolvedInstanceDataset`` whose equilibria calibrated their
sampling box, sharing its ``raw/base_graph.pt`` -- so the asymmetric family's coupling matrices exist in
exactly one place per root and cannot disagree between the train and val/test data (see
``envs/asym_pume_traffic`` for why a second copy would be unsafe). ``processed_file_names`` is what keeps
them apart inside it.

Why not in ``datasets.py``: the expert dataset rolls out an algorithm, so it needs ``algorithms`` +
``dynamics``, while ``datasets.py`` depends only on torch/PyG/tqdm. Keeping that module a leaf is worth
a second file.
"""

from abc import ABC, abstractmethod

import torch
import torch_geometric.data
import tqdm

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import (
    OperatorEvaluations,
    normalize_example,
    operator_examples,
    sample_and_eval_operator,
)
from l2s_games.dynamics import simulate
from l2s_games.rollout_sampling import RecordedField, with_endpoint

# Attrs the stored `Data` carries beyond the instance graph itself: one instance's evaluated points and
# their operator values. `traffic._DROPPED_ATTRS` lists them too, so `model_input` strips them from the
# example it builds and they never reach a collated batch (the model reads only `feats`).
_EVALUATION_ATTRS = ("points", "targets", "preconditioner_diagonal")


class OperatorDataset(torch_geometric.data.InMemoryDataset, ABC):
    """Base for an on-disk cache of operator examples: stores per *instance*, indexes per *point*.

    The stored unit is one instance's ``OperatorEvaluations`` -- its graph plus ``points`` / ``targets`` /
    ``preconditioner_diagonal``, each ``[points_per_instance, E]``. Storing whole instances rather than
    expanded examples is deliberate: ``model_input`` clones the graph per point (~13 KB for traffic), so a
    file of examples would cost ``points_per_instance`` times as much. ``operator_examples`` rebuilds one
    example at a time in ``get``, so the clone is transient.

    ``len`` and ``get`` therefore disagree with the underlying store: this dataset has
    ``n_instances * points_per_instance`` items, and ``get`` decodes a flat index into
    ``(which instance, which point)``. Subclasses implement one hook, ``evaluate_instance``.

    Args:
        root: shared with the root's ``SolvedInstanceDataset`` -- ``raw/base_graph.pt`` must already
            exist there, since the sampling ceiling is calibrated from that dataset's equilibria.
        family: the VI family. Required to *generate*; also used on every read, for ``model_input`` and
            ``transform``.
        n_instances: how many instances to draw and evaluate (generation only).
        points_per_instance: points to evaluate per instance. A *generation* input -- the subclass that
            needs it reads it in ``evaluate_instance`` -- and on a read it is overwritten by the stored
            value, which is authoritative (see ``_read_config``). ``None`` on a read adopts the file's.
        normalizer: fitted on the train split and applied lazily per ``get``; settable afterwards, since
            it is fit *from* this dataset.
        quiet: suppress the generation progress bar.
        **kwargs: forwarded to ``InMemoryDataset``. Leave ``transform`` unset -- PyG would apply it to
            the ``(item, target)`` tuple ``get`` returns; the family's own ``transform`` is applied inside.
    """

    def __init__(
        self,
        root,
        family=None,
        n_instances=None,
        points_per_instance=None,
        normalizer=None,
        quiet=False,
        **kwargs,
    ):
        self.family = family
        self.n_instances = n_instances
        self.normalizer = normalizer
        self.quiet = quiet
        # Assigned before super().__init__, which may run process() (which reads it) -- and before
        # anything can call len(), which multiplies by it. _read_config then makes the file authoritative.
        self.points_per_instance = points_per_instance
        super().__init__(root, **kwargs)
        self.load(self.processed_paths[0])
        self._read_config()

    @property
    def raw_file_names(self):
        """Shared with ``SolvedInstanceDataset``, so one root holds one ``base_graph`` (hence one copy of
        the asymmetric coupling matrices) for its instances *and* its operator examples."""
        return ["base_graph.pt"]

    @property
    @abstractmethod
    def processed_file_names(self):
        """``[evaluations, config]`` -- distinct per subclass so both can share one root's processed dir."""

    @abstractmethod
    def evaluate_instance(self, params):
        """One instance's ``OperatorEvaluations``: pick its points and evaluate the operator there.

        The single axis the subclasses differ on. Per-instance rather than batched, because for PUME a
        batch dimension is a Python loop over independent ``PUMEModel`` builds anyway (see
        ``pume_traffic.operator_and_preconditioner``) -- so batching buys nothing and would force the base
        class into a batch-shaped hook for no gain.
        """

    def download(self):
        raise AssertionError(
            f"no base graph at {self.raw_paths[0]}. Generate this root's SolvedInstanceDataset first "
            "(scripts/generate_traffic_dataset.py): its equilibria calibrate the sampling ceiling these "
            "examples are drawn under."
        )

    def process(self):
        """Draw ``n_instances`` fresh instances, evaluate each, and cache them with a config sidecar.

        Instances are drawn fresh from the family rather than reused from ``instances.pt`` -- matching the
        streaming ``uniform`` source's semantics, and leaving the noise free to differ from the solved
        instances' (which is the out-of-distribution axis; see ``_config``).
        """
        assert self.family is not None and self.n_instances is not None, (
            f"no cache at {self.processed_paths[0]}. Pass family and n_instances to build the dataset."
        )
        progress = range(self.n_instances) if self.quiet else tqdm.trange(self.n_instances)
        records = [self.evaluate_instance(self.family.sample_params()) for _ in progress]
        self.save([self._to_data(record) for record in records], self.processed_paths[0])
        torch.save(self._config(records[0]), self.processed_paths[1])

    def _config(self, record):
        """The sidecar written beside the evaluations.

        ``points_per_instance`` is **load-bearing**: a reader needs it to decode a flat index, and it is
        measured off the generated data rather than predicted, so the expert dataset does not have to know
        its rollout's width in advance. The noise settings are **provenance**: nothing reads them back and
        nothing asserts on them, which is the point -- generating the train examples under different noise
        from the solved instances is how out-of-distribution generalization gets measured, so the record
        has to distinguish the two without constraining them.
        """
        return {
            "points_per_instance": len(record.points),
            "noise_scale": self.family.noise_scale,
            "noise_type": self.family.noise_type,
        }

    def _read_config(self):
        """Adopt the stored ``points_per_instance``, and reject a constructor arg that disagrees.

        The file is authoritative. A mismatch has to raise rather than resolve silently: ``process()`` is
        skipped whenever the cache exists, so passing a new value *looks* like it should regenerate and
        cannot -- and quietly keeping either value would decode every index into the wrong
        ``(instance, point)`` pair.
        """
        stored = torch.load(self.processed_paths[1], weights_only=False)["points_per_instance"]
        assert self.points_per_instance in (None, stored), (
            f"{self.processed_paths[0]} holds {stored} points per instance, but "
            f"{self.points_per_instance} was requested. The cache is not regenerated when it already "
            "exists -- delete it, or pass the value it was built with."
        )
        self.points_per_instance = stored

    @staticmethod
    def _to_data(record):
        """One ``OperatorEvaluations`` as a storable ``Data``: the instance graph plus the three stacks.

        PyG's default ``__cat_dim__`` is 0 for non-``index`` keys and ``__inc__`` is 0, so ``collate``
        concatenates the stacks along their point axis and ``separate`` slices them back out per instance.
        """
        data = record.params.clone()
        for attr, value in zip(_EVALUATION_ATTRS, (record.points, record.targets, record.preconditioner_diagonal)):
            data[attr] = value
        return data

    @property
    def n_stored_instances(self):
        """How many instances the file holds -- ``len(self)`` counts *examples*, one per point."""
        return super().len()

    def evaluations(self, index):
        """The ``index``-th instance's stored ``OperatorEvaluations``, unexpanded.

        The cheap way in: no ``model_input`` clone and no ``transform``, so a caller wanting the raw
        tensors in bulk -- fitting a normalizer over every target, say -- does not pay a graph clone and a
        line-graph shortest-path solve per *point*, which indexing through ``get`` would cost.
        """
        data = super().get(index)
        return OperatorEvaluations(data, *(data[attr] for attr in _EVALUATION_ATTRS))

    def len(self):
        return self.n_stored_instances * self.points_per_instance

    def get(self, index):
        """The ``index``-th ``(model input, target)`` example, featurized and normalized on access.

        Nothing featurized is cached: ``operator_examples`` -> ``model_input`` clones the instance and
        ``normalize_example`` applies the family ``transform`` fresh, exactly as every other source does.
        """
        instance, point = divmod(index, self.points_per_instance)
        raw, target = next(operator_examples(self.family, self.evaluations(instance), order=[point]))
        return normalize_example(raw, target, self.family.transform, self.normalizer)


class UniformOperatorDataset(OperatorDataset):
    """Operator examples at points drawn **uniformly** over the calibrated cost box.

    The on-disk equivalent of ``data.UniformSampledOperatorStream``: every example is a fresh instance's
    ``sample_domain`` draw, so coverage is spread over the whole segment a rollout traverses rather than
    concentrated on the path one takes. One ``operator_and_preconditioner`` call per instance covers all
    its points, and because those points share a rank-1 ``params`` that call builds a single ``PUMEModel``
    for the lot.

    ``points_per_instance`` (on the base) is required to generate and is what the stored value is checked
    against on a read.
    """

    @property
    def processed_file_names(self):
        return ["operators.pt", "operators_config.pt"]

    def evaluate_instance(self, params):
        return sample_and_eval_operator(self.family, params, self.points_per_instance)


class ExpertTrajectoryOperatorDataset(OperatorDataset):
    """Operator examples at the states a converging algorithm visits on the **true** operator.

    The expert demonstration source: instead of covering the domain, it covers the path a good solver
    takes, which is the distribution a learned field is actually rolled out on. ``RecordedField`` keeps
    every ``(state, value, diagonal)`` triple the rollout asks for, so the evaluations the algorithm
    already paid for *are* the training targets -- one example per evaluation, no re-solving. Recording at
    the field level keeps it algorithm-agnostic: ``extragradient`` queries twice per step and
    ``projection`` once, and both are legitimate ``(state, operator value)`` pairs.

    The converged endpoint is appended as one extra example. It is the one state the algorithm never
    queried (``simulate`` returns it without evaluating there), and the only near-zero target the model
    ever sees -- so it is what teaches the field where its root is.

    ``points_per_instance`` is therefore ``n_steps * evals_per_step + 1``, uniform across instances since
    every rollout runs the full ``n_steps``. ``algo`` must be chosen per operator: ``projection`` for the
    potential and multiplicatively-coupled families, ``optimistic`` once the additive coupling makes the
    field rotation-dominated (see ``envs/asym_pume_traffic``).
    """

    def __init__(self, root, algo="projection", h=0.1, n_steps=500, **kwargs):
        self.algo = algo
        self.h = h
        self.n_steps = n_steps
        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return ["trajectories.pt", "trajectories_config.pt"]

    def evaluate_instance(self, params):
        """Roll this instance out on the true operator from a uniform start, keeping every evaluation.

        Single-instance (rank-1 ``params``), so ``simulate`` is driven directly rather than through
        ``rollout_sampling.batched_rollout``, which is shaped for a collated batch. The field is negated
        for descent -- toward the operator's zero -- so the *recorded* values stay the unnegated operator,
        which is the target convention the whole pipeline regresses.
        """
        start = self.family.sample_domain(params, 1)[0]
        recorder = RecordedField(self.family, params)
        trajectory = simulate(
            lambda z: -recorder(z),
            ALGORITHMS[self.algo](self.h),
            start,
            self.n_steps,
            project=lambda z: self.family.project(params, z),
        )
        endpoint = trajectory[-1]
        target, diagonal = self.family.operator_and_preconditioner(params, endpoint)
        return with_endpoint(recorder.evaluations(params), endpoint, target, diagonal)
