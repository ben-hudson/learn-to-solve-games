"""Operator examples: where an instance's cost points come from, and how they are served to the model.

Two halves, matching the two stages of ``datasets.EquilibriumDataset``:

- **Generation** (``POINT_SOURCES``, the ``process()`` stage). A point source is a plain function
  ``instances -> [OperatorEvaluations]`` that picks each instance's cost points and evaluates the operator
  there. Two exist, and they differ *only* in where the points come from: ``uniform_points`` spreads them
  over the calibrated cost box, ``expert_points`` puts them on the path a converging algorithm actually
  walks on the true operator. Each is handed the **solved** instances, so it calibrates its own sampling
  ceiling from their equilibria -- which is why the ceiling never has to be threaded in from outside.
- **Reading** (``OperatorExamples``). The stored dataset serves *instances*; a field model wants
  *(instance, point)* examples. That is a view over the same file, not a second one.

Why the operator examples ride on the instances rather than living in their own dataset: they describe the
same instances, so two files would have to be kept describing the same thing -- which is what previously
forced a pinned-instance argument, a coverage cap, a config sidecar recording it, and cross-file matching at
load time. All of that was bookkeeping created by the split.
"""

import torch
import tqdm

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import (
    OperatorEvaluations,
    normalize_example,
    operator_examples,
    sample_and_eval_operator,
)
from l2s_games.dynamics import simulate
from l2s_games.envs import GAMES, make_game
from l2s_games.rollout_sampling import RecordedField, with_endpoint

# The three per-instance tensors a point source produces, in `OperatorEvaluations` field order. Listed in
# `traffic._DROPPED_ATTRS` too, so `model_input` strips them from the example it builds: they ride on the
# instance graph, so without that a batch would carry every *other* point of each instance beside the one
# being trained on.
EVALUATION_ATTRS = ("points", "targets", "preconditioner_diagonal")


def _calibrated_family(instances, game, base_graph, n_cal, n_stds, **family_kwargs):
    """A family whose ``sample_domain`` box is calibrated from the solved ``instances``.

    The calibration prefix is deliberately the *first* ``n_cal`` instances: consumers split train/val/test
    with train first, so the instances that set the sampling box are always inside train and the box never
    sees a held-out equilibrium.
    """
    ceiling = GAMES[game].calibrate_ceiling(instances[:n_cal], n_stds)
    return make_game(game, base_graph=base_graph, sampling_ceiling=ceiling, **family_kwargs)


def uniform_points(instances, points_per_instance=32, n_cal=128, n_stds=3.0, quiet=False, **family_kwargs):
    """Points drawn **uniformly** over the calibrated cost box, one block per instance.

    Coverage of the whole segment a rollout traverses -- from the free-flow-time start up to a few sigma
    past the equilibria -- rather than of the path any one algorithm takes. One
    ``operator_and_preconditioner`` call covers an instance's whole block, and since those points share a
    rank-1 ``params`` that call builds a single ``PUMEModel`` for the lot.
    """
    family = _calibrated_family(instances, n_cal=n_cal, n_stds=n_stds, **family_kwargs)
    progress = instances if quiet else tqdm.tqdm(instances, desc="uniform points")
    return [sample_and_eval_operator(family, instance, points_per_instance) for instance in progress]


def expert_points(instances, algo="projection", h=0.1, n_steps=500, n_cal=128, n_stds=3.0, quiet=False, **family_kwargs):
    """Points visited by rolling out the **true** operator, plus the equilibrium the rollout reaches.

    The expert demonstration source: it covers the distribution a learned field is actually rolled out on
    rather than the whole domain. ``RecordedField`` keeps every ``(state, value, diagonal)`` triple the
    rollout asks for, so the evaluations the algorithm already paid for *are* the training targets -- one
    example per evaluation, nothing re-solved. Recording at the field level keeps it algorithm-agnostic:
    ``extragradient`` queries twice per step and ``projection`` once, and both are legitimate pairs.

    The converged endpoint is appended as one extra example. It is the one state the algorithm never
    queried (``simulate`` returns it without evaluating there) and the only near-zero target in the set, so
    it is what teaches the field where its root is. Hence ``n_steps * evals_per_step + 1`` points per
    instance, uniform across instances since every rollout runs the full ``n_steps``.

    ``algo`` must be chosen per operator: ``projection`` for the potential and multiplicatively-coupled
    families, ``optimistic`` once additive coupling makes the field rotation-dominated (see
    ``envs/asym_pume_traffic``).
    """
    family = _calibrated_family(instances, n_cal=n_cal, n_stds=n_stds, **family_kwargs)
    progress = instances if quiet else tqdm.tqdm(instances, desc=f"{algo} rollouts")
    return [_rollout_evaluations(family, instance, algo, h, n_steps) for instance in progress]


def _rollout_evaluations(family, params, algo, h, n_steps):
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
        ALGORITHMS[algo](h),
        start,
        n_steps,
        project=lambda z: family.project(params, z),
    )
    endpoint = trajectory[-1]
    target, diagonal = family.operator_and_preconditioner(params, endpoint)
    return with_endpoint(recorder.evaluations(params), endpoint, target, diagonal)


POINT_SOURCES = {"uniform": uniform_points, "expert": expert_points}


class OperatorExamples(torch.utils.data.Dataset):
    """``(model input, operator value)`` examples over an ``EquilibriumDataset``'s attached evaluations.

    Stores per *instance*, serves per *point*: instance ``i`` owns the contiguous example range
    ``[i*ppi, (i+1)*ppi)``, so ``__getitem__`` decodes a flat index with ``divmod``. That layout is
    load-bearing twice over -- a file of expanded examples would clone the instance graph per point (~13 KB
    for traffic), and a **contiguous slice is therefore an instance-disjoint split**, which is how
    train/val/test are taken.

    A plain ``Dataset`` rather than an ``InMemoryDataset`` subclass, deliberately: nothing here owns files,
    so there is no ``len``/``get`` re-entrancy to work around when reading ``points_per_instance``, no
    collision with PyG's own ``transform`` (which would be applied to the ``(item, target)`` tuple this
    returns), and no second set of ``processed_file_names`` to keep distinct.

    Args:
        instances: the processed instances of an ``EquilibriumDataset`` (or a slice of them). Each must
            carry the ``EVALUATION_ATTRS``, i.e. the root was generated with an ``evaluate_fn``.
        family_factory: zero-arg callable returning the VI family -- **not** a live family. A family holds
            the PUME solver, which holds a thread lock and so cannot be pickled to a ``DataLoader`` worker;
            construction costs ~0.02 s, so building one per worker is free. Only ``model_input`` and
            ``transform`` are ever used from it: the points and targets are already stored, so nothing here
            touches the operator, the solver, or ``sample_domain``.
        normalizer: applied per ``__getitem__``; settable afterwards, since it is fit *from* this data.
    """

    def __init__(self, instances, family_factory, normalizer=None):
        assert all(attr in instances[0] for attr in EVALUATION_ATTRS), (
            "these instances carry no operator examples -- regenerate the dataset with an evaluate_fn "
            "(scripts/generate_traffic_dataset.py --operator-dataset uniform|expert)"
        )
        self.instances = instances
        self.family_factory = family_factory
        self.normalizer = normalizer
        self.points_per_instance = instances[0].points.shape[0]
        self._family = None

    @property
    def family(self):
        """Built lazily, so each ``DataLoader`` worker constructs its own rather than unpickling one."""
        if self._family is None:
            self._family = self.family_factory()
        return self._family

    def evaluations(self, index):
        """The ``index``-th instance's stored ``OperatorEvaluations``, unexpanded.

        The cheap read: no ``model_input`` clone and no ``transform``, so a caller wanting the raw tensors
        in bulk -- fitting a normalizer over every target, say -- does not pay a graph clone and a
        line-graph shortest-path solve per *point*, which indexing through ``__getitem__`` would cost.
        """
        instance = self.instances[index]
        return OperatorEvaluations(instance, *(instance[attr] for attr in EVALUATION_ATTRS))

    def __len__(self):
        return len(self.instances) * self.points_per_instance

    def __getitem__(self, index):
        """Featurize + normalize one example on access -- nothing featurized is ever cached."""
        instance, point = divmod(index, self.points_per_instance)
        raw, target = next(operator_examples(self.family, self.evaluations(instance), order=[point]))
        return normalize_example(raw, target, self.family.transform, self.normalizer)
