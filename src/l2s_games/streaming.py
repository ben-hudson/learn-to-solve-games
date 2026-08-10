"""The **streaming** training path: data sources that generate their examples at train time.

Everything here belongs to ``scripts/train_field_gnn.py`` and ``scripts/train_field_mlp.py`` -- the trainers
that build their data as they go, drawing fresh instances and solving the operator per batch. Nothing else
imports it, which is why it lives in one file: the cached path
(``datasets`` / ``operator_datasets`` / ``scripts/train_field_simple.py``) reads pre-solved roots off disk
instead, so this whole module retires with those two scripts.

Three groups, all with the same reason for being here rather than in ``data`` / ``operator_count`` /
``callbacks``:

- **Sources.** ``OperatorStream`` and its subclasses (plus ``caching`` / ``instance_sampling`` /
  ``rollout_sampling``, which build on it) are infinite iterables that solve the operator inside
  ``DataLoader`` workers. ``LazyOperatorDataset`` and ``build_dataset`` /
  ``build_streaming_{operator,solution}_dataset`` assemble their fixed cal/val/test counterparts eagerly at
  startup -- which is the thing a cached root makes unnecessary.
- **The operator budget.** ``SharedCounter`` and ``OperatorCountCallback`` exist because a stream's spend
  grows with the epoch count, so it has to be measured. A cached source's budget is ``len(train)``, known
  before training starts, so neither is needed there.
- **``AsinhWarp`` and ``VizRolloutCallback``**, reachable only through ``--target_warp`` and the on-policy
  training mode respectively.

``fit_normalizer`` in ``data`` supersedes the ``_fit_normalizer`` here: that one fits from a materialized
list of examples, which only an eager source can hand it.
"""

import functools
import multiprocessing as mp
import os

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import (
    GlobalStandardizer,
    Normalizer,
    Standardizer,
    _clone,
    _collate_examples,
    eval_operator,
    normalize_example,
    operator_examples,
    sample_and_eval_operator,
    solution_examples,
)
from l2s_games.dynamics import simulate
from l2s_games.viz import plot_trajectory_arrows


class SharedCounter:
    """Process-safe cumulative counter shared across ``DataLoader`` workers (``spawn``-safe).

    Backed by a ``Manager`` ``Value`` + ``Lock`` proxy pair -- both picklable and reconnecting to the
    manager server across the ``spawn`` boundary -- so the counter can be baked into the picklable
    ``family_factory`` and shared by every streaming worker and the main process. ``add`` is atomic
    under the lock. ``__getstate__`` drops the unpicklable ``Manager`` (kept alive in the main
    process) so only the proxies are pickled into workers.
    """

    def __init__(self):
        self._manager = mp.Manager()
        self._value = self._manager.Value("q", 0)
        self._lock = self._manager.Lock()

    def add(self, n):
        with self._lock:
            self._value.value += n

    @property
    def value(self):
        return self._value.value

    def __getstate__(self):
        return {"_value": self._value, "_lock": self._lock}

    def __setstate__(self, state):
        self.__dict__.update(state)


class AsinhWarp(nn.Module):
    """Stateless, invertible tail-compressing warp: ``transform(z) = asinh(z)``, ``inverse = sinh``.

    Composed by ``Normalizer`` *after* the (fitted) target scale, so ``asinh(y/scale)`` factors as this
    warp on the scaled target. Odd and zero-preserving (``0 -> 0``, so the equilibrium is untouched) and
    ~linear near 0 / logarithmic in the tail (compresses a heavy-tailed field smoothly rather than
    hard-clipping). Smooth and invertible, so it stays jacrev-transparent in the inference field. No
    fitted state -- the warp is a config choice (see ``--target_warp``), not learned from data.
    """

    def transform(self, z):
        return torch.asinh(z)

    def inverse_transform(self, w):
        return torch.sinh(w)


class LazyOperatorDataset(Dataset):
    """Lazily featurize + normalize raw ``(input item, target)`` examples held in memory as a list.

    ``__getitem__`` clones the raw item, applies the family's ``transform`` (builds ``feats`` fresh),
    then standardizes ``feats`` and the target -- so no featurized tensor is ever cached.

    Serves the same ``(instance, point)`` examples as
    ``operator_datasets.OperatorDataset``, and the two coexist deliberately. That one persists them
    to disk, which needs the sampling ceiling fixed *before* generation; the fixed cal/val/test splits
    cannot satisfy that, because the ceiling is calibrated from the very equilibria those splits are split
    from. So they are built eagerly here at startup instead, and this class wraps the resulting list.
    """

    def __init__(self, examples, transform, normalizer):
        self.examples = examples
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        raw, target = self.examples[index]
        return normalize_example(raw, target, self.transform, self.normalizer)


def collate_examples(family):
    """DataLoader ``collate_fn``: batch inputs via the family's seam, targets via ``default_collate``.

    Returns the ``(inputs, target)`` tuple ``FieldModel`` trains on. For flat games ``collate_fn``
    is ``default_collate``, so this reduces to stacking dicts; traffic overrides it with a dense
    graph stack. Returns a picklable ``functools.partial`` (not a closure) so the streaming train
    loader's workers can pickle it; ``family.collate_fn`` is a staticmethod, picklable by reference
    and free of the route-choice solver.
    """
    return functools.partial(_collate_examples, family.collate_fn)


def examples_at_points(family, params, points, index=None):
    """A list of raw ``(model_input, target)`` examples for one instance at explicit ``points``.

    ``eval_operator`` + ``operator_examples`` for the sources that build their examples eagerly (the
    fixed splits, the fixed-instance stream and the on-policy collector, which must re-evaluate because
    it rolls out the *learned* field). Buffered sources hold the ``OperatorEvaluations`` instead and skip
    this.
    """
    return list(operator_examples(family, eval_operator(family, params, points), index=index))


def _solve_instance(family, params, points_per_instance):
    """The ``(raw input item, target)`` examples for one instance: sample points, evaluate the operator.

    Shared by the eager ``_examples_for_instances`` and the streaming ``UniformSampledOperatorStream``,
    both of which pass the full ``points_per_instance`` so one evaluation call amortizes over that many
    points.
    """
    return list(operator_examples(family, sample_and_eval_operator(family, params, points_per_instance)))


def _examples_for_instances(family, instances, points_per_instance):
    """A flat list of ``(raw input item, target)`` examples over instances."""
    return [example for params in instances for example in _solve_instance(family, params, points_per_instance)]


def _fit_normalizer(
    family, examples, target_scaler=functools.partial(GlobalStandardizer.fit, center=False), warp="none"
):
    """Fit the feats standardizer and the target scaler (+ optional warp) on the train examples.

    Feats are per-feature standardized. ``target_scaler`` is a fit-callable ``targets -> module``: the
    default global-standardizes the operator *field* target with ``mean = 0`` (zero-preserving,
    isotropic); the solution baseline passes ``Standardizer.fit`` instead, treating the equilibrium
    ``z*`` as a generic per-feature-standardized target (see ``build_streaming_solution_dataset``).
    ``warp`` composes an optional stateless nonlinearity on the scaled target: ``"asinh"`` adds
    ``AsinhWarp`` (tames a heavy tail), ``"none"`` (the default everywhere) adds nothing, so the target
    stays linear in the field and preserves its direction.
    """
    transform = family.transform
    feats = torch.stack([transform(_clone(raw))["feats"] for raw, _ in examples])
    targets = torch.stack([target for _, target in examples])
    target_warp = AsinhWarp() if warp == "asinh" else None
    return Normalizer(Standardizer.fit(feats), target_scaler(targets), target_warp)


def build_dataset(family, n_train, n_val, n_test, points_per_instance):
    """Train/val/test ``FieldDataset``s plus the fitted ``Normalizer``.

    The normalizer is fit on the train split and shared with all three, so val/test contribute no
    statistics. Each split draws its own instances, so the test split measures generalization to
    unseen parametrizations. Pair with ``collate_examples(family)`` when building the DataLoaders.
    """
    splits = [
        _examples_for_instances(family, [family.sample_params() for _ in range(n)], points_per_instance)
        for n in (n_train, n_val, n_test)
    ]
    normalizer = _fit_normalizer(family, splits[0])
    datasets = tuple(LazyOperatorDataset(split, family.transform, normalizer) for split in splits)
    return datasets, normalizer


class OperatorStream(IterableDataset):
    """Base for an infinite stream of normalized ``(item, target)`` operator examples.

    Factors the shared seam every source needs: hold a picklable ``family_factory`` (not a live
    family) and build the family -- hence its route-choice solver -- **lazily inside each worker
    process** on first iteration, so nothing solver-related is pickled across the worker boundary;
    then featurize + normalize each raw ``(item, target)`` a subclass produces via ``_raw_stream``,
    exactly as a ``FieldDataset`` would. Subclasses implement ``_raw_stream(family)`` -- an infinite
    iterator of raw ``(input item, target)`` pairs -- to define *where* the examples come from
    (uniform sampling, on-policy rollouts, expert demonstrations, ...). Reproducible per-worker
    streams come from ``lightning.seed_everything(seed, workers=True)`` at the call site (the Trainer
    installs the per-worker seeding); this dataset owns no seeding of its own.
    """

    def __init__(self, family_factory, normalizer):
        self.family_factory = family_factory
        self.normalizer = normalizer

    def _raw_stream(self, family):
        """Infinite iterator of raw ``(input item, target)`` pairs -- defined by the subclass."""
        raise NotImplementedError

    def __iter__(self):
        # The family is built once per __iter__ (~once per worker per epoch), not per sample; the
        # per-epoch rebuild is cheap relative to a full epoch of solves.
        family = self.family_factory()
        transform = family.transform
        for raw, target in self._raw_stream(family):
            yield normalize_example(raw, target, transform, self.normalizer)


class UniformSampledOperatorStream(OperatorStream):
    """Infinite stream of freshly-sampled instances: one fresh instance -> normalized examples.

    Each step samples a new instance, ``points_per_instance`` domain points, solves the operator for
    the targets, and yields the normalized ``(item, target)`` pairs -- so every example is a distinct
    parametrization and minibatches are maximally diverse.
    """

    def __init__(self, family_factory, normalizer, points_per_instance):
        super().__init__(family_factory, normalizer)
        self.points_per_instance = points_per_instance

    def _raw_stream(self, family):
        while True:
            # One joint operator solve per fresh instance yields points_per_instance examples,
            # amortizing the expensive route-choice solve over that many training points.
            yield from _solve_instance(family, family.sample_params(), self.points_per_instance)


def build_streaming_operator_dataset(
    family_factory,
    cal_instances,
    val_instances,
    test_instances,
    points_per_instance,
    stream_factory=None,
    warp="none",
    cache_instances=0,
    refresh_every=1,
):
    """A streaming train dataset plus fixed val/test ``FieldDataset``s and the fitted ``Normalizer``.

    Operator-field target (``--amortization partial``): the model regresses the operator value at a
    domain point. The sibling ``build_streaming_solution_dataset`` builds the ``z*``-target variant.
    ``warp`` selects the target nonlinearity composed on the (global-standardized) field target --
    ``"none"`` (default: linear, so the field's direction survives) or ``"asinh"`` (tail compression).
    The tail is left to the family's supply-diagonal preconditioning, which handles it on the per-edge
    axis a global warp cannot; see ``--target_warp`` for that trade-off.

    The cal / val / test instances are pre-solved and passed in (split from a cached
    ``EquilibriumDataset``); this builds their ``(input, target)`` examples with the family's
    (calibrated) ``sample_domain`` + ``operator``. The normalizer is fit once on the **calibration**
    examples, then frozen and shared with the stream and the fixed val/test splits -- preserving the
    fit-on-a-fixed-sample invariant while training draws unbounded fresh instances. Val/test stay
    fixed so their metrics are stable across epochs. ``points_per_instance`` drives the train stream
    (points solved jointly per fresh instance) and the calibration density; val/test always solve each
    instance once -- the equilibrium rollout depends only on the instance, so extra points there just
    repeat identical rollouts. Pair with ``collate_examples(family)`` for the DataLoaders.

    ``stream_factory`` builds the train stream's per-worker family; it defaults to ``family_factory``.
    Pass a distinct factory (e.g. one carrying an operator-call counter) to instrument the train
    stream without counting the one-time cal/val/test build, which always uses ``family_factory``.

    ``cache_instances > 0`` swaps the unbounded uniform stream for the **cached** one (see
    ``caching.CachedOperatorStream``): that many fresh instances are solved per ``refresh_every``-epoch
    window and then reused, so the train operator budget is set by config rather than growing with the
    epoch count. ``0`` (the default) keeps the unbounded stream.
    """
    stream_factory = stream_factory or family_factory
    family = family_factory()
    cal = _examples_for_instances(family, cal_instances, points_per_instance)
    # Solve each fixed val/test instance once: the rollout residual is a function of the instance
    # alone (the sampled cost point is overwritten by the rollout state), so >1 point is redundant.
    val = _examples_for_instances(family, val_instances, 1)
    test = _examples_for_instances(family, test_instances, 1)
    normalizer = _fit_normalizer(family, cal, warp=warp)
    if cache_instances > 0:
        # Imported here: caching.py builds on this module's OperatorStream + group helpers, so a
        # module-level import would be a cycle.
        from l2s_games.caching import CachedOperatorStream

        train_ds = CachedOperatorStream(
            stream_factory, normalizer, points_per_instance, n_instances=cache_instances, refresh_every=refresh_every
        )
    else:
        train_ds = UniformSampledOperatorStream(stream_factory, normalizer, points_per_instance)
    val_ds, test_ds = (LazyOperatorDataset(split, family.transform, normalizer) for split in (val, test))
    # The calibration set (a fixed FieldDataset) doubles as the model-sizing sample source.
    cal_ds = LazyOperatorDataset(cal, family.transform, normalizer)
    return (train_ds, val_ds, test_ds, cal_ds), normalizer


def build_streaming_solution_dataset(family_factory, cal_instances, val_instances, test_instances):
    """Fixed cal/val/test ``z*``-target ``FieldDataset``s plus the fitted ``Normalizer``.

    The solution-target sibling of ``build_streaming_operator_dataset``. The fixed splits use each
    instance's cached ``equilibrium`` (exact and free), and the normalizer's target scaler is a
    per-feature ``Standardizer`` fit on those equilibria -- ``z*`` is a generic regression target, not
    a field, so it uses neither the field's global scale nor a warp. There is no ``train_ds`` -- the
    streaming train part is the expert solution stream (``ExpertOperatorStream(solution_target=True)``),
    built in the training script with its counting family + algorithm args. Pair with
    ``collate_examples(family)`` for the DataLoaders.
    """
    family = family_factory()
    cal, val, test = (
        solution_examples(family, instances) for instances in (cal_instances, val_instances, test_instances)
    )
    normalizer = _fit_normalizer(family, cal, target_scaler=Standardizer.fit, warp="none")
    val_ds, test_ds, cal_ds = (LazyOperatorDataset(split, family.transform, normalizer) for split in (val, test, cal))
    return (val_ds, test_ds, cal_ds), normalizer


class OperatorCountCallback(L.Callback):
    """Log the cumulative ground-truth operator point-evaluation budget once per epoch.

    The ``SharedCounter`` (see ``operator_count``) accumulates every training-data operator call
    across the streaming workers and the main process; this logs its current (monotonic) value, so
    the logged series *is* the cumulative-sum curve -- no in-dashboard cumsum needed. Logging goes
    through ``pl_module.log`` (never ``experiment.log``) so wandb's step bookkeeping stays in sync;
    register it as a ``wandb.define_metric`` step_metric at the call site to plot other metrics
    against the budget.

    Logged from **two** epoch-boundary hooks with ``on_epoch=True`` -- once in ``on_train_epoch_end``
    (pairing with ``train/loss`` / ``train/mse``) and once in ``on_validation_epoch_end`` (pairing with
    the ``val/*`` metrics). Both are needed because Lightning's ``WandbLogger.log_metrics`` does *not*
    forward a ``step`` to ``wandb.log`` -- it lets wandb auto-increment ``_step`` once per
    ``log_metrics`` call. The training-epoch flush and the validation-loop flush are separate
    ``log_metrics`` calls, so they land on *different* ``_step`` s: logging the budget only at
    ``on_train_epoch_end`` put it on the train flush's step, which no ``val/*`` metric ever shares, so
    selecting it as the custom x-axis for a val metric returned "no data" (nothing to pair against).
    Logging it in the validation flush too puts a copy on the val metrics' step (validation does not
    touch the training family's counter, so the value matches the same epoch's train-flush value).
    """

    def __init__(self, counter, key="train/operator_evals"):
        super().__init__()
        self.counter = counter
        self.key = key

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(self.key, float(self.counter.value), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Shares the validation flush's wandb _step so val/* metrics can be plotted against the budget.
        pl_module.log(self.key, float(self.counter.value), on_step=False, on_epoch=True)


class VizRolloutCallback(L.Callback):
    """Log the rollout trajectory + true/learned operators along it, for fixed instances through training.

    Each validation epoch, for each fixed held-out instance rolls out the learned field and draws one
    plot over the full domain: the trajectory as a blue line, with the true (crimson) and learned
    (blue) operators arrowed (magnitude-scaled, shared scale) at ``n_arrows`` points along it (see
    ``viz.plot_trajectory_arrows``). Both fields are shown because a lookahead/momentum algorithm does
    not step straight along the learned field, so the trajectory tangent isn't the learned direction.
    Logged as ``viz/rollout`` via the Lightning logger when it is wandb (so wandb's step bookkeeping
    stays consistent -- never ``experiment.log`` directly), else saved to
    ``{save_dir}/rollout_viz/epoch_{n}.png``.
    """

    def __init__(self, family, instances, algo, h, n_steps, save_dir, n_arrows=20):
        super().__init__()
        self.family = family
        self.instances = instances
        self.algo = algo
        self.h = h
        self.n_steps = n_steps
        self.save_dir = save_dir
        self.n_arrows = n_arrows
        # Fixed random starts, one per instance, so the trajectory across epochs is comparable.
        self.starts = [family.sample_domain(params, 1)[0] for params in instances]

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        n = len(self.instances)
        cols = min(n, 3)
        rows = -(-n // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
        axes = axes.ravel()
        for ax, params, z0 in zip(axes, self.instances, self.starts):
            true_field = lambda z, p=params: self.family.operator(p, z)
            learned_field = pl_module.conditioned_field(self.family, params)
            project = lambda z, p=params: self.family.project(p, z)
            traj = simulate(
                lambda z: -learned_field(z), ALGORITHMS[self.algo](self.h), z0, self.n_steps, project=project
            )
            summary = ", ".join(f"p{i}={v:.2f}" for i, v in enumerate(params.tolist()))
            plot_trajectory_arrows(
                ax, traj, true_field, learned_field, lim=self.family.lim, n_arrows=self.n_arrows, title=summary
            )
            ax.legend(fontsize=8, loc="upper right")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"rollout ({self.algo}): trajectory + true/learned operator -- epoch {epoch}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if isinstance(trainer.logger, WandbLogger):
            # Go through the Lightning logger (not experiment.log) so wandb's step counter stays in sync
            # with the metric logging -- a direct experiment.log desyncs the step and drops points on sync.
            trainer.logger.log_image(key="viz/rollout", images=[fig], step=trainer.global_step)
        else:
            out_dir = os.path.join(self.save_dir, "rollout_viz")
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"), dpi=110)
        plt.close(fig)
