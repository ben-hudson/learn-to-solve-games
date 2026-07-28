"""Dataset construction for the amortized field model.

An *instance* is one ``params`` value drawn from the family; a *sample* is a point in that
instance's domain. A training example is ``(family.model_input(params, point), operator value)``,
where ``model_input`` returns the **raw** input item (``{point, params}`` for flat games, a PyG
``Data`` with ``.cost`` for traffic). ``FieldDataset`` applies the family's ``transform`` -- which
builds ``feats`` (and any structure) -- **lazily on every ``__getitem__``**, so nothing is cached;
featurization lives in ``transforms.py``. Train/val/test instances are drawn independently so the
test split measures generalization to unseen parametrizations.

Normalization is **not** a transform (it fits on train and inverts at inference), so it lives here
as ``Standardizer`` / ``Normalizer`` and is applied by this agnostic dataset layer via key access
(``item["feats"]`` works for both a dict and a ``Data``). It is fit on the train split only, so
val/test see no statistics of their own; constant features (e.g. traffic's ``b`` / ``power``) map
to 0 rather than dividing by zero. Reproducibility is via ``lightning.seed_everything`` at the
call site.
"""

import functools
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset, default_collate, random_split

# Item key for the per-coordinate diagonal mapping the operator target back to the raw field; see
# examples_at_points. Lives here (not in monotonicity.py, its consumer) so the data layer owns its own
# schema and nothing in the pipeline imports the constraint code.
METRIC_DIAGONAL = "metric_diagonal"
# Item key for the index of the example's instance within its source's instance set -- stable across
# epochs, and what the monotonicity constraint groups same-instance points by. Diagnostics-only for the
# model (the backbones read only feats + structure); it rides along on the collated batch. Lives here
# with METRIC_DIAGONAL for the same reason -- and because group_examples sets it, so a home in a module
# that imports this one would be a cycle.
INSTANCE_INDEX = "instance_index"


class Standardizer(nn.Module):
    """Per-feature ``(x - mean) / std`` map, fit from data, with its inverse.

    ``mean`` / ``std`` are registered buffers so the fitted stats move with the module (Lightning
    ships them to the model's device with the rest of the module) and serialize into ``state_dict``.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def fit(cls, x):
        dims = tuple(range(x.dim() - 1))  # reduce all but the last (feature) axis
        std = x.std(dim=dims)
        # constant features (e.g. traffic's b / power) have zero variance -> map them to 0
        # rather than dividing by zero (matches sklearn's StandardScaler).
        return cls(x.mean(dim=dims), torch.where(std > 0, std, torch.ones_like(std)))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


class GlobalStandardizer(nn.Module):
    """Global (isotropic) affine normalizer ``(x - mean) / std``, fit with a single scalar mean/std.

    Unlike ``Standardizer`` (per-feature mean/std), ``fit`` pools over **all** axes -- every edge *and*
    every sample -- to one scalar ``mean`` and one scalar ``std``. That isotropy is the point: a single
    scale preserves the field's cross-edge relative magnitude (and, with ``center=False``, its
    direction), whereas a per-edge scale would warp the field geometry the dynamics act on.
    ``center=False`` forces ``mean = 0`` -- zero-preserving, so the operator field's equilibrium
    (``F = 0``) is untouched; the field target uses this. ``center=True`` keeps the mean (for a generic
    target, or -- later -- global feature standardization, which this class is general enough to serve).

    Scale trade-off: ``std`` is the conventional unit-variance choice and reusable for features, but is
    inflated by heavy tails; a robust ``median|y|`` (what the old ``AsinhScaler`` used) would resist the
    operator's blow-up tail better. The PUME operator is only mildly heavy-tailed, so ``std`` is fine
    here. ``mean``/``std`` are registered buffers (they move with the module and serialize).
    """

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def fit(cls, x, center=True):
        mean = x.mean() if center else torch.zeros((), dtype=x.dtype)  # single scalar over all edges+samples
        std = x.std()
        return cls(mean, std if std > 0 else torch.ones_like(std))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, z):
        return z * self.std + self.mean


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


class Normalizer(nn.Module):
    """The fitted feats standardizer and target scaler (+ optional warp) the model trains/predicts through.

    ``input`` / ``target`` / ``target_warp`` are submodules, so a ``Normalizer`` owned by a model (see
    ``FieldModel``) moves to the model's device and serializes into ``state_dict`` automatically. The
    target round-trip is a two-stage composition: a fitted scale (``target``) and an optional stateless
    nonlinearity (``target_warp``, ``None`` = no warp), applied scale-then-warp and inverted
    warp-then-scale. Both stages must invert at inference, which is why the warp lives here rather than
    in the one-directional dataset ``transform``s.
    """

    def __init__(self, input, target, target_warp=None):
        super().__init__()
        self.input = input
        self.target = target
        self.target_warp = target_warp

    def transform_target(self, y):
        """Scale the real-unit target, then apply the optional warp -- the network's regression space."""
        z = self.target.transform(y)
        return self.target_warp.transform(z) if self.target_warp is not None else z

    def inverse_target(self, y):
        if self.target_warp is not None:
            y = self.target_warp.inverse_transform(y)
        return self.target.inverse_transform(y)


def _clone(item):
    """Copy a raw item so the stored original stays pristine across epochs (Data or dict)."""
    return item.clone() if hasattr(item, "clone") else dict(item)


def normalize_input(raw, transform, normalizer):
    """Featurize a raw input item and standardize its ``feats`` -- the one place that input shape lives.

    Clones the raw item (so a stored original stays pristine), applies the family ``transform`` (builds
    ``feats`` fresh), then standardizes ``feats``. This is the input half of ``_normalize_example``,
    factored out so every model-ready-input builder -- the datasets, ``FieldModel.conditioned_field``,
    and ``OnPolicyOperatorStream`` -- featurizes/standardizes identically. Representation-agnostic: it
    only touches the family ``transform`` seam and ``normalizer.input``, so it works for flat and graph
    games alike.
    """
    item = transform(_clone(raw))
    item["feats"] = normalizer.input.transform(item["feats"])
    return item


def _normalize_example(raw, target, transform, normalizer):
    """Featurize + standardize a raw ``(input item, target)`` pair -- input via ``normalize_input``,
    target clip-then-standardized. Shared by the map-style ``OperatorDataset`` and the streaming
    ``OperatorStream`` subclasses so both featurize/normalize identically.
    """
    return normalize_input(raw, transform, normalizer), normalizer.transform_target(target)


class OperatorDataset(Dataset):
    """Lazily featurize + normalize raw ``(input item, target)`` examples.

    ``__getitem__`` clones the raw item, applies the family's ``transform`` (builds ``feats`` fresh),
    then standardizes ``feats`` and the target -- so no featurized tensor is ever cached.
    """

    def __init__(self, examples, transform, normalizer):
        self.examples = examples
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        raw, target = self.examples[index]
        return _normalize_example(raw, target, self.transform, self.normalizer)


def _collate_examples(family_collate_fn, pairs):
    inputs, targets = zip(*pairs)
    return family_collate_fn(list(inputs)), default_collate(list(targets))


def collate_examples(family):
    """DataLoader ``collate_fn``: batch inputs via the family's seam, targets via ``default_collate``.

    Returns the ``(inputs, target)`` tuple ``FieldModel`` trains on. For flat games ``collate_fn``
    is ``default_collate``, so this reduces to stacking dicts; traffic overrides it with a dense
    graph stack. Returns a picklable ``functools.partial`` (not a closure) so the streaming train
    loader's workers can pickle it; ``family.collate_fn`` is a staticmethod, picklable by reference
    and free of the route-choice solver.
    """
    return functools.partial(_collate_examples, family.collate_fn)


class SolvedGroup(NamedTuple):
    """One instance's solved points: ``params`` plus ``points`` / ``targets`` / ``metrics``, each ``[K, d]``.

    The unit every *buffered* source retains (see ``caching.CachedOperatorStream`` and the expert stream
    in ``rollout_sampling``). Deliberately **not** a list of ``(model_input, target)`` examples:
    ``model_input`` clones the instance per point -- ~13 KB for a traffic graph -- so a buffer of examples
    costs ``K`` times what a group does, which for a million cached traffic points is 13 GB rather than
    1.3 GB. ``group_examples`` rebuilds the examples one at a time, at yield time, so the clone is
    transient and only the tensors are held.
    """

    params: Any
    points: torch.Tensor
    targets: torch.Tensor
    metrics: torch.Tensor


def solve_group(family, params, points):
    """Solve the operator jointly for one instance's ``points`` -> a ``SolvedGroup``.

    The single place a domain point is paired with its operator target: the operator (an expensive
    route-choice solve for traffic) runs **once for all points**, and the metric diagonal falls out of
    the same call, so it is free (see ``VariationalInequalityFamily.operator_and_metric``).
    """
    with torch.no_grad():
        targets, metrics = family.operator_and_metric(params, points)
    return SolvedGroup(params, points, targets, metrics)


def sample_group(family, params, n):
    """``solve_group`` at ``n`` freshly sampled domain points -- one instance's worth of training data."""
    return solve_group(family, params, family.sample_domain(params, n))


def group_examples(family, group, index=None, order=None):
    """Iterate a ``SolvedGroup`` into raw ``(model_input, target)`` examples, one per point.

    Each item is tagged with ``METRIC_DIAGONAL``: the per-coordinate diagonal that maps the operator's
    value back to the family's **raw** (unpreconditioned) field, i.e. ``metric_diagonal * target`` (all
    ones for families that are already raw). It rides along for the monotonicity constraint, which must
    be stated about the raw field -- the preconditioned one is not monotone. ``index``, when given, adds
    the ``INSTANCE_INDEX`` tag. Both are set by key access, so this works for a PyG ``Data`` (traffic)
    and a plain dict (flat games) alike.

    ``order`` picks which point comes out when: the cached sources reshuffle it per pass so the
    monotonicity pairs -- matched by halves within an instance -- vary without any new solves.
    """
    order = range(len(group.points)) if order is None else order
    for j in order:
        item = family.model_input(group.params, group.points[j])
        item[METRIC_DIAGONAL] = group.metrics[j]
        if index is not None:
            item[INSTANCE_INDEX] = torch.tensor(index)
        yield item, group.targets[j]


def examples_at_points(family, params, points, index=None):
    """A list of raw ``(model_input, target)`` examples for one instance at explicit ``points``.

    ``solve_group`` + ``group_examples`` for the sources that build their examples eagerly (the fixed
    splits, the fixed-instance stream and the on-policy collector, which must re-solve because it rolls
    out the *learned* field). Buffered sources hold the group instead and skip this.
    """
    return list(group_examples(family, solve_group(family, params, points), index=index))


def _solve_instance(family, params, points_per_instance):
    """The ``(raw input item, target)`` examples for one instance: sample points, solve the operator.

    Shared by the eager ``_examples_for_instances`` and the streaming ``UniformSampledOperatorStream``,
    both of which pass the full ``points_per_instance`` so one solve amortizes over that many points.
    """
    return list(group_examples(family, sample_group(family, params, points_per_instance)))


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
    datasets = tuple(OperatorDataset(split, family.transform, normalizer) for split in splits)
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
            yield _normalize_example(raw, target, transform, self.normalizer)


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


def split_instances(instances, counts):
    """Split a flat list of instances into disjoint sublists of sizes ``counts`` (reproducibly).

    A cache larger than ``sum(counts)`` is allowed -- the leftover is randomly held out and dropped.
    Reproducibility comes from the global RNG (seed via ``lightning.seed_everything`` at the call
    site); this function owns no seeding of its own.
    """
    counts = list(counts)
    remainder = len(instances) - sum(counts)
    assert remainder >= 0, f"need {sum(counts)} instances but the cache has only {len(instances)}"
    subsets = random_split(instances, counts + [remainder])
    return [[instances[i] for i in subset.indices] for subset in subsets[: len(counts)]]


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
    ``SolvedInstanceDataset``); this builds their ``(input, target)`` examples with the family's
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
    val_ds, test_ds = (OperatorDataset(split, family.transform, normalizer) for split in (val, test))
    # The calibration set (a fixed FieldDataset) doubles as the model-sizing sample source.
    cal_ds = OperatorDataset(cal, family.transform, normalizer)
    return (train_ds, val_ds, test_ds, cal_ds), normalizer


def solution_examples(family, instances):
    """Raw ``(parameters-only input, equilibrium z*)`` examples from cached solved instances.

    The full-amortization target (``--amortization full``): each instance's free-flow-time start fills
    the query column (``model_input`` -- no point that would leak the answer), regressed onto the
    cached ``equilibrium_cost`` ``z*`` (solved offline, see ``SolvedInstanceDataset``). Mirrors the
    ``solution_target=True`` path of ``rollout_sampling.ExpertOperatorStream`` for the fixed splits.
    """
    return [
        (family.model_input(inst, inst.free_flow_time), inst.equilibrium_cost.float()) for inst in instances
    ]


def build_streaming_solution_dataset(family_factory, cal_instances, val_instances, test_instances):
    """Fixed cal/val/test ``z*``-target ``FieldDataset``s plus the fitted ``Normalizer``.

    The solution-target sibling of ``build_streaming_operator_dataset``. The fixed splits use each
    instance's cached ``equilibrium_cost`` (exact and free), and the normalizer's target scaler is a
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
    val_ds, test_ds, cal_ds = (OperatorDataset(split, family.transform, normalizer) for split in (val, test, cal))
    return (val_ds, test_ds, cal_ds), normalizer
