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
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, default_collate

from l2s_games.transforms import NormClip


@dataclass(frozen=True)
class Standardizer:
    """Per-feature ``(x - mean) / std`` map, fit from data, with its inverse."""

    mean: Tensor
    std: Tensor

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


@dataclass(frozen=True)
class Normalizer:
    """The fitted feats/target standardizers the model trains and predicts through."""

    input: Standardizer
    target: Standardizer
    clip: NormClip | None = None

    def clip_field(self, y):
        """Norm-clip a **real-unit** field (identity when clipping is disabled)."""
        return y if self.clip is None else self.clip(y)

    def transform_target(self, y):
        """Clip the real-unit target's norm, then standardize it.

        The traffic operator is heavy-tailed: at low costs (near the free-flow-time floor) BPR's
        power term makes the residual blow up to ~100× the typical magnitude, so a handful of
        outliers dominate an MSE fit. Clipping the field's *norm* (see ``NormClip``) saturates those
        blow-ups without rotating the field, so the model learns the operator's true direction; the
        equilibrium ``F = 0`` is untouched. The clip is applied in real units *before* standardizing
        so it is a scalar scaling of the field -- de-standardizing at inference recovers that same
        direction.
        """
        return self.target.transform(self.clip_field(y))

    def inverse_target(self, y):
        return self.target.inverse_transform(y)


def _clone(item):
    """Copy a raw item so the stored original stays pristine across epochs (Data or dict)."""
    return item.clone() if hasattr(item, "clone") else dict(item)


def _normalize_example(raw, target, transform, normalizer):
    """Featurize a raw item and standardize its feats + target -- the one place that shape lives.

    Clones the raw item, applies the family ``transform`` (builds ``feats`` fresh), then standardizes
    ``feats`` and clip-then-standardizes the target. Shared by the map-style ``FieldDataset`` and the
    streaming ``StreamingFieldDataset`` so both featurize/normalize identically.
    """
    item = transform(_clone(raw))
    item["feats"] = normalizer.input.transform(item["feats"])
    return item, normalizer.transform_target(target)


class FieldDataset(Dataset):
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


def _solve_instance(family, params, points_per_instance):
    """The ``(raw input item, target)`` examples for one instance: sample points, solve the operator.

    The operator (an expensive route-choice solve for traffic) is run **once, jointly for all points**
    of the instance, then sliced per point. Shared by the eager ``_examples_for_instances`` and the
    streaming generator (which calls it with ``points_per_instance=1``).
    """
    points = family.sample_domain(params, points_per_instance)
    with torch.no_grad():
        targets = family.operator(params, points)
    return [(family.model_input(params, points[j]), targets[j]) for j in range(len(points))]


def _examples_for_instances(family, instances, points_per_instance):
    """A flat list of ``(raw input item, target)`` examples over instances."""
    return [example for params in instances for example in _solve_instance(family, params, points_per_instance)]


def _fit_normalizer(family, examples, target_clip):
    """Fit feats/target standardizers on the (transformed) train examples.

    The target standardizer is fit on the **clipped** targets -- the same tensor the model regresses
    (``transform_target`` standardizes ``clip(y)``), so the fitted std matches the target's true
    dynamic range. Fitting on the unclipped targets instead lets the heavy BPR tail inflate std, so
    the clipped target divides down to ~0 and the model gets no signal.
    """
    transform = family.transform
    feats = torch.stack([transform(_clone(raw))["feats"] for raw, _ in examples])
    targets = torch.stack([target for _, target in examples])
    clip = NormClip(target_clip) if target_clip else None
    clipped_targets = clip(targets) if clip else targets
    return Normalizer(Standardizer.fit(feats), Standardizer.fit(clipped_targets), clip)


def build_dataset(family, n_train, n_val, n_test, points_per_instance, target_clip=None):
    """Train/val/test ``FieldDataset``s plus the fitted ``Normalizer``.

    The normalizer is fit on the train split and shared with all three, so val/test contribute no
    statistics. Each split draws its own instances, so the test split measures generalization to
    unseen parametrizations. Pair with ``collate_examples(family)`` when building the DataLoaders.
    """
    splits = [
        _examples_for_instances(family, [family.sample_params() for _ in range(n)], points_per_instance)
        for n in (n_train, n_val, n_test)
    ]
    normalizer = _fit_normalizer(family, splits[0], target_clip)
    datasets = tuple(FieldDataset(split, family.transform, normalizer) for split in splits)
    return datasets, normalizer


class StreamingFieldDataset(IterableDataset):
    """Infinite stream of freshly-sampled instances: one fresh instance -> one normalized example.

    Each step samples a new instance, one domain point, solves the operator for the target, and yields
    the same normalized ``(item, target)`` a ``FieldDataset`` would -- so every example is a distinct
    parametrization and minibatches are maximally diverse. Holds a picklable ``family_factory`` (not a
    live family) and builds the family -- hence its route-choice solver -- **lazily inside each worker
    process** on first iteration, so nothing solver-related is pickled across the worker boundary.
    Reproducible per-worker streams come from ``lightning.seed_everything(seed, workers=True)`` at the
    call site (the Trainer installs the per-worker seeding); this dataset owns no seeding of its own.
    """

    def __init__(self, family_factory, normalizer):
        self.family_factory = family_factory
        self.normalizer = normalizer

    def __iter__(self):
        # Built once per __iter__ (~once per worker per epoch), not per sample; the per-epoch rebuild
        # is cheap relative to a full epoch of solves.
        family = self.family_factory()
        transform = family.transform
        while True:
            (raw, target), = _solve_instance(family, family.sample_params(), 1)
            yield _normalize_example(raw, target, transform, self.normalizer)


def build_streaming_dataset(family_factory, n_bootstrap, n_val, n_test, points_per_instance, target_clip=None):
    """A streaming train dataset plus fixed val/test ``FieldDataset``s and the fitted ``Normalizer``.

    The normalizer (+clip) is fit once on a fixed **bootstrap** set of freshly-solved instances, then
    frozen and shared with the stream and the fixed val/test splits -- preserving the fit-on-a-fixed-
    sample invariant while training draws unbounded fresh instances. Val/test stay pre-solved so their
    metrics are stable across epochs. Pair with ``collate_examples(family)`` for the DataLoaders.
    """
    family = family_factory()
    bootstrap, val, test = (
        _examples_for_instances(family, [family.sample_params() for _ in range(n)], points_per_instance)
        for n in (n_bootstrap, n_val, n_test)
    )
    normalizer = _fit_normalizer(family, bootstrap, target_clip)
    train_ds = StreamingFieldDataset(family_factory, normalizer)
    val_ds, test_ds = (FieldDataset(split, family.transform, normalizer) for split in (val, test))
    # The bootstrap set (a fixed FieldDataset) doubles as the model-sizing sample source.
    bootstrap_ds = FieldDataset(bootstrap, family.transform, normalizer)
    return (train_ds, val_ds, test_ds, bootstrap_ds), normalizer
