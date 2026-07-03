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

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset, default_collate

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
        item = self.transform(_clone(raw))
        item["feats"] = self.normalizer.input.transform(item["feats"])
        return item, self.normalizer.transform_target(target)


def collate_examples(family):
    """DataLoader ``collate_fn``: batch inputs via the family's seam, targets via ``default_collate``.

    Returns the ``(inputs, target)`` tuple ``FieldModel`` trains on. For flat games ``collate_fn``
    is ``default_collate``, so this reduces to stacking dicts; traffic overrides it with a dense
    graph stack.
    """

    def collate(pairs):
        inputs, targets = zip(*pairs)
        return family.collate_fn(list(inputs)), default_collate(list(targets))

    return collate


def _examples_for_instances(family, instances, points_per_instance):
    """A list of ``(raw input item, target)`` examples over instances.

    Targets are the operator values -- computed **once** here (an expensive route-choice solve for
    traffic), never in the lazy transform.
    """
    examples = []
    for params in instances:
        points = family.sample_domain(params, points_per_instance)
        with torch.no_grad():
            targets = family.operator(params, points)
        examples.extend((family.model_input(params, points[j]), targets[j]) for j in range(len(points)))
    return examples


def _fit_normalizer(family, examples, target_clip):
    """Fit feats/target standardizers on the (transformed) train examples.

    Standardizers are fit on the **unclipped** targets; ``target_clip`` bounds the real-unit field
    norm and is applied at transform time (see ``NormClip`` / ``Normalizer.clip_field``).
    """
    transform = family.transform
    feats = torch.stack([transform(_clone(raw))["feats"] for raw, _ in examples])
    targets = torch.stack([target for _, target in examples])
    clip = NormClip(target_clip) if target_clip else None
    return Normalizer(Standardizer.fit(feats), Standardizer.fit(targets), clip)


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
