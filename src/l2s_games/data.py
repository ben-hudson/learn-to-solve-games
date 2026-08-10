"""The vocabulary every training source shares: what an example *is*, and how it is normalized.

An *instance* is one ``params`` value drawn from the family; a *sample* is a point in that instance's
domain. A training example is ``(family.model_input(params, point), operator value)``, where ``model_input``
returns the **raw** input item (``{point, params}`` for flat games, a PyG ``Data`` with ``.cost`` for
traffic). ``operator_examples`` builds those pairs from an ``OperatorEvaluations`` -- one instance's points
and the operator's value at each -- which is the unit both the cached and the streaming sources hold.

Featurization is deliberately *not* here: the family's ``transform`` builds ``feats`` (and any structure)
lazily, per access, so nothing featurized is ever cached; it lives in ``transforms.py``.

Normalization is **not** a transform either (it fits on train and inverts at inference), so it lives here as
``Standardizer`` / ``GlobalStandardizer`` / ``Normalizer``, applied through key access (``item["feats"]``
works for a dict and a ``Data`` alike). Fit on the train split only, so val/test contribute no statistics of
their own; constant features (traffic's ``b`` / ``power``) map to 0 rather than dividing by zero.
``fit_normalizer`` is the entry point for a cached source, which can stream its feats
(``Standardizer.fit_iter``) and read every target for free; the eager, materialized-list variant belongs to
the streaming path and lives in ``streaming`` with it.

Reproducibility is via ``lightning.seed_everything`` at the call site -- nothing here seeds.
"""

import functools
from typing import Any, NamedTuple

import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, default_collate, random_split

# Item key for the per-coordinate diagonal mapping the operator target back to the raw field; set by
# operator_examples. Lives here (not in monotonicity.py, its consumer) so the data layer owns its own
# schema and nothing in the pipeline imports the constraint code.
PRECONDITIONER_DIAGONAL = "preconditioner_diagonal"
# Item key for the index of the example's instance within its source's instance set -- stable across
# epochs, and what the monotonicity constraint groups same-instance points by. Diagnostics-only for the
# model (the backbones read only feats + structure); it rides along on the collated batch. Lives here
# with PRECONDITIONER_DIAGONAL for the same reason -- and because operator_examples sets it, so a home in
# a module that imports this one would be a cycle.
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

    @classmethod
    def fit_iter(cls, chunks):
        """``fit`` for a population too large to stack: incremental over an iterable of ``[..., k]`` tensors.

        The cached operator datasets fit over every train example, whose stacked ``feats`` would be
        ``n_examples x n_edges x k`` floats -- gigabytes on a wide root -- so the caller passes a generator
        and nothing is materialized. Delegates the accumulation to sklearn's ``StandardScaler.partial_fit``,
        whose ``_incremental_mean_and_var`` is numerically careful where a hand-rolled sum-of-squares is not,
        and whose ``scale_`` already maps zero-variance features to 1 -- the convention ``fit`` above was
        written to match. Differs from ``fit`` only in using the population std (sklearn's ddof=0) rather than
        torch's unbiased default, a ``sqrt(n / (n - 1))`` factor.

        One ``partial_fit`` per chunk, unbatched: its validation costs tens of microseconds against the
        ~0.7 ms the caller already spends featurizing each example, so batching would buy nothing.
        """
        scaler = StandardScaler()
        for chunk in chunks:
            scaler.partial_fit(chunk.reshape(-1, chunk.shape[-1]).numpy())
        as_float32 = functools.partial(torch.as_tensor, dtype=torch.float32)
        return cls(as_float32(scaler.mean_), as_float32(scaler.scale_))

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
    ``feats`` fresh), then standardizes ``feats``. This is the input half of ``normalize_example``,
    factored out so every model-ready-input builder -- the datasets, ``FieldModel.conditioned_field``,
    and the streaming sources -- featurizes/standardizes identically. Representation-agnostic: it
    only touches the family ``transform`` seam and ``normalizer.input``, so it works for flat and graph
    games alike.
    """
    item = transform(_clone(raw))
    item["feats"] = normalizer.input.transform(item["feats"])
    return item


def normalize_example(raw, target, transform, normalizer):
    """Featurize + standardize a raw ``(input item, target)`` pair -- input via ``normalize_input``,
    target clip-then-standardized. Shared by ``operator_datasets.OperatorDataset`` (through
    ``collate_normalized_examples``) and by every ``streaming.OperatorStream`` subclass, so each source
    featurizes/normalizes identically however its examples are stored.
    """
    return normalize_input(raw, transform, normalizer), normalizer.transform_target(target)



def _collate_examples(family_collate_fn, pairs):
    inputs, targets = zip(*pairs)
    return family_collate_fn(list(inputs)), default_collate(list(targets))



def _collate_normalized_examples(family_collate_fn, transform, normalizer, pairs):
    normalized = [normalize_example(raw, target, transform, normalizer) for raw, target in pairs]
    return _collate_examples(family_collate_fn, normalized)


def collate_normalized_examples(family, normalizer):
    """``collate_examples`` for sources that serve **raw** examples: featurize + standardize first.

    The streams must normalize at yield time -- their examples are transient -- but a cached source
    (``operator_datasets.OperatorDataset``) serves real-unit examples and stays free of experiment state,
    so its normalizer enters here, at the DataLoader boundary, where the fit-on-train invariant is visible
    at the call site. Goes through ``normalize_example``, so every source still featurizes/normalizes
    identically. Picklable like ``collate_examples``: ``family.transform`` and ``family.collate_fn`` are
    solver-free, and the normalizer is plain tensors.
    """
    return functools.partial(_collate_normalized_examples, family.collate_fn, family.transform, normalizer)


class OperatorEvaluations(NamedTuple):
    """One instance's evaluated points: ``params`` plus ``points`` / ``targets`` /
    ``preconditioner_diagonal``, each ``[points_per_instance, d]``.

    An *evaluation* is one operator call; contrast ``datasets.EquilibriumDataset``, where "solved"
    means solved to equilibrium -- hundreds of these. The unit every *buffered* source retains (see
    ``caching.CachedOperatorStream`` and the expert stream in ``rollout_sampling``, both streaming) and the unit
    ``operator_datasets`` persists. Deliberately **not** a list of ``(model_input, target)`` examples:
    ``model_input`` clones the instance per point -- ~13 KB for a traffic graph -- so a buffer of examples
    costs ``points_per_instance`` times what this does, which for a million cached traffic points is
    13 GB rather than 1.3 GB. ``operator_examples`` rebuilds the examples one at a time, at yield time,
    so the clone is transient and only the tensors are held.
    """

    params: Any
    points: torch.Tensor
    targets: torch.Tensor
    preconditioner_diagonal: torch.Tensor


def eval_operator(family, params, points):
    """Evaluate the operator jointly for one instance's ``points`` -> an ``OperatorEvaluations``.

    The single place a domain point is paired with its operator target: the operator (an expensive
    route-choice solve for traffic) runs **once for all points**, and the preconditioner diagonal falls
    out of the same call, so it is free (see ``VariationalInequalityFamily.operator_and_preconditioner``).
    """
    with torch.no_grad():
        targets, preconditioner_diagonal = family.operator_and_preconditioner(params, points)
    return OperatorEvaluations(params, points, targets, preconditioner_diagonal)


def sample_and_eval_operator(family, params, n):
    """``eval_operator`` at ``n`` freshly sampled domain points -- one instance's worth of training data."""
    return eval_operator(family, params, family.sample_domain(params, n))


def operator_examples(family, evaluations, index=None, order=None):
    """Iterate an ``OperatorEvaluations`` into raw ``(model_input, target)`` examples, one per point.

    Each item is tagged with ``PRECONDITIONER_DIAGONAL``: the per-coordinate diagonal that maps the
    operator's value back to the family's **raw** (unpreconditioned) field, i.e.
    ``preconditioner_diagonal * target`` (all ones for families that are already raw). It rides along for
    the monotonicity constraint, which must be stated about the raw field -- the preconditioned one is
    not monotone. ``index``, when given, adds the ``INSTANCE_INDEX`` tag. Both are set by key access, so
    this works for a PyG ``Data`` (traffic) and a plain dict (flat games) alike.

    ``order`` picks which point comes out when: the cached sources reshuffle it per pass so the
    monotonicity pairs -- matched by halves within an instance -- vary without any new evaluations.
    """
    order = range(len(evaluations.points)) if order is None else order
    for j in order:
        item = family.model_input(evaluations.params, evaluations.points[j])
        item[PRECONDITIONER_DIAGONAL] = evaluations.preconditioner_diagonal[j]
        if index is not None:
            item[INSTANCE_INDEX] = torch.tensor(index)
        yield item, evaluations.targets[j]



def fit_normalizer(feats, targets, target_scaler=functools.partial(GlobalStandardizer.fit, center=False)):
    """The ``Normalizer`` a model trains through, from the two populations given.

    For the **cached** sources, whose two halves want different sample sets: ``targets`` are stored raw so
    reading every one is a single ``torch.cat``, while ``feats`` have to be *built* per example, so they arrive
    as an iterable and are streamed (see ``Standardizer.fit_iter``). Taking both from the caller is what keeps
    the fit-on-train choice visible at the call site rather than implied by a dataset's argument.

    ``target_scaler`` defaults to the operator field's zero-preserving global scale; the solution baseline
    passes ``Standardizer.fit``, treating ``z*`` as a generic per-feature target (no global scale, no warp).
    Supersedes ``_fit_normalizer`` -- which fits from a materialized example list, and retires with the
    streaming sources that need that shape.
    """
    return Normalizer(Standardizer.fit_iter(feats), target_scaler(targets))




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



def solution_examples(family, instances):
    """Raw ``(parameters-only input, equilibrium z*)`` examples from cached solved instances.

    The full-amortization target (``--amortization full``): each instance's free-flow-time start fills
    the query column (``model_input`` -- no point that would leak the answer), regressed onto the
    cached ``equilibrium`` ``z*`` (solved offline, see ``EquilibriumDataset``). Mirrors the
    ``solution_target=True`` path of ``rollout_sampling.ExpertOperatorStream`` for the fixed splits.
    """
    return [(family.model_input(inst, inst.free_flow_time), inst.equilibrium.float()) for inst in instances]


