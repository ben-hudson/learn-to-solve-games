import torch

from .base import FieldModel
from .graphormer import GraphormerFieldModel
from .mlp import MLPFieldModel

__all__ = [
    "FieldModel",
    "MLPFieldModel",
    "GraphormerFieldModel",
    "conditioned_field",
]


def conditioned_field(model, family, params, normalizer):
    """Return a plain field callable ``v(z)`` for a fixed instance, in real units.

    Runs the family's conditioning seams: ``model_input`` -> ``transform`` (builds ``feats``) ->
    ``collate_fn`` (a batch of one), then de-standardizes the prediction. Works for any
    representation (flat dict or graph ``Data``), so a trained field model is a drop-in field for
    ``simulate`` / plotting. It is grad-transparent (``torch.func.jacrev`` works -- ``feats`` and
    normalization are differentiable in ``z``, structure is not a function of ``z``); value-only
    callers handle their own ``no_grad`` / ``detach``.
    """

    def v(z):
        z = torch.as_tensor(z, dtype=torch.float32)
        item = family.transform(family.model_input(params, z))
        item["feats"] = normalizer.input.transform(item["feats"])
        prediction = model(family.collate_fn([item])).squeeze(0)
        # No norm clip here: the field must stay jacrev-transparent for Consensus / quiver, but
        # NormClip (a BaseTransform) copy-copies its input and breaks jacrev. Clipping the training
        # target already teaches the model the reduced-range, correctly-directed field.
        return normalizer.target.inverse_transform(prediction)

    return v
