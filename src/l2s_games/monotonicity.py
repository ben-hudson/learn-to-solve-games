"""Monotonicity as a constraint on the learned field.

A VI operator is **monotone** iff ``<F(x) - F(y), x - y> >= 0`` for every pair of feasible points, and
that -- not regression accuracy -- is what makes the rollout algorithms in ``algorithms.py`` converge on
it. A field model fit purely by MSE can be accurate and still non-monotone, hence unrollable, so this
module states monotonicity as a constrained-optimization problem over ``cooper`` and lets the training
loop enforce it.

Three things make it nearly free, and all three are properties of the existing pipeline:

- **The pairs are already in the batch.** ``FixedInstanceOperatorStream`` emits ``points_per_instance``
  examples per instance, each tagged ``instance_index``, so a batch contains many same-instance points
  whose predictions are already computed. Grouping by that tag yields pairs with no extra sampling, no
  extra operator solves, and no extra model forwards.
- **The constraint is stated about the *raw* field.** The model regresses the *preconditioned* operator
  ``M^-1 E``, which is measurably non-monotone (a positive per-coordinate reweighting does not preserve
  the sign of ``sum_i dF_i dx_i``, and ``M`` varies with the point). Constraining that would fight the
  target. So predictions are mapped back with the ``metric_diagonal`` each example carries -- see
  ``data.examples_at_points`` -- giving ``E_hat = M * F_hat``, where monotonicity does hold of the data.
- **The ground truth is checkable for free.** The same computation run on the batch's *targets* must
  come out ~0 in the raw space, which is a correctness gate on this whole path rather than a diagnostic.

The aggregation over an instance's pairs is a **soft maximum** (temperature ``tau``), i.e. approximately
the worst pair rather than the mean: the requirement is monotonicity *over the training domain*, and a
mean would let a single bad direction hide behind the satisfied ones. That is only legitimate
because the ``QuadraticPenalty`` formulation used here has **no dual variable**. A sampled ``max`` is a
biased estimator of a sup, so with a multiplier there would be nothing well-defined for it to converge
to; a later switch to ``Lagrangian`` must revisit the aggregation together with the multiplier class.
"""

import cooper
import torch

from l2s_games.data import METRIC_DIAGONAL
from l2s_games.instance_sampling import INSTANCE_INDEX


def monotonicity_ratios(field, points, normalize=True):
    """Per-pair monotonicity ratios under a **one-to-one matching** of one instance's points.

    ``field``/``points`` are ``[P, E]`` and are matched by halves -- point ``i`` with point ``i + P // 2``
    -- giving ``P // 2`` disjoint pairs (a trailing point is dropped when ``P`` is odd). Since the stream
    draws an instance's points i.i.d. from the domain, every matching is statistically equivalent, so
    halving is arbitrary but not arbitrary-in-a-way-that-matters. Each entry is

        ``(F_i - F_j)^T (x_i - x_j) / ||x_i - x_j||^2``   (``normalize=True``, the default)

    which is the **Rayleigh quotient of the secant map** -- for a linear field ``F = Ax`` it is exactly
    ``dx^T A dx / dx^T dx``, the Rayleigh quotient of ``A``'s symmetric part along ``dx``. Note this is
    *not* the "modulus": a field is strongly monotone *with modulus mu* when the inequality holds with
    ``mu ||dx||^2`` for **all** pairs, so the modulus is an infimum over pairs -- a property of the
    operator, not of a pair. The per-pair value is the ratio; the modulus is what an aggregate estimates.

    Dividing by ``||dx||^2`` is free in the sense that matters: the divisor is a positive scalar that does
    not depend on the model, so it changes neither the sign of any entry nor the feasible set, and leaves
    the gradient a per-pair reweighting. (Dividing per *coordinate* would not be free -- that is exactly
    the reweighting that destroys monotonicity in the first place.) What it buys is scale independence:
    the unnormalized inner product grows like ``||dx||^2``, so its tolerance would have to be recalibrated
    per pair separation and per instance. ``normalize=False`` keeps the raw inner product for comparison.

    Differencing **before** contracting is numerically load-bearing, not stylistic. The algebraically
    equivalent gram form ``(a_i - a_j)^T(b_i - b_j) = d_i + d_j - K_ij - K_ji`` computes a small result as
    a difference of large ones, and the raw (unpreconditioned) field is *stiff* -- values reach ~1e12 near
    free-flow time, which is exactly why the operator gets preconditioned. In float32 that cancellation
    destroys the sign for close pairs, reporting ~1e9 violations for a field that is in fact monotone.
    """
    half = points.shape[0] // 2
    delta_field = field[:half] - field[half : 2 * half]
    delta_points = points[:half] - points[half : 2 * half]
    inner = (delta_field * delta_points).sum(dim=-1)
    if not normalize:
        return inner
    # Two independent domain draws are never coincident; the floor only guards an exact duplicate.
    return inner / (delta_points * delta_points).sum(dim=-1).clamp(min=1e-12)


def monotonicity_violations(field, points, normalize=True):
    """``relu(-ratio)`` per pair: zero where the field is monotone, the depth of the failure where not."""
    return torch.relu(-monotonicity_ratios(field, points, normalize))


def soft_max(values, temperature):
    """Softmax-weighted mean of ``values`` -- the worst entry as ``temperature -> 0``, the mean as it grows.

    A differentiable stand-in for ``max`` that spreads gradient over the worst few entries instead of
    exactly one, which keeps the signal from being a single-sample statistic.
    """
    return torch.sum(torch.softmax(values / temperature, dim=0) * values)


class MonotonicityCMP(cooper.ConstrainedMinimizationProblem):
    """Regression loss subject to per-instance monotonicity of the model's raw field.

    One inequality constraint whose violation tensor holds **one entry per instance** present in the
    batch, so ``cooper`` sees a per-instance statement rather than one batch-wide average. The
    formulation is ``QuadraticPenalty`` -- no multiplier -- so ``compute_primal_lagrangian()`` returns an
    ordinary differentiable loss and the training loop needs no dual optimizer; swapping
    ``formulation_type`` (plus a multiplier) is the upgrade path to a Lagrangian.

    ``constrain_raw=False`` states the constraint about the preconditioned field the model literally
    predicts, which the operator itself violates -- kept only so the two are comparable.
    """

    def __init__(
        self,
        model,
        family,
        temperature,
        tolerance,
        penalty_mu,
        normalize=True,
        constrain_raw=True,
    ):
        super().__init__()
        self.model = model
        self.family = family
        self.temperature = temperature
        self.tolerance = tolerance
        self.normalize = normalize
        self.constrain_raw = constrain_raw
        self.monotone = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.QuadraticPenalty,
            penalty_coefficient=cooper.penalty_coefficients.DensePenaltyCoefficient(torch.tensor(penalty_mu)),
        )

    def raw_field(self, values, inputs):
        """Map target-space ``values`` (a prediction or a target) to the field the constraint is about.

        De-standardizes into real units, then applies the example's ``metric_diagonal`` to undo the
        operator's preconditioning. Real units rather than target space so the ratio -- and hence the
        tolerance -- is comparable to a ground-truth measurement; with no target warp the inverse is a
        single multiply, so this is nearly free and stays differentiable.
        """
        field = self.model.inverse_target(values)
        return field * inputs[METRIC_DIAGONAL] if self.constrain_raw else field

    def _per_instance(self, values, inputs):
        """Per-instance soft-max violations plus the pooled per-pair violations, grouped by instance.

        Instances contributing a single point to the batch carry no pair and are skipped; with the fixed
        stream every instance contributes ``points_per_instance`` of them, so this is an edge case at
        batch boundaries, not the norm.
        """
        field = self.raw_field(values, inputs)
        points = self.family.initial_point(inputs)  # the raw domain point each example carries
        instance_index = inputs[INSTANCE_INDEX]
        aggregates, pooled = [], []
        for index in instance_index.unique():
            rows = instance_index == index
            if rows.sum() < 2:
                continue
            violations = monotonicity_violations(field[rows], points[rows], self.normalize)
            aggregates.append(soft_max(violations, self.temperature))
            pooled.append(violations)
        return torch.stack(aggregates), torch.cat(pooled)

    def compute_cmp_state(self, batch):
        """The regression loss plus one monotonicity violation per instance, over every train source.

        ``batch`` is the ``{source: (inputs, targets)}`` mapping the model trains on, so predictions and
        constraints are built per source and concatenated -- the constraint needs each source's own
        collated inputs (its metric, points and instance tags), which a merged prediction tensor loses.
        """
        predictions, targets, aggregates, pooled, true_pooled = [], [], [], [], []
        for inputs, source_targets in batch.values():
            prediction = self.model(inputs)
            predictions.append(prediction)
            targets.append(source_targets)
            instance_violations, pairs = self._per_instance(prediction, inputs)
            aggregates.append(instance_violations)
            pooled.append(pairs)
            # The identical computation on the *targets*: in raw space this must be ~0, which is a
            # correctness gate on the metric wiring, and it floors what the model can be asked to reach.
            with torch.no_grad():
                true_pooled.append(self._per_instance(source_targets, inputs)[1])

        prediction, target = torch.cat(predictions), torch.cat(targets)
        violation = torch.cat(aggregates) - self.tolerance
        return cooper.CMPState(
            loss=self.model.regression_loss(prediction, target),
            # QuadraticPenalty has no dual variable, so the constraint must declare it contributes
            # nothing to a dual update -- cooper's own sanity check rejects the pairing otherwise.
            observed_constraints={self.monotone: cooper.ConstraintState(violation=violation, contributes_to_dual_update=False)},
            misc={
                "prediction": prediction,
                "targets": target,
                "pair_violations": torch.cat(pooled),
                "true_pair_violations": torch.cat(true_pooled),
            },
        )
