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

The constraint is stated **per pair**, not per instance: the violation tensor holds one entry for every
matched pair in the batch. Under ``QuadraticPenalty`` that is the natural granularity, because cooper's
penalty already sums over the violation dimension (``0.5 * mu * sum_p relu(v_p)^2``), and that sum is
zero *iff every pair is monotone* -- so nothing is averaged away and no aggregation code is needed. This
is also the granularity the eventual ``ImplicitMultiplier`` wants, since it parametrises one multiplier
per pair from that pair's features.

Two consequences to keep in mind. ``mu``'s effective scale tracks the number of pairs in a batch, since a
sum grows with count -- it is not comparable across ``--batch_fixed`` or ``--points_per_instance``
changes. And the squared hinge's gradient, ``2 * relu(v)``, vanishes as a pair approaches feasibility, so
the last sliver of violation is only weakly pushed; that is inherent to the quadratic penalty rather than
a tuning problem.
"""

import cooper
import torch

from l2s_games.data import INSTANCE_INDEX, METRIC_DIAGONAL


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


class MonotonicityCMP(cooper.ConstrainedMinimizationProblem):
    """Regression loss subject to monotonicity of the model's raw field, one constraint entry per pair.

    One inequality constraint whose violation tensor holds **one entry per matched pair** in the batch, so
    nothing is averaged: ``QuadraticPenalty`` sums the squared hinges, and that sum vanishes only when
    every pair is monotone. No multiplier, so ``compute_primal_lagrangian()`` returns an ordinary
    differentiable loss and the training loop needs no dual optimizer; swapping ``formulation_type`` (plus
    a multiplier) is the upgrade path to a Lagrangian, and a per-pair violation is already the granularity
    an ``ImplicitMultiplier`` needs.

    ``constrain_raw=False`` states the constraint about the preconditioned field the model literally
    predicts, which the operator itself violates -- kept only so the two are comparable.
    """

    def __init__(
        self,
        model,
        family,
        tolerance,
        penalty_mu,
        normalize=True,
        constrain_raw=True,
    ):
        super().__init__()
        self.model = model
        self.family = family
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

    def pair_violations(self, values, inputs):
        """Per-pair violations for every instance in the batch, concatenated.

        Pairs are formed **within** an instance -- monotonicity is a property of one operator, so a
        cross-instance pair is meaningless -- by grouping on ``instance_index``. An instance contributing a
        single point carries no pair and is skipped; with the fixed stream each contributes
        ``points_per_instance``, so that is an edge case at batch boundaries rather than the norm.
        """
        field = self.raw_field(values, inputs)
        points = self.family.initial_point(inputs)  # the raw domain point each example carries
        instance_index = inputs[INSTANCE_INDEX]
        groups = [instance_index == index for index in instance_index.unique()]
        return torch.cat(
            [
                monotonicity_violations(field[rows], points[rows], self.normalize)
                for rows in groups
                if rows.sum() >= 2
            ]
        )

    def compute_cmp_state(self, batch):
        """The regression loss plus one monotonicity violation per pair, over every train source.

        ``batch`` is the ``{source: (inputs, targets)}`` mapping the model trains on, so predictions and
        constraints are built per source and concatenated -- the constraint needs each source's own
        collated inputs (its metric, points and instance tags), which a merged prediction tensor loses.
        """
        predictions, targets, violations, true_violations = [], [], [], []
        for inputs, source_targets in batch.values():
            prediction = self.model(inputs)
            predictions.append(prediction)
            targets.append(source_targets)
            violations.append(self.pair_violations(prediction, inputs))
            # The identical computation on the *targets*: in raw space this must be ~0, which is a
            # correctness gate on the metric wiring, and it floors what the model can be asked to reach.
            with torch.no_grad():
                true_violations.append(self.pair_violations(source_targets, inputs))

        prediction, target = torch.cat(predictions), torch.cat(targets)
        pair_violations = torch.cat(violations)
        return cooper.CMPState(
            loss=self.model.regression_loss(prediction, target),
            # QuadraticPenalty has no dual variable, so the constraint must declare it contributes
            # nothing to a dual update -- cooper's own sanity check rejects the pairing otherwise.
            observed_constraints={
                self.monotone: cooper.ConstraintState(
                    violation=pair_violations - self.tolerance, contributes_to_dual_update=False
                )
            },
            misc={
                "prediction": prediction,
                "targets": target,
                "pair_violations": pair_violations,
                "true_pair_violations": torch.cat(true_violations),
            },
        )
