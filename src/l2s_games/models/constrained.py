"""Field training under a monotonicity constraint.

``ConstrainedFieldModel`` is a ``FieldModel`` whose training objective is a ``cooper`` constrained
minimization problem (``monotonicity.MonotonicityCMP``) instead of the plain regression loss: fit the
operator field, subject to the model's *raw* field being monotone on the training domain.

The whole point of starting from the ``QuadraticPenalty`` formulation is that it has **no dual
variable**, so the primal Lagrangian is just a differentiable scalar and ``training_step`` returns it
like any other loss -- Lightning's automatic optimization, the warmup/cosine schedule, and gradient
clipping all keep working untouched. Introducing a multiplier later is what forces manual optimization
(and with it: registering the multiplier as a submodule so it reaches the device and the checkpoint,
excluding it from the primal optimizer, and handing ``cooper`` the raw torch optimizers).
"""

import cooper
import torch
from torch.nn import functional as F

from l2s_games.models.base import FieldModel
from l2s_games.monotonicity import MonotonicityCMP


class ConstrainedFieldModel(FieldModel):
    """A ``FieldModel`` trained on ``loss + penalty * monotonicity violation``.

    ``penalty_mu`` is the initial penalty coefficient and is raised multiplicatively whenever an
    instance's violation exceeds ``tolerance`` (``MultiplicativePenaltyCoefficientUpdater``), so the
    exchange rate against the regression loss is found by a feasibility-driven schedule rather than
    hand-tuned. Note that updater's ``has_restart`` default resets the coefficient to ``penalty_mu`` on
    any fully-feasible step, so it hovers near its initial value rather than escalating; pass
    ``has_restart=False`` if a ratchet is wanted.

    ``penalty_mu=0`` measures the constraint without acting on it: the penalty term is exactly zero and
    the updater keeps a zero coefficient at zero, so the objective is *mathematically* identical to
    unconstrained while every monotonicity metric is still logged. It is not bit-identical -- adding a
    structurally-zero term changes gradient accumulation order, and the training trajectory is chaotic
    enough to amplify that over epochs (measured: the two agree to 4 significant figures at epoch 4 and
    reach the same loss by epoch 19, but diverge mid-run).
    """

    def __init__(
        self,
        net,
        lr,
        family,
        tolerance=1e-3,
        penalty_mu=1.0,
        penalty_growth=1.01,
        normalize=True,
        constrain_raw=True,
        **kwargs,
    ):
        super().__init__(net, lr, **kwargs)
        self.cmp = MonotonicityCMP(
            model=self,
            family=family,
            tolerance=tolerance,
            penalty_mu=penalty_mu,
            normalize=normalize,
            constrain_raw=constrain_raw,
        )
        self.penalty_updater = cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater(
            growth_factor=penalty_growth, violation_tolerance=tolerance
        )

    def on_fit_start(self):
        # The penalty coefficient is a plain tensor holder, not a Parameter or buffer, so Lightning's
        # device move skips it and the penalty term would try to combine a CPU coefficient with an
        # on-device violation. cooper's CMP.to() moves the constraint's coefficients (and, once there is
        # one, its multipliers) explicitly.
        self.cmp.to(self.device)

    def training_step(self, batch, _):
        cmp_state = self.cmp.compute_cmp_state(batch)
        # No dual variable under QuadraticPenalty, so the primal Lagrangian *is* the training loss and
        # Lightning steps it as usual; compute_dual_lagrangian would return an empty store.
        lagrangian = cmp_state.compute_primal_lagrangian().lagrangian
        self._log_training_metrics(cmp_state, lagrangian)
        # Raise the penalty coefficient for constraints still violated beyond the tolerance. Called after
        # the loss is built so the logged coefficient is the one that produced this step's gradient.
        self.penalty_updater.step(cmp_state.observed_constraints)
        return lagrangian

    def _log_training_metrics(self, cmp_state, lagrangian):
        """Log the objective split into its parts, plus what the constraint currently sees.

        ``train/loss`` stays the *optimized* quantity (now the Lagrangian) and ``train/regression_loss``
        isolates the fit term, so the two curves show the price the constraint is charging.
        ``monotonicity/true_violation`` is the identical measurement on the batch's targets: in raw space
        it must sit at ~0, so a non-zero reading means the metric diagonal is mis-wired rather than that
        the model is at fault, and it floors what the model can be asked to achieve.
        """
        prediction, targets = cmp_state.misc["prediction"], cmp_state.misc["targets"]
        pair_violations = cmp_state.misc["pair_violations"]
        batch_size = targets.shape[0]
        log = lambda name, value: self.log(name, value, on_step=False, on_epoch=True, batch_size=batch_size)

        log("train/loss", lagrangian)
        log("train/regression_loss", cmp_state.loss)
        log("train/mse", F.mse_loss(self.inverse_target(prediction), self.inverse_target(targets)))
        # The penalized quantity is the *sum* of squared hinges over pairs, so track both its worst entry
        # and its extent: max alone hides how widespread the violation is, mean alone hides its depth.
        log("monotonicity/max_pair_violation", pair_violations.max())
        log("monotonicity/mean_pair_violation", pair_violations.mean())
        log("monotonicity/frac_pairs_violating", (pair_violations > self.cmp.tolerance).float().mean())
        log("monotonicity/true_violation", cmp_state.misc["true_pair_violations"].max())
        log("monotonicity/penalty", self.cmp.monotone.penalty_coefficient.value.max())

    def on_save_checkpoint(self, checkpoint):
        # The penalty coefficient is a plain tensor holder, not a Parameter or buffer, so Lightning would
        # not persist it and a resumed run would restart the schedule from penalty_mu.
        checkpoint["cmp"] = self.cmp.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.cmp.load_state_dict(checkpoint["cmp"])
