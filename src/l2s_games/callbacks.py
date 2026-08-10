"""Validation callbacks: score an equilibrium estimate by the analytic operator residual.

Lives above the model/dynamics/algorithms layers as glue -- keeps the models free of any dynamics
dependency (the project pipeline is family -> dataset -> model -> dynamics). Both amortization modes
produce an equilibrium estimate ``z_end`` and log the same two metrics at it (``_log_equilibrium_metrics``);
they differ only in how ``z_end`` is produced, so each has its own thin callback:

- ``FieldRolloutCallback`` -- the field model has no solution of its own, so it rolls out one ``algo``
  on the learned field to reach ``z_end`` (compose a list to sweep several algorithms).
- ``SolutionPredictionCallback`` -- the solution model predicts ``z*`` directly, so ``z_end`` is just
  its (projected) prediction; no rollout.

The streaming path's own callbacks -- ``OperatorCountCallback`` (the streaming operator budget) and
``VizRolloutCallback`` (on-policy rollout plots) -- live in ``streaming`` with the sources that need them.
"""

import lightning as L

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import natural_map, simulate


def _log_equilibrium_metrics(pl_module, family, inputs, z_end, name, equilibrium):
    """Log the two endpoint metrics at an equilibrium estimate ``z_end`` (mean over the batch, per epoch).

    ``val/{name}/residual`` -- the **natural map** norm ``||z_end - project(z_end - operator(z_end))||``
    of the *analytic* operator -- and ``val/{name}/eq_dist`` -- the distance ``||z_end - equilibrium||``
    to the reference ``z*``. Shared by the field (rollout) and solution (direct) callbacks so both sit
    on the same axes and neither reimplements the scoring.

    The natural map rather than the bare ``||operator||`` because the feasible set is constrained
    (traffic clamps ``costs >= free_flow_time``): at a boundary solution the operator points out of the
    feasible set, so its norm stays bounded away from zero there while the natural map vanishes. Both
    are exact solution certificates on the interior -- they coincide wherever no constraint is active,
    so this only changes the metric at the boundary.

    Taken with a **unit** step, which is a convention: the natural map's zero set is the solution set
    for any step ``h > 0``, so ``h`` sets only the scale. ``1`` because (a) an algorithm's ``h`` is
    tuned damping (a solver knob), and a convergence measure that inherited it would rescale when the
    step size changed, breaking comparability across the algorithm sweep and against the solution model
    (which has no rollout, hence no ``h``); and (b) the default preconditioned PUME operator is already
    in domain (cost) units, so ``h`` is a dimensionless damping factor whose undamped value is 1 -- this
    is the residual of the undamped preconditioned fixed-point iteration. Under ``--no-precondition``
    the raw flow-residual operator is not cost-scaled, so the *magnitude* is arbitrary (the zero is not).
    """
    params = family.params_from_batch(inputs)
    residual = natural_map(family, params, z_end).norm(dim=-1).mean()
    eq_dist = (z_end - equilibrium).norm(dim=-1).mean()
    batch_size = z_end.shape[0]
    pl_module.log(f"val/{name}/residual", residual, on_epoch=True, batch_size=batch_size)
    pl_module.log(f"val/{name}/eq_dist", eq_dist, on_epoch=True, batch_size=batch_size)


class FieldRolloutCallback(L.Callback):
    """Score a **field model** by rolling out ``algo`` on its learned field (``--amortization partial``).

    The field model has no predicted solution, so ``z_end`` is the endpoint of rolling out ``algo`` on
    the learned field from ``family.initial_point`` -- each example's uniformly sampled domain point
    (drawn by ``sample_domain`` at dataset-build time, so fixed across epochs), matching the start
    distribution the on-policy collector trains on -- projected onto the feasible set each step. One
    callback runs one algorithm; compose a list to sweep several, each logging
    ``val/{algo}/{residual,eq_dist}``.

    The reference ``z*`` for ``eq_dist`` comes from the family's ``reference_equilibrium`` seam rather than
    from a constructor argument: for traffic it is each instance's own cached ``equilibrium``, which the
    collated batch already carries, and for the Nash-centered matrix-game charts it is the origin. It used to
    be a constructor default of ``0.0`` that no caller ever overrode, so ``eq_dist`` was silently reporting
    ``||z_end||`` on the traffic families.
    """

    def __init__(self, family, algo, n_steps, h):
        super().__init__()
        self.family = family
        self.algo = algo
        self.n_steps = n_steps
        self.h = h

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs, _ = batch
        field = pl_module.batched_field(self.family, inputs)
        z0 = self.family.initial_point(inputs)
        project = lambda z: self.family.project(inputs, z)
        # consensus' torch.func.grad manages its own grad tracking, so the ambient no-grad is fine;
        # params stay out of autograd (avoids the flash-attention grad-mask kernel error).
        endpoint = simulate(lambda z: -field(z), ALGORITHMS[self.algo](self.h), z0, self.n_steps, project=project)[-1]
        equilibrium = self.family.reference_equilibrium(inputs)
        _log_equilibrium_metrics(pl_module, self.family, inputs, endpoint, self.algo, equilibrium)


class SolutionPredictionCallback(L.Callback):
    """Score a **solution model** at its directly-predicted equilibrium (``--amortization full``).

    ``z_end`` is the model's projected, de-standardized prediction (``SolutionModel.solve``) -- no
    rollout. The reference ``z*`` for ``eq_dist`` is exact here: the solution model's validation
    *target* **is** the cached equilibrium, read off the batch. Logs ``val/solution/{residual,eq_dist}``,
    the same axes as ``FieldRolloutCallback``.
    """

    def __init__(self, family):
        super().__init__()
        self.family = family

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        z_end = pl_module.solve(self.family, inputs)
        # The target is the standardized z*; the model owns the de-standardization (its normalizer).
        equilibrium = pl_module.inverse_target(targets)
        _log_equilibrium_metrics(pl_module, self.family, inputs, z_end, "solution", equilibrium)


