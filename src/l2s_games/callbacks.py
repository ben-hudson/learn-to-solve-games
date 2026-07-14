"""Validation callbacks: score an equilibrium estimate by the analytic operator residual.

Lives above the model/dynamics/algorithms layers as glue -- keeps the models free of any dynamics
dependency (the project pipeline is family -> dataset -> model -> dynamics). Both amortization modes
produce an equilibrium estimate ``z_end`` and log the same two metrics at it (``_log_equilibrium_metrics``);
they differ only in how ``z_end`` is produced, so each has its own thin callback:

- ``FieldRolloutCallback`` -- the field model has no solution of its own, so it rolls out one ``algo``
  on the learned field to reach ``z_end`` (compose a list to sweep several algorithms).
- ``SolutionPredictionCallback`` -- the solution model predicts ``z*`` directly, so ``z_end`` is just
  its (projected) prediction; no rollout.

``VizRolloutCallback`` logs the rollout + true/learned field visualizations through training for the
on-policy training mode (see ``rollout_sampling``).
"""

import os

import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate
from l2s_games.viz import plot_trajectory_arrows


def _log_equilibrium_metrics(pl_module, family, inputs, z_end, name, equilibrium):
    """Log the two endpoint metrics at an equilibrium estimate ``z_end`` (mean over the batch, per epoch).

    ``val/{name}/residual`` -- the analytic operator norm ``||operator(params, z_end)||`` (zero at a
    true equilibrium) -- and ``val/{name}/eq_dist`` -- the distance ``||z_end - equilibrium||`` to the
    reference ``z*``. Shared by the field (rollout) and solution (direct) callbacks so both sit on the
    same axes and neither reimplements the scoring.
    """
    params = family.params_from_batch(inputs)
    residual = family.operator(params, z_end).norm(dim=-1).mean()
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

    ``equilibrium`` is the reference ``z*`` for ``eq_dist``. It defaults to the origin (the
    Nash-centered chart's equilibrium for the matrix-game scope); the traffic setting will pass its
    pre-computed per-instance equilibria as a tensor.
    """

    def __init__(self, family, algo, n_steps, h, equilibrium=0.0):
        super().__init__()
        self.family = family
        self.algo = algo
        self.n_steps = n_steps
        self.h = h
        self.equilibrium = equilibrium

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs, _ = batch
        field = pl_module.batched_field(self.family, inputs)
        z0 = self.family.initial_point(inputs)
        project = lambda z: self.family.project(inputs, z)
        # consensus' torch.func.grad manages its own grad tracking, so the ambient no-grad is fine;
        # params stay out of autograd (avoids the flash-attention grad-mask kernel error).
        endpoint = simulate(lambda z: -field(z), ALGORITHMS[self.algo](self.h), z0, self.n_steps, project=project)[-1]
        _log_equilibrium_metrics(pl_module, self.family, inputs, endpoint, self.algo, self.equilibrium)


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


class OperatorCountCallback(L.Callback):
    """Log the cumulative ground-truth operator point-evaluation budget once per epoch.

    The ``SharedCounter`` (see ``operator_count``) accumulates every training-data operator call
    across the streaming workers and the main process; this logs its current (monotonic) value, so
    the logged series *is* the cumulative-sum curve -- no in-dashboard cumsum needed. Logging goes
    through ``pl_module.log`` (never ``experiment.log``) so wandb's step bookkeeping stays in sync;
    register it as a ``wandb.define_metric`` step_metric at the call site to plot other metrics
    against the budget.

    Logged at the **epoch boundary** with ``on_epoch=True`` (not per training batch), so it lands on
    the same logging step as the epoch-aggregated metrics it exists to be plotted against -- the model
    logs ``train/loss`` / ``train/mse`` with ``on_step=False, on_epoch=True`` and the val metrics
    ``on_epoch=True`` (see ``models.base`` and ``FieldRolloutCallback``). Logged per training batch instead,
    the budget occupied its own logging steps that no ``on_epoch`` metric ever shared, so selecting it
    as a custom wandb x-axis returned "no data" (nothing to pair against). Per-epoch is also the right
    granularity: every plottable metric here is per-epoch.
    """

    def __init__(self, counter, key="train/operator_evals"):
        super().__init__()
        self.counter = counter
        self.key = key

    def on_train_epoch_end(self, trainer, pl_module):
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
