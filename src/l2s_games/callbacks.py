"""Validation callback: roll out the learned field with one algorithm, log the analytic residual.

Lives above the model/dynamics/algorithms layers as glue -- keeps ``FieldModel`` free of any
dynamics dependency (the project pipeline is family -> dataset -> field model -> dynamics). One
callback runs one algorithm; compose a list to sweep several.

``VizRolloutCallback`` logs the rollout + true/learned field visualizations through training for the
on-policy training mode (see ``rollout_sampling``); the on-policy buffer itself is owned by
``rollout_sampling.OnPolicyOperatorStream``, not a callback.
"""

import os

import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate
from l2s_games.viz import plot_trajectory_arrows


class RolloutCallback(L.Callback):
    """Each validation batch, roll out ``algo`` on the learned field and log two endpoint metrics.

    The rollout starts from ``family.initial_point`` -- each example's uniformly sampled domain
    point (drawn by ``sample_domain`` at dataset-build time, so fixed across epochs), matching the
    start distribution the on-policy collector trains on -- and is projected onto the feasible set
    each step. Logged (mean over the batch, aggregated over the epoch):
    ``val/{algo}/residual`` -- the analytic operator norm ``||operator(params, z_end)||`` -- and
    ``val/{algo}/dist_to_eq`` -- the distance ``||z_end - z*||`` from the endpoint to the true
    equilibrium ``z*``.

    ``equilibrium`` is ``z*``, broadcast over the batch. It defaults to the origin (the Nash-centered
    chart's equilibrium for the matrix-game scope); the traffic setting will pass its pre-computed
    per-instance equilibria as a tensor.
    """

    def __init__(self, family, algo, n_steps, h, equilibrium=0.0):
        super().__init__()
        self.family = family
        self.algo = algo
        self.n_steps = n_steps
        self.h = h
        self.equilibrium = equilibrium

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        field = pl_module.batched_field(self.family, inputs)
        z0 = self.family.initial_point(inputs)
        project = lambda z: self.family.project(inputs, z)
        params = self.family.params_from_batch(inputs)
        # consensus' torch.func.grad manages its own grad tracking, so the ambient no-grad is fine;
        # params stay out of autograd (avoids the flash-attention grad-mask kernel error).
        traj = simulate(lambda z: -field(z), ALGORITHMS[self.algo](self.h), z0, self.n_steps, project=project)
        endpoint = traj[-1]
        residual = self.family.operator(params, endpoint).norm(dim=-1).mean()
        dist_to_eq = (endpoint - self.equilibrium).norm(dim=-1).mean()
        pl_module.log(f"val/{self.algo}/residual", residual, on_epoch=True, batch_size=targets.shape[0])
        pl_module.log(f"val/{self.algo}/eq_dist", dist_to_eq, on_epoch=True, batch_size=targets.shape[0])


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
    ``on_epoch=True`` (see ``models.base`` and ``RolloutCallback``). Logged per training batch instead,
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
