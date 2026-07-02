"""Validation callback: roll out the learned field with one algorithm, log the analytic residual.

Lives above the model/dynamics/algorithms layers as glue -- keeps ``FieldModel`` free of any
dynamics dependency (the project pipeline is family -> dataset -> field model -> dynamics). One
callback runs one algorithm; compose a list to sweep several.
"""

import lightning as L

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate


class EquilibriumRolloutCallback(L.Callback):
    """Each validation batch, roll out ``algo`` on the learned field and log its endpoint residual.

    The rollout starts from ``family.initial_point`` and is projected onto the feasible set each
    step; the endpoint's analytic residual ``||operator(params, z)||`` is logged as
    ``val/{algo}/residual`` (mean over the batch, aggregated over the epoch).
    """

    def __init__(self, family, algo, n_steps, h):
        super().__init__()
        self.family = family
        self.algo = algo
        self.n_steps = n_steps
        self.h = h

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        field = pl_module.batched_field(self.family, inputs)
        z0 = self.family.initial_point(inputs)
        project = lambda z: self.family.project(inputs, z)
        params = self.family.params_from_batch(inputs)
        # consensus' torch.func.grad manages its own grad tracking, so the ambient no-grad is fine;
        # params stay out of autograd (avoids the flash-attention grad-mask kernel error).
        traj = simulate(lambda z: -field(z), ALGORITHMS[self.algo](self.h), z0, self.n_steps, project=project)
        residual = self.family.operator(params, traj[-1]).norm(dim=-1).mean()
        pl_module.log(f"val/{self.algo}/residual", residual, on_epoch=True, batch_size=targets.shape[0])
