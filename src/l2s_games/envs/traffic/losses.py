import torch

from typing import List
from l2s_games.envs.traffic.game import PotentialCongestion


class PotentialLoss(torch.nn.Module):
    """Dual potential loss for congestion games, averaged over the batch.

    Descends ``-D(c) = sum_e Phi*_e(c_e) + sum_d <g^d, V^d(c)>``, the convex dual of the
    Beckmann-entropy SUE program, whose unique minimizer is the equilibrium cost vector.
    Its exact gradient is the excess supply ``s(c) - f(c)``: the supply term because
    ``grad Phi* = s``, the demand term by Danskin -- so no gradient flows through the
    inner demand solve. Self-supervised: no equilibrium labels are needed.

    With ``precondition=True`` the gradient is rescaled by PUME's supply-diagonal metric
    (``operator_and_preconditioner``): the step follows ``M^{-1}(s(c) - f(c))``, a descent
    direction for ``-D`` in the ``M``-metric with the same zero, pulling the stiff flow
    residual back toward cost units.

    The returned value is the mean squared norm of the (preconditioned) excess supply --
    a monitor in ``val/residual``'s units, not the potential itself, whose gradient is
    what ``PotentialGradient`` actually descends.
    """

    def __init__(self, precondition: bool = False):
        super().__init__()
        self.precondition = precondition

    def forward(self, games: List[PotentialCongestion], pred_costs: torch.Tensor):
        squared_excess_supply = PotentialGradient.apply(games, pred_costs, self.precondition)
        return squared_excess_supply.sum(dim=-1).mean()


class PotentialGradient(torch.autograd.Function):
    """Excess supply as the analytic gradient of the dual potential at the predicted costs.

    The forward value is the elementwise squared excess supply (a monitor with the same
    zero); backward returns the excess supply itself, the exact gradient of ``-D`` with
    respect to the costs. PUME solves one instance at a time in float64 on the CPU, so
    the batch is looped there and the result cast back to the input's dtype and device.
    """

    @staticmethod
    def forward(ctx, games: List[PotentialCongestion], costs: torch.Tensor, precondition: bool):
        fields = []
        # move to the CPU before the float64 cast, which MPS does not support
        for game, sample in zip(games, costs.cpu().double()):
            # evaluate on the feasible box: outside it the demand solve degenerates. The
            # gradient is still returned at the raw prediction (straight-through), and the
            # boundary field points back inside, so infeasible predictions get pushed in
            sample = game.project_costs(sample)
            if precondition:
                excess_supply, metric_diagonal = game.operator_and_preconditioner(sample)
                fields.append(excess_supply / metric_diagonal)
            else:
                fields.append(game.operator(sample))
        field = torch.stack(fields).to(dtype=costs.dtype, device=costs.device)
        ctx.save_for_backward(field)
        return field.square()

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        (field,) = ctx.saved_tensors
        return None, field * grad_loss, None
