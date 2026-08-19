import torch

from l2s_games.envs.traffic.game import PotentialCongestion
from torch_geometric.utils import scatter
from typing import List


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


class WardropAprLoss(torch.nn.Module):
    """Hard per-node deviation gain at the predicted costs, worst case per game, batch-averaged.

    The traffic analogue of ``NashAprLoss``: at each node a traveller bound for destination
    ``d`` plays the mixed next-edge strategy ``pi^d``. With edge payoffs
    ``Q^d(e) = -c_e + V^d(head(e))`` -- predicted edge cost plus the perturbed cost-to-go
    downstream -- the gain at a node is ``max_e Q^d(e) - <pi^d, Q^d>``: the saving from
    committing to the single best outgoing edge, holding the costs and the downstream
    strategy fixed. The loss is the largest gain over nodes and destinations, so it is the
    worst one-step Wardrop-condition violation in the network.

    ``V`` and ``pi`` are the perturbed (entropy) best response to the predicted costs, from
    the differentiable PUMCM solve, so the gain includes the entropy smoothing residual and
    does not reach zero at the stochastic user equilibrium. It also never consults the
    supply side: it scores only the internal consistency of the demand response.

    Predictions are projected onto the feasible cost box before the solve; the projection's
    gradient is zero outside the box (no straight-through), so pair this with a loss that
    pushes infeasible predictions back inside.
    """

    def forward(self, games: List[PotentialCongestion], pred_costs: torch.Tensor):
        # PUME solves one instance at a time in float64 on the CPU; MPS cannot cast to
        # float64, so move first. The .cpu()/.double() ops keep the autograd graph intact.
        gains = [
            self._worst_deviation_gain(game, game.project_costs(sample))
            for game, sample in zip(games, pred_costs.cpu().double())
        ]
        return torch.stack(gains).to(dtype=pred_costs.dtype, device=pred_costs.device).mean()

    def _worst_deviation_gain(self, game: PotentialCongestion, costs: torch.Tensor):
        best_response = game.best_response(costs)
        values, policies = best_response["value"], best_response["policy"]
        tails, heads = game.edge_index
        payoffs = values[:, heads] - costs
        best = scatter(payoffs, tails, dim=-1, dim_size=game.n_nodes, reduce="max")
        expected = scatter(policies * payoffs, tails, dim=-1, dim_size=game.n_nodes, reduce="sum")
        # a destination places no probability on its own out-edges (termination state), so
        # its own row carries no strategy and its gap is meaningless
        is_dest = scatter(policies, tails, dim=-1, dim_size=game.n_nodes, reduce="sum") == 0
        return (best - expected)[~is_dest].max()


class PolicyKLDiv(torch.nn.Module):
    """Flow-weighted KL divergence between the policies at the predicted and induced costs,
    averaged over the batch -- the smooth (Fenchel-Young) Wardrop gap.

    Travellers best-respond to the predicted costs; the loss scores that response in the
    perturbed utility they actually experience -- the congested times induced by their own
    flows. Per node and destination the Fenchel-Young gap
    ``H(Q_t) + Omega(pi_c) - <pi_c, Q_t>`` reduces to ``KL(pi_c || pi_t)``, the divergence
    between the response to the predicted costs and the best response to the induced times.
    It is zero at every node exactly at the stochastic user equilibrium -- no smoothing
    floor, unlike the hard ``WardropAprLoss``. Weighting each node's KL by its throughflow
    collapses the sum over nodes into an occupancy-weighted log-ratio over edges:
    ``sum_d sum_e flow^d_e (log pi_c^d_e - log pi_t^d_e)``.

    Gradients flow through both solves and the congestion map (PUMCM implicit diff): the
    prediction is pulled toward the best response to the realized times and toward inducing
    times its own response is optimal for. Self-supervised: no equilibrium labels.
    """

    def forward(self, games: List[PotentialCongestion], pred_costs: torch.Tensor):
        divergences = [
            self._flow_weighted_kl(game, game.project_costs(sample))
            for game, sample in zip(games, pred_costs.cpu().double())
        ]
        return torch.stack(divergences).to(dtype=pred_costs.dtype, device=pred_costs.device).mean()

    def _flow_weighted_kl(self, game: PotentialCongestion, costs: torch.Tensor):
        response = game.best_response(costs, return_demand=True)
        # the response's flows congest the network; project because extreme predictions can
        # push the induced times past the box, where the solve's exponentials underflow
        induced_times = game.project_costs(game.travel_time(response["demand"].sum(dim=0)))
        induced_response = game.best_response(induced_times)
        # untravelled edges carry no weight and are exactly where pi = 0, so drop them
        # before the log
        travelled = response["demand"] > 0
        log_ratio = response["policy"][travelled].log() - induced_response["policy"][travelled].log()
        return (response["demand"][travelled] * log_ratio).sum()


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
