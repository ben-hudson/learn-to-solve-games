import torch


class NormLoss(torch.nn.Module):
    """``||prediction - target||`` over each sample's flattened field, averaged over samples.

    Zero-residual samples are masked out of the mean: the norm's gradient at exactly zero is NaN.
    """

    def forward(self, prediction, target):
        # flatten start_dim=1 because the operator is a vector, so we have n_players*n_actions
        residual_norm = (prediction - target).flatten(start_dim=1).norm(dim=-1)
        return residual_norm[residual_norm > 0].mean()


class NashAprLoss(torch.nn.Module):
    """Nash approximation loss for bimatrix games (Duan et al. 2023, Eq. 1), averaged over the batch.

    ``NashApr(σ, u) = max_i max_{a_i} [u_i(a_i, σ_{-i}) - u_i(σ)]`` is the largest utility any player
    gains by deviating to a pure best response, so it is zero exactly at a Nash equilibrium. It is
    computed from the strategies and payoffs alone -- no equilibrium labels -- which sidesteps the
    equilibrium-selection problem that breaks supervised regression onto a solver's solution.
    """

    def forward(self, strategies, A, B):
        x, y = strategies.unbind(dim=-2)
        # each player's expected payoff per pure action against the opponent's mixed strategy:
        # (Ay)_a = u_1(a, y) and (x^T B)_a = u_2(x, a)
        pure_payoffs_1 = torch.einsum("bij,bj->bi", A, y)
        pure_payoffs_2 = torch.einsum("bi,bij->bj", x, B)
        # deviation gain = best pure payoff - current expected payoff u_i(x, y)
        gain_1 = pure_payoffs_1.max(dim=-1).values - torch.einsum("bi,bi->b", x, pure_payoffs_1)
        gain_2 = pure_payoffs_2.max(dim=-1).values - torch.einsum("bj,bj->b", y, pure_payoffs_2)
        return torch.maximum(gain_1, gain_2).mean()
