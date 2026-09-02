"""Jacobian-free field surrogates, shared by every environment.

Both losses report the squared ascent field at (or near) the prediction, and both substitute the
*field* for the true Jacobian-vector product in the backward. Gradient descent on the surrogate
therefore moves the prediction *along* the field -- ``PotentialLoss`` reproduces
``algorithms.SimpleProjection``, ``EGLoss`` reproduces ``algorithms.ExtraGradient`` -- pushed back
through whatever network produced it. The true gradient of the squared residual is ``J^T v``, which
needs a Jacobian that is typically stiff, often rotational, and sometimes (wherever the field passes
through a clamp) identically zero in the directions that matter.

Everything game-specific is injected, on the same contract ``algorithms.Algorithm`` uses: an ascent
field ``field_fn(iterate)`` and, where there is a feasible set, a ``project_fn`` for the lookahead
point. The environments wrap these with their own signatures --
``envs.zero_sum.losses.PotentialLoss`` binds the bimatrix operator and the simplex projection,
``envs.spe.models.SpeSolutionModel`` binds the negated market-clearing operator and no projection.

Whether the forward value falls to zero at a solution is a property of the environment, not of the
loss: a zero-sum game's field is a nonzero vector normal to the simplex at an interior equilibrium,
so its surrogate is not monotone over training and ``val/residual`` is the metric to read; an
unconstrained equation like ``PriceSPE`` has ``field = 0`` at equilibrium, so there the value is a
genuine fit signal.
"""

import torch


def no_projection(iterate):
    """Identity: the lookahead of a variational inequality with no constraint set needs no fixing."""
    return iterate


class PotentialLoss(torch.nn.Module):
    """Squared ascent field at the prediction; the field itself as the descent direction.

    One field evaluation per step. Converges on strongly monotone or cocoercive operators and fails
    on a purely rotational one -- the same terms under which ``algorithms.SimpleProjection`` does.
    """

    def forward(self, field_fn, iterate):
        return PotentialGradient.apply(iterate, field_fn).sum(dim=-1).mean()


class PotentialGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, iterate, field_fn):
        field = field_fn(iterate)
        ctx.save_for_backward(field)
        return field.square()

    @staticmethod
    def backward(ctx, grad_loss):
        (field,) = ctx.saved_tensors
        # one gradient per forward input, positionally: (iterate, field_fn). Returning ``-field``
        # makes the optimizer's descent step ``z <- z - lr * dL/dz`` move the iterate *along* the
        # field, the ascent convention every dynamic in ``algorithms`` follows.
        return -field * grad_loss, None


class EGLoss(PotentialLoss):
    """Extragradient surrogate: the squared field at the *lookahead* point, and the field there.

    ``step_size`` is the fixed lookahead distance ``h`` in ``z_half = project(z + h * v(z))``,
    independent of the optimizer's learning rate, which only scales how far the update follows the
    lookahead field. Two field evaluations per step, and convergence on merely monotone operators --
    including the rotational fields where ``PotentialLoss`` stalls or spirals.

    ``project_fn`` is applied to the lookahead point, as Korpelevich's constrained form does; leaving
    it at ``no_projection`` queries the field wherever plain extrapolation lands, which is correct
    only when there is no feasible set to leave.
    """

    def __init__(self, step_size: float, project_fn=no_projection):
        super().__init__()
        self.step_size = step_size
        self.project_fn = project_fn

    def forward(self, field_fn, iterate):
        return EGGradient.apply(iterate, field_fn, self.step_size, self.project_fn).sum(dim=-1).mean()


class EGGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, iterate, field_fn, step_size, project_fn):
        lookahead = project_fn(iterate + step_size * field_fn(iterate))
        field = field_fn(lookahead)
        ctx.save_for_backward(field)
        return field.square()

    @staticmethod
    def backward(ctx, grad_loss):
        (field,) = ctx.saved_tensors
        # positionally: (iterate, field_fn, step_size, project_fn). See ``PotentialGradient``
        # for the sign; everything else here is a constant.
        return -field * grad_loss, None, None, None
