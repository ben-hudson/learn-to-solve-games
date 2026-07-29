"""Tests for the game-dynamics algorithms (``algorithms.ALGORITHMS`` + ``dynamics.simulate``).

Covers the properties the ``step(z, v, project)`` contract exists to guarantee: every point an algorithm
*queries the field at* is feasible (not just every iterate it returns), the lookahead methods converge on
a purely rotational field where plain projection diverges, ``optimistic`` gets that for one field
evaluation per step, and threading ``project`` through leaves the unconstrained dynamics untouched.

Built on ``toy.rotational_field`` with the wells switched off, which is exactly ``v(z) = R z`` for the
antisymmetric ``R = omega * [[0,-1],[1,0]]``: purely rotational, zero at the origin, and with a *known*
``omega``, so the convergence thresholds follow from the closed-form per-step factor rather than from
whatever magnitude a sampled instance happens to have. Nothing here needs PUME or the traffic data.
"""

import pytest
import torch

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate
from l2s_games.envs.toy import rotational_field

_LOOKAHEAD = ["extragradient", "optimistic"]  # the methods that evaluate the field off the iterate
_Z0 = torch.tensor([0.2, 0.1])
# v(z) = R z with omega = 1. Descending on -v, one extragradient step scales the iterate by
# sqrt(1 - a^2 + a^4) with a = h * omega, so h = 0.2 contracts ~2% per step: unambiguous over 400 steps,
# while plain forward Euler *grows* by sqrt(1 + a^2) per step on the same field.
_OMEGA, _H, _N_STEPS = 1.0, 0.2, 400


class RecordingField:
    """A field that keeps every point it was evaluated at (the states an expert rollout would record)."""

    def __init__(self, field):
        self.field = field
        self.queries = []

    def __call__(self, z):
        self.queries.append(torch.as_tensor(z).clone())
        return self.field(z)


@pytest.fixture
def descent_field():
    """``-v`` for the purely rotational field, i.e. the direction ``simulate`` is rolled out along."""
    field = rotational_field(omega=_OMEGA)
    return lambda z: -field(z)


@pytest.mark.parametrize("name", _LOOKAHEAD)
def test_queried_points_are_feasible(name):
    """Every point the field is *asked about* lies in the feasible set, not just every iterate.

    This is the property the expert data stream depends on (``rollout_sampling.RecordedField`` turns
    exactly these queries into training examples), and the one an unprojected lookahead breaks: the
    returned iterates would still be feasible because ``simulate`` projects them, while the lookahead
    the field actually saw was not.
    """
    lo, hi = -1.0, 1.0
    project = lambda z: z.clamp(lo, hi)
    # A strong outward field: any unprojected intermediate step escapes the box immediately.
    field = RecordingField(lambda z: 10.0 * torch.sign(z))
    simulate(field, ALGORITHMS[name](0.5), torch.tensor([0.9, -0.9]), n_steps=10, project=project)

    assert field.queries, "the field was never evaluated"
    worst = max(query.abs().max().item() for query in field.queries)
    assert worst <= hi + 1e-6, f"{name} queried the field at an infeasible point (|z|max = {worst})"


@pytest.mark.parametrize("name", _LOOKAHEAD)
def test_lookahead_converges_where_projection_diverges(name, descent_field):
    """Lookahead methods drive the rotational field to its zero; plain projection spirals outward.

    Both halves matter: a purely rotational field is monotone but not *strongly* monotone, which is
    exactly the class where forward Euler has no guarantee and the extragradient family does.
    """
    rollout = lambda algo: simulate(descent_field, algo, _Z0, _N_STEPS)

    lookahead_end = rollout(ALGORITHMS[name](_H))[-1].norm()
    projection_end = rollout(ALGORITHMS["projection"](_H))[-1].norm()

    assert lookahead_end < 0.1 * _Z0.norm(), f"{name} did not converge (||z|| = {lookahead_end:.3g})"
    assert projection_end > 10.0 * _Z0.norm(), "projection unexpectedly contracted a rotational field"


def test_optimistic_costs_one_field_evaluation_per_step(descent_field):
    """``optimistic`` (Popov) evaluates the field once per step, ``extragradient`` twice.

    The reason ``optimistic`` is the cheaper driver for ground-truth rollouts, where each evaluation is a
    solve billed against the training budget -- so it is worth pinning, not incidental.
    """
    n_steps = 20
    counts = {}
    for name in _LOOKAHEAD:
        field = RecordingField(descent_field)
        simulate(field, ALGORITHMS[name](_H), _Z0, n_steps)
        counts[name] = len(field.queries)

    assert counts["optimistic"] == n_steps + 1  # one per step, plus the first step's initialization
    assert counts["extragradient"] == 2 * n_steps


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_identity_project_leaves_dynamics_unchanged(name, descent_field):
    """Passing an explicit identity ``project`` is indistinguishable from the default (unconstrained).

    Guards the plumbing itself: threading the projection through ``step`` must not perturb the flat-game
    dynamics, whose domains are unconstrained. Parametrized over the whole registry, so it also covers
    ``momentum`` / ``consensus``, which now project their returned iterate.
    """
    rollout = lambda **kwargs: simulate(descent_field, ALGORITHMS[name](_H), _Z0, 25, **kwargs)

    assert torch.equal(rollout(), rollout(project=lambda z: z))
