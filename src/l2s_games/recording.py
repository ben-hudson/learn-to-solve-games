"""Recording a rollout's field evaluations, so the rollout doubles as its own data collection.

Rolling out the ground-truth operator already pays one evaluation per visited state, and those evaluations
*are* the regression targets -- but ``simulate`` returns only the iterates, so without this they would be
discarded and a subsample of the trajectory re-evaluated. ``RecordedField`` wraps the analytic field and keeps
every triple it is asked for, turning ``n_steps`` evaluations into ``n_steps`` examples.

Used by ``operator_datasets.ExpertOperatorDataset`` (single-instance generation, via ``evaluations``) and by
the streaming expert source (batched, via ``groups``); ``with_endpoint`` appends the converged endpoint, the
one state the algorithm never queried and the only near-zero target in the set.
"""

import torch

from l2s_games.data import OperatorEvaluations


class RecordedField:
    """An analytic field that keeps every ``(state, value, preconditioner diagonal)`` triple it is asked for.

    Rolling out the ground-truth operator already pays one evaluation per visited state, and those
    evaluations *are* the regression targets -- but ``simulate`` returns only the iterates, so they used
    to be discarded and a subsample of the trajectory re-evaluated (``trajectory_examples``). Recording
    them instead makes the expert rollout its own data collection: ``n_steps`` evals, ``n_steps`` examples.

    Recording at the **field** level rather than per step is what keeps this algorithm-agnostic:
    ``extragradient`` calls the field twice per step (at the iterate and at the lookahead) and both are
    legitimate ``(state, operator value)`` pairs, while ``projection`` and ``optimistic`` call it once.
    The negation for descent stays outside (``batched_rollout`` rolls out ``-field``), so the retained
    values are the unnegated operator -- the target convention the whole pipeline regresses.

    Two readers, matching the two rollout shapes: ``evaluations`` for a single instance (rank-1 ``params``,
    the ``operator_datasets`` generation path) and ``groups`` for a batch (rank-2, the streaming path).
    Every recorded state is feasible under every algorithm: the lookahead methods project their
    intermediate points (their textbook constrained forms do -- see ``algorithms``), so the field is only
    ever asked about projected points. Which points those are does vary: ``projection`` queries the
    iterates, while ``optimistic`` queries its extrapolated points ``z_bar`` -- both feasible, both on the
    distribution the rollout actually traverses.
    """

    def __init__(self, family, params):
        self.family = family
        self.params = params
        self.states, self.values, self.preconditioner_diagonals = [], [], []

    def __call__(self, z):
        values, preconditioner_diagonal = self.family.operator_and_preconditioner(self.params, z)
        self.states.append(z.detach().clone())
        self.values.append(values)
        self.preconditioner_diagonals.append(preconditioner_diagonal)
        return values

    def _stacked(self):
        """The three recordings stacked over field calls -- ``[n_calls, ...]`` each."""
        return (
            torch.stack(record)
            for record in (self.states, self.values, self.preconditioner_diagonals)
        )

    def evaluations(self, params):
        """One ``OperatorEvaluations`` for a **single**-instance rollout (rank-1 ``params``).

        Each field call recorded a bare ``[d]`` row, so the stacks are already the
        ``[points_per_instance, d]`` an ``OperatorEvaluations`` wants -- no column to select.
        """
        return OperatorEvaluations(params, *self._stacked())

    def groups(self, instances):
        """One ``OperatorEvaluations`` per instance of a **batched** rollout, ``[n_calls, d]`` each.

        The recordings stack to ``[n_calls, B, d]`` (field calls over the ``B``-instance batch), so
        instance ``b``'s evaluations are column ``b`` of each stack.
        """
        states, values, diagonals = self._stacked()
        return [
            OperatorEvaluations(inst, states[:, b], values[:, b], diagonals[:, b])
            for b, inst in enumerate(instances)
        ]

def with_endpoint(evaluations, point, target, preconditioner_diagonal):
    """Extend one instance's ``OperatorEvaluations`` with its equilibrium endpoint.

    The endpoint is the one state the algorithm never queried (``simulate`` returns it but never
    evaluated the field there), so it costs one extra evaluation -- and it is the only near-zero target
    the model ever sees. Keeping it inside the same ``OperatorEvaluations`` holds one record per instance,
    hence a uniform ``points_per_instance``.
    """
    return OperatorEvaluations(
        evaluations.params,
        torch.cat([evaluations.points, point[None]]),
        torch.cat([evaluations.targets, target[None]]),
        torch.cat([evaluations.preconditioner_diagonal, preconditioner_diagonal[None]]),
    )
