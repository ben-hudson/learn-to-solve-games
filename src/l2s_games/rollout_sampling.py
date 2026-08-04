"""Training points from *rolling out* a field: on-policy (learned field) and expert (true field).

The uniform pipeline (``data.build_dataset``) trains on points drawn uniformly over the domain.
The two streams here instead sample the points a *solver actually visits*, rolling out a batch of
instances jointly and evaluating the ground-truth operator at the visited states:

- ``OnPolicyOperatorStream`` rolls out the **current learned field**, so the model is trained on the
  state distribution its own field induces. The field changes as it trains, so the stream refreshes
  its buffer periodically. It holds a live model ref (``num_workers=0``).
- ``ExpertOperatorStream`` rolls out the **true operator** with a converging algorithm, exposing both
  the expert *trajectory* (the path a good solver takes) and the *equilibrium solution* (the
  converged endpoint). It is model-free -- holds only the picklable ``family_factory`` -- so it runs
  on ``DataLoader`` workers, which it must: the rollout is ``n_steps`` operator solves.

Both share ``batched_rollout``, differing in *which* batched field is rolled out -- and that difference
decides where their training targets come from. The expert rolls out the ground truth, so the operator
values the algorithm consumes **are** the regression targets: a ``RecordedField`` keeps them and the
rollout doubles as the data collection, ~1 eval per example. The on-policy stream rolls out the
*learned* field, whose values are worthless as targets, so it must subsample the visited states and
re-solve the true operator there (``trajectory_examples``) -- which is why only it still pays that
second solve.

Everything runs through the existing family seams -- ``model_input`` / ``transform`` / ``collate_fn``
(conditioning), ``batched_field`` (the batched learned field, real units) or ``operator_and_preconditioner``
(the batched analytic field), ``params_from_batch`` / ``project`` (off the collated batch), ``simulate``
+ ``ALGORITHMS`` (the rollout) -- so they are blind to the concrete representation and work for both the
flat (RPS/matrix) and graph (traffic) families.
"""

import torch

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import OperatorStream, OperatorEvaluations, examples_at_points, operator_examples, normalize_input
from l2s_games.dynamics import simulate


def batched_rollout(family, batch, field, algo, n_steps, z0):
    """Roll out ``-field`` over the whole instance batch; returns the trajectory ``[T+1, B, d]`` (CPU).

    ``field`` is a batched real-unit field ``v(Z): [B, d] -> [B, d]`` (the learned batched field for
    the on-policy collector, the batched analytic operator for the expert), rolled out in descent
    (``-field``, toward the operator's zero) with an **already-constructed** ``algo`` instance -- so
    this makes no assumption about the algorithm's constructor (extra params like momentum ``beta``
    are the caller's concern). ``simulate`` detaches every iterate, so no graph leaks into data
    loading; consensus manages its own autograd internally (caller runs with inference_mode off).
    """
    project = lambda z: family.project(batch, z)
    return simulate(lambda z: -field(z), algo, z0, n_steps, project=project).cpu()


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


def _with_endpoints(instances, groups, points, targets, diagonals):
    """The batched counterpart of ``with_endpoint``: extend every instance's evaluations, or -- when no
    trajectory was kept (the solutions-only baseline) -- make a record of just the endpoint."""
    endpoints = [
        OperatorEvaluations(inst, points[b, None], targets[b, None], diagonals[b, None])
        for b, inst in enumerate(instances)
    ]
    if not groups:
        return endpoints
    return [
        with_endpoint(group, endpoint.points[0], endpoint.targets[0], endpoint.preconditioner_diagonal[0])
        for group, endpoint in zip(groups, endpoints)
    ]


def trajectory_examples(family, instances, traj, n_points):
    """Raw ``(model_input, target)`` examples from ``n_points`` states subsampled across a rollout.

    Flattens the trajectory ``[T+1, B, d]`` over time and instances, tracking each row's instance so
    its params drive the target solve, subsamples ``n_points`` of the visited states, then solves the
    analytic operator per instance over its picked points (one solve per instance) via
    ``examples_at_points``. The assembled examples are shuffled so a minibatch is not dominated by a
    single instance.

    The **on-policy** collector's path: it rolls out the learned field, so the values that rollout
    produced are not ground truth and the visited states have to be re-solved here. The expert rolls out
    the ground truth and keeps its values instead (``RecordedField``), paying no second solve.
    """
    n_steps, n_inst = traj.shape[0], traj.shape[1]
    flat_points = traj.reshape(n_steps * n_inst, -1)  # [(T+1)*B, d]
    flat_inst = torch.arange(n_inst).repeat(n_steps)  # instance index per flat row
    pick = torch.randint(0, flat_points.shape[0], (n_points,))
    points, picked_inst = flat_points[pick], flat_inst[pick]
    examples = []
    for b, inst in enumerate(instances):
        pts = points[picked_inst == b]  # [m, d]
        if pts.shape[0] > 0:
            examples += examples_at_points(family, inst, pts)
    return [examples[i] for i in torch.randperm(len(examples))]


class OnPolicyOperatorStream(OperatorStream):
    """Infinite stream of on-policy rollout points, refreshed from the *current* model each refresh.

    Owns its rollout and its buffer: every ``refresh_every`` epochs (cadenced in ``_raw_stream``) it
    draws ``n_instances`` fresh instances, rolls out the learned field with ``algo`` over them, and
    refills ``self._buffer`` via ``_rollout_buffer``, then cycles that buffer for the rest of the
    window. The per-epoch length is bounded by ``Trainer(limit_train_batches=...)``, not by buffer
    exhaustion, so the epoch always has the same number of batches (which Lightning fixes from epoch 0).

    The refresh serves two purposes here (unlike the model-free ``ExpertOperatorStream``, where it only
    rotates instances): it re-rolls out under the **current** field so the state distribution tracks
    the moving field, *and* it rotates in fresh instances for diversity -- matching the expert stream,
    so the on-policy source is not pinned to a fixed instance subset.

    Holds a **live** ``model`` reference (weights update in place, so it always rolls out the current
    field), which requires ``num_workers=0`` -- the model cannot be pickled to a worker process.
    """

    def __init__(self, family_factory, normalizer, model, algo, h, n_steps, n_instances, points_per_instance, refresh_every):
        super().__init__(family_factory, normalizer)
        self.model = model
        self.algo = algo
        self.h = h
        self.n_steps = n_steps
        self.n_instances = n_instances
        self.points_per_instance = points_per_instance
        self.refresh_every = refresh_every
        self._buffer = None
        self._epoch = -1

    def _rollout_buffer(self, family):
        """Fresh raw ``(model_input, target)`` examples for the on-policy training buffer.

        Draws ``n_instances`` fresh instances, rolls out the *learned* batched field (on the model's
        device) from one uniform start per instance, then delegates the shared subsample + per-instance
        target solve to ``trajectory_examples``. The target is the analytic operator, unnegated -- the
        model regresses the operator itself, exactly as the uniform pipeline does. Uniform coverage
        (exploration / cold-start, while the field is near-random) is supplied by the sibling
        ``UniformSampledOperatorStream``, so this stream is purely on-policy.

        The model (hence the learned field) lives on this device; the rollout must run there, while
        featurization here and the operator solve inside ``trajectory_examples`` stay on CPU (the
        stream's normalizer stats and the raw instances are CPU). ``batched_rollout`` returns the
        trajectory on CPU.
        """
        instances = [family.sample_params() for _ in range(self.n_instances)]
        device = next(self.model.parameters()).device
        # One uniform start per instance; the seed also sizes the collated batch (batched_field_input
        # overwrites the point columns with the rollout state each step, so the seed value is arbitrary).
        z0 = torch.stack([family.sample_domain(inst, 1)[0] for inst in instances])  # [B, d]
        items = [
            normalize_input(family.model_input(inst, z), family.transform, self.normalizer)
            for inst, z in zip(instances, z0)
        ]
        # Ship the collated batch to the model's device for the rollout (the field runs the model);
        # the batch is a plain dict of tensors for every family, so move it entry-wise.
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in family.collate_fn(items).items()}
        field = self.model.batched_field(family, batch)
        traj = batched_rollout(family, batch, field, ALGORITHMS[self.algo](self.h), self.n_steps, z0.to(device))
        return trajectory_examples(family, instances, traj, self.n_instances * self.points_per_instance)

    def _raw_stream(self, family):
        self._epoch += 1
        if self._buffer is None or self._epoch % self.refresh_every == 0:
            self._buffer = self._rollout_buffer(family)
        while True:
            yield from self._buffer


class ExpertOperatorStream(OperatorStream):
    """Infinite stream of expert demonstrations: roll out the *true* operator, sample along it + z*.

    Unlike ``OnPolicyOperatorStream`` (which rolls out the *learned* field and so holds a live model),
    the expert rolls out the ground-truth operator with a converging algorithm, so it holds only the
    picklable ``family_factory`` and runs on ``DataLoader`` workers (``num_workers > 0``) -- essential
    because the rollout is ``n_steps`` operator solves (each an expensive route-choice solve for
    traffic).

    Like ``OnPolicyOperatorStream`` it owns a buffer refreshed every ``refresh_every`` epochs (its
    ``_raw_stream``): one ``_expert_batch`` solve per refresh window, cycled for the rest of the window,
    so a solved chunk trains the model across epochs instead of being regenerated-and-discarded every
    time the epoch (bounded by ``Trainer(limit_train_batches=...)``) ends mid-chunk. This keeps the
    logged operator-eval budget equal to the distinct solves the model actually trains on, rather than
    ``n_workers * epochs`` re-solves. Being model-free the expert distribution is stationary, so the
    refresh exists only to rotate in fresh instances for diversity (``_expert_batch`` draws new ones),
    not to track a moving field.

    Each refresh draws ``n_instances`` fresh instances and rolls them out **jointly** (one batched
    operator solve per step over the whole batch), then yields ordinary ``(model_input, operator)``
    examples at:

    - **every** state the rollout evaluated the operator at, when ``include_trajectory`` -- kept by
      ``RecordedField`` as one ``OperatorEvaluations`` per instance rather than re-solved, so a window costs
      ``n_instances * (n_steps + 1)`` evals and yields that many examples (~1 example per eval, where
      subsampling-and-re-solving yielded one per ``n_steps / points_per_instance``), and
    - the converged endpoint ``z*`` (the equilibrium solution), one per instance -- all ``n_instances``
      solved in a single batched operator call -- when ``include_solution``. That solve is the ``+ 1``
      above and is genuinely extra: ``traj[-1]`` is the one state the algorithm never queried.

    Both are plain operator examples, so they blend into the same regression MSE. The two ``include_*``
    gates let a solutions-only baseline select just the equilibria by config, not a rewrite.
    ``algo`` must be non-Jacobian (``consensus`` is excluded: ``jacrev`` does not compose through the
    analytic traffic operator); ``algo_kwargs`` overrides its extra hyperparameters (e.g. momentum
    ``beta``) picklably through the registry.

    ``solution_target`` switches the stream from *operator-field* demonstrations to *full-amortization*
    demonstrations for the solution-prediction baseline (``--amortization full``): each example becomes
    ``(model_input(inst, free_flow_time), z*)`` -- a **parameters-only** input (the free-flow start,
    carrying no query point that would leak the answer) regressed onto the **equilibrium** ``z*``,
    rather than a domain point regressed onto its operator value. It requires ``include_trajectory``
    off (trajectory examples carry operator-value targets and cannot mix into a ``z*`` regression) and
    skips the endpoint residual solve (the target is ``z*`` itself, not the ~0 operator there).
    """

    def __init__(
        self,
        family_factory,
        normalizer,
        algo,
        h,
        n_steps,
        n_instances,
        refresh_every,
        algo_kwargs=None,
        include_trajectory=True,
        include_solution=True,
        solution_target=False,
    ):
        super().__init__(family_factory, normalizer)
        assert not (solution_target and include_trajectory), (
            "solution_target regresses z* (not operator values), so the operator-target trajectory "
            "examples cannot be mixed in -- set include_trajectory=False"
        )
        self.algo = algo
        self.h = h
        self.n_steps = n_steps
        self.n_instances = n_instances
        self.refresh_every = refresh_every
        self.algo_kwargs = algo_kwargs or {}
        self.include_trajectory = include_trajectory
        self.include_solution = include_solution
        self.solution_target = solution_target
        self._buffer = None
        self._epoch = -1

    def _expert_batch(self, family):
        """Roll out a fresh batch of instances on the true operator; return ``(groups, examples)``.

        ``groups`` hold the operator-target demonstrations -- one ``OperatorEvaluations`` per instance, carrying
        the states the rollout evaluated plus the equilibrium endpoint. ``examples`` hold the ``z*``-target
        ones, which are not operator values and so are not groups. Exactly one of the two is populated by
        each of the configurations in use (``partial`` -> groups, ``full`` -> examples).
        """
        instances = [family.sample_params() for _ in range(self.n_instances)]
        z0 = torch.stack([family.sample_domain(inst, 1)[0] for inst in instances])  # [B, d]
        items = [
            normalize_input(family.model_input(inst, z), family.transform, self.normalizer)
            for inst, z in zip(instances, z0)
        ]
        # All-CPU (no model), so no device move; params_from_batch points the operator at the batch's
        # real-unit attrs -- the same batched analytic path the validation FieldRolloutCallback uses.
        batch = family.collate_fn(items)
        params = family.params_from_batch(batch)
        algo = ALGORITHMS[self.algo](self.h, **self.algo_kwargs)  # fresh instance per rollout
        # Keep the operator values the rollout consumes when they are the targets we want. Under
        # solution_target they are not (z* is), so the plain field is used there and nothing is retained.
        recorder = None if self.solution_target else RecordedField(family, params)
        field = recorder if recorder is not None else lambda z: family.operator(params, z)
        traj = batched_rollout(family, batch, field, algo, self.n_steps, z0)  # [T+1, B, d]

        groups, examples = [], []
        if self.include_trajectory:
            groups = recorder.groups(instances)
        if self.include_solution:
            z_star = traj[-1]  # [B, d] -- each instance's converged endpoint is its equilibrium solution
            if self.solution_target:
                # Full-amortization target: regress z* directly from a parameters-only input. The
                # free-flow-time start fills the query column (see model_input), carrying no point
                # that would leak z*; no residual solve -- the target is z* itself.
                examples += [(family.model_input(inst, inst.free_flow_time), z_star[b]) for b, inst in enumerate(instances)]
            else:
                # Operator-field target: solve the operator (~0 there) for all B distinct endpoints in
                # ONE batched call -- batching the B instances together, as the rollout does. This is the
                # one solve the recording cannot supply: traj[-1] is the state the algorithm never queried.
                with torch.no_grad():
                    residuals, diagonals = family.operator_and_preconditioner(params, z_star)
                groups = _with_endpoints(instances, groups, z_star, residuals, diagonals)
        return groups, examples

    def _raw_stream(self, family):
        # Refresh the buffer at epoch start (once per refresh_every epochs), then cycle it -- one
        # expensive rollout solve per window, reused across epochs, so operator evals track the
        # distinct solves the model trains on rather than a per-epoch regenerate-and-discard. The
        # per-epoch length is bounded by Trainer(limit_train_batches=...), not buffer exhaustion.
        # State persists across epochs: num_workers=0 keeps the object in-process; num_workers>0 runs
        # with persistent_workers=True (see the training scripts), so each worker's replica survives.
        self._epoch += 1
        if self._buffer is None or self._epoch % self.refresh_every == 0:
            self._buffer = self._expert_batch(family)
        groups, examples = self._buffer
        # Every group holds the same number of points (one rollout, one endpoint solve), so a flat
        # permutation over group x point decodes by divmod -- interleaving the instances so a minibatch is
        # a mix rather than one instance's consecutive states, as the old shuffled example list was.
        width = len(groups[0].points) if groups else 0
        while True:
            for flat in torch.randperm(len(groups) * width).tolist():
                index, point = divmod(flat, width)
                yield from operator_examples(family, groups[index], index=index, order=[point])
            yield from examples
