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

Both share the same batched machinery (``batched_rollout`` + ``trajectory_examples``), differing only
in *which* batched field is rolled out. Everything runs through the existing family seams --
``model_input`` / ``transform`` / ``collate_fn`` (conditioning), ``batched_field`` (the batched
learned field, real units) or ``operator`` (the batched analytic field), ``params_from_batch`` /
``project`` (off the collated batch), ``simulate`` + ``ALGORITHMS`` (the rollout) -- so they are blind
to the concrete representation and work for both the flat (RPS/matrix) and graph (traffic) families.
"""

import torch

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import OperatorStream, examples_at_points, normalize_input
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


def trajectory_examples(family, instances, traj, n_points):
    """Raw ``(model_input, target)`` examples from ``n_points`` states subsampled across a rollout.

    Flattens the trajectory ``[T+1, B, d]`` over time and instances, tracking each row's instance so
    its params drive the target solve, subsamples ``n_points`` of the visited states, then solves the
    analytic operator per instance over its picked points (one solve per instance) via
    ``examples_at_points``. The assembled examples are shuffled so a minibatch is not dominated by a
    single instance.
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
    """Infinite stream of on-policy rollout points, refreshed from the *current* model each epoch.

    Owns its rollout and its buffer: at the start of each epoch (its ``__iter__``, cadenced by
    ``refresh_every``) it rolls out the learned field with ``algo`` and refills ``self._buffer`` via
    ``_rollout_buffer``, then cycles that buffer for the rest of the epoch. The per-epoch length is
    bounded by ``Trainer(limit_train_batches=...)``, not by buffer exhaustion, so the epoch always has
    the same number of batches (which Lightning fixes from epoch 0). Rolling out at epoch start
    mirrors the old ``on_train_epoch_start`` timing.

    Holds a **live** ``model`` reference (weights update in place, so it always rolls out the current
    field), which requires ``num_workers=0`` -- the model cannot be pickled to a worker process.
    """

    def __init__(self, family_factory, normalizer, model, instances, algo, h, n_steps, buffer_size, refresh_every):
        super().__init__(family_factory, normalizer)
        self.model = model
        self.instances = instances
        self.algo = algo
        self.h = h
        self.n_steps = n_steps
        self.buffer_size = buffer_size
        self.refresh_every = refresh_every
        self._buffer = None
        self._epoch = -1

    def _rollout_buffer(self, family):
        """Fresh raw ``(model_input, target)`` examples for the on-policy training buffer.

        Rolls out the *learned* batched field (on the model's device) from one uniform start per
        instance, then delegates the shared subsample + per-instance target solve to
        ``trajectory_examples``. The target is the analytic operator, unnegated -- the model regresses
        the operator itself, exactly as the uniform pipeline does. Uniform coverage (exploration /
        cold-start, while the field is near-random) is supplied by the sibling
        ``UniformSampledOperatorStream``, so this stream is purely on-policy.

        The model (hence the learned field) lives on this device; the rollout must run there, while
        featurization here and the operator solve inside ``trajectory_examples`` stay on CPU (the
        stream's normalizer stats and the raw instances are CPU). ``batched_rollout`` returns the
        trajectory on CPU.
        """
        device = next(self.model.parameters()).device
        # One uniform start per instance; the seed also sizes the collated batch (batched_field_input
        # overwrites the point columns with the rollout state each step, so the seed value is arbitrary).
        z0 = torch.stack([family.sample_domain(inst, 1)[0] for inst in self.instances])  # [B, d]
        items = [
            normalize_input(family.model_input(inst, z), family.transform, self.normalizer)
            for inst, z in zip(self.instances, z0)
        ]
        # Ship the collated batch to the model's device for the rollout (the field runs the model);
        # the batch is a plain dict of tensors for every family, so move it entry-wise.
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in family.collate_fn(items).items()}
        field = self.model.batched_field(family, batch)
        traj = batched_rollout(family, batch, field, ALGORITHMS[self.algo](self.h), self.n_steps, z0.to(device))
        return trajectory_examples(family, self.instances, traj, self.buffer_size)

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
    ``__iter__``): one ``_expert_batch`` solve per refresh window, cycled for the rest of the window,
    so a solved chunk trains the model across epochs instead of being regenerated-and-discarded every
    time the epoch (bounded by ``Trainer(limit_train_batches=...)``) ends mid-chunk. This keeps the
    logged operator-eval budget equal to the distinct solves the model actually trains on, rather than
    ``n_workers * epochs`` re-solves. Being model-free the expert distribution is stationary, so the
    refresh exists only to rotate in fresh instances for diversity (``_expert_batch`` draws new ones),
    not to track a moving field.

    Each refresh draws ``n_instances`` fresh instances and rolls them out **jointly** (one batched
    operator solve per step over the whole batch), then yields ordinary ``(model_input, operator)``
    examples at:

    - ``n_instances * points_per_instance`` states subsampled along the trajectory (the expert path),
      when ``include_trajectory``, and
    - the converged endpoint ``z*`` (the equilibrium solution), one per instance -- all ``n_instances``
      solved in a single batched operator call -- when ``include_solution``.

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
        points_per_instance,
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
        self.points_per_instance = points_per_instance
        self.refresh_every = refresh_every
        self.algo_kwargs = algo_kwargs or {}
        self.include_trajectory = include_trajectory
        self.include_solution = include_solution
        self.solution_target = solution_target
        self._buffer = None
        self._epoch = -1

    def _expert_batch(self, family):
        """Roll out a fresh batch of instances on the true operator; return trajectory + solution examples."""
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
        field = lambda z: family.operator(params, z)
        traj = batched_rollout(family, batch, field, algo, self.n_steps, z0)  # [T+1, B, d]

        examples = []
        if self.include_trajectory:
            examples += trajectory_examples(family, instances, traj, self.n_instances * self.points_per_instance)
        if self.include_solution:
            z_star = traj[-1]  # [B, d] -- each instance's converged endpoint is its equilibrium solution
            if self.solution_target:
                # Full-amortization target: regress z* directly from a parameters-only input. The
                # free-flow-time start fills the query column (see model_input), carrying no point
                # that would leak z*; no residual solve -- the target is z* itself.
                examples += [(family.model_input(inst, inst.free_flow_time), z_star[b]) for b, inst in enumerate(instances)]
            else:
                # Operator-field target: solve the operator (~0 there) for all B distinct endpoints in
                # ONE batched call -- batching the B instances together, as the rollout does.
                with torch.no_grad():
                    residuals = family.operator(params, z_star)
                examples += [(family.model_input(inst, z_star[b]), residuals[b]) for b, inst in enumerate(instances)]
        return examples

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
        while True:
            yield from self._buffer
