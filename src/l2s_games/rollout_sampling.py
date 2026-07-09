"""On-policy training points: roll out the *learned* field, sample the operator along it.

The uniform pipeline (``data.build_dataset``) trains on points drawn uniformly over the
domain. ``OnPolicyOperatorStream`` (below) instead samples the points a solver *actually
visits*: it rolls out the current model's field with a pluggable algorithm from random starts
and evaluates the ground-truth operator at the visited states, so the model is trained on the
state distribution its own field induces. The field changes as it trains, so the stream
regenerates its buffer periodically -- one refreshing stream per training loader.

Everything runs through the existing family seams -- ``model_input`` / ``transform`` /
``collate_fn`` (conditioning), ``batched_field`` (the batched learned field, real units),
``project`` (off the collated batch), ``simulate`` + ``ALGORITHMS`` (the rollout), and
``operator`` (the per-instance target) -- so it is blind to the concrete representation and
works for both the flat (RPS/matrix) and graph (traffic) families.
"""

import torch

from l2s_games.algorithms import ALGORITHMS
from l2s_games.data import OperatorStream, normalize_input
from l2s_games.dynamics import simulate


class OnPolicyOperatorStream(OperatorStream):
    """Infinite stream of on-policy rollout points, refreshed from the *current* model each epoch.

    Owns its rollout and its buffer: at the start of each epoch (its ``__iter__``, cadenced by
    ``refresh_every``) it rolls out the learned field with ``algo`` and refills ``self._buffer`` via
    ``_rollout_buffer``, then cycles that buffer for the rest of the epoch. The per-epoch length is
    bounded by ``Trainer(limit_train_batches=...)``, not by buffer exhaustion, so the epoch always has
    the same number of batches (which Lightning fixes from epoch 0). Rolling out at epoch start
    mirrors the old ``on_train_epoch_start`` timing; ``simulate`` detaches every iterate, so no graph
    leaks into data loading.

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

        Rolls out ``-learned_field`` (descent, matching ``RolloutCallback``) with
        ``ALGORITHMS[self.algo]`` from one uniform start per instance, then samples ``buffer_size`` of
        the visited states. The target at each chosen point is the analytic operator, unnegated -- the
        model regresses the operator itself, exactly as the uniform pipeline does. Uniform coverage
        (exploration / cold-start, while the field is near-random) is supplied by the sibling
        ``UniformSampledOperatorStream``, so this stream is purely on-policy.

        Representation-agnostic: it drives the same family seams the validation rollout uses
        (``model_input`` -> ``transform`` -> ``collate_fn`` for the batch, ``batched_field`` for the
        learned field, ``params_from_batch`` / ``project`` off the collated batch, ``operator`` for the
        targets), so it works for flat games and graph instances alike. Targets are solved once per
        instance over its picked points (one route-choice solve per instance for traffic), then the
        assembled examples are shuffled so a minibatch is not dominated by a single instance.
        """
        # The model (hence the learned field) lives on this device; the rollout must run there, while
        # featurization above and the operator solve below stay on CPU (the stream's normalizer stats
        # and the raw instances are CPU, matching every other stream). See the two moves below.
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
        project = lambda z: family.project(batch, z)
        # simulate() detaches every iterate, so the trajectory is a set of sample locations with no
        # graph; consensus manages its own autograd internally (caller runs with inference_mode off).
        traj = simulate(lambda z: -field(z), ALGORITHMS[self.algo](self.h), z0.to(device), self.n_steps, project=project)  # [T+1,B,d]

        # Back to CPU (a single ~20MB copy per refresh) so the per-instance operator solve + model_input
        # below run on CPU against the CPU raw instances -- the operator path the uniform stream uses.
        traj = traj.cpu()
        # Flatten the trajectory, tracking each row's instance so targets + model_input use its params.
        n_steps, n_inst = traj.shape[0], traj.shape[1]
        flat_points = traj.reshape(n_steps * n_inst, -1)  # [(T+1)*B, d]
        flat_inst = torch.arange(n_inst).repeat(n_steps)  # instance index per flat row
        pick = torch.randint(0, flat_points.shape[0], (self.buffer_size,))
        points, picked_inst = flat_points[pick], flat_inst[pick]

        # Solve the operator per instance over its picked points, then assemble + shuffle the examples.
        examples = []
        with torch.no_grad():
            for b, inst in enumerate(self.instances):
                pts = points[picked_inst == b]  # [m, d]
                if pts.shape[0] == 0:
                    continue
                targets = family.operator(inst, pts)
                examples += [(family.model_input(inst, pts[j]), targets[j]) for j in range(pts.shape[0])]
        return [examples[i] for i in torch.randperm(len(examples))]

    def _raw_stream(self, family):
        self._epoch += 1
        if self._buffer is None or self._epoch % self.refresh_every == 0:
            self._buffer = self._rollout_buffer(family)
        while True:
            yield from self._buffer
