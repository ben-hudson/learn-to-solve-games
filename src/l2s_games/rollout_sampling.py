"""On-policy training points: roll out the *learned* field, sample the operator along it.

The uniform pipeline (``data.build_dataset``) trains on points drawn uniformly over the
domain. ``OnPolicyOperatorStream`` (below) instead samples the points a solver *actually
visits*: it rolls out the current model's field with a pluggable algorithm from random starts
and evaluates the ground-truth operator at the visited states, so the model is trained on the
state distribution its own field induces. The field changes as it trains, so the stream
regenerates its buffer periodically -- one refreshing stream per training loader.

Everything runs through the existing family seams -- ``model_input`` / ``transform`` /
``collate_fn`` (conditioning), ``batched_field`` (the batched learned field, real units),
``simulate`` + ``ALGORITHMS`` (the rollout), and ``operator`` (the target) -- so it is blind
to the concrete representation, though only the flat (RPS/matrix) path is wired for now.
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

    def _learned_field(self, family, params, z0):
        """The model's field over the instance batch, real units: ``v(Z)``, ``Z [B, d] -> [B, d]``.

        Builds the collated batch the same way training does -- ``normalize_input`` (transform +
        standardize ``feats``) then ``collate_fn`` -- and hands it to ``FieldModel.batched_field``,
        which splices the rollout state ``Z`` into the batch and de-standardizes the prediction. The
        seed point per instance is arbitrary (``batched_field_input`` overwrites the point columns
        with ``Z`` each step); we use ``z0`` for concreteness.
        """
        items = [
            normalize_input(family.model_input(p, z), family.transform, self.normalizer)
            for p, z in zip(params, z0)
        ]
        return self.model.batched_field(family, family.collate_fn(items))

    def _rollout_buffer(self, family):
        """Fresh raw ``(model_input, target)`` examples for the on-policy training buffer.

        Rolls out ``-learned_field`` (descent, matching ``RolloutCallback``) with
        ``ALGORITHMS[self.algo]`` from one random start per instance, then samples ``buffer_size`` of
        the visited states. The target at each chosen point is the analytic operator, unnegated -- the
        model regresses the operator itself, exactly as the uniform pipeline does. Uniform coverage
        (exploration / cold-start, while the field is near-random) is supplied by the sibling
        ``UniformSampledOperatorStream``, so this stream is purely on-policy. Returns ``buffer_size``
        raw ``({"point", "params"}, target)`` pairs, the shape ``OperatorStream`` normalizes.
        """
        params = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in self.instances])  # [B, n_params]
        z0 = torch.cat([family.sample_domain(p, 1) for p in self.instances], dim=0)  # [B, d]

        field = self._learned_field(family, params, z0)
        project = lambda z: family.project(params, z)
        # simulate() detaches every iterate, so the trajectory is a set of sample locations with no
        # graph; consensus manages its own autograd internally (caller runs with inference_mode off).
        traj = simulate(lambda z: -field(z), ALGORITHMS[self.algo](self.h), z0, self.n_steps, project=project)  # [T+1,B,d]

        # Flatten trajectory points with their per-instance params, subsample to buffer_size.
        flat_points = traj.reshape(-1, traj.shape[-1])  # [(T+1)*B, d]
        flat_params = params.unsqueeze(0).expand(traj.shape[0], -1, -1).reshape(-1, params.shape[-1])
        pick = torch.randint(0, flat_points.shape[0], (self.buffer_size,))
        points, point_params = flat_points[pick], flat_params[pick]

        with torch.no_grad():
            targets = family.operator(point_params, points)
        return [(family.model_input(point_params[i], points[i]), targets[i]) for i in range(points.shape[0])]

    def _raw_stream(self, family):
        self._epoch += 1
        if self._buffer is None or self._epoch % self.refresh_every == 0:
            self._buffer = self._rollout_buffer(family)
        while True:
            yield from self._buffer
