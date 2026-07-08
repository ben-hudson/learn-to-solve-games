"""On-policy training points: roll out the *learned* field, sample the operator along it.

The uniform pipeline (``data.build_dataset``) trains on points drawn uniformly over the
domain. This module instead samples the points a solver *actually visits*: it rolls out the
current model's field with a pluggable algorithm from random starts and evaluates the
ground-truth operator at the visited states, so the model is trained on the state
distribution its own field induces. The field changes as it trains, so a caller regenerates
these examples periodically (see ``callbacks.RolloutBufferCallback``).

Everything runs through the existing family seams -- ``model_input`` / ``transform`` /
``collate_fn`` (conditioning), ``batched_field`` (the batched learned field, real units),
``simulate`` + ``ALGORITHMS`` (the rollout), and ``operator`` (the target) -- so it is blind
to the concrete representation, though only the flat (RPS/matrix) path is wired for now.
"""

import torch

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate


def _batched_learned_field(model, family, normalizer, params, z0):
    """The model's field over the instance batch, real units: ``v(Z)``, ``Z [B, d] -> [B, d]``.

    Builds the collated batch the same way training does -- ``transform(model_input(...))``,
    standardize ``feats``, ``collate_fn`` -- then hands it to ``FieldModel.batched_field``,
    which splices the rollout state ``Z`` into the batch and de-standardizes the prediction.
    The seed point per instance is arbitrary (``batched_field_input`` overwrites the point
    columns with ``Z`` each step); we use ``z0`` for concreteness.
    """
    items = []
    for p, z in zip(params, z0):
        item = family.transform(family.model_input(p, z))
        item["feats"] = normalizer.input.transform(item["feats"])
        items.append(item)
    batch = family.collate_fn(items)
    return model.batched_field(family, batch)


def rollout_examples(model, family, normalizer, params_list, algo_name, h, n_steps, buffer_size, blend_uniform_frac):
    """Fresh raw ``(model_input, target)`` examples for the on-policy training buffer.

    Rolls out ``-learned_field`` (descent, matching ``EquilibriumRolloutCallback``) with
    ``ALGORITHMS[algo_name]`` from one random start per instance in ``params_list``, then
    mixes the visited states with a ``blend_uniform_frac`` share of uniform ``sample_domain``
    points (exploration + cold-start coverage while the field is still near-random). The
    target at each chosen point is the analytic operator, unnegated -- the model regresses
    the operator itself, exactly as the uniform pipeline does. Returns ``buffer_size`` raw
    ``({"point", "params"}, target)`` pairs, the shape ``FieldDataset.examples`` holds.
    """
    params = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in params_list])  # [B, n_params]
    z0 = torch.cat([family.sample_domain(p, 1) for p in params_list], dim=0)  # [B, d]

    field = _batched_learned_field(model, family, normalizer, params, z0)
    project = lambda z: family.project(params, z)
    # simulate() detaches every iterate, so the trajectory is a set of sample locations with no
    # graph; consensus manages its own autograd internally (caller runs with inference_mode off).
    traj = simulate(lambda z: -field(z), ALGORITHMS[algo_name](h), z0, n_steps, project=project)  # [T+1, B, d]

    n_onpolicy = round(buffer_size * (1.0 - blend_uniform_frac))
    n_uniform = buffer_size - n_onpolicy

    # On-policy: flatten trajectory points with their per-instance params, subsample to n_onpolicy.
    flat_points = traj.reshape(-1, traj.shape[-1])  # [(T+1)*B, d]
    flat_params = params.unsqueeze(0).expand(traj.shape[0], -1, -1).reshape(-1, params.shape[-1])
    pick = torch.randint(0, flat_points.shape[0], (n_onpolicy,))
    on_points, on_params = flat_points[pick], flat_params[pick]

    # Uniform blend: fresh uniform domain points, each tied to a random instance's params.
    inst = torch.randint(0, len(params_list), (n_uniform,))
    uni_params = params[inst]
    uni_points = family.sample_domain(params_list[0], n_uniform)  # params ignored by flat sample_domain

    all_params = torch.cat([on_params, uni_params], dim=0)
    all_points = torch.cat([on_points, uni_points], dim=0)
    with torch.no_grad():
        targets = family.operator(all_params, all_points)
    return [
        (family.model_input(all_params[i], all_points[i]), targets[i]) for i in range(all_points.shape[0])
    ]
