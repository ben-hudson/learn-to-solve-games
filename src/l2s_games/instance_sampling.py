"""Training points from a *fixed* instance set, with the domain points resampled every visit.

Sits between the two extremes the pipeline already has: ``UniformSampledOperatorStream`` (see
``data.py``) draws a **fresh** instance every step, so the model never sees a parametrization twice,
while the fixed ``OperatorDataset`` splits freeze both the instance *and* the point.
``FixedInstanceOperatorStream`` pins the *instances* to a set of ``n`` drawn once, but keeps the point
distribution identical to the uniform stream's (a fresh ``sample_domain`` draw per visit) -- the knob
for asking how many distinct parametrizations amortization actually needs, with the point sampling
held constant.

Each example is tagged with ``instance_index`` -- the position of its instance in the passed
``instances`` list, stable across epochs and workers. It is **diagnostics-only**: the backbones read
only ``feats`` and the graph structure, so the tag rides along on the collated batch (see the traffic
``collate_fn``, which stacks every tensor attribute) without ever reaching the network.
"""

import torch

from l2s_games.data import OperatorStream, examples_at_points

INSTANCE_INDEX = "instance_index"


class FixedInstanceOperatorStream(OperatorStream):
    """Infinite stream over a fixed instance set: fresh domain points each time an instance comes up.

    Every pass visits all ``instances`` in a reshuffled order, drawing ``points_per_instance`` fresh
    domain points per visit and solving the operator jointly for them (one solve per visit, as the
    uniform stream does per fresh instance).

    Model-free -- it holds only the picklable ``family_factory`` and the instances -- so it runs on
    ``DataLoader`` workers, which it must: each visit is an operator solve. Each worker holds a copy of
    the same instance list but seeds its own RNG (via ``lightning.seed_everything(seed, workers=True)``
    at the call site), so visit order and points differ per worker while the instance *set* -- the
    invariant this stream exists to impose -- is identical everywhere.
    """

    def __init__(self, family_factory, normalizer, instances, points_per_instance):
        super().__init__(family_factory, normalizer)
        self.instances = instances
        self.points_per_instance = points_per_instance

    def _raw_stream(self, family):
        while True:
            for index in torch.randperm(len(self.instances)).tolist():
                params = self.instances[index]
                points = family.sample_domain(params, self.points_per_instance)
                for raw, target in examples_at_points(family, params, points):
                    # Key access (not attribute) so this stays representation-agnostic: it tags a PyG
                    # Data (traffic) and a plain dict (flat games) alike.
                    raw[INSTANCE_INDEX] = torch.tensor(index)
                    yield raw, target
