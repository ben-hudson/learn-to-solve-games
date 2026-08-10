"""Training points from a *bounded* set of solved ``(instance, point)`` pairs, reused once spent.

The two sampling sources resample on every visit -- ``UniformSampledOperatorStream`` draws a fresh
instance (see ``data.py``), ``FixedInstanceOperatorStream`` pins the instances but redraws the points
(see ``instance_sampling.py``) -- so both spend ``steps_per_epoch * batch`` ground-truth operator
evaluations *per epoch, forever*. Their only budget knob is the epoch count, which ties the data budget
to the optimization budget and makes them incomparable to the rollout sources, which already bound their
spend with a per-window buffer (see ``rollout_sampling``).

``CachedOperatorStream`` gives the sampling sources the same bound: it solves ``n_instances`` groups per
``refresh_every``-epoch window and then cycles them, so the budget is

    n_workers * n_instances * points_per_instance * ceil(epochs / refresh_every)

-- a config number. It serves both sources from one class: pass ``instances`` and it is "fixed instances
with **frozen** points", pass none and it draws fresh instances per window. So under caching the two
differ only in whether the instance set rotates across windows.

Three properties worth knowing:

- **The window fills lazily.** The first pass solves a group, yields its examples, and keeps it,
  resuming across epoch boundaries until the quota is spent -- so there is no startup stall and early
  training sees exactly the uncached stream's fresh-sample distribution.
- **The cache holds ``OperatorEvaluations``s, not examples.** ``model_input`` clones the instance per point, so
  retaining examples would cost ``points_per_instance`` times as much memory (13 GB rather than 1.3 GB
  for a million cached traffic points); ``operator_examples`` rebuilds them at yield time.
- **One cache per worker.** PyTorch does not shard ``IterableDataset``s, so each of the ``n_workers``
  dataset replicas fills its own -- hence the ``n_workers`` factor above, matching how the expert and
  on-policy buffers are counted. The cache lives on the replica, so ``persistent_workers=True`` is
  required for it to survive epoch boundaries (the training scripts set it whenever workers are used);
  without it every epoch would rebuild and the budget would grow again. ``INSTANCE_INDEX`` is therefore
  replica-local, which is harmless: a batch is collated inside a single worker, so no batch ever mixes
  two workers' indices and the monotonicity constraint still groups same-instance points correctly.
"""

import torch

from l2s_games.data import operator_examples, sample_and_eval_operator
from l2s_games.streaming import OperatorStream


class CachedOperatorStream(OperatorStream):
    """Infinite stream over a bounded, reused set of solved ``(instance, point)`` pairs.

    ``instances`` pins the instance set (``n_instances`` follows from it); otherwise ``n_instances``
    fresh ones are drawn per refresh window. Model-free either way -- it holds only the picklable
    ``family_factory`` -- so it runs on ``DataLoader`` workers, which it must while filling: each group
    is an operator solve.

    Every pass over a filled cache reshuffles the instance order (as ``FixedInstanceOperatorStream``
    does) *and* the point order within each instance. The latter is free and varies which points the
    monotonicity constraint pairs, since pairs are matched by halves of an instance's points in the
    batch (see ``monotonicity.monotonicity_ratios``) -- no new solves, different pairs.
    """

    def __init__(
        self,
        family_factory,
        normalizer,
        points_per_instance,
        n_instances=None,
        instances=None,
        refresh_every=1,
    ):
        super().__init__(family_factory, normalizer)
        self.points_per_instance = points_per_instance
        self.instances = instances
        self.n_instances = len(instances) if instances is not None else n_instances
        self.refresh_every = refresh_every
        self._groups = None
        self._epoch = -1

    def _solve(self, family, index):
        """Solve one instance's points -- the ``index``-th pinned instance, or a fresh draw."""
        params = self.instances[index] if self.instances is not None else family.sample_params()
        return sample_and_eval_operator(family, params, self.points_per_instance)

    def _examples(self, family, index):
        """The cached group's examples, in a fresh point order."""
        group = self._groups[index]
        return operator_examples(family, group, index=index, order=torch.randperm(len(group.points)).tolist())

    def _raw_stream(self, family):
        # Refresh at epoch start (once per refresh_every epochs), mirroring ExpertOperatorStream's
        # cadence. Clearing rather than refilling here is what makes the fill lazy: the loop below
        # solves as it yields, and picks up where it left off when an epoch ends mid-fill (the epoch
        # length is bounded by Trainer(limit_train_batches=...), not by the cache).
        self._epoch += 1
        if self._groups is None or self._epoch % self.refresh_every == 0:
            self._groups = []
        while True:
            while len(self._groups) < self.n_instances:
                self._groups.append(self._solve(family, len(self._groups)))
                yield from self._examples(family, len(self._groups) - 1)
            for index in torch.randperm(self.n_instances).tolist():
                yield from self._examples(family, index)
