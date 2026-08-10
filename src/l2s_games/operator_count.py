"""Cumulative counters for ground-truth operator point-evaluations (the training budget).

The ground-truth operator is an expensive route-choice solve; training generates its data across
several sources, some running inside ``DataLoader`` worker processes (see ``rollout_sampling`` /
``data``). To log the cumulative point-evaluation budget, the family increments a counter on every
``operator`` call. Two flavours share one ``add(n)`` / ``value`` interface so ``operator`` stays
branch-free (it always calls ``self.operator_counter.add(...)``):

- ``LocalCounter`` -- an in-process accumulator, the cheap no-op default for families that are not
  counted (the main validation/collate family, the one-time dataset build, sandbox, tests).
The streaming path's process-safe counterpart, ``SharedCounter``, lives in ``streaming``: only a source that
generates data inside ``DataLoader`` workers needs to share a total across processes, and a cached source's
budget is ``len(train)``, known before training starts.
"""


class LocalCounter:
    """Plain in-process cumulative counter (the branch-free default; nothing reads it when unused)."""

    def __init__(self):
        self._value = 0

    def add(self, n):
        self._value += n

    @property
    def value(self):
        return self._value


