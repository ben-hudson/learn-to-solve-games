"""Cumulative counters for ground-truth operator point-evaluations (the training budget).

The ground-truth operator is an expensive route-choice solve; training generates its data across
several sources, some running inside ``DataLoader`` worker processes (see ``rollout_sampling`` /
``data``). To log the cumulative point-evaluation budget, the family increments a counter on every
``operator`` call. Two flavours share one ``add(n)`` / ``value`` interface so ``operator`` stays
branch-free (it always calls ``self.operator_counter.add(...)``):

- ``LocalCounter`` -- an in-process accumulator, the cheap no-op default for families that are not
  counted (the main validation/collate family, the one-time dataset build, sandbox, tests).
- ``SharedCounter`` -- a process-safe counter shared with streaming workers. Only the families that
  generate training data carry one, so the total is scoped to the training budget.
"""

import multiprocessing as mp


class LocalCounter:
    """Plain in-process cumulative counter (the branch-free default; nothing reads it when unused)."""

    def __init__(self):
        self._value = 0

    def add(self, n):
        self._value += n

    @property
    def value(self):
        return self._value


class SharedCounter:
    """Process-safe cumulative counter shared across ``DataLoader`` workers (``spawn``-safe).

    Backed by a ``Manager`` ``Value`` + ``Lock`` proxy pair -- both picklable and reconnecting to the
    manager server across the ``spawn`` boundary -- so the counter can be baked into the picklable
    ``family_factory`` and shared by every streaming worker and the main process. ``add`` is atomic
    under the lock. ``__getstate__`` drops the unpicklable ``Manager`` (kept alive in the main
    process) so only the proxies are pickled into workers.
    """

    def __init__(self):
        self._manager = mp.Manager()
        self._value = self._manager.Value("q", 0)
        self._lock = self._manager.Lock()

    def add(self, n):
        with self._lock:
            self._value.value += n

    @property
    def value(self):
        return self._value.value

    def __getstate__(self):
        return {"_value": self._value, "_lock": self._lock}

    def __setstate__(self, state):
        self.__dict__.update(state)
