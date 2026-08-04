"""Shared test helpers: an operator-call counter and a stream-draining shorthand.

``CountingFamily`` is the in-process stand-in for ``operator_count.SharedCounter``, which only the traffic
family accepts -- every budget assertion in ``test_caching`` / ``test_expert_recording`` is about how many
point-evaluations a source spends, so it needs to count them for the flat families too.
"""

import itertools


class CountingFamily:
    """Wraps a family, counting the point-evaluations its operator is asked for.

    Both entry points are counted, and neither double-counts: whichever the caller enters, the delegate
    call lands on the *wrapped* family, whose internal hop between ``operator`` and ``operator_and_preconditioner``
    never re-enters this wrapper. (Families differ in which way that hop goes -- ``pume_traffic``'s
    ``operator`` delegates to ``operator_and_preconditioner``, the flat families' default does the reverse -- so
    counting only one of them would silently miss whole code paths.)
    """

    def __init__(self, family):
        self._family = family
        self.n_evaluations = 0

    def __getattr__(self, name):
        return getattr(self._family, name)

    def operator(self, params, points):
        self.n_evaluations += len(points)
        return self._family.operator(params, points)

    def operator_and_preconditioner(self, params, points):
        self.n_evaluations += len(points)
        return self._family.operator_and_preconditioner(params, points)


def take(stream, n):
    """The first ``n`` examples of an (infinite) stream."""
    return list(itertools.islice(iter(stream), n))
