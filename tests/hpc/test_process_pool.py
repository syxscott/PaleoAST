"""Regression tests for ``hpc/process_pool.py``.

The pool is exercised through a fake ``_pool`` rather than a real one: these
tests are about index bookkeeping and error attribution, both of which are
observable without spawning processes, and a real Pool would make them slow and
flaky on CI.

The defect pinned here: ``map()`` collected futures into a bare list that was
only appended to when ``apply_async`` succeeded, then attributed failures using
``enumerate(results)`` to index ``chunks``. One failed submission therefore
shifted every later chunk's failures by one position -- and misattributing
which input failed is precisely what that code was rewritten to prevent.
"""

from __future__ import annotations

import logging

import pytest

from hpc.process_pool import ProcessPool
from utils.exceptions import ComputationError

ITEMS = ["a0", "a1", "b0", "b1", "c0", "c1"]


def _bare_pool() -> ProcessPool:
    """A ProcessPool with no real Manager and no real worker Pool."""
    pool = ProcessPool.__new__(ProcessPool)
    pool._n_workers = 1
    pool._max_tasks = None
    pool._progress_callback = None
    pool._pool = None
    pool._manager = None
    pool._task_queue = None
    pool._result_queue = None
    pool._progress_queue = None
    pool._tasks = {}
    pool._results = {}
    pool._errors = {}
    pool._shutdown = False
    pool._logger = logging.getLogger("test.process_pool")
    return pool


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def get(self, timeout=None):
        return self._value


class _FakePool:
    """Applies ``func`` per item, with injectable submission and item failures.

    ``raise_on_call`` names the 0-based ``apply_async`` call that should blow
    up at submission time. ``none_in`` names the item whose result should come
    back as None, i.e. a worker that returned nothing for it.
    """

    def __init__(self, raise_on_call=None, none_in=()):
        self.raise_on_call = raise_on_call
        self.none_in = set(none_in)
        self.calls = 0
        self.chunks_seen = []

    def apply_async(self, _worker, args=(), **_kwargs):
        func, chunk = args
        index = self.calls
        self.calls += 1
        if index == self.raise_on_call:
            raise ValueError("Pool not running")
        self.chunks_seen.append(list(chunk))
        return _FakeFuture([None if item in self.none_in else func(item) for item in chunk])


def _double(item):
    return item.upper()


class TestMapAttributesFailuresToTheRightChunk:
    def test_failed_submission_does_not_shift_later_failures(self):
        """The chunk after a failed submission must still be attributed to itself.

        Chunk 1 fails to submit; chunk 2's worker returns None for "c1". The
        error must name "c1". Indexing by position in the submitted list
        instead names "b1" -- the item from the chunk that never ran.
        """
        pool = _bare_pool()
        pool._pool = _FakePool(raise_on_call=1, none_in={"c1"})

        with pytest.raises(ComputationError) as excinfo:
            pool.map(_double, ITEMS, chunk_size=2)

        message = str(excinfo.value)
        assert "c1" in message, f"failed item not attributed to its own chunk: {message}"
        assert "'b1'" not in message, f"attributed to a chunk that never ran: {message}"

    def test_survivors_are_returned_in_submission_order(self):
        pool = _bare_pool()
        pool._pool = _FakePool(raise_on_call=1, none_in={"c1"})

        with pytest.raises(ComputationError):
            pool.map(_double, ITEMS, chunk_size=2)

        # Best-effort mode: same run, no raise, and the survivor list is honest
        # about the hole rather than silently short.
        pool2 = _bare_pool()
        pool2._pool = _FakePool(raise_on_call=1, none_in={"c1"})
        out = pool2.map(_double, ITEMS, chunk_size=2, raise_on_error=False)
        assert out == ["A0", "A1", "C0", None]

    def test_all_good_run_returns_every_result(self):
        pool = _bare_pool()
        pool._pool = _FakePool()
        assert pool.map(_double, ITEMS, chunk_size=2) == ["A0", "A1", "B0", "B1", "C0", "C1"]

    def test_every_submitted_chunk_was_the_one_expected(self):
        pool = _bare_pool()
        fake = _FakePool()
        pool._pool = fake
        pool.map(_double, ITEMS, chunk_size=2)
        assert fake.chunks_seen == [["a0", "a1"], ["b0", "b1"], ["c0", "c1"]]

    def test_callback_skips_failed_items(self):
        seen = []
        pool = _bare_pool()
        pool._pool = _FakePool(none_in={"a1"})
        with pytest.raises(ComputationError):
            pool.map(_double, ITEMS, chunk_size=2, callback=seen.append)
        assert seen == ["A0", "B0", "B1", "C0", "C1"]


class TestMapRejectsUseAfterShutdown:
    def test_map_after_shutdown_raises_instead_of_resurrecting(self):
        pool = _bare_pool()
        pool._shutdown = True
        pool._pool = _FakePool()
        with pytest.raises(ComputationError, match="shut down"):
            pool.map(_double, ITEMS, chunk_size=2)

    def test_empty_items_returns_empty_without_touching_the_pool(self):
        pool = _bare_pool()
        pool._pool = _FakePool(raise_on_call=0)  # would raise if called
        assert pool.map(_double, [], chunk_size=2) == []
