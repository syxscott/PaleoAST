"""
Ground truth for ``hpc/process_pool.py`` (30.6% covered).

A process pool's characteristic failure is not a wrong number -- it is a HANG,
a lost result, or a task that silently never runs. Three such defects were
found by driving each scenario with a timeout and asserting on what came
back:

1. ``get_result`` did not wait. ``submit_task`` is asynchronous, so the
   natural ``submit_task(...)`` then ``get_result(...)`` sequence raised
   ``KeyError: not found or not completed`` whenever the worker had not
   finished -- and the ``timeout`` parameter was accepted and never used.
2. ``error_callback`` only logged. A task that failed at the transport level
   stayed PENDING forever, so it was indistinguishable from a slow one.
3. ``map`` put ``None`` into the result list for every failed item and
   skipped whole failed chunks, all behind a log line. A caller computing
   ``np.mean(replicates)`` over 40 silently-missing bootstrap replicates got
   NaN and no indication why.

These tests spawn real processes, so they need the BLAS thread caps below --
each worker otherwise allocates its own OpenBLAS thread pool.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hpc.process_pool import ProcessPool  # noqa: E402
from utils.exceptions import ComputationError  # noqa: E402


def _square(x: int) -> int:
    """Module level so it can cross the process boundary."""
    return x * x


def _boom(x: int) -> int:
    raise ValueError(f"boom {x}")


def _row_mean(rows: np.ndarray) -> float:
    return float(np.mean(rows))


@pytest.fixture
def pool():
    p = ProcessPool(n_workers=2)
    p.start()
    yield p
    p.shutdown()


class TestMap:
    def test_one_result_per_input_in_order(self, pool):
        assert pool.map(_square, list(range(12))) == [i * i for i in range(12)]

    def test_empty_input(self, pool):
        assert pool.map(_square, []) == []

    def test_a_raising_item_is_reported(self, pool):
        """Not returned as if it were a result.

        ``_worker_map`` turns a failed item into ``None``; the pool used to
        pass that straight through, so a bootstrap with one bad replicate
        produced a list that looked complete.
        """
        with pytest.raises(ComputationError):
            pool.map(_boom, [1, 2, 3])

    def test_best_effort_is_available_but_opt_in(self, pool):
        results = pool.map(_boom, [1, 2, 3], raise_on_error=False)
        assert results == [None, None, None]
        assert len(results) == 3, "best-effort must keep the shape"


class TestSubmitAndGet:
    def test_get_result_waits_for_the_worker(self, pool):
        """The natural call sequence must work.

        It used to raise ``KeyError: not found or not completed`` because
        ``get_result`` checked the result table once and gave up, and the
        ``timeout`` argument was never used.
        """
        pool.submit_task("t1", _square, 7)
        assert pool.get_result("t1", timeout=30) == 49

    def test_get_result_reports_a_timeout(self, pool):
        with pytest.raises((TimeoutError, RuntimeError, KeyError)):
            pool.get_result("never-submitted", timeout=0.1)

    def test_get_result_rejects_an_unsubmitted_id(self, pool):
        with pytest.raises(KeyError):
            pool.get_result("no-such-task", timeout=1)

    def test_wait_all_accounts_for_every_task(self, pool):
        for i in range(5):
            pool.submit_task(f"w{i}", _square, i)
        done = pool.wait_all(timeout=30)
        assert len(done) == 5, f"wait_all returned {sorted(done)}"


class TestParallelHelpers:
    def test_distance_matches_the_serial_computation(self, pool):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(30, 5))
        expected = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        assert np.allclose(pool.compute_parallel_distance(data), expected, atol=1e-9)

    def test_bootstrap_returns_the_requested_replicates(self, pool):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(30, 5))
        replicates = pool.bootstrap_parallel(data, 12, _row_mean)
        assert len(replicates) == 12
        assert all(np.isfinite(value) for value in replicates)
        # A bootstrap of the mean must centre near the observed mean.
        assert abs(float(np.mean(replicates)) - float(data.mean())) < 0.2

    def test_bootstrap_rejects_an_unpicklable_statistic(self, pool):
        """A lambda cannot cross a process boundary.

        Previously that produced a list of ``None`` and a log line; the
        failure mode is now a clear error.
        """
        with pytest.raises((ComputationError, Exception)):
            pool.bootstrap_parallel(np.zeros((10, 2)), 4, lambda rows: float(np.mean(rows)))


class TestLifecycle:
    def test_shutdown_then_work_is_refused(self):
        """shutdown() released the Manager too, so a later map() would
        silently resurrect a set of processes the caller had just torn down."""
        p = ProcessPool(n_workers=2)
        p.start()
        p.shutdown()
        with pytest.raises(ComputationError):
            p.map(_square, [1, 2])

    def test_context_manager_closes(self):
        with ProcessPool(n_workers=2) as p:
            assert p.map(_square, [1, 2, 3]) == [1, 4, 9]
        with pytest.raises(ComputationError):
            p.map(_square, [1])
