"""
Tests for the HPC layer (``hpc/task_scheduler.py``), previously 30% covered.

A task scheduler is the one component where a mistake is not a wrong number
but a wrong *time*: a task that silently never runs looks exactly like a task
that is merely slow. These tests pin the three failure modes that were live:

1. ``add_task(func, task_id, dependencies, priority, *args, **kwargs)`` put
   the options before ``*args`` and made them positional, so the natural
   ``add_task(add, 2, 3)`` bound ``2`` to ``task_id`` and ``3`` to
   ``dependencies``, then died inside ``set(3)``. The variadic-args feature
   was unreachable.
2. ``get_result`` on an id it had never seen waited out the whole timeout and
   then raised ``TimeoutError``, which reads like "the task is slow" rather
   than "you passed a wrong id".
3. ``add_task`` after ``shutdown()`` returned a plausible task id for work
   that could never run.
"""

from __future__ import annotations

import time

import pytest

from hpc.task_scheduler import TaskPriority, TaskScheduler


def _add(a: int, b: int) -> int:
    return a + b


def _boom() -> None:
    raise ValueError("intentional failure")


def _sleep(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


@pytest.fixture
def scheduler():
    s = TaskScheduler()
    s.start()
    yield s
    s.shutdown(wait=False)


class TestPositionalArgumentsReachTheTarget:
    def test_positional_args_are_forwarded(self, scheduler):
        """The regression: options used to swallow the caller's arguments."""
        task = scheduler.add_task(_add, 2, 3)
        assert scheduler.get_result(task, timeout=10) == 5

    def test_several_positional_args(self, scheduler):
        task = scheduler.add_task(_add, 10, 20)
        assert scheduler.get_result(task, timeout=10) == 30

    def test_a_float_positional_arg(self, scheduler):
        task = scheduler.add_task(_sleep, 0.01)
        assert scheduler.get_result(task, timeout=10) == pytest.approx(0.01)

    def test_keyword_args_still_reach_the_target(self, scheduler):
        task = scheduler.add_task(_add, a=7, b=8)
        assert scheduler.get_result(task, timeout=10) == 15

    def test_options_are_still_keyword_only(self, scheduler):
        task = scheduler.add_task(_add, 1, 1, priority=TaskPriority.HIGH)
        assert scheduler.get_result(task, timeout=10) == 2

    def test_explicit_task_id_is_honoured(self, scheduler):
        task = scheduler.add_task(_add, 1, 1, task_id="named-task")
        assert task == "named-task"

    def test_duplicate_task_id_is_rejected(self, scheduler):
        scheduler.add_task(_add, 1, 1, task_id="dup")
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(_add, 2, 2, task_id="dup")


class TestDependencies:
    def test_a_chain_runs_in_order(self, scheduler):
        first = scheduler.add_task(_sleep, 0.05)
        second = scheduler.add_task(_add, 1, 1, dependencies=[first])
        third = scheduler.add_task(_add, 5, 5, dependencies=[second])
        assert scheduler.get_result(second, timeout=15) == 2
        assert scheduler.get_result(third, timeout=15) == 10

    def test_missing_dependency_is_rejected_immediately(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            scheduler.add_task(_add, 1, 1, dependencies=["no_such_task"])

    def test_a_dependency_on_a_finished_task_is_allowed(self, scheduler):
        done = scheduler.add_task(_add, 1, 1)
        scheduler.get_result(done, timeout=10)
        later = scheduler.add_task(_add, 2, 2, dependencies=[done])
        assert scheduler.get_result(later, timeout=10) == 4


class TestFailuresAreSurfaced:
    def test_a_failing_task_raises(self, scheduler):
        task = scheduler.add_task(_boom)
        with pytest.raises(RuntimeError, match="intentional failure"):
            scheduler.get_result(task, timeout=10)

    def test_a_failure_does_not_poison_its_neighbours(self, scheduler):
        good = scheduler.add_task(_add, 1, 2)
        scheduler.add_task(_boom)
        assert scheduler.get_result(good, timeout=10) == 3

    def test_status_reports_the_failure(self, scheduler):
        good = scheduler.add_task(_add, 1, 1)
        bad = scheduler.add_task(_boom)
        for task in (good, bad):
            try:
                scheduler.get_result(task, timeout=10)
            except Exception:
                pass
        status = scheduler.get_status()
        assert status["total_tasks"] == 2
        assert status["failed"] == 1
        assert status["completed"] == 1


class TestUnknownAndUnrunnableTasks:
    def test_unknown_id_raises_at_once(self, scheduler):
        """Not a TimeoutError after the full wait -- that reads as "slow"."""
        start = time.time()
        with pytest.raises(KeyError, match="Unknown task id"):
            scheduler.get_result("never-existed", timeout=10)
        assert time.time() - start < 1.0, "waited for the timeout on an unknown id"

    def test_add_after_shutdown_is_refused(self):
        s = TaskScheduler()
        s.start()
        task = s.add_task(_add, 1, 1)
        s.get_result(task, timeout=10)
        s.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            s.add_task(_add, 2, 2)

    def test_get_result_without_start_says_so(self):
        s = TaskScheduler()
        task = s.add_task(_add, 1, 1)
        with pytest.raises(RuntimeError, match="never started"):
            s.get_result(task, timeout=5)


class TestPrioritiesAndStatus:
    @pytest.mark.parametrize(
        "priority",
        [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH, TaskPriority.CRITICAL],
    )
    def test_every_priority_completes(self, scheduler, priority):
        task = scheduler.add_task(_sleep, 0.01, priority=priority)
        assert scheduler.get_result(task, timeout=10) is not None

    def test_status_counts_add_up(self, scheduler):
        for i in range(5):
            scheduler.add_task(_add, i, i)
        scheduler.wait_all(timeout=15)
        status = scheduler.get_status()
        assert status["total_tasks"] == 5
        assert status["completed"] == 5
        assert status["failed"] == 0
