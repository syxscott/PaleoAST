"""Tests for presets/runner.py: RunQueue state machine and the run manifest."""

import json
import os

import pytest

from presets.runner import (
    ERROR,
    OK,
    PENDING,
    RUNNING,
    SKIPPED,
    RunQueue,
    RunQueueItem,
    build_manifest,
    write_manifest,
)


def _item(analysis="pca", **over):
    return RunQueueItem(analysis_id=analysis, params={"n_components": 2}, **over)


class TestItemBasics:
    def test_initial_state(self):
        item = _item(preset_name="Saved")
        assert item.status == PENDING
        assert item.label == "Saved"
        assert _item().label == "pca"


class TestRunQueueFlow:
    def test_ok_flow(self):
        seen = []
        queue = RunQueue(
            executor=lambda item, finish: finish(OK, None),
            on_change=lambda item: seen.append(item.status),
        )
        queue.add(_item())
        started = queue.run_next()
        assert started is not None and started.status == OK
        assert started.started_at is not None and started.finished_at is not None
        # notifications: RUNNING then OK
        assert seen == [RUNNING, OK]
        assert queue.is_done() and not queue.is_busy()
        assert queue.run_next() is None  # nothing pending

    def test_error_flow_carries_message(self):
        def executor(item, finish):
            finish(ERROR, "boom")

        queue = RunQueue(executor=executor)
        item = _item()
        queue.add(item)
        queue.run_next()
        assert item.status == ERROR
        assert item.error == "boom"

    def test_guard_skip(self):
        executed = []
        queue = RunQueue(
            guard=lambda item: "has_data",
            executor=lambda item, finish: executed.append(item),
        )
        item = _item()
        queue.add(item)
        started = queue.run_next()
        assert started.status == SKIPPED
        assert started.error == "has_data"
        assert not executed
        assert queue.is_done()

    def test_executor_exception_finishes_error(self):
        def boom(item, finish):
            raise ValueError("exploded")

        queue = RunQueue(executor=boom)
        item = _item()
        queue.add(item)
        queue.run_next()
        assert item.status == ERROR
        assert "exploded" in item.error

    def test_executor_exception_after_finish_only_logs(self):
        def nasty(item, finish):
            finish(OK, None)
            raise ValueError("late")

        queue = RunQueue(executor=nasty)
        item = _item()
        queue.add(item)
        started = queue.run_next()  # must not raise
        assert started.status == OK

    def test_finish_twice_raises_when_called_later(self):
        captured = {}

        def executor(item, finish):
            captured["finish"] = finish  # async completion

        queue = RunQueue(executor=executor)
        queue.add(_item())
        queue.run_next()
        captured["finish"](OK, None)
        with pytest.raises(RuntimeError, match="finish\\(\\) called twice"):
            captured["finish"](OK, None)

    def test_finish_bad_status_recorded_as_error(self):
        # A synchronous bad-status raise is caught at the executor boundary
        # and re-recorded as an ERROR finish.
        queue = RunQueue(executor=lambda item, finish: finish(SKIPPED, None))
        item = _item()
        queue.add(item)
        queue.run_next()
        assert item.status == ERROR
        assert "status" in item.error

    def test_cannot_start_while_busy(self):
        captured = {}

        def async_executor(item, finish):
            captured["finish"] = finish  # never called yet

        queue = RunQueue(executor=async_executor)
        queue.add(_item())
        queue.add(_item())
        queue.run_next()
        assert queue.is_busy()
        with pytest.raises(RuntimeError, match="still running"):
            queue.run_next()
        # completing the async item unblocks the queue
        captured["finish"](OK, None)
        assert not queue.is_busy()
        queue.run_next()

    def test_remove_and_clear_results(self):
        queue = RunQueue(executor=lambda item, finish: finish(OK, None))
        queue.add(_item())
        queue.add(_item(analysis="pcoa"))
        queue.run_next()
        queue.run_next()
        assert queue.is_done()
        queue.clear_results()
        assert all(i.status == PENDING for i in queue.items)
        assert all(i.error is None for i in queue.items)
        queue.remove(0)
        assert [i.analysis_id for i in queue.items] == ["pcoa"]

    def test_no_executor_raises(self):
        queue = RunQueue()
        queue.add(_item())
        with pytest.raises(RuntimeError, match="no executor"):
            queue.run_next()


class TestManifest:
    def test_manifest_shape(self):
        items = [_item(preset_name="P1")]
        items[0].status = OK
        items[0].started_at = 100.0
        items[0].finished_at = 100.5
        manifest = build_manifest(items, meta={"app_version": "1.0.1"})
        assert manifest["format"] == "paleoast_run_manifest_v1"
        assert manifest["meta"]["app_version"] == "1.0.1"
        assert manifest["summary"] == {OK: 1}
        record = manifest["items"][0]
        assert record["analysis"] == "pca"
        assert record["preset"] == "P1"
        assert record["duration_s"] == pytest.approx(0.5)
        assert record["params"] == {"n_components": 2}

    def test_manifest_pending_has_no_duration(self):
        manifest = build_manifest([_item()])
        assert manifest["items"][0]["duration_s"] is None
        assert manifest["summary"] == {PENDING: 1}

    def test_write_manifest_atomic(self, tmp_path):
        path = str(tmp_path / "manifest.json")
        write_manifest(path, [_item()])
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["format"] == "paleoast_run_manifest_v1"
        assert not os.path.exists(path + ".tmp")
