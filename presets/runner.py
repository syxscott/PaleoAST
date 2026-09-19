# =============================================================================
# FILE: presets/runner.py
# =============================================================================
"""
Sequential run queue with guards and a JSON run manifest.

:class:`RunQueue` is a Qt-free state machine: items move
``pending -> running -> ok | error | skipped``.  The queue itself never
runs anything -- an *executor* callback is invoked per item and must call
its ``finish(status, error)`` callback exactly once, synchronously or from
an async completion handler (the UI wires this to the analysis callbacks).
Guard callbacks can veto an item before execution, mirroring the
Overwrite/Cancel run guards of the reference GUIs.

After a pass, :func:`build_manifest` produces a machine-readable record of
what ran with which parameters (surface-morphometrics-gui config-snapshot
convention) and :func:`write_manifest` stores it atomically.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .schema import atomic_write_json

logger = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
OK = "ok"
ERROR = "error"
SKIPPED = "skipped"

FinishCallback = Callable[[str, str | None], None]
Executor = Callable[["RunQueueItem", FinishCallback], None]
Guard = Callable[["RunQueueItem"], str | None]


@dataclass
class RunQueueItem:
    """One queued analysis run."""

    analysis_id: str
    params: dict[str, Any] = field(default_factory=dict)
    preset_name: str = ""
    status: str = PENDING
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def label(self) -> str:
        return self.preset_name or self.analysis_id


class RunQueue:
    """Sequential queue: guards, one-running-at-a-time, status callbacks."""

    def __init__(
        self,
        guard: Guard | None = None,
        executor: Executor | None = None,
        on_change: Callable[[RunQueueItem], None] | None = None,
    ) -> None:
        self._guard = guard or (lambda item: None)
        self._executor = executor
        self._on_change = on_change or (lambda item: None)
        self._items: list[RunQueueItem] = []

    # ------------------------------------------------------------------
    def add(self, item: RunQueueItem) -> RunQueueItem:
        self._items.append(item)
        return item

    def remove(self, index: int) -> RunQueueItem:
        return self._items.pop(index)

    @property
    def items(self) -> list[RunQueueItem]:
        return list(self._items)

    def clear_results(self) -> None:
        """Reset every non-pending item so the pass can be re-run."""
        for item in self._items:
            if item.status != PENDING:
                item.status = PENDING
                item.error = None
                item.started_at = item.finished_at = None
                self._on_change(item)

    def is_busy(self) -> bool:
        return any(i.status == RUNNING for i in self._items)

    def is_done(self) -> bool:
        return all(i.status in (OK, ERROR, SKIPPED) for i in self._items)

    def _notify(self, item: RunQueueItem) -> None:
        self._on_change(item)

    # ------------------------------------------------------------------
    def run_next(self) -> RunQueueItem | None:
        """
        Start the first pending item (after the guard check).

        Returns the item that was started (already finished for a
        synchronous executor or a guard skip), or ``None`` when nothing is
        pending.  With an async executor, call :meth:`run_next` again from
        the finish handler to advance.
        """
        if self.is_busy():
            raise RuntimeError("RunQueue: an item is still running; cannot start another")
        item = next((i for i in self._items if i.status == PENDING), None)
        if item is None:
            return None

        reason = self._guard(item)
        if reason is not None:
            item.status = SKIPPED
            item.error = reason
            logger.info("RunQueue: skipped %s (guard: %s)", item.label, reason)
            self._notify(item)
            return item

        item.status = RUNNING
        item.error = None
        item.started_at = time.time()
        self._notify(item)

        if self._executor is None:
            raise RuntimeError("RunQueue has no executor configured")

        finished = {"done": False}

        def finish(status: str, error: str | None = None) -> None:
            if finished["done"]:
                raise RuntimeError(f"RunQueue: finish() called twice for {item.label}")
            if status not in (OK, ERROR):
                raise ValueError(f"RunQueue: finish status must be '{OK}' or '{ERROR}', got {status!r}")
            finished["done"] = True
            item.status = status
            item.error = error
            item.finished_at = time.time()
            self._notify(item)

        try:
            self._executor(item, finish)
        except Exception as exc:
            if not finished["done"]:
                finish(ERROR, str(exc))
            else:
                logger.warning("RunQueue: executor raised after finishing %s: %s", item.label, exc)
        return item


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_manifest(items: list[RunQueueItem], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Machine-readable record of a queue pass (statuses, params, timings)."""
    records = []
    for item in items:
        records.append(
            {
                "analysis": item.analysis_id,
                "preset": item.preset_name,
                "params": dict(item.params),
                "status": item.status,
                "error": item.error,
                "started_at": item.started_at,
                "finished_at": item.finished_at,
                "duration_s": (
                    round(item.finished_at - item.started_at, 3)
                    if item.started_at and item.finished_at
                    else None
                ),
            }
        )
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "format": "paleoast_run_manifest_v1",
        "meta": dict(meta or {}),
        "summary": counts,
        "items": records,
    }


def write_manifest(path: str, items: list[RunQueueItem], meta: dict[str, Any] | None = None) -> str:
    """Atomically write the manifest JSON; returns the path."""
    atomic_write_json(path, build_manifest(items, meta))
    logger.info("Run manifest written: %s", path)
    return path
