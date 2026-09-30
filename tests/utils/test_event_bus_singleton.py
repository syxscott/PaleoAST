# =============================================================================
# FILE: tests/utils/test_event_bus_singleton.py
# =============================================================================
"""
Regression tests for the EventBus singleton (utils/event_bus.py).

The bug these guard: ``EventBus.__new__`` returned a raw ``super().__new__(cls)``
whose C++ side was not yet initialised, and ``__init__`` then began with
``if self._initialized:``. Touching an attribute on a QObject before
``QObject.__init__`` has run sends PyQt6 into unbounded recursion. The process
dies with a native stack overflow (exit code 127 on Windows), not a catchable
``RecursionError`` -- so a plain in-process assertion cannot report it: pytest
is simply killed, and everything after looks like a collection error.

``test_constructing_the_singleton_does_not_crash_the_process`` therefore runs
the construction in a *subprocess* and inspects its exit status. That is the
only way to assert "this does not take the interpreter down".
"""

import subprocess
import sys

import pytest

pytest.importorskip("PyQt6", reason="EventBus is a QObject; PyQt6 is required")

from utils.event_bus import EventBus, get_event_bus


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """Every test starts and ends with a fresh singleton."""
    EventBus.reset_instance()
    yield
    EventBus.reset_instance()


class TestEventBusConstruction:
    def test_constructing_the_singleton_does_not_crash_the_process(self):
        """`EventBus()` must not overflow the stack.

        Run out-of-process: the failure mode is a hard native crash, so an
        in-process test would take pytest down with it and report nothing
        useful. Exit code 0 (and the sentinel on stdout) means it survived.
        """
        code = (
            "import faulthandler; faulthandler.enable()\n"
            "from utils.event_bus import EventBus\n"
            "bus = EventBus()\n"
            "assert bus is EventBus()\n"
            "print('SURVIVED')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"constructing EventBus() killed the interpreter "
            f"(exit {proc.returncode}).\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert "SURVIVED" in proc.stdout, proc.stdout

    def test_state_manager_can_be_used_after_a_bus_reset(self):
        """The real crash path: get_state_manager() -> set_data_matrix -> emit.

        ``models.state_manager.set_data_matrix`` publishes on the bus, so this
        is the call chain that took the interpreter down in production.
        """
        import numpy as np

        from models.data_matrix import DataMatrix
        from models.state_manager import get_state_manager

        EventBus.reset_instance()
        state = get_state_manager()
        matrix = DataMatrix(
            data=np.array([[1.0, 2.0], [3.0, 4.0]]),
            row_labels=["r1", "r2"],
            col_labels=["a", "b"],
        )
        state.set_data_matrix(matrix)
        assert state.data_matrix is matrix


class TestEventBusSingleton:
    def test_repeated_construction_returns_one_instance(self):
        assert EventBus() is EventBus() is EventBus.get_instance() is get_event_bus()

    def test_reset_instance_yields_a_new_object(self):
        first = EventBus()
        EventBus.reset_instance()
        second = EventBus()
        assert second is not first
        assert EventBus() is second

    def test_logger_is_attached_once(self):
        bus = EventBus()
        assert bus._logger is not None
        # A second construction must not replace the logger on the shared object.
        logger = bus._logger
        EventBus()
        assert bus._logger is logger

    def test_signals_can_be_connected_and_emitted(self):
        """A QObject whose base was built in __new__ must still deliver signals."""
        bus = EventBus()
        received = []
        bus.data_changed.connect(received.append)
        bus.emit_data_changed({"matrix": 1})
        assert received == [{"matrix": 1}]
