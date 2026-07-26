"""
Pytest configuration for models tests.

This conftest patches only the utils submodules that depend on PyQt6
(event_bus) so DataMatrix tests can run in a headless environment.
Real StateManager is left untouched so its tests work.
"""

from __future__ import annotations

import sys
from types import ModuleType

_PROJECT_ROOT = __import__("pathlib").Path(__file__).parent.parent.parent


def _patch_event_bus_only():
    """Patch only utils.event_bus (PyQt6 dep) without replacing state_manager."""
    mock_event_bus = ModuleType("utils.event_bus")

    class MockQObject:
        pass

    class MockSignal:
        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    mock_event_bus.QObject = MockQObject
    mock_event_bus.pyqtSignal = lambda: MockSignal

    # Provide get_event_bus returning a no-op singleton for headless tests.
    class _MockEventBus:
        def __getattr__(self, name):
            return lambda *a, **kw: None

    _instance = _MockEventBus()

    def get_event_bus():
        return _instance

    mock_event_bus.get_event_bus = get_event_bus

    sys.modules["utils.event_bus"] = mock_event_bus


_patch_event_bus_only()