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


def pytest_collection_finish(session):
    """收集完成后恢复真实的 utils.event_bus。

    mock 必须在模块导入期存在 (models 的测试模块在收集期导入,
    其依赖的 utils 子模块需要无头 mock), 但 sys.modules 替换是
    进程级全局的——若跨测试目录残留, 后续依赖真实事件总线的
    测试 (如 spreadsheet 的 data_changed 连接) 会全部失效。
    PyQt6 可用时恢复真实模块; 不可用时保留 mock 维持无头兼容。
    """
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        return
    import sys as _sys
    import importlib

    _sys.modules.pop("utils.event_bus", None)
    importlib.invalidate_caches()
