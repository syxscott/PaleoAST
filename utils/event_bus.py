# =============================================================================
# FILE: utils/event_bus.py
# =============================================================================
"""
Event Bus for PaleoAST

Centralized pub/sub event system for decoupled communication between
components. Uses PyQt signals for thread-safe event dispatching.

Architecture:
    - EventBus singleton provides a single channel for all application events
    - Components publish events without knowing who consumes them
    - Components subscribe to events without knowing who publishes them
    - Thread-safe via Qt signal/slot mechanism

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import threading
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class EventBus(QObject):
    """
    Centralized event bus using PyQt signals.

    Provides thread-safe publish/subscribe for application-wide events.
    Implemented as a singleton to ensure a single channel across the app.
    """

    # Singleton instance
    _instance: "EventBus | None" = None
    _instance_lock = threading.Lock()

    # =========================================================================
    # Signals - Data Events
    # =========================================================================

    data_changed = pyqtSignal(object, name="data_changed")
    """Emitted when data matrix is loaded, changed, or cleared."""

    metadata_changed = pyqtSignal(str, int, object, name="metadata_changed")
    """Emitted when metadata changes. (scope, index, metadata_dict)"""

    undo_stack_changed = pyqtSignal(name="undo_stack_changed")
    """Emitted when undo/redo stack changes."""

    # =========================================================================
    # Signals - Analysis Events
    # =========================================================================

    analysis_started = pyqtSignal(str, name="analysis_started")
    """Emitted when an analysis begins. (analysis_name)"""

    analysis_completed = pyqtSignal(str, object, name="analysis_completed")
    """Emitted when an analysis completes. (analysis_name, result)"""

    analysis_failed = pyqtSignal(str, str, name="analysis_failed")
    """Emitted when an analysis fails. (analysis_name, error_message)"""

    # =========================================================================
    # Signals - Visualization Events
    # =========================================================================

    visualization_updated = pyqtSignal(str, name="visualization_updated")
    """Emitted when a visualization is updated. (plot_id)"""

    plot_settings_changed = pyqtSignal(dict, name="plot_settings_changed")
    """Emitted when plot settings change. (settings_dict)"""

    # =========================================================================
    # Singleton Access
    # =========================================================================

    def __new__(cls, *args, **kwargs) -> "EventBus":
        """Create the singleton, initialising the QObject here.

        The ``QObject`` base must be constructed *inside* ``__new__``, before
        the object is published to ``cls._instance``.

        The previous version did the opposite: ``__new__`` returned a raw
        ``super().__new__(cls)`` with its C++ side still uninitialised, and
        ``__init__`` then did ``if self._initialized: ...; super().__init__()``.
        Reading an attribute on a QObject whose ``__init__`` has not run puts
        PyQt6 into unbounded recursion -- the interpreter dies with a native
        "stack overflow" (WinError / exit 127), not a catchable
        ``RecursionError``. Instantiating ``EventBus()`` crashed the process,
        which took down ``models.state_manager`` and, through it, most of the
        test suite at ``tests/controllers/test_data_controller.py``.

        Initialising the base class in ``__new__`` and leaving ``__init__`` a
        no-op sidesteps the whole problem: by the time any attribute is
        touched the QObject is fully constructed, and ``cls()`` still returns
        the one shared instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    QObject.__init__(instance)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Set up per-instance state. The QObject base is built in ``__new__``.

        Only the first construction does any work; later ``EventBus()`` calls
        return the same object and must not re-register anything.
        """
        if getattr(self, "_logger", None) is not None:
            return
        self._logger = logging.getLogger(f"{__name__}.EventBus")
        self._logger.info("EventBus initialized")

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get the singleton EventBus instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton. For testing only."""
        with cls._instance_lock:
            cls._instance = None

    # =========================================================================
    # Publish Methods
    # =========================================================================

    def emit_data_changed(self, matrix: Any) -> None:
        """Emit data_changed event."""
        self._logger.debug("Event: data_changed")
        self.data_changed.emit(matrix)

    def emit_metadata_changed(self, scope: str, index: int, metadata: dict[str, Any]) -> None:
        """Emit metadata_changed event."""
        self._logger.debug(f"Event: metadata_changed scope={scope} index={index}")
        self.metadata_changed.emit(scope, index, metadata)

    def emit_undo_stack_changed(self) -> None:
        """Emit undo_stack_changed event."""
        self.undo_stack_changed.emit()

    def emit_analysis_started(self, analysis_name: str) -> None:
        """Emit analysis_started event."""
        self._logger.debug(f"Event: analysis_started {analysis_name}")
        self.analysis_started.emit(analysis_name)

    def emit_analysis_completed(self, analysis_name: str, result: Any) -> None:
        """Emit analysis_completed event."""
        self._logger.debug(f"Event: analysis_completed {analysis_name}")
        self.analysis_completed.emit(analysis_name, result)

    def emit_analysis_failed(self, analysis_name: str, error: str) -> None:
        """Emit analysis_failed event."""
        self._logger.warning(f"Event: analysis_failed {analysis_name}: {error}")
        self.analysis_failed.emit(analysis_name, error)

    def emit_visualization_updated(self, plot_id: str) -> None:
        """Emit visualization_updated event."""
        self.visualization_updated.emit(plot_id)

    def emit_plot_settings_changed(self, settings: dict[str, Any]) -> None:
        """Emit plot_settings_changed event."""
        self.plot_settings_changed.emit(settings)


def get_event_bus() -> EventBus:
    """Convenience function to get the EventBus singleton."""
    return EventBus()
