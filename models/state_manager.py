# =============================================================================
# FILE: models/state_manager.py
# =============================================================================
"""
Thread-Safe State Manager for PaleoAST

This module implements the central state management system that maintains
application-wide state including the current data matrix, metadata,
and visualization settings.

Architecture:
    Uses Singleton pattern with Read-Write lock for thread safety.
    Supports concurrent read access and exclusive write access.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

from utils.event_bus import get_event_bus

from .column_metadata import ColumnMetadataManager
from .data_matrix import DataMatrix
from .row_metadata import RowMetadataManager

logger = logging.getLogger(__name__)


class StateManager:
    """
    Thread-safe global state manager for PaleoAST.

    This class implements a singleton pattern with read-write locking
    to ensure thread-safe access to application state from multiple threads.

    State Components:
        - data_matrix: Current DataMatrix being analyzed
        - column_metadata: ColumnMetadataManager for column properties
        - row_metadata: RowMetadataManager for row properties
        - analysis_results: Cache of completed analysis results
        - visualization_settings: Current plot/visualization settings
        - undo_stack: Stack of previous states for undo functionality

    Thread Safety:
        - Uses RLock for reentrant locking
        - Read operations acquire read lock (shared access)
        - Write operations acquire write lock (exclusive access)
        - Lock acquisition is automatic via context managers

    Example:
        >>> state = StateManager.get_instance()
        >>> with state.read_lock():
        ...     matrix = state.data_matrix
        ...     print(matrix.shape)
    """

    _instance: Optional["StateManager"] = None
    _instance_lock = threading.Lock()
    _MAX_CACHE_SIZE = 100

    def __new__(cls) -> "StateManager":
        """
        Create or return the singleton instance.

        This ensures only one StateManager exists throughout the
        application lifecycle.

        Note: This method does NOT acquire the instance lock. Locking
        is handled in get_instance() to avoid re-entrant deadlocks.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the StateManager.

        Only initializes once (idempotent). Subsequent calls do nothing.
        """
        if self._initialized:
            return

        self._logger = logging.getLogger(f"{__name__}.StateManager")

        # Initialize locks
        self._read_write_lock = threading.RLock()
        self._state_lock = threading.Lock()

        # Initialize state variables
        self._data_matrix: DataMatrix | None = None
        self._column_metadata: ColumnMetadataManager | None = None
        self._row_metadata: RowMetadataManager | None = None
        self._analysis_cache: OrderedDict[str, Any] = OrderedDict()
        self._visualization_settings: dict[str, Any] = {}
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._modified: bool = False
        self._current_file: str | None = None

        self._initialized = True

        self._logger.info("StateManager initialized")

    # =========================================================================
    # Singleton Access
    # =========================================================================

    @classmethod
    def get_instance(cls) -> "StateManager":
        """
        Get the singleton instance with thread-safe double-checked locking.

        This method uses the double-checked locking pattern to ensure
        thread-safe singleton creation with minimal lock contention.

        The double-check pattern:
        1. First check without lock: fast path when instance exists
        2. Acquire lock only when instance needs creation
        3. Second check with lock held: prevent race between threads
           that both passed the first check

        Returns:
            StateManager: The singleton instance

        Thread Safety:
            - Multiple threads can call get_instance() concurrently
            - Only one thread will create the instance
            - All threads receive the same instance reference
        """
        # First check: fast path, no lock needed if instance exists
        if cls._instance is None:
            # Acquire lock for instance creation
            with cls._instance_lock:
                # Second check: ensure instance wasn't created by
                # another thread while waiting for the lock
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        Warning:
            This should only be used for testing or when absolutely
            necessary, as it clears all application state.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None

    # =========================================================================
    # Lock Context Managers
    # =========================================================================

    def read_lock(self) -> "ReadLockContext":
        """
        Acquire read lock for concurrent read access.

        Multiple threads can hold read locks simultaneously.

        Returns:
            ReadLockContext: Context manager for read access

        Example:
            >>> state = StateManager.get_instance()
            >>> with state.read_lock():
            ...     data = state.data_matrix
        """
        return ReadLockContext(self._read_write_lock)

    def write_lock(self) -> "WriteLockContext":
        """
        Acquire write lock for exclusive write access.

        Only one thread can hold the write lock.

        Returns:
            WriteLockContext: Context manager for write access
        """
        return WriteLockContext(self._read_write_lock)

    # =========================================================================
    # Data Matrix Operations
    # =========================================================================

    @property
    def data_matrix(self) -> DataMatrix | None:
        """Get current data matrix (thread-safe)."""
        with self._read_write_lock:
            return self._data_matrix

    @property
    def has_data(self) -> bool:
        """Check if data matrix is loaded."""
        with self._read_write_lock:
            return self._data_matrix is not None

    def set_data_matrix(
        self,
        matrix: DataMatrix,
        _record_undo: bool = True,
        _reset_metadata: bool = True,
        mark_modified: bool | None = None,
    ) -> None:
        """
        Set the current data matrix.

        Parameters:
            matrix: New DataMatrix to set
            _record_undo: When False, the previous state is *not*
                pushed onto the undo stack. Programmatic loads (e.g.
                the spreadsheet rebuilding itself from the EventBus)
                should pass ``False`` to avoid polluting the user's
                undo history with synthetic entries.
            _reset_metadata: When True (the default) the per-column
                and per-row metadata managers are recreated from the
                new matrix's labels. Callers that want to preserve
                existing metadata (e.g. a simple value edit) should
                pass ``False``.
            mark_modified: Whether to flag the project as having
                unsaved changes. Loading a file from disk is NOT a
                modification (the on-disk state matches), so callers
                that load fresh data should pass ``False``. When None
                (default) the flag is set True, preserving the
                historical behaviour for in-app edits and transforms.
        """
        with self._read_write_lock:
            if _record_undo:
                self._push_undo()
            self._data_matrix = matrix
            self._logger.info(f"set_data_matrix: shape=({matrix.n_samples} x {matrix.n_variables})")
            if _reset_metadata:
                # Build a fresh metadata manager but carry over any
                # existing per-column / per-row attributes whose
                # *label* still exists in the new matrix. This avoids
                # silently erasing user-defined group / colour / marker
                # data when the dataset is replaced with a new one
                # that happens to reuse some labels.
                preserved_col = (
                    self._column_metadata.to_dict() if self._column_metadata else {}
                )
                preserved_row = (
                    self._row_metadata.to_dict() if self._row_metadata else {}
                )
                self._column_metadata = ColumnMetadataManager(
                    n_columns=matrix.n_variables, column_labels=matrix.col_labels
                )
                self._column_metadata.from_dict_by_label(preserved_col, matrix.col_labels)
                self._row_metadata = RowMetadataManager(
                    n_rows=matrix.n_samples, row_labels=matrix.row_labels
                )
                self._row_metadata.from_dict_by_label(preserved_row, matrix.row_labels)
            self._analysis_cache.clear()
            if mark_modified is not None:
                self._modified = mark_modified
            else:
                self._modified = True
        get_event_bus().emit_data_changed(matrix)

    def clear_data(self, _record_undo: bool = True) -> None:
        """Clear the current data matrix.

        Parameters:
            _record_undo: When False, the previous state is *not*
                pushed onto the undo stack.
        """
        with self._read_write_lock:
            self._logger.info("clear_data: clearing current data matrix and analysis cache")
            if _record_undo:
                self._push_undo()
            self._data_matrix = None
            self._column_metadata = None
            self._row_metadata = None
            self._analysis_cache.clear()
            self._modified = True
        get_event_bus().emit_data_changed(None)

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    @property
    def column_metadata(self) -> ColumnMetadataManager | None:
        """Get column metadata manager."""
        with self._read_write_lock:
            return self._column_metadata

    @property
    def row_metadata(self) -> RowMetadataManager | None:
        """Get row metadata manager."""
        with self._read_write_lock:
            return self._row_metadata

    def set_col_metadata(self, col_index: int, metadata_dict: dict[str, Any]) -> None:
        """
        Set column metadata from a dictionary.

        Parameters:
            col_index: Column index
            metadata_dict: Dictionary with metadata fields (color, group, data_type, etc.)

        Raises:
            IndexError: If ``col_index`` is out of range.
        """
        with self._read_write_lock:
            if self._column_metadata is None:
                return
            # Bounds check before delegating — otherwise an out-of-range
            # index would raise ``IndexError`` deep inside the metadata
            # manager with a confusing message.
            if self._data_matrix is None:
                return
            if not (0 <= col_index < self._data_matrix.n_variables):
                raise IndexError(
                    f"set_col_metadata: col_index {col_index} out of range "
                    f"[0, {self._data_matrix.n_variables})"
                )
            if "data_type" in metadata_dict:
                self._column_metadata.set_data_type(col_index, metadata_dict["data_type"])
            if "group" in metadata_dict:
                self._column_metadata.set_group(col_index, metadata_dict["group"])
            if "color" in metadata_dict:
                self._column_metadata.set_color(col_index, metadata_dict["color"])
            if "marker" in metadata_dict:
                self._column_metadata.set_marker(col_index, metadata_dict["marker"])
        get_event_bus().emit_metadata_changed("column", col_index, metadata_dict)

    def set_row_metadata(self, row_index: int, metadata_dict: dict[str, Any]) -> None:
        """
        Set row metadata from a dictionary.

        Parameters:
            row_index: Row index
            metadata_dict: Dictionary with metadata fields (color, group, marker, etc.)

        Raises:
            IndexError: If ``row_index`` is out of range.
        """
        with self._read_write_lock:
            if self._row_metadata is None:
                return
            if self._data_matrix is None:
                return
            if not (0 <= row_index < self._data_matrix.n_samples):
                raise IndexError(
                    f"set_row_metadata: row_index {row_index} out of range "
                    f"[0, {self._data_matrix.n_samples})"
                )
            if "group" in metadata_dict:
                self._row_metadata.set_group(row_index, metadata_dict["group"])
            if "color" in metadata_dict:
                self._row_metadata.set_color(row_index, metadata_dict["color"])
            if "marker" in metadata_dict:
                self._row_metadata.set_marker(row_index, metadata_dict["marker"])
        get_event_bus().emit_metadata_changed("row", row_index, metadata_dict)

    # =========================================================================
    # Analysis Cache
    # =========================================================================

    def cache_result(self, key: str, result: Any) -> None:
        """
        Cache an analysis result with LRU eviction.

        Parameters:
            key: Unique identifier for the result
            result: The result to cache
        """
        with self._read_write_lock:
            self._logger.debug(f"cache_result: storing result with key='{key}'")
            if key in self._analysis_cache:
                self._analysis_cache.move_to_end(key)
            else:
                if len(self._analysis_cache) >= self._MAX_CACHE_SIZE:
                    self._analysis_cache.popitem(last=False)
                self._analysis_cache[key] = result

    def get_cached_result(self, key: str) -> Any | None:
        """
        Retrieve a cached analysis result (LRU-aware).

        Parameters:
            key: Unique identifier for the result

        Returns:
            The cached result or None if not found
        """
        with self._read_write_lock:
            self._logger.debug(f"get_cached_result: retrieving result with key='{key}'")
            if key in self._analysis_cache:
                self._analysis_cache.move_to_end(key)
                return self._analysis_cache[key]
            return None

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        with self._read_write_lock:
            self._analysis_cache.clear()

    # =========================================================================
    # Visualization Settings
    # =========================================================================

    def get_visualization_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a visualization setting.

        Parameters:
            key: Setting name
            default: Default value if not found
        """
        with self._read_write_lock:
            return self._visualization_settings.get(key, default)

    def set_visualization_setting(self, key: str, value: Any) -> None:
        """
        Set a visualization setting.

        Parameters:
            key: Setting name
            value: Setting value
        """
        with self._read_write_lock:
            self._visualization_settings[key] = value

    def get_all_visualization_settings(self) -> dict[str, Any]:
        """
        Get all visualization settings.

        Returns:
            Dictionary of all settings
        """
        with self._read_write_lock:
            return self._visualization_settings.copy()

    # =========================================================================
    # Undo/Redo Operations
    # =========================================================================

    def _push_undo(self) -> None:
        """Push current state onto undo stack."""
        if self._data_matrix is not None:
            state = {
                "data_matrix": self._data_matrix.copy(),
                "column_metadata": (self._column_metadata.to_dict() if self._column_metadata else None),
                "row_metadata": (self._row_metadata.to_dict() if self._row_metadata else None),
            }
            self._undo_stack.append(state)

            # Limit undo stack size
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)

            # Clear redo stack on new action
            self._redo_stack.clear()

    def can_undo(self) -> bool:
        """Check if undo is available."""
        with self._read_write_lock:
            return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        with self._read_write_lock:
            return len(self._redo_stack) > 0

    def undo(self) -> None:
        """Undo the last state change."""
        with self._read_write_lock:
            if not self._undo_stack:
                return
            self._logger.debug(
                f"undo: undo_stack size={len(self._undo_stack)}, redo_stack size={len(self._redo_stack)}"
            )
            if self._data_matrix is not None:
                current_state = {
                    "data_matrix": self._data_matrix.copy(),
                    "column_metadata": (self._column_metadata.to_dict() if self._column_metadata else None),
                    "row_metadata": (self._row_metadata.to_dict() if self._row_metadata else None),
                }
                self._redo_stack.append(current_state)
            state = self._undo_stack.pop()
            self._data_matrix = state["data_matrix"]
            if state["column_metadata"] is not None and self._data_matrix:
                self._column_metadata = ColumnMetadataManager(
                    n_columns=self._data_matrix.n_variables, column_labels=self._data_matrix.col_labels
                )
                self._column_metadata.from_dict(state["column_metadata"])
            if state["row_metadata"] is not None and self._data_matrix:
                self._row_metadata = RowMetadataManager(
                    n_rows=self._data_matrix.n_samples, row_labels=self._data_matrix.row_labels
                )
                self._row_metadata.from_dict(state["row_metadata"])
        get_event_bus().emit_undo_stack_changed()

    def redo(self) -> None:
        """Redo the last undone change."""
        with self._read_write_lock:
            if not self._redo_stack:
                return
            self._logger.debug(
                f"redo: undo_stack size={len(self._undo_stack)}, redo_stack size={len(self._redo_stack)}"
            )
            if self._data_matrix is not None:
                current_state = {
                    "data_matrix": self._data_matrix.copy(),
                    "column_metadata": (self._column_metadata.to_dict() if self._column_metadata else None),
                    "row_metadata": (self._row_metadata.to_dict() if self._row_metadata else None),
                }
                self._undo_stack.append(current_state)
            state = self._redo_stack.pop()
            self._data_matrix = state["data_matrix"]
            if state["column_metadata"] is not None and self._data_matrix:
                self._column_metadata = ColumnMetadataManager(
                    n_columns=self._data_matrix.n_variables, column_labels=self._data_matrix.col_labels
                )
                self._column_metadata.from_dict(state["column_metadata"])
            if state["row_metadata"] is not None and self._data_matrix:
                self._row_metadata = RowMetadataManager(
                    n_rows=self._data_matrix.n_samples, row_labels=self._data_matrix.row_labels
                )
                self._row_metadata.from_dict(state["row_metadata"])
        get_event_bus().emit_undo_stack_changed()

    # =========================================================================
    # File Operations
    # =========================================================================

    @property
    def current_file(self) -> str | None:
        """Get the current file path."""
        with self._read_write_lock:
            return self._current_file

    @property
    def is_modified(self) -> bool:
        """Check if data has been modified since last save."""
        with self._read_write_lock:
            return self._modified

    def mark_saved(self, filepath: str | None = None) -> None:
        """
        Mark the data as saved.

        Parameters:
            filepath: The file path it was saved to
        """
        with self._read_write_lock:
            if filepath is not None:
                self._current_file = filepath
                self._logger.info(f"mark_saved: data saved to '{filepath}'")
            self._modified = False

    # =========================================================================
    # Clear/Reset
    # =========================================================================

    def clear(self) -> None:
        """Clear all state."""
        with self._read_write_lock:
            self._data_matrix = None
            self._column_metadata = None
            self._row_metadata = None
            self._analysis_cache.clear()
            self._visualization_settings.clear()
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._modified = False
            self._current_file = None

    # =========================================================================
    # Status Information
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get current state status information.

        Returns:
            Dictionary containing status information
        """
        with self._read_write_lock:
            status = {
                "has_data": self._data_matrix is not None,
                "is_modified": self._modified,
                "current_file": self._current_file,
                "undo_available": len(self._undo_stack) > 0,
                "redo_available": len(self._redo_stack) > 0,
                "cache_size": len(self._analysis_cache),
            }

            if self._data_matrix is not None:
                status.update(
                    {
                        "n_samples": self._data_matrix.n_samples,
                        "n_variables": self._data_matrix.n_variables,
                        "has_missing": self._data_matrix.has_missing,
                    }
                )

            return status

    def __repr__(self) -> str:
        return f"StateManager(has_data={self.has_data}, modified={self._modified})"


# Context manager classes for lock handling
class ReadLockContext:
    """
    Context manager for read lock acquisition.

    Implementation Note:
        This is a PSEUDO read-write lock using a single threading.RLock.
        Read and write operations are mutually exclusive - a true RWLock
        (which would allow multiple concurrent readers) is not used because
        the 'readerwriterlock' package is not a project dependency.

        Benefits of current approach:
        - No external dependency required
        - Provides thread-safe state access
        - Prevents data races on shared state

        Limitations:
        - Multiple threads cannot read concurrently (throughput limited)
        - For high-read-concurrency scenarios, consider adding
          'readerwriterlock' to dependencies and using RWLock

        Thread Safety:
            Uses RLock for reentrant locking - the same thread can
            acquire the read lock multiple times (nested calls).
    """

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock

    def __enter__(self) -> "ReadLockContext":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._lock.release()


class WriteLockContext:
    """
    Context manager for write lock acquisition.

    Implementation Note:
        This is a PSEUDO read-write lock using a single threading.RLock.
        Read and write operations are mutually exclusive - a true RWLock
        (which would allow multiple concurrent readers) is not used because
        the 'readerwriterlock' package is not a project dependency.

        Benefits of current approach:
        - No external dependency required
        - Provides exclusive access for writes
        - Prevents data races on shared state

        Limitations:
        - All reads are blocked during a write (even if no writes occur)
        - For high-read-concurrency scenarios, consider adding
          'readerwriterlock' to dependencies and using RWLock

        Thread Safety:
            Uses RLock for reentrant locking - the same thread can
            acquire the write lock multiple times (nested calls).
    """

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock

    def __enter__(self) -> "WriteLockContext":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._lock.release()


# Convenience function for getting the singleton
def get_state_manager() -> StateManager:
    """
    Get the global StateManager instance.

    Returns:
        StateManager: The singleton state manager
    """
    return StateManager.get_instance()
