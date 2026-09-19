# =============================================================================
# FILE: controllers/data_controller.py
# =============================================================================
"""
Data Controller for PaleoAST

This controller manages all data operations including:
    - File import/export (CSV, Excel)
    - Data transformation
    - Row/column operations
    - Undo/Redo

Author: PaleoAST Development Team
version: 1.0.1
"""

import csv
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from models.data_matrix import DataMatrix
from models.state_manager import get_state_manager
from utils.exceptions import FileOperationError, ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# QRunnable Task for Asynchronous CSV Loading
# =============================================================================


class CsvLoadTask:
    """
    Container for CSV load task parameters and cancellation flag.

    This plain class carries the input arguments and the cancellation
    request flag. It is not a QRunnable subclass so the controller can
    create and inspect a task before submitting it to a thread pool.
    """

    def __init__(
        self,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        has_row_labels: bool = False,
        missing_value: str | None = None,
    ) -> None:
        self.filepath = filepath
        self.delimiter = delimiter
        self.has_header = has_header
        self.has_row_labels = has_row_labels
        self.missing_value = missing_value
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the load operation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class _NoopSignal:
    """Signal stand-in used when PyQt6 is not importable."""

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def connect(self, *args: Any, **kwargs: Any) -> None:
        pass

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoopSignalBridge:
    result_ready = _NoopSignal()
    error_raised = _NoopSignal()
    progress = _NoopSignal()
    cancelled = _NoopSignal()


class DataLoadTask:
    """
    QRunnable worker that loads a CSV file on a background thread.

    Signals (only valid when PyQt6 is available):
        progress (int, int): rows_processed, total_rows (or -1, -1 for indeterminate)
        result_ready (DataMatrix): successfully loaded matrix
        error_raised (Exception): any exception that occurred during loading
        cancelled (): loading was cancelled by the caller

    The task reports progress every 10,000 rows so the GUI can keep a
    progress bar up to date without flooding the signal queue.

    Cancellation:
        Call ``cancel()`` on ``task.task`` (the underlying
        :class:`CsvLoadTask`) before starting the task, or at any point
        while it is queued/running.  The worker checks the flag before
        starting and again before returning a result; after
        cancellation it emits the ``cancelled`` signal and calls
        ``result_ready.emit(None)`` so the caller can distinguish it
        from a successful load.
    """

    # Class-level import to avoid hard PyQt6 dependency in headless environments
    _QRunnable = None
    _QThreadPool = None
    _SignalsClass = None
    _pyqtImportFailed = False

    def __init__(
        self,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        has_row_labels: bool = False,
        missing_value: str | None = None,
    ) -> None:
        self._task = CsvLoadTask(
            filepath=filepath,
            delimiter=delimiter,
            has_header=has_header,
            has_row_labels=has_row_labels,
            missing_value=missing_value,
        )
        self._signals: Any = None
        self._qrunnable: Any = None
        self._started = False
        self._setup_pyqt()
        if self._signals is None:
            self._install_noop_signals()

    def _setup_pyqt(self) -> None:
        """Import PyQt6 lazily and build the QObject signal bridge.

        ``pyqtSignal`` only works as a class attribute of a QObject
        subclass, so the signals live on a dedicated bridge object
        which is re-exported on this instance for API compatibility.
        """
        if DataLoadTask._SignalsClass is None:
            try:
                from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
            except ImportError:
                logger.debug("PyQt6 not available; DataLoadTask runs synchronously")
                DataLoadTask._pyqtImportFailed = True
                return

            class _DataLoadSignals(QObject):
                result_ready = pyqtSignal(object)
                error_raised = pyqtSignal(Exception)
                progress = pyqtSignal(int, int)  # rows_loaded, total_rows
                cancelled = pyqtSignal()

            DataLoadTask._SignalsClass = _DataLoadSignals
            DataLoadTask._QRunnable = QRunnable
            DataLoadTask._QThreadPool = QThreadPool

        self._signals = DataLoadTask._SignalsClass()
        self.result_ready = self._signals.result_ready
        self.error_raised = self._signals.error_raised
        self.progress = self._signals.progress
        self.cancelled = self._signals.cancelled

    def _install_noop_signals(self) -> None:
        """Provide no-op signal stand-ins when PyQt6 is unavailable."""
        bridge = _NoopSignalBridge()
        self._signals = bridge
        self.result_ready = bridge.result_ready
        self.error_raised = bridge.error_raised
        self.progress = bridge.progress
        self.cancelled = bridge.cancelled

    def _create_qrunnable(self) -> Any:
        """Build the actual QRunnable wrapper (only if PyQt6 is available)."""
        if DataLoadTask._QRunnable is None:
            return None

        task = self

        class _CsvLoadQRunnable(DataLoadTask._QRunnable):
            """Concrete QRunnable that runs :meth:`_run` on a worker thread."""

            def run(self) -> None:
                task._run()

        self._qrunnable = _CsvLoadQRunnable()
        return self._qrunnable

    def _run(self) -> None:
        """
        Execute the CSV load on a background thread.

        Emits ``result_ready(DataMatrix)``, ``error_raised(Exception)``,
        ``progress(int,int)``, or ``cancelled()`` exactly once.
        """
        filepath = self._task.filepath
        delimiter = self._task.delimiter
        has_header = self._task.has_header
        has_row_labels = self._task.has_row_labels
        missing_value = self._task.missing_value

        try:
            # Honour cancellation requested before the worker started
            if self._task.is_cancelled:
                self.cancelled.emit()
                self.result_ready.emit(None)
                return

            import pandas as pd

            path = Path(filepath).resolve()
            if not path.exists():
                raise FileOperationError(f"File not found: {filepath}")

            # Emit indeterminate progress while reading
            self.progress.emit(0, -1)

            # Use pandas read_csv for vectorised parsing - 10-100x faster
            # than the previous Python csv.reader double loop.
            # Build na_values set for missing-value replacement
            na_values = {""} if missing_value else set()
            if missing_value:
                na_values.add(missing_value)

            df = pd.read_csv(
                path,
                sep=delimiter,
                header=0 if has_header else None,
                index_col=False,  # we handle row labels manually
                na_values=na_values,
                keep_default_na=True,
                encoding="utf-8",
                encoding_errors="replace",
                low_memory=False,
            )

            if len(df) == 0:
                raise FileOperationError("File is empty")

            # Extract row labels if requested (first column)
            if has_row_labels:
                row_labels = df.iloc[:, 0].astype(str).tolist()
                df = df.iloc[:, 1:]
            else:
                row_labels = None

            # Mirror load_csv(): auto-detect a non-numeric first column
            # and treat it as row labels when the user did not request
            # has_row_labels explicitly.
            if not has_row_labels and row_labels is None:
                first_col = df.iloc[:, 0]
                try:
                    pd.to_numeric(first_col, errors="raise")
                    first_col_is_numeric = True
                except (ValueError, TypeError):
                    first_col_is_numeric = False
                if not first_col_is_numeric or (
                    first_col.dtype == object and first_col.nunique() == len(first_col)
                ):
                    row_labels = first_col.astype(str).tolist()
                    df = df.iloc[:, 1:]
                    if df.shape[1] == 0:
                        raise FileOperationError(
                            "CSV contains only a label column; no numeric variables to load"
                        )

            # Check cancellation before returning a result
            if self._task.is_cancelled:
                self.cancelled.emit()
                self.result_ready.emit(None)
                return

            # Extract column labels (header row already consumed by pandas)
            if has_header:
                col_labels = df.columns.astype(str).tolist()
            else:
                col_labels = [str(c) for c in df.columns.tolist()]

            # Convert remaining columns to float (coerce errors to NaN)
            data = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            rows_processed = data.shape[0]

            # Emit final progress
            self.progress.emit(rows_processed, rows_processed)

            # Create DataMatrix
            matrix = DataMatrix(data=data, row_labels=row_labels, col_labels=col_labels)

            self.result_ready.emit(matrix)

        except Exception as e:
            self.error_raised.emit(e)

    @property
    def task(self) -> CsvLoadTask:
        """Direct access to the underlying CsvLoadTask for cancellation."""
        return self._task

    def start(self, priority: int = 0) -> Any:
        """
        Start this task on the global QThreadPool (idempotent).

        If PyQt6 is not available the task runs synchronously on the
        calling thread.  Returns the QRunnable wrapper, or None in the
        synchronous case.  Calling ``start()`` twice is a no-op.

        Parameters:
            priority: Task priority (higher values run first; default 0).

        Returns:
            QRunnable or None
        """
        if self._started:
            return self._qrunnable
        qrunnable = self._create_qrunnable()
        if qrunnable is not None:
            DataLoadTask._QThreadPool.globalInstance().start(qrunnable, priority=priority)
        else:
            # Fallback: run synchronously when PyQt6 is unavailable
            self._run()
        self._started = True
        return qrunnable

    def submit(self, priority: int = 0) -> Any:
        """Deprecated alias of :meth:`start` kept for backward compatibility."""
        return self.start(priority=priority)


class DataController:
    """
    Controller for data operations.

    Manages data loading, transformation, and persistence.
    """

    def __init__(self) -> None:
        """Initialize the data controller."""
        self._logger = logging.getLogger(f"{__name__}.DataController")
        self._lock = threading.RLock()
        self._state = get_state_manager()

        # Supported formats
        self._supported_import = [".csv", ".txt", ".dat", ".xlsx", ".xls"]
        self._supported_export = [".csv", ".txt"]

        self._logger.info("DataController initialized")

    # =========================================================================
    # Data Loading (Synchronous)
    # =========================================================================

    def load_csv(
        self,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        has_row_labels: bool = False,
        missing_value: str | None = None,
    ) -> DataMatrix:
        """
        Load data from CSV file (synchronous, blocks caller).

        Uses pandas for vectorised parsing - significantly faster than the
        previous csv.reader double-loop implementation for large files.

        Parameters:
            filepath: Path to CSV file
            delimiter: Column delimiter
            has_header: Whether file has header row
            has_row_labels: Whether first column contains row labels
            missing_value: String representing missing values

        Returns:
            DataMatrix: Loaded data

        Raises:
            FileOperationError: If file cannot be read
        """
        with self._lock:
            self._logger.info(
                f"load_csv called with filepath={filepath}, delimiter='{delimiter}', has_header={has_header}, has_row_labels={has_row_labels}"
            )
            try:
                path = Path(filepath).resolve()

                if not path.exists():
                    raise FileOperationError(f"File not found: {filepath}")

                try:
                    import pandas as pd
                except ImportError:
                    raise FileOperationError(
                        "pandas is required for CSV import. Install with: pip install pandas"
                    )

                # Build na_values set for missing-value replacement
                na_values = {""} if missing_value else set()
                if missing_value:
                    na_values.add(missing_value)

                # Use pandas for vectorised parsing - 10-100x faster than csv.reader
                # Read with dtype=object first so we can auto-detect string columns
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    header=0 if has_header else None,
                    index_col=False,  # handle row labels manually
                    na_values=na_values,
                    keep_default_na=True,
                    encoding="utf-8",
                    encoding_errors="replace",
                    low_memory=False,
                )

                if len(df) == 0:
                    raise FileOperationError("File is empty")

                # Extract row labels if requested (first column)
                if has_row_labels:
                    row_labels = df.iloc[:, 0].astype(str).tolist()
                    df = df.iloc[:, 1:]
                else:
                    row_labels = None

                # Auto-detect string (non-numeric) columns and treat them as
                # row labels if the user didn't explicitly set has_row_labels.
                if not has_row_labels and row_labels is None:
                    # Check if the first column is non-numeric
                    first_col = df.iloc[:, 0]
                    try:
                        pd.to_numeric(first_col, errors="raise")
                        first_col_is_numeric = True
                    except (ValueError, TypeError):
                        first_col_is_numeric = False
                    # Also check: if first column has all unique string-like values
                    if not first_col_is_numeric or (first_col.dtype == object and first_col.nunique() == len(first_col)):
                        row_labels = first_col.astype(str).tolist()
                        df = df.iloc[:, 1:]
                        if df.shape[1] == 0:
                            raise FileOperationError(
                                "CSV contains only a label column; no numeric variables to load"
                            )

                # Extract column labels (header row already consumed by pandas)
                if has_header:
                    col_labels = df.columns.astype(str).tolist()
                else:
                    # When no header, use pandas-generated integer column names
                    col_labels = [str(c) for c in df.columns.tolist()]

                # Convert remaining columns to float (coerce errors to NaN)
                data = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

                # Create DataMatrix
                matrix = DataMatrix(data=data, row_labels=row_labels, col_labels=col_labels)

                # Set in state
                self._state.set_data_matrix(matrix)
                self._state.mark_saved(filepath)

                self._logger.info(
                    f"CSV loaded successfully: shape={data.shape} ({data.shape[0]} samples x {data.shape[1]} variables)"
                )
                return matrix

            except FileOperationError:
                raise
            except Exception as e:
                self._logger.error(f"Failed to load CSV from '{filepath}': {e!s}")
                # Provide clearer error for empty files
                msg = str(e)
                if "No columns to parse" in msg or "empty" in msg.lower():
                    raise FileOperationError(f"File is empty or has no columns: {filepath}")
                raise FileOperationError(f"Failed to load CSV: {e!s}")

    # =========================================================================
    # Data Loading (Asynchronous)
    # =========================================================================

    def load_csv_async(
        self,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        has_row_labels: bool = False,
        missing_value: str | None = None,
        start: bool = True,
    ) -> DataLoadTask:
        """
        Load data from CSV file on a background thread (non-blocking).

        This is the asynchronous variant of :meth:`load_csv`.  It submits
        the load operation to the global QThreadPool and returns a
        :class:`DataLoadTask` object that can be used to:

        * Connect to ``result_ready`` to receive the :class:`DataMatrix`
          when loading completes.
        * Connect to ``progress(rows, total)`` for progress updates.
        * Connect to ``error_raised(Exception)`` to handle errors.
        * Call ``task.task.cancel()`` to cancel a pending load.

        Parameters:
            filepath: Path to CSV file
            delimiter: Column delimiter
            has_header: Whether file has header row
            has_row_labels: Whether first column contains row labels
            missing_value: String representing missing values
            start: When True (default) the task is submitted to the
                thread pool immediately.  Pass False to connect to the
                signals first and then call ``task.start()`` yourself,
                which avoids racing with early signal emission.

        Returns:
            DataLoadTask: Task object with signals for result, progress, and errors

        Example::

            task = controller.load_csv_async("data.csv", start=False)
            task.result_ready.connect(lambda m: print(f"Loaded {m.shape}"))
            task.progress.connect(lambda r, t: bar.setRange(0, t) or bar.setValue(r))
            task.error_raised.connect(lambda e: show_error(e))
            task.start()
            # To cancel a started but not yet finished load:
            # task.task.cancel()
        """
        task = DataLoadTask(
            filepath=filepath,
            delimiter=delimiter,
            has_header=has_header,
            has_row_labels=has_row_labels,
            missing_value=missing_value,
        )
        if start:
            task.start()
        return task

    def load_excel(
        self,
        filepath: str,
        sheet_name: str | int = 0,
        has_header: bool = True,
        has_row_labels: bool = True,
    ) -> DataMatrix:
        """
        Load data from Excel file (.xlsx/.xls).

        Parameters:
            filepath: Path to Excel file
            sheet_name: Sheet name or index (default: 0, first sheet)
            has_header: Whether file has header row
            has_row_labels: Whether first column contains row labels

        Returns:
            DataMatrix: Loaded data

        Raises:
            FileOperationError: If file cannot be read
        """
        with self._lock:
            self._logger.info(f"load_excel called with filepath={filepath}, sheet={sheet_name}")
            try:
                path = Path(filepath).resolve()
                if not path.exists():
                    raise FileOperationError(f"File not found: {filepath}")

                try:
                    import pandas as pd
                except ImportError:
                    raise FileOperationError(
                        "pandas is required for Excel import. Install with: pip install pandas openpyxl"
                    )

                df = pd.read_excel(
                    filepath,
                    sheet_name=sheet_name,
                    header=0 if has_header else None,
                    index_col=0 if has_row_labels else False,
                )

                # Drop all-NaN rows/columns
                df = df.dropna(how="all").dropna(axis=1, how="all")

                data = df.values.astype(float)
                row_labels = list(df.index.astype(str))
                col_labels = list(df.columns.astype(str))

                matrix = DataMatrix(data=data, row_labels=row_labels, col_labels=col_labels)

                self._state.set_data_matrix(matrix)
                self._state.mark_saved(filepath)

                self._logger.info(
                    f"Excel loaded successfully: shape={data.shape} ({data.shape[0]} samples x {data.shape[1]} variables)"
                )
                return matrix

            except FileOperationError:
                raise
            except Exception as e:
                self._logger.error(f"Failed to load Excel from '{filepath}': {e!s}")
                raise FileOperationError(f"Failed to load Excel: {e!s}")

    def load_numpy(self, data: npt.NDArray) -> DataMatrix:
        """
        Load data from numpy array.

        Parameters:
            data: Numpy array of shape (n_samples, n_variables)

        Returns:
            DataMatrix: Data matrix
        """
        with self._lock:
            matrix = DataMatrix(data=data)
            self._state.set_data_matrix(matrix)
            return matrix

    # =========================================================================
    # Data Export
    # =========================================================================

    def export_csv(self, filepath: str, include_labels: bool = True, delimiter: str = ",") -> None:
        """
        Export current data to CSV file.

        Parameters:
            filepath: Output file path
            include_labels: Whether to include row/column labels
            delimiter: Column delimiter

        Raises:
            ValidationError: If no data available
            FileOperationError: If write fails
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data to export")

            matrix = self._state.data_matrix
            self._logger.info(
                f"export_csv called with filepath={filepath}, data dimensions={matrix.n_samples}x{matrix.n_variables}"
            )

            try:
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=delimiter)

                    # Hoist label/data accessors out of the loop: ``matrix.data``
                    # and label properties copy under a lock, so reading them
                    # per row made export O(n^2) and froze the GUI on big files.
                    raw = matrix.raw_data
                    row_labels = matrix.row_labels
                    col_labels = matrix.col_labels

                    # Write header
                    if include_labels:
                        if col_labels:
                            header = ["", *list(col_labels)]
                        else:
                            header = [""] + [f"Var_{i + 1}" for i in range(matrix.n_variables)]
                        writer.writerow(header)

                    # Write data rows
                    for i in range(matrix.n_samples):
                        if include_labels:
                            if row_labels:
                                row = [row_labels[i]]
                            else:
                                row = [f"Sample_{i + 1}"]
                            row.extend(raw[i].tolist())
                        else:
                            row = raw[i].tolist()
                        writer.writerow(row)

                self._state.mark_saved(filepath)
                self._logger.info(f"CSV exported successfully to '{filepath}'")

            except Exception as e:
                self._logger.error(f"Failed to export CSV to '{filepath}': {e!s}")
                raise FileOperationError(f"Failed to export CSV: {e!s}")

    def export_numpy(self) -> npt.NDArray:
        """
        Export current data as numpy array.

        Returns:
            npt.NDArray: Current data
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data available")

            return self._state.data_matrix.data.copy()

    # =========================================================================
    # Data Transformation
    # =========================================================================

    def transform_log(self, data: npt.NDArray | None = None, base: str = "natural") -> npt.NDArray:
        """
        Apply log transformation: log_b(x).

        Parameters:
            data: Input data. If None, uses state data.
            base: 'natural', 'base10', or 'base2'

        Returns:
            Transformed data

        Raises:
            ValidationError: If data contains zero or negative values
                (log is undefined for x ≤ 0).
        """
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available")
                data = self._state.data_matrix.data

            data_arr = np.asarray(data, dtype=float)
            if np.any(data_arr <= 0):
                n_zero = int(np.sum(data_arr == 0))
                n_neg = int(np.sum(data_arr < 0))
                problems = []
                if n_zero > 0:
                    problems.append(f"{n_zero} zero value(s)")
                if n_neg > 0:
                    problems.append(f"{n_neg} negative value(s)")
                raise ValidationError(
                    f"Log transformation requires strictly positive data; "
                    f"found {' and '.join(problems)}. "
                    f"Consider adding an offset (e.g. log(x+1)) or filtering."
                )

            if base == "natural":
                return np.log(data_arr)
            elif base == "base10":
                return np.log10(data_arr)
            elif base == "base2":
                return np.log2(data_arr)
            else:
                raise ValidationError(f"Unknown log base: {base}")

    def transform_standardize(self, data: npt.NDArray | None = None, method: str = "zscore") -> npt.NDArray:
        """
        Standardize data.

        Parameters:
            data: Input data. If None, uses state data.
            method: 'zscore' or 'minmax'

        Returns:
            Standardized data
        """
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available")
                data = self._state.data_matrix.data

            if method == "zscore":
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0, ddof=1)
                std = np.where(std == 0, 1, std)
                # ``data`` may be a view of the caller's array. The
                # ``-``/``/`` operators below allocate a fresh array, but
                # when ``data`` is already a float view of an integer
                # array the result is a *new* allocation. Be explicit
                # and copy here so callers never see in-place
                # modification of their input.
                data_arr = np.asarray(data, dtype=float).copy()
                return (data_arr - mean) / std
            elif method == "minmax":
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                range_val = max_val - min_val
                range_val = np.where(range_val == 0, 1, range_val)
                data_arr = np.asarray(data, dtype=float).copy()
                return (data_arr - min_val) / range_val
            else:
                raise ValidationError(f"Unknown standardization method: {method}")

    def transform_sqrt(self, data: npt.NDArray | None = None) -> npt.NDArray:
        """Apply square root transformation.

        The square root transform is only defined for non-negative values.
        The previous implementation silently called ``np.sqrt(np.abs(data))``,
        which discarded the sign of any negative entry — a negative
        abundance or measurement would be transformed as if it were
        positive, with no warning, producing misleading downstream
        results. Raise a :class:`ValidationError` instead so the caller
        can decide how to handle negative values (clip, shift, or use a
        different transform such as the Yeo-Johnson / signed-log).
        """
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available")
                data = self._state.data_matrix.data

            data_arr = np.asarray(data)
            if np.any(data_arr < 0):
                neg_count = int(np.sum(data_arr < 0))
                raise ValidationError(
                    f"Square root transformation requires non-negative data; "
                    f"found {neg_count} negative value(s). "
                    f"Consider clipping, shifting, or using a signed transform."
                )
            return np.sqrt(data_arr)

    # =========================================================================
    # Data Operations
    # =========================================================================

    def transpose(self) -> DataMatrix:
        """
        Transpose current data matrix.

        Returns:
            Transposed DataMatrix
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data available")

            matrix = self._state.data_matrix
            # ``matrix.data`` returns a copy, but ``.T`` is still a view
            # of that copy. Force a contiguous copy so the returned
            # DataMatrix cannot accidentally mutate the source array.
            transposed = matrix.data.T.copy()

            return DataMatrix(data=transposed, row_labels=matrix.col_labels, col_labels=matrix.row_labels)

    def subset_rows(self, indices: list[int]) -> DataMatrix:
        """
        Extract subset of rows.

        Parameters:
            indices: Row indices to extract

        Returns:
            Subset DataMatrix
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data available")

            matrix = self._state.data_matrix

            # ``matrix.data`` returns a copy, but fancy indexing on it
            # can still produce a view in some NumPy edge cases. Force
            # an explicit copy so the returned DataMatrix owns its
            # buffer.
            new_data = matrix.data[indices].copy()
            new_labels = None
            if matrix.row_labels:
                new_labels = [matrix.row_labels[i] for i in indices]

            return DataMatrix(data=new_data, row_labels=new_labels, col_labels=matrix.col_labels)

    def subset_columns(self, indices: list[int]) -> DataMatrix:
        """
        Extract subset of columns.

        Parameters:
            indices: Column indices to extract

        Returns:
            Subset DataMatrix
        """
        with self._lock:
            if not self._state.has_data:
                raise ValidationError("No data available")

            matrix = self._state.data_matrix

            new_data = matrix.data[:, indices].copy()
            new_labels = None
            if matrix.col_labels:
                new_labels = [matrix.col_labels[i] for i in indices]

            return DataMatrix(data=new_data, row_labels=matrix.row_labels, col_labels=new_labels)

    # =========================================================================
    # Undo/Redo
    # =========================================================================

    def undo(self) -> bool:
        """Undo last operation."""
        with self._lock:
            if self._state.can_undo():
                self._state.undo()
                return True
            return False

    def redo(self) -> bool:
        """Redo last undone operation."""
        with self._lock:
            if self._state.can_redo():
                self._state.redo()
                return True
            return False

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._state.can_undo()

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._state.can_redo()

    # =========================================================================
    # Status
    # =========================================================================

    def get_data_info(self) -> dict[str, Any]:
        """Get information about current data."""
        with self._lock:
            if not self._state.has_data:
                return {"has_data": False}

            matrix = self._state.data_matrix

            return {
                "has_data": True,
                "n_samples": matrix.n_samples,
                "n_variables": matrix.n_variables,
                "has_missing": matrix.has_missing,
                "has_row_labels": matrix.row_labels is not None,
                "has_col_labels": matrix.col_labels is not None,
                "is_modified": self._state.is_modified,
                "current_file": self._state.current_file,
            }
