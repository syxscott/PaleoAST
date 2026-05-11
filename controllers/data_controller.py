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
Version: 1.0.0
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
    # Data Loading
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
        Load data from CSV file.

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
                path = Path(filepath)

                if not path.exists():
                    raise FileOperationError(f"File not found: {filepath}")

                # Read file
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    rows = list(reader)

                if len(rows) == 0:
                    raise FileOperationError("File is empty")

                # Determine data dimensions
                if has_row_labels:
                    row_labels = [row[0] for row in rows]
                    data_rows = [row[1:] for row in rows]
                else:
                    row_labels = None
                    data_rows = rows

                if has_header and row_labels is None:
                    # First row is header
                    col_labels = data_rows[0]
                    data_rows = data_rows[1:]
                elif has_header and row_labels is not None:
                    col_labels = data_rows[0]
                    data_rows = data_rows[1:]
                    row_labels = row_labels[1:]
                else:
                    col_labels = None

                # Convert to numpy array
                data_list = []
                for row in data_rows:
                    row_data = []
                    for val in row:
                        if missing_value and val.strip() == missing_value:
                            row_data.append(np.nan)
                        else:
                            try:
                                row_data.append(float(val.strip()))
                            except ValueError:
                                row_data.append(np.nan)
                    data_list.append(row_data)

                data = np.array(data_list, dtype=float)

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
                raise FileOperationError(f"Failed to load CSV: {e!s}")

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
                path = Path(filepath)
                if not path.exists():
                    raise FileOperationError(f"File not found: {filepath}")

                try:
                    import pandas as pd
                except ImportError:
                    raise FileOperationError("pandas is required for Excel import. Install with: pip install pandas openpyxl")

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

                    # Write header
                    if include_labels:
                        if matrix.col_labels:
                            header = [""] + list(matrix.col_labels)
                        else:
                            header = [""] + [f"Var_{i + 1}" for i in range(matrix.n_variables)]
                        writer.writerow(header)

                    # Write data rows
                    for i in range(matrix.n_samples):
                        if include_labels:
                            if matrix.row_labels:
                                row = [matrix.row_labels[i]]
                            else:
                                row = [f"Sample_{i + 1}"]
                            row.extend(matrix.data[i].tolist())
                        else:
                            row = matrix.data[i].tolist()
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
        Apply log transformation.

        Parameters:
            data: Input data. If None, uses state data.
            base: 'natural', 'base10', or 'base2'

        Returns:
            Transformed data
        """
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available")
                data = self._state.data_matrix.data

            if base == "natural":
                return np.log(data)
            elif base == "base10":
                return np.log10(data)
            elif base == "base2":
                return np.log2(data)
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
                return (data - mean) / std
            elif method == "minmax":
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                range_val = max_val - min_val
                range_val = np.where(range_val == 0, 1, range_val)
                return (data - min_val) / range_val
            else:
                raise ValidationError(f"Unknown standardization method: {method}")

    def transform_sqrt(self, data: npt.NDArray | None = None) -> npt.NDArray:
        """Apply square root transformation."""
        with self._lock:
            if data is None:
                if not self._state.has_data:
                    raise ValidationError("No data available")
                data = self._state.data_matrix.data

            return np.sqrt(np.abs(data))

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
            transposed = matrix.data.T

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

            new_data = matrix.data[indices]
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

            new_data = matrix.data[:, indices]
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
