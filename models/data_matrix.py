# =============================================================================
# FILE: models/data_matrix.py
# =============================================================================
"""
Core Data Matrix Module for PaleoAST

This module implements the DataMatrix class, which is the central data structure
for storing and manipulating paleontological data in the PaleoAST application.

Mathematical Context:
    A data matrix X ∈ ℝ^(n×p) where:
    - n = number of samples (rows)
    - p = number of variables/features (columns)
    
    Each element x_ij represents the value of variable j for sample i.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, List, Tuple, Any, Sequence, Dict
from dataclasses import dataclass, field
import threading
from copy import deepcopy

from utils.exceptions import (
    DataValidationError,
    MatrixDimensionError,
)
from utils.validators import validate_data_array, check_missing_values
from utils.matrix_ops import ensure_matrix

import logging

logger = logging.getLogger(__name__)


class DataMatrix:
    """
    Core data matrix class for storing and manipulating paleontological data.
    
    This class extends numpy.ndarray to provide specialized functionality
    for paleontological data analysis, including masked values, metadata
    management, and efficient matrix operations.
    
    Attributes:
        data: The underlying numpy array containing the data
        row_labels: Labels for each row (sample names)
        col_labels: Labels for each column (variable names)
        masked: Boolean mask indicating missing values
    
    Mathematical Representation:
        X = [x_ij] where:
        - x_ij ∈ ℝ for continuous data
        - x_ij ∈ {0, 1} for binary data
        - x_ij ∈ ℤ≥0 for count data
        - x_ij = NaN for missing data
    
    Example:
        >>> import numpy as np
        >>> data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> matrix = DataMatrix(data, row_labels=['Sample_A', 'Sample_B'])
        >>> matrix.shape
        (2, 3)
        >>> matrix.n_samples
        2
        >>> matrix.n_variables
        3
    """
    
    def __init__(
        self,
        data: Union[npt.NDArray, List[List[float]]],
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        name: str = "Unnamed"
    ) -> None:
        """
        Initialize a DataMatrix instance.
        
        Parameters:
            data: 2D array-like data of shape (n_samples, n_variables)
            row_labels: Optional list of row labels (sample names)
            col_labels: Optional list of column labels (variable names)
            name: Name for this data matrix
        
        Raises:
            DataValidationError: If data validation fails
            MatrixDimensionError: If dimensions are invalid
        """
        # Convert to numpy array and validate
        self._data = validate_data_array(
            data,
            allow_nan=True,
            allow_inf=False,
            name="data_matrix"
        )
        
        n_samples, n_variables = self._data.shape
        
        # Initialize or validate row labels
        if row_labels is None:
            self._row_labels = [f"Sample_{i+1}" for i in range(n_samples)]
        else:
            if len(row_labels) != n_samples:
                raise MatrixDimensionError(
                    "Row label count must match number of samples",
                    details={
                        "n_samples": n_samples,
                        "n_labels": len(row_labels)
                    }
                )
            self._row_labels = list(row_labels)
        
        # Initialize or validate column labels
        if col_labels is None:
            self._col_labels = [f"Var_{j+1}" for j in range(n_variables)]
        else:
            if len(col_labels) != n_variables:
                raise MatrixDimensionError(
                    "Column label count must match number of variables",
                    details={
                        "n_variables": n_variables,
                        "n_labels": len(col_labels)
                    }
                )
            self._col_labels = list(col_labels)
        
        # Set name
        self._name = name

        # Thread lock for concurrent access
        self._lock = threading.RLock()

        self._logger = logging.getLogger(f"{__name__}.DataMatrix")
        self._logger.info(f"DataMatrix initialized: shape=({n_samples} x {n_variables}), name='{name}'")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def data(self) -> npt.NDArray:
        """
        Get the underlying data array.
        
        Returns:
            npt.NDArray: Copy of the underlying data array
        """
        with self._lock:
            return self._data.copy()
    
    @property
    def raw_data(self) -> npt.NDArray:
        """
        Get the underlying data array without copying.
        
        Warning:
            This returns a reference to the actual data.
            Modifying it directly will modify the DataMatrix.
        
        Returns:
            npt.NDArray: Reference to the data array
        """
        with self._lock:
            return self._data
    
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the data matrix.
        
        Returns:
            Tuple[int, int]: (n_samples, n_variables)
        """
        with self._lock:
            return self._data.shape
    
    @property
    def n_samples(self) -> int:
        """
        Get the number of samples (rows).
        
        Returns:
            int: Number of samples
        """
        with self._lock:
            return self._data.shape[0]
    
    @property
    def n_variables(self) -> int:
        """
        Get the number of variables (columns).
        
        Returns:
            int: Number of variables
        """
        with self._lock:
            return self._data.shape[1]
    
    @property
    def row_labels(self) -> List[str]:
        """
        Get the row labels.
        
        Returns:
            List[str]: Copy of row labels
        """
        with self._lock:
            return list(self._row_labels)
    
    @row_labels.setter
    def row_labels(self, labels: List[str]) -> None:
        """
        Set row labels.
        
        Parameters:
            labels: New row labels
        
        Raises:
            MatrixDimensionError: If label count doesn't match
        """
        with self._lock:
            if len(labels) != self._data.shape[0]:
                raise MatrixDimensionError(
                    "Row label count must match number of samples",
                    details={
                        "n_samples": self._data.shape[0],
                        "n_labels": len(labels)
                    }
                )
            self._row_labels = list(labels)
    
    @property
    def col_labels(self) -> List[str]:
        """
        Get the column labels.
        
        Returns:
            List[str]: Copy of column labels
        """
        with self._lock:
            return list(self._col_labels)
    
    @col_labels.setter
    def col_labels(self, labels: List[str]) -> None:
        """
        Set column labels.
        
        Parameters:
            labels: New column labels
        
        Raises:
            MatrixDimensionError: If label count doesn't match
        """
        with self._lock:
            if len(labels) != self._data.shape[1]:
                raise MatrixDimensionError(
                    "Column label count must match number of variables",
                    details={
                        "n_variables": self._data.shape[1],
                        "n_labels": len(labels)
                    }
                )
            self._col_labels = list(labels)
    
    @property
    def name(self) -> str:
        """
        Get the matrix name.
        
        Returns:
            str: Matrix name
        """
        with self._lock:
            return self._name
    
    @name.setter
    def name(self, name: str) -> None:
        """
        Set the matrix name.
        
        Parameters:
            name: New name
        """
        with self._lock:
            self._name = str(name)
    
    @property
    def nan_mask(self) -> npt.NDArray:
        """
        Get the boolean mask for missing values.
        
        Returns:
            npt.NDArray: Boolean array where True indicates missing values
        """
        with self._lock:
            return np.isnan(self._data)
    
    @property
    def has_missing(self) -> bool:
        """
        Check if matrix contains any missing values.
        
        Returns:
            bool: True if any NaN values present
        """
        with self._lock:
            return bool(np.any(np.isnan(self._data)))
    
    @property
    def missing_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about missing values.
        
        Returns:
            Dict: Missing value statistics
        """
        with self._lock:
            return check_missing_values(self._data, report_positions=False)
    
    # =========================================================================
    # Data Access Methods
    # =========================================================================
    
    def __getitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> Union[float, npt.NDArray, 'DataMatrix']:
        """
        Get item(s) from the data matrix.
        
        Parameters:
            key: Index, slice, or tuple of indices/slices
        
        Returns:
            Single value, array, or DataMatrix view depending on indexing
        """
        with self._lock:
            result = self._data[key]
            
            if isinstance(result, np.ndarray):
                if result.ndim == 0:
                    return float(result)
                elif isinstance(key, tuple) and isinstance(key[0], int) and isinstance(key[1], int):
                    return float(result)
                else:
                    return self._create_view(result, key)
            return result
    
    def _create_view(
        self,
        data: npt.NDArray,
        key: Tuple
    ) -> 'DataMatrixView':
        """
        Create a view of the data matrix with updated slicing.
        """
        if isinstance(key, tuple):
            row_key = key[0] if len(key) > 0 else slice(None)
            col_key = key[1] if len(key) > 1 else slice(None)
        else:
            row_key = key
            col_key = slice(None)
        
        # Determine row labels
        if isinstance(row_key, slice):
            row_labels = self._row_labels[row_key]
        elif isinstance(row_key, (list, np.ndarray)):
            row_labels = [self._row_labels[i] for i in row_key]
        else:
            row_labels = [self._row_labels[row_key]]
        
        # Determine column labels
        if isinstance(col_key, slice):
            col_labels = self._col_labels[col_key]
        elif isinstance(col_key, (list, np.ndarray)):
            col_labels = [self._col_labels[i] for i in col_key]
        else:
            col_labels = [self._col_labels[col_key]]
        
        view = DataMatrixView(
            data=data,
            row_labels=row_labels,
            col_labels=col_labels,
            parent=self
        )
        return view
    
    def __setitem__(
        self,
        key: Tuple[int, int],
        value: float
    ) -> None:
        """
        Set a single value in the matrix.
        
        Parameters:
            key: (row_index, col_index)
            value: New value
        """
        with self._lock:
            self._data[key] = value
    
    def get_row(self, index: int) -> npt.NDArray:
        """
        Get a single row as a 1D array.
        
        Parameters:
            index: Row index
        
        Returns:
            npt.NDArray: Row data
        """
        with self._lock:
            return self._data[index].copy()
    
    def get_column(self, index: int) -> npt.NDArray:
        """
        Get a single column as a 1D array.
        
        Parameters:
            index: Column index
        
        Returns:
            npt.NDArray: Column data
        """
        with self._lock:
            return self._data[:, index].copy()
    
    def get_row_by_label(self, label: str) -> Optional[npt.NDArray]:
        """
        Get a row by its label.
        
        Parameters:
            label: Row label to search for
        
        Returns:
            Optional[npt.NDArray]: Row data if found, None otherwise
        """
        with self._lock:
            try:
                index = self._row_labels.index(label)
                return self._data[index].copy()
            except ValueError:
                return None
    
    def get_column_by_label(self, label: str) -> Optional[npt.NDArray]:
        """
        Get a column by its label.
        
        Parameters:
            label: Column label to search for
        
        Returns:
            Optional[npt.NDArray]: Column data if found, None otherwise
        """
        with self._lock:
            try:
                index = self._col_labels.index(label)
                return self._data[:, index].copy()
            except ValueError:
                return None
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    
    def transpose(self) -> 'DataMatrix':
        """
        Transpose the data matrix.
        
        Mathematical Operation:
            X^T where (X^T)_ij = X_ji
        
        Returns:
            DataMatrix: Transposed matrix
        """
        with self._lock:
            return DataMatrix(
                data=self._data.T.copy(),
                row_labels=self._col_labels.copy(),
                col_labels=self._row_labels.copy(),
                name=f"{self._name}_transposed"
            )
    
    def subset_rows(
        self,
        indices: Union[List[int], npt.NDArray]
    ) -> 'DataMatrix':
        """
        Create a new DataMatrix with only the specified rows.
        
        Parameters:
            indices: List or array of row indices to include
        
        Returns:
            DataMatrix: Matrix with subset of rows
        """
        with self._lock:
            if isinstance(indices, list):
                indices = np.array(indices)

            self._logger.debug(f"subset_rows: selecting {len(indices)} rows from {self._data.shape[0]}")

            new_data = self._data[indices].copy()
            new_row_labels = [self._row_labels[i] for i in indices]

            return DataMatrix(
                data=new_data,
                row_labels=new_row_labels,
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_rows_subset"
            )
    
    def subset_columns(
        self,
        indices: Union[List[int], npt.NDArray]
    ) -> 'DataMatrix':
        """
        Create a new DataMatrix with only the specified columns.
        
        Parameters:
            indices: List or array of column indices to include
        
        Returns:
            DataMatrix: Matrix with subset of columns
        """
        with self._lock:
            if isinstance(indices, list):
                indices = np.array(indices)

            self._logger.debug(f"subset_columns: selecting {len(indices)} columns from {self._data.shape[1]}")

            new_data = self._data[:, indices].copy()
            new_col_labels = [self._col_labels[i] for i in indices]

            return DataMatrix(
                data=new_data,
                row_labels=self._row_labels.copy(),
                col_labels=new_col_labels,
                name=f"{self._name}_cols_subset"
            )
    
    def remove_constant_columns(self) -> 'DataMatrix':
        """
        Remove columns with zero or near-zero variance.
        
        Constant columns provide no information for multivariate
        analysis and can cause numerical issues.
        
        Returns:
            DataMatrix: Matrix without constant columns
        """
        with self._lock:
            # Compute variance for each column
            variances = np.var(self._data, axis=0, ddof=1)
            tolerance = 1e-10
            
            # Find non-constant columns
            non_constant_mask = variances > tolerance
            non_constant_indices = np.where(non_constant_mask)[0]
            
            if len(non_constant_indices) == self.n_variables:
                # No columns to remove
                return self.copy()
            
            return self.subset_columns(non_constant_indices.tolist())
    
    def remove_rows_with_missing(
        self,
        threshold: float = 1.0
    ) -> 'DataMatrix':
        """
        Remove rows with excessive missing values.
        
        Parameters:
            threshold: Fraction of columns that can be missing (0 to 1).
                      Default 1.0 means remove only rows that are entirely NaN.
        
        Returns:
            DataMatrix: Matrix without rows exceeding threshold
        """
        with self._lock:
            nan_counts = np.sum(np.isnan(self._data), axis=1)
            max_allowed_nan = int(threshold * self.n_variables)
            
            valid_mask = nan_counts <= max_allowed_nan
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == self.n_samples:
                return self.copy()
            
            return self.subset_rows(valid_indices.tolist())
    
    # =========================================================================
    # Statistics Methods
    # =========================================================================
    
    def column_means(self) -> npt.NDArray:
        """
        Compute mean for each column.
        
        Returns:
            npt.NDArray: Column means, NaN values are ignored
        """
        with self._lock:
            return np.nanmean(self._data, axis=0)
    
    def column_stds(self, ddof: int = 1) -> npt.NDArray:
        """
        Compute standard deviation for each column.
        
        Parameters:
            ddof: Delta degrees of freedom for variance
        
        Returns:
            npt.NDArray: Column standard deviations
        """
        with self._lock:
            return np.nanstd(self._data, axis=0, ddof=ddof)
    
    def column_sums(self) -> npt.NDArray:
        """
        Compute sum for each column.
        
        Returns:
            npt.NDArray: Column sums
        """
        with self._lock:
            return np.nansum(self._data, axis=0)
    
    def row_means(self) -> npt.NDArray:
        """
        Compute mean for each row.
        
        Returns:
            npt.NDArray: Row means
        """
        with self._lock:
            return np.nanmean(self._data, axis=1)
    
    # =========================================================================
    # Imputation Methods
    # =========================================================================
    
    def impute_mean(self) -> 'DataMatrix':
        """
        Impute missing values with column means.
        
        Mathematical Operation:
            x_ij = μ_j if x_ij = NaN
            where μ_j = (1/n) Σ_i x_ij for non-NaN values
        
        Returns:
            DataMatrix: Matrix with imputed values
        """
        with self._lock:
            result = self._data.copy()
            col_means = np.nanmean(result, axis=0)

            # Find NaN positions and fill with column means
            nan_mask = np.isnan(result)
            missing_count = int(np.sum(nan_mask))
            self._logger.info(f"impute_mean: imputing {missing_count} missing values with column means")

            for j in range(result.shape[1]):
                result[nan_mask[:, j], j] = col_means[j]

            return DataMatrix(
                data=result,
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_mean_imputed"
            )
    
    def impute_median(self) -> 'DataMatrix':
        """
        Impute missing values with column medians.
        
        Mathematical Operation:
            x_ij = median(x_j) if x_ij = NaN
        
        Returns:
            DataMatrix: Matrix with imputed values
        """
        with self._lock:
            result = self._data.copy()
            col_medians = np.nanmedian(result, axis=0)

            nan_mask = np.isnan(result)
            missing_count = int(np.sum(nan_mask))
            self._logger.info(f"impute_median: imputing {missing_count} missing values with column medians")

            for j in range(result.shape[1]):
                result[nan_mask[:, j], j] = col_medians[j]

            return DataMatrix(
                data=result,
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_median_imputed"
            )
    
    def impute_knn(self, k: int = 5) -> 'DataMatrix':
        """
        Impute missing values using K-Nearest Neighbors.
        
        Mathematical Algorithm:
            For each sample with missing values:
            1. Find k nearest neighbors using complete cases
            2. Impute missing values with weighted average of neighbors
        
        This method is particularly useful when data has complex
        multivariate relationships that mean/median cannot capture.
        
        Parameters:
            k: Number of nearest neighbors to use
        
        Returns:
            DataMatrix: Matrix with KNN-imputed values
        """
        with self._lock:
            from scipy.spatial.distance import cdist

            result = self._data.copy()
            nan_mask = np.isnan(result)
            missing_count = int(np.sum(nan_mask))

            if not np.any(nan_mask):
                return self.copy()

            self._logger.info(f"impute_knn: imputing {missing_count} missing values with k={k} nearest neighbors")
            
            # Identify rows with and without missing values
            has_nan = np.any(nan_mask, axis=1)
            complete_mask = ~has_nan
            
            if np.sum(complete_mask) < k:
                # Not enough complete rows for KNN, fall back to mean
                return self.impute_mean()
            
            complete_data = self._data[complete_mask]
            incomplete_indices = np.where(has_nan)[0]
            
            for idx in incomplete_indices:
                row = result[idx]
                row_nan = nan_mask[idx]
                
                if not np.any(row_nan):
                    continue
                
                # Get non-NaN values for this row
                valid_cols = ~row_nan
                if np.sum(valid_cols) == 0:
                    # All values missing, use mean of complete rows
                    for j in np.where(row_nan)[0]:
                        result[idx, j] = np.mean(complete_data[:, j])
                    continue
                
                # Compute distances using valid columns only
                dists = cdist(
                    result[idx, valid_cols].reshape(1, -1),
                    complete_data[:, valid_cols]
                )[0]
                
                # Get k nearest neighbors
                neighbor_indices = np.argsort(dists)[:k]
                neighbors = complete_data[neighbor_indices]
                
                # Impute each missing column
                for j in np.where(row_nan)[0]:
                    result[idx, j] = np.mean(neighbors[:, j])
            
            return DataMatrix(
                data=result,
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_knn_imputed"
            )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def copy(self) -> 'DataMatrix':
        """
        Create a deep copy of this DataMatrix.
        
        Returns:
            DataMatrix: Independent copy of this matrix
        """
        with self._lock:
            return DataMatrix(
                data=self._data.copy(),
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=self._name
            )
    
    def to_numpy(self) -> npt.NDArray:
        """
        Convert to numpy array.
        
        Returns:
            npt.NDArray: Data as numpy array
        """
        with self._lock:
            return self._data.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict: Dictionary with all data and metadata
        """
        with self._lock:
            return {
                "name": self._name,
                "data": self._data.tolist(),
                "row_labels": self._row_labels,
                "col_labels": self._col_labels,
                "shape": self._data.shape,
                "has_missing": self.has_missing,
                "missing_info": self.missing_info
            }
    
    def __repr__(self) -> str:
        """
        String representation of the DataMatrix.
        
        Returns:
            str: Human-readable description
        """
        with self._lock:
            return (
                f"DataMatrix(name='{self._name}', "
                f"shape={self._data.shape}, "
                f"missing={self.has_missing})"
            )
    
    def __str__(self) -> str:
        """
        Detailed string representation.
        
        Returns:
            str: Formatted string showing data and metadata
        """
        with self._lock:
            lines = [
                f"DataMatrix: {self._name}",
                f"Shape: {self._data.shape[0]} samples × {self._data.shape[1]} variables",
                f"Has missing values: {self.has_missing}",
                "",
                "Row Labels:",
                ", ".join(self._row_labels[:5]) + 
                ("..." if len(self._row_labels) > 5 else ""),
                "",
                "Column Labels:",
                ", ".join(self._col_labels[:5]) + 
                ("..." if len(self._col_labels) > 5 else ""),
            ]
            return "\n".join(lines)


class DataMatrixView:
    """
    A view into a subset of a DataMatrix.
    
    This class provides a lightweight view that references the parent
    DataMatrix's data, allowing efficient subset operations without
    copying data.
    
    Warning:
        Modifications to this view affect the parent DataMatrix.
        Use DataMatrix.subset_* methods for creating independent copies.
    """
    
    def __init__(
        self,
        data: npt.NDArray,
        row_labels: List[str],
        col_labels: List[str],
        parent: DataMatrix
    ) -> None:
        """
        Initialize a DataMatrixView.
        
        Parameters:
            data: View data array
            row_labels: Row labels for the view
            col_labels: Column labels for the view
            parent: Reference to parent DataMatrix
        """
        self._data = data
        self._row_labels = row_labels
        self._col_labels = col_labels
        self._parent = parent
    
    @property
    def data(self) -> npt.NDArray:
        return self._data
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape
    
    @property
    def n_samples(self) -> int:
        return self._data.shape[0]
    
    @property
    def n_variables(self) -> int:
        return self._data.shape[1]
    
    @property
    def row_labels(self) -> List[str]:
        return list(self._row_labels)
    
    @property
    def col_labels(self) -> List[str]:
        return list(self._col_labels)
    
    def to_matrix(self) -> DataMatrix:
        """
        Convert view to independent DataMatrix.
        
        Returns:
            DataMatrix: Independent copy of this view's data
        """
        return DataMatrix(
            data=self._data.copy(),
            row_labels=self._row_labels.copy(),
            col_labels=self._col_labels.copy(),
            name="View_converted"
        )
    
    def __repr__(self) -> str:
        return (
            f"DataMatrixView(shape={self._data.shape}, "
            f"parent='{self._parent.name}')"
        )
