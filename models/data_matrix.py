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

Why Paleontological Data Requires Metadata:
==============================================================================
Paleontological specimens carry essential contextual information that cannot be
captured in a simple numeric matrix alone. Each specimen (row) represents a
fossil individual with unique provenance:

    - Collector: Who collected the specimen (field researcher name)
    - Stratigraphic position: Geological formation, age, and position within strata
    - Specimen number: Museum/institution catalog number (e.g., "AMNH-F-12345")
    - Taxonomic identification: Species name, with confidence levels
    - Geographic location: Site name, GPS coordinates, paleocoordinates
    - Taphonomic notes: Preservation state, diagenetic alterations

This metadata is CRITICAL because:
1. The same morphological character may vary due to ontogeny, geography, or preservation
2. Phylogenetic analysis depends on correct taxon sampling
3. Stratigraphic constraints inform divergence time estimation
4. Museum specimen numbers enable reproducibility and museum visits

References:
    - Maddison et al. (1997) NEXUS format. Syst. Biol. 46(4):590-621
    - DeQueiroz et al. (2001) The duties of the taxonomic journal. Syst. Biol. 50:847-849

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import threading
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from utils.exceptions import (
    MatrixDimensionError,
)
from utils.validators import check_missing_values, validate_data_array

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
        data: Union[npt.NDArray, list[list[float]]],
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        name: str = "Unnamed",
        metadata: dict[str, Any] | None = None,
        specimen_metadata: list[dict[str, Any]] | None = None,
        column_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize a DataMatrix instance.

        Parameters:
            data: 2D array-like data of shape (n_samples, n_variables)
            row_labels: Optional list of row labels (sample names)
            col_labels: Optional list of column labels (variable names)
            name: Name for this data matrix
            metadata: Optional dictionary of general matrix metadata
            specimen_metadata: Optional list of per-specimen metadata dicts.
                Length must match n_samples. Each dict contains fields like
                'collector', 'stratigraphy', 'specimen_number', 'taxonomy', etc.
            column_metadata: Optional dict mapping column label to metadata dict.
                Each column metadata dict may contain 'description', 'units',
                'data_type', 'coding_scheme', etc.

        Raises:
            DataValidationError: If data validation fails
            MatrixDimensionError: If dimensions are invalid

        Example:
            >>> meta = {"project": "Cambrian Explosion", "analyst": "Dr. Smith"}
            >>> spec_meta = [{"specimen_id": "AMNH-001", "formation": "Burgess Shale"}]
            >>> col_meta = {"Var_1": {"description": "Carapace length", "units": "mm"}}
            >>> matrix = DataMatrix([[1.0, 2.0]], metadata=meta,
            ...                     specimen_metadata=spec_meta,
            ...                     column_metadata=col_meta)
        """
        # Convert to numpy array and validate
        self._data = validate_data_array(data, allow_nan=True, allow_inf=False, name="data_matrix")

        n_samples, n_variables = self._data.shape

        # Initialize or validate row labels
        if row_labels is None:
            self._row_labels = [f"Sample_{i + 1}" for i in range(n_samples)]
        else:
            if len(row_labels) != n_samples:
                raise MatrixDimensionError(
                    "Row label count must match number of samples",
                    details={"n_samples": n_samples, "n_labels": len(row_labels)},
                )
            self._row_labels = list(row_labels)

        # Initialize or validate column labels
        if col_labels is None:
            self._col_labels = [f"Var_{j + 1}" for j in range(n_variables)]
        else:
            if len(col_labels) != n_variables:
                raise MatrixDimensionError(
                    "Column label count must match number of variables",
                    details={"n_variables": n_variables, "n_labels": len(col_labels)},
                )
            self._col_labels = list(col_labels)

        # Set name
        self._name = name

        # Initialize metadata
        self._metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self._specimen_metadata: list[dict[str, Any]] = self._init_specimen_metadata(
            n_samples, specimen_metadata
        )
        self._column_metadata: dict[str, dict[str, Any]] = self._init_column_metadata(
            n_variables, column_metadata
        )

        # Thread lock for concurrent access
        self._lock = threading.RLock()

        self._logger = logging.getLogger(f"{__name__}.DataMatrix")
        self._logger.info(f"DataMatrix initialized: shape=({n_samples} x {n_variables}), name='{name}'")

    def _init_specimen_metadata(
        self, n_samples: int, specimen_metadata: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Initialize specimen metadata with validation."""
        if specimen_metadata is None:
            return [{} for _ in range(n_samples)]
        if len(specimen_metadata) != n_samples:
            raise MatrixDimensionError(
                "Specimen metadata length must match number of samples",
                details={"n_samples": n_samples, "n_spec_meta": len(specimen_metadata)},
            )
        return [dict(m) for m in specimen_metadata]  # Deep copy

    def _init_column_metadata(
        self, n_variables: int, column_metadata: dict[str, dict[str, Any]] | None
    ) -> dict[str, dict[str, Any]]:
        """Initialize column metadata with validation."""
        if column_metadata is None:
            return {label: {} for label in self._col_labels}
        # Validate that all keys correspond to existing columns
        result: dict[str, dict[str, Any]] = {}
        for i, label in enumerate(self._col_labels):
            if label in column_metadata:
                result[label] = dict(column_metadata[label])
            else:
                result[label] = {}
        return result

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
    def shape(self) -> tuple[int, int]:
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
    def row_labels(self) -> list[str]:
        """
        Get the row labels.

        Returns:
            List[str]: Copy of row labels
        """
        with self._lock:
            return list(self._row_labels)

    @row_labels.setter
    def row_labels(self, labels: list[str]) -> None:
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
                    details={"n_samples": self._data.shape[0], "n_labels": len(labels)},
                )
            self._row_labels = list(labels)

    @property
    def col_labels(self) -> list[str]:
        """
        Get the column labels.

        Returns:
            List[str]: Copy of column labels
        """
        with self._lock:
            return list(self._col_labels)

    @col_labels.setter
    def col_labels(self, labels: list[str]) -> None:
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
                    details={"n_variables": self._data.shape[1], "n_labels": len(labels)},
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
            # Return an explicit copy. ``np.isnan`` returns a view of the
            # underlying data; callers that mutate the mask (e.g. via
            # in-place boolean assignment) would otherwise corrupt
            # ``self._data`` itself.
            return np.isnan(self._data).copy()

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
    def missing_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about missing values.

        Returns:
            Dict: Missing value statistics
        """
        with self._lock:
            return check_missing_values(self._data, report_positions=False)

    # =========================================================================
    # Metadata Properties
    # =========================================================================

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get the general matrix metadata.

        Returns:
            Dict[str, Any]: Copy of matrix metadata
        """
        with self._lock:
            return dict(self._metadata)

    @metadata.setter
    def metadata(self, metadata: dict[str, Any]) -> None:
        """
        Set the general matrix metadata.

        Parameters:
            metadata: New metadata dictionary
        """
        with self._lock:
            self._metadata = dict(metadata)

    @property
    def specimen_metadata(self) -> list[dict[str, Any]]:
        """
        Get the per-specimen metadata.

        Each entry corresponds to a row (sample) in the matrix.

        Returns:
            List[Dict[str, Any]]: Copy of specimen metadata list
        """
        with self._lock:
            return [dict(m) for m in self._specimen_metadata]

    @specimen_metadata.setter
    def specimen_metadata(self, specimen_metadata: list[dict[str, Any]]) -> None:
        """
        Set the per-specimen metadata.

        Parameters:
            specimen_metadata: New list of metadata dicts

        Raises:
            MatrixDimensionError: If length doesn't match n_samples
        """
        with self._lock:
            if len(specimen_metadata) != self._data.shape[0]:
                raise MatrixDimensionError(
                    "Specimen metadata length must match number of samples",
                    details={"n_samples": self._data.shape[0], "n_spec_meta": len(specimen_metadata)},
                )
            self._specimen_metadata = [dict(m) for m in specimen_metadata]

    @property
    def column_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Get the per-column (variable) metadata.

        Returns:
            Dict[str, Dict[str, Any]]: Copy of column metadata
        """
        with self._lock:
            return {k: dict(v) for k, v in self._column_metadata.items()}

    @column_metadata.setter
    def column_metadata(self, column_metadata: dict[str, dict[str, Any]]) -> None:
        """
        Set the column metadata.

        Parameters:
            column_metadata: New column metadata dict

        Raises:
            MatrixDimensionError: If keys don't match column labels
        """
        with self._lock:
            missing = set(self._col_labels) - set(column_metadata.keys())
            extra = set(column_metadata.keys()) - set(self._col_labels)
            if missing or extra:
                raise MatrixDimensionError(
                    "Column metadata keys must match existing column labels",
                    details={"missing_columns": list(missing), "extra_columns": list(extra)},
                )
            self._column_metadata = {k: dict(v) for k, v in column_metadata.items()}

    def get_specimen_metadata(self, specimen_id: Union[int, str]) -> dict[str, Any]:
        """
        Get metadata for a specific specimen.

        Parameters:
            specimen_id: Integer index or row label string

        Returns:
            Dict[str, Any]: Specimen metadata

        Raises:
            IndexError: If specimen_id is out of range
            ValueError: If specimen_id is invalid string label
        """
        with self._lock:
            if isinstance(specimen_id, int):
                if 0 <= specimen_id < len(self._specimen_metadata):
                    return dict(self._specimen_metadata[specimen_id])
                raise IndexError(f"Specimen index {specimen_id} out of range [0, {len(self._specimen_metadata)}])")
            else:
                # Try as row label
                try:
                    idx = self._row_labels.index(specimen_id)
                    return dict(self._specimen_metadata[idx])
                except ValueError:
                    raise ValueError(f"Specimen label '{specimen_id}' not found in row labels")

    def set_specimen_metadata(
        self, specimen_id: Union[int, str], key: str, value: Any
    ) -> None:
        """
        Set a specific metadata field for a specimen.

        Parameters:
            specimen_id: Integer index or row label string
            key: Metadata key to set
            value: Metadata value

        Raises:
            IndexError: If specimen_id is out of range
            ValueError: If specimen_id is invalid string label
        """
        with self._lock:
            if isinstance(specimen_id, int):
                if 0 <= specimen_id < len(self._specimen_metadata):
                    self._specimen_metadata[specimen_id][key] = value
                    return
                raise IndexError(f"Specimen index {specimen_id} out of range [0, {len(self._specimen_metadata)}])")
            else:
                try:
                    idx = self._row_labels.index(specimen_id)
                    self._specimen_metadata[idx][key] = value
                    return
                except ValueError:
                    raise ValueError(f"Specimen label '{specimen_id}' not found in row labels")

    def get_column_metadata(self, column: Union[int, str]) -> dict[str, Any]:
        """
        Get metadata for a specific column (variable).

        Parameters:
            column: Integer index or column label string

        Returns:
            Dict[str, Any]: Column metadata

        Raises:
            IndexError: If column index is out of range
            KeyError: If column label is not found
        """
        with self._lock:
            if isinstance(column, int):
                if 0 <= column < len(self._col_labels):
                    label = self._col_labels[column]
                    return dict(self._column_metadata[label])
                raise IndexError(f"Column index {column} out of range [0, {len(self._col_labels)}])")
            else:
                if column in self._column_metadata:
                    return dict(self._column_metadata[column])
                raise KeyError(f"Column label '{column}' not found")

    # =========================================================================
    # Data Access Methods
    # =========================================================================

    def __getitem__(
        self, key: Union[int, slice, tuple[Union[int, slice], ...]]
    ) -> Union[float, npt.NDArray, "DataMatrix"]:
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
                if result.ndim == 0 or (isinstance(key, tuple) and isinstance(key[0], int) and isinstance(key[1], int)):
                    return float(result)
                else:
                    return self._create_view(result, key)
            return result

    def _create_view(self, data: npt.NDArray, key: tuple) -> "DataMatrixView":
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

        view = DataMatrixView(data=data, row_labels=row_labels, col_labels=col_labels, parent=self)
        return view

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
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

    def get_row_by_label(self, label: str) -> npt.NDArray | None:
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

    def get_column_by_label(self, label: str) -> npt.NDArray | None:
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

    def transpose(self) -> "DataMatrix":
        """
        Transpose the data matrix.

        Mathematical Operation:
            X^T where (X^T)_ij = X_ji

        Note:
            Transposition swaps rows and columns:
            - Original specimen_metadata (per-row) becomes column_metadata (per-column)
            - Original column_metadata (per-column) becomes specimen_metadata (per-row)

        Returns:
            DataMatrix: Transposed matrix
        """
        with self._lock:
            # Original specimen_metadata (per-row) becomes new column_metadata (per-column)
            # Original col_labels[i] becomes new column label, so use specimen_metadata[i]
            new_column_meta: dict[str, dict[str, Any]] = {
                row_label: dict(self._specimen_metadata[i])
                for i, row_label in enumerate(self._row_labels)
            }

            # Original column_metadata (per-column) becomes new specimen_metadata (per-row)
            # Original col_labels[i] becomes new row label
            new_specimen_meta: list[dict[str, Any]] = [
                dict(self._column_metadata.get(col_label, {}))
                for col_label in self._col_labels
            ]

            return DataMatrix(
                data=self._data.T.copy(),
                row_labels=self._col_labels.copy(),
                col_labels=self._row_labels.copy(),
                name=f"{self._name}_transposed",
                metadata=self._metadata.copy(),
                specimen_metadata=new_specimen_meta,
                column_metadata=new_column_meta,
            )

    def subset_rows(self, indices: Union[list[int], npt.NDArray]) -> "DataMatrix":
        """
        Create a new DataMatrix with only the specified rows.

        Parameters:
            indices: List or array of row indices to include

        Returns:
            DataMatrix: Matrix with subset of rows, preserving specimen metadata
        """
        with self._lock:
            if isinstance(indices, list):
                indices = np.array(indices)

            self._logger.debug(f"subset_rows: selecting {len(indices)} rows from {self._data.shape[0]}")

            new_data = self._data[indices].copy()
            new_row_labels = [self._row_labels[i] for i in indices]
            new_spec_meta = [self._specimen_metadata[i].copy() for i in indices]

            return DataMatrix(
                data=new_data,
                row_labels=new_row_labels,
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_rows_subset",
                metadata=self._metadata.copy(),
                specimen_metadata=new_spec_meta,
                column_metadata={k: v.copy() for k, v in self._column_metadata.items()},
            )

    def subset_columns(self, indices: Union[list[int], npt.NDArray]) -> "DataMatrix":
        """
        Create a new DataMatrix with only the specified columns.

        Parameters:
            indices: List or array of column indices to include

        Returns:
            DataMatrix: Matrix with subset of columns, preserving column metadata
        """
        with self._lock:
            if isinstance(indices, list):
                indices = np.array(indices)

            self._logger.debug(f"subset_columns: selecting {len(indices)} columns from {self._data.shape[1]}")

            new_data = self._data[:, indices].copy()
            new_col_labels = [self._col_labels[i] for i in indices]
            new_col_meta = {self._col_labels[i]: self._column_metadata[self._col_labels[i]].copy() for i in indices}

            return DataMatrix(
                data=new_data,
                row_labels=self._row_labels.copy(),
                col_labels=new_col_labels,
                name=f"{self._name}_cols_subset",
                metadata=self._metadata.copy(),
                specimen_metadata=[m.copy() for m in self._specimen_metadata],
                column_metadata=new_col_meta,
            )

    def remove_constant_columns(self) -> "DataMatrix":
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

    def remove_rows_with_missing(self, threshold: float = 1.0) -> "DataMatrix":
        """
        Remove rows with excessive missing values.

        Parameters:
            threshold: Fraction of columns that can be missing (0 to 1, exclusive).
                      A row is dropped if its NaN count is strictly greater
                      than ``threshold * n_variables``.
                      - 0.0 keeps only fully complete rows.
                      - 1.0 (default) keeps rows with at most ``n_variables - 1``
                        NaN values, i.e. drops only rows that are entirely NaN.

        Returns:
            DataMatrix: Matrix without rows exceeding threshold
        """
        with self._lock:
            nan_counts = np.sum(np.isnan(self._data), axis=1)
            # Cap threshold below 1.0 so threshold == 1.0 means "drop rows that
            # are entirely NaN" (matching the documented behaviour).
            max_allowed_nan = int(threshold * self.n_variables)
            if threshold >= 1.0:
                max_allowed_nan = self.n_variables - 1

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

    def impute_mean(self) -> "DataMatrix":
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

            # Handle all-NaN columns explicitly: nanmean returns NaN for them
            # (with a RuntimeWarning), which would leave the column untouched.
            # Fall back to 0 for those columns.
            all_nan_cols = np.all(nan_mask, axis=0)
            if np.any(all_nan_cols):
                self._logger.warning(
                    f"impute_mean: {int(np.sum(all_nan_cols))} all-NaN column(s) detected, falling back to 0"
                )
                col_means = np.where(all_nan_cols, 0.0, col_means)

            for j in range(result.shape[1]):
                result[nan_mask[:, j], j] = col_means[j]

            return DataMatrix(
                data=result,
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_mean_imputed",
                metadata=self._metadata.copy(),
                specimen_metadata=[m.copy() for m in self._specimen_metadata],
                column_metadata={k: v.copy() for k, v in self._column_metadata.items()},
            )

    def impute_median(self) -> "DataMatrix":
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

            # Handle all-NaN columns: nanmedian returns NaN for them.
            all_nan_cols = np.all(nan_mask, axis=0)
            if np.any(all_nan_cols):
                self._logger.warning(
                    f"impute_median: {int(np.sum(all_nan_cols))} all-NaN column(s) detected, falling back to 0"
                )
                col_medians = np.where(all_nan_cols, 0.0, col_medians)

            for j in range(result.shape[1]):
                result[nan_mask[:, j], j] = col_medians[j]

            return DataMatrix(
                data=result,
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_median_imputed",
                metadata=self._metadata.copy(),
                specimen_metadata=[m.copy() for m in self._specimen_metadata],
                column_metadata={k: v.copy() for k, v in self._column_metadata.items()},
            )

    def impute_knn(self, k: int = 5) -> "DataMatrix":
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
                result[idx]
                row_nan = nan_mask[idx]

                if not np.any(row_nan):
                    continue

                # Get non-NaN values for this row
                valid_cols = ~row_nan
                if np.sum(valid_cols) == 0:
                    # All values missing, use mean of complete rows
                    for j in np.where(row_nan)[0]:
                        col_mean = np.nanmean(complete_data[:, j]) if complete_data.size > 0 else 0.0
                        result[idx, j] = col_mean if not np.isnan(col_mean) else 0.0
                    continue

                # Compute distances using valid columns only
                dists = cdist(result[idx, valid_cols].reshape(1, -1), complete_data[:, valid_cols])[0]

                # Get k nearest neighbors
                neighbor_indices = np.argsort(dists)[:k]
                neighbors = complete_data[neighbor_indices]

                # Impute each missing column
                for j in np.where(row_nan)[0]:
                    col_mean = np.nanmean(neighbors[:, j])
                    result[idx, j] = col_mean if not np.isnan(col_mean) else 0.0

            return DataMatrix(
                data=result,
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=f"{self._name}_knn_imputed",
                metadata=self._metadata.copy(),
                specimen_metadata=[m.copy() for m in self._specimen_metadata],
                column_metadata={k: v.copy() for k, v in self._column_metadata.items()},
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> "DataMatrix":
        """
        Create a deep copy of this DataMatrix.

        Preserves all metadata (matrix, specimen, and column).

        Returns:
            DataMatrix: Independent copy of this matrix
        """
        with self._lock:
            return DataMatrix(
                data=self._data.copy(),
                row_labels=self._row_labels.copy(),
                col_labels=self._col_labels.copy(),
                name=self._name,
                metadata=self._metadata.copy(),
                specimen_metadata=[m.copy() for m in self._specimen_metadata],
                column_metadata={k: v.copy() for k, v in self._column_metadata.items()},
            )

    def to_numpy(self) -> npt.NDArray:
        """
        Convert to numpy array.

        Returns:
            npt.NDArray: Data as numpy array
        """
        with self._lock:
            return self._data.copy()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Includes all data, labels, and metadata for complete serialization.

        Returns:
            Dict: Dictionary with all data and metadata

        Example:
            >>> matrix = DataMatrix([[1.0, 2.0]], row_labels=['A'],
            ...                     metadata={'project': 'Test'},
            ...                     specimen_metadata=[{'id': 'S1'}])
            >>> d = matrix.to_dict()
            >>> d['metadata']
            {'project': 'Test'}
            >>> d['specimen_metadata']
            [{'id': 'S1'}]
        """
        with self._lock:
            return {
                "name": self._name,
                "data": self._data.tolist(),
                "row_labels": self._row_labels,
                "col_labels": self._col_labels,
                "shape": self._data.shape,
                "has_missing": self.has_missing,
                "missing_info": self.missing_info,
                "metadata": dict(self._metadata),
                "specimen_metadata": [dict(m) for m in self._specimen_metadata],
                "column_metadata": {k: dict(v) for k, v in self._column_metadata.items()},
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataMatrix":
        """
        Create a DataMatrix from a dictionary representation.

        Parameters:
            data: Dictionary with keys 'data', 'row_labels', 'col_labels',
                  and optionally 'metadata', 'specimen_metadata', 'column_metadata'

        Returns:
            DataMatrix: New instance reconstructed from dict

        Raises:
            KeyError: If required keys are missing

        Example:
            >>> d = {
            ...     'name': 'Test',
            ...     'data': [[1.0, 2.0], [3.0, 4.0]],
            ...     'row_labels': ['A', 'B'],
            ...     'col_labels': ['X', 'Y'],
            ...     'metadata': {'project': 'Test'},
            ...     'specimen_metadata': [{'id': 'A'}, {'id': 'B'}],
            ...     'column_metadata': {'X': {'units': 'mm'}}
            ... }
            >>> matrix = DataMatrix.from_dict(d)
        """
        required_keys = {"data", "row_labels", "col_labels"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f"Missing required keys in dictionary: {missing}")

        return cls(
            data=data["data"],
            row_labels=data["row_labels"],
            col_labels=data["col_labels"],
            name=data.get("name", "Unnamed"),
            metadata=data.get("metadata"),
            specimen_metadata=data.get("specimen_metadata"),
            column_metadata=data.get("column_metadata"),
        )

    def __repr__(self) -> str:
        """
        String representation of the DataMatrix.

        Returns:
            str: Human-readable description
        """
        with self._lock:
            return f"DataMatrix(name='{self._name}', shape={self._data.shape}, missing={self.has_missing})"

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
                ", ".join(self._row_labels[:5]) + ("..." if len(self._row_labels) > 5 else ""),
                "",
                "Column Labels:",
                ", ".join(self._col_labels[:5]) + ("..." if len(self._col_labels) > 5 else ""),
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

    def __init__(self, data: npt.NDArray, row_labels: list[str], col_labels: list[str], parent: DataMatrix) -> None:
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
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def n_samples(self) -> int:
        return self._data.shape[0]

    @property
    def n_variables(self) -> int:
        return self._data.shape[1]

    @property
    def row_labels(self) -> list[str]:
        return list(self._row_labels)

    @property
    def col_labels(self) -> list[str]:
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
            name="View_converted",
        )

    def __repr__(self) -> str:
        return f"DataMatrixView(shape={self._data.shape}, parent='{self._parent.name}')"
