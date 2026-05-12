# =============================================================================
# FILE: utils/validators.py
# =============================================================================
"""
Data Validation Module for PaleoAST

This module provides comprehensive validation functions for ensuring
data integrity and compatibility with statistical computations.

Validation Functions:
    - validate_data_array: Validate input data arrays
    - validate_column_metadata: Validate column metadata structures
    - validate_row_labels: Validate row label arrays
    - validate_distance_metric: Validate distance metric specifications
    - check_missing_values: Check for missing value patterns
    - check_infinite_values: Check for infinite values
    - check_constant_columns: Check for constant/variance-zero columns

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
from collections.abc import Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from .exceptions import DataValidationError, InvalidDataTypeError

logger = logging.getLogger(__name__)


def validate_data_array(
    data: Union[npt.NDArray, list, tuple],
    allow_nan: bool = True,
    allow_inf: bool = False,
    min_values: int | None = None,
    dtype: np.dtype | None = None,
    name: str = "data",
    preserve_dimensions: bool = False,
) -> npt.NDArray:
    """
    Validate input data array for statistical computations.

    This function performs comprehensive validation of input data
    including type checking, NaN handling, and value constraints.

    Parameters:
        data: Input data to validate (will be converted to numpy array)
        allow_nan: Whether NaN values are permitted in the data
        allow_inf: Whether infinite values are permitted in the data
        min_values: Minimum number of non-NaN values required
        dtype: Required dtype for the array. If None, any numeric dtype
        name: Name for error messages (e.g., "X", "Y", "distance_matrix")
        preserve_dimensions: If True, 1D arrays remain 1D; if False, 1D arrays
                           are reshaped to 2D (n, 1) for consistency with
                           statistical computation conventions

    Returns:
        npt.NDArray: Validated numpy array

    Raises:
        DataValidationError: If validation fails

    Example:
        >>> validate_data_array([[1, 2], [3, 4]], allow_nan=False)
        array([[1, 2],
               [3, 4]])
        >>> validate_data_array([1, 2, np.nan], allow_nan=True, min_values=2)
        array([ 1.,  2., nan])
    """
    logger.debug(
        f"Validating data array '{name}': allow_nan={allow_nan}, allow_inf={allow_inf}, min_values={min_values}"
    )
    # Convert to numpy array
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (list, tuple)):
        try:
            arr = np.array(data)
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert '{name}' to numpy array: {e}")
            raise DataValidationError(
                f"Cannot convert {name} to numpy array", details={"error": str(e), "input_type": str(type(data))}
            )
    else:
        logger.warning(f"'{name}' must be array-like, got {type(data)}")
        raise DataValidationError(
            f"{name} must be array-like (list, tuple, or numpy array)", details={"input_type": str(type(data))}
        )

    # Check dimensionality
    if arr.ndim == 0:
        logger.warning(f"'{name}' must be at least 1-dimensional, got shape {arr.shape}")
        raise DataValidationError(f"{name} must be at least 1-dimensional", details={"shape": arr.shape})

    # Handle 1D arrays by reshaping to 2D if not preserving dimensions
    if arr.ndim == 1 and not preserve_dimensions:
        arr = arr.reshape(-1, 1)

    # Check dtype if specified
    if dtype is not None:
        if not np.issubdtype(arr.dtype, np.dtype(dtype).dtype):
            # Try to convert if numeric
            try:
                arr = arr.astype(dtype)
            except (ValueError, TypeError):
                raise DataValidationError(
                    f"{name} dtype mismatch: expected {dtype}, got {arr.dtype}",
                    details={"expected_dtype": str(dtype), "actual_dtype": str(arr.dtype)},
                )

    # Check for NaN values
    nan_mask = np.isnan(arr)
    nan_count = np.sum(nan_mask)

    if not allow_nan and nan_count > 0:
        nan_positions = np.where(nan_mask)
        logger.warning(f"'{name}' contains {nan_count} NaN value(s) but allow_nan=False")
        raise DataValidationError(
            f"{name} contains {nan_count} NaN value(s)",
            details={"nan_count": int(nan_count), "positions": list(zip(nan_positions[0][:5], nan_positions[1][:5]))},
        )

    # Check for infinite values
    inf_mask = np.isinf(arr)
    inf_count = np.sum(inf_mask)

    if not allow_inf and inf_count > 0:
        inf_positions = np.where(inf_mask)
        logger.warning(f"'{name}' contains {inf_count} infinite value(s) but allow_inf=False")
        raise DataValidationError(
            f"{name} contains {inf_count} infinite value(s)",
            details={"inf_count": int(inf_count), "positions": list(zip(inf_positions[0][:5], inf_positions[1][:5]))},
        )

    # Check minimum values requirement
    if min_values is not None:
        valid_count = np.sum(~np.isnan(arr)) if allow_nan else arr.size
        if valid_count < min_values:
            logger.warning(f"'{name}' has {valid_count} valid values, need at least {min_values}")
            raise DataValidationError(
                f"{name} must have at least {min_values} valid values",
                details={"required": min_values, "actual": valid_count},
            )

    return arr


def validate_column_metadata(
    metadata: dict[int, dict[str, Any]],
    n_cols: int,
    valid_types: list[str] | None = None,
    valid_markers: list[str] | None = None,
) -> bool:
    """
    Validate column metadata dictionary structure.

    Column metadata stores per-column properties like data type,
    grouping assignment, and visualization settings.

    Expected Structure:
        {
            0: {"type": "continuous", "group": "control", "color": "#FF0000", "marker": "o"},
            1: {"type": "nominal", "group": None, "color": "#00FF00", "marker": "s"},
            ...
        }

    Parameters:
        metadata: Dictionary mapping column index to metadata dict
        n_cols: Expected number of columns in the data matrix
        valid_types: List of valid data type strings (default: all DataType values)
        valid_markers: List of valid matplotlib marker strings

    Returns:
        bool: True if validation passes

    Raises:
        DataValidationError: If metadata structure is invalid

    Example:
        >>> metadata = {
        ...     0: {"type": "continuous", "color": "#0077BB", "marker": "o"},
        ...     1: {"type": "nominal", "color": "#EE7733", "marker": "s"}
        ... }
        >>> validate_column_metadata(metadata, n_cols=2)
        True
    """
    logger.debug(f"Validating column metadata: {len(metadata)} entries for {n_cols} columns")
    if valid_types is None:
        from config.constants import DataType

        valid_types = [DataType.NOMINAL, DataType.ORDINAL, DataType.CONTINUOUS, DataType.BINARY, DataType.COUNT]

    if valid_markers is None:
        from config.colors import CHART_MARKERS

        valid_markers = CHART_MARKERS

    # Check metadata is a dictionary
    if not isinstance(metadata, dict):
        logger.warning(f"Column metadata must be a dictionary, got {type(metadata)}")
        raise DataValidationError("Column metadata must be a dictionary", details={"type": str(type(metadata))})

    # Check each column entry
    for col_idx, col_meta in metadata.items():
        # Validate column index
        if not isinstance(col_idx, int):
            raise DataValidationError(
                f"Column index must be integer, got {type(col_idx)}", details={"column_index": str(col_idx)}
            )

        if col_idx < 0 or col_idx >= n_cols:
            raise DataValidationError(
                f"Column index {col_idx} out of range for {n_cols} columns",
                details={"column_index": col_idx, "n_cols": n_cols},
            )

        # Validate metadata dict
        if not isinstance(col_meta, dict):
            raise DataValidationError(
                f"Column {col_idx} metadata must be a dictionary", details={"type": str(type(col_meta))}
            )

        # Validate 'type' field if present
        if "type" in col_meta:
            col_type = col_meta["type"]
            if col_type not in valid_types:
                raise DataValidationError(
                    f"Invalid data type '{col_type}' for column {col_idx}",
                    details={"column": col_idx, "invalid_type": col_type, "valid_types": valid_types},
                )

        # Validate 'color' field if present
        if "color" in col_meta:
            color = col_meta["color"]
            if not _is_valid_hex_color(color):
                raise DataValidationError(
                    f"Invalid color format for column {col_idx}",
                    details={"color": str(color), "expected_format": "#RRGGBB or #RGB"},
                )

        # Validate 'marker' field if present
        if "marker" in col_meta:
            marker = col_meta["marker"]
            if marker not in valid_markers:
                raise DataValidationError(
                    f"Invalid marker '{marker}' for column {col_idx}",
                    details={"column": col_idx, "invalid_marker": marker, "valid_markers": valid_markers},
                )

    return True


def validate_row_labels(
    labels: Sequence[str] | None, n_rows: int, allow_duplicates: bool = False, allow_none: bool = True
) -> list[str] | None:
    """
    Validate row label sequence.

    Parameters:
        labels: Sequence of row labels (sample names)
        n_rows: Expected number of rows in the data matrix
        allow_duplicates: Whether duplicate labels are permitted
        allow_none: Whether None/empty labels are permitted

    Returns:
        Optional[List[str]]: Validated list of labels, or None if labels is None

    Raises:
        DataValidationError: If label validation fails

    Example:
        >>> validate_row_labels(["Sample_A", "Sample_B", "Sample_C"], n_rows=3)
        ['Sample_A', 'Sample_B', 'Sample_C']
    """
    logger.debug(
        f"Validating row labels: n_rows={n_rows}, allow_duplicates={allow_duplicates}, allow_none={allow_none}"
    )
    if labels is None:
        if not allow_none:
            logger.warning("Row labels cannot be None")
            raise DataValidationError("Row labels cannot be None")
        return None

    # Convert to list if needed
    if isinstance(labels, (list, tuple, np.ndarray)):
        label_list = list(labels)
    else:
        logger.warning(f"Row labels must be list-like, got {type(labels)}")
        raise DataValidationError(
            "Row labels must be list-like (list, tuple, or numpy array)", details={"type": str(type(labels))}
        )

    # Check length
    if len(label_list) != n_rows:
        raise DataValidationError(
            f"Row label count ({len(label_list)}) must match row count ({n_rows})",
            details={"n_labels": len(label_list), "n_rows": n_rows},
        )

    # Check for duplicates
    if not allow_duplicates:
        seen = set()
        duplicates = []
        for i, label in enumerate(label_list):
            if label in seen:
                duplicates.append((i, label))
            seen.add(label)

        if duplicates:
            logger.warning(f"Row labels contain {len(duplicates)} duplicate(s): {duplicates[:5]}")
            raise DataValidationError(
                f"Row labels contain {len(duplicates)} duplicate(s)",
                details={"duplicates": duplicates[:5]},  # Show first 5
            )

    # Check for None values
    none_count = sum(1 for l in label_list if l is None or l == "")
    if none_count > 0 and not allow_none:
        raise DataValidationError(
            f"Row labels contain {none_count} None/empty value(s)", details={"none_count": none_count}
        )

    return label_list


def validate_distance_metric(metric: str, valid_metrics: list[str] | None = None) -> str:
    """
    Validate a distance metric specification.

    Parameters:
        metric: Distance metric name (case-insensitive)
        valid_metrics: List of valid metric names. If None, uses defaults.

    Returns:
        str: Validated metric name (lowercase)

    Raises:
        InvalidDataTypeError: If metric is not valid

    Example:
        >>> validate_distance_metric("euclidean")
        'euclidean'
        >>> validate_distance_metric("BRAY_CURTIS")
        'bray_curtis'
    """
    if valid_metrics is None:
        from config.constants import DistanceMetric

        valid_metrics = [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.BRAY_CURTIS,
            DistanceMetric.JACCARD,
            DistanceMetric.CANBERRA,
            DistanceMetric.CHEBYCHEV,
        ]

    logger.debug(f"Validating distance metric: '{metric}'")
    metric_lower = metric.lower().strip()

    if metric_lower not in valid_metrics:
        logger.warning(f"Invalid distance metric: '{metric}', valid metrics: {valid_metrics}")
        raise InvalidDataTypeError(
            f"Invalid distance metric: '{metric}'", details={"provided": metric, "valid_metrics": valid_metrics}
        )

    return metric_lower


def check_missing_values(
    matrix: npt.NDArray, report_positions: bool = False, max_positions: int = 10
) -> dict[str, Any]:
    """
    Check for missing values (NaN) in a matrix.

    This function provides a comprehensive report on missing values
    including counts, proportions, and optionally positions.

    Parameters:
        matrix: Input matrix to check
        report_positions: Whether to include row/column positions of NaNs
        max_positions: Maximum number of positions to report

    Returns:
        Dict containing:
            - total_nan: Total count of NaN values
            - nan_proportion: Proportion of matrix that is NaN
            - rows_with_nan: Count of rows containing at least one NaN
            - cols_with_nan: Count of columns containing at least one NaN
            - nan_by_row: NaN count per row (if requested)
            - nan_by_col: NaN count per column (if requested)
            - positions: List of (row, col) positions of NaNs (if requested)

    Example:
        >>> import numpy as np
        >>> X = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        >>> check_missing_values(X)
        {'total_nan': 2, 'nan_proportion': 0.222..., 'rows_with_nan': 2, 'cols_with_nan': 2}
    """
    logger.debug(f"Checking for missing values in matrix of shape {matrix.shape}")
    nan_mask = np.isnan(matrix)
    total_nan = int(np.sum(nan_mask))
    total_elements = matrix.size

    # Rows and columns with NaN
    rows_with_nan = int(np.any(nan_mask, axis=1).sum())
    cols_with_nan = int(np.any(nan_mask, axis=0).sum())

    result = {
        "total_nan": total_nan,
        "nan_proportion": total_nan / total_elements if total_elements > 0 else 0,
        "rows_with_nan": rows_with_nan,
        "cols_with_nan": cols_with_nan,
    }

    # Per-row NaN counts
    result["nan_by_row"] = np.sum(nan_mask, axis=1).tolist()

    # Per-column NaN counts
    result["nan_by_col"] = np.sum(nan_mask, axis=0).tolist()

    # Position reporting
    if report_positions and total_nan > 0:
        nan_positions = np.where(nan_mask)
        positions = list(zip(nan_positions[0].tolist()[:max_positions], nan_positions[1].tolist()[:max_positions]))
        result["positions"] = positions
        if total_nan > max_positions:
            result["positions_truncated"] = True
            result["total_positions"] = total_nan

    return result


def check_infinite_values(matrix: npt.NDArray) -> dict[str, Any]:
    """
    Check for infinite values in a matrix.

    Parameters:
        matrix: Input matrix to check

    Returns:
        Dict containing:
            - has_pos_inf: Whether any positive infinity exists
            - has_neg_inf: Whether any negative infinity exists
            - total_inf: Total count of infinite values
            - positions: List of (row, col) positions (up to 10)

    Example:
        >>> import numpy as np
        >>> X = np.array([[1, np.inf, 3], [-np.inf, 5, 6]])
        >>> check_infinite_values(X)
        {'has_pos_inf': True, 'has_neg_inf': True, 'total_inf': 2, ...}
    """
    logger.debug(f"Checking for infinite values in matrix of shape {matrix.shape}")
    pos_inf_mask = np.isposinf(matrix)
    neg_inf_mask = np.isneginf(matrix)
    inf_mask = pos_inf_mask | neg_inf_mask

    has_pos_inf = bool(np.any(pos_inf_mask))
    has_neg_inf = bool(np.any(neg_inf_mask))
    total_inf = int(np.sum(inf_mask))

    result = {
        "has_pos_inf": has_pos_inf,
        "has_neg_inf": has_neg_inf,
        "total_inf": total_inf,
    }

    if total_inf > 0:
        positions = np.where(inf_mask)
        result["positions"] = list(zip(positions[0].tolist()[:10], positions[1].tolist()[:10]))

    return result


def check_constant_columns(matrix: npt.NDArray, tolerance: float = 1e-10) -> dict[str, Any]:
    """
    Identify columns with zero or near-zero variance.

    Constant columns (zero variance) cause issues in:
    - Standardization (division by zero in z-score)
    - Correlation matrix (undefined correlation)
    - PCA (zero eigenvalues)

    Mathematical Definition:
        A column is constant if: var(x) = 0
        Or approximately constant if: var(x) < tolerance

    Parameters:
        matrix: Input matrix of shape (n_samples, n_features)
        tolerance: Variance below which column is considered constant

    Returns:
        Dict containing:
            - constant_cols: List of column indices with zero variance
            - near_constant_cols: List of column indices with near-zero variance
            - variance_by_col: Variance of each column
            - std_by_col: Standard deviation of each column

    Example:
        >>> X = np.array([[1, 2, 3], [1, 4, 6], [1, 6, 9]], dtype=float)
        >>> check_constant_columns(X)
        {'constant_cols': [0], 'near_constant_cols': [], ...}
    """
    logger.debug(f"Checking for constant columns in matrix of shape {matrix.shape}, tolerance={tolerance}")
    # Compute variance for each column
    variance_by_col = np.var(matrix, axis=0, ddof=1)
    std_by_col = np.sqrt(variance_by_col)

    # Identify constant columns (exactly zero variance)
    constant_mask = variance_by_col < tolerance
    constant_cols = np.where(constant_mask)[0].tolist()

    # Identify near-constant columns (small but non-zero variance)
    near_constant_mask = (variance_by_col >= tolerance) & (variance_by_col < tolerance * 1000)
    near_constant_cols = np.where(near_constant_mask)[0].tolist()

    return {
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "variance_by_col": variance_by_col.tolist(),
        "std_by_col": std_by_col.tolist(),
        "has_constant": len(constant_cols) > 0,
        "has_near_constant": len(near_constant_cols) > 0,
    }


def _is_valid_hex_color(color: str) -> bool:
    """
    Check if a string is a valid hex color code.

    Parameters:
        color: String to check

    Returns:
        bool: True if valid hex color format
    """
    if not isinstance(color, str):
        return False

    # Remove # prefix
    color = color.lstrip("#")

    # Check length (3 or 6 hex digits)
    if len(color) not in (3, 6):
        return False

    # Check all characters are hex digits
    try:
        int(color, 16)
        return True
    except ValueError:
        return False
