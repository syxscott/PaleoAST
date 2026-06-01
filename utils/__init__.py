# =============================================================================
# FILE: utils/__init__.py
# =============================================================================
"""
PaleoAST Utilities Package

This package contains utility modules for matrix operations, data validation,
exception handling, decorators, and parallel computing.

Author: PaleoAST Development Team
version: 1.0.1
"""

from .decorators import (
    cache_result,
    log_execution_time,
    memoize,
    thread_safe,
    validate_inputs,
)
from .exceptions import (
    ComputationError,
    ConvergenceError,
    DataValidationError,
    FileFormatError,
    FileOperationError,
    InvalidDataTypeError,
    MatrixDimensionError,
    MorphometricsError,
    PaleoASTError,
    PlottingError,
    StatisticalError,
    ValidationError,
)
from .matrix_ops import (
    center_matrix,
    correlation_matrix,
    covariance_matrix,
    ensure_matrix,
    euclidean_distance_matrix,
    mahalanobis_distance,
    pairwise_distances,
    standardize_matrix,
    validate_matrix_shape,
)
from .validators import (
    check_constant_columns,
    check_infinite_values,
    check_missing_values,
    validate_column_metadata,
    validate_data_array,
    validate_distance_metric,
    validate_row_labels,
)

__all__ = [
    "ComputationError",
    "ConvergenceError",
    "DataValidationError",
    "FileFormatError",
    "FileOperationError",
    "InvalidDataTypeError",
    "MatrixDimensionError",
    "MorphometricsError",
    # Exceptions
    "PaleoASTError",
    "PlottingError",
    "StatisticalError",
    "ValidationError",
    "cache_result",
    "center_matrix",
    "check_constant_columns",
    "check_infinite_values",
    "check_missing_values",
    "correlation_matrix",
    "covariance_matrix",
    # Matrix operations
    "ensure_matrix",
    "euclidean_distance_matrix",
    "log_execution_time",
    "mahalanobis_distance",
    "memoize",
    "pairwise_distances",
    "standardize_matrix",
    # Decorators
    "thread_safe",
    "validate_column_metadata",
    # Validators
    "validate_data_array",
    "validate_distance_metric",
    "validate_inputs",
    "validate_matrix_shape",
    "validate_row_labels",
]
