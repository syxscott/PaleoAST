# =============================================================================
# FILE: utils/__init__.py
# =============================================================================
"""
PaleoAST Utilities Package

This package contains utility modules for matrix operations, data validation,
exception handling, decorators, and parallel computing.

Author: PaleoAST Development Team
Version: 1.0.0
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
    # Exceptions
    "PaleoASTError",
    "DataValidationError",
    "MatrixDimensionError",
    "ConvergenceError",
    "InvalidDataTypeError",
    "FileFormatError",
    "FileOperationError",
    "ComputationError",
    "MorphometricsError",
    "PlottingError",
    "StatisticalError",
    "ValidationError",
    # Matrix operations
    "ensure_matrix",
    "validate_matrix_shape",
    "center_matrix",
    "standardize_matrix",
    "covariance_matrix",
    "correlation_matrix",
    "mahalanobis_distance",
    "euclidean_distance_matrix",
    "pairwise_distances",
    # Validators
    "validate_data_array",
    "validate_column_metadata",
    "validate_row_labels",
    "validate_distance_metric",
    "check_missing_values",
    "check_infinite_values",
    "check_constant_columns",
    # Decorators
    "thread_safe",
    "memoize",
    "log_execution_time",
    "validate_inputs",
    "cache_result",
]
