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

from .exceptions import (
    PaleoASTError,
    DataValidationError,
    MatrixDimensionError,
    ConvergenceError,
    InvalidDataTypeError,
    FileFormatError,
    ComputationError,
)

from .matrix_ops import (
    ensure_matrix,
    validate_matrix_shape,
    center_matrix,
    standardize_matrix,
    covariance_matrix,
    correlation_matrix,
    mahalanobis_distance,
    euclidean_distance_matrix,
    pairwise_distances,
)

from .validators import (
    validate_data_array,
    validate_column_metadata,
    validate_row_labels,
    validate_distance_metric,
    check_missing_values,
    check_infinite_values,
    check_constant_columns,
)

from .decorators import (
    thread_safe,
    memoize,
    log_execution_time,
    validate_inputs,
    cache_result,
)

__all__ = [
    # Exceptions
    'PaleoASTError',
    'DataValidationError',
    'MatrixDimensionError',
    'ConvergenceError',
    'InvalidDataTypeError',
    'FileFormatError',
    'ComputationError',
    # Matrix operations
    'ensure_matrix',
    'validate_matrix_shape',
    'center_matrix',
    'standardize_matrix',
    'covariance_matrix',
    'correlation_matrix',
    'mahalanobis_distance',
    'euclidean_distance_matrix',
    'pairwise_distances',
    # Validators
    'validate_data_array',
    'validate_column_metadata',
    'validate_row_labels',
    'validate_distance_metric',
    'check_missing_values',
    'check_infinite_values',
    'check_constant_columns',
    # Decorators
    'thread_safe',
    'memoize',
    'log_execution_time',
    'validate_inputs',
    'cache_result',
]
