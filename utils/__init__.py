# =============================================================================
# FILE: utils/__init__.py
# =============================================================================
"""
PaleoAST Utilities Package

This package contains utility modules for matrix operations, data validation,
exception handling, decorators, and parallel computing.

Author: PaleoAST Development Team
version: 1.1.0
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
from .statistics_core import (
    MISSING_GROUP,
    NonlinearFitResult,
    PermutationResult,
    aic_from_log_likelihood,
    aicc_from_log_likelihood,
    fit_nonlinear,
    gaussian_log_likelihood,
    group_indices,
    make_rng,
    permutation_pvalue,
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
    "MISSING_GROUP",
    "ComputationError",
    "ConvergenceError",
    "DataValidationError",
    "FileFormatError",
    "FileOperationError",
    "InvalidDataTypeError",
    "MatrixDimensionError",
    "MorphometricsError",
    # Exceptions
    "NonlinearFitResult",
    "PaleoASTError",
    "PermutationResult",
    "PlottingError",
    "StatisticalError",
    "ValidationError",
    "aic_from_log_likelihood",
    "aicc_from_log_likelihood",
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
    "fit_nonlinear",
    "gaussian_log_likelihood",
    "group_indices",
    "log_execution_time",
    "mahalanobis_distance",
    "make_rng",
    "memoize",
    "pairwise_distances",
    "permutation_pvalue",
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
