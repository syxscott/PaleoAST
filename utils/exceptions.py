# =============================================================================
# FILE: utils/exceptions.py
# =============================================================================
"""
Custom Exception Classes for PaleoAST

This module defines all custom exception classes used throughout the
application for precise error handling and informative error messages.

Exception Hierarchy:
    PaleoASTError (base)
    ├── DataValidationError
    ├── MatrixDimensionError
    ├── ConvergenceError
    ├── InvalidDataTypeError
    ├── FileFormatError
    └── ComputationError

Author: PaleoAST Development Team
Version: 1.0.0
"""

from typing import Optional, Any, Dict


class PaleoASTError(Exception):
    """
    Base exception class for all PaleoAST-specific errors.
    
    This serves as the parent class for all custom exceptions,
    allowing users to catch any PaleoAST-related error with
    a single exception handler.
    
    Attributes:
        message (str): Human-readable error message
        details (Dict[str, Any]): Additional error context details
        original_exception (Optional[Exception]): The original exception if
            this exception wraps another exception
    
    Example:
        >>> try:
        ...     raise PaleoASTError("Test error", details={"context": "testing"})
        ... except PaleoASTError as e:
        ...     print(f"Caught PaleoAST error: {e}")
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Initialize a PaleoASTError instance.
        
        Parameters:
            message: The primary error message describing what went wrong.
            details: Optional dictionary of additional context for the error.
            original_exception: Optional underlying exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        """
        Return a string representation of the exception.
        
        Returns:
            str: The error message, potentially with additional context.
        """
        base_message = self.message
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_message = f"{base_message} ({detail_str})"
        if self.original_exception:
            base_message = f"{base_message}\nCaused by: {self.original_exception}"
        return base_message
    
    def get_full_traceback(self) -> str:
        """
        Generate a full error traceback including original exception.
        
        Returns:
            str: Formatted error message with full context for debugging.
        """
        lines = [
            "=" * 60,
            "PaleoAST Error Traceback",
            "=" * 60,
            f"Error Type: {self.__class__.__name__}",
            f"Message: {self.message}",
        ]
        if self.details:
            lines.append("Additional Details:")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")
        if self.original_exception:
            import traceback
            lines.append("Original Exception:")
            lines.append(traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            ))
        lines.append("=" * 60)
        return "\n".join(lines)


class DataValidationError(PaleoASTError):
    """
    Exception raised when data validation fails.
    
    This exception is raised when input data does not meet the required
    validation criteria, such as containing invalid values, wrong data types,
    or violating business rules.
    
    Common Causes:
        - Missing required columns in input data
        - Invalid data type in a column
        - Values outside expected range
        - Corrupt or malformed data
    
    Example:
        >>> raise DataValidationError(
        ...     "Column 'age' contains negative values",
        ...     details={"column": "age", "invalid_count": 5}
        ... )
    """
    pass


class MatrixDimensionError(PaleoASTError):
    """
    Exception raised when matrix dimensions are incompatible.
    
    This exception is raised during matrix operations when the dimensions
    of input matrices do not match the requirements of the operation.
    
    Mathematical Context:
        For matrix multiplication A @ B, requires A.shape[1] == B.shape[0]
        For matrix addition A + B, requires A.shape == B.shape
    
    Attributes:
        expected_shape: The expected matrix dimensions (n_rows, n_cols)
        actual_shape: The actual matrix dimensions received
        operation: The operation that was being attempted
    
    Example:
        >>> raise MatrixDimensionError(
        ...     "Cannot multiply matrices with incompatible dimensions",
        ...     details={
        ...         "expected": "(3, 4)",
        ...         "actual": "(3, 5)",
        ...         "operation": "matrix_multiplication"
        ...     }
        ... )
    """
    pass


class ConvergenceError(PaleoASTError):
    """
    Exception raised when an iterative algorithm fails to converge.
    
    This exception is raised when numerical algorithms that require
    iterative convergence (such as PCA, NMDS, GPA) fail to reach the
    convergence criteria within the maximum number of iterations.
    
    Mathematical Context:
        Most iterative algorithms check convergence using a tolerance:
        ||x_new - x_old|| < tolerance
        
        Where tolerance is typically set to 1e-6 or 1e-8 for
        numerical stability in scientific computing.
    
    Attributes:
        algorithm: Name of the algorithm that failed to converge
        iterations: Number of iterations performed before failure
        max_iterations: Maximum allowed iterations
        final_tolerance: The tolerance value at termination
    
    Example:
        >>> raise ConvergenceError(
        ...     "NMDS failed to converge after 500 iterations",
        ...     details={
        ...         "algorithm": "NMDS",
        ...         "iterations": 500,
        ...         "max_iterations": 500,
        ...         "final_stress": 0.15,
        ...         "target_stress": 0.01
        ...     }
        ... )
    """
    pass


class InvalidDataTypeError(PaleoASTError):
    """
    Exception raised when an invalid data type is specified or encountered.
    
    This exception is raised when a data type operation is attempted
    with an unsupported or invalid data type specification.
    
    Valid Data Types (from config.constants.DataType):
        - "nominal": Categorical data without order
        - "ordinal": Categorical data with meaningful order
        - "continuous": Real-valued quantitative data
        - "binary": Presence/absence (0/1) data
        - "count": Non-negative integer counts
    
    Example:
        >>> raise InvalidDataTypeError(
        ...     "Unsupported data type for this operation",
        ...     details={
        ...         "provided_type": "invalid_type",
        ...         "valid_types": ["nominal", "ordinal", "continuous"]
        ...     }
        ... )
    """
    pass


class FileFormatError(PaleoASTError):
    """
    Exception raised when file format is invalid or unsupported.
    
    This exception is raised when attempting to read or write files
    with unsupported formats, corrupt file structures, or invalid
    file contents.
    
    Supported Import Formats:
        .csv, .tsv, .txt, .xlsx, .xls, .json, .mat
    
    Supported Export Formats:
        .csv, .xlsx, .json, .png, .svg, .pdf, .tiff
    
    Example:
        >>> raise FileFormatError(
        ...     "File format not supported",
        ...     details={
        ...         "extension": ".xyz",
        ...         "supported": [".csv", ".xlsx", ".json"]
        ...     }
        ... )
    """
    pass


class ComputationError(PaleoASTError):
    """
    Exception raised when a mathematical computation fails.
    
    This exception is raised when numerical computations fail due to
    mathematical issues such as division by zero, singular matrices,
    or numerical overflow/underflow.
    
    Common Causes:
        - Division by zero in statistical calculations
        - Singular or near-singular matrix (determinant ≈ 0)
        - Numerical overflow (values too large)
        - Numerical underflow (values too small)
        - Invalid mathematical operation (e.g., log of negative)
    
    Example:
        >>> raise ComputationError(
        ...     "Cannot compute covariance matrix: matrix is singular",
        ...     details={
        ...         "operation": "covariance",
        ...         "matrix_rank": 4,
        ...         "matrix_shape": (5, 5),
        ...         "condition_number": float('inf')
        ...     }
        ... )
    """
    pass


class StatisticalError(PaleoASTError):
    """
    Exception raised when a statistical computation fails.
    
    This exception is raised specifically for statistical analysis
    failures, such as insufficient sample size, invalid test
    assumptions, or failed hypothesis tests.
    
    Example:
        >>> raise StatisticalError(
        ...     "Sample size too small for t-test",
        ...     details={
        ...         "minimum_required": 3,
        ...         "actual_size": 2
        ...     }
        ... )
    """
    pass


class MorphometricsError(PaleoASTError):
    """
    Exception raised when geometric morphometrics analysis fails.
    
    This exception is raised for errors specific to GPA, TPS,
    and other geometric morphometrics operations.
    
    Example:
        >>> raise MorphometricsError(
        ...     "Landmark configurations have different numbers of points",
        ...     details={
        ...         "config1_landmarks": 15,
        ...         "config2_landmarks": 14
        ...     }
        ... )
    """
    pass


class PlottingError(PaleoASTError):
    """
    Exception raised when visualization generation fails.
    
    This exception is raised when matplotlib operations fail
    due to invalid data, figure configuration errors, or
    export format issues.
    
    Example:
        >>> raise PlottingError(
        ...     "Cannot create scatter plot: insufficient data points",
        ...     details={"n_points": 0, "minimum_required": 1}
        ... )
    """
    pass


# =============================================================================
# COMPATIBILITY ALIASES (for backward compatibility)
# =============================================================================

# Alias for ValidationError -> DataValidationError
ValidationError = DataValidationError
"""
Alias for DataValidationError for backward compatibility.
"""

# File Operation Error
class FileOperationError(PaleoASTError):
    """
    Exception raised when file operations fail.
    
    This exception is raised when file reading, writing, or parsing
    operations fail.
    
    Example:
        >>> raise FileOperationError(
        ...     "Failed to read CSV file",
        ...     details={"file": "data.csv", "reason": "Permission denied"}
        ... )
    """
    pass
