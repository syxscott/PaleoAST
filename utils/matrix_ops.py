# =============================================================================
# FILE: utils/matrix_ops.py
# =============================================================================
"""
Matrix Operations Module for PaleoAST

This module provides comprehensive matrix operations required for
statistical computations in paleontological data analysis.

Mathematical Functions:

1. Center Matrix:
   Z = X - μ
   where μ is the column-wise mean vector

2. Standardize Matrix (Z-score):
   z_ij = (x_ij - μ_j) / σ_j
   where μ_j is column mean, σ_j is column std dev

3. Covariance Matrix:
   S = (1/(n-1)) * Z^T * Z
   where Z is the centered matrix

4. Correlation Matrix:
   R_ij = S_ij / (σ_i * σ_j)
   where S is covariance matrix

5. Mahalanobis Distance:
   D(x, μ, Σ) = sqrt((x - μ)^T * Σ^(-1) * (x - μ))

6. Euclidean Distance Matrix:
   D_ij = ||x_i - x_j||_2 = sqrt(Σ_k (x_ik - x_jk)^2)

Author: PaleoAST Development Team
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
import logging
from typing import Optional, Union, Tuple

from .exceptions import (
    MatrixDimensionError,
    ComputationError,
)

logger = logging.getLogger(__name__)


def ensure_matrix(
    data: Union[npt.NDArray, list, tuple],
    dtype: Optional[np.dtype] = None,
    copy: bool = False
) -> npt.NDArray:
    """
    Ensure input data is converted to a NumPy ndarray.
    
    This function provides a robust way to convert various input
    types to NumPy arrays with proper dtype handling.
    
    Parameters:
        data: Input data that can be array-like (list, tuple, ndarray)
        dtype: Optional dtype to cast the array to. If None, uses
            the inferred dtype from the input data.
        copy: If True, always return a copy of the data. If False,
            return a view when possible.
    
    Returns:
        npt.NDArray: A NumPy ndarray with the specified dtype.
    
    Raises:
        MatrixDimensionError: If data cannot be converted to a 2D array.
    
    Example:
        >>> ensure_matrix([[1, 2], [3, 4]])
        array([[1, 2],
               [3, 4]])
        >>> ensure_matrix([1, 2, 3])
        array([[1],
               [2],
               [3]])
    """
    if isinstance(data, np.ndarray):
        result = data.copy() if copy else data
    elif isinstance(data, (list, tuple)):
        result = np.array(data)
    else:
        raise MatrixDimensionError(
            f"Cannot convert {type(data)} to numpy array",
            details={"input_type": str(type(data))}
        )
    
    # Handle 1D arrays by reshaping to column vectors
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    
    if dtype is not None:
        result = result.astype(dtype, copy=copy)
    
    return result


def validate_matrix_shape(
    matrix: npt.NDArray,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    min_cols: Optional[int] = None,
    max_cols: Optional[int] = None,
    exact_shape: Optional[Tuple[int, int]] = None,
    allow_empty: bool = False
) -> bool:
    """
    Validate that a matrix meets specified shape requirements.
    
    Parameters:
        matrix: The matrix to validate.
        min_rows: Minimum number of rows required.
        max_rows: Maximum number of rows allowed.
        min_cols: Minimum number of columns required.
        max_cols: Maximum number of columns allowed.
        exact_shape: If specified, matrix must match this exact shape.
        allow_empty: Whether to allow empty matrices (0x0, 0xn, m x 0).
    
    Returns:
        bool: True if matrix meets all requirements.
    
    Raises:
        MatrixDimensionError: If matrix fails any validation check.
    
    Example:
        >>> data = np.random.rand(10, 5)
        >>> validate_matrix_shape(data, min_rows=5, min_cols=3)
        True
    """
    if not isinstance(matrix, np.ndarray):
        raise MatrixDimensionError(
            "Input must be a numpy ndarray",
            details={"input_type": str(type(matrix))}
        )
    
    n_rows, n_cols = matrix.shape
    
    # Check for empty matrix
    if not allow_empty:
        if n_rows == 0 or n_cols == 0:
            raise MatrixDimensionError(
                "Matrix cannot be empty",
                details={"shape": matrix.shape}
            )
    
    # Check exact shape if specified
    if exact_shape is not None:
        expected_rows, expected_cols = exact_shape
        if n_rows != expected_rows or n_cols != expected_cols:
            raise MatrixDimensionError(
                f"Matrix shape must be {exact_shape}",
                details={
                    "expected": exact_shape,
                    "actual": matrix.shape
                }
            )
    
    # Check row constraints
    if min_rows is not None and n_rows < min_rows:
        raise MatrixDimensionError(
            f"Matrix must have at least {min_rows} rows",
            details={
                "minimum_rows": min_rows,
                "actual_rows": n_rows
            }
        )
    
    if max_rows is not None and n_rows > max_rows:
        raise MatrixDimensionError(
            f"Matrix cannot have more than {max_rows} rows",
            details={
                "maximum_rows": max_rows,
                "actual_rows": n_rows
            }
        )
    
    # Check column constraints
    if min_cols is not None and n_cols < min_cols:
        raise MatrixDimensionError(
            f"Matrix must have at least {min_cols} columns",
            details={
                "minimum_cols": min_cols,
                "actual_cols": n_cols
            }
        )
    
    if max_cols is not None and n_cols > max_cols:
        raise MatrixDimensionError(
            f"Matrix cannot have more than {max_cols} columns",
            details={
                "maximum_cols": max_cols,
                "actual_cols": n_cols
            }
        )
    
    return True


def center_matrix(matrix: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """
    Center a matrix by subtracting the mean along specified axis.
    
    Mathematical Operation (axis=0):
        Z_ij = X_ij - μ_j
        
        where μ_j = (1/n) * Σ_i=1^n X_ij is the column-wise mean.
    
    This operation is essential for PCA and other multivariate
    techniques that require centered data.
    
    Parameters:
        matrix: Input matrix of shape (n_samples, n_features)
        axis: Axis along which to compute mean.
            0 = column-wise (subtract column means)
            1 = row-wise (subtract row means)
    
    Returns:
        npt.NDArray: Centered matrix of the same shape.
    
    Mathematical Example:
        Given X = [[2, 4], [4, 6], [6, 8]]
        Column means: μ = [4, 6]
        Centered: Z = [[-2, -2], [0, 0], [2, 2]]
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> center_matrix(X)
        array([[-2., -2.],
               [ 0.,  0.],
               [ 2.,  2.]])
    """
    logger.debug(f"Centering matrix of shape {matrix.shape} along axis {axis}")
    validate_matrix_shape(matrix, allow_empty=False)

    mean = np.mean(matrix, axis=axis, keepdims=True)
    centered = matrix - mean

    return centered


def standardize_matrix(
    matrix: npt.NDArray,
    axis: int = 0,
    ddof: int = 1
) -> npt.NDArray:
    """
    Standardize a matrix to zero mean and unit variance (Z-score).
    
    Mathematical Operation (axis=0):
        z_ij = (X_ij - μ_j) / σ_j
        
        where:
        - μ_j = (1/(n-1)) * Σ_i=1^n X_ij is the column mean
        - σ_j = sqrt((1/(n-1)) * Σ_i=1^n (X_ij - μ_j)^2 is the sample std dev
        
        Note: Using ddof=1 gives the sample standard deviation.
        Using ddof=0 gives the population standard deviation.
    
    This produces a matrix where each column has mean=0 and std=1,
    which is required for PCA based on correlation matrix and
    many machine learning algorithms.
    
    Parameters:
        matrix: Input matrix of shape (n_samples, n_features)
        axis: Axis along which to standardize.
            0 = column-wise standardization
            1 = row-wise standardization
        ddof: Delta degrees of freedom for standard deviation.
            ddof=1 gives sample std (default for statistics)
            ddof=0 gives population std
    
    Returns:
        npt.NDArray: Standardized matrix of the same shape.
    
    Mathematical Example:
        Given X = [[2, 4], [4, 6], [6, 8]]
        Column means: μ = [4, 6]
        Column stds: σ = [2, 2] (with ddof=1)
        Standardized: Z = [[-1, -1], [0, 0], [1, 1]]
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        >>> standardize_matrix(X)
        array([[-1.22474487, -1.22474487],
               [ 0.        ,  0.        ],
               [ 1.22474487,  1.22474487]])
    """
    logger.debug(f"Standardizing matrix of shape {matrix.shape} along axis {axis} with ddof={ddof}")
    validate_matrix_shape(matrix, allow_empty=False)

    mean = np.mean(matrix, axis=axis, keepdims=True)
    std = np.std(matrix, axis=axis, keepdims=True, ddof=ddof)
    
    # Handle zero standard deviation to avoid division by zero
    if np.any(std == 0):
        std = np.where(std == 0, 1, std)
    
    standardized = (matrix - mean) / std
    
    return standardized


def covariance_matrix(
    matrix: npt.NDArray,
    rowvar: bool = False,
    ddof: int = 1,
    bias: bool = False
) -> npt.NDArray:
    """
    Compute the covariance matrix of a dataset.
    
    Mathematical Definition:
        For centered matrix Z (n × p):
        S = (1/(n-1)) * Z^T * Z
        
        Element-wise:
        S_ij = cov(X_i, X_j) = (1/(n-1)) * Σ_k=1^n (X_ki - μ_i)(X_kj - μ_j)
        
        where:
        - n is the number of observations (samples)
        - p is the number of variables (features)
        - μ_i is the mean of variable i
    
    For row vectors (rowvar=True), computes covariance between rows.
    
    Parameters:
        matrix: Input array of shape (n_samples, n_features)
        rowvar: If True, each row is a variable (feature).
                If False, each column is a variable.
        ddof: Delta degrees of freedom for divisor.
            ddof=1 gives unbiased estimator (default)
            ddof=0 gives MLE
        bias: If True, uses n instead of n-1 in divisor.
            Overrides ddof if True.
    
    Returns:
        npt.NDArray: Covariance matrix of shape (p, p) if rowvar=False.
    
    Raises:
        ComputationError: If matrix contains invalid values for covariance.
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        >>> covariance_matrix(X)
        array([[4., 4.],
               [4., 4.]])
    """
    validate_matrix_shape(matrix, min_rows=2, allow_empty=False)
    n, p = matrix.shape
    if n * p > 10000:
        logger.info(f"Computing covariance matrix for large dataset: {n} samples x {p} features")
    else:
        logger.debug(f"Computing covariance matrix: {n} samples x {p} features")

    # Handle the ddof/bias interaction
    if bias:
        divisor = matrix.shape[0]
    else:
        divisor = matrix.shape[0] - ddof
    
    if divisor <= 0:
        raise ComputationError(
            "Cannot compute covariance: insufficient degrees of freedom",
            details={"n": matrix.shape[0], "ddof": ddof}
        )
    
    # Center the data
    mean = np.mean(matrix, axis=0, keepdims=True)
    centered = matrix - mean
    
    # Compute covariance using matrix multiplication
    if rowvar:
        # Treat rows as variables: centered is (p, n), result should be (p, p)
        cov = centered @ centered.T / divisor
    else:
        # Treat columns as variables: centered is (n, p), result should be (p, p)
        cov = centered.T @ centered / divisor
    
    return cov


def correlation_matrix(
    matrix: npt.NDArray,
    method: str = "pearson"
) -> npt.NDArray:
    """
    Compute the correlation matrix of a dataset.
    
    Mathematical Definition (Pearson Correlation):
        r_ij = cov(X_i, X_j) / (σ_i * σ_j)
        
        where:
        - cov(X_i, X_j) is the covariance between variables i and j
        - σ_i and σ_j are the standard deviations
        
        Expanded form:
        r_ij = Σ_k (X_ki - μ_i)(X_kj - μ_j) / 
               sqrt(Σ_k (X_ki - μ_i)^2 * Σ_k (X_kj - μ_j)^2)
        
        Properties:
        - -1 ≤ r_ij ≤ 1
        - r_ii = 1 (self-correlation)
        - r_ij = r_ji (symmetry)
    
    For standardized data (Z-scores), the correlation matrix equals
    the covariance matrix.
    
    Parameters:
        matrix: Input array of shape (n_samples, n_features)
        method: Correlation method. Currently only "pearson" supported.
    
    Returns:
        npt.NDArray: Correlation matrix of shape (n_features, n_features).
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        >>> correlation_matrix(X)
        array([[1., 1.],
               [1., 1.]])
    """
    validate_matrix_shape(matrix, min_rows=2, allow_empty=False)
    n, p = matrix.shape
    if n * p > 10000:
        logger.info(f"Computing correlation matrix for large dataset: {n} samples x {p} features")
    else:
        logger.debug(f"Computing correlation matrix: {n} samples x {p} features, method={method}")

    if method.lower() != "pearson":
        raise NotImplementedError(
            f"Correlation method '{method}' not yet implemented"
        )

    # Use NumPy's corrcoef which is optimized and stable
    # Transpose so rows are variables if rowvar convention
    corr = np.corrcoef(matrix.T)

    return corr


def mahalanobis_distance(
    x: npt.NDArray,
    mean: npt.NDArray,
    cov: npt.NDArray,
    inverted: bool = False
) -> float:
    """
    Compute the Mahalanobis distance from points to a distribution.
    
    Mathematical Definition:
        D(x, μ, Σ) = sqrt((x - μ)^T * Σ^(-1) * (x - μ))
        
        where:
        - x is the point vector (1 × p)
        - μ is the mean vector (1 × p)
        - Σ is the covariance matrix (p × p)
        - Σ^(-1) is the inverse covariance matrix (precision matrix)
    
    The Mahalanobis distance measures how many standard deviations
    a point is from the mean, accounting for correlations between
    variables. For uncorrelated variables with unit variance,
    this reduces to the Euclidean distance.
    
    Geometric Interpretation:
        - D = 1: Point lies on the covariance ellipse (1σ contour)
        - D = 2: Point lies on the 2σ contour
        - D = 3: Point lies on the 3σ contour (useful for outlier detection)
    
    Parameters:
        x: Point or points array of shape (p,) or (n, p)
        mean: Mean vector of shape (p,)
        cov: Covariance matrix of shape (p, p)
        inverted: If True, cov is already the precision matrix (Σ^(-1)).
                  If False, will compute the inverse.
    
    Returns:
        float: Mahalanobis distance. If x has multiple rows, returns
               array of distances.
    
    Example:
        >>> import numpy as np
        >>> x = np.array([2, 3])
        >>> mean = np.array([1, 2])
        >>> cov = np.array([[1, 0.5], [0.5, 1]])
        >>> mahalanobis_distance(x, mean, cov)
        2.0
    """
    logger.debug(f"Computing Mahalanobis distance: point shape={np.asarray(x).shape}, inverted={inverted}")
    x = ensure_matrix(x)
    mean = ensure_matrix(mean).flatten()
    cov = ensure_matrix(cov)

    # Handle single point case
    single_point = x.ndim == 1 or x.shape[0] == 1
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    if x.shape[1] != mean.shape[0]:
        raise MatrixDimensionError(
            "Point dimension must match mean dimension",
            details={
                "point_dim": x.shape[1],
                "mean_dim": mean.shape[0]
            }
        )
    
    if cov.shape[0] != cov.shape[1]:
        raise MatrixDimensionError(
            "Covariance matrix must be square",
            details={"cov_shape": cov.shape}
        )
    
    if cov.shape[0] != mean.shape[0]:
        raise MatrixDimensionError(
            "Covariance matrix dimension must match mean dimension",
            details={
                "cov_dim": cov.shape[0],
                "mean_dim": mean.shape[0]
            }
        )
    
    # Compute inverse if not already provided
    if not inverted:
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            raise ComputationError(
                "Covariance matrix is singular or near-singular",
                details={"cov_shape": cov.shape}
            )
    else:
        precision = cov
    
    # Center the points
    diff = x - mean
    
    # Compute Mahalanobis distance: sqrt(diff @ precision @ diff.T)
    # For multiple points, this is equivalent to:
    # sqrt(sum(diff * (diff @ precision), axis=1))
    mahal_sq = np.sum(diff @ precision * diff, axis=1)
    
    # Handle numerical precision issues
    mahal_sq = np.maximum(mahal_sq, 0)
    
    distances = np.sqrt(mahal_sq)
    
    if single_point:
        return float(distances[0])
    return distances


def euclidean_distance_matrix(
    matrix: npt.NDArray,
    squared: bool = False
) -> npt.NDArray:
    """
    Compute the pairwise Euclidean distance matrix.
    
    Mathematical Definition:
        D_ij = ||x_i - x_j||_2 = sqrt(Σ_k=1^p (x_ik - x_jk)^2)
        
        where:
        - x_i and x_j are row vectors of length p
        - ||·||_2 denotes the L2 (Euclidean) norm
        
        For squared distances (squared=True):
        D_ij² = Σ_k=1^p (x_ik - x_jk)²
    
    This is equivalent to computing the Euclidean norm of the
    difference between all pairs of row vectors.
    
    Optimization:
        D² = ||X||² - 2*X*X^T + ||X||² (vectorized)
    
    Parameters:
        matrix: Input array of shape (n_samples, n_features)
        squared: If True, return squared distances (faster computation).
                 If False, return actual Euclidean distances.
    
    Returns:
        npt.NDArray: Distance matrix of shape (n_samples, n_samples)
                     where D[i,j] = distance from row i to row j.
    
    Properties:
        - D[i,i] = 0 (distance from point to itself)
        - D[i,j] = D[j,i] (symmetry)
        - D[i,j] ≥ 0 (non-negativity)
        - D[i,j] ≤ D[i,k] + D[k,j] (triangle inequality)
    
    Example:
        >>> X = np.array([[0, 0], [3, 4], [0, 4]])
        >>> euclidean_distance_matrix(X)
        array([[0.        , 5.        , 4.        ],
               [5.        , 0.        , 3.        ],
               [4.        , 3.        , 0.        ]])
    """
    validate_matrix_shape(matrix, min_rows=2, allow_empty=False)
    n = matrix.shape[0]
    if n > 500:
        logger.info(f"Computing Euclidean distance matrix for {n} points ({n * (n - 1) // 2} pairs)")
    else:
        logger.debug(f"Computing Euclidean distance matrix for {n} points")

    # Compute squared Euclidean distances using vectorized operations
    # D_sq[i,j] = ||x_i||² - 2*x_i·x_j + ||x_j||²
    sq_norms = np.sum(matrix ** 2, axis=1)
    sq_dist = sq_norms[:, np.newaxis] - 2 * matrix @ matrix.T + sq_norms[np.newaxis, :]
    
    # Handle numerical precision (can have small negatives)
    sq_dist = np.maximum(sq_dist, 0)
    
    if squared:
        return sq_dist
    else:
        return np.sqrt(sq_dist)


def pairwise_distances(
    matrix1: npt.NDArray,
    matrix2: Optional[npt.NDArray] = None,
    metric: str = "euclidean",
    **kwargs
) -> npt.NDArray:
    """
    Compute pairwise distances between points in two arrays.
    
    This is a wrapper around scipy.spatial.distance.pdist/cdist
    with additional distance metrics specific to paleontological data.
    
    Supported Metrics:
        'euclidean': Euclidean (L2) distance
        'manhattan': Manhattan (L1) distance
        'bray_curtis': Bray-Curtis dissimilarity
        'jaccard': Jaccard distance
        'canberra': Canberra distance
        'chebychev': Chebychev (L-infinity) distance
    
    Mathematical Definitions:
        
        Manhattan (L1):
            D_ij = Σ_k |x_ik - x_jk|
        
        Bray-Curtis:
            D_ij = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)
        
        Jaccard:
            D_ij = 1 - |A∩B| / |A∪B|
            where A, B are sets of non-zero indices
    
    Parameters:
        matrix1: First array of shape (n1, p) where p is dimensionality
        matrix2: Second array of shape (n2, p). If None, computes
                distances within matrix1.
        metric: Distance metric to use (see supported metrics above)
        **kwargs: Additional arguments passed to distance functions
    
    Returns:
        npt.NDArray: 
            - If matrix2 is None: distance matrix of shape (n1, n1)
            - If matrix2 is provided: distance matrix of shape (n1, n2)
    
    Example:
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> pairwise_distances(X, metric='manhattan')
        array([[0., 2., 4.],
               [2., 0., 2.],
               [4., 2., 0.]])
    """
    matrix1 = ensure_matrix(matrix1)

    if matrix2 is not None:
        matrix2 = ensure_matrix(matrix2)

        if matrix1.shape[1] != matrix2.shape[1]:
            logger.error(f"Column mismatch: {matrix1.shape[1]} vs {matrix2.shape[1]}")
            raise MatrixDimensionError(
                "Matrices must have same number of columns",
                details={
                    "matrix1_cols": matrix1.shape[1],
                    "matrix2_cols": matrix2.shape[1]
                }
            )

        logger.info(f"Computing cross-distances: {matrix1.shape[0]} x {matrix2.shape[0]} points, metric='{metric}'")
        # Compute cross-distances using custom implementation
        return _compute_cross_distances(matrix1, matrix2, metric, **kwargs)
    else:
        n = matrix1.shape[0]
        if n > 500:
            logger.info(f"Computing pairwise distances for {n} points, metric='{metric}'")
        else:
            logger.debug(f"Computing pairwise distances for {n} points, metric='{metric}'")
        # Compute self-distances
        return _compute_self_distances(matrix1, metric, **kwargs)


def _compute_self_distances(
    matrix: npt.NDArray,
    metric: str,
    **kwargs
) -> npt.NDArray:
    """
    Compute pairwise distances within a single matrix.
    
    For internal use. Handles special metrics that need custom
    implementation for self-distances.
    """
    from scipy.spatial.distance import pdist, squareform
    
    try:
        # Try scipy first for standard metrics
        dist_condensed = pdist(matrix, metric=metric, **kwargs)
        return squareform(dist_condensed)
    except ValueError:
        # Custom metrics for paleontological data
        if metric == "bray_curtis":
            return _bray_curtis_distance_matrix(matrix)
        elif metric == "jaccard":
            return _jaccard_distance_matrix(matrix)
        else:
            raise ValueError(f"Unknown metric: {metric}")


def _compute_cross_distances(
    matrix1: npt.NDArray,
    matrix2: npt.NDArray,
    metric: str,
    **kwargs
) -> npt.NDArray:
    """
    Compute pairwise distances between two matrices.
    
    For internal use. Handles special metrics for cross-distances.
    """
    from scipy.spatial.distance import cdist
    
    try:
        # Try scipy first for standard metrics
        return cdist(matrix1, matrix2, metric=metric, **kwargs)
    except ValueError:
        # Custom metrics
        if metric == "bray_curtis":
            return _bray_curtis_cross_distance(matrix1, matrix2)
        elif metric == "jaccard":
            return _jaccard_cross_distance(matrix1, matrix2)
        else:
            raise ValueError(f"Unknown metric: {metric}")


def _bray_curtis_distance_matrix(matrix: npt.NDArray) -> npt.NDArray:
    """
    Compute Bray-Curtis distance matrix.
    
    Mathematical Definition:
        BC_ij = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)
        
        Properties:
        - BC_ij ∈ [0, 1]
        - BC_ij = 0 when x_i = x_j (identical compositions)
        - BC_ij = 1 when one sample has all zeros except one taxon
                 where the other sample has all its abundance
    
    This metric is widely used in ecological studies for
    comparing species compositions between samples.
    """
    n = matrix.shape[0]
    dist = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            numerator = np.sum(np.abs(matrix[i] - matrix[j]))
            denominator = np.sum(matrix[i] + matrix[j])
            
            if denominator > 0:
                bc = numerator / denominator
            else:
                bc = 0.0
            
            dist[i, j] = bc
            dist[j, i] = bc
    
    return dist


def _bray_curtis_cross_distance(
    matrix1: npt.NDArray,
    matrix2: npt.NDArray
) -> npt.NDArray:
    """
    Compute Bray-Curtis cross-distance between two matrices.
    """
    n1, n2 = matrix1.shape[0], matrix2.shape[0]
    dist = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            numerator = np.sum(np.abs(matrix1[i] - matrix2[j]))
            denominator = np.sum(matrix1[i] + matrix2[j])
            
            if denominator > 0:
                dist[i, j] = numerator / denominator
            else:
                dist[i, j] = 0.0
    
    return dist


def _jaccard_distance_matrix(matrix: npt.NDArray) -> npt.NDArray:
    """
    Compute Jaccard distance matrix.
    
    Mathematical Definition:
        J_ij = 1 - |A_i ∩ A_j| / |A_i ∪ A_j|
        
        where A_i and A_j are sets of non-zero element indices.
        
        For binary data (presence/absence):
        - |A_i ∩ A_j| = number of positions where both have 1
        - |A_i ∪ A_j| = number of positions where at least one has 1
        
        Equivalently:
        J_ij = (FP + FN) / (TP + FP + FN)
        
        where TP = true positives, FP = false positives, FN = false negatives.
    """
    n = matrix.shape[0]
    dist = np.zeros((n, n))
    
    # Convert to binary presence/absence
    binary = (matrix > 0).astype(int)
    
    for i in range(n):
        for j in range(i + 1, n):
            intersection = np.sum(binary[i] & binary[j])
            union = np.sum(binary[i] | binary[j])
            
            if union > 0:
                jaccard = 1 - intersection / union
            else:
                jaccard = 0.0
            
            dist[i, j] = jaccard
            dist[j, i] = jaccard
    
    return dist


def _jaccard_cross_distance(
    matrix1: npt.NDArray,
    matrix2: npt.NDArray
) -> npt.NDArray:
    """
    Compute Jaccard cross-distance between two matrices.
    """
    n1, n2 = matrix1.shape[0], matrix2.shape[0]
    dist = np.zeros((n1, n2))
    
    binary1 = (matrix1 > 0).astype(int)
    binary2 = (matrix2 > 0).astype(int)
    
    for i in range(n1):
        for j in range(n2):
            intersection = np.sum(binary1[i] & binary2[j])
            union = np.sum(binary1[i] | binary2[j])
            
            if union > 0:
                dist[i, j] = 1 - intersection / union
            else:
                dist[i, j] = 0.0
    
    return dist
