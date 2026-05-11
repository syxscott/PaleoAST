# =============================================================================
# FILE: utils/transformations.py
# =============================================================================
"""
Data Transformation & Preprocessing Pipeline for PaleoAST

Provides pure functions for common ecological and paleontological
data transformations, plus KNN-based missing value imputation.

All functions accept and return numpy arrays without side effects.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def log_transform(data: npt.NDArray, base: float = 10, offset: float = 1.0) -> npt.NDArray:
    """
    Logarithmic transformation: log_b(x + offset).

    The offset handles zero values (common in ecological abundance data).

    Parameters:
        data: Input array
        base: Logarithm base (10, 2, or np.e)
        offset: Constant added before log (default: 1.0 for log(x+1))

    Returns:
        Transformed array
    """
    result = data.astype(float).copy()
    valid = ~np.isnan(result)
    if base == np.e:
        result[valid] = np.log(result[valid] + offset)
    elif base == 2:
        result[valid] = np.log2(result[valid] + offset)
    else:
        result[valid] = np.log10(result[valid] + offset)
    return result


def sqrt_transform(data: npt.NDArray) -> npt.NDArray:
    """
    Square root transformation: sqrt(x).

    Commonly used for count data to stabilize variance.

    Parameters:
        data: Input array (must be non-negative)

    Returns:
        Transformed array
    """
    result = data.astype(float).copy()
    valid = ~np.isnan(result)
    result[valid] = np.sqrt(np.maximum(result[valid], 0))
    return result


def zscore_standardize(data: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """
    Z-score standardization: (x - mean) / std.

    Parameters:
        data: Input array
        axis: 0 = column-wise, 1 = row-wise

    Returns:
        Standardized array
    """
    result = np.atleast_2d(data.astype(float).copy())
    if axis == 0:
        for j in range(result.shape[1]):
            col = result[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                m = np.nanmean(col[valid])
                s = np.nanstd(col[valid], ddof=1)
                if s > 0:
                    result[valid, j] = (col[valid] - m) / s
    else:
        for i in range(result.shape[0]):
            row = result[i]
            valid = ~np.isnan(row)
            if valid.sum() > 1:
                m = np.nanmean(row[valid])
                s = np.nanstd(row[valid], ddof=1)
                if s > 0:
                    result[i, valid] = (row[valid] - m) / s
    return result


def percent_standardize(data: npt.NDArray, axis: int = 0) -> npt.NDArray:
    """
    Percentage (proportion) standardization.

    axis=0: each value as % of its column total
    axis=1: each value as % of its row total (common in ecology)

    Parameters:
        data: Input array
        axis: 0 = column-wise, 1 = row-wise

    Returns:
        Standardized array (values sum to 1.0 along axis)
    """
    result = data.astype(float).copy()
    valid = ~np.isnan(result)
    result[~valid] = 0

    totals = result.sum(axis=axis, keepdims=True)
    totals[totals == 0] = 1  # avoid division by zero

    result = result / totals
    result[~valid] = np.nan
    return result


def hellinger_transform(data: npt.NDArray) -> npt.NDArray:
    """
    Hellinger transformation: sqrt(x_i / sum(x)).

    Recommended for abundance data before PCA/PCoA.
    Reduces the influence of very abundant species.

    Parameters:
        data: Non-negative abundance matrix (n_samples x n_variables)

    Returns:
        Hellinger-transformed matrix
    """
    result = data.astype(float).copy()
    valid = ~np.isnan(result)
    result[~valid] = 0

    row_sums = result.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    result = np.sqrt(result / row_sums)
    result[~valid] = np.nan
    return result


def wisconsin_double_standardize(data: npt.NDArray) -> npt.NDArray:
    """
    Wisconsin double standardization.

    Step 1: Column maxima standardization (divide by column max)
    Step 2: Row totals standardization (divide by row total)

    Commonly used in community ecology.

    Parameters:
        data: Non-negative abundance matrix (n_samples x n_variables)

    Returns:
        Wisconsin-standardized matrix
    """
    result = data.astype(float).copy()
    valid = ~np.isnan(result)
    result[~valid] = 0

    # Step 1: divide each value by its column maximum
    col_max = result.max(axis=0)
    col_max[col_max == 0] = 1
    result = result / col_max

    # Step 2: divide each value by its row total
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    result = result / row_sums

    result[~valid] = np.nan
    return result


def boxcox_transform(data: npt.NDArray, column: int = 0, lambda_val: float | None = None) -> tuple[npt.NDArray, float]:
    """
    Box-Cox transformation for a single column.

    y = (x^λ - 1) / λ  if λ ≠ 0
    y = ln(x)           if λ = 0

    Parameters:
        data: 1D array or 2D array (uses column)
        column: Column index if 2D
        lambda_val: If None, finds optimal lambda

    Returns:
        (transformed_data, lambda_used)
    """
    from scipy import stats as sp_stats

    if data.ndim == 2:
        col = data[:, column].copy()
    else:
        col = data.copy()

    valid_mask = ~np.isnan(col) & (col > 0)
    valid_data = col[valid_mask]

    if len(valid_data) < 3:
        return col, 1.0

    if lambda_val is None:
        transformed, lambda_opt = sp_stats.boxcox(valid_data)
    else:
        if abs(lambda_val) < 1e-10:
            transformed = np.log(valid_data)
        else:
            transformed = (valid_data ** lambda_val - 1) / lambda_val
        lambda_opt = lambda_val

    result = col.copy()
    result[valid_mask] = transformed
    return result, lambda_opt


def impute_knn(data: npt.NDArray, k: int = 5) -> npt.NDArray:
    """
    KNN-based missing value imputation.

    For each sample with missing values, finds K nearest neighbors
    (using non-missing dimensions) and imputes with their mean.

    Parameters:
        data: Data matrix with NaN values
        k: Number of neighbors

    Returns:
        Imputed data matrix
    """
    result = data.astype(float).copy()
    n_samples, n_vars = result.shape

    for i in range(n_samples):
        missing_mask = np.isnan(result[i])
        if not missing_mask.any():
            continue

        observed_mask = ~missing_mask
        if not observed_mask.any():
            # All missing: impute with column means
            for j in np.where(missing_mask)[0]:
                col_mean = np.nanmean(result[:, j])
                result[i, j] = col_mean if not np.isnan(col_mean) else 0
            continue

        # Compute distances using observed dimensions only
        distances = np.zeros(n_samples)
        for other in range(n_samples):
            if other == i:
                distances[other] = np.inf
                continue
            both_observed = observed_mask & ~np.isnan(result[other])
            if both_observed.sum() == 0:
                distances[other] = np.inf
                continue
            diff = result[i, both_observed] - result[other, both_observed]
            distances[other] = np.sqrt(np.sum(diff ** 2))

        # Find K nearest neighbors
        neighbor_idx = np.argsort(distances)[:k]
        # Filter out infinite distances
        neighbor_idx = neighbor_idx[np.isfinite(distances[neighbor_idx])]

        if len(neighbor_idx) == 0:
            # Fallback to column means
            for j in np.where(missing_mask)[0]:
                col_mean = np.nanmean(result[:, j])
                result[i, j] = col_mean if not np.isnan(col_mean) else 0
            continue

        # Impute with neighbor means
        for j in np.where(missing_mask)[0]:
            neighbor_vals = result[neighbor_idx, j]
            neighbor_vals = neighbor_vals[~np.isnan(neighbor_vals)]
            if len(neighbor_vals) > 0:
                result[i, j] = np.mean(neighbor_vals)
            else:
                col_mean = np.nanmean(result[:, j])
                result[i, j] = col_mean if not np.isnan(col_mean) else 0

    return result


def impute_column_mean(data: npt.NDArray) -> npt.NDArray:
    """Simple column-mean imputation for missing values."""
    result = data.astype(float).copy()
    for j in range(result.shape[1]):
        col = result[:, j]
        missing = np.isnan(col)
        if missing.any():
            col_mean = np.nanmean(col)
            result[missing, j] = col_mean if not np.isnan(col_mean) else 0
    return result
