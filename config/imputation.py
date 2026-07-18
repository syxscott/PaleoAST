# config/imputation.py
"""
Data Imputation Module for PaleoAST

Provides multiple strategies for handling missing values (NaN) in data matrices.

Imputation Methods:
    - Mean imputation: Replace NaN with column mean
    - Median imputation: Replace NaN with column median
    - KNN imputation: Use k-nearest neighbors for imputation
    - Row/Column removal: Remove rows or columns with NaN

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ImputationMethod(Enum):
    """Imputation method types."""

    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"
    REMOVE_ROWS = "remove_rows"
    REMOVE_COLUMNS = "remove_columns"


@dataclass
class ImputationResult:
    """Result of an imputation operation."""

    data: np.ndarray
    method: ImputationMethod
    nan_removed: int
    rows_removed: int
    columns_removed: int
    summary: str

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "nan_removed": self.nan_removed,
            "rows_removed": self.rows_removed,
            "columns_removed": self.columns_removed,
            "summary": self.summary,
        }


@dataclass
class MissingValueReport:
    """Report on missing values in a dataset."""

    total_nan: int
    nan_proportion: float
    rows_with_nan: int
    cols_with_nan: int
    nan_by_row: np.ndarray
    nan_by_col: np.ndarray
    column_names: list | None = None

    def to_dict(self) -> dict:
        return {
            "total_nan": self.total_nan,
            "nan_proportion": self.nan_proportion,
            "rows_with_nan": self.rows_with_nan,
            "cols_with_nan": self.cols_with_nan,
            "nan_by_row": self.nan_by_row.tolist() if self.nan_by_row is not None else None,
            "nan_by_col": self.nan_by_col.tolist() if self.nan_by_col is not None else None,
        }

    def summary(self) -> str:
        # Use the i18n system for translatable strings instead of
        # hardcoded Chinese — the previous version embedded Chinese
        # text that English users could not localise.
        from config.i18n import _

        lines = [
            _("Missing Value Report"),
            f"{'=' * 40}",
            _("Total NaN count: {0}").format(self.total_nan),
            _("NaN proportion: {0:.2f}%").format(self.nan_proportion * 100),
            _("Rows with NaN: {0}").format(self.rows_with_nan),
            _("Columns with NaN: {0}").format(self.cols_with_nan),
        ]
        return "\n".join(lines)


def analyze_missing_values(data: np.ndarray, column_names: list | None = None) -> MissingValueReport:
    """
    Analyze missing values in a data matrix.

    Parameters:
        data: Input data matrix
        column_names: Optional list of column names for reporting

    Returns:
        MissingValueReport with detailed missing value statistics
    """
    nan_mask = np.isnan(data)
    total_nan = int(np.sum(nan_mask))
    total_elements = data.size

    rows_with_nan = int(np.any(nan_mask, axis=1).sum())
    cols_with_nan = int(np.any(nan_mask, axis=0).sum())

    nan_by_row = np.sum(nan_mask, axis=1)
    nan_by_col = np.sum(nan_mask, axis=0)

    return MissingValueReport(
        total_nan=total_nan,
        nan_proportion=total_nan / total_elements if total_elements > 0 else 0,
        rows_with_nan=rows_with_nan,
        cols_with_nan=cols_with_nan,
        nan_by_row=nan_by_row,
        nan_by_col=nan_by_col,
        column_names=column_names,
    )


def impute_mean(data: np.ndarray) -> ImputationResult:
    """
    Impute missing values using column means.

    Replaces each NaN with the mean of its column (ignoring NaN).
    Columns that are entirely NaN are filled with 0 as a fallback.

    Parameters:
        data: Input data matrix (will not be modified)

    Returns:
        ImputationResult with imputed data
    """
    logger.info("Performing mean imputation")
    result_data = data.copy().astype(float)
    nan_mask = np.isnan(result_data)

    nan_count = int(np.sum(nan_mask))
    rows_removed = 0

    # Compute column means (ignoring NaN)
    col_means = np.nanmean(result_data, axis=0)

    # Find columns that are all NaN. np.nanmean returns NaN with a RuntimeWarning
    # for these, so substitute 0 explicitly to avoid leaving them as NaN.
    all_nan_cols = np.all(nan_mask, axis=0)
    if np.any(all_nan_cols):
        nan_cols = np.where(all_nan_cols)[0]
        logger.warning(f"impute_mean: {len(nan_cols)} all-NaN column(s) found; falling back to 0 for these columns")
        col_means = np.where(all_nan_cols, 0.0, col_means)

    # Impute
    for col_idx in range(result_data.shape[1]):
        col_nan_mask = nan_mask[:, col_idx]
        if np.any(col_nan_mask):
            result_data[col_nan_mask, col_idx] = col_means[col_idx]

    summary = f"均值填充: {nan_count} 个 NaN → 列均值"
    logger.info(summary)

    return ImputationResult(
        data=result_data,
        method=ImputationMethod.MEAN,
        nan_removed=nan_count,
        rows_removed=rows_removed,
        columns_removed=0,
        summary=summary,
    )


def impute_median(data: np.ndarray) -> ImputationResult:
    """
    Impute missing values using column medians.

    Replaces each NaN with the median of its column (ignoring NaN).
    Columns that are entirely NaN are filled with 0 as a fallback.

    Parameters:
        data: Input data matrix (will not be modified)

    Returns:
        ImputationResult with imputed data
    """
    logger.info("Performing median imputation")
    result_data = data.copy().astype(float)
    nan_mask = np.isnan(result_data)

    nan_count = int(np.sum(nan_mask))
    rows_removed = 0

    # Compute column medians (ignoring NaN)
    col_medians = np.nanmedian(result_data, axis=0)

    # Find columns that are all NaN. np.nanmedian returns NaN for these,
    # so substitute 0 to avoid leaving them as NaN.
    all_nan_cols = np.all(nan_mask, axis=0)
    if np.any(all_nan_cols):
        nan_cols = np.where(all_nan_cols)[0]
        logger.warning(f"impute_median: {len(nan_cols)} all-NaN column(s) found; falling back to 0 for these columns")
        col_medians = np.where(all_nan_cols, 0.0, col_medians)

    # Impute
    for col_idx in range(result_data.shape[1]):
        col_nan_mask = nan_mask[:, col_idx]
        if np.any(col_nan_mask):
            result_data[col_nan_mask, col_idx] = col_medians[col_idx]

    summary = f"中位数填充: {nan_count} 个 NaN → 列中位数"
    logger.info(summary)

    return ImputationResult(
        data=result_data,
        method=ImputationMethod.MEDIAN,
        nan_removed=nan_count,
        rows_removed=rows_removed,
        columns_removed=0,
        summary=summary,
    )


def impute_knn(data: np.ndarray, k: int = 5, distance_metric: str = "euclidean") -> ImputationResult:
    """
    Impute missing values using K-Nearest Neighbors.

    For each NaN value, finds the k nearest neighbors (based on non-NaN features)
    and uses their weighted average to impute the missing value.

    Parameters:
        data: Input data matrix
        k: Number of nearest neighbors
        distance_metric: Distance metric for neighbor finding

    Returns:
        ImputationResult with imputed data
    """
    logger.info(f"Performing KNN imputation with k={k}")
    result_data = data.copy().astype(float)
    nan_mask = np.isnan(result_data)

    nan_count = int(np.sum(nan_mask))

    _n_samples, n_features = result_data.shape

    # For each sample with missing values
    for sample_idx in np.where(np.any(nan_mask, axis=1))[0]:
        sample = result_data[sample_idx]
        missing_features = np.where(np.isnan(sample))[0]

        for feat_idx in missing_features:
            # Get features that are known for this sample (non-NaN)
            known_mask = ~np.isnan(sample)

            if np.sum(known_mask) == 0:
                # All features of this sample are NaN — fall back to the
                # global column mean of feat_idx (or 0 if the column is
                # also entirely NaN). The previous implementation simply
                # continued and left the cell as NaN, contradicting the
                # "→ 0" summary message.
                col_mean = np.nanmean(result_data[:, feat_idx])
                if not np.isfinite(col_mean):
                    col_mean = 0.0
                result_data[sample_idx, feat_idx] = col_mean
                continue

            # Find candidate neighbors: samples that have the target feature value
            candidate_mask = ~np.isnan(result_data[:, feat_idx])

            # Further filter: candidates must not have NaN in the same features as our sample
            for j in range(n_features):
                if known_mask[j]:  # If sample has a known value at j
                    # Candidate must also have a known (non-NaN) value at j
                    candidate_mask &= ~np.isnan(result_data[:, j])

            candidate_indices = np.where(candidate_mask)[0]
            if len(candidate_indices) == 0:
                # No usable neighbours: fall back to column mean of feat_idx.
                col_mean = np.nanmean(result_data[:, feat_idx])
                if not np.isfinite(col_mean):
                    col_mean = 0.0
                result_data[sample_idx, feat_idx] = col_mean
                continue

            # Calculate distances using known features
            sample_known = sample[known_mask]
            distances = np.zeros(len(candidate_indices))

            for i, cand_idx in enumerate(candidate_indices):
                cand_known = result_data[cand_idx, known_mask]
                distances[i] = np.sqrt(np.sum((sample_known - cand_known) ** 2))

            if len(distances) <= k:
                # Not enough neighbors, use weighted mean
                if np.sum(distances) > 0:
                    weights = 1.0 / (distances + 1e-10)
                    weights /= weights.sum()
                    result_data[sample_idx, feat_idx] = np.sum(weights * result_data[candidate_indices, feat_idx])
                else:
                    result_data[sample_idx, feat_idx] = np.mean(result_data[candidate_indices, feat_idx])
            else:
                # Use k nearest neighbors with distance weighting
                k_nearest_idx = np.argsort(distances)[:k]
                k_distances = distances[k_nearest_idx]
                k_values = result_data[candidate_indices[k_nearest_idx], feat_idx]

                # Distance-weighted average (1/d)
                weights = 1.0 / (k_distances + 1e-10)
                weights /= weights.sum()
                result_data[sample_idx, feat_idx] = np.sum(weights * k_values)

    summary = f"KNN填充 (k={k}): {nan_count} 个 NaN → 邻居加权均值"
    logger.info(summary)

    return ImputationResult(
        data=result_data,
        method=ImputationMethod.KNN,
        nan_removed=nan_count,
        rows_removed=0,
        columns_removed=0,
        summary=summary,
    )


def remove_rows_with_nan(data: np.ndarray) -> ImputationResult:
    """
    Remove rows containing any NaN values.

    Parameters:
        data: Input data matrix

    Returns:
        ImputationResult with rows removed
    """
    logger.info("Removing rows with NaN")
    nan_mask = np.isnan(data)
    rows_with_nan = np.any(nan_mask, axis=1)

    rows_removed = int(np.sum(rows_with_nan))
    nan_count = int(np.sum(nan_mask))
    result_data = data[~rows_with_nan]

    summary = f"删除行: 移除 {rows_removed} 行, {nan_count} 个 NaN"
    logger.info(summary)

    return ImputationResult(
        data=result_data,
        method=ImputationMethod.REMOVE_ROWS,
        nan_removed=nan_count,
        rows_removed=rows_removed,
        columns_removed=0,
        summary=summary,
    )


def remove_columns_with_nan(data: np.ndarray) -> ImputationResult:
    """
    Remove columns containing any NaN values.

    Parameters:
        data: Input data matrix

    Returns:
        ImputationResult with columns removed
    """
    logger.info("Removing columns with NaN")
    nan_mask = np.isnan(data)
    cols_with_nan = np.any(nan_mask, axis=0)

    cols_removed = int(np.sum(cols_with_nan))
    nan_count = int(np.sum(nan_mask))
    result_data = data[:, ~cols_with_nan]

    summary = f"删除列: 移除 {cols_removed} 列, {nan_count} 个 NaN"
    logger.info(summary)

    return ImputationResult(
        data=result_data,
        method=ImputationMethod.REMOVE_COLUMNS,
        nan_removed=nan_count,
        rows_removed=0,
        columns_removed=cols_removed,
        summary=summary,
    )


def impute(data: np.ndarray, method: ImputationMethod, **kwargs) -> ImputationResult:
    """
    Apply specified imputation method.

    Parameters:
        data: Input data matrix
        method: Imputation method to use
        **kwargs: Additional arguments for specific methods (e.g., k for KNN)

    Returns:
        ImputationResult with imputed data

    Raises:
        ValueError: If method is invalid or data is invalid
    """
    if data.size == 0:
        raise ValueError("Cannot impute empty data")

    nan_count = int(np.sum(np.isnan(data)))
    if nan_count == 0:
        logger.info("No missing values found, returning copy of data")
        return ImputationResult(
            data=data.copy(), method=method, nan_removed=0, rows_removed=0, columns_removed=0, summary="无缺失值"
        )

    if method == ImputationMethod.MEAN:
        return impute_mean(data)
    elif method == ImputationMethod.MEDIAN:
        return impute_median(data)
    elif method == ImputationMethod.KNN:
        k = kwargs.get("k", 5)
        return impute_knn(data, k=k)
    elif method == ImputationMethod.REMOVE_ROWS:
        return remove_rows_with_nan(data)
    elif method == ImputationMethod.REMOVE_COLUMNS:
        return remove_columns_with_nan(data)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
