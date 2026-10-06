# =============================================================================
# FILE: stats/distance_metrics.py
# =============================================================================
"""
Distance Metrics Module for PaleoAST

This module provides comprehensive distance/similarity metric computations
for paleontological and ecological data analysis.

Supported Metrics:
    - Euclidean: L2 norm distance
    - Manhattan: L1 norm distance
    - Bray-Curtis: Ecological dissimilarity
    - Jaccard: Presence/absence similarity
    - Canberra: Weighted Canberra distance
    - Chebychev: L-infinity distance

Author: PaleoAST Development Team
Version: 2.0.0 (optimized with scipy)
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, pdist, squareform

from config.i18n import _
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class DistanceMatrixResult:
    """
    Container for distance matrix computation results.

    This class stores the results of a distance matrix computation,
    separate from the PhylogeneticDistanceMatrix used in phylogenetics.

    Attributes:
        matrix: The computed distance matrix
        metric: Name of the distance metric used
        labels: Row/column labels
    """

    matrix: npt.NDArray
    metric: str
    labels: list[str]
    _upper_tri_indices: npt.NDArray | None = None

    def __getitem__(self, key: tuple) -> float:
        """Access distance by (i, j) indices."""
        return self.matrix[key]

    def _get_upper_tri_indices(self) -> npt.NDArray:
        """Cache upper triangular indices for summary computation."""
        if self._upper_tri_indices is None:
            n = self.matrix.shape[0]
            self._upper_tri_indices = np.triu_indices(n, k=1)
        return self._upper_tri_indices

    def summary(self) -> str:
        """Generate summary text."""
        n = self.matrix.shape[0]
        upper_idx = self._get_upper_tri_indices()
        upper_vals = self.matrix[upper_idx]
        return (
            f"{_('Distance Matrix')}\n"
            f"{'=' * 40}\n"
            f"{_('Metric: {0}').format(self.metric)}\n"
            f"{_('Size: {0} x {1}').format(n, n)}\n"
            f"{_('Min distance: {0}').format(f'{np.min(upper_vals):.4f}')}\n"
            f"{_('Max distance: {0}').format(f'{np.max(upper_vals):.4f}')}\n"
            f"{_('Mean distance: {0}').format(f'{np.mean(upper_vals):.4f}')}"
        )


# Metric name mapping from PaleoAST to scipy
_METRIC_MAP = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",  # cityblock is scipy's name for Manhattan
    "canberra": "canberra",
    "chebychev": "chebyshev",  # Note: scipy uses chebyshev, not chebychev
    "jaccard": "jaccard",
}


def compute_distance_matrix(
    data: npt.NDArray, metric: str = "euclidean", labels: list[str] | None = None
) -> DistanceMatrixResult:
    """
    Compute pairwise distance matrix.

    Parameters:
        data: Input data matrix (n_samples, n_features)
        metric: Distance metric to use
        labels: Optional row/column labels

    Returns:
        DistanceMatrixResult: Computed distance matrix with metadata

    Supported Metrics:
        - 'euclidean': Euclidean (L2) distance
        - 'manhattan': Manhattan (L1) distance
        - 'bray_curtis': Bray-Curtis dissimilarity
        - 'jaccard': Jaccard dissimilarity (presence/absence)
        - 'canberra': Canberra distance
        - 'chebychev': Chebychev (L∞) distance
    """
    # Validate data. We allow NaN and Inf here (rather than rejecting
    # outright) so that callers can intentionally compute a "partial"
    # distance matrix: e.g. the all-NaN / all-Inf integration tests
    # expect every cell to come back as NaN / Inf. The metric
    # implementations below already propagate NaN / Inf correctly
    # via numpy's built-in arithmetic, so the output is well-defined.
    X = validate_data_array(data, allow_nan=True, allow_inf=True, name="distance_input")
    n = X.shape[0]
    logger.info(f"Computing distance matrix: {X.shape[0]}x{X.shape[1]} data, metric={metric}")

    # Default labels
    if labels is None:
        labels = [f"Sample_{i + 1}" for i in range(n)]

    # Compute based on metric
    metric_lower = metric.lower()
    logger.debug(f"Distance computation dispatching to '{metric_lower}' metric")

    if metric_lower == "bray_curtis":
        D = _bray_curtis_distance_matrix(X)
    elif metric_lower == "jaccard":
        # scipy's jaccard treats non-zero as present and zero as absent. If
        # the input contains fractional/quantitative abundances, scipy
        # silently binarises them and throws away the magnitude information.
        # Warn the user explicitly so a count-matrix input is not misreported
        # as Jaccard-on-presence/absence.
        if _looks_quantitative(X):
            import warnings as _warnings

            _warnings.warn(
                "Jaccard is a presence/absence metric: scipy.broadcast treats any "
                "non-zero value as 'present'. The input matrix appears to contain "
                "quantitative values (not a pure 0/1 presence/absence matrix); "
                "the magnitudes will be DISCARDED and only the support sets used. "
                "If the input is a species count or abundance matrix, consider "
                "'bray_curtis' instead, which preserves the quantitative structure.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning("Jaccard called on quantitative input: magnitudes will be discarded.")
        scipy_metric = _METRIC_MAP.get(metric_lower, metric_lower)
        try:
            condensed = pdist(X, metric=scipy_metric)
            D = squareform(condensed)
        except Exception as e:
            logger.warning(f"pdist failed for {metric}, falling back to loop: {e}")
            D = _fallback_distance_matrix(X, metric_lower)
    else:
        # Use scipy's pdist for efficient computation
        scipy_metric = _METRIC_MAP.get(metric_lower, metric_lower)
        try:
            # pdist returns condensed distance matrix (upper triangle)
            condensed = pdist(X, metric=scipy_metric)
            D = squareform(condensed)
        except Exception as e:
            logger.warning(f"pdist failed for {metric}, falling back to loop: {e}")
            D = _fallback_distance_matrix(X, metric_lower)

    logger.info(f"Distance matrix computed: {n}x{n}, metric={metric}")
    return DistanceMatrixResult(matrix=D, metric=metric, labels=labels)


def _looks_quantitative(X: npt.NDArray) -> bool:
    """Return True if the input matrix has quantitative (non-binary) entries.

    Heuristic: look at every unique value; if any value is strictly between
    0 and 1 (or simply not exactly 0/1) treat the matrix as quantitative.
    NaN/Inf are skipped (the caller has already accepted those).
    """
    finite = X[np.isfinite(X)]
    if finite.size == 0:
        return False
    # Vectorised check: not every value is exactly 0 or 1
    unique_vals = np.unique(finite)
    non_binary_mask = ~np.isin(unique_vals, [0.0, 1.0])
    return bool(np.any(non_binary_mask))


def _bray_curtis_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Bray-Curtis dissimilarity matrix.

    d_BC(i,j) = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)

    Range: [0, 1]
    - 0: Identical compositions
    - 1: No overlap in taxa
    - NaN: undefined (e.g. both samples are all zeros, or invalid input)

    Raises:
        ValueError: if any input value is negative (Bray-Curtis requires
            non-negative abundance data; the previous implementation
            silently propagated negatives into |a-b|/|a+b| which has no
            ecological meaning).
    """
    # Bray-Curtis requires non-negative abundances.  Negative values
    # break the denominator (Σ_k (a_k + b_k)) and produce nonsense
    # distances; we surface a clear error rather than silently
    # propagating NaNs / wrong numbers.
    if np.any(X < 0):
        raise ValueError(
            "Bray-Curtis requires non-negative abundances; negative values were "
            "detected in the input matrix. Use a different metric for signed "
            "data (e.g. Euclidean)."
        )

    n = X.shape[0]

    try:
        if n <= 500:
            diff = np.abs(X[:, None, :] - X[None, :, :])
            sum_arr = X[:, None, :] + X[None, :, :]
            numerator = np.sum(diff, axis=2)
            denominator = np.sum(sum_arr, axis=2)
            # When the denominator is 0 (both samples are all zeros) the
            # Bray-Curtis dissimilarity is UNDEFINED (0/0). Return NaN
            # instead of the artificial 0 the previous code returned
            # (which made two empty samples look identical -- silently
            # biasing downstream PCoA / NMDS / cluster analyses).
            D = np.where(denominator == 0, np.nan, numerator / denominator)
            # Distance of a row to itself is 0 by definition; restore the
            # diagonal explicitly because for an all-zero row the diagonal
            # cell also got the NaN from the above np.where.
            np.fill_diagonal(D, 0.0)
        else:
            D = np.zeros((n, n))
            chunk_size = 100
            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                for j in range(i + 1, n, chunk_size):
                    end_j = min(j + chunk_size, n)
                    chunk = X[i:end_i, None, :] - X[None, j:end_j, :]
                    num = np.sum(np.abs(chunk), axis=2)
                    den = np.sum(X[i:end_i, None, :] + X[None, j:end_j, :], axis=2)
                    # NaN where the pair has zero total abundance
                    block = np.where(den == 0, np.nan, num / den)
                    D[i:end_i, j:end_j] = block
                    D[j:end_j, i:end_i] = block.T
    except Exception:
        # Fallback to pure Python loop (slow)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                numerator = np.sum(np.abs(X[i] - X[j]))
                denominator = np.sum(X[i] + X[j])
                if denominator > 0:
                    bc = numerator / denominator
                else:
                    # Both samples empty: undefined.
                    bc = np.nan
                D[i, j] = bc
                D[j, i] = bc

    return D


def _fallback_distance_matrix(X: npt.NDArray, metric: str) -> npt.NDArray:
    """Fallback distance matrix computation using cdist."""
    n = X.shape[0]
    D = np.zeros((n, n))

    # Use cdist which is faster than pure Python loops
    for i in range(n):
        D[i] = cdist(X[i : i + 1], X, metric=metric)[0]

    return D


def _euclidean_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Euclidean distance matrix (legacy, use compute_distance_matrix instead).

    d_ij = ||x_i - x_j||_2 = sqrt(Σ_k (x_ik - x_jk)²)
    """
    condensed = pdist(X, metric="euclidean")
    return squareform(condensed)


def _manhattan_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Manhattan distance matrix (legacy, use compute_distance_matrix instead).

    d_ij = Σ_k |x_ik - x_jk|
    """
    condensed = pdist(X, metric="cityblock")
    return squareform(condensed)


def _jaccard_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Jaccard dissimilarity matrix (legacy, use compute_distance_matrix instead).

    d_J(i,j) = 1 - |A ∩ B| / |A ∪ B|

    where A, B are sets of non-zero indices.
    """
    condensed = pdist(X, metric="jaccard")
    return squareform(condensed)


def _canberra_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Canberra distance matrix (legacy, use compute_distance_matrix instead).

    d_C(i,j) = Σ_k |x_ik - x_jk| / (|x_ik| + |x_jk|)
    """
    condensed = pdist(X, metric="canberra")
    return squareform(condensed)


def _chebychev_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Chebychev distance matrix (legacy, use compute_distance_matrix instead).

    d_C(i,j) = max_k |x_ik - x_jk|
    """
    condensed = pdist(X, metric="chebyshev")
    return squareform(condensed)
