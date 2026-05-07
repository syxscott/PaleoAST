# =============================================================================
# FILE: statistics/distance_metrics.py
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
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
import threading

from utils.exceptions import MatrixDimensionError
from utils.validators import validate_data_array


@dataclass
class DistanceMatrix:
    """
    Container for distance matrix results.
    
    Attributes:
        matrix: The computed distance matrix
        metric: Name of the distance metric used
        labels: Row/column labels
    """
    matrix: npt.NDArray
    metric: str
    labels: List[str]
    
    def __getitem__(self, key: tuple) -> float:
        """Access distance by (i, j) indices."""
        return self.matrix[key]
    
    def summary(self) -> str:
        """Generate summary text."""
        n = self.matrix.shape[0]
        return (
            f"Distance Matrix\n"
            f"{'=' * 40}\n"
            f"Metric: {self.metric}\n"
            f"Size: {n} × {n}\n"
            f"Min distance: {np.min(self.matrix[np.triu_indices(n, k=1)]):.4f}\n"
            f"Max distance: {np.max(self.matrix):.4f}\n"
            f"Mean distance: {np.mean(self.matrix[np.triu_indices(n, k=1)]):.4f}"
        )


def compute_distance_matrix(
    data: npt.NDArray,
    metric: str = 'euclidean',
    labels: Optional[List[str]] = None
) -> DistanceMatrix:
    """
    Compute pairwise distance matrix.
    
    Parameters:
        data: Input data matrix (n_samples, n_features)
        metric: Distance metric to use
        labels: Optional row/column labels
    
    Returns:
        DistanceMatrix: Computed distance matrix with metadata
    
    Supported Metrics:
        - 'euclidean': Euclidean (L2) distance
        - 'manhattan': Manhattan (L1) distance
        - 'bray_curtis': Bray-Curtis dissimilarity
        - 'jaccard': Jaccard dissimilarity
        - 'canberra': Canberra distance
        - 'chebychev': Chebychev (L∞) distance
    """
    # Validate data
    X = validate_data_array(data, allow_nan=False, name="distance_input")
    n = X.shape[0]
    
    # Default labels
    if labels is None:
        labels = [f"Sample_{i+1}" for i in range(n)]
    
    # Initialize distance matrix
    D = np.zeros((n, n))
    
    # Compute based on metric
    metric_lower = metric.lower()
    
    if metric_lower == 'euclidean':
        D = _euclidean_distance_matrix(X)
    elif metric_lower == 'manhattan':
        D = _manhattan_distance_matrix(X)
    elif metric_lower == 'bray_curtis':
        D = _bray_curtis_distance_matrix(X)
    elif metric_lower == 'jaccard':
        D = _jaccard_distance_matrix(X)
    elif metric_lower == 'canberra':
        D = _canberra_distance_matrix(X)
    elif metric_lower == 'chebychev':
        D = _chebychev_distance_matrix(X)
    else:
        raise ValueError(f"Unknown distance metric: '{metric}'")
    
    return DistanceMatrix(matrix=D, metric=metric, labels=labels)


def _euclidean_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Euclidean distance matrix.
    
    d_ij = ||x_i - x_j||_2 = sqrt(Σ_k (x_ik - x_jk)²)
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        diff = X - X[i]
        D[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return D


def _manhattan_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Manhattan distance matrix.
    
    d_ij = Σ_k |x_ik - x_jk|
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        D[i] = np.sum(np.abs(X - X[i]), axis=1)
    
    return D


def _bray_curtis_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Bray-Curtis dissimilarity matrix.
    
    d_BC(i,j) = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)
    
    Range: [0, 1]
    - 0: Identical compositions
    - 1: No overlap in taxa
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            numerator = np.sum(np.abs(X[i] - X[j]))
            denominator = np.sum(X[i] + X[j])
            
            if denominator > 0:
                bc = numerator / denominator
            else:
                bc = 0.0
            
            D[i, j] = bc
            D[j, i] = bc
    
    return D


def _jaccard_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Jaccard dissimilarity matrix.
    
    d_J(i,j) = 1 - |A ∩ B| / |A ∪ B|
    
    where A, B are sets of non-zero indices.
    For binary data: d_J = (FP + FN) / (TP + FP + FN)
    """
    n = X.shape[0]
    
    # Convert to binary presence/absence
    X_binary = (X > 0).astype(int)
    
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            intersection = np.sum(X_binary[i] & X_binary[j])
            union = np.sum(X_binary[i] | X_binary[j])
            
            if union > 0:
                jd = 1 - intersection / union
            else:
                jd = 0.0
            
            D[i, j] = jd
            D[j, i] = jd
    
    return D


def _canberra_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Canberra distance matrix.
    
    d_C(i,j) = Σ_k |x_ik - x_jk| / (|x_ik| + |x_jk|)
    
    Weighted version of Manhattan distance, sensitive to
    small differences near zero.
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        diff = np.abs(X - X[i])
        denom = np.abs(X) + np.abs(X[i])
        # Avoid division by zero
        denom = np.where(denom == 0, 1, denom)
        D[i] = np.sum(diff / denom, axis=1)
    
    return D


def _chebychev_distance_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    Compute Chebychev (L-infinity) distance matrix.
    
    d_C(i,j) = max_k |x_ik - x_jk|
    
    The maximum absolute difference along any dimension.
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        D[i] = np.max(np.abs(X - X[i]), axis=1)
    
    return D
