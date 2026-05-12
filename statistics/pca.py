# =============================================================================
# FILE: statistics/pca.py
# =============================================================================
"""
Principal Component Analysis (PCA) Module for PaleoAST

This module implements PCA for multivariate statistical analysis of
paleontological data, supporting both covariance-based and correlation-based PCA.

Mathematical Foundation:

PCA finds orthogonal linear combinations of variables that maximize variance.
Given data matrix X ∈ ℝ^(n×p):

1. Center the data: Z = X - μ (subtract column means)
2. Compute covariance matrix: S = (1/(n-1)) * Z^T * Z
3. Eigenvalue decomposition: S * V = V * Λ
   where Λ = diag(λ₁, λ₂, ..., λ_p) contains eigenvalues
         V contains eigenvectors as columns
4. Principal components: PC_k = X * v_k
   where v_k is the k-th eigenvector

Variance Explained:
    Individual: r_k² = λ_k / Σλ_i
    Cumulative: R_k² = Σᵢ₌₁ᵏ λ_i / Σλ_i

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import (
    ComputationError,
    MatrixDimensionError,
)
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class PCAResult:
    """
    Container for PCA analysis results.

    Attributes:
        scores: Principal component scores (n_samples × n_components)
        loadings: Variable loadings (n_components × n_variables)
        eigenvalues: Eigenvalues for each PC
        explained_variance: Variance explained by each PC
        cumulative_variance: Cumulative variance explained
        eigenvalues_raw: Raw eigenvalues before scaling
        mean_vector: Column means used for centering
        std_vector: Column standard deviations (for correlation PCA)
        n_components: Number of components extracted
        method: PCA method used ('covariance' or 'correlation')
        singular_values: Singular values from SVD

    Mathematical Relationships:
        - eigenvalues = singular_values² / (n - 1)
        - scores = X_centered @ loadings
        - loadings[:, k] = eigenvector_k * sqrt(eigenvalue_k)
    """

    scores: npt.NDArray
    loadings: npt.NDArray
    eigenvalues: npt.NDArray
    explained_variance: npt.NDArray
    cumulative_variance: npt.NDArray
    eigenvalues_raw: npt.NDArray
    mean_vector: npt.NDArray
    std_vector: npt.NDArray | None
    n_components: int
    method: str
    singular_values: npt.NDArray

    def __getitem__(self, key: str) -> Any:
        """Access result attributes by name."""
        return getattr(self, key)

    def get_scores(self, n_components: int | None = None) -> npt.NDArray:
        """
        Get scores for specified number of components.

        Parameters:
            n_components: Number of components to return. If None, returns all.
        """
        if n_components is None:
            return self.scores
        return self.scores[:, :n_components]

    def summary(self) -> str:
        """
        Generate summary text of PCA results.

        Returns:
            Formatted summary string
        """
        lines = [
            _("Principal Component Analysis Results"),
            "=" * 50,
            _("Method: {0}").format(self.method.upper()),
            _("Number of components: {0}").format(self.n_components),
            "",
            _("Component | Eigenvalue | Variance % | Cumulative %"),
            "-" * 50,
        ]

        for i in range(self.n_components):
            lines.append(
                f"PC{i + 1:7d} | {self.eigenvalues[i]:10.4f} | "
                f"{self.explained_variance[i]:9.2f} | "
                f"{self.cumulative_variance[i]:10.2f}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "scores": self.scores.tolist(),
            "loadings": self.loadings.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "explained_variance": self.explained_variance.tolist(),
            "cumulative_variance": self.cumulative_variance.tolist(),
            "n_components": self.n_components,
            "method": self.method,
        }


class PCAAnalyzer:
    """
    Principal Component Analysis engine for paleontological data.

    This class implements PCA with support for both variance-covariance
    and correlation-based approaches, along with comprehensive diagnostics.

    Features:
        - Covariance-based and correlation-based PCA
        - Scree plot data generation
        - Variable loadings analysis
        - Score plotting data
        - Confidence ellipses support

    Mathematical Reference:
        Given centered data Z ∈ ℝ^(n×p):

        Covariance PCA:
            S = (1/(n-1)) * Z^T * Z
            Eigenvectors V from S * V = V * Λ
            Scores: T = Z * V
            Loadings: P = V * sqrt(Λ)

        Correlation PCA (using standardized data):
            Z_stand = (Z) / σ (element-wise)
            Same process with Z_stand

    Example:
        >>> analyzer = PCAAnalyzer()
        >>> result = analyzer.analyze(data, n_components=3, method='correlation')
        >>> print(result.summary())
        >>> scores = result.get_scores(n_components=2)
    """

    def __init__(self) -> None:
        """Initialize the PCA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.PCAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: PCAResult | None = None
        self._logger.info("PCAAnalyzer initialized")

    def analyze(
        self,
        data: npt.NDArray,
        n_components: int | None = None,
        method: str = "covariance",
        impute_missing: bool = True,
    ) -> PCAResult:
        """
        Perform Principal Component Analysis.

        Parameters:
            data: Input data matrix of shape (n_samples, n_variables)
            n_components: Number of components to extract. If None, extracts
                        min(n_samples, n_variables) components.
            method: PCA method - 'covariance' or 'correlation'
            impute_missing: If True, imputes missing values with column means

        Returns:
            PCAResult: Complete PCA analysis results

        Raises:
            ComputationError: If matrix is singular or computation fails
            MatrixDimensionError: If input dimensions are invalid
        """
        with self._lock:
            # Validate and prepare data
            X = validate_data_array(data, allow_nan=True, name="PCA_input")

            n_samples, n_variables = X.shape
            self._logger.info(
                f"PCA analyze started: {n_samples} samples x {n_variables} variables, "
                f"n_components={n_components}, method={method}"
            )

            # Handle missing values
            if impute_missing and np.any(np.isnan(X)):
                col_means = np.nanmean(X, axis=0)
                nan_mask = np.isnan(X)
                for j in range(n_variables):
                    X[nan_mask[:, j], j] = col_means[j]

            # Determine number of components
            max_components = min(n_samples - 1, n_variables)
            if n_components is None:
                n_components = max_components
            else:
                n_components = min(n_components, max_components)

            if n_components < 1:
                raise MatrixDimensionError(
                    "Cannot perform PCA: insufficient dimensions",
                    details={"n_samples": n_samples, "n_variables": n_variables},
                )

            # Perform PCA based on method
            if method == "covariance":
                result = self._pca_covariance(X, n_components)
            elif method == "correlation":
                result = self._pca_correlation(X, n_components)
            else:
                raise ValueError(f"Unknown PCA method: '{method}'. Use 'covariance' or 'correlation'.")

            self._last_result = result
            self._logger.info(
                f"PCA completed: top eigenvalues={result.eigenvalues[:3].tolist()}, "
                f"cumulative variance={result.cumulative_variance[-1]:.2f}%"
            )
            return result

    def _pca_covariance(self, X: npt.NDArray, n_components: int) -> PCAResult:
        """
        Perform covariance-based PCA.

        Mathematical Steps:
            1. Center: Z = X - μ (subtract column means)
            2. Covariance: S = (1/(n-1)) * Z^T * Z
            3. Eigendecomposition: S = V * Λ * V^T
            4. Scores: T = Z * V
            5. Loadings: P = V * sqrt(Λ)
        """
        n_samples, n_variables = X.shape

        # Step 1: Center the data
        mean_vector = np.mean(X, axis=0)
        Z = X - mean_vector
        logger.debug(f"Covariance PCA: data centered, mean range=[{mean_vector.min():.4f}, {mean_vector.max():.4f}]")

        # Step 2: Compute covariance matrix using SVD for numerical stability
        # For centered data: S = (1/(n-1)) * Z^T * Z
        # SVD of Z: Z = U * Σ * V^T
        # Then: S = V * (Σ²/(n-1)) * V^T
        try:
            U, singular_values, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise ComputationError("SVD computation failed during PCA", original_exception=e)

        # Eigenvalues from singular values
        # λ_i = σ_i² / (n-1)
        eigenvalues_raw = (singular_values**2) / (n_samples - 1)

        # Step 3: Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues_raw = eigenvalues_raw[sorted_indices]
        singular_values = singular_values[sorted_indices]
        Vt = Vt[sorted_indices]

        # Select top n_components
        eigenvalues = eigenvalues_raw[:n_components]
        singular_values = singular_values[:n_components]
        V = Vt[:n_components].T

        # Step 4: Compute scores (project data onto eigenvectors)
        # T = Z * V (n × k)
        scores = Z @ V

        # Step 5: Compute loadings
        # P = V * sqrt(Λ) (p × k)
        loadings = V * np.sqrt(eigenvalues)

        # Compute explained variance
        total_variance = np.sum(eigenvalues_raw)
        if total_variance > 0:
            explained_variance = (eigenvalues / total_variance) * 100
        else:
            explained_variance = np.zeros_like(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance)

        return PCAResult(
            scores=scores,
            loadings=loadings,
            eigenvalues=eigenvalues,
            explained_variance=explained_variance,
            cumulative_variance=cumulative_variance,
            eigenvalues_raw=eigenvalues_raw,
            mean_vector=mean_vector,
            std_vector=None,
            n_components=n_components,
            method="covariance",
            singular_values=singular_values,
        )

    def _pca_correlation(self, X: npt.NDArray, n_components: int) -> PCAResult:
        """
        Perform correlation-based PCA (using standardized data).

        Mathematical Steps:
            1. Standardize: Z = (X - μ) / σ (z-scores)
            2. Correlation matrix: R = (1/(n-1)) * Z^T * Z
            3. Eigendecomposition: R = V * Λ * V^T
            4. Scores: T = Z * V
            5. Loadings: P = V * sqrt(Λ)

        Note: For standardized data, correlation matrix equals
              covariance matrix, so we can use the same algorithm.
        """
        n_samples, n_variables = X.shape

        # Step 1: Compute mean and std for standardization
        mean_vector = np.mean(X, axis=0)
        std_vector = np.std(X, axis=0, ddof=1)

        # Handle zero standard deviation
        std_vector = np.where(std_vector == 0, 1.0, std_vector)

        # Standardize: Z = (X - μ) / σ
        Z = (X - mean_vector) / std_vector
        logger.debug(f"Correlation PCA: data standardized, std range=[{std_vector.min():.4f}, {std_vector.max():.4f}]")

        # Step 2: SVD of standardized data
        try:
            U, singular_values, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise ComputationError("SVD computation failed during correlation PCA", original_exception=e)

        # Eigenvalues from singular values
        eigenvalues_raw = (singular_values**2) / (n_samples - 1)

        # Step 3: Sort by eigenvalues
        sorted_indices = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues_raw = eigenvalues_raw[sorted_indices]
        singular_values = singular_values[sorted_indices]
        Vt = Vt[sorted_indices]

        # Select top n_components
        eigenvalues = eigenvalues_raw[:n_components]
        singular_values = singular_values[:n_components]
        V = Vt[:n_components].T

        # Step 4: Compute scores
        scores = Z @ V

        # Step 5: Compute loadings
        loadings = V * np.sqrt(eigenvalues)

        # Compute explained variance
        total_variance = np.sum(eigenvalues_raw)
        if total_variance > 0:
            explained_variance = (eigenvalues / total_variance) * 100
        else:
            explained_variance = np.zeros_like(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance)

        return PCAResult(
            scores=scores,
            loadings=loadings,
            eigenvalues=eigenvalues,
            explained_variance=explained_variance,
            cumulative_variance=cumulative_variance,
            eigenvalues_raw=eigenvalues_raw,
            mean_vector=mean_vector,
            std_vector=std_vector,
            n_components=n_components,
            method="correlation",
            singular_values=singular_values,
        )

    def get_scree_data(self, result: PCAResult | None = None, n_points: int = 20) -> dict[str, Any]:
        """
        Generate data for scree plot.

        Parameters:
            result: PCA result to use. If None, uses last result.
            n_points: Number of eigenvalues to include

        Returns:
            Dictionary with eigenvalues and explained variance for plotting
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No PCA result available. Run analyze() first.")

        n_eigenvalues = min(n_points, len(result.eigenvalues_raw))

        return {
            "components": np.arange(1, n_eigenvalues + 1),
            "eigenvalues": result.eigenvalues_raw[:n_eigenvalues],
            "explained_variance": (result.eigenvalues_raw[:n_eigenvalues] / np.sum(result.eigenvalues_raw) * 100),
            "cumulative_variance": np.cumsum(
                result.eigenvalues_raw[:n_eigenvalues] / np.sum(result.eigenvalues_raw) * 100
            ),
        }

    def get_loadings_data(self, result: PCAResult | None = None) -> dict[str, Any]:
        """
        Get loadings data for biplot visualization.

        Returns:
            Dictionary with loadings vectors
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No PCA result available. Run analyze() first.")

        return {
            "loadings": result.loadings,
            "pc1_loadings": result.loadings[:, 0],
            "pc2_loadings": result.loadings[:, 1] if result.n_components >= 2 else None,
        }

    @property
    def last_result(self) -> PCAResult | None:
        """Get the last computed PCA result."""
        with self._lock:
            return self._last_result
