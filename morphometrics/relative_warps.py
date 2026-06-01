# =============================================================================
# FILE: morphometrics/relative_warps.py
# =============================================================================
"""
Relative Warps Analysis Module for PaleoAST

Relative Warps Analysis performs PCA on Procrustes-aligned coordinates
to extract the major axes of shape variation.

Mathematical Foundation:

Given N Procrustes-aligned configurations Y₁, Y₂, ..., Yₙ ∈ ℝ^k:
    k = n_landmarks × n_dimensions (flattened)

1. Flatten each configuration to a vector
   y_i = vec(Y_i) ∈ ℝ^k

2. Compute covariance matrix
   S = (1/(n-1)) * Σ(y_i - ȳ)(y_i - ȳ)'

3. PCA on covariance matrix
   S * v_j = λ_j * v_j

4. Relative Warps (RW_j):
   RW_j = y * v_j

Relative warps are analogous to principal components in traditional PCA
but applied to shape space.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


@dataclass
class RelativeWarpsResult:
    """
    Container for Relative Warps analysis results.
    """

    relative_warps: npt.NDArray  # RW scores (n_specimens, n_components)
    eigenvalues: npt.NDArray
    explained_variance: npt.NDArray
    cumulative_variance: npt.NDArray
    eigenvectors: npt.NDArray  # Shape change vectors
    mean_shape: npt.NDArray  # Mean Procrustes configuration
    n_components: int
    n_landmarks: int
    n_dims: int

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Relative Warps Analysis Results"),
            "=" * 50,
            _("Number of specimens: {0}").format(self.relative_warps.shape[0]),
            _("Number of landmarks: {0}").format(self.n_landmarks),
            _("Dimensionality: {0}D").format(self.n_dims),
            _("Number of components: {0}").format(self.n_components),
            "",
            _("RW  | Eigenvalue | Variance % | Cumulative"),
            "-" * 50,
        ]

        for i in range(min(10, self.n_components)):
            lines.append(
                f"RW{i + 1:2d} | {self.eigenvalues[i]:10.4f} | "
                f"{self.explained_variance[i]:9.2f} | "
                f"{self.cumulative_variance[i]:10.2f}"
            )

        return "\n".join(lines)


class RelativeWarpsAnalyzer:
    """
    Relative Warps Analysis engine.

    Performs PCA on Procrustes-aligned landmark configurations
    to extract major patterns of shape variation.
    """

    def __init__(self) -> None:
        """Initialize the Relative Warps analyzer."""
        self._logger = logging.getLogger(f"{__name__}.RelativeWarpsAnalyzer")
        self._logger.info("RelativeWarpsAnalyzer initialized")
        self._lock = threading.RLock()
        self._last_result: RelativeWarpsResult | None = None

    def analyze(self, aligned_configurations: npt.NDArray, n_components: int | None = None) -> RelativeWarpsResult:
        """
        Perform Relative Warps Analysis.

        Parameters:
            aligned_configurations: 3D array from GPA (n_specimens, n_landmarks, n_dims)
            n_components: Number of relative warps to extract

        Returns:
            RelativeWarpsResult: Relative Warps analysis results
        """
        with self._lock:
            # Validate input
            if aligned_configurations.ndim != 3:
                raise ComputationError("Aligned configurations must be 3D (n_specimens, n_landmarks, n_dims)")

            n_specimens, n_landmarks, n_dims = aligned_configurations.shape
            self._logger.info(
                f"Relative Warps analysis started: n_specimens={n_specimens}, "
                f"n_landmarks={n_landmarks}, n_dimensions={n_dims}"
            )

            # Determine number of components
            max_components = min(n_specimens - 1, n_landmarks * n_dims)
            if n_components is None:
                n_components = max_components
            else:
                n_components = min(n_components, max_components)

            # Flatten configurations
            # Shape: (n_specimens, n_landmarks * n_dims)
            flattened = aligned_configurations.reshape(n_specimens, n_landmarks * n_dims)

            # Compute mean shape
            mean_shape = np.mean(aligned_configurations, axis=0)

            # Center the data
            flattened_centered = flattened - mean_shape.flatten()

            # PCA via SVD (more numerically stable than eigendecomposition)
            try:
                U, singular_values, Vt = np.linalg.svd(flattened_centered, full_matrices=False)
            except np.linalg.LinAlgError as e:
                raise ComputationError("SVD failed during Relative Warps analysis", original_exception=e)

            # Eigenvalues from singular values
            eigenvalues = (singular_values**2) / (n_specimens - 1)

            # Select top components
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = Vt[:n_components].T

            # Compute relative warps (projections)
            relative_warps = flattened_centered @ eigenvectors

            # Compute explained variance
            total_variance = np.sum(eigenvalues)
            explained_variance = (eigenvalues / total_variance) * 100
            cumulative_variance = np.cumsum(explained_variance)
            self._logger.info(
                f"Relative Warps analysis completed: {n_components} components, "
                f"PC1 variance={explained_variance[0]:.2f}%, "
                f"cumulative={cumulative_variance[-1]:.2f}%"
            )

            result = RelativeWarpsResult(
                relative_warps=relative_warps,
                eigenvalues=eigenvalues,
                explained_variance=explained_variance,
                cumulative_variance=cumulative_variance,
                eigenvectors=eigenvectors,
                mean_shape=mean_shape,
                n_components=n_components,
                n_landmarks=n_landmarks,
                n_dims=n_dims,
            )

            self._last_result = result
            return result

    def get_shape_at_warp(
        self, result: RelativeWarpsResult | None = None, warp_number: int = 0, warp_score: float = 3.0
    ) -> npt.NDArray:
        """
        Reconstruct shape at a specific position along a relative warp.

        Parameters:
            result: Relative warps result. If None, uses last result.
            warp_number: Which relative warp (0-indexed)
            warp_score: Score along the warp (standard deviations)

        Returns:
            npt.NDArray: Reconstructed configuration
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No Relative Warps result available")

        if warp_number >= result.n_components:
            raise ComputationError(f"Warp number {warp_number} exceeds available components")

        # Start from mean shape
        shape = result.mean_shape.flatten()

        # Add contribution from specified warp
        eigenvector = result.eigenvectors[:, warp_number]
        std_dev = np.sqrt(result.eigenvalues[warp_number])

        shape = shape + eigenvector * warp_score * std_dev

        # Reshape to configuration
        return shape.reshape(result.n_landmarks, result.n_dims)

    @property
    def last_result(self) -> RelativeWarpsResult | None:
        """Get the last Relative Warps result."""
        with self._lock:
            return self._last_result
