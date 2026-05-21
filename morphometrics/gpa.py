# =============================================================================
# FILE: morphometrics/gpa.py
# =============================================================================
"""
Generalized Procrustes Analysis (GPA) Module for PaleoAST

GPA is the fundamental preprocessing step in geometric morphometrics that
removes non-shape variation from landmark configurations.

Mathematical Foundation:

Given N landmark configurations X₁, X₂, ..., Xₙ ∈ ℝ^(k×m):
    k = number of landmarks
    m = number of dimensions (2 for 2D, 3 for 3D)

The GPA algorithm minimizes Procrustes distance by iteratively:

1. Translation: Remove centroid from each configuration
   X_centered = X - centroid(X)

2. Scaling: Normalize to unit centroid size
   X_scaled = X_centered / size(X_centered)

   where size = sqrt(trace(X_centered' * X_centered))

3. Rotation: Optimal rotation via SVD
   X_rotated = R * X_scaled

   where R maximizes trace(Y' * R * X_scaled)
   and R = U * V' from SVD of Y' * X_scaled = U * Σ * V'

Procrustes Distance:
    d_P(X, Y) = sqrt(||X - Y||² / size(X))

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.constants import GPA_CONVERGENCE_TOLERANCE, GPA_MAX_ITERATIONS
from config.i18n import _
from utils.exceptions import ComputationError, MorphometricsError

logger = logging.getLogger(__name__)


@dataclass
class GPAResult:
    """
    Container for GPA analysis results.
    """

    aligned_configurations: npt.NDArray  # (n_specimens, k, m)
    consensus: npt.NDArray  # (k, m) - mean configuration
    procrustes_distances: npt.NDArray  # Distance to consensus
    rotations: list[npt.NDArray]  # Rotation matrices
    scales: npt.NDArray  # Scale factors
    centroids: npt.NDArray  # Original centroids
    n_iterations: int
    converged: bool
    final_sse: float  # Sum of squared errors

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Generalized Procrustes Analysis Results')}\n"
            f"{'=' * 50}\n"
            f"{_('Number of specimens: {0}').format(self.aligned_configurations.shape[0])}\n"
            f"{_('Number of landmarks: {0}').format(self.aligned_configurations.shape[1])}\n"
            f"{_('Dimensions: {0}').format(self.aligned_configurations.shape[2])}\n"
            f"{_('Iterations: {0}').format(self.n_iterations)}\n"
            f"{_('Converged: {0}').format(self.converged)}\n"
            f"{_('Final SSE: {0}').format(f'{self.final_sse:.6f}')}"
        )


class GPAAnalyzer:
    """
    Generalized Procrustes Analysis engine for landmark data.

    GPA removes non-shape variation (translation, scaling, rotation)
    from landmark configurations to obtain shape coordinates.
    """

    def __init__(self) -> None:
        """Initialize the GPA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.GPAAnalyzer")
        self._logger.info("GPAAnalyzer initialized")
        self._lock = threading.RLock()
        self._last_result: GPAResult | None = None
        self._tolerance = GPA_CONVERGENCE_TOLERANCE
        self._max_iterations = GPA_MAX_ITERATIONS

    def analyze(
        self, configurations: npt.NDArray, n_iterations: int | None = None, tolerance: float | None = None
    ) -> GPAResult:
        """
        Perform Generalized Procrustes Analysis.

        Parameters:
            configurations: 3D array of shape (n_specimens, n_landmarks, n_dims)
                          or 2D array (n_specimens, n_landmarks*n_dims) for flat format
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for Procrustes distance

        Returns:
            GPAResult: GPA analysis results

        Raises:
            MorphometricsError: If configurations are invalid
        """
        with self._lock:
            # Validate and prepare configurations
            X = self._prepare_configurations(configurations)

            n_specimens, n_landmarks, n_dims = X.shape
            self._logger.info(
                f"GPA alignment started: n_specimens={n_specimens}, n_landmarks={n_landmarks}, n_dimensions={n_dims}"
            )

            if n_iterations is None:
                n_iterations = self._max_iterations
            if tolerance is None:
                tolerance = self._tolerance

            # Initialize aligned configurations
            aligned = X.copy()
            rotations = []
            scales = np.ones(n_specimens)
            centroids = np.zeros((n_specimens, n_dims))

            # Compute initial centroids and sizes
            for i in range(n_specimens):
                centroids[i] = np.mean(aligned[i], axis=0)

            # Iterative Procrustes superimposition
            prev_sse = float("inf")

            for iteration in range(n_iterations):
                # Step 1: Translate to common origin
                for i in range(n_specimens):
                    aligned[i] = aligned[i] - centroids[i]

                # Step 2: Scale to unit size
                sizes = self._compute_sizes(aligned)
                for i in range(n_specimens):
                    aligned[i] = aligned[i] / sizes[i]
                    scales[i] *= sizes[i]

                # Step 3: Compute consensus (mean)
                consensus = np.mean(aligned, axis=0)

                # Step 4: Rotate each specimen to consensus
                new_aligned = np.zeros_like(aligned)
                iter_rotations = []

                for i in range(n_specimens):
                    rotation = self._find_rotation(consensus, aligned[i])
                    iter_rotations.append(rotation)
                    new_aligned[i] = aligned[i] @ rotation.T

                aligned = new_aligned
                rotations = iter_rotations

                # Compute sum of squared Procrustes distances
                sse = 0.0
                for i in range(n_specimens):
                    diff = aligned[i] - consensus
                    sse += np.sum(diff**2)

                self._logger.debug(f"GPA iteration {iteration + 1}: SSE={sse:.6f}, delta={abs(prev_sse - sse):.6e}")

                # Check convergence
                if abs(prev_sse - sse) < tolerance:
                    self._logger.info(f"GPA converged after {iteration + 1} iterations with final SSE={sse:.6f}")
                    result = GPAResult(
                        aligned_configurations=aligned,
                        consensus=consensus,
                        procrustes_distances=self._compute_distances_to_consensus(aligned, consensus),
                        rotations=rotations,
                        scales=scales,
                        centroids=centroids,
                        n_iterations=iteration + 1,
                        converged=True,
                        final_sse=sse,
                    )
                    self._last_result = result
                    return result

                prev_sse = sse

            # Did not converge within max iterations
            self._logger.warning(f"GPA did not converge after {n_iterations} iterations, final SSE={prev_sse:.6f}")
            result = GPAResult(
                aligned_configurations=aligned,
                consensus=consensus,
                procrustes_distances=self._compute_distances_to_consensus(aligned, consensus),
                rotations=rotations,
                scales=scales,
                centroids=centroids,
                n_iterations=n_iterations,
                converged=False,
                final_sse=prev_sse,
            )

            self._last_result = result
            return result

    def _prepare_configurations(self, configurations: npt.NDArray) -> npt.NDArray:
        """
        Prepare configurations array to standard 3D format.

        Parameters:
            configurations: Input array (n, k, m) or (n, k*m)

        Returns:
            npt.NDArray: Standardized 3D array (n, k, m)
        """
        if configurations.ndim == 2:
            # Flat format: each row is flattened configuration
            n_specimens = configurations.shape[0]
            flat_dim = configurations.shape[1]

            # Determine dimensions
            if flat_dim % 3 == 0:
                n_dims = 3
            elif flat_dim % 2 == 0:
                n_dims = 2
            else:
                raise MorphometricsError("Cannot determine dimensions from flat configuration")

            n_landmarks = flat_dim // n_dims
            X = configurations.reshape(n_specimens, n_landmarks, n_dims)
        elif configurations.ndim == 3:
            X = configurations
        else:
            raise MorphometricsError("Configurations must be 2D (n_specimens, k*m) or 3D (n_specimens, k, m)")

        return X.astype(float)

    def _compute_sizes(self, configurations: npt.NDArray) -> npt.NDArray:
        """
        Compute centroid size for each configuration (vectorized).

        Centroid size = sqrt(trace(X' * X))

        where X is the centered configuration matrix.
        """
        # Vectorized: compute mean for all configurations at once
        means = np.mean(configurations, axis=1, keepdims=True)  # (n, 1, m)
        centered = configurations - means  # Broadcasting

        # Compute centroid sizes for all at once
        sizes = np.sqrt(np.sum(centered**2, axis=(1, 2)))

        return sizes

    def _find_rotation(self, reference: npt.NDArray, target: npt.NDArray) -> npt.NDArray:
        """
        Find optimal rotation to align target to reference using SVD.

        Given reference Y and target X, find R that minimizes:
            ||Y - X @ R||²

        Solution: R = V * U' where U * Σ * V' = X' * Y

        Returns:
            npt.NDArray: Rotation matrix R
        """
        # Compute cross-covariance
        H = target.T @ reference

        # SVD
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError as e:
            raise ComputationError("SVD failed during GPA rotation", original_exception=e)

        # Compute rotation
        R = Vt.T @ U.T

        # Ensure proper rotation (determinant = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            U[:, -1] *= -1
            R = Vt.T @ U.T

        return R

    def _compute_distances_to_consensus(self, aligned: npt.NDArray, consensus: npt.NDArray) -> npt.NDArray:
        """
        Compute Procrustes distances to consensus configuration (vectorized).
        """
        # Vectorized: diff[i] = aligned[i] - consensus for all i
        diff = aligned - consensus  # Broadcasting: (n, k, m) - (k, m) -> (n, k, m)
        distances = np.sqrt(np.sum(diff**2, axis=(1, 2)))

        return distances

    @property
    def last_result(self) -> GPAResult | None:
        """Get the last GPA result."""
        with self._lock:
            return self._last_result
