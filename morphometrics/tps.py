# =============================================================================
# FILE: morphometrics/tps.py
# =============================================================================
"""
Thin-Plate Spline (TPS) Analysis Module for PaleoAST

TPS is used to describe shape deformations between configurations and
to visualize how landmark configurations differ.

Mathematical Foundation:

The Thin-Plate Spline interpolates a deformation from source to target
landmarks by minimizing bending energy.

Bending Energy Function:
    E = ∫∫(f_xx² + 2*f_xy² + f_yy²) dxdy

where f is the TPS interpolation function.

TPS Interpolation Function:
    f(x, y) = a₀ + a₁*x + a₂*y + Σᵢ wᵢ * U(rᵢ)

where:
    U(r) = r² * log(r) is the radial basis function
    rᵢ = sqrt((x-xᵢ)² + (y-yᵢ)²) is the distance to landmark i

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError, MorphometricsError

logger = logging.getLogger(__name__)


@dataclass
class TPSResult:
    """
    Container for TPS analysis results.
    """

    source: npt.NDArray
    target: npt.NDArray
    warp_coefficients: dict[str, Any]  # TPS coefficients
    bending_energy: float
    deformation_grid: npt.NDArray | None  # Grid of warped points
    grid_shape: tuple[int, int] | None

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Thin-Plate Spline Analysis Results')}\n"
            f"{'=' * 50}\n"
            f"{_('Source landmarks: {0}').format(self.source.shape[0])}\n"
            f"{_('Target landmarks: {0}').format(self.target.shape[0])}\n"
            f"{_('Bending energy: {0}').format(f'{self.bending_energy:.6f}')}\n"
            f"{_('Grid shape: {0}').format(self.grid_shape)}"
        )


class TPSAnalyzer:
    """
    Thin-Plate Spline analyzer for landmark deformations.

    TPS computes the non-affine component of shape variation
    and allows visualization of deformation grids.
    """

    def __init__(self) -> None:
        """Initialize the TPS analyzer."""
        self._logger = logging.getLogger(f"{__name__}.TPSAnalyzer")
        self._logger.info("TPSAnalyzer initialized")
        self._lock = threading.RLock()
        self._last_result: TPSResult | None = None

    def analyze(self, source: npt.NDArray, target: npt.NDArray) -> TPSResult:
        """
        Perform TPS analysis between source and target configurations.

        Parameters:
            source: Source landmark configuration (n_landmarks, n_dims)
            target: Target landmark configuration (n_landmarks, n_dims)

        Returns:
            TPSResult: TPS analysis results with deformation coefficients
        """
        with self._lock:
            # Validate configurations
            source = self._validate_configuration(source)
            target = self._validate_configuration(target)

            if source.shape[0] != target.shape[0]:
                raise MorphometricsError(
                    "Source and target must have same number of landmarks",
                    details={"source_landmarks": source.shape[0], "target_landmarks": target.shape[0]},
                )

            n_landmarks, n_dims = source.shape
            self._logger.info(f"TPS fit started: n_landmarks={n_landmarks}, n_dimensions={n_dims}")

            if n_dims not in [2, 3]:
                raise MorphometricsError(f"TPS requires 2D or 3D configurations, got {n_dims}D")

            # Build TPS system using Bookstein's constrained formulation
            # [K  P] [w]   [q]
            # [P^T 0] [a] = [0]
            #
            # where K is kernel matrix, P = [1, x, y] (or [1, x, y, z])
            # w = non-affine weights, a = affine coefficients

            K = self._build_kernel_matrix(source)

            # Build P matrix: [1, x, y] for 2D or [1, x, y, z] for 3D
            if n_dims == 2:
                P = np.column_stack([np.ones(n_landmarks), source])
            else:
                P = np.column_stack([np.ones(n_landmarks), source])

            n_affine = P.shape[1]  # 3 for 2D, 4 for 3D

            # Build block system
            # Top:    [K, P]     (n_landmarks x (n_landmarks + n_affine))
            # Bottom: [P^T, 0]  (n_affine x (n_landmarks + n_affine))
            top = np.hstack([K, P])
            bottom = np.hstack([P.T, np.zeros((n_affine, n_affine))])
            L = np.vstack([top, bottom])

            # Right-hand side: [target; 0]
            rhs = np.vstack([target, np.zeros((n_affine, n_dims))])

            # Solve the block system
            try:
                solution = np.linalg.solve(L, rhs)
            except np.linalg.LinAlgError:
                solution, _, _, _ = np.linalg.lstsq(L, rhs, rcond=None)

            # Extract weights and affine coefficients
            non_affine = solution[:n_landmarks]  # w: (n_landmarks, n_dims)
            affine = solution[n_landmarks:]  # a: (n_affine, n_dims)

            # Compute bending energy
            # E = w' * K * w where K_ij = U(||landmark_i - landmark_j||)
            K = self._build_kernel_matrix(source)
            bending_energy = float(np.sum(non_affine * (K @ non_affine)))
            self._logger.info(f"TPS fit completed: bending_energy={bending_energy:.6f}")

            result = TPSResult(
                source=source,
                target=target,
                warp_coefficients={"affine": affine, "non_affine": non_affine, "full_coefficients": solution},
                bending_energy=bending_energy,
                deformation_grid=None,
                grid_shape=None,
            )

            self._last_result = result
            return result

    def warp_grid(self, result: TPSResult | None = None, grid_shape: tuple[int, int] = (20, 20)) -> npt.NDArray:
        """
        Generate warped grid for visualization.

        Parameters:
            result: TPS result to use. If None, uses last result.
            grid_shape: Shape of output grid (n_rows, n_cols)

        Returns:
            npt.NDArray: Warped grid points (n_rows*n_cols, 2)
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No TPS result available. Run analyze() first.")

        source = result.source
        source.shape[0]
        source.shape[1]

        # Generate grid points spanning the source configuration
        x_min, x_max = source[:, 0].min(), source[:, 0].max()
        y_min, y_max = source[:, 1].min(), source[:, 1].max()

        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin

        # Create grid
        x_grid = np.linspace(x_min, x_max, grid_shape[1])
        y_grid = np.linspace(y_min, y_max, grid_shape[0])
        xx, yy = np.meshgrid(x_grid, y_grid)

        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Warp grid points using TPS
        warped = self._warp_points(grid_points, result)

        return warped.reshape(grid_shape[0], grid_shape[1], 2)

    def _validate_configuration(self, config: npt.NDArray) -> npt.NDArray:
        """Validate and convert configuration to standard format."""
        if isinstance(config, list):
            config = np.array(config)

        if config.ndim == 1:
            # Flattened format
            if len(config) % 2 == 0:
                n_dims = 2
            else:
                n_dims = 3
            n_landmarks = len(config) // n_dims
            config = config.reshape(n_landmarks, n_dims)
        elif config.ndim == 2:
            pass
        else:
            raise MorphometricsError("Configuration must be 1D (flat) or 2D (n_landmarks, n_dims)")

        return config.astype(float)

    def _build_kernel_matrix(self, landmarks: npt.NDArray) -> npt.NDArray:
        """
        Build TPS kernel matrix.

        K_ij = U(r_ij) = r_ij² * log(r_ij)

        where r_ij = ||landmark_i - landmark_j||
        """
        n = landmarks.shape[0]
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.linalg.norm(landmarks[i] - landmarks[j])
                    if r > 0:
                        K[i, j] = r**2 * np.log(r)

        return K

    def _warp_points(self, points: npt.NDArray, result: TPSResult) -> npt.NDArray:
        """
        Warp arbitrary points using TPS coefficients.
        """
        source = result.source
        source.shape[0]
        n_dims = source.shape[1]
        coeffs = result.warp_coefficients["full_coefficients"]

        n_points = points.shape[0]
        warped = np.zeros((n_points, n_dims))

        for p in range(n_points):
            point = points[p]

            # Compute distances to landmarks
            distances = np.array(
                [np.linalg.norm(point - lm) if np.linalg.norm(point - lm) > 0 else 1e-10 for lm in source]
            )

            # Compute U(r) values
            U = distances**2 * np.log(distances)

            # Affine contribution
            if n_dims == 2:
                affine_vals = np.array([1, point[0], point[1]])
            else:
                affine_vals = np.array([1, point[0], point[1], point[2]])

            # Non-affine contribution
            non_affine_vals = U

            # Combine
            combined = np.concatenate([affine_vals, non_affine_vals])

            warped[p] = coeffs.T @ combined

        return warped

    @property
    def last_result(self) -> TPSResult | None:
        """Get the last TPS result."""
        with self._lock:
            return self._last_result
