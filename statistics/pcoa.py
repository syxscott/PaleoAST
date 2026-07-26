# =============================================================================
# FILE: statistics/pcoa.py
# =============================================================================
"""
Principal Coordinate Analysis (PCoA) Module for PaleoAST

PCoA (also called Metric MDS) finds principal coordinates that best
represent the distances in a dissimilarity matrix.

Mathematical Foundation:

Given a distance/dissimilarity matrix D ∈ ℝ^(n×n):

1. Square the distances: D² = [d²_ij]
2. Double centering: B = -0.5 * J * D² * J
   where J = I - (1/n) * 1*1^T is the centering matrix
3. Eigendecomposition: B = U * Λ * U^T
   where Λ = diag(λ₁, λ₂, ..., λₙ) are eigenvalues
4. Coordinates: PCoA_i = sqrt(|λ_i|) * U_i
   (using absolute eigenvalue because negative eigenvalues indicate
   non-Euclidean structure; the sign is absorbed into eigenvector direction)

Negative Eigenvalues:
    Negative eigenvalues are a natural consequence of non-Euclidean distance
    metrics (e.g., Bray-Curtis, Jaccard, unweighted UniFrac). They indicate
    that the distance matrix cannot be perfectly represented in Euclidean
    space. This is NOT an error condition.

    - The ABSOLUTE value |λ_i| determines the axis length (variance explained)
    - The SIGN of λ_i indicates whether samples on that axis diverge (negative)
      or converge (positive) relative to the centroid
    - Negative eigenvalues are preserved because they contain meaningful
      biological information about dissimilarity structure

    This implementation follows the convention of R's cmdscale(eig=TRUE) and
    ape::pcoa(), which also preserve negative eigenvalues. Users should
    interpret negative eigenvalues as indicators of non-metric distance
    structure; the absolute value represents the axis's contribution to
    explained variance.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class PCoAResult:
    """
    Container for PCoA analysis results.
    """

    coordinates: npt.NDArray
    eigenvalues: npt.NDArray
    proportion_explained: npt.NDArray
    cumulative_proportion: npt.NDArray
    distance_matrix: npt.NDArray
    n_components: int
    metric: str

    def get_coordinates(self, n_components: int | None = None) -> npt.NDArray:
        """Get coordinates for specified number of components."""
        if n_components is None:
            return self.coordinates
        return self.coordinates[:, :n_components]

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Principal Coordinate Analysis Results"),
            "=" * 50,
            _("Distance metric: {0}").format(self.metric),
            _("Number of coordinates: {0}").format(self.n_components),
            "",
            _("Coord | Eigenvalue | Proportion | Cumulative"),
            "-" * 50,
        ]
        for i in range(min(10, self.n_components)):
            lines.append(
                f"PC{i + 1:4d} | {self.eigenvalues[i]:10.4f} | "
                f"{self.proportion_explained[i]:10.4f} | "
                f"{self.cumulative_proportion[i]:10.4f}"
            )
        return "\n".join(lines)


class PCoAAnalyzer:
    """
    Principal Coordinate Analysis engine.

    PCoA is a dimension reduction technique that operates on distance matrices,
    making it suitable for ecological data with Bray-Curtis, Jaccard, and
    other non-Euclidean distances.
    """

    def __init__(self) -> None:
        """Initialize the PCoA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.PCoAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: PCoAResult | None = None
        self._logger.info("PCoAAnalyzer initialized")

    def analyze(
        self, distance_matrix: npt.NDArray, n_components: int | None = None, metric: str = "unknown"
    ) -> PCoAResult:
        """
        Perform Principal Coordinate Analysis.

        Parameters:
            distance_matrix: Square distance/dissimilarity matrix
            n_components: Number of coordinates to extract
            metric: Name of distance metric (for reference)

        Returns:
            PCoAResult: PCoA analysis results
        """
        with self._lock:
            # Validate distance matrix
            D = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            n = D.shape[0]
            self._logger.info(
                f"PCoA analyze started: distance matrix {D.shape[0]}x{D.shape[1]}, "
                f"n_components={n_components}, metric={metric}"
            )

            # Check square matrix
            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square", details={"shape": D.shape})

            if n < 2:
                raise MatrixDimensionError("PCoA requires at least 2 samples", details={"n_samples": n})

            # Determine number of components
            if n_components is None:
                n_components = min(n - 1, 20)
            else:
                n_components = min(n_components, n - 1)

            # Step 1: Square the distances
            D_sq = D**2

            # Step 2: Double centering
            # B = -0.5 * J * D² * J
            # where J = I - (1/n) * 1*1^T
            n_float = float(n)
            ones = np.ones((n, n))
            J = np.eye(n) - (1.0 / n_float) * ones

            # Matrix multiplication: B = -0.5 * J * D² * J
            B = -0.5 * J @ D_sq @ J

            # Step 3: Eigendecomposition
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(B)
            except np.linalg.LinAlgError as e:
                raise ComputationError("Eigendecomposition failed during PCoA", original_exception=e)

            # Check for negative eigenvalues (common with non-Euclidean distances)
            # and warn explicitly - they are NOT truncated, but preserved
            negative_mask = eigenvalues < 0
            negative_count = np.sum(negative_mask)
            if negative_count > 0:
                import warnings
                warnings.warn(
                    f"PCoA: {negative_count} negative eigenvalue(s) detected "
                    f"(metric='{metric}'). Negative eigenvalues indicate non-Euclidean "
                    f"distance structure (common for Bray-Curtis, Jaccard, etc.). "
                    f"The absolute value represents axis length; the sign indicates "
                    f"divergence direction. Negative eigenvalues are preserved, "
                    f"following R cmdscale()/ape::pcoa() conventions.",
                    UserWarning,
                    stacklevel=2,
                )
                self._logger.warning(
                    f"PCoA: {negative_count} negative eigenvalue(s) preserved. "
                    f"Non-Euclidean metric '{metric}' detected."
                )

            # Sort by absolute value (descending) to prioritize axes by
            # variance explained regardless of sign
            abs_eigenvalues = np.abs(eigenvalues)
            sorted_indices = np.argsort(abs_eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # Step 4: Compute coordinates using absolute eigenvalues
            # PCoA_i = sqrt(|λ_i|) * U_i
            # The sign of λ_i is preserved in the eigenvector direction,
            # so using sqrt(|λ_i|) captures the magnitude while the
            # eigenvector sign carries the original sign information
            sqrt_abs_eigenvalues = np.sqrt(np.abs(eigenvalues))
            coordinates = eigenvectors * sqrt_abs_eigenvalues

            # Select top n_components
            coordinates = coordinates[:, :n_components]
            eigenvalues = eigenvalues[:n_components]

            # Compute proportion explained using absolute values
            # This ensures negative eigenvalues contribute proportionally
            total_abs_eigenvalue = np.sum(np.abs(eigenvalues))
            if total_abs_eigenvalue > 0:
                proportion = np.abs(eigenvalues) / total_abs_eigenvalue * 100
            else:
                proportion = np.zeros(n_components)

            cumulative = np.cumsum(proportion)

            result = PCoAResult(
                coordinates=coordinates,
                eigenvalues=eigenvalues,
                proportion_explained=proportion,
                cumulative_proportion=cumulative,
                distance_matrix=D,
                n_components=n_components,
                metric=metric,
            )

            self._last_result = result
            self._logger.info(
                f"PCoA completed: top eigenvalues={eigenvalues[:3].tolist()}, "
                f"cumulative proportion={cumulative[-1]:.2f}%"
            )
            return result

    @property
    def last_result(self) -> PCoAResult | None:
        """Get the last PCoA result."""
        with self._lock:
            return self._last_result
