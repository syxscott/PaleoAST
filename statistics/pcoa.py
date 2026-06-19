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
4. Coordinates: PCoA_i = sqrt(max(λ_i, 0)) * U_i

Author: PaleoAST Development Team
version: 1.0.1
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

            # Sort eigenvalues in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            # Handle negative eigenvalues (can occur with non-Euclidean distances)
            # Set them to zero for coordinates
            negative_count = np.sum(eigenvalues < 0)
            if negative_count > 0:
                self._logger.warning(
                    f"PCoA: {negative_count} negative eigenvalue(s) detected. "
                    f"This indicates non-Euclidean distance metric '{metric}'. "
                    f"Coordinates for negative eigenvalues will be set to zero."
                )
            eigenvalues_positive = np.maximum(eigenvalues, 0)

            # Step 4: Compute coordinates
            # PCoA_i = sqrt(λ_i) * U_i
            sqrt_eigenvalues = np.sqrt(eigenvalues_positive)
            coordinates = eigenvectors * sqrt_eigenvalues

            # Select top n_components
            coordinates = coordinates[:, :n_components]
            eigenvalues = eigenvalues[:n_components]

            # Compute proportion explained.
            # The denominator MUST be the sum of *all* positive eigenvalues,
            # not just the top n_components — otherwise the cumulative
            # proportion cannot reach 100% and the scree plot misleads the
            # user about how much variance the remaining coordinates carry.
            total_eigenvalue = np.sum(eigenvalues_positive)
            if total_eigenvalue > 0:
                proportion = eigenvalues_positive[:n_components] / total_eigenvalue * 100
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
