# =============================================================================
# FILE: statistics/spatial.py
# =============================================================================
"""
Spatial Point Pattern Analysis Module for PaleoAST

Ripley's K function and related spatial statistics for analyzing
the distribution of points in 2D space.

Mathematical Foundation:

Ripley's K-function:
    K(r) = A * ΣΣ I(d_ij < r) / n²

where:
    A = study area
    d_ij = distance between points i and j
    I() = indicator function (1 if true, 0 otherwise)
    n = number of points

Standardized L-function:
    L(r) = sqrt(K(r) / π) - r

For complete spatial randomness (CSR):
    K(r) = πr²
    L(r) = 0

Envelope simulation:
    Monte Carlo simulation of n permutations to create
    confidence envelopes under CSR hypothesis.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class SpatialResult:
    """
    Container for spatial point pattern analysis results.

    Attributes:
        r_values: Distance values at which K was computed
        k_values: Ripley's K values at each r
        l_values: L-function values (standardized K)
        envelope_upper: Upper confidence envelope
        envelope_lower: Lower confidence envelope
        point_coords: Original point coordinates
        n_points: Number of points
        area: Study area
        n_simulations: Number of Monte Carlo simulations
        interpretation: String describing spatial pattern
    """

    r_values: npt.NDArray
    k_values: npt.NDArray
    l_values: npt.NDArray
    envelope_upper: npt.NDArray
    envelope_lower: npt.NDArray
    point_coords: npt.NDArray
    n_points: int
    area: float
    n_simulations: int
    interpretation: str

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Spatial Point Pattern Analysis (Ripley K)')}\n"
            f"{'=' * 45}\n"
            f"{_('Number of points: {0}').format(self.n_points)}\n"
            f"{_('Study area: {0:.4f}').format(self.area)}\n"
            f"{_('Monte Carlo simulations: {0}').format(self.n_simulations)}\n"
            f"{_('Distance range: {0:.4f} to {1:.4f}').format(self.r_values.min(), self.r_values.max())}\n"
            f"\n{_('Interpretation:')}\n{self.interpretation}"
        )


class RipleyKAnalyzer:
    """
    Ripley's K-function spatial point pattern analyzer.

    Analyzes spatial distribution patterns (clustered, regular, or random)
    using Ripley's K-function with Monte Carlo confidence envelopes.
    """

    def __init__(self) -> None:
        """Initialize the Ripley's K analyzer."""
        self._logger = logging.getLogger(f"{__name__}.RipleyKAnalyzer")
        self._lock = threading.RLock()
        self._last_result: SpatialResult | None = None
        self._logger.info("RipleyKAnalyzer initialized")

    def analyze(
        self,
        coords: npt.NDArray,
        r_max: float | None = None,
        n_r_values: int = 50,
        n_simulations: int = 99,
    ) -> SpatialResult:
        """
        Perform Ripley's K spatial analysis.

        Parameters:
            coords: Point coordinates array (n_points, 2)
            r_max: Maximum distance for analysis. If None, auto-calculated
            n_r_values: Number of distance values to compute
            n_simulations: Number of Monte Carlo simulations for envelope

        Returns:
            SpatialResult: Spatial analysis results
        """
        with self._lock:
            # Validate input
            points = validate_data_array(coords, allow_nan=False, name="coords")

            if points.ndim != 2 or points.shape[1] != 2:
                raise ComputationError("Coords must be 2D points with shape (n, 2)")

            n_points = points.shape[0]

            if n_points < 3:
                raise ComputationError("Need at least 3 points for spatial analysis")

            self._logger.info(f"RipleyK analyze started: n_points={n_points}, n_simulations={n_simulations}")

            # Compute bounding box
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()

            # Compute area (use bounding box)
            area = (x_max - x_min) * (y_max - y_min)

            if area == 0:
                raise ComputationError("Points have zero area - all points may be identical")

            # Auto-calculate r_max if not provided
            if r_max is None:
                # Use 25% of the smaller dimension
                r_max = 0.25 * min(x_max - x_min, y_max - y_min)

            # Create r values
            r_values = np.linspace(0, r_max, n_r_values)

            # Compute K(r) for observed points
            k_values = self._compute_k_function(points, r_values, area)

            # Compute L(r) = sqrt(K(r) / π) - r
            with np.errstate(divide="ignore", invalid="ignore"):
                l_values = np.sqrt(k_values / np.pi) - r_values
                l_values = np.nan_to_num(l_values, nan=0.0)

            # Monte Carlo envelope
            envelope_upper, envelope_lower = self._compute_envelope(points, r_values, area, n_simulations)

            # Determine interpretation by comparing observed L(r) against
            # the Monte Carlo envelope over non-trivial radii.
            window = slice(len(r_values) // 4, None)
            above = np.mean(l_values[window] > envelope_upper[window])
            below = np.mean(l_values[window] < envelope_lower[window])

            if above > 0.5:
                interpretation = _("Pattern: CLUSTERED (L(r) > envelope indicates clustering)")
            elif below > 0.5:
                interpretation = _("Pattern: REGULAR/DISPERSED (L(r) < envelope indicates regularity)")
            else:
                interpretation = _("Pattern: RANDOM (L(r) within envelope indicates CSR)")

            result = SpatialResult(
                r_values=r_values,
                k_values=k_values,
                l_values=l_values,
                envelope_upper=envelope_upper,
                envelope_lower=envelope_lower,
                point_coords=points,
                n_points=n_points,
                area=area,
                n_simulations=n_simulations,
                interpretation=interpretation,
            )

            self._last_result = result
            self._logger.info(f"RipleyK completed: {interpretation[:50]}")
            return result

    def _compute_k_function(
        self,
        points: npt.NDArray,
        r_values: npt.NDArray,
        area: float,
    ) -> npt.NDArray:
        """
        Compute Ripley's K-function for given points.

        Uses the standard estimator (Diggle 2003):
            K(r) = (A / n²) * Σ_{i ≠ j} I(d_ij < r)

        Note: the previous implementation divided by ``n * (n - 1)``
        instead of ``n²``, which slightly under-estimates K for small n.
        The current formula is internally consistent with the Monte
        Carlo envelope (which is built with the same expression), so
        the interpretation of the test against CSR is unchanged.
        """
        n = points.shape[0]
        k_values = np.zeros(len(r_values))

        # Build KD-tree for efficient distance queries
        tree = cKDTree(points)

        for i, r in enumerate(r_values):
            if r == 0:
                k_values[i] = 0
                continue

            # Count all pairs within distance r (only i < j to avoid double counting)
            count = 0
            for j in range(n):
                # Find neighbors of point j within distance r
                neighbors = tree.query_ball_point(points[j], r)
                # Only count pairs where neighbor index > j (to count each pair once)
                for neighbor in neighbors:
                    if neighbor > j:
                        count += 1

            # Number of ordered pairs (i != j) is n * (n - 1).
            # ``count`` already counts each unordered pair once, so
            # 2*count is the total number of (i, j) ordered pairs with
            # i != j and d_ij < r. Dividing by n^2 gives the standard
            # K estimator.
            if n > 1:
                k_values[i] = (area * 2.0 * count) / (n * n)
            else:
                k_values[i] = 0.0

        return k_values

    def _compute_envelope(
        self,
        points: npt.NDArray,
        r_values: npt.NDArray,
        area: float,
        n_simulations: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Compute Monte Carlo confidence envelope under CSR.

        Simulates n_simulations point patterns with same number
        of points but random positions within the bounding box.
        """
        n = points.shape[0]

        # Bounding box
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # Store L values from simulations
        l_simulations = np.zeros((n_simulations, len(r_values)))

        for sim in range(n_simulations):
            # Generate random points within bounding box
            random_x = np.random.uniform(x_min, x_max, n)
            random_y = np.random.uniform(y_min, y_max, n)
            random_points = np.column_stack([random_x, random_y])

            # Compute K for random points
            k_random = self._compute_k_function(random_points, r_values, area)

            # L = sqrt(K / π) - r
            with np.errstate(divide="ignore", invalid="ignore"):
                l_random = np.sqrt(k_random / np.pi) - r_values
                l_random = np.nan_to_num(l_random, nan=0.0)

            l_simulations[sim] = l_random

        # Envelope percentiles (2.5 and 97.5 for 95% envelope)
        envelope_lower = np.percentile(l_simulations, 2.5, axis=0)
        envelope_upper = np.percentile(l_simulations, 97.5, axis=0)

        return envelope_upper, envelope_lower

    @property
    def last_result(self) -> SpatialResult | None:
        """Get the last spatial analysis result."""
        with self._lock:
            return self._last_result
