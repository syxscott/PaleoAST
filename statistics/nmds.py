# =============================================================================
# FILE: statistics/nmds.py
# =============================================================================
"""
Non-metric Multidimensional Scaling (NMDS) Module for PaleoAST

NMDS is an ordination technique that finds a configuration of points in
low-dimensional space that preserves the rank order of pairwise distances.

Mathematical Foundation:

NMDS minimizes a stress function that measures the disagreement between
the rank order of distances in the original space and the reduced space.

Stress Formula:
    Stress = sqrt(Σ(d_ij - d̂_ij)² / Σd_ij²)

where:
    d_ij = original distance between points i and j
    d̂_ij = distance in reduced ordination space

The SMACOF algorithm (Scaling by MAjorizing a COmplicated Function)
is used for iterative optimization.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.constants import NMDS_MAX_ITERATIONS, NMDS_RANDOM_RESTARTS
from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class NMDSResult:
    """
    Container for NMDS analysis results.

    Attributes:
        coordinates: Configuration matrix in reduced space (n_samples × n_dimensions)
        stress: Final stress value
        stress_formula: Which stress formula was used ('raw_stress' = default,
                       denominator uses d_target, or 'stress_1' = Kruskal 1964 canonical,
                       denominator uses d_hat)
        n_iterations: Number of iterations to convergence
        converged: Whether SMACOF converged within tolerance
        distance_matrix: Original dissimilarity matrix
        stress_history: Per-iteration stress values
        metric: Distance metric name (for reference)
        n_restarts: Number of random restarts performed
    """

    coordinates: npt.NDArray
    stress: float
    n_iterations: int
    converged: bool
    distance_matrix: npt.NDArray
    stress_history: list
    metric: str
    n_restarts: int
    stress_formula: str = "raw_stress"

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Non-metric MDS Results')}\n"
            f"{'=' * 40}\n"
            f"{_('Distance metric: {0}').format(self.metric)}\n"
            f"{_('Stress formula: {0}').format(self.stress_formula)}\n"
            f"{_('Final stress: {0}').format(f'{self.stress:.4f}')}\n"
            f"{_('Iterations: {0}').format(self.n_iterations)}\n"
            f"{_('Converged: {0}').format('Yes' if self.converged else 'No')}\n"
            f"{_('Random restarts: {0}').format(self.n_restarts)}"
        )


class NMDSAnalyzer:
    """
    Non-metric Multidimensional Scaling analyzer.

    NMDS uses an iterative approach (SMACOF algorithm) to find
    an ordination that preserves the rank order of dissimilarities.
    Multiple random restarts help avoid local minima.
    """

    def __init__(self) -> None:
        """Initialize the NMDS analyzer."""
        self._logger = logging.getLogger(f"{__name__}.NMDSAnalyzer")
        self._lock = threading.RLock()
        self._last_result: NMDSResult | None = None
        self._max_iterations = NMDS_MAX_ITERATIONS
        self._n_restarts = NMDS_RANDOM_RESTARTS
        self._tolerance = 1e-6
        self._logger.info("NMDSAnalyzer initialized")

    def analyze(
        self,
        distance_matrix: npt.NDArray,
        n_dimensions: int = 2,
        metric: str = "euclidean",
        max_iterations: int | None = None,
        n_restarts: int | None = None,
        random_seed: int | None = None,
        tolerance: float | None = None,
        progress_callback=None,
        method: str = "raw_stress",
    ) -> NMDSResult:
        """
        Perform Non-metric MDS.

        Parameters:
            distance_matrix: Dissimilarity matrix
            n_dimensions: Number of dimensions for ordination
            metric: Original distance metric (for reference)
            max_iterations: Maximum iterations per restart
            n_restarts: Number of random restarts
            random_seed: Random seed for reproducibility
            tolerance: Convergence tolerance for stress change
            progress_callback: Optional callable(restart_index, total_restarts, stress)
                               called at the end of each restart with the final stress.
                               Intended for GUI progress bars.
            method: Stress formula to use. One of:
                - 'raw_stress' (default, backward-compatible): sqrt(sum((d_hat - d_tilde)^2) / sum(d_target^2)).
                  Denominator uses original distances (d_target). Matches the
                  v1.0.0 algorithm and the reference test fixture.
                - 'stress_1' (Kruskal 1964 canonical, R vegan::monoMDS):
                  sqrt(sum((d_hat - d_tilde)^2) / sum(d_hat^2)).
                  Denominator uses configuration distances (d_hat).

        Returns:
            NMDSResult: NMDS analysis results with best configuration
        """
        with self._lock:
            # Validate input
            D = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            n = D.shape[0]
            self._logger.info(
                f"NMDS analyze started: distance matrix {D.shape[0]}x{D.shape[1]}, "
                f"n_dimensions={n_dimensions}, n_restarts={n_restarts}, metric={metric}"
            )

            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square", details={"shape": D.shape})

            # Set parameters
            if max_iterations is None:
                max_iterations = self._max_iterations
            if n_restarts is None:
                n_restarts = self._n_restarts
            if tolerance is not None:
                self._tolerance = tolerance

            # Store best result across all restarts
            best_stress = float("inf")
            best_coordinates = None
            best_iterations = 0
            best_history = []

            # Run multiple restarts
            for restart in range(n_restarts):
                if random_seed is not None:
                    np.random.seed(random_seed + restart)

                # Initialize random configuration
                X = np.random.randn(n, n_dimensions) * 0.01
                self._logger.debug(f"NMDS restart {restart + 1}/{n_restarts} started")

                # Run SMACOF optimization
                result = self._smacof(D, X, max_iterations, restart, method=method)
                self._logger.debug(
                    f"NMDS restart {restart + 1}/{n_restarts} finished: "
                    f"stress={result['stress']:.6f}, iterations={result['n_iterations']}"
                )

                if result["stress"] < best_stress:
                    best_stress = result["stress"]
                    best_coordinates = result["coordinates"]
                    best_iterations = result["n_iterations"]
                    best_history = result["stress_history"]

                # Report progress if callback is provided
                if progress_callback is not None:
                    progress_callback(restart, n_restarts, result["stress"])

            # NMDS convergence threshold: stress < 0.05 is considered converged
            # stress < 0.10 is acceptable, stress > 0.20 is poor fit
            CONVERGED_THRESHOLD = 0.05
            nmds_result = NMDSResult(
                coordinates=best_coordinates,
                stress=best_stress,
                n_iterations=best_iterations,
                converged=best_stress < CONVERGED_THRESHOLD,
                distance_matrix=D,
                stress_history=best_history,
                metric=metric,
                n_restarts=n_restarts,
                stress_formula=method,
            )

            self._last_result = nmds_result
            self._logger.info(
                f"NMDS completed: final stress={best_stress:.6f}, "
                f"iterations={best_iterations}, converged={nmds_result.converged}"
            )
            if best_stress > 0.20:
                self._logger.warning(
                    f"NMDS poor fit: stress={best_stress:.4f} > 0.20, consider increasing n_restarts or n_dimensions"
                )
            elif not nmds_result.converged:
                self._logger.warning(f"NMDS did not converge: best stress={best_stress:.6f}")
            return nmds_result

    def _smacof(
        self,
        D: npt.NDArray,
        X_init: npt.NDArray,
        max_iterations: int,
        restart_id: int,
        method: str = "raw_stress",  # 与 analyze() 的默认保持一致
    ) -> dict[str, Any]:
        """
        SMACOF algorithm for NMDS optimization.

        This is the *non-metric* SMACOF: each iteration first performs an
        isotonic regression of the current configuration distances ``d̂``
        against the fixed original dissimilarities ``D`` to obtain the
        disparities ``d̃`` (target distances that monotonically follow the
        rank order of ``D``), then applies a Guttman majorization step
        that minimizes ``Σ(d̃ - d̂)²``. Without the isotonic-regression
        step this collapses to *metric* MDS and the rank-order
        preservation that defines NMDS (Kruskal 1964) is lost.

        Stress formulas:
        * raw_stress (default, backward-compatible with v1.0.0 algorithm):
            sqrt(sum((d_hat - d_tilde)^2) / sum(d_target^2))
        * stress_1 (Kruskal 1964 canonical, R vegan::monoMDS):
            sqrt(sum((d_hat - d_tilde)^2) / sum(d_hat^2))

        Performance optimisations vs. the original implementation:
        * IsotonicRegression is instantiated **once** outside the iteration
          loop and reused via :meth:`fit_transform` (previously a new object
          was created every iteration, causing ~50 restart × 500 iter = 25,000
          redundant allocations).
        * Goodness-of-fit (stress) is computed with fully vectorised NumPy
          operations with no Python-level loops.
        """
        from sklearn.isotonic import IsotonicRegression

        n = X_init.shape[0]  # n_samples
        X = X_init.copy()
        stress_history: list[float] = []

        # Pre-compute the upper-triangular indices of the dissimilarity
        # matrix. Only the off-diagonal (i < j) entries are used by NMDS.
        iu, ju = np.triu_indices(n, k=1)
        d_target = D[iu, ju]

        # Pre-allocate working arrays to avoid repeated allocation inside the loop
        d_hat = np.empty(len(d_target), dtype=float)
        d_tilde = np.empty(len(d_target), dtype=float)
        D_hat = np.empty((n, n), dtype=float)
        D_tilde = np.zeros((n, n), dtype=float)
        B = np.empty((n, n), dtype=float)

        # Create IsotonicRegression ONCE and reuse it across all iterations.
        # This eliminates ~max_iterations object allocations per restart.
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

        for iteration in range(max_iterations):
            # Compute distances in current configuration
            self._compute_distances_inplace(X, D_hat)
            d_hat[:] = D_hat[iu, ju]

            # Isotonic regression: find the monotone sequence d_tilde that
            # follows the rank order of the fixed d_target while minimizing
            # sum((d_hat - d_tilde)^2). This is the defining step of NMDS
            # (Kruskal 1964). Pool-adjacent-violators algorithm via sklearn.
            # Reusing the same iso instance with fit_transform is significantly
            # faster than creating a new IsotonicRegression() each iteration.
            d_tilde = iso.fit_transform(d_target, d_hat)

            # Stress formula selection.
            # Default = 'raw_stress' (sqrt(sum((d_hat - d_tilde)^2) / sum(d_target^2)))
            # to preserve backward compatibility with existing tests, callers, and
            # the original algorithm shipped in v1.0.0 / v1.0.1. The denominator uses
            # the original distances (d_target) per Kruskal 1964 / Borg & Groenen 1997
            # for normalized stress.
            #
            # 'stress_1' (sqrt(sum((d_hat - d_tilde)^2) / sum(d_hat^2))) is the
            # alternative form sometimes reported in newer textbooks and used by
            # R vegan::monoMDS; opt in by passing method='stress_1'.
            diff = d_hat - d_tilde
            numerator = np.dot(diff, diff)
            if method == "raw_stress":
                # Default: denominator = sum(d_target^2)
                denom = np.dot(d_target, d_target)
            elif method == "stress_1":
                # Alternative: denominator = sum(d_hat^2)
                denom = np.dot(d_hat, d_hat)
            else:
                raise ValueError(
                    f"Unknown NMDS stress method '{method}'. "
                    "Use 'raw_stress' (default) or 'stress_1'."
                )
            if denom > 0:
                stress = np.sqrt(numerator / denom)
            else:
                stress = float("inf")
            stress_history.append(stress)

            # Log convergence progress every 50 iterations
            if iteration > 0 and iteration % 50 == 0:
                logger.debug(
                    f"SMACOF restart={restart_id} iteration={iteration}: "
                    f"stress={stress:.6f}, change={abs(stress_history[-1] - stress_history[-2]):.8f}"
                )

            # Check convergence
            if iteration > 0:
                stress_change = abs(stress_history[-1] - stress_history[-2])
                if stress_change < self._tolerance:
                    logger.debug(f"SMACOF restart={restart_id} converged at iteration {iteration}: stress={stress:.6f}")
                    break

            # Build the working disparity matrix D_tilde from the
            # upper-triangular disparities and mirror it symmetrically.
            D_tilde[iu, ju] = d_tilde
            D_tilde[ju, iu] = d_tilde

            # Guttman transform using the disparities D_tilde.
            # B[i,j] = -d_tilde_ij / d_hat_ij  if d_hat_ij > 0  else 0
            # B[i,i] = -sum_{j!=i} B[i,j]      (row sums zero)
            mask = D_hat > 0
            np.divide(-D_tilde, D_hat, out=B, where=mask)
            np.fill_diagonal(B, 0.0)
            row_sums = B.sum(axis=1)
            np.fill_diagonal(B, -row_sums)

            # Guttman transform: X_new = (1/n) * B @ X
            # In-place update via explicit allocation to avoid aliasing X
            X_new = B @ X
            X_new /= n
            X = X_new

        return {
            "coordinates": X,
            "stress": stress_history[-1] if stress_history else float("inf"),
            "n_iterations": len(stress_history),
            "stress_history": stress_history,
        }

    def _compute_distances(self, X: npt.NDArray) -> npt.NDArray:
        """
        Compute pairwise Euclidean distances using scipy for efficiency.
        """
        from scipy.spatial.distance import cdist

        n = X.shape[0]
        if n <= 500:
            # Use broadcasting for small matrices
            diff = X[:, None, :] - X[None, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
        else:
            # Use cdist for large matrices (more memory efficient)
            D = cdist(X, X, metric="euclidean")

        return D

    def _compute_distances_inplace(self, X: npt.NDArray, out: npt.NDArray) -> None:
        """
        Compute pairwise Euclidean distances into a pre-allocated array.

        This avoids the allocation overhead of :meth:`_compute_distances`
        when called inside the SMACOF hot loop.

        Parameters:
            X: Configuration matrix of shape (n, n_dimensions)
            out: Pre-allocated output array of shape (n, n)
        """
        from scipy.spatial.distance import cdist

        n = X.shape[0]
        if n <= 500:
            # Use broadcasting for small matrices
            diff = X[:, None, :] - X[None, :, :]
            np.sqrt(np.sum(diff**2, axis=2), out=out)
        else:
            # Use cdist for large matrices (more memory efficient)
            cdist(X, X, metric="euclidean", out=out)

    def get_shepard_data(self, result: NMDSResult | None = None) -> dict[str, npt.NDArray | float]:
        """
        Get data for Shepard diagram.

        Returns original vs. ordination distances for assessing fit.
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No NMDS result available")

        # Get original and ordination distances
        D_orig = result.distance_matrix
        D_ord = self._compute_distances(result.coordinates)

        # Flatten for plotting (upper triangle only)
        n = D_orig.shape[0]
        indices = np.triu_indices(n, k=1)

        return {
            "original": D_orig[indices],
            "ordination": D_ord[indices],
            "stress": float(result.stress),
        }

    @property
    def last_result(self) -> NMDSResult | None:
        """Get the last computed NMDS result."""
        with self._lock:
            return self._last_result
