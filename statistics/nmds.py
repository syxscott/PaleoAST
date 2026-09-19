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

Stress formulas:
    NMDS minimises the agreement between the *disparities* d̃ (the monotone
    least-squares fit of the configuration distances d̂ to the original
    dissimilarities d, obtained by isotonic regression) and d̂:

    Kruskal stress-1 (Kruskal 1964, R vegan::monoMDS default):
        stress_1 = sqrt(Σ(d̂_ij - d̃_ij)² / Σd̂_ij²)

    Normalised "raw" stress (this module's default, kept for backward
    compatibility with v1.0.0):
        raw_stress = sqrt(Σ(d̂_ij - d̃_ij)² / Σd_ij²)

    where:
        d_ij  = original dissimilarity between points i and j
        d̂_ij = distance in the reduced ordination space
        d̃_ij = fitted disparity (isotonic regression of d̂ on d)

    Both values are computed for the winning configuration and reported as
    ``result.stress`` (the formula selected through ``method``) and
    ``result.stress_1`` (Kruskal stress-1, always available).

The SMACOF algorithm (Scaling by MAjorizing a COmplicated Function)
is used for iterative optimization.

Author: PaleoAST Development Team
version: 1.0.2
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.constants import NMDS_MAX_ITERATIONS, NMDS_RANDOM_RESTARTS
from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError
from utils.validators import validate_data_array


def _pava_increasing(y_sorted: npt.NDArray) -> npt.NDArray:
    """Pool-adjacent-violators isotonic fit of ``y_sorted`` (already sorted by predictor).

    O(n) amortised via a block stack; equal to sklearn's
    ``IsotonicRegression(increasing=True)`` evaluated at the data points.
    """
    n = len(y_sorted)
    out = np.empty(n, dtype=float)
    block_sums: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    for i in range(n):
        s = float(y_sorted[i])
        w = 1.0
        start = i
        while block_weights and s / w <= block_sums[-1] / block_weights[-1]:
            s += block_sums.pop()
            w += block_weights.pop()
            start = block_starts.pop()
        block_sums.append(s)
        block_weights.append(w)
        block_starts.append(start)
    for k, start in enumerate(block_starts):
        end = block_starts[k + 1] if k + 1 < len(block_starts) else n
        out[start:end] = block_sums[k] / block_weights[k]
    return out


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
        stress_1: Kruskal (1964) stress-1 of the reported configuration,
            ``sqrt(Σ(d̂-d̃)² / Σd̂²)``.  Always computed (independently of the
            ``method`` that drove the optimisation) so both conventions are
            available; ``None`` only when no configuration was produced.
            New in 1.0.2 - appended last so existing positional construction
            of ``NMDSResult`` keeps working.
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
    stress_1: float | None = None

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
            n_restarts: Number of random restarts (must be >= 1)
            random_seed: Random seed for reproducibility
            tolerance: Convergence tolerance for stress change. Applies to
                this call only; the analyzer keeps its default (1e-6) for
                subsequent calls.
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
            NMDSResult: NMDS analysis results with best configuration.
                ``result.stress`` follows the selected ``method``; Kruskal
                stress-1 of the same configuration is always additionally
                reported as ``result.stress_1``.

        Raises:
            ValidationError: if ``n_restarts`` is smaller than 1.
            MatrixDimensionError: if the matrix is not square.
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
            if n_restarts < 1:
                raise ValidationError(
                    "NMDS requires at least one random restart (n_restarts >= 1)",
                    details={"n_restarts": n_restarts},
                )

            # A caller supplied tolerance applies to *this* run only.  The
            # previous implementation assigned it to ``self._tolerance``,
            # permanently changing the convergence criterion of every later
            # call on the same analyzer instance (e.g. after a GUI dialog had
            # passed a looser tolerance).
            tolerance_value = self._tolerance if tolerance is None else float(tolerance)

            # Store best result across all restarts
            best_stress = float("inf")
            best_coordinates = None
            best_iterations = 0
            best_history = []
            best_stress_1: float | None = None

            # Run multiple restarts
            for restart in range(n_restarts):
                # Initialise the configuration from a LOCAL generator.  The old
                # code called np.random.seed(), which re-seeded NumPy's global
                # legacy RNG and silently perturbed every other consumer of
                # np.random in the same process.  RandomState() reproduces the
                # very same MT19937 stream as np.random.seed(seed) followed by
                # np.random.randn(), so results stay reproducible while global
                # state is left untouched.
                rng_seed = None if random_seed is None else int(random_seed) + restart
                rng = np.random.RandomState(rng_seed)

                # Initialize random configuration
                X = rng.randn(n, n_dimensions) * 0.01
                self._logger.debug(f"NMDS restart {restart + 1}/{n_restarts} started")

                # Run SMACOF optimization
                result = self._smacof(
                    D, X, max_iterations, restart, method=method, tolerance=tolerance_value
                )
                self._logger.debug(
                    f"NMDS restart {restart + 1}/{n_restarts} finished: "
                    f"stress={result['stress']:.6f}, iterations={result['n_iterations']}"
                )

                if result["stress"] < best_stress:
                    best_stress = result["stress"]
                    best_coordinates = result["coordinates"]
                    best_iterations = result["n_iterations"]
                    best_history = result["stress_history"]
                    best_stress_1 = result.get("stress_1")

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
                stress_1=best_stress_1,
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
        tolerance: float | None = None,
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

        Stress formulas (both are evaluated every iteration):
        * raw_stress (default, backward-compatible with the v1.0.0 algorithm):
            sqrt(sum((d_hat - d_tilde)^2) / sum(d_target^2))
        * stress_1 (Kruskal 1964 canonical, R vegan::monoMDS):
            sqrt(sum((d_hat - d_tilde)^2) / sum(d_hat^2))

        ``method`` only selects which of the two drives the optimisation
        (stopping rule and restart selection); both are returned so the
        caller can report Kruskal stress-1 regardless of the default.

        Args:
            tolerance: Stress-change convergence threshold for this run.
                Defaults to ``self._tolerance``.

        Performance optimisations vs. the original implementation:
        * The isotonic regression (pool-adjacent-violators) is computed with
          a small dependency-free NumPy/Python helper instead of sklearn,
          which is not part of this project's dependency set.
        * Goodness-of-fit (stress) is computed with fully vectorised NumPy
          operations with no Python-level distance loops.
        """
        n = X_init.shape[0]  # n_samples
        X = X_init.copy()
        stress_history: list[float] = []
        if tolerance is None:
            tolerance = self._tolerance

        # Pre-compute the upper-triangular indices of the dissimilarity
        # matrix. Only the off-diagonal (i < j) entries are used by NMDS.
        iu, ju = np.triu_indices(n, k=1)
        d_target = D[iu, ju]

        # The isotonic fit is taken with respect to d_target, which is fixed
        # across iterations: sort once, then map results back per iteration.
        sort_order = np.argsort(d_target, kind="stable")
        unsort_idx = np.empty(len(sort_order), dtype=np.intp)
        unsort_idx[sort_order] = np.arange(len(sort_order))

        # Pre-allocate working arrays to avoid repeated allocation inside the loop
        d_hat = np.empty(len(d_target), dtype=float)
        D_hat = np.empty((n, n), dtype=float)
        D_tilde = np.zeros((n, n), dtype=float)
        B = np.zeros((n, n), dtype=float)

        # sum(d_target^2) is constant across iterations; hoist it out.
        denom_target = float(np.dot(d_target, d_target))
        stress_raw = float("inf")
        stress_1 = float("inf")

        for iteration in range(max_iterations):
            # Compute distances in current configuration
            self._compute_distances_inplace(X, D_hat)
            d_hat[:] = D_hat[iu, ju]

            # Isotonic regression: find the monotone sequence d_tilde that
            # follows the rank order of the fixed d_target while minimizing
            # sum((d_hat - d_tilde)^2). This is the defining step of NMDS
            # (Kruskal 1964), solved with the pool-adjacent-violators
            # algorithm below.
            d_tilde = _pava_increasing(d_hat[sort_order])[unsort_idx]

            # Goodness of fit.  Both normalisations are evaluated every
            # iteration and returned; ``method`` only decides which one is
            # reported as ``stress`` (and therefore which one drives restart
            # selection and the stopping rule).
            #
            # NOTE on attribution: the *canonical* Kruskal (1964) stress-1
            # divides by sum(d_hat^2) -- the configuration distances.  The
            # denominator used by 'raw_stress' here, sum(d_target^2) (the
            # original dissimilarities), is NOT a Kruskal formula; it is a
            # legacy normalisation kept so that v1.0.0 results and the
            # reference test fixtures remain reproducible.
            if method not in ("raw_stress", "stress_1"):
                raise ValueError(
                    f"Unknown NMDS stress method '{method}'. "
                    "Use 'raw_stress' (default) or 'stress_1'."
                )
            diff = d_hat - d_tilde
            numerator = float(np.dot(diff, diff))
            denom_hat = float(np.dot(d_hat, d_hat))
            stress_raw = np.sqrt(numerator / denom_target) if denom_target > 0 else float("inf")
            stress_1 = np.sqrt(numerator / denom_hat) if denom_hat > 0 else float("inf")
            stress = stress_raw if method == "raw_stress" else stress_1
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
                if stress_change < tolerance:
                    logger.debug(f"SMACOF restart={restart_id} converged at iteration {iteration}: stress={stress:.6f}")
                    break

            # Build the working disparity matrix D_tilde from the
            # upper-triangular disparities and mirror it symmetrically.
            D_tilde[iu, ju] = d_tilde
            D_tilde[ju, iu] = d_tilde

            # Guttman transform using the disparities D_tilde.
            # B[i,j] = -d_tilde_ij / d_hat_ij  if d_hat_ij > 0  else 0
            # B[i,i] = -sum_{j!=i} B[i,j]      (row sums zero)
            #
            # ``np.divide(..., where=mask)`` only writes the entries where the
            # mask is True, so every other cell must be reset explicitly:
            # ``B`` is reused across iterations, and without the reset the
            # coincident-point cells (d_hat == 0, including the diagonal)
            # would carry over the previous iteration's values and corrupt the
            # majorization step.
            mask = D_hat > 0
            np.divide(-D_tilde, D_hat, out=B, where=mask)
            B[~mask] = 0.0
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
            # Both normalisations of the last *evaluated* configuration, so
            # Kruskal stress-1 is available even when raw_stress drove the
            # optimisation (and vice versa).
            "raw_stress": stress_raw if stress_history else float("inf"),
            "stress_1": stress_1 if stress_history else float("inf"),
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
