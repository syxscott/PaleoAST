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
Version: 1.0.0
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import threading
import random

from utils.exceptions import ComputationError, ConvergenceError, MatrixDimensionError
from utils.validators import validate_data_array
from config.constants import NMDS_MAX_ITERATIONS, NMDS_RANDOM_RESTARTS
from config.i18n import _

logger = logging.getLogger(__name__)


@dataclass
class NMDSResult:
    """
    Container for NMDS analysis results.
    """
    coordinates: npt.NDArray
    stress: float
    n_iterations: int
    converged: bool
    distance_matrix: npt.NDArray
    stress_history: list
    metric: str
    n_restarts: int
    
    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Non-metric MDS Results')}\n"
            f"{'=' * 40}\n"
            f"{_('Distance metric: {0}').format(self.metric)}\n"
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
        self._last_result: Optional[NMDSResult] = None
        self._max_iterations = NMDS_MAX_ITERATIONS
        self._n_restarts = NMDS_RANDOM_RESTARTS
        self._tolerance = 1e-6
        self._logger.info("NMDSAnalyzer initialized")
    
    def analyze(
        self,
        distance_matrix: npt.NDArray,
        n_dimensions: int = 2,
        metric: str = 'euclidean',
        max_iterations: Optional[int] = None,
        n_restarts: Optional[int] = None,
        random_seed: Optional[int] = None
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
                raise MatrixDimensionError(
                    "Distance matrix must be square",
                    details={"shape": D.shape}
                )
            
            # Set parameters
            if max_iterations is None:
                max_iterations = self._max_iterations
            if n_restarts is None:
                n_restarts = self._n_restarts
            
            # Store best result across all restarts
            best_stress = float('inf')
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
                result = self._smacof(
                    D, X, max_iterations, restart
                )
                self._logger.debug(
                    f"NMDS restart {restart + 1}/{n_restarts} finished: "
                    f"stress={result['stress']:.6f}, iterations={result['n_iterations']}"
                )

                if result['stress'] < best_stress:
                    best_stress = result['stress']
                    best_coordinates = result['coordinates']
                    best_iterations = result['n_iterations']
                    best_history = result['stress_history']
            
            nmds_result = NMDSResult(
                coordinates=best_coordinates,
                stress=best_stress,
                n_iterations=best_iterations,
                converged=best_stress < 0.001,
                distance_matrix=D,
                stress_history=best_history,
                metric=metric,
                n_restarts=n_restarts
            )

            self._last_result = nmds_result
            self._logger.info(
                f"NMDS completed: final stress={best_stress:.6f}, "
                f"iterations={best_iterations}, converged={nmds_result.converged}"
            )
            if best_stress > 0.20:
                self._logger.warning(
                    f"NMDS poor fit: stress={best_stress:.4f} > 0.20, "
                    f"consider increasing n_restarts or n_dimensions"
                )
            if not nmds_result.converged:
                self._logger.error(
                    f"NMDS did not converge after {n_restarts} restarts, "
                    f"best stress={best_stress:.6f}"
                )
            return nmds_result
    
    def _smacof(
        self,
        D: npt.NDArray,
        X_init: npt.NDArray,
        max_iterations: int,
        restart_id: int
    ) -> Dict[str, Any]:
        """
        SMACOF algorithm for NMDS optimization.
        
        The SMACOF algorithm minimizes stress by iteratively
        updating the configuration using the Guttman transform.
        """
        n, k = X_init.shape
        X = X_init.copy()
        stress_history = []
        
        for iteration in range(max_iterations):
            # Compute distances in current configuration
            D_hat = self._compute_distances(X)

            # Compute stress
            numerator = np.sum((D - D_hat) ** 2)
            denominator = np.sum(D ** 2)
            stress = np.sqrt(numerator / denominator)
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
                    logger.debug(
                        f"SMACOF restart={restart_id} converged at iteration {iteration}: "
                        f"stress={stress:.6f}"
                    )
                    break
            
            # Compute stress weights (avoid division by zero)
            B = np.zeros_like(D)
            mask = D_hat > 0
            B[mask] = -D[mask] / D_hat[mask]
            np.fill_diagonal(B, 0)
            row_sums = np.sum(B, axis=1)
            np.fill_diagonal(B, -row_sums)
            
            # Guttman transform
            X_new = (B @ X) / n
            
            # Update configuration
            X = X_new
        
        return {
            'coordinates': X,
            'stress': stress_history[-1] if stress_history else float('inf'),
            'n_iterations': len(stress_history),
            'stress_history': stress_history
        }
    
    def _compute_distances(self, X: npt.NDArray) -> npt.NDArray:
        """
        Compute pairwise Euclidean distances.
        """
        n = X.shape[0]
        D = np.zeros((n, n))
        
        for i in range(n):
            diff = X - X[i]
            D[i] = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return D
    
    def get_shepard_data(
        self,
        result: Optional[NMDSResult] = None
    ) -> Dict[str, npt.NDArray]:
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
            'original': D_orig[indices],
            'ordination': D_ord[indices],
            'stress': result.stress
        }
    
    @property
    def last_result(self) -> Optional[NMDSResult]:
        """Get the last computed NMDS result."""
        with self._lock:
            return self._last_result
