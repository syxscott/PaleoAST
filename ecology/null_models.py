# =============================================================================
# FILE: ecology/null_models.py
# =============================================================================
"""
Null Model Analysis for PaleoAST

Implements EcoSim-style null model analysis for testing co-occurrence patterns.

Stone, L. & Roberts, A. (1990). The checkerboard score: tests of
，刷到欠漏和物种组合随机性的统计方法.

Gotelli, N.J. & Entsminger, G.L. (2001). EcoSim: null models software for
ecology. Aquatic Sciences, 63(1), 5-11.

Mathematical Foundation:
==============================================================================

C-score (Checkerboard Score):
    C-score(i,j) = (r_i - 1) * (r_j - 1)

    where r_i = total occurrences of species i

    For a matrix, the observed C-score is the average
    over all species pairs.

Null Model Algorithms:
    1. Fixed-Fixed (swap): Row sums and column sums preserved
    2. Quick Swap: Random pairs of cells swapped

Significance:
    SES = (observed - mean_simulated) / std_simulated
    p-value = proportion of simulated >= observed

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Result Class
# =============================================================================


@dataclass
class NullModelResult:
    """
    Container for null model analysis results.

    Attributes:
        observed_score: Observed co-occurrence index value
        simulated_scores: Distribution of simulated index values
        mean_simulated: Mean of simulated values
        std_simulated: Standard deviation of simulated values
        standardized_effect_size: SES = (observed - mean) / std
        p_value: Proportion of simulated >= observed
        n_permutations: Number of permutations
        algorithm: Null model algorithm used
        metric: Co-occurrence metric used
        n_species: Number of species
        n_sites: Number of sites
    """

    observed_score: float
    simulated_scores: npt.NDArray[np.float64]
    mean_simulated: float
    std_simulated: float
    standardized_effect_size: float
    p_value: float
    n_permutations: int
    algorithm: str
    metric: str
    n_species: int
    n_sites: int

    def summary(self) -> str:
        """Generate summary text."""
        if self.p_value < 0.001:
            sig = "***"
        elif self.p_value < 0.01:
            sig = "**"
        elif self.p_value < 0.05:
            sig = "*"
        else:
            sig = ""

        if self.standardized_effect_size > 2:
            interpretation = _("Aggregation")
        elif self.standardized_effect_size < -2:
            interpretation = _("Segregation")
        else:
            interpretation = _("Random")

        return (
            f"{_('Null Model Analysis')}\n"
            f"{'=' * 50}\n"
            f"{_('Metric: {0}').format(self.metric.upper())}\n"
            f"{_('Algorithm: {0}').format(self.algorithm.upper())}\n"
            f"{_('Species: {0}, Sites: {1}').format(self.n_species, self.n_sites)}\n"
            f"{_('Permutations: {0}').format(self.n_permutations)}\n"
            f"\n"
            f"{_('Observed score: {0:.4f}').format(self.observed_score)}\n"
            f"{_('Mean simulated: {0:.4f}').format(self.mean_simulated)}\n"
            f"{_('Std simulated: {0:.4f}').format(self.std_simulated)}\n"
            f"{_('SES: {0:.4f}').format(self.standardized_effect_size)}\n"
            f"{_('P-value: {0:.4f} {1}').format(self.p_value, sig)}\n"
            f"\n"
            f"{_('Interpretation: {0}').format(interpretation)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "observed_score": self.observed_score,
            "simulated_scores": self.simulated_scores.tolist(),
            "mean_simulated": self.mean_simulated,
            "std_simulated": self.std_simulated,
            "standardized_effect_size": self.standardized_effect_size,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "n_species": self.n_species,
            "n_sites": self.n_sites,
            "summary": self.summary(),
        }


# =============================================================================
# Main Analyzer Class
# =============================================================================


class NullModelAnalyzer:
    """
    EcoSim-style null model analysis for co-occurrence patterns.

    Tests whether observed co-occurrence patterns differ significantly from
    random expectation using Monte Carlo simulations.

    Example:
        >>> analyzer = NullModelAnalyzer()
        >>> # Binary presence/absence matrix (species x sites)
        >>> presence = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]])
        >>> result = analyzer.analyze(presence, n_permutations=9999)
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize null model analyzer."""
        self._logger = logging.getLogger(f"{__name__}.NullModelAnalyzer")
        self._lock = threading.RLock()
        self._last_result: NullModelResult | None = None

    @property
    def last_result(self) -> NullModelResult | None:
        """Get last computed result."""
        with self._lock:
            return self._last_result

    def analyze(
        self,
        presence_matrix: npt.NDArray,
        metric: str = "c_score",
        n_permutations: int = 9999,
        algorithm: str = "swap",
        n_workers: int | None = None,
        random_seed: int | None = None,
    ) -> NullModelResult:
        """
        Perform null model analysis.

        Parameters:
            presence_matrix: Binary presence/absence matrix (n_species, n_sites)
            metric: Co-occurrence metric ("c_score", "checkerboard", "combo")
            n_permutations: Number of randomizations
            algorithm: Randomization algorithm ("swap", "shuffle")
            n_workers: Number of parallel workers (None = sequential)
            random_seed: Random seed for reproducibility

        Returns:
            NullModelResult with observed and simulated statistics

        Raises:
            ValidationError: If input data is invalid
        """
        with self._lock:
            self._logger.info(f"Null model analysis: metric={metric}, n_perm={n_permutations}")

            # Validate input
            presence_matrix = np.asarray(presence_matrix, dtype=np.int32)
            if presence_matrix.ndim != 2:
                raise ValidationError(_("Presence matrix must be 2D (species x sites)"))

            n_species, n_sites = presence_matrix.shape

            if n_species < 2:
                raise ValidationError(_("Need at least 2 species for null model analysis"))

            if n_sites < 2:
                raise ValidationError(_("Need at least 2 sites for null model analysis"))

            # Compute observed score
            if metric == "c_score":
                observed = self._compute_c_score(presence_matrix)
            elif metric == "checkerboard":
                observed = self._compute_checkerboard(presence_matrix)
            elif metric == "combo":
                observed = self._compute_combo_score(presence_matrix)
            else:
                raise ValidationError(
                    _("Unknown metric: {0}. Use 'c_score', 'checkerboard', or 'combo'").format(metric)
                )

            # Run permutations
            if random_seed is not None:
                np.random.seed(random_seed)

            if n_workers is not None and n_workers > 1:
                simulated = self._run_parallel(presence_matrix, n_permutations, algorithm, n_workers)
            else:
                simulated = self._run_sequential(presence_matrix, n_permutations, algorithm)

            # Compute statistics
            mean_sim = float(np.mean(simulated))
            std_sim = float(np.std(simulated))
            ses = (observed - mean_sim) / std_sim if std_sim > 0 else 0.0
            p_value = float(np.mean(simulated >= observed))

            result = NullModelResult(
                observed_score=observed,
                simulated_scores=simulated,
                mean_simulated=mean_sim,
                std_simulated=std_sim,
                standardized_effect_size=ses,
                p_value=p_value,
                n_permutations=n_permutations,
                algorithm=algorithm,
                metric=metric,
                n_species=n_species,
                n_sites=n_sites,
            )

            self._last_result = result
            self._logger.info(f"Null model: observed={observed:.4f}, SES={ses:.4f}, p={p_value:.4f}")
            return result

    def _run_sequential(
        self,
        matrix: npt.NDArray,
        n_permutations: int,
        algorithm: str,
    ) -> npt.NDArray[np.float64]:
        """Run permutations sequentially."""
        simulated = np.zeros(n_permutations)
        working = matrix.copy()

        for i in range(n_permutations):
            permuted = self._permute_matrix(working, algorithm)
            if len(permuted) > 0:
                simulated[i] = self._compute_c_score(permuted)

        return simulated

    def _run_parallel(
        self,
        matrix: npt.NDArray,
        n_permutations: int,
        algorithm: str,
        n_workers: int,
    ) -> npt.NDArray[np.float64]:
        """Run permutations in parallel using multiprocessing."""
        try:
            from multiprocessing import Pool

            chunk_size = max(1, n_permutations // n_workers)
            args_list = [
                (matrix.copy(), min(chunk_size, n_permutations - i * chunk_size), algorithm) for i in range(n_workers)
            ]

            with Pool(n_workers) as pool:
                results = pool.starmap(_worker_permute, args_list)

            simulated = np.concatenate(results)
            return simulated[:n_permutations]

        except ImportError:
            self._logger.warning("Multiprocessing not available, running sequentially")
            return self._run_sequential(matrix, n_permutations, algorithm)

    def _permute_matrix(
        self,
        matrix: npt.NDArray,
        algorithm: str,
    ) -> npt.NDArray:
        """Apply permutation algorithm to matrix."""
        if algorithm == "shuffle":
            return self._shuffle_matrix(matrix)
        elif algorithm == "swap":
            return self._swap_matrix(matrix)
        else:
            return self._shuffle_matrix(matrix)

    def _shuffle_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """Simple random shuffle of matrix elements."""
        result = matrix.copy()
        flat = result.flatten()
        np.random.shuffle(flat)
        return flat.reshape(matrix.shape)

    def _swap_matrix(self, matrix: npt.NDArray) -> npt.NDArray:
        """
        Swap algorithm - preserves row and column sums.

        Randomly select two rows and two columns.
        If cells form a 2x2 submatrix [[a,b],[c,d]], swap to [[c,d],[a,b]]
        """
        result = matrix.copy()
        n_species, n_sites = result.shape

        # Number of swaps = total cells * swap_factor
        n_swaps = int(n_species * n_sites * 0.1)

        for _ in range(n_swaps):
            # Select two random rows and columns
            rows = np.random.choice(n_species, 2, replace=False)
            cols = np.random.choice(n_sites, 2, replace=False)

            r1, r2 = rows
            c1, c2 = cols

            # Get 2x2 submatrix
            a = result[r1, c1]
            b = result[r1, c2]
            c = result[r2, c1]
            d = result[r2, c2]

            # Check if swapping would change the matrix
            # A valid swap exchanges 1s and 0s while preserving sums
            if (a == c and b == d) or (a == b and c == d):
                # Swapping doesn't change the pattern
                continue

            # Perform swap: [[a,b],[c,d]] -> [[c,d],[a,b]]
            result[r1, c1] = c
            result[r1, c2] = d
            result[r2, c1] = a
            result[r2, c2] = b

        return result

    def _compute_c_score(self, matrix: npt.NDArray) -> float:
        """
        Compute C-score for presence/absence matrix.

        C-score(i,j) = (r_i - 1) * (r_j - 1)

        Average C-score = mean of all pairwise species combinations
        """
        n_species, _n_sites = matrix.shape
        row_sums = np.sum(matrix, axis=1)

        c_scores = []
        for i in range(n_species):
            for j in range(i + 1, n_species):
                c_ij = float((row_sums[i] - 1) * (row_sums[j] - 1))
                c_scores.append(c_ij)

        return float(np.mean(c_scores)) if c_scores else 0.0

    def _compute_checkerboard(self, matrix: npt.NDArray) -> float:
        """
        Compute checkerboard count.

        Counts the number of perfect checkerboard patterns:
        [[1, 0], [0, 1]] or [[0, 1], [1, 0]]
        """
        n_species, n_sites = matrix.shape
        checkerboards = 0

        for i in range(n_species):
            for j in range(i + 1, n_species):
                for k in range(n_sites):
                    for l in range(k + 1, n_sites):
                        # Check for checkerboard pattern
                        if (matrix[i, k] == 1 and matrix[i, l] == 0 and matrix[j, k] == 0 and matrix[j, l] == 1) or (
                            matrix[i, k] == 0 and matrix[i, l] == 1 and matrix[j, k] == 1 and matrix[j, l] == 0
                        ):
                            checkerboards += 1

        return float(checkerboards)

    def _compute_combo_score(self, matrix: npt.NDArray) -> float:
        """
        Compute combined score = normalized C-score + checkerboard.

        Combines both metrics for a more robust test.
        """
        c_score = self._compute_c_score(matrix)
        checkerboard = self._compute_checkerboard(matrix)

        # Normalize checkerboard to similar scale as C-score
        n_species, n_sites = matrix.shape
        max_checkerboard = n_species * (n_species - 1) / 2 * n_sites * (n_sites - 1) / 2
        norm_checkerboard = checkerboard / max_checkerboard if max_checkerboard > 0 else 0

        return (c_score + norm_checkerboard) / 2


def _worker_permute(
    matrix: npt.NDArray,
    n_perms: int,
    algorithm: str,
) -> npt.NDArray:
    """
    Worker function for parallel permutation.

    This is a module-level function to allow pickling for multiprocessing.
    """
    results = np.zeros(n_perms)

    for i in range(n_perms):
        if algorithm == "shuffle":
            result = matrix.copy()
            flat = result.flatten()
            np.random.shuffle(flat)
            result = flat.reshape(matrix.shape)
        elif algorithm == "swap":
            result = _swap_matrix_worker(matrix)
        else:
            result = matrix.copy()
            flat = result.flatten()
            np.random.shuffle(flat)
            result = flat.reshape(matrix.shape)

        results[i] = _compute_c_score_worker(result)

    return results


def _swap_matrix_worker(matrix: npt.NDArray) -> npt.NDArray:
    """Swap algorithm in worker function."""
    result = matrix.copy()
    n_species, n_sites = result.shape
    n_swaps = int(n_species * n_sites * 0.1)

    for _ in range(n_swaps):
        rows = np.random.choice(n_species, 2, replace=False)
        cols = np.random.choice(n_sites, 2, replace=False)

        r1, r2 = rows
        c1, c2 = cols

        a = result[r1, c1]
        b = result[r1, c2]
        c = result[r2, c1]
        d = result[r2, c2]

        if (a == c and b == d) or (a == b and c == d):
            continue

        result[r1, c1] = c
        result[r1, c2] = d
        result[r2, c1] = a
        result[r2, c2] = b

    return result


def _compute_c_score_worker(matrix: npt.NDArray) -> float:
    """C-score computation in worker function."""
    n_species, _ = matrix.shape
    row_sums = np.sum(matrix, axis=1)

    c_scores = []
    for i in range(n_species):
        for j in range(i + 1, n_species):
            c_ij = float((row_sums[i] - 1) * (row_sums[j] - 1))
            c_scores.append(c_ij)

    return float(np.mean(c_scores)) if c_scores else 0.0
