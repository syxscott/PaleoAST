# =============================================================================
# FILE: ecology/rarefaction.py
# =============================================================================
"""
Rarefaction Analysis Module for PaleoAST

Rarefaction allows comparison of species richness across samples with
different sample sizes by extrapolating to a common sample size.

Mathematical Foundation:

Individual-based Rarefaction:
    E[S_n] = Σ_{i=1}^{S} [1 - C(N-n, n_i) / C(N, n)]

where:
    E[S_n] = expected number of species in sample of size n
    S = total number of species
    N = total number of individuals
    n_i = number of individuals of species i

Hurlbert's Formula:
    E[S_n] = Σ_{i=1}^{S} [1 - (N - n_i choose n) / (N choose n)]

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading

import numpy as np
import numpy.typing as npt

from models.diversity_result import RarefactionResult
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


def compute_rarefaction(
    abundances: npt.NDArray, sample_name: str = "Sample", max_n: int | None = None, n_points: int = 50
) -> RarefactionResult:
    """
    Compute individual-based rarefaction curve.

    Parameters:
        abundances: Array of taxon abundances
        sample_name: Name for the sample
        max_n: Maximum sample size to evaluate. If None, uses N/2
        n_points: Number of sample sizes to evaluate

    Returns:
        RarefactionResult: Rarefaction curve data
    """
    # Validate input
    abundances = validate_data_array(abundances, allow_nan=False, name="abundances")

    # validate_data_array reshapes 1D to 2D; flatten back for 1D operations
    abundances = abundances.flatten()

    abundances = abundances[abundances > 0]

    N = int(np.sum(abundances))  # Total individuals
    S = len(abundances)  # Observed richness
    logger.info(f"compute_rarefaction started: n_taxa={S}, total_individuals={N}, max_n={max_n}, n_points={n_points}")

    if N == 0:
        raise ComputationError("No individuals in sample")

    # Determine sample sizes to evaluate
    if max_n is None:
        max_n = N // 2

    max_n = min(max_n, N - 1)

    sample_sizes = np.linspace(1, max_n, n_points).astype(int)
    sample_sizes = np.unique(sample_sizes)  # Remove duplicates

    expected_taxa = np.zeros(len(sample_sizes))

    for i, n in enumerate(sample_sizes):
        expected_taxa[i] = _rarefaction_formula(abundances, N, n)

    return RarefactionResult(
        sample_name=sample_name, expected_taxa=expected_taxa, sample_sizes=sample_sizes, method="individual"
    )


def _rarefaction_formula(abundances: npt.NDArray, N: int, n: int) -> float:
    """
    Compute expected species richness at sample size n.

    Uses Hurlbert's formula:
    E[S_n] = Σ_{i=1}^{S} [1 - (N-n_i choose n) / (N choose n)]
    """
    S = len(abundances)

    if n >= N:
        return float(S)

    if n == 0:
        return 0.0

    # Precompute binomial coefficient
    # C(N, n) = N! / (n! * (N-n)!)
    log_N_choose_n = _log_factorial(N) - _log_factorial(n) - _log_factorial(N - n)

    expected_S = 0.0

    for ni in abundances:
        if N - ni < n:
            # Species is definitely present (can't avoid it in the sample)
            expected_S += 1.0
        else:
            # P(species excluded) = C(N-ni, n) / C(N, n)
            log_term = _log_factorial(N - ni) - _log_factorial(n) - _log_factorial(N - ni - n)
            prob_excluded = np.exp(log_term - log_N_choose_n)
            expected_S += 1.0 - prob_excluded

    return expected_S


def _log_factorial(n: int) -> float:
    """
    Compute log(n!) using Stirling's approximation for large n.
    """
    if n <= 1:
        return 0.0

    if n < 60:
        return np.sum(np.log(np.arange(1, n + 1)))

    # Use Stirling's approximation for large n
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)


def compute_sample_based_rarefaction(
    occurrence_matrix: npt.NDArray, sample_sizes: npt.NDArray | None = None
) -> list[RarefactionResult]:
    """
    Compute sample-based rarefaction (interpolation by sample).

    Parameters:
        occurrence_matrix: 2D binary matrix (n_samples, n_species)
        sample_sizes: Array of sample subset sizes

    Returns:
        List of RarefactionResult for each original sample
    """
    n_samples, _n_species = occurrence_matrix.shape

    results = []

    for i in range(n_samples):
        # Count occurrences
        occurrences = np.sum(occurrence_matrix[: i + 1], axis=0)
        occurrences = occurrences[occurrences > 0]

        if len(occurrences) == 0:
            continue

        # Compute rarefaction
        # For sample-based rarefaction, the unit is samples (rows), not species
        n_total_samples = i + 1  # number of samples accumulated so far
        if sample_sizes is None:
            max_k = n_total_samples
            sample_k = np.arange(1, max_k + 1)
        else:
            sample_k = sample_sizes[sample_sizes <= n_total_samples]

        expected_species = np.zeros(len(sample_k))

        for j, k in enumerate(sample_k):
            # Sample-based: expected species at k samples
            expected_species[j] = np.sum(
                1
                - np.array(
                    [
                        _combinations(n_total_samples - int(occ), k) / _combinations(n_total_samples, k)
                        for occ in occurrences
                    ]
                )
            )

        results.append(
            RarefactionResult(
                sample_name=f"Sample_{i + 1}",
                expected_taxa=expected_species,
                sample_sizes=sample_k.astype(int),
                method="sample",
            )
        )

    return results


def _combinations(n: int, k: int) -> float:
    """Compute C(n, k) using log space."""
    if k > n or k < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    return np.exp(_log_factorial(n) - _log_factorial(k) - _log_factorial(n - k))


class RarefactionAnalyzer:
    """
    Rarefaction curve analyzer.
    """

    def __init__(self) -> None:
        """Initialize the rarefaction analyzer."""
        self._lock = threading.RLock()
        self._last_result: RarefactionResult | None = None

    def analyze(
        self, abundances: npt.NDArray, sample_name: str = "Sample", max_n: int | None = None, n_points: int = 50
    ) -> RarefactionResult:
        """
        Perform individual-based rarefaction analysis.
        """
        with self._lock:
            result = compute_rarefaction(abundances, sample_name, max_n, n_points)
            self._last_result = result
            return result

    @property
    def last_result(self) -> RarefactionResult | None:
        """Get the last analysis result."""
        with self._lock:
            return self._last_result
