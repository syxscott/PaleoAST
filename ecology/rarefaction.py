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
version: 1.0.1
"""

import math
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

    # Rarefaction is defined over INDIVIDUALS, so the input must be integer
    # counts. `int(np.sum(...))` used to truncate silently: a
    # relative-abundance vector like [0.1, 0.2, 0.3, 0.4] summed to exactly
    # 1.0, became N=1, and produced the nonsense curve "1 individual yields
    # 4 species".
    #
    # The test MUST be on the individual values, not on the sum: these
    # fractions sum to an integer, so a sum-integrality check passes them.
    non_integer = ~np.isclose(abundances, np.round(abundances), rtol=0.0, atol=1e-9)
    if np.any(non_integer):
        offenders = np.unique(np.asarray(abundances)[non_integer])[:5]
        raise ComputationError(
            "Rarefaction requires integer abundance counts (total individuals), "
            f"but got fractional values {list(offenders)}. Pass raw counts, "
            "not a relative-abundance or percentage vector."
        )

    N = int(round(float(np.sum(abundances))))  # Total individuals
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
    # Exact integer arithmetic: the previous log-gamma route lost ~1e-4 of
    # relative accuracy for N >= 60, where _log_factorial switches to
    # Stirling. At k = 1 the expectation is exactly 1.0 and it came out
    # 1.0001, which is a visible artefact on a reported species count.
    total_choose_n = _combinations(N, n)
    expected_S = 0.0

    for ni in abundances:
        if N - ni < n:
            # Species is definitely present (can't avoid it in the sample)
            expected_S += 1.0
        else:
            # P(species excluded) = C(N-ni, n) / C(N, n)
            prob_excluded = _combinations(N - ni, n) / total_choose_n
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
    """Compute sample-based rarefaction (interpolation by sample).

    Sample-based rarefaction fixes the total pool of samples at the full
    observed set (``N = n_samples``) and asks, for each sub-sample size
    ``k`` from 1 to ``N``, how many species we expect to see on average
    when drawing ``k`` samples uniformly without replacement:

        E[S(k)] = Σ_i [ 1 − C(N − n_i, k) / C(N, k) ]

    where ``n_i`` is the number of samples in which species ``i`` occurs.

    The previous implementation instead iterated over prefixes of the
    sample matrix (``n_total = i + 1`` for each ``i``) and recomputed a
    truncated curve per prefix — that is *species-accumulation*
    semantics, not rarefaction. Under accumulation the curve depends on
    sample ordering and the per-prefix ``n_total`` shrinks the
    combinatorial denominator, so the resulting "rarefaction" values
    were neither a valid rarefaction curve nor an unbiased accumulation
    curve. Compute a single canonical curve over the full dataset and
    return it as a one-element list (preserving the original return
    type).

    Parameters:
        occurrence_matrix: 2D binary matrix (n_samples, n_species)
        sample_sizes: Optional array of sub-sample sizes ``k``. Values
            larger than ``N`` are clipped to ``N``; if omitted, ``k``
            runs from 1 to ``N``.

    Returns:
        List with a single :class:`RarefactionResult` describing the
        rarefaction curve of the full dataset.
    """
    n_samples, _n_species = occurrence_matrix.shape
    if n_samples == 0:
        return []

    # Number of samples in which each species occurs (over the FULL
    # dataset). This is the canonical rarefaction denominator.
    occurrences = np.sum(occurrence_matrix, axis=0)
    occurrences = occurrences[occurrences > 0]

    if len(occurrences) == 0:
        return []

    n_total = n_samples  # fixed total
    if sample_sizes is None:
        sample_k = np.arange(1, n_total + 1)
    else:
        sample_k = np.asarray(sample_sizes)
        # Rarefaction is only defined for k ≤ N; clip rather than drop
        # so the caller's requested grid is preserved in shape.
        sample_k = np.clip(sample_k, 1, n_total)

    expected_species = np.zeros(len(sample_k))
    for j, k in enumerate(sample_k):
        k_int = int(k)
        # E[S(k)] = Σ_i [1 - C(N - n_i, k) / C(N, k)]
        denom = _combinations(n_total, k_int)
        if denom <= 0:
            expected_species[j] = 0.0
            continue
        terms = np.array([1.0 - _combinations(n_total - int(occ), k_int) / denom for occ in occurrences])
        expected_species[j] = float(np.sum(terms))

    return [
        RarefactionResult(
            sample_name="pooled",
            expected_taxa=expected_species,
            sample_sizes=sample_k.astype(int),
            method="sample",
        )
    ]


def _combinations(n: int, k: int) -> float:
    """Compute C(n, k), exactly.

    ``math.comb`` needs Python ints; sample-size grids arrive as numpy arrays
    whose dtype may be float, so coerce rather than trust the caller.
    """
    n = int(n)
    k = int(k)
    if k > n or k < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    # ``math.comb`` is exact arbitrary-precision integer arithmetic and is
    # instantaneous for the sample sizes a rarefaction curve ever sees (the
    # log-gamma route was only needed to avoid materialising the integer, and
    # its Stirling branch for n >= 60 cost ~1e-4 of relative accuracy on a
    # quantity whose exact value at k = 1 is 1.0). Fall back to the log route
    # only if the integers would actually overflow a float.
    try:
        return float(math.comb(n, k))
    except (OverflowError, ValueError):
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
