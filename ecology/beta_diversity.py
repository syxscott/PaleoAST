# =============================================================================
# FILE: ecology/beta_diversity.py
# =============================================================================
"""
Beta Diversity Decomposition and Coverage-based Rarefaction for PaleoAST

Implements two related analyses:

1. Beta Diversity Decomposition
   Decomposes overall beta diversity into turnover (species replacement)
   and nestedness (species gain/loss) components.

   Baselga, A. (2010). Partitioning the turnover and nestedness components
   of beta diversity. Global Ecology and Biogeography, 19(1), 134-143.

2. Coverage-based Rarefaction (iNEXT)
   Extrapolates species richness to unobserved diversity based on
   sample coverage estimation.

   Chao et al. (2014). Rarefaction and extrapolation of species diversity.
   Methods in Ecology and Evolution, 5(7), 677-686.

Mathematical Framework:
==============================================================================

Beta Diversity Decomposition:

For presence-absence data between sites i and j:
    a = species present in both sites
    b = species only in site i
    c = species only in site j

Jaccard: J = (b + c) / (a + b + c)
Sorensen: S = 2a / (2a + b + c)

Turnover component:
    J_tu = 2*min(b, c) / (a + 2*min(b, c))
    S_tu (beta_sim) = min(b, c) / (2a + min(b, c))

Nestedness component (Baselga 2012 / Baselga 2010):
    J_ne = beta_jac - beta_jtu = a*|b - c| / [(a + b + c)*(a + 2*min(b, c))]
    S_ne = beta_sor - beta_sim = 2*a*max(b, c) / [(2a + b + c)*(2a + min(b, c))]

Coverage-based Rarefaction:

Sample coverage (Chao & Jost 2012, Eq. 3):
    Chat_n = 1 - (f1/n) * ((n-1)*f1 / ((n-1)*f1 + 2*f2))

Interpolated coverage of a rarefied subsample of m individuals
(Chao & Jost 2012, Eq. 4; Chao et al. 2014, Table 1):
    Chat(m) = 1 - sum_i (x_i/N) * C(N-x_i, m-1) / C(N-1, m-1)

All diversity values reported by this module are Hill numbers
(^0D = S, ^1D = exp(H'), ^2D = 1/lambda), including the q=1 and q=2
asymptotic estimators.

Author: PaleoAST Development Team
version: 1.0.2
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln
from scipy.stats import norm

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class BetaDiversityResult:
    """
    Container for beta diversity decomposition results.

    Attributes:
        sample_names: Names of samples
        total_beta: Total beta diversity matrix
        turnover_component: Turnover (species replacement) component
        nestedness_component: Nestedness (species gain/loss) component
        decomposition_type: "jaccard" or "sorensen"
        n_samples: Number of samples
        pairwise_results: List of pairwise comparisons
    """

    sample_names: list[str]
    total_beta: npt.NDArray[np.float64]
    turnover_component: npt.NDArray[np.float64]
    nestedness_component: npt.NDArray[np.float64]
    decomposition_type: str
    n_samples: int
    pairwise_results: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary text."""
        mean_total = np.mean(self.total_beta[np.triu_indices(self.n_samples, k=1)])
        mean_turnover = np.mean(self.turnover_component[np.triu_indices(self.n_samples, k=1)])
        mean_nestedness = np.mean(self.nestedness_component[np.triu_indices(self.n_samples, k=1)])
        turnover_pct = (mean_turnover / mean_total * 100) if mean_total > 0 else 0
        nestedness_pct = (mean_nestedness / mean_total * 100) if mean_total > 0 else 0

        return (
            f"{_('Beta Diversity Decomposition')}\n"
            f"{'=' * 50}\n"
            f"{_('Method: {0}').format(self.decomposition_type.upper())}\n"
            f"{_('Number of samples: {0}').format(self.n_samples)}\n"
            f"{_('Mean total beta: {0:.4f}').format(mean_total)}\n"
            f"{_('Mean turnover: {0:.4f} ({1:.1f}%)').format(mean_turnover, turnover_pct)}\n"
            f"{_('Mean nestedness: {0:.4f} ({1:.1f}%)').format(mean_nestedness, nestedness_pct)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_names": self.sample_names,
            "total_beta": self.total_beta.tolist(),
            "turnover_component": self.turnover_component.tolist(),
            "nestedness_component": self.nestedness_component.tolist(),
            "decomposition_type": self.decomposition_type,
            "n_samples": self.n_samples,
            "pairwise_results": self.pairwise_results,
            "summary": self.summary(),
        }


@dataclass
class CoverageRarefactionResult:
    """
    Container for coverage-based rarefaction results.

    Attributes:
        sample_names: Names of samples
        coverage_levels: Coverage values (0-1)
        expected_richness: Expected species richness at each coverage
        confidence_lower: Lower confidence bounds
        confidence_upper: Upper confidence bounds
        asymptote_estimate: Asymptotic richness estimate
        n_iterations: Number of bootstrap iterations
    """

    sample_names: list[str]
    coverage_levels: npt.NDArray[np.float64]
    expected_richness: npt.NDArray[np.float64]
    confidence_lower: npt.NDArray[np.float64]
    confidence_upper: npt.NDArray[np.float64]
    asymptote_estimate: npt.NDArray[np.float64]
    sample_sizes: npt.NDArray[np.float64] | None = None
    method: str = "inext"

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"{_('Coverage-based Rarefaction (iNEXT-style)')}\n",
            f"{'=' * 50}\n",
            f"{_('Method: {0}').format(self.method.upper())}\n",
            f"{_('Number of samples: {0}').format(len(self.sample_names))}\n",
            "",
        ]
        final_richness = float(self.expected_richness[-1]) if len(self.expected_richness) else 0.0
        final_coverage = float(self.coverage_levels[-1]) if len(self.coverage_levels) else 0.0
        for i, name in enumerate(self.sample_names):
            sample_size = self.sample_sizes[i] if self.sample_sizes is not None and i < len(self.sample_sizes) else 0
            asymptote = self.asymptote_estimate[i] if i < len(self.asymptote_estimate) else 0
            lines.append(
                f"{name}: n={sample_size:.0f}, "
                f"asymptote={asymptote:.1f}, "
                f"aggregate S@{final_coverage:.1%}={final_richness:.1f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_names": self.sample_names,
            "coverage_levels": self.coverage_levels.tolist(),
            "expected_richness": self.expected_richness.tolist(),
            "confidence_lower": self.confidence_lower.tolist(),
            "confidence_upper": self.confidence_upper.tolist(),
            "asymptote_estimate": self.asymptote_estimate.tolist(),
            "sample_sizes": self.sample_sizes.tolist() if self.sample_sizes is not None else None,
            "method": self.method,
            "summary": self.summary(),
        }


# =============================================================================
# Beta Diversity Decomposition
# =============================================================================


class BetaDiversityAnalyzer:
    """
    Computes beta diversity decomposition into turnover and nestedness.

    Uses presence-absence data to decompose overall beta diversity.

    Example:
        >>> analyzer = BetaDiversityAnalyzer()
        >>> # Abundance matrix (samples x species)
        >>> abundance = np.array([[10, 5, 0], [8, 0, 3], [0, 2, 7]])
        >>> result = analyzer.decompose_beta_diversity(abundance, metric="jaccard")
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize beta diversity analyzer."""
        self._logger = logging.getLogger(f"{__name__}.BetaDiversityAnalyzer")
        self._lock = threading.RLock()
        self._last_result: BetaDiversityResult | None = None

    @property
    def last_result(self) -> BetaDiversityResult | None:
        """Get last computed result."""
        return self._last_result

    def decompose_beta_diversity(
        self,
        abundance_matrix: npt.NDArray,
        sample_names: list[str] | None = None,
        metric: str = "jaccard",
    ) -> BetaDiversityResult:
        """
        Decompose beta diversity into turnover and nestedness.

        Parameters:
            abundance_matrix: 2D array (n_samples, n_species) of abundances
            sample_names: Optional list of sample names
            metric: "jaccard" or "sorensen"

        Returns:
            BetaDiversityResult with decomposed matrices

        Raises:
            ValidationError: If input data is invalid
        """
        self._logger.info(f"Computing beta diversity decomposition: {metric}")

        # Validate input
        abundance_matrix = np.asarray(abundance_matrix, dtype=np.float64)
        if abundance_matrix.ndim != 2:
            raise ValidationError(_("Abundance matrix must be 2D"))

        n_samples, _n_species = abundance_matrix.shape

        if sample_names is None:
            sample_names = [f"Sample_{i + 1}" for i in range(n_samples)]
        elif len(sample_names) != n_samples:
            raise ValidationError(
                _("Number of sample names ({0}) must match matrix rows ({1})").format(len(sample_names), n_samples)
            )

        if metric not in ("jaccard", "sorensen"):
            raise ValidationError(_("Metric must be 'jaccard' or 'sorensen', got '{0}'").format(metric))

        # Convert to presence-absence (boolean for bitwise operations)
        presence = abundance_matrix > 0

        # Initialize matrices
        total_beta = np.zeros((n_samples, n_samples))
        turnover = np.zeros((n_samples, n_samples))
        nestedness = np.zeros((n_samples, n_samples))
        pairwise_results = []

        # Compute pairwise decomposition
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Presence vectors
                p_i = presence[i]
                p_j = presence[j]

                # Compute a, b, c
                a = np.sum(p_i & p_j)  # Shared species
                b = np.sum(p_i & ~p_j)  # Only in i
                c = np.sum(~p_i & p_j)  # Only in j

                total = a + b + c
                if total == 0:
                    # No species in either sample
                    turn_val = 0.0
                    nest_val = 0.0
                    total_val = 0.0
                elif metric == "jaccard":
                    # Baselga (2012) Jaccard-based partition:
                    #   βjac = (b + c) / (a + b + c)
                    #   βjtu = 2·min(b,c) / (a + 2·min(b,c))           (turnover)
                    #   βjne = βjac - βjtu
                    #         = a·|b - c| / [(a + b + c)·(a + 2·min(b,c))]
                    total_val = (b + c) / total
                    min_bc = min(b, c)
                    denom_turn = a + 2 * min_bc
                    turn_val = (2 * min_bc) / denom_turn if denom_turn > 0 else 0.0
                    nest_val = (a * abs(b - c)) / (total * denom_turn) if denom_turn > 0 else 0.0
                else:  # sorensen
                    # Baselga (2010) Sørensen-based partition:
                    #   βsor = (b + c) / (2a + b + c)
                    #   βsim = min(b,c) / (2a + min(b,c))              (turnover)
                    #   βsne = βsor - βsim
                    #         = 2a·max(b,c) / [(2a + b + c)·(2a + min(b,c))]
                    denom = 2 * a + b + c
                    total_val = (b + c) / denom if denom > 0 else 0.0
                    min_bc = min(b, c)
                    denom_turn = 2 * a + min_bc
                    turn_val = min_bc / denom_turn if denom_turn > 0 else 0.0
                    nest_val = (2 * a * max(b, c)) / (denom * denom_turn) if (denom > 0 and denom_turn > 0) else 0.0

                total_beta[i, j] = total_beta[j, i] = total_val
                turnover[i, j] = turnover[j, i] = turn_val
                nestedness[i, j] = nestedness[j, i] = nest_val

                pairwise_results.append(
                    {
                        "sample_i": sample_names[i],
                        "sample_j": sample_names[j],
                        "shared_species": int(a),
                        "only_i": int(b),
                        "only_j": int(c),
                        "total_beta": float(total_val),
                        "turnover": float(turn_val),
                        "nestedness": float(nest_val),
                    }
                )

        result = BetaDiversityResult(
            sample_names=sample_names,
            total_beta=total_beta,
            turnover_component=turnover,
            nestedness_component=nestedness,
            decomposition_type=metric,
            n_samples=n_samples,
            pairwise_results=pairwise_results,
        )

        self._last_result = result
        self._logger.info(f"Beta diversity: {n_samples} samples, mean beta={np.mean(total_beta):.4f}")
        return result


# =============================================================================
# Coverage-based Rarefaction
# =============================================================================


def _integerize_abundances(values: npt.NDArray, context: str = "") -> npt.NDArray:
    """
    Round an abundance array to whole counts, preserving its shape.

    Abundances are expected to be whole numbers, but they can arrive as floats
    from parsed files, from arithmetic on matrices or from a multinomial
    bootstrap.  Values are matched to the nearest integer with ``np.isclose``
    (a plain ``astype(int)`` would truncate 4.9999999 to 4 and silently change
    the frequency spectrum); any deviation beyond the tolerance is reported
    through the module logger.  Non-finite and non-positive entries are
    zeroed.

    Parameters
    ----------
    values : array-like
        Species abundances of any shape.
    context : str, optional
        Extra text for the warning message.

    Returns
    -------
    np.ndarray
        Array of the same shape holding integer-valued counts.
    """
    arr = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(arr) & (arr > 0)
    if arr.size and not np.allclose(arr[keep], np.rint(arr[keep]), rtol=0.0, atol=1e-6):
        deviations = np.abs(arr[keep] - np.rint(arr[keep]))
        logger.warning(
            "Non-integer species abundances%s were rounded to the nearest "
            "count for the frequency-spectrum estimators (max deviation %.6g); "
            "abundance data should be whole numbers.",
            f" in {context}" if context else "",
            float(deviations.max()) if deviations.size else 0.0,
        )
    return np.where(keep, np.rint(np.where(np.isfinite(arr), arr, 0.0)), 0.0)


def _as_positive_counts(species_counts: npt.NDArray, context: str = "") -> npt.NDArray:
    """
    Flatten an abundance vector to strictly positive whole counts.

    Parameters
    ----------
    species_counts : array-like
        Species abundances (zeros and negatives are discarded).
    context : str, optional
        Extra text for the rounding warning.

    Returns
    -------
    np.ndarray
        Strictly positive integer-valued abundances as float64.
    """
    counts = _integerize_abundances(species_counts, context=context).reshape(-1)
    return counts[counts > 0]


def _frequency_classes(counts: npt.NDArray) -> tuple[float, float]:
    """
    Number of singleton (``f1``) and doubleton (``f2``) species.

    ``np.isclose`` is used instead of ``==`` because the counts may carry
    floating-point noise (e.g. ``0.99999999``), which ``== 1`` would drop and
    thereby bias the coverage estimate low.

    Parameters
    ----------
    counts : np.ndarray
        Positive integer-valued abundances (see :func:`_as_positive_counts`).

    Returns
    -------
    tuple of float
        ``(f1, f2)``.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts.size == 0:
        return 0.0, 0.0
    f1 = float(np.count_nonzero(np.isclose(counts, 1.0)))
    f2 = float(np.count_nonzero(np.isclose(counts, 2.0)))
    return f1, f2


def _sample_coverage(counts: npt.NDArray, n: float, f1: float, f2: float) -> float:
    """
    Sample coverage of an individual-based sample (Chao & Jost 2012, Eq. 3).

    .. math:: \\hat C_n = 1 - \\frac{f_1}{n}\\,\\frac{(n-1)f_1}{(n-1)f_1 + 2f_2}

    With ``f2 = 0`` the correction factor tends to 1 and the estimator reduces
    to the Good-Turing coverage ``1 - f1/n``; with ``f1 = 0`` the sample is
    declared complete (coverage 1).

    Parameters
    ----------
    counts : np.ndarray
        Positive abundances (only used for the empty-sample shortcut).
    n : float
        Total number of individuals.
    f1, f2 : float
        Numbers of singleton and doubleton species.

    Returns
    -------
    float
        Coverage estimate in ``[0, 1]``.
    """
    n = float(n)
    if n <= 0 or counts is None or len(counts) == 0:
        return 0.0
    if f1 <= 0:
        return 1.0
    if n < 2:
        # A single individual is necessarily a singleton: coverage is zero.
        return 0.0
    denom = (n - 1.0) * f1 + 2.0 * f2
    gamma = ((n - 1.0) * f1) / denom if denom > 0 else 0.0
    return float(min(1.0, max(0.0, 1.0 - (f1 / n) * gamma)))


def _chao1_unseen(f1: float, f2: float) -> float:
    """
    Chao (1984) lower-bound estimate of the number of unseen species.

    ``f1^2 / (2 f2)`` when doubletons exist, otherwise the bias-corrected
    ``f1 (f1 - 1) / 2`` (which also handles the ``f2 = 0`` case without a
    division by zero).

    Parameters
    ----------
    f1, f2 : float
        Numbers of singleton and doubleton species.

    Returns
    -------
    float
        Estimated number of unseen species (``>= 0``).
    """
    if f1 <= 0:
        return 0.0
    if f2 > 0:
        return float((f1 * f1) / (2.0 * f2))
    return float(max(0.0, (f1 * (f1 - 1.0)) / 2.0))


def _chao_shen_proportions(
    counts: npt.NDArray, n: float, f1: float
) -> tuple[npt.NDArray, float]:
    """
    Bias-corrected relative abundances of the observed species.

    Chao & Shen (1992) completeness correction, used by Chao & Jost (2012) and
    iNEXT to turn the observed sample into the "complete" (asymptotic) sample:

    .. math:: \\hat p_i = \\frac{i}{n}\\Bigl[1 - \\frac{f_1}{n}
              \\frac{(i+1)f_{i+1}}{i f_i + \\delta_{i1}f_1}\\Bigr]

    The correction deflates rare species, so ``sum(p_hat)`` is strictly smaller
    than 1 whenever singletons are present; the residual probability mass is
    carried by the unseen species (see :func:`_chao1_unseen`).

    Parameters
    ----------
    counts : np.ndarray
        Positive integer-valued abundances.
    n : float
        Total number of individuals.
    f1 : float
        Number of singleton species.

    Returns
    -------
    tuple
        ``(p_hat, residual)`` with ``p_hat`` the corrected proportions of the
        observed species and ``residual = 1 - sum(p_hat)``.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = float(n)
    if counts.size == 0 or n <= 0:
        return np.zeros(counts.size), 0.0
    x = np.rint(counts).astype(int)
    freq = np.bincount(x, minlength=int(x.max()) + 2).astype(np.float64)
    p_hat = np.zeros(counts.size, dtype=np.float64)
    if f1 > 0:
        for k, i in enumerate(x):
            denom = i * freq[i] + (f1 if i == 1 else 0.0)
            correction = 0.0
            if denom > 0:
                correction = (f1 / n) * ((i + 1) * freq[i + 1]) / denom
            p_hat[k] = (i / n) * max(0.0, 1.0 - correction)
    else:
        p_hat = x / n
    residual = float(max(0.0, 1.0 - float(np.sum(p_hat))))
    return p_hat, residual


def _hill_number(counts: npt.NDArray, n: float, q: int) -> float:
    """
    Observed Hill number of order ``q`` for a vector of abundances.

    ``q=0`` returns species richness, ``q=1`` ``exp(H')`` and ``q=2``
    ``1/lambda`` (inverse Simpson concentration).

    Parameters
    ----------
    counts : np.ndarray
        Positive abundances.
    n : float
        Total number of individuals.
    q : int
        Hill order (0, 1 or 2).

    Returns
    -------
    float
        Hill number (``>= 0``).
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    counts = counts[counts > 0]
    if counts.size == 0 or n <= 0:
        return 0.0
    if q == 0:
        return float(counts.size)
    p = counts / float(n)
    if q == 1:
        return float(math.exp(-float(np.sum(p * np.log(p)))))
    lam = float(np.sum(p * p))
    return float(1.0 / lam) if lam > 0 else 0.0


def _hill_from_index(value: float, q: int) -> float:
    """
    Convert an entropy / concentration index into the Hill number of order q.

    ``q=0`` passes the richness through, ``q=1`` exponentiates a Shannon
    entropy and ``q=2`` inverts a Simpson concentration ``1 - lambda``
    (the form returned by :func:`_rarefaction_simpson`).

    Parameters
    ----------
    value : float
        Index value on the natural scale used by the rarefaction helpers.
    q : int
        Hill order.

    Returns
    -------
    float
        Hill number of order ``q``.
    """
    if q == 0:
        return float(value)
    if q == 1:
        return float(math.exp(max(-700.0, min(700.0, float(value)))))
    # q == 2: value is the Gini-Simpson index 1 - lambda
    lam = 1.0 - float(value)
    if lam <= 1e-12:
        lam = 1e-12
    return float(1.0 / lam)


def _hill_asymptote(
    counts: npt.NDArray, n: float, q: int, f1: float, f2: float, s_obs: int
) -> float:
    """
    Asymptotic (complete-sample) Hill number of order ``q``.

    - ``q=0``: Chao1 richness, ``S_obs + f1^2/(2 f2)``.
    - ``q=1``: ``exp(H_inf)`` with the Chao-Shen corrected proportions and the
      residual mass spread over the Chao1 number of unseen species; reduces to
      ``exp(H')`` when ``f1 = 0``.
    - ``q=2``: ``1 / lambda_inf`` on the same corrected proportions; reduces to
      the observed inverse Simpson when ``f1 = 0``.

    Parameters
    ----------
    counts : np.ndarray
        Positive integer-valued abundances.
    n : float
        Total number of individuals.
    q : int
        Hill order (0, 1 or 2).
    f1, f2 : float
        Numbers of singleton and doubleton species.
    s_obs : int
        Observed richness.

    Returns
    -------
    float
        Asymptotic Hill number, guaranteed ``>=`` the observed Hill number.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts.size == 0 or n <= 0:
        return 0.0
    if q == 0:
        return float(s_obs) + _chao1_unseen(f1, f2)

    p_hat, residual = _chao_shen_proportions(counts, n, f1)
    n_unseen = _chao1_unseen(f1, f2)
    observed = _hill_number(counts, n, q)
    if n_unseen <= 0 or residual <= 0:
        return observed

    p0 = residual / n_unseen
    if q == 1:
        positive = p_hat[p_hat > 0]
        entropy = -float(np.sum(positive * np.log(positive))) if positive.size else 0.0
        entropy -= n_unseen * p0 * math.log(p0)
        asymptote = float(math.exp(entropy))
    else:  # q == 2
        lam = float(np.sum(p_hat * p_hat)) + n_unseen * p0 * p0
        asymptote = float(1.0 / lam) if lam > 0 else observed
    # The asymptotic estimate may not be smaller than the observed diversity.
    return max(asymptote, observed)


def _sample_size_for_coverage(counts: npt.NDArray, n: int, target: float) -> int:
    """
    Invert the interpolated coverage curve (Chao & Jost 2012, Eq. 4).

    Returns the subsample size ``m`` whose interpolated coverage ``Chat(m)`` is
    closest to ``target``.  ``Chat(m)`` is monotone increasing in ``m``, so a
    plain integer bisection finds the crossing point; the previous linear
    heuristic ``m = n * c / Chat(n)`` was not the coverage of the rarefied
    sample and mis-aligned the curve (most visibly for uneven samples).

    Parameters
    ----------
    counts : np.ndarray
        Positive integer-valued abundances.
    n : int
        Total number of individuals in the sample.
    target : float
        Target coverage in ``[0, 1]``.

    Returns
    -------
    int
        Subsample size in ``[1, n]``.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = int(n)
    if n <= 1 or counts.size == 0:
        return max(1, min(n, 1))
    top = _rarefied_coverage(counts, n, n)
    if target >= top:
        return n
    if target <= 0.0:
        return 1

    lo, hi = 1, n  # Chat(lo) <= target < Chat(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _rarefied_coverage(counts, mid, n) <= target:
            lo = mid
        else:
            hi = mid
    # Return whichever endpoint is closest to the requested coverage.
    c_lo = _rarefied_coverage(counts, lo, n)
    c_hi = _rarefied_coverage(counts, hi, n)
    return lo if abs(target - c_lo) <= abs(c_hi - target) else hi


def _coverage_curve_point(
    counts: npt.NDArray,
    n: int,
    q: int,
    c_level: float,
    coverage_i: float,
    asymptote: float,
) -> float:
    """
    Hill number of order ``q`` at a target coverage (CRÉ curve).

    Covers both branches of Chao & Jost (2012) in one place so the rarefaction
    (interpolation), the extrapolation and the asymptote share a single
    transform:

    - ``c_level < Chat``: classic individual-based rarefaction of the observed
      sample, evaluated at the subsample size ``m`` solving ``Chat(m) = c_level``.
    - ``c_level >= Chat``: linear in the *standardised coverage*
      ``(c - Chat)/(1 - Chat)`` between the observed Hill number and the
      asymptotic Hill number, i.e. the asymptote is reached at coverage 1.

    Parameters
    ----------
    counts : np.ndarray
        Positive integer-valued abundances.
    n : int
        Total number of individuals.
    q : int
        Hill order (0, 1 or 2).
    c_level : float
        Target coverage.
    coverage_i : float
        Sample coverage ``Chat`` of the full sample.
    asymptote : float
        Asymptotic Hill number from :func:`_hill_asymptote`.

    Returns
    -------
    float
        Diversity value of order ``q`` at ``c_level``.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    counts = counts[counts > 0]
    if counts.size == 0 or n <= 0:
        return 0.0
    observed = _hill_number(counts, n, q)
    if coverage_i <= 0:
        return observed
    if c_level < coverage_i:
        m = _sample_size_for_coverage(counts, n, c_level)
        m = int(min(max(1, m), n))
        if q == 0:
            return float(_rarefaction_species(counts, m))
        if q == 1:
            return _hill_from_index(_rarefaction_shannon(counts, m, n), 1)
        return _hill_from_index(_rarefaction_simpson(counts, m, n), 2)
    if coverage_i >= 1.0 or c_level >= 1.0:
        return float(asymptote)
    frac = (c_level - coverage_i) / (1.0 - coverage_i)
    return float(observed + (asymptote - observed) * frac)


def coverage_rarefaction_hill(
    abundance_matrix: npt.NDArray,
    sample_names: list[str] | None = None,
    q: int = 0,
    n_points: int = 50,
    confidence_level: float = 0.95,
    n_bootstrap: int = 100,
    seed: int | None = None,
) -> CoverageRarefactionResult:
    """
    Compute coverage-based rarefaction and extrapolation using Hill numbers.

    Implements the coverage-based rarefaction and extrapolation (CRÉ)
    framework of Chao & Jost (2012) with asymptotic estimators from
    Chao et al. (2014).

    Parameters
    ----------
    abundance_matrix : array-like, shape (n_samples, n_species)
        Matrix of species abundances where rows are samples and
        columns are species.
    sample_names : list of str, optional
        Names for each sample.
    q : int, default=0
        Order of the Hill number:
        - q=0: species richness (S)
        - q=1: Shannon diversity (^1D = exp(H'))
        - q=2: inverse Simpson diversity (^2D = 1/lambda)
    n_points : int, default=50
        Number of coverage levels to evaluate.
    confidence_level : float, default=0.95
        Confidence level for bootstrap intervals.
    n_bootstrap : int, default=100
        Number of bootstrap replicates.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    CoverageRarefactionResult
        Results containing coverage levels, expected richness,
        confidence intervals, and asymptote estimates.

    Notes
    -----
    **Coverage estimator** (Chao & Jost 2012, Eq. 3):
        Chat_n = 1 - (f_1/n) · ((n-1)·f_1 / ((n-1)·f_1 + 2·f_2))

    **Asymptotic diversity estimators** (Chao et al. 2014), all on the
    Hill-number scale so that the rarefaction curve, the extrapolation and
    the asymptote are directly comparable:
        - q=0: ^0D_inf = S_obs + f_1²/(2·f_2)          (Chao1 richness)
        - q=1: ^1D_inf = exp(H_inf) on the Chao-Shen bias-corrected
          proportions, with the residual mass carried by the unseen
          species; degenerates to exp(H') when f_1 = 0.
        - q=2: ^2D_inf = 1/lambda_inf on the same corrected proportions;
          degenerates to the observed 1/lambda when f_1 = 0.

    **Rarefaction/extrapolation** (Chao & Jost 2012):
        For coverage C < Chat: interpolate with the classic rarefaction
        formula evaluated at the sample size m solving Chat(m) = C
        (inversion of Eq. 4, see :func:`_sample_size_for_coverage`).
        For coverage C > Chat: scale linearly in (C - Chat)/(1 - Chat)
        from the observed Hill number toward the asymptotic Hill number.

    **Bootstrap CI**: Uses local RNG (np.random.default_rng(seed))
    to avoid polluting global random state.

    References
    ----------
    Chao, A., & Jost, L. (2012). Coverage-based rarefaction and
        extrapolation: sampling and projecting species diversity.
        Methods in Ecology and Evolution, 3(5), 873-882.

    Chao, A., Hsieh, T. C., Chazdon, R. L., Colwell, R. K., &
        Gotelli, N. J. (2014). Rarefaction and extrapolation with
        Hill numbers: a framework for sampling and estimation.
        Methods in Ecology and Evolution, 5(7), 677-686.
    """
    # Validate input
    abundance_matrix = np.asarray(abundance_matrix, dtype=np.float64)
    if abundance_matrix.ndim != 2:
        raise ValidationError(_("Abundance matrix must be 2D"))
    if abundance_matrix.shape[0] == 0 or abundance_matrix.shape[1] == 0:
        raise ValidationError(_("Abundance matrix cannot be empty"))

    # The frequency-spectrum estimators need whole-number abundances: round
    # once for the whole matrix (single warning) instead of truncating row by
    # row, which also drops stray negative / non-finite cells.
    abundance_matrix = _integerize_abundances(abundance_matrix, context="abundance_matrix")

    n_samples, n_species = abundance_matrix.shape

    if sample_names is None:
        sample_names = [f"Sample_{i + 1}" for i in range(n_samples)]
    elif len(sample_names) != n_samples:
        raise ValidationError(
            _("Number of sample names ({0}) must match matrix rows ({1})").format(
                len(sample_names), n_samples
            )
        )

    if q not in (0, 1, 2):
        raise ValidationError(_("q must be 0, 1, or 2, got '{0}'").format(q))

    # Coverage levels from 0.1 to 0.99
    coverage_levels = np.linspace(0.1, 0.99, n_points)

    # Bootstrap CI using LOCAL RNG (avoids global seed pollution)
    rng = np.random.default_rng(seed)
    # Two-sided normal quantile for the requested confidence level
    z = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    # Storage for per-sample results
    sample_coverage = np.zeros(n_samples)
    asymptote = np.zeros(n_samples)
    richness_curve = np.zeros((n_samples, n_points))
    ci_lower_curve = np.zeros((n_samples, n_points))
    ci_upper_curve = np.zeros((n_samples, n_points))

    for i in range(n_samples):
        row = abundance_matrix[i]
        species_counts = row[row > 0]
        total_n = int(np.sum(species_counts))
        s_obs = int(species_counts.size)

        if total_n == 0 or s_obs == 0:
            sample_coverage[i] = 0.0
            asymptote[i] = 0.0
            continue

        # Count singletons and doubletons (np.isclose based: floats from
        # parsed files must not silently drop out of the frequency spectrum)
        f1, f2 = _frequency_classes(species_counts)

        # ---- Coverage estimator (Chao & Jost 2012, Eq. 3) ----
        coverage_i = _sample_coverage(species_counts, total_n, f1, f2)
        sample_coverage[i] = coverage_i

        # ---- Asymptotic Hill number (Chao et al. 2014) ----
        asymptote[i] = _hill_asymptote(species_counts, total_n, q, f1, f2, s_obs)

        # ---- Rarefaction/extrapolation at each coverage level ----
        # True non-parametric bootstrap: resample from multinomial (Chao & Jost 2012)
        bootstrap_curves = []
        for _bootstrap_idx in range(n_bootstrap):
            # Multinomial resampling: sample N individuals from p_hat
            boot_counts = _multinomial_resample(species_counts, total_n, rng)
            boot_counts = _as_positive_counts(boot_counts)
            boot_N = int(np.sum(boot_counts))

            if boot_N == 0 or boot_counts.size == 0:
                # Empty resample: fall back to the observed sample
                bootstrap_curves.append(np.zeros(n_points))
                continue

            boot_f1, boot_f2 = _frequency_classes(boot_counts)
            boot_coverage = _sample_coverage(boot_counts, boot_N, boot_f1, boot_f2)
            boot_asymptote = _hill_asymptote(
                boot_counts, boot_N, q, boot_f1, boot_f2, int(boot_counts.size)
            )
            boot_curve = np.array(
                [
                    _coverage_curve_point(
                        boot_counts, boot_N, q, c_level, boot_coverage, boot_asymptote
                    )
                    for c_level in coverage_levels
                ]
            )
            bootstrap_curves.append(boot_curve)

        bootstrap_curves = np.array(bootstrap_curves) if bootstrap_curves else np.zeros((0, n_points))

        # ---- Point estimate: computed from the OBSERVED sample ----
        # Chao et al. (2014) compute the rarefaction/extrapolation point
        # estimate from the observed data; bootstrap replicates are used
        # only to estimate the standard error / confidence bounds.
        observed_curve = np.array(
            [
                _coverage_curve_point(
                    species_counts, total_n, q, c_level, coverage_i, asymptote[i]
                )
                for c_level in coverage_levels
            ]
        )
        richness_curve[i] = observed_curve

        # ---- Bootstrap replicates feed only the s.e. / CI bounds ----
        if bootstrap_curves.shape[0] > 1:
            boot_se = np.std(bootstrap_curves, axis=0, ddof=1)
            ci_lower_curve[i] = np.maximum(observed_curve - z * boot_se, 0.0)
            ci_upper_curve[i] = observed_curve + z * boot_se
        else:
            ci_lower_curve[i] = observed_curve
            ci_upper_curve[i] = observed_curve

    # Aggregate across samples
    result = CoverageRarefactionResult(
        sample_names=sample_names,
        coverage_levels=coverage_levels,
        expected_richness=np.mean(richness_curve, axis=0),
        confidence_lower=np.mean(ci_lower_curve, axis=0),
        confidence_upper=np.mean(ci_upper_curve, axis=0),
        asymptote_estimate=asymptote,
        sample_sizes=abundance_matrix.sum(axis=1),
        method=f"coverage_rarefaction_hill_q{q}",
    )
    return result


def _rarefied_coverage(species_counts: npt.NDArray, n: int, N: int) -> float:
    """
    Interpolated (rarefied) sample coverage at sample size n.

    Chao & Jost (2012), Eq. 4 (see also Chao et al. 2014, Table 1,
    last row): the coverage of a rarefied subsample of ``n`` individuals
    drawn without replacement from the observed sample of ``N`` individuals,

        C(n) = 1 - sum_i (x_i / N) * C(N - x_i, n - 1) / C(N - 1, n - 1)

    which is the Good-Turing deficiency recomputed with the *expected* number
    of singletons of the subsample: a species with abundance x_i is a
    singleton in the subsample with probability
    ``x_i * C(N - x_i, n - 1) / C(N, n)``, and dividing that expectation by
    ``n`` gives the summand above because ``n * C(N, n) = N * C(N-1, n-1)``.
    The previous implementation used ``C(N - x_i, n) / C(N - 1, n)``, i.e.
    ``n`` in place of ``n - 1``, which is not the coverage of any subsample
    (it does not reduce to ``1 - f1/N`` at ``n = N`` and is not monotone in
    ``n``).

    Singletons contribute exactly 1/N each at n = N, and rarer species are
    increasingly likely to be missed, so C(n) < C(N) for n < N and the curve
    is strictly increasing in n.

    Parameters
    ----------
    species_counts : array-like
        Species abundances in the full sample (positive counts).
    n : int
        Rarefied sample size (n <= N).
    N : int
        Total sample size of the full sample.

    Returns
    -------
    float
        Estimated coverage C(n) of the rarefied sample, in ``[0, 1]``.
    """
    counts = np.asarray(species_counts, dtype=np.float64).reshape(-1)
    counts = counts[counts > 0]
    N = float(N)
    n = int(n)
    if n <= 0 or counts.size == 0 or N <= 0:
        return 0.0
    if n >= N:
        # At (or beyond) the full sample size the Good-Turing estimate applies,
        # which is exactly the n = N limit of the formula below.
        f1, _ = _frequency_classes(counts)
        return float(min(1.0, max(0.0, 1.0 - f1 / N)))

    k = n - 1  # Eq. 4 uses n - 1 in both binomial coefficients
    # log C(N-1, k): denominator binomial coefficient
    log_ref = gammaln(N) - gammaln(k + 1) - gammaln(N - k)
    # A species with x_i > N - k cannot be a singleton of the subsample
    # (C(N - x_i, k) = 0), so it drops out of the deficiency sum.
    x = counts[counts <= N - k]
    if x.size == 0:
        return 1.0
    # log C(N - x_i, k), vectorised over species
    log_num = gammaln(N - x + 1) - gammaln(k + 1) - gammaln(N - x - k + 1)
    deficiency = float(np.sum((x / N) * np.exp(log_num - log_ref)))
    return float(min(1.0, max(0.0, 1.0 - deficiency)))


def _rarefaction_species(species_counts: npt.NDArray, n: int) -> float:
    """
    Rarefied species richness (q=0) at sample size n.

    Uses Hurlbert's formula:
    E[S(n)] = Σ [1 - C(N - k_i, n) / C(N, n)]

    Parameters
    ----------
    species_counts : array-like
        Species abundances
    n : int
        Subsample size

    Returns
    -------
    float
        Expected number of species
    """
    if n <= 0:
        return 0.0
    N = int(np.sum(species_counts))
    if n >= N:
        return float(len(species_counts))

    expected_s = 0.0
    for k in species_counts:
        # P(species absent in sample of size n) = C(N-k, n) / C(N, n)
        log_prob_absent = (
            _lgamma(N - k + 1)
            - _lgamma(n + 1)
            - _lgamma(N - k - n + 1)
            - (_lgamma(N + 1) - _lgamma(n + 1) - _lgamma(N - n + 1))
        )
        prob_present = 1.0 - math.exp(log_prob_absent)
        expected_s += prob_present
    return expected_s


def _rarefaction_shannon(species_counts: npt.NDArray, n: int, N: int) -> float:
    """
    Rarefied Shannon entropy (q=1) at sample size n.

    Based on Chao & Jost 2012 Eq. (4):
    H_n = H_N * (1 - (N-n)/N * (1 - C_n/C_N))

    where H_N is observed Shannon entropy, C_n is the interpolated
    (rarefied) coverage at size n (Chao & Jost 2012, Eq. 4) and C_N is
    the observed coverage estimated by Good-Turing as 1 - f1/N.

    Parameters
    ----------
    species_counts : array-like
        Species abundances
    n : int
        Subsample size
    N : int
        Total sample size

    Returns
    -------
    float
        Expected Shannon entropy
    """
    if n <= 0:
        return 0.0
    if n >= N:
        p = species_counts / N
        return float(-np.sum(p * np.log(p)))

    p = species_counts / N
    H_N = -np.sum(p * np.log(p))

    # Coverage at full sample (Good-Turing)
    f1, _ = _frequency_classes(species_counts)
    C_N = 1.0 - (f1 / N) if N > 0 else 0.0

    # Interpolated (rarefied) coverage at n (Chao & Jost 2012, Eq. 4)
    C_n = _rarefied_coverage(species_counts, n, N)

    if C_n > 0 and C_N > 0:
        H_n = H_N * (1 - ((N - n) / N) * (1 - C_n / C_N))
    else:
        H_n = H_N * n / N

    return max(0.0, H_n)


def _rarefaction_simpson(species_counts: npt.NDArray, n: int, N: int) -> float:
    """
    Rarefied Simpson concentration (q=2) at sample size n.

    Based on Chao & Jost 2012 Eq. (5):
    D_n = D_N * (1 - (N-n)/N * (1 - C_n/C_N))

    where D_N is observed Simpson concentration, C_n is the interpolated
    (rarefied) coverage at size n (Chao & Jost 2012, Eq. 4) and C_N is
    the observed coverage estimated by Good-Turing as 1 - f1/N.

    Parameters
    ----------
    species_counts : array-like
        Species abundances
    n : int
        Subsample size
    N : int
        Total sample size

    Returns
    -------
    float
        Simpson diversity index ``1 - lambda`` of the rarefied sample (the
        Gini-Simpson convention used by Chao & Jost 2012, Eq. 5).  Use
        ``_hill_from_index(value, 2)`` for the Hill number ``1/lambda``.
    """
    if n <= 0:
        return 0.0
    if n >= N:
        p = species_counts / N
        D_N = 1.0 - np.sum(p**2)
        return max(0.0, D_N)

    p = species_counts / N
    D_N = 1.0 - np.sum(p**2)

    # Coverage at full sample (Good-Turing)
    f1, _ = _frequency_classes(species_counts)
    C_N = 1.0 - (f1 / N) if N > 0 else 0.0

    # Interpolated (rarefied) coverage at n (Chao & Jost 2012, Eq. 4)
    C_n = _rarefied_coverage(species_counts, n, N)

    if C_n > 0 and C_N > 0:
        D_n = D_N * (1 - ((N - n) / N) * (1 - C_n / C_N))
    else:
        D_n = D_N * n / N

    return max(0.0, D_n)


def _multinomial_resample(species_counts: npt.NDArray, N: int, rng: np.random.Generator) -> npt.NDArray:
    """
    Resample species abundances via multinomial bootstrap (Chao & Jost 2012).

    For abundance-based bootstrap, we sample N individuals from a multinomial
    distribution with probabilities proportional to the observed species
    proportions p_hat = species_counts / N.

    This is the standard non-parametric bootstrap for species abundance data,
    as implemented in the R iNEXT package.

    Parameters
    ----------
    species_counts : array-like
        Observed species abundances (counts).
    N : int
        Total number of individuals (sample size).
    rng : np.random.Generator
        Local random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Resampled species abundances (counts).

    References
    ----------
    Chao, A., & Jost, L. (2012). Coverage-based rarefaction and
        extrapolation: sampling and projecting species diversity.
        Methods in Ecology and Evolution, 3(5), 873-882.
    """
    if N <= 0:
        return np.zeros_like(species_counts)

    # Species proportions
    p_hat = species_counts / N

    # Multinomial resampling: sample N individuals
    # Result is counts for each species
    resampled = rng.multinomial(N, p_hat)

    return np.asarray(resampled, dtype=np.float64)


def _lgamma(x: float) -> float:
    """Log gamma function wrapper for numerical stability."""
    return math.lgamma(x) if x > 0 else 0.0


class CoverageRarefactionAnalyzer:
    """
    Computes coverage-based rarefaction curves.

    Estimates unobserved species richness based on sample coverage.
    Provides backward-compatible wrapper around coverage_rarefaction_hill().

    Example:
        >>> analyzer = CoverageRarefactionAnalyzer()
        >>> abundance = np.array([[25, 10, 5], [15, 20, 8]])
        >>> result = analyzer.analyze(abundance)
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize coverage rarefaction analyzer."""
        self._logger = logging.getLogger(f"{__name__}.CoverageRarefactionAnalyzer")
        self._last_result: CoverageRarefactionResult | None = None

    @property
    def last_result(self) -> CoverageRarefactionResult | None:
        """Get last computed result."""
        return self._last_result

    def coverage_rarefaction_hill(
        self,
        abundance_matrix: npt.NDArray,
        sample_names: list[str] | None = None,
        q: int = 0,
        n_points: int = 50,
        confidence_level: float = 0.95,
        n_bootstrap: int = 100,
        seed: int | None = None,
    ) -> CoverageRarefactionResult:
        """
        Compute coverage-based rarefaction using Hill numbers.

        Wrapper around the module-level coverage_rarefaction_hill() function.

        Parameters
        ----------
        abundance_matrix : array-like, shape (n_samples, n_species)
            Matrix of species abundances.
        sample_names : list of str, optional
            Names for each sample.
        q : int, default=0
            Order of Hill number (0=richness, 1=Shannon, 2=Simpson).
        n_points : int, default=50
            Number of coverage levels to evaluate.
        confidence_level : float, default=0.95
            Confidence level for bootstrap intervals.
        n_bootstrap : int, default=100
            Number of bootstrap replicates.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        CoverageRarefactionResult
        """
        result = coverage_rarefaction_hill(
            abundance_matrix=abundance_matrix,
            sample_names=sample_names,
            q=q,
            n_points=n_points,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        self._last_result = result
        return result

    def analyze(
        self,
        abundance_matrix: npt.NDArray,
        sample_names: list[str] | None = None,
        n_points: int = 50,
        confidence_level: float = 0.95,
    ) -> CoverageRarefactionResult:
        """
        Compute coverage-based rarefaction.

        Parameters:
            abundance_matrix: 2D array (n_samples, n_species) of abundances
            sample_names: Optional list of sample names
            n_points: Number of coverage levels to evaluate
            confidence_level: Confidence level for intervals

        Returns:
            CoverageRarefactionResult with rarefaction data

        Raises:
            ValidationError: If input data is invalid
        """
        self._logger.info(f"Computing coverage-based rarefaction: {n_points} points")

        # Validate input
        abundance_matrix = np.asarray(abundance_matrix, dtype=np.float64)
        if abundance_matrix.ndim != 2:
            raise ValidationError(_("Abundance matrix must be 2D"))

        n_samples, n_species = abundance_matrix.shape

        if sample_names is None:
            sample_names = [f"Sample_{i + 1}" for i in range(n_samples)]
        elif len(sample_names) != n_samples:
            raise ValidationError(
                _("Number of sample names ({0}) must match matrix rows ({1})").format(len(sample_names), n_samples)
            )

        # Compute coverage for each sample
        coverages = np.zeros(n_samples)
        richness = np.zeros(n_samples)
        asymptote = np.zeros(n_samples)
        expected_at_coverage = np.zeros((n_samples, n_points))
        ci_lower = np.zeros((n_samples, n_points))
        ci_upper = np.zeros((n_samples, n_points))

        # Coverage levels from 0.1 to 0.99
        coverage_levels = np.linspace(0.1, 0.99, n_points)

        # Two-sided normal quantile for the requested confidence level
        z = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

        # Whole-number abundances are required by the frequency spectrum;
        # round once (single warning) instead of mixing truncated and
        # float-valued counts, and keep the total consistent with the counts.
        abundance_matrix = _integerize_abundances(
            abundance_matrix, context="abundance_matrix"
        )

        for i in range(n_samples):
            row = abundance_matrix[i]
            species_counts = row[row > 0]
            total_n = int(np.sum(species_counts))

            if total_n == 0 or species_counts.size == 0:
                coverages[i] = 0.0
                richness[i] = 0.0
                asymptote[i] = 0.0
                continue

            # Count singletons, doubletons, etc. (np.isclose based so that
            # 0.99999999 is not silently dropped from the spectrum)
            f1, f2 = _frequency_classes(species_counts)
            s_obs = int(species_counts.size)  # Observed richness

            # Coverage estimate (Chao & Jost 2012, Eq. 3):
            #     Chat = 1 - (f1/n) * ((n-1) f1 / ((n-1) f1 + 2 f2))
            coverage_i = _sample_coverage(species_counts, total_n, f1, f2)

            coverages[i] = coverage_i
            richness[i] = s_obs

            # Asymptotic richness: Chao1 lower bound.  The previous
            # "2 * s_obs when singletons exist" fallback was an arbitrary
            # doubling with no estimator behind it.
            asymptote[i] = _hill_asymptote(species_counts, total_n, 0, f1, f2, s_obs)

            # Rarefaction/extrapolation at each coverage level, using the same
            # machinery as coverage_rarefaction_hill(): the interpolation
            # sample size is obtained by inverting the interpolated coverage
            # curve Chat(m) = c_level (Chao & Jost 2012, Eq. 4) instead of the
            # linear ratio m = n * c / Chat, which is not the coverage of any
            # subsample.
            for j, c_level in enumerate(coverage_levels):
                expected_at_coverage[i, j] = _coverage_curve_point(
                    species_counts, total_n, 0, c_level, coverage_i, asymptote[i]
                )

                # Approximate CI using Poisson-like variance.  The upper bound
                # is deliberately NOT clipped to n_species (the number of
                # columns of the input matrix): the Chao1 asymptote counts
                # species that are absent from the matrix by definition, so
                # clipping there truncated the interval below the estimate.
                var = expected_at_coverage[i, j] * (1 - c_level) / c_level
                std = math.sqrt(max(0.0, var))
                ci_lower[i, j] = max(0.0, expected_at_coverage[i, j] - z * std)
                ci_upper[i, j] = expected_at_coverage[i, j] + z * std

        result = CoverageRarefactionResult(
            sample_names=sample_names,
            coverage_levels=coverage_levels,
            expected_richness=np.mean(expected_at_coverage, axis=0),
            confidence_lower=np.mean(ci_lower, axis=0),
            confidence_upper=np.mean(ci_upper, axis=0),
            asymptote_estimate=asymptote,
            sample_sizes=abundance_matrix.sum(axis=1),
            method="inext",
        )

        self._last_result = result
        self._logger.info(f"Coverage rarefaction: {n_samples} samples analyzed")
        return result

    def _rarefaction_at_n(
        self,
        species_counts: npt.NDArray,
        n: int,
    ) -> float:
        """
        Compute expected species richness at sample size n.

        Uses classic Hurlbert's formula:
        E[S(n)] = sum_{i=1}^{S} [1 - C(N - n_i, n) / C(N, n)]

        Parameters:
            species_counts: Array of species abundances
            n: Target sample size

        Returns:
            Expected species richness
        """
        if n <= 0:
            return 0.0

        N = np.sum(species_counts)
        if n >= N:
            return len(species_counts)

        expected_s = 0.0
        for count in species_counts:
            if count <= N - n:
                # Use approximation for large numbers
                comb = self._combination_approx(N - count, n)
                comb_total = self._combination_approx(N, n)
                prob = 1.0 - (comb / comb_total) if comb_total > 0 else 0.0
            else:
                # Direct calculation for small values
                prob = 1.0 - self._hypergeometric_prob(count, N, n)
            expected_s += prob

        return expected_s

    def _combination_approx(self, n: int, k: int) -> float:
        """Approximate combination using log gamma."""
        if k < 0 or k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0

        # Use log gamma for numerical stability
        import math

        log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        return math.exp(log_comb)

    def _hypergeometric_prob(self, K: int, N: int, n: int) -> float:
        """Compute the probability of observing *zero* occurrences of a
        species in a sample of size ``n`` drawn without replacement from
        a population of size ``N`` containing ``K`` occurrences of the
        species.

        This is the hypergeometric tail probability P(X = 0):

            P(X = 0) = C(N - K, n) * C(K, 0) / C(N, n)
                     = C(N - K, n) / C(N, n)

        The previous implementation was written as a generic
        ``P(X = k)`` but its numerator and denominator terms cancelled
        exactly, leaving only ``exp(-lgamma(N + 1))`` = ``1 / N!`` —
        independent of ``K`` and ``n``. As a result every rarefaction /
        coverage estimate silently returned the same probability
        regardless of species abundance. Use the log-gamma form below
        for numerical stability on large counts.

        Parameters
        ----------
        K : int
            Total count of the focal species in the population.
        N : int
            Total population size (sum of all species counts).
        n : int
            Sample size (number of individuals drawn).

        Returns
        -------
        float
            ``P(X = 0)`` — probability the species is absent from the
            sample. The caller computes ``1 - P(X = 0)`` to obtain the
            probability of presence.
        """
        if K < 0 or N < 0 or n < 0:
            return 0.0
        if K > N or n > N:
            return 0.0
        if n == 0:
            # Drawing nothing -> the species is certainly absent.
            return 1.0
        if K == 0:
            # Species not in the population -> certainly absent.
            return 1.0
        if N - K < n:
            # Fewer non-focal individuals than the sample size, so the
            # species must appear at least once in the sample.
            return 0.0

        import math

        # log C(N - K, n) + log C(K, 0) - log C(N, n)
        # C(K, 0) = 1 -> log = 0
        log_num = (
            math.lgamma(N - K + 1)
            - math.lgamma(n + 1)
            - math.lgamma(N - K - n + 1)
        )
        log_den = (
            math.lgamma(N + 1)
            - math.lgamma(n + 1)
            - math.lgamma(N - n + 1)
        )
        return math.exp(log_num - log_den)
