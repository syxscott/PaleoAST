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
    S_tu = 2*min(b, c) / (2a + b + c)

Nestedness component:
    J_ne = |b - c| / (a + b + c)
    S_ne = |b - c| / (2a + b + c)

Coverage-based Rarefaction:

Coverage: C = 1 - (f1/N) * qD_1
where f1 = singletons, N = total individuals, qD_1 = Hill number of order 1

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

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
                    #         = a·|b - c| / [(2a + b + c)·(2a + min(b,c))]
                    denom = 2 * a + b + c
                    total_val = (b + c) / denom if denom > 0 else 0.0
                    min_bc = min(b, c)
                    denom_turn = 2 * a + min_bc
                    turn_val = min_bc / denom_turn if denom_turn > 0 else 0.0
                    nest_val = (a * abs(b - c)) / (denom * denom_turn) if (denom > 0 and denom_turn > 0) else 0.0

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
        - q=1: Shannon entropy (exp(H'))
        - q=2: Simpson concentration (1/D)
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
        C_n = 1 - (f_1/n) · ((n-1)·f_1 / ((n-1)·f_1 + 2·f_2))

    **Asymptotic diversity estimators** (Chao et al. 2014):
        - q=0: S_hat = S_obs + f_1² / (2·f_2)  (Chao1)
        - q=1: S_hat = S_obs + f_1 · γ  where γ = (n-1)·f_1 / ((n-1)·f_1 + 2·f_2)
        - q=2: S_hat = S_obs + f_1 · γ²

    **Rarefaction/extrapolation** (Chao & Jost 2012):
        For coverage C < C_obs: interpolate using classic rarefaction
        For coverage C > C_obs: extrapolate toward asymptotic estimator

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
    z = 1.96 if confidence_level == 0.95 else 2.576

    # Storage for per-sample results
    sample_coverage = np.zeros(n_samples)
    asymptote = np.zeros(n_samples)
    richness_curve = np.zeros((n_samples, n_points))
    ci_lower_curve = np.zeros((n_samples, n_points))
    ci_upper_curve = np.zeros((n_samples, n_points))

    for i in range(n_samples):
        row = abundance_matrix[i]
        total_n = int(np.sum(row))
        species_counts = row[row > 0]

        if total_n == 0 or len(species_counts) == 0:
            sample_coverage[i] = 0.0
            asymptote[i] = 0.0
            continue

        # Count singletons and doubletons
        f1 = float(np.sum(species_counts == 1))
        f2 = float(np.sum(species_counts == 2))
        s_obs = len(species_counts)

        # ---- Coverage estimator (Chao & Jost 2012, Eq. 3) ----
        if f1 > 0 and total_n > 1:
            gamma_factor = ((total_n - 1) * f1) / ((total_n - 1) * f1 + 2 * f2) if ((total_n - 1) * f1 + 2 * f2) > 0 else 0.0
        else:
            gamma_factor = 0.0
        coverage_i = 1.0 - (f1 / total_n) * gamma_factor if total_n > 0 else 0.0
        sample_coverage[i] = coverage_i

        # ---- Asymptotic diversity estimator (Chao et al. 2014) ----
        if q == 0:
            # Chao1 for species richness
            if f2 > 0:
                asymptote[i] = s_obs + (f1**2) / (2 * f2)
            elif f1 > 1:
                asymptote[i] = s_obs + f1 * (f1 - 1) / 2
            else:
                asymptote[i] = float(s_obs)
        elif q == 1:
            # Shannon entropy estimator (Chao et al. 2014)
            if f1 > 0 and f2 >= 0:
                asymptote[i] = s_obs + f1 * gamma_factor
            else:
                # Observed Shannon diversity
                p = species_counts / total_n
                n0 = s_obs - np.sum(p * np.log(p))
                asymptote[i] = s_obs + n0
        else:  # q == 2
            # Simpson concentration estimator (Chao et al. 2014)
            if f1 > 0 and f2 >= 0:
                asymptote[i] = s_obs + f1 * (gamma_factor**2)
            else:
                p = species_counts / total_n
                n0 = s_obs - np.sum(p**2)
                asymptote[i] = s_obs + n0

        # ---- Rarefaction/extrapolation at each coverage level ----
        bootstrap_curves = []
        for _ in range(n_bootstrap):
            boot_curve = np.zeros(n_points)
            for j, c_level in enumerate(coverage_levels):
                if c_level <= coverage_i:
                    # Interpolation (rarefaction)
                    # Sample size that gives this coverage
                    m = max(1, int(total_n * c_level / coverage_i)) if coverage_i > 0 else 1
                    m = min(m, total_n - 1)
                    if q == 0:
                        boot_curve[j] = _rarefaction_species(species_counts, m)
                    elif q == 1:
                        boot_curve[j] = _rarefaction_shannon(species_counts, m, total_n)
                    else:
                        boot_curve[j] = _rarefaction_simpson(species_counts, m, total_n)
                else:
                    # Extrapolation toward asymptote
                    ratio = c_level / coverage_i if coverage_i > 0 else 1.0
                    boot_curve[j] = s_obs + (asymptote[i] - s_obs) * (ratio - 1)
                    boot_curve[j] = min(boot_curve[j], asymptote[i])
            bootstrap_curves.append(boot_curve)

        bootstrap_curves = np.array(bootstrap_curves)

        # Use median bootstrap curve as point estimate
        richness_curve[i] = np.median(bootstrap_curves, axis=0)

        # Percentile-based CI
        alpha = 1 - confidence_level
        ci_lower_curve[i] = np.percentile(bootstrap_curves, 100 * alpha / 2, axis=0)
        ci_upper_curve[i] = np.percentile(bootstrap_curves, 100 * (1 - alpha / 2), axis=0)

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

    where H_N is observed Shannon entropy, C_n is coverage at size n.

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

    # Coverage at full sample
    f1 = float(np.sum(species_counts == 1))
    C_N = 1.0 - (f1 / N) if N > 0 else 0.0

    # Approximate coverage at n
    if n < N:
        f1_n = max(1.0, f1 * (n / N))
        C_n = 1.0 - (f1_n / n) if n > 0 else 0.0
    else:
        C_n = C_N

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

    where D_N is observed Simpson concentration, C_n is coverage.

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
        Expected Simpson concentration (1/D)
    """
    if n <= 0:
        return 0.0
    if n >= N:
        p = species_counts / N
        D_N = 1.0 - np.sum(p**2)
        return max(0.0, D_N)

    p = species_counts / N
    D_N = 1.0 - np.sum(p**2)

    # Coverage
    f1 = float(np.sum(species_counts == 1))
    C_N = 1.0 - (f1 / N) if N > 0 else 0.0

    if n < N:
        f1_n = max(1.0, f1 * (n / N))
        C_n = 1.0 - (f1_n / n) if n > 0 else 0.0
    else:
        C_n = C_N

    if C_n > 0 and C_N > 0:
        D_n = D_N * (1 - ((N - n) / N) * (1 - C_n / C_N))
    else:
        D_n = D_N * n / N

    return max(0.0, D_n)


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

        for i in range(n_samples):
            row = abundance_matrix[i]
            total_n = np.sum(row)
            species_counts = row[row > 0]

            if total_n == 0:
                coverages[i] = 0.0
                richness[i] = 0.0
                asymptote[i] = 0.0
                continue

            # Count singletons, doubletons, etc.
            f1 = np.sum(species_counts == 1)  # Singletons
            f2 = np.sum(species_counts == 2)  # Doubletons
            s_obs = len(species_counts)  # Observed richness

            # Coverage estimate (Chao et al. 2014)
            if f1 > 0 and total_n > 0:
                coverage_i = 1.0 - (f1 / total_n) * ((total_n - 1) / (total_n - f1 + 1))
            else:
                coverage_i = 1.0 - f1 / total_n if total_n > 0 else 0.0

            coverages[i] = coverage_i
            richness[i] = s_obs

            # Estimate asymptote using Chao1-like estimator
            if f1 > 0 and f2 > 0:
                asymptote[i] = s_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1))
            else:
                asymptote[i] = s_obs * 2 if f1 > 0 else s_obs

            # Rarefaction/extrapolation at each coverage level
            for j, c_level in enumerate(coverage_levels):
                if c_level <= coverage_i:
                    # Interpolation (rarefaction)
                    # Sample size that gives this coverage
                    m = int(total_n * c_level / coverage_i) if coverage_i > 0 else 0
                    m = max(1, min(m, total_n - 1))
                    # Expected richness at m individuals
                    rarefaction = self._rarefaction_at_n(species_counts, m)
                    expected_at_coverage[i, j] = rarefaction
                else:
                    # Extrapolation
                    # Use asymptotic estimator scaled by coverage ratio
                    ratio = c_level / coverage_i if coverage_i > 0 else 1.0
                    expected_at_coverage[i, j] = s_obs + (asymptote[i] - s_obs) * (ratio - 1)
                    expected_at_coverage[i, j] = min(expected_at_coverage[i, j], asymptote[i])

                # Approximate CI using Poisson-like variance
                var = expected_at_coverage[i, j] * (1 - c_level) / c_level
                std = math.sqrt(max(0, var))
                z = 1.96 if confidence_level == 0.95 else 2.576
                ci_lower[i, j] = max(0, expected_at_coverage[i, j] - z * std)
                ci_upper[i, j] = min(n_species, expected_at_coverage[i, j] + z * std)

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
