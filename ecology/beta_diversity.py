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
Version: 1.0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError, ValidationError

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
        for i, name in enumerate(self.sample_names):
            lines.append(
                f"{name}: S={self.expected_richness[i]:.1f}, "
                f"coverage={self.coverage_levels[i]:.1%}, "
                f"asymptote={self.asymptote_estimate[i]:.1f}"
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

        n_samples, n_species = abundance_matrix.shape

        if sample_names is None:
            sample_names = [f"Sample_{i + 1}" for i in range(n_samples)]
        elif len(sample_names) != n_samples:
            raise ValidationError(
                _("Number of sample names ({0}) must match matrix rows ({1})").format(
                    len(sample_names), n_samples
                )
            )

        if metric not in ("jaccard", "sorensen"):
            raise ValidationError(
                _("Metric must be 'jaccard' or 'sorensen', got '{0}'").format(metric)
            )

        # Convert to presence-absence (boolean for bitwise operations)
        presence = (abundance_matrix > 0)

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
                    total_val = (b + c) / total
                    min_bc = min(b, c)
                    turn_val = (2 * min_bc) / (a + 2 * min_bc) if (a + 2 * min_bc) > 0 else 0.0
                    nest_val = abs(b - c) / total
                else:  # sorensen
                    denom = 2 * a + b + c
                    total_val = 2 * a / denom if denom > 0 else 0.0
                    min_bc = min(b, c)
                    turn_val = (2 * min_bc) / denom if denom > 0 else 0.0
                    nest_val = abs(b - c) / denom if denom > 0 else 0.0

                total_beta[i, j] = total_beta[j, i] = total_val
                turnover[i, j] = turnover[j, i] = turn_val
                nestedness[i, j] = nestedness[j, i] = nest_val

                pairwise_results.append({
                    "sample_i": sample_names[i],
                    "sample_j": sample_names[j],
                    "shared_species": int(a),
                    "only_i": int(b),
                    "only_j": int(c),
                    "total_beta": float(total_val),
                    "turnover": float(turn_val),
                    "nestedness": float(nest_val),
                })

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


class CoverageRarefactionAnalyzer:
    """
    Computes coverage-based rarefaction curves.

    Estimates unobserved species richness based on sample coverage.

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
                _("Number of sample names ({0}) must match matrix rows ({1})").format(
                    len(sample_names), n_samples
                )
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
                coverage_i = 1.0 - (f1 / total_n) * (
                    (total_n - 1) / (total_n - f1 + 1)
                )
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
        log_comb = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
        return math.exp(log_comb)

    def _hypergeometric_prob(self, k: int, N: int, n: int) -> float:
        """Compute hypergeometric probability."""
        if k < 0 or k > min(n, N):
            return 0.0

        import math
        log_prob = (
            math.lgamma(k + 1)
            + math.lgamma(N - k + 1)
            + math.lgamma(n - k + 1)
            + math.lgamma(N - n + 1)
            - math.lgamma(N + 1)
            - math.lgamma(k + 1)
            - math.lgamma(N - k + 1)
            - math.lgamma(n - k + 1)
            - math.lgamma(N - n + 1)
        )
        return math.exp(log_prob)
