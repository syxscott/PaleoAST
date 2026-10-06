# =============================================================================
# FILE: ecology/diversity.py
# =============================================================================
"""
Diversity Analysis Module for PaleoAST

This module implements alpha diversity indices for biodiversity analysis.

Supported Indices:
    - Species Richness (S): Number of unique taxa
    - Shannon Index (H'): -Σ p_i ln(p_i)
    - Simpson Index (D): 1 - Σ p_i²
    - Pielou's Evenness (J): H' / ln(S)
    - Margalef Index: (S-1) / ln(N)
    - Fisher's Alpha: Solves S = α ln(1 + N/α)
    - Chao-1: S_obs + f₁² / (2f₂)

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import math
import threading

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from models.diversity_result import DiversityIndexResult, DiversityResult
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


def compute_diversity_indices(abundances: npt.NDArray, sample_name: str = "Sample") -> DiversityResult:
    """
    Compute all diversity indices for a single sample.

    Parameters:
        abundances: Array of taxon abundances (counts)
        sample_name: Name/identifier for the sample

    Returns:
        DiversityResult: Complete diversity analysis results
    """
    # Validate input
    abundances = validate_data_array(abundances, allow_nan=False, name="abundances")

    # validate_data_array reshapes 1D to 2D; flatten back for 1D operations
    abundances = abundances.flatten()

    # Remove zeros and negative values
    abundances = abundances[abundances > 0]

    if len(abundances) == 0:
        raise ComputationError("No positive abundances found in sample")

    # `individuals` is reported to the user and drives Fisher's alpha, so it
    # must be a real count. `int(np.sum(...))` used to truncate silently: a
    # relative-abundance vector like [0.1, 0.2, 0.3, 0.4] summed to exactly
    # 1.0, became N=1, and the result claimed "4 taxa, 1 individual".
    #
    # The test MUST be on the individual values, not on the sum: these
    # fractions sum to an integer, so a sum-integrality check passes them.
    non_integer = ~np.isclose(abundances, np.round(abundances), rtol=0.0, atol=1e-9)
    if np.any(non_integer):
        offenders = np.unique(np.asarray(abundances)[non_integer])[:5]
        raise ComputationError(
            "Diversity indices require integer abundance counts (individuals), "
            f"but got fractional values {list(offenders)}. Pass raw counts, "
            "not a relative-abundance or percentage vector."
        )

    N = round(float(np.sum(abundances)))  # Total individuals
    S = len(abundances)  # Number of taxa
    logger.info(f"compute_diversity_indices started: n_taxa={S}, total_abundance={N}, sample_name='{sample_name}'")

    # Compute proportions
    p = abundances / N

    indices = {}

    # Shannon Index: H' = -Σ p_i ln(p_i)
    shannon = -np.sum(p * np.log(p))
    indices["shannon"] = DiversityIndexResult(
        index_name="Shannon Index (H')",
        value=float(shannon),
        formula=r"H' = -\sum_{i=1}^{S} p_i \ln(p_i)",
        interpretation=f"Shannon index of {shannon:.4f} indicates moderate diversity",
    )

    # Simpson Index (1-D): D = 1 - Σ p_i²
    simpson = 1 - np.sum(p**2)
    if S == 1:
        simpson_interpretation = "monospecific assemblage (S=1), Simpson=1 by definition - diversity indices are not meaningful for single-taxon samples"
    else:
        simpson_interpretation = (
            f"Simpson index of {simpson:.4f} indicates {'high diversity' if simpson > 0.7 else 'moderate diversity'}"
        )
    indices["simpson"] = DiversityIndexResult(
        index_name="Simpson Index (1-D)",
        value=float(simpson),
        formula=r"1 - D = 1 - \sum_{i=1}^{S} p_i^2",
        interpretation=simpson_interpretation,
    )

    # Pielou's Evenness: J = H' / ln(S)
    # Omitted for S=1 because J = H'/ln(1) is undefined (division by zero)
    if S > 1:
        pielou = shannon / np.log(S)
        indices["pielou"] = DiversityIndexResult(
            index_name="Pielou's Evenness (J)",
            value=float(pielou),
            formula=r"J = H' / \ln(S)",
            interpretation=f"Evenness of {pielou:.4f} indicates {'more even' if pielou > 0.6 else 'less even'} distribution",
        )

    # Margalef Index: (S-1) / ln(N)
    # Requires S >= 2 and N > 1 for meaningful diversity measurement
    if N > 1 and S >= 2:
        margalef = (S - 1) / np.log(N)
        indices["margalef"] = DiversityIndexResult(
            index_name="Margalef Index",
            value=float(margalef),
            formula=r"D_{Mg} = (S-1) / \ln(N)",
            interpretation=f"Margalef index of {margalef:.4f}",
        )
    elif S < 2:
        logger.debug("Margalef Index skipped: requires at least 2 taxa (S>=2) for meaningful measurement")

    # Fisher's Alpha
    fisher_alpha = _compute_fisher_alpha(S, N)
    if fisher_alpha is not None:
        indices["fisher_alpha"] = DiversityIndexResult(
            index_name="Fisher's Alpha (α)",
            value=float(fisher_alpha),
            formula=r"S = \alpha \ln(1 + N/\alpha)",
            interpretation=f"Fisher's alpha of {fisher_alpha:.4f} indicates {'high' if fisher_alpha > 20 else 'moderate'} diversity",
        )

    # Chao-1 estimator
    freq_counts = _compute_frequency_counts(abundances)
    f1 = freq_counts.get(1, 0)  # Taxa appearing once
    f2 = freq_counts.get(2, 0)  # Taxa appearing twice

    if f2 > 0:
        chao1 = S + (f1**2) / (2 * f2)
        indices["chao1"] = DiversityIndexResult(
            index_name="Chao-1 Estimator",
            value=float(chao1),
            formula=r"\hat{S}_{Chao1} = S_{obs} + \frac{f_1^2}{2f_2}",
            interpretation=f"Chao-1 estimate of {chao1:.1f} (S_obs={S})",
        )
    elif f1 > 0:
        chao1 = S + (f1 * (f1 - 1)) / 2
        indices["chao1"] = DiversityIndexResult(
            index_name="Chao-1 Estimator (adjusted)",
            value=float(chao1),
            formula=r"\hat{S}_{Chao1} = S + \frac{f_1(f_1-1)}{2}",
            interpretation=f"Chao-1 estimate of {chao1:.1f} (S_obs={S})",
        )
    else:
        # No rare species observed (f1=f2=0) - use observed richness as estimate
        indices["chao1"] = DiversityIndexResult(
            index_name="Chao-1 Estimator",
            value=float(S),
            formula=r"S_{obs} (no rare species observed)",
            interpretation=f"No rare species found; observed richness S={S} used as estimate",
        )

    logger.info(
        f"compute_diversity_indices completed: Shannon={shannon:.4f}, "
        f"Simpson={simpson:.4f}, n_taxa={S}, total_abundance={N}"
    )
    return DiversityResult(sample_name=sample_name, taxa_count=S, individuals=N, indices=indices)


def _compute_fisher_alpha(S: int, N: int) -> float | None:
    """
    Compute Fisher's alpha using Newton-Raphson iteration.

    Solves: S = α ln(1 + N/α)
    """
    if S <= 0 or N <= 0:
        return None

    # Initial guess
    alpha = 1.0

    for _ in range(100):
        # f(α) = α ln(1 + N/α) - S
        # f'(α) = ln(1 + N/α) - N/(α + N)

        f = alpha * np.log(1 + N / alpha) - S
        f_prime = np.log(1 + N / alpha) - N / (alpha + N)

        if abs(f_prime) < 1e-10:
            break

        alpha_new = alpha - f / f_prime

        if abs(alpha_new - alpha) < 1e-6:
            return alpha_new

        alpha = alpha_new

        if alpha <= 0:
            logger.warning(f"Fisher's alpha failed to converge for S={S}, N={N}")
            return None

    logger.warning(f"Fisher's alpha did not converge within 100 iterations for S={S}, N={N}")
    return None


def _compute_frequency_counts(abundances: npt.NDArray) -> dict[int, int]:
    """
    Compute frequency counts of abundances.

    Returns dictionary where keys are abundance values
    and values are counts of taxa with that abundance.
    """
    unique, counts = np.unique(abundances, return_counts=True)
    return dict(zip(unique.astype(int), counts, strict=False))


def chao1_confidence_interval(abundances: npt.NDArray, confidence_level: float = 0.95) -> tuple[float, float, float]:
    """
    Compute Chao1 richness estimator with confidence interval.

    Implements the bias-corrected Chao1 estimator and its variance
    following Chao (1987) and the log-transformation method for
    confidence intervals.

    Parameters
    ----------
    abundances : array-like
        Vector of species abundances (counts). Zeros are ignored.
    confidence_level : float, default=0.95
        Confidence level for the interval (e.g., 0.95 for 95% CI).

    Returns
    -------
    tuple[float, float, float]
        (chao1_estimate, ci_lower, ci_upper)

    Notes
    -----
    **Chao1 estimator** (Chao 1987):
        S_hat_Chao1 = S_obs + f_1² / (2·f_2)

    where f_1 = number of singletons and f_2 = number of doubletons.

    **Variance** (Chao 1987, Eq. 5):
        var(Chao1) = f_2 · [ (α/4)·(f_1/f_2)⁴ + (α²/2)·(f_1/f_2)³
                        + (α²/2)·(f_1/f_2)² + (α²/4)·(f_1/f_2) ]

    where α = 2·f_2 / ((n-1)·f_1 + 2·f_2), n = total individuals.

    **95% Confidence interval** (log-transformation, Chao & Jost 2012):
        K = exp(z_{α/2} · √(log(1 + var/Chao1²)))
        CI = [Chao1 / K,  Chao1 · K]

    References
    ----------
    Chao, A. (1987). Estimating the population size for capture-recapture
        data with unequal catchability. Biometrics, 43(4), 783-791.

    Chao, A., Chiu, C.-H., & Jost, L. (2014). Uncovering species diversity
        in ecological communities. Methods in Ecology and Evolution, 5(7),
        675-684.
    """
    # Validate and flatten input
    abundances = np.asarray(abundances, dtype=np.float64).flatten()
    abundances = abundances[abundances > 0]

    if len(abundances) == 0:
        return (0.0, 0.0, 0.0)

    n = int(np.sum(abundances))  # total individuals
    s_obs = len(abundances)  # observed richness

    # Frequency counts
    f1 = float(np.sum(abundances == 1))  # singletons
    f2 = float(np.sum(abundances == 2))  # doubletons

    # Handle edge cases
    if n == 0:
        return (float(s_obs), float(s_obs), float(s_obs))

    # ---- Chao1 point estimate ----
    if f2 > 0:
        chao1 = s_obs + (f1**2) / (2 * f2)
    elif f1 > 0:
        # Bias-corrected form when f2 == 0 but f1 > 0
        chao1 = s_obs + (f1 * (f1 - 1)) / 2
    else:
        chao1 = float(s_obs)

    # ---- Variance estimation (Chao 1987) ----
    if f2 > 0 and f1 > 0:
        alpha = (2 * f2) / ((n - 1) * f1 + 2 * f2)
        ratio = f1 / f2
        var_chao1 = f2 * (
            (alpha / 4) * ratio**4 + (alpha**2 / 2) * ratio**3 + (alpha**2 / 2) * ratio**2 + (alpha**2 / 4) * ratio
        )
    elif f1 > 1:
        # When f2 == 0 but f1 > 1: use approximate variance
        # var ≈ f1(f1-1)/2 (Chao 1987, Eq. 6)
        var_chao1 = f1 * (f1 - 1) / 2
    else:
        var_chao1 = 0.0

    # ---- Log-transformation CI ----
    z = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    if var_chao1 > 0 and chao1 > 0:
        log_ratio = math.log(1 + var_chao1 / (chao1**2))
        if log_ratio > 0:
            k = math.exp(z * math.sqrt(log_ratio))
            ci_lower = chao1 / k
            ci_upper = chao1 * k
        else:
            ci_lower = chao1
            ci_upper = chao1
    else:
        ci_lower = chao1
        ci_upper = chao1

    return (float(chao1), float(ci_lower), float(ci_upper))


class DiversityAnalyzer:
    """
    Diversity analyzer for community data.
    """

    def __init__(self) -> None:
        """Initialize the diversity analyzer."""
        self._lock = threading.RLock()
        self._last_result: DiversityResult | None = None

    def analyze_sample(self, abundances: npt.NDArray, sample_name: str = "Sample") -> DiversityResult:
        """
        Analyze diversity of a single sample.
        """
        with self._lock:
            result = compute_diversity_indices(abundances, sample_name)
            self._last_result = result
            return result

    def analyze_multiple(
        self,
        abundance_matrix: npt.NDArray,
        sample_names: list[str] | None = None,
        compute_standardized: bool = True,
        q_values: tuple[int, ...] = (0, 1, 2),
        coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
        n_bootstrap: int = 100,
        seed: int | None = 42,
    ) -> list[DiversityResult]:
        """
        Analyze diversity for multiple samples.

        Raw Shannon / Simpson / Pielou / Chao1 are computed as before
        (per-sample, on each sample's own N). When ``compute_standardized``
        is true, coverage-based standardized Hill numbers (q = 0, 1, 2)
        are also evaluated at the supplied coverage levels, using the
        helper from :mod:`ecology.beta_diversity`. Hill numbers are
        stored in each ``DiversityResult.metadata`` under
        ``standardized_hill``:

            {
                "q0": {"C=0.50": <S@50%>, "C=0.80": <S@80%>, ...},
                "q1": {"C=0.50": <exp(H')@50%>, ...},
                "q2": {"C=0.50": <1/λ@50%>, ...},
                "asymptote": {"q0": <Chao1>, "q1": <exp(H'_asymp)>,
                              "q2": <1/λ_asymp>},
                "note": str,
            }

        Comparing standardized Hill numbers across samples with unequal N
        is the only way to compare like-with-like — see Chao & Jost (2012)
        and Chao et al. (2014). The raw indices are still returned so
        callers that want the un-standardized values (e.g. for display
        next to a literature comparison that did the same) keep working.

        Parameters
        ----------
        abundance_matrix : array-like, shape (n_samples, n_taxa)
        sample_names : list of str, optional
        compute_standardized : bool, default True
            Whether to also compute coverage-based Hill numbers.
        q_values : tuple of {0, 1, 2}
            Hill orders to compute.
        coverage_levels : tuple of float in (0, 1)
            Coverage levels at which to evaluate Hill numbers.
        n_bootstrap : int, default 100
            Bootstrap replicates for the standardized estimator CIs.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        list of DiversityResult
        """
        with self._lock:
            abundance_matrix = np.asarray(abundance_matrix, dtype=float)
            if abundance_matrix.ndim == 1:
                abundance_matrix = abundance_matrix.reshape(1, -1)
            if sample_names is None:
                sample_names = [f"Sample_{i + 1}" for i in range(abundance_matrix.shape[0])]

            n_samples = abundance_matrix.shape[0]
            sample_sizes = abundance_matrix.sum(axis=1)
            sample_sizes_unequal = np.ptp(sample_sizes) > 0 if n_samples > 1 else False

            # Per-sample raw indices (unchanged behaviour)
            results = [compute_diversity_indices(abundance_matrix[i], sample_names[i]) for i in range(n_samples)]

            if not compute_standardized:
                return results

            # Coverage-based standardized Hill numbers, one q at a time.
            # The beta_diversity.coverage_rarefaction_hill helper
            # aggregates expected_richness across samples (returns the
            # sample-mean curve), but we want PER-SAMPLE values. We
            # therefore call it once per sample (single-row matrix) so
            # the result's expected_richness curve and asymptote index
            # back to that one sample unambiguously.
            from ecology.beta_diversity import coverage_rarefaction_hill

            for q in q_values:
                if q not in (0, 1, 2):
                    raise ValueError(f"q must be 0, 1, or 2, got {q}")
                for i, res in enumerate(results):
                    single_row = abundance_matrix[i : i + 1]
                    raref = coverage_rarefaction_hill(
                        single_row,
                        sample_names=[sample_names[i]],
                        q=q,
                        n_points=max(len(coverage_levels) * 4, 20),
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                    cl = np.asarray(raref.coverage_levels)
                    meta = res.metadata.setdefault("standardized_hill", {})
                    for c_target in coverage_levels:
                        idx = int(np.argmin(np.abs(cl - c_target)))
                        meta.setdefault(f"q{q}", {})[f"C={c_target:.2f}"] = float(raref.expected_richness[idx])
                    meta.setdefault("asymptote", {})[f"q{q}"] = float(raref.asymptote_estimate[0])

            # Warning + explanatory note when sample sizes differ
            if sample_sizes_unequal:
                note = (
                    "Sample sizes differ across rows; raw Shannon/Simpson "
                    "values are NOT comparable across samples. Use "
                    "metadata['standardized_hill'] (coverage-based Hill "
                    "numbers) for valid cross-sample comparison "
                    "(Chao & Jost 2012; Chao et al. 2014)."
                )
                for res in results:
                    res.metadata.setdefault("standardized_hill", {})["note"] = note
                logger.warning(note)

            return results

    @property
    def last_result(self) -> DiversityResult | None:
        """Get the last analysis result."""
        with self._lock:
            return self._last_result
