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
version: 1.0.1
"""

import logging
import threading

import numpy as np
import numpy.typing as npt

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

    N = int(np.sum(abundances))  # Total individuals
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
    indices["simpson"] = DiversityIndexResult(
        index_name="Simpson Index (1-D)",
        value=float(simpson),
        formula=r"1 - D = 1 - \sum_{i=1}^{S} p_i^2",
        interpretation=f"Simpson index of {simpson:.4f} indicates {'high diversity' if simpson > 0.7 else 'moderate diversity'}",
    )

    # Pielou's Evenness: J = H' / ln(S)
    if S > 1:
        pielou = shannon / np.log(S)
        indices["pielou"] = DiversityIndexResult(
            index_name="Pielou's Evenness (J)",
            value=float(pielou),
            formula=r"J = H' / \ln(S)",
            interpretation=f"Evenness of {pielou:.4f} indicates {'more even' if pielou > 0.6 else 'less even'} distribution",
        )

    # Margalef Index: (S-1) / ln(N)
    if N > 1:
        margalef = (S - 1) / np.log(N)
        indices["margalef"] = DiversityIndexResult(
            index_name="Margalef Index",
            value=float(margalef),
            formula=r"D_{Mg} = (S-1) / \ln(N)",
            interpretation=f"Margalef index of {margalef:.4f}",
        )

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
            return None

    return alpha


def _compute_frequency_counts(abundances: npt.NDArray) -> dict[int, int]:
    """
    Compute frequency counts of abundances.

    Returns dictionary where keys are abundance values
    and values are counts of taxa with that abundance.
    """
    unique, counts = np.unique(abundances, return_counts=True)
    return dict(zip(unique.astype(int), counts))


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
        self, abundance_matrix: npt.NDArray, sample_names: list[str] | None = None
    ) -> list[DiversityResult]:
        """
        Analyze diversity for multiple samples.

        Parameters:
            abundance_matrix: 2D array (n_samples, n_taxa)
            sample_names: Optional list of sample names

        Returns:
            List of DiversityResult objects
        """
        with self._lock:
            if sample_names is None:
                sample_names = [f"Sample_{i + 1}" for i in range(abundance_matrix.shape[0])]

            # Use list comprehension for slightly better performance
            results = [
                compute_diversity_indices(abundance_matrix[i], sample_names[i])
                for i in range(abundance_matrix.shape[0])
            ]

            return results

    @property
    def last_result(self) -> DiversityResult | None:
        """Get the last analysis result."""
        with self._lock:
            return self._last_result
