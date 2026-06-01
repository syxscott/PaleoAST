# =============================================================================
# FILE: stratigraphy/markov.py
# =============================================================================
"""
Markov Chain Analysis for Stratigraphic Facies Transitions

Tests whether vertical facies transitions follow a random (independent)
sequence or exhibit first-order Markov dependency.

Mathematical Foundation:

Given a transition count matrix T, the expected count under independence is:

    E_ij = (row_i_total × col_j_total) / grand_total

The chi-squared statistic for Markovity:

    χ² = Σ (T_ij - E_ij)² / E_ij

with df = (n-1)² degrees of freedom.

The Difference Matrix (Powers & Easterling 1982):

    D_ij = T_ij - E_ij

Positive D_ij indicates preferred transition; negative indicates avoided.

Reference: Powers & Easterling (1982) "Improved methodology for using
embedded Markov chains to describe vertical changes in rock columns."
Mathematical Geology, 14, 121-136.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ComputationError

logger = logging.getLogger(__name__)


@dataclass
class MarkovResult:
    """Result of Markov chain analysis."""

    transition_matrix: npt.NDArray
    expected_matrix: npt.NDArray
    difference_matrix: npt.NDArray
    chi_squared: float
    p_value: float
    df: int
    n_transitions: int
    is_markovian: bool
    facies_names: list[str]

    def summary(self) -> str:
        sig = "Yes" if self.is_markovian else "No"
        lines = [
            _("Markov Chain Analysis"),
            "=" * 45,
            f"χ² = {self.chi_squared:.4f}, df = {self.df}, p = {self.p_value:.4f}",
            f"{_('Markovian')}? {sig}",
            f"{_('Transitions')}: {self.n_transitions}",
            "",
            _("Transition Matrix (observed):"),
        ]
        # Header
        header = "      " + "".join(f"{n:>8}" for n in self.facies_names)
        lines.append(header)
        for i, name in enumerate(self.facies_names):
            row = f"{name:>6}" + "".join(f"{self.transition_matrix[i, j]:>8.0f}" for j in range(len(self.facies_names)))
            lines.append(row)

        lines.append("")
        lines.append(_("Difference Matrix (observed - expected):"))
        lines.append(header)
        for i, name in enumerate(self.facies_names):
            row = f"{name:>6}" + "".join(f"{self.difference_matrix[i, j]:>8.2f}" for j in range(len(self.facies_names)))
            lines.append(row)

        return "\n".join(lines)


class MarkovAnalyzer:
    """Markov chain analysis for facies transitions."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.MarkovAnalyzer")

    def analyze(
        self,
        sequence: list[int] | npt.NDArray,
        facies_names: list[str] | None = None,
    ) -> MarkovResult:
        """
        Analyze Markov property of a stratigraphic sequence.

        Parameters:
            sequence: Ordered sequence of facies codes (integers)
            facies_names: Names for each facies type

        Returns:
            MarkovResult
        """
        sequence = np.array(sequence)
        if len(sequence) < 2:
            raise ValueError("Sequence must have at least 2 elements for Markov chain analysis")
        # Use the number of *unique* facies codes, not max+1. The
        # previous formula ``int(np.max(sequence)) + 1`` would create
        # empty rows in the transition matrix whenever the facies
        # codes are not contiguous (e.g. {0, 2, 5} would allocate
        # 6 states, three of which are unused).
        n_states = int(len(np.unique(sequence)))

        if facies_names is None:
            facies_names = [f"Facies_{i}" for i in range(n_states)]

        # Build transition count matrix
        T = np.zeros((n_states, n_states), dtype=float)
        for i in range(len(sequence) - 1):
            T[sequence[i], sequence[i + 1]] += 1

        n_transitions = int(T.sum())

        # Expected matrix under independence
        row_sums = T.sum(axis=1, keepdims=True)
        col_sums = T.sum(axis=0, keepdims=True)
        grand_total = T.sum()
        E = (row_sums @ col_sums) / grand_total if grand_total > 0 else np.zeros_like(T)

        # Difference matrix
        D = T - E

        # Chi-squared test
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2_terms = np.where(E > 0, (T - E) ** 2 / E, 0)
        chi2 = float(np.sum(chi2_terms))
        df = (n_states - 1) ** 2

        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df) if df > 0 else 1.0

        return MarkovResult(
            transition_matrix=T,
            expected_matrix=E,
            difference_matrix=D,
            chi_squared=chi2,
            p_value=p_value,
            df=df,
            n_transitions=n_transitions,
            is_markovian=p_value < 0.05,
            facies_names=facies_names[:n_states],
        )
