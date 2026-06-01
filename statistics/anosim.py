# =============================================================================
# FILE: statistics/anosim.py
# =============================================================================
"""
Analysis of Similarities (ANOSIM) Module for PaleoAST

ANOSIM is a non-parametric test for differences between groups
based on distance/similarity matrices.

Mathematical Foundation:

ANOSIM statistic R:
    R = (r̄_B - r̄_W) / [½ * n(n-1)]

where:
    r̄_B = mean rank of between-group similarities
    r̄_W = mean rank of within-group similarities
    n = total number of samples

Significance is assessed via permutation test.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from config.constants import PERMUTATION_TESTS
from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class ANOSIMResult:
    """
    Container for ANOSIM analysis results.

    Attributes:
        statistic: R statistic value
        p_value: Permutation-based p-value
        n_permutations: Number of permutations used
        groups: List of unique group identifiers
        n_groups: Number of groups
        n_samples: Total number of samples
        metric: Distance metric used
    """

    statistic: float
    p_value: float
    n_permutations: int
    groups: list[Any]
    n_groups: int
    n_samples: int
    metric: str

    def summary(self) -> str:
        """Generate summary text."""
        sig_marker = "**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "")
        return (
            f"{_('Analysis of Similarities (ANOSIM)')}\n"
            f"{'=' * 45}\n"
            f"{_('Test statistic (R): {0}').format(f'{self.statistic:.4f}')}\n"
            f"{_('P-value: {0}').format(f'{self.p_value:.4f} {sig_marker}')}\n"
            f"{_('Permutations: {0}').format(self.n_permutations)}\n"
            f"{_('Groups: {0}').format(self.n_groups)}\n"
            f"{_('Distance metric: {0}').format(self.metric)}"
        )


class ANOSIMAnalyzer:
    """
    Analysis of Similarities (ANOSIM) analyzer.

    ANOSIM tests whether there are significant differences
    between groups of samples based on their distance matrix.
    """

    def __init__(self) -> None:
        """Initialize the ANOSIM analyzer."""
        self._logger = logging.getLogger(f"{__name__}.ANOSIMAnalyzer")
        self._lock = threading.RLock()
        self._last_result: ANOSIMResult | None = None
        self._n_permutations = PERMUTATION_TESTS
        self._logger.info("ANOSIM initialized")

    def analyze(
        self,
        distance_matrix: npt.NDArray,
        groups: list[Any],
        n_permutations: int | None = None,
        metric: str = "euclidean",
    ) -> ANOSIMResult:
        """
        Perform ANOSIM analysis.

        Parameters:
            distance_matrix: Square distance/dissimilarity matrix
            groups: List of group assignments (integers or strings)
            n_permutations: Number of permutations for p-value
            metric: Distance metric used (for reference)

        Returns:
            ANOSIMResult: ANOSIM analysis results
        """
        with self._lock:
            # Validate input
            D = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            n = D.shape[0]
            unique_groups = sorted(set(groups), key=lambda x: str(x))
            self._logger.info(
                f"ANOSIM analyze started: n_samples={n}, n_groups={len(unique_groups)}, n_permutations={n_permutations}"
            )

            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square")

            if len(groups) != n:
                raise ComputationError("Group assignments must match distance matrix size")

            if n_permutations is None:
                n_permutations = self._n_permutations

            # Compute observed R statistic
            R_obs = self._compute_R_statistic(D, groups)

            # Permutation test
            permuted_R = np.zeros(n_permutations)
            groups_array = np.array(groups)

            for i in range(n_permutations):
                # Randomly permute group assignments
                perm_indices = np.random.permutation(n)
                permuted_groups = groups_array[perm_indices]
                permuted_R[i] = self._compute_R_statistic(D, permuted_groups)

            # Calculate p-value
            p_value = np.mean(permuted_R >= R_obs)

            # Get unique groups
            unique_groups = sorted(set(groups), key=lambda x: str(x))

            result = ANOSIMResult(
                statistic=R_obs,
                p_value=p_value,
                n_permutations=n_permutations,
                groups=unique_groups,
                n_groups=len(unique_groups),
                n_samples=n,
                metric=metric,
            )

            self._last_result = result
            self._logger.info(f"ANOSIM completed: R={R_obs:.4f}, p-value={p_value:.4f}")
            return result

    def _compute_R_statistic(self, D: npt.NDArray, groups: np.ndarray) -> float:
        """
        Compute ANOSIM R statistic.

        R = (r̄_B - r̄_W) / [½ * n(n-1)]
        """
        n = D.shape[0]

        # Compute all pairwise similarities (1 - distance)
        S = 1 - D
        np.fill_diagonal(S, 1.0)

        # Rank all similarities
        ranks = np.zeros((n, n))

        # Get upper triangle values
        upper_tri_indices = np.triu_indices(n, k=1)
        sim_values = S[upper_tri_indices]

        # Rank from largest to smallest
        rank_order = np.argsort(-sim_values)
        ranks[upper_tri_indices[0][rank_order], upper_tri_indices[1][rank_order]] = np.arange(1, len(sim_values) + 1)

        # Make symmetric
        ranks = ranks + ranks.T

        # Separate between-group and within-group ranks
        r_B_list = []
        r_W_list = []

        for i in range(n):
            for j in range(i + 1, n):
                if groups[i] != groups[j]:
                    r_B_list.append(ranks[i, j])
                else:
                    r_W_list.append(ranks[i, j])

        r_B = np.mean(r_B_list) if r_B_list else 0
        r_W = np.mean(r_W_list) if r_W_list else 0

        # Compute R statistic
        # ANOSIM R = (r_B - r_W) / (N/2) where N = n(n-1)/2 is the total number of pairs
        # The denominator N/2 = n(n-1)/4 represents the scaling factor
        N = n * (n - 1) / 2
        R = (r_B - r_W) / (N / 2)

        return R

    @property
    def last_result(self) -> ANOSIMResult | None:
        """Get the last ANOSIM result."""
        with self._lock:
            return self._last_result
