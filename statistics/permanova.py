# =============================================================================
# FILE: statistics/permanova.py
# =============================================================================
"""
Permutational Multivariate Analysis of Variance (PERMANOVA) Module

PERMANOVA is a non-parametric multivariate analysis that tests whether
groups differ in their multivariate centroids.

Mathematical Foundation:

PERMANOVA F statistic:
    F = (SS_B / (g-1)) / (SS_W / (n-g))

where:
    SS_B = sum of squares between groups
    SS_W = sum of squares within groups
    g = number of groups
    n = total number of samples

SS_B and SS_W are computed from distance matrices:
    SS_T = Σᵢ Σⱼ d²_ij / n
    SS_B = Σ_g n_g * d̄²_g. - SS_T/n * Σ_g n_g
    SS_W = SS_T - SS_B

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
class PERMANOVAResult:
    """
    Container for PERMANOVA analysis results.
    """

    f_statistic: float
    p_value: float
    n_permutations: int
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    groups: list[Any]
    n_groups: int
    n_samples: int
    metric: str

    def summary(self) -> str:
        """Generate summary text."""
        sig = "**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "")
        return (
            f"{_('PERMANOVA Results')}\n"
            f"{'=' * 50}\n"
            f"{_('F statistic: {0}').format(f'{self.f_statistic:.4f}')}\n"
            f"{_('P-value: {0}').format(f'{self.p_value:.4f} {sig}')}\n"
            f"{_('Permutations: {0}').format(self.n_permutations)}\n"
            f"\n"
            f"{_('Degrees of freedom:')}\n"
            f"{_('Between groups: {0}').format(self.df_between)}\n"
            f"{_('Within groups: {0}').format(self.df_within)}\n"
            f"\n"
            f"{_('Sum of squares:')}\n"
            f"{_('Between (SS_B): {0}').format(f'{self.ss_between:.4f}')}\n"
            f"{_('Within (SS_W): {0}').format(f'{self.ss_within:.4f}')}\n"
            f"\n"
            f"{_('Mean squares:')}\n"
            f"{_('Between (MS_B): {0}').format(f'{self.ms_between:.4f}')}\n"
            f"{_('Within (MS_W): {0}').format(f'{self.ms_within:.4f}')}"
        )


class PERMANOVAAnalyzer:
    """
    PERMANOVA analyzer for multivariate group comparisons.

    PERMANOVA tests whether groups differ significantly based on
    a distance matrix, without assuming normality.
    """

    def __init__(self) -> None:
        """Initialize the PERMANOVA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.PERMANOVAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: PERMANOVAResult | None = None
        self._n_permutations = PERMUTATION_TESTS
        self._logger.info("PERMANOVA initialized")

    def analyze(
        self,
        distance_matrix: npt.NDArray,
        groups: list[Any],
        n_permutations: int | None = None,
        metric: str = "euclidean",
        random_seed: int | None = None,
    ) -> PERMANOVAResult:
        """
        Perform PERMANOVA analysis.

        Parameters:
            distance_matrix: Square distance/dissimilarity matrix
            groups: List of group assignments
            n_permutations: Number of permutations for p-value
            metric: Distance metric used (for reference)
            random_seed: Optional seed for the permutation RNG so that
                the resulting p-value is reproducible. Without a seed,
                ``np.random`` is used as-is and two calls with the same
                data may return slightly different p-values.

        Returns:
            PERMANOVAResult: PERMANOVA analysis results
        """
        with self._lock:
            # Validate input
            D = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            n = D.shape[0]
            unique_groups = sorted(set(groups), key=lambda x: str(x))
            self._logger.info(
                f"PERMANOVA analyze started: n_samples={n}, n_groups={len(unique_groups)}, "
                f"n_permutations={n_permutations}, random_seed={random_seed}"
            )

            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square")

            if len(groups) != n:
                raise ComputationError("Group assignments must match distance matrix size")

            if n_permutations is None:
                n_permutations = self._n_permutations

            groups_array = np.array(groups)
            unique_groups = sorted(set(groups), key=lambda x: str(x))
            g = len(unique_groups)

            # Compute observed F statistic
            F_obs, ss_between, ss_within, df_g, df_res = self._compute_F_statistic(D, groups_array, g, n)

            # Permutation test. Use a dedicated Generator when a seed is
            # supplied so the test is fully reproducible; fall back to
            # ``np.random`` otherwise for backward compatibility.
            if random_seed is not None:
                rng = np.random.default_rng(random_seed)
            else:
                rng = np.random

            permuted_F = np.zeros(n_permutations)

            for i in range(n_permutations):
                # Randomly permute group assignments
                perm_indices = rng.permutation(n)
                permuted_groups = groups_array[perm_indices]
                permuted_F[i] = self._compute_F_statistic(D, permuted_groups, g, n)[0]

            # Calculate p-value
            p_value = float((1 + np.sum(permuted_F >= F_obs)) / (n_permutations + 1))

            # Mean squares
            ms_between = ss_between / df_g if df_g > 0 else 0
            ms_within = ss_within / df_res if df_res > 0 else 0

            result = PERMANOVAResult(
                f_statistic=F_obs,
                p_value=p_value,
                n_permutations=n_permutations,
                df_between=df_g,
                df_within=df_res,
                ss_between=ss_between,
                ss_within=ss_within,
                ms_between=ms_between,
                ms_within=ms_within,
                groups=unique_groups,
                n_groups=g,
                n_samples=n,
                metric=metric,
            )

            self._last_result = result
            self._logger.info(f"PERMANOVA completed: F={F_obs:.4f}, p-value={p_value:.4f}")
            return result

    def _compute_F_statistic(self, D: npt.NDArray, groups: np.ndarray, g: int, n: int) -> tuple:
        """
        Compute PERMANOVA F statistic from distance matrix.

        Uses Anderson (2001) formula:
            SS_T = (1/n) * Σ_i Σ_j d²_ij
            SS_W = Σ_g (1/n_g) * Σ_{i<j in g} d²_ij
            SS_B = SS_T - SS_W
            F = (SS_B / (g-1)) / (SS_W / (n-g))

        Returns:
            tuple: (F, SS_B, SS_W, df_g, df_res)
        """
        # Square distances
        D_sq = D**2

        # Total sum of squares (Anderson 2001): sum over unordered
        # pairs only. The full squared distance matrix contains each
        # pair twice, so use the upper triangle.
        SS_T = np.sum(D_sq[np.triu_indices(n, k=1)]) / n

        # Within-group sum of squares (vectorized)
        # For each group g with n_g samples, compute sum of squared distances
        # ss_within = sum_g (1/n_g) * sum_{i<j in g} d_ij^2
        # Use upper triangle of distance matrix for unordered pairs
        ss_within = 0.0
        triu_idx = np.triu_indices(n, k=1)
        D_sq_triu = D_sq[triu_idx]
        for grp in np.unique(groups):
            grp_mask = groups == grp
            grp_indices = np.where(grp_mask)[0]
            n_g = len(grp_indices)
            if n_g < 2:
                continue
            # Build index mask for pairs within this group
            # Create boolean mask for upper triangle elements where both i and j are in group
            idx_i = triu_idx[0]
            idx_j = triu_idx[1]
            in_grp_i = grp_mask[idx_i]
            in_grp_j = grp_mask[idx_j]
            grp_pair_mask = in_grp_i & in_grp_j
            grp_sum = np.sum(D_sq_triu[grp_pair_mask])
            ss_within += (1.0 / n_g) * grp_sum

        # Between-group sum of squares
        ss_between = SS_T - ss_within

        # Degrees of freedom
        df_g = g - 1
        df_res = n - g

        # F statistic
        if df_g <= 0:
            # Single group - test not applicable
            F = 0.0
        elif df_res <= 0:
            # Perfect separation (all within-group variance is zero)
            F = float("inf")
        else:
            MS_between = ss_between / df_g
            MS_within = ss_within / df_res
            F = MS_between / MS_within if MS_within > 0 else float("inf")

        return F, ss_between, ss_within, df_g, df_res

    @property
    def last_result(self) -> PERMANOVAResult | None:
        """Get the last PERMANOVA result."""
        with self._lock:
            return self._last_result
