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
Version: 1.0.0
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import threading

from utils.exceptions import ComputationError, MatrixDimensionError
from utils.validators import validate_data_array
from config.constants import PERMUTATION_TESTS
from config.i18n import _

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
    groups: List[Any]
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
        self._last_result: Optional[PERMANOVAResult] = None
        self._n_permutations = PERMUTATION_TESTS
        self._logger.info("PERMANOVA initialized")
    
    def analyze(
        self,
        distance_matrix: npt.NDArray,
        groups: List[Any],
        n_permutations: Optional[int] = None,
        metric: str = 'euclidean'
    ) -> PERMANOVAResult:
        """
        Perform PERMANOVA analysis.
        
        Parameters:
            distance_matrix: Square distance/dissimilarity matrix
            groups: List of group assignments
            n_permutations: Number of permutations for p-value
            metric: Distance metric used (for reference)
        
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
                f"n_permutations={n_permutations}"
            )
            
            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square")
            
            if len(groups) != n:
                raise ComputationError(
                    "Group assignments must match distance matrix size"
                )
            
            if n_permutations is None:
                n_permutations = self._n_permutations
            
            groups_array = np.array(groups)
            unique_groups = sorted(set(groups), key=lambda x: str(x))
            g = len(unique_groups)
            
            # Compute observed F statistic
            F_obs, ss_between, ss_within, df_g, df_res = self._compute_F_statistic(D, groups_array, g, n)
            
            # Permutation test
            permuted_F = np.zeros(n_permutations)
            
            for i in range(n_permutations):
                # Randomly permute group assignments
                perm_indices = np.random.permutation(n)
                permuted_groups = groups_array[perm_indices]
                permuted_F[i] = self._compute_F_statistic(
                    D, permuted_groups, g, n
                )[0]
            
            # Calculate p-value
            p_value = np.mean(permuted_F >= F_obs)
            
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
                metric=metric
            )

            self._last_result = result
            self._logger.info(
                f"PERMANOVA completed: F={F_obs:.4f}, p-value={p_value:.4f}"
            )
            return result
    
    def _compute_F_statistic(
        self,
        D: npt.NDArray,
        groups: np.ndarray,
        g: int,
        n: int
    ) -> tuple:
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
        D_sq = D ** 2

        # Total sum of squares
        SS_T = np.sum(D_sq) / n

        # Within-group sum of squares
        ss_within = 0.0
        for grp in np.unique(groups):
            grp_indices = np.where(groups == grp)[0]
            n_g = len(grp_indices)
            if n_g < 2:
                continue
            grp_sum = 0.0
            for i in range(len(grp_indices)):
                for j in range(i + 1, len(grp_indices)):
                    grp_sum += D_sq[grp_indices[i], grp_indices[j]]
            ss_within += (2.0 / n_g) * grp_sum

        # Between-group sum of squares
        ss_between = SS_T - ss_within

        # Degrees of freedom
        df_g = g - 1
        df_res = n - g

        # F statistic
        if df_res > 0 and df_g > 0 and ss_within > 0:
            MS_between = ss_between / df_g
            MS_within = ss_within / df_res
            F = MS_between / MS_within
        else:
            F = 0.0

        return F, ss_between, ss_within, df_g, df_res
    
    @property
    def last_result(self) -> Optional[PERMANOVAResult]:
        """Get the last PERMANOVA result."""
        with self._lock:
            return self._last_result
