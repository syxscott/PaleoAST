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

import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import threading

from utils.exceptions import ComputationError, MatrixDimensionError
from utils.validators import validate_data_array
from config.constants import PERMUTATION_TESTS


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
            f"PERMANOVA Results\n"
            f"{'=' * 50}\n"
            f"F statistic: {self.f_statistic:.4f}\n"
            f"P-value: {self.p_value:.4f} {sig}\n"
            f"Permutations: {self.n_permutations}\n"
            f"\n"
            f"Degrees of freedom:\n"
            f"  Between groups: {self.df_between}\n"
            f"  Within groups: {self.df_within}\n"
            f"\n"
            f"Sum of squares:\n"
            f"  Between (SS_B): {self.ss_between:.4f}\n"
            f"  Within (SS_W): {self.ss_within:.4f}\n"
            f"\n"
            f"Mean squares:\n"
            f"  Between (MS_B): {self.ms_between:.4f}\n"
            f"  Within (MS_W): {self.ms_within:.4f}"
        )


class PERMANOVAAnalyzer:
    """
    PERMANOVA analyzer for multivariate group comparisons.
    
    PERMANOVA tests whether groups differ significantly based on
    a distance matrix, without assuming normality.
    """
    
    def __init__(self) -> None:
        """Initialize the PERMANOVA analyzer."""
        self._lock = threading.RLock()
        self._last_result: Optional[PERMANOVAResult] = None
        self._n_permutations = PERMUTATION_TESTS
    
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
        
        Returns:
            tuple: (F, SS_B, SS_W, df_g, df_res)
        """
        # Square distances
        D_sq = D ** 2
        
        # Total sum of squares (relative to grand centroid)
        # SS_T = Σᵢ Σⱼ d²_ij / n
        SS_T = np.sum(D_sq) / n
        
        # Group sizes
        group_sizes = {unique: np.sum(groups == unique) for unique in np.unique(groups)}
        
        # Between-group sum of squares
        # SS_B = Σ_g n_g * d̄²_g. - SS_T/n * Σ_g n_g
        ss_between = 0.0
        for grp, size in group_sizes.items():
            # Get indices for this group
            grp_indices = np.where(groups == grp)[0]
            
            # Sum of distances within group
            group_dist_sum = 0.0
            for i in range(len(grp_indices)):
                for j in range(i + 1, len(grp_indices)):
                    group_dist_sum += D_sq[grp_indices[i], grp_indices[j]]
            
            # Add diagonal (self-distances, typically 0)
            group_dist_sum += np.sum(D_sq[grp_indices, grp_indices])
            
            # Mean within group
            n_g = size
            if n_g > 1:
                mean_dist_sq = group_dist_sum / (n_g * n_g)
            else:
                mean_dist_sq = 0
            
            ss_between += n_g * mean_dist_sq
        
        # Total mean
        grand_mean_sq = SS_T / n
        ss_between = ss_between - n * grand_mean_sq
        
        # Within-group sum of squares
        ss_within = SS_T - ss_between
        
        # Degrees of freedom
        df_g = g - 1
        df_res = n - g
        
        # F statistic
        if df_res > 0 and df_g > 0:
            MS_between = ss_between / df_g
            MS_within = ss_within / df_res
            F = MS_between / MS_within if MS_within > 0 else 0
        else:
            F = 0
        
        return F, ss_between, ss_within, df_g, df_res
    
    @property
    def last_result(self) -> Optional[PERMANOVAResult]:
        """Get the last PERMANOVA result."""
        with self._lock:
            return self._last_result
