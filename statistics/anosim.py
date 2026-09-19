# =============================================================================
# FILE: statistics/anosim.py
# =============================================================================
"""
Analysis of Similarities (ANOSIM) Module for PaleoAST

ANOSIM is a non-parametric test for differences between groups
based on distance/similarity matrices.

Mathematical Foundation:

ANOSIM statistic R:
    R = (r̄_B - r̄_W) / [n(n-1)/4]

where:
    r̄_B = mean rank of between-group similarities
    r̄_W = mean rank of within-group similarities
    n = total number of samples

The denominator is half of the total number of pairs, n(n-1)/4, which keeps
R within [-1, 1]: R ≈ 1 means the groups are more dissimilar between each
other than within, R ≈ 0 means no structure, R < 0 means the groups are more
similar to each other than within.

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
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError
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
        random_seed: int | None = None,
    ) -> ANOSIMResult:
        """
        Perform ANOSIM analysis.

        Parameters:
            distance_matrix: Square distance/dissimilarity matrix
            groups: List of group assignments (integers or strings)
            n_permutations: Number of permutations for p-value
            metric: Distance metric used (for reference)
            random_seed: Optional seed for the permutation RNG so the
                p-value is reproducible. Without a seed the global
                ``np.random`` state is used, which means successive
                calls with identical inputs may return slightly
                different p-values.

        Returns:
            ANOSIMResult: ANOSIM analysis results

        Raises:
            ValidationError: if the design cannot produce both between- and
                within-group pairs (fewer than 2 samples or 2 groups).
        """
        with self._lock:
            # Validate input
            D = validate_data_array(distance_matrix, allow_nan=False, name="distance_matrix")

            n = D.shape[0]

            if D.shape[0] != D.shape[1]:
                raise MatrixDimensionError("Distance matrix must be square")

            if len(groups) != n:
                raise ComputationError("Group assignments must match distance matrix size")

            # Group bookkeeping is validated *before* it is used (both for the
            # log message and for the result), and it needs the sample count
            # check above to have run first.
            unique_groups = sorted(set(groups), key=lambda x: str(x))
            self._logger.info(
                f"ANOSIM analyze started: n_samples={n}, n_groups={len(unique_groups)}, "
                f"n_permutations={n_permutations}, random_seed={random_seed}"
            )

            # ANOSIM contrasts between- with within-group rank sums, so both
            # kinds of pair have to exist (mirrors vegan::anosim, which
            # requires at least two groups plus replication).  Without these
            # checks the statistic silently degenerates: an empty side is
            # treated as a mean rank of 0 and R leaves its [-1, 1] range.
            if n < 2:
                raise ValidationError("ANOSIM requires at least 2 samples", details={"n_samples": n})
            if len(unique_groups) < 2:
                raise ValidationError(
                    "ANOSIM requires at least 2 groups",
                    details={"n_groups": len(unique_groups)},
                )
            if n == len(unique_groups):
                # Every group holds exactly one sample -> no within-group pairs.
                raise ValidationError(
                    "ANOSIM requires at least one within-group pair, so at least "
                    "one group must contain 2 or more samples",
                    details={"n_samples": n, "n_groups": len(unique_groups)},
                )

            if n_permutations is None:
                n_permutations = self._n_permutations

            # Compute observed R statistic
            R_obs = self._compute_R_statistic(D, groups)

            # Permutation test. Use a dedicated Generator when a seed
            # is supplied so the test is reproducible.
            if random_seed is not None:
                rng = np.random.default_rng(random_seed)
            else:
                rng = np.random

            permuted_R = np.zeros(n_permutations)
            groups_array = np.array(groups)

            for i in range(n_permutations):
                # Randomly permute group assignments
                perm_indices = rng.permutation(n)
                permuted_groups = groups_array[perm_indices]
                permuted_R[i] = self._compute_R_statistic(D, permuted_groups)

            # Calculate p-value
            p_value = float((1 + np.sum(permuted_R >= R_obs)) / (n_permutations + 1))

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

        R = (r̄_B - r̄_W) / [n(n-1)/4]

        The denominator is half of the total number of pairs N = n(n-1)/2
        (Legendre & Fortin 1989; identical to ``m <- n*(n-1)/4`` in
        vegan::anosim), which keeps R within [-1, 1].

        Raises:
            ValidationError: if either the between- or the within-group pair
                set is empty.  Averaging over an empty set used to be silently
                replaced by 0, which produced out-of-range statistics such as
                R = 2.0 or R = -1.02.
        """
        n = D.shape[0]
        if n < 2:
            raise ValidationError(
                "ANOSIM requires at least 2 samples", details={"n_samples": n}
            )

        # Compute all pairwise similarities (1 - distance)
        S = 1 - D
        np.fill_diagonal(S, 1.0)

        # Rank all similarities
        ranks = np.zeros((n, n))

        # Get upper triangle values
        upper_tri_indices = np.triu_indices(n, k=1)
        sim_values = S[upper_tri_indices]

        # Rank from largest to smallest using stable sort to preserve order for ties
        # Then assign ranks accounting for ties by averaging
        order = np.argsort(-sim_values, kind="stable")
        sorted_vals = sim_values[order]

        # Compute ranks with tie-handling: assign average rank to tied values
        n_vals = len(sorted_vals)
        ranks_list = np.empty(n_vals, dtype=float)
        i = 0
        while i < n_vals:
            val = sorted_vals[i]
            j = i
            while j < n_vals and sorted_vals[j] == val:
                j += 1
            # All values from i to j-1 are tied; assign average rank
            avg_rank = (i + 1 + j) / 2.0
            ranks_list[i:j] = avg_rank
            i = j

        # Reorder back to original positions
        unranked_order = np.argsort(order)
        final_ranks = ranks_list[unranked_order]
        ranks[upper_tri_indices[0], upper_tri_indices[1]] = final_ranks

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

        if not r_B_list:
            raise ValidationError(
                "ANOSIM needs at least one between-group pair: every sample "
                "belongs to the same group",
                details={"n_samples": n},
            )
        if not r_W_list:
            raise ValidationError(
                "ANOSIM needs at least one within-group pair: at least one "
                "group must contain 2 or more samples",
                details={"n_samples": n, "n_groups": len(set(groups))},
            )
        r_B = float(np.mean(r_B_list))
        r_W = float(np.mean(r_W_list))

        # Compute R statistic
        # ANOSIM R = (r_B - r_W) / (N/2) where N = n(n-1)/2 is the total number
        # of pairs.  The denominator N/2 = n(n-1)/4 is the scaling factor that
        # keeps R bounded by [-1, 1].
        N = n * (n - 1) / 2
        R = (r_B - r_W) / (N / 2)

        return R

    @property
    def last_result(self) -> ANOSIMResult | None:
        """Get the last ANOSIM result."""
        with self._lock:
            return self._last_result
