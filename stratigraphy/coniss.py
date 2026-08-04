# =============================================================================
# FILE: stratigraphy/coniss.py
# =============================================================================
"""
CONISS (Constrained Incremental Sum of Squares) for PaleoAST

CONISS is a hierarchical clustering method constrained to preserve
stratigraphic order. Used for pollen diagrams and microfossil
abundance zonation.

Mathematical Foundation:

CONISS uses incremental sum of squares to define cluster boundaries:

    ISS = Σ_j Σ_{i∈C_j} Σ_k (x_ik - x̄_jk)²

where x̄_jk is the mean of variable k in zone j.

The method proceeds by merging adjacent zones that minimize
the increase in ISS, subject to the constraint that only
adjacent stratigraphic levels can be merged.

Reference: Grimm (1987) "CONISS: A FORTRAN 78 program for
stratigraphically constrained cluster analysis." Computers &
Geosciences, 13, 13-35.

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
class CONISSResult:
    """Result of CONISS zonation."""

    linkage_matrix: npt.NDArray
    zone_labels: npt.NDArray
    n_zones: int
    iss_total: float
    iss_by_zone: list[float]
    depth_levels: npt.NDArray | None

    def summary(self) -> str:
        lines = [
            _("CONISS Zonation"),
            "=" * 40,
            f"{_('Total ISS')}: {self.iss_total:.4f}",
            f"{_('Zones')}: {self.n_zones}",
        ]
        for i, iss in enumerate(self.iss_by_zone):
            lines.append(f"  Zone {i + 1}: ISS = {iss:.4f}")
        return "\n".join(lines)


class CONISSAnalyzer:
    """
    CONISS constrained clustering for stratigraphic data.

    Performs constrained hierarchical clustering where only
    adjacent stratigraphic levels can be merged.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.CONISSAnalyzer")

    def analyze(
        self,
        data: npt.NDArray,
        n_zones: int = 4,
        depths: npt.NDArray | None = None,
        compute_broken_stick: bool = False,
        n_permutations: int = 999,
    ) -> tuple[CONISSResult, dict | None]:
        """
        Perform CONISS zonation.

        Parameters:
            data: Abundance matrix (n_levels x n_variables)
            n_zones: Number of zones to extract
            depths: Depth/age for each level (optional)
            compute_broken_stick: If True, compute broken-stick significance test
            n_permutations: Number of permutations for broken-stick test

        Returns:
            Tuple of (CONISSResult, broken_stick_significance)
            broken_stick_significance is None if compute_broken_stick=False
        """
        n_levels, _n_vars = data.shape

        if n_levels < 2:
            raise ComputationError("CONISS requires at least 2 stratigraphic levels")

        # Build constrained linkage using incremental sum of squares
        linkage_matrix = self._constrained_linkage(data)

        # Extract zones by cutting the dendrogram
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(linkage_matrix, n_zones, criterion="maxclust")

        # Compute ISS per zone
        iss_total = 0.0
        iss_by_zone = []
        for zone_id in sorted(set(labels)):
            mask = labels == zone_id
            zone_data = data[mask]
            zone_mean = zone_data.mean(axis=0)
            zone_iss = np.sum((zone_data - zone_mean) ** 2)
            iss_by_zone.append(float(zone_iss))
            iss_total += zone_iss

        result = CONISSResult(
            linkage_matrix=linkage_matrix,
            zone_labels=labels,
            n_zones=len(set(labels)),
            iss_total=float(iss_total),
            iss_by_zone=iss_by_zone,
            depth_levels=depths,
        )

        # Compute broken-stick significance if requested
        broken_stick_sig = None
        if compute_broken_stick:
            # BD values are the ISS increases (third column of linkage matrix)
            bd_values = linkage_matrix[:, 2]
            broken_stick_sig = broken_stick_test(bd_values, n_permutations)

        return result, broken_stick_sig

    def _constrained_linkage(self, data: npt.NDArray) -> npt.NDArray:
        """
        Build constrained linkage matrix.

        At each step, merge the pair of adjacent clusters
        that results in the smallest increase in ISS.
        """
        n = data.shape[0]
        # Each level starts as its own cluster
        clusters = {i: [i] for i in range(n)}
        cluster_order = list(range(n))

        linkage = np.zeros((n - 1, 4))
        current_iss = 0.0

        for step in range(n - 1):
            # Find the adjacent pair whose merge minimizes ISS increase
            best_merge = None
            best_increase = np.inf

            for k in range(len(cluster_order) - 1):
                c1_id = cluster_order[k]
                c2_id = cluster_order[k + 1]

                # ISS increase from merging c1 and c2
                merged_data = np.vstack([data[clusters[c1_id]], data[clusters[c2_id]]])
                merged_mean = merged_data.mean(axis=0)
                merged_iss = np.sum((merged_data - merged_mean) ** 2)

                c1_data = data[clusters[c1_id]]
                c1_iss = np.sum((c1_data - c1_data.mean(axis=0)) ** 2)

                c2_data = data[clusters[c2_id]]
                c2_iss = np.sum((c2_data - c2_data.mean(axis=0)) ** 2)

                increase = merged_iss - c1_iss - c2_iss

                if increase < best_increase:
                    best_increase = increase
                    best_merge = k

            # Merge the best pair. Guard against the theoretical case
            # where the inner loop never finds a valid merge (e.g.
            # all clusters are identical); ``best_merge`` is initialised
            # to 0 but could in principle be ``None`` after edits.
            if best_merge is None:
                raise RuntimeError(
                    "CONISS: failed to find a valid cluster merge — "
                    "check that the input data contains at least 2 distinct rows."
                )
            k = best_merge
            c1_id = cluster_order[k]
            c2_id = cluster_order[k + 1]

            new_id = n + step
            clusters[new_id] = clusters[c1_id] + clusters[c2_id]

            # Record in linkage matrix
            n1 = len(clusters[c1_id])
            n2 = len(clusters[c2_id])
            linkage[step] = [c1_id, c2_id, best_increase, n1 + n2]

            # Update cluster order
            cluster_order = [*cluster_order[:k], new_id, *cluster_order[k + 2 :]]

            current_iss += best_increase

        return linkage


def broken_stick_test(bd_values: npt.NDArray, n_permutations: int = 999) -> dict:
    """
    Broken-stick model significance test for CONISS zone selection.

    Determines the optimal number of zones by comparing the observed
    within-group inertia (BD values from linkage matrix) to the
    expected distribution under a broken-stick model.

    The broken-stick model (Bennett 1996 / Grimm 1987) assumes that
    a stick of unit length is randomly broken into n segments.
    Each segment represents a zone, with expected length = 1/n.

    Parameters
    ----------
    bd_values : array-like
        BD (Barton-David) inertia values from CONISS linkage matrix
        (typically the ISS increase at each merge step, third column
        of linkage matrix).
    n_permutations : int, default 999
        Number of Monte Carlo permutations for significance testing.

    Returns
    -------
    dict with keys:
        - 'significant_zones': int, number of statistically significant zones
        - 'p_values': list of float, p-value for each zone boundary
        - 'broken_stick_expectation': list of float, expected contribution
          under broken-stick model for each zone

    Notes
    -----
    The broken-stick model:

    1. Randomly assign 1 unit length to n segments (expected length = 1/n)
    2. Compare observed group inertia to broken-stick expectation
    3. A zone is significant when its contribution > broken-stick expectation

    References
    ----------
    Bennett, K.D. (1996). "Determination of the number of zones in a
        biostratigraphical sequence." New Phytologist, 132: 155-170.

    Grimm, E.C. (1987). "CONISS: A FORTRAN 78 program for
        stratigraphically constrained cluster analysis." Computers &
        Geosciences, 13: 13-35.
    """
    bd_values = np.asarray(bd_values, dtype=float)
    n_levels = len(bd_values) + 1  # n zones = n_levels - 1 merges possible

    # Compute broken-stick expected values
    # For n segments, expected length of segment i (sorted descending) is:
    # E[i] = 1/(n) + 1/(n-1) + ... + 1/(n-i+1)
    broken_stick_expectation = []
    for k in range(1, n_levels):
        # Expected contribution of zone k under broken-stick
        expected = 1.0 / (n_levels - k + 1)
        broken_stick_expectation.append(expected)

    # Normalize BD values to sum to 1 (proportion of total inertia)
    total_bd = np.sum(bd_values)
    if total_bd == 0:
        return {
            "significant_zones": 0,
            "p_values": [1.0] * (n_levels - 1),
            "broken_stick_expectation": broken_stick_expectation,
        }

    normalized_bd = bd_values / total_bd

    # Monte Carlo permutation test
    # Under null hypothesis, zone contributions are randomly distributed
    n_permutations = max(n_permutations, 99)  # minimum 99 for valid p-values
    permutation_max = np.zeros(n_permutations)

    for perm in range(n_permutations):
        # Random permutation of normalized BD values
        perm_bd = np.random.permutation(normalized_bd)
        # Compute maximum deviation from broken-stick expectation
        perm_max = np.max(np.abs(np.cumsum(perm_bd) - np.array(broken_stick_expectation)))
        permutation_max[perm] = perm_max

    # Compute p-values for each zone boundary
    observed_cumsum = np.cumsum(normalized_bd)
    p_values = []
    significant_zones = 0

    for k in range(len(normalized_bd)):
        observed_dev = np.abs(observed_cumsum[k] - broken_stick_expectation[k])
        # p-value = proportion of permutations with larger deviation
        p_val = np.mean(permutation_max >= observed_dev)
        p_values.append(float(p_val))
        if p_val < 0.05 and k + 1 > significant_zones:
            significant_zones = k + 1

    # Number of significant zones = number of boundaries where p < 0.05
    # But we need at least 1 zone
    significant_zones = max(1, sum(1 for p in p_values if p < 0.05))

    return {
        "significant_zones": significant_zones,
        "p_values": p_values,
        "broken_stick_expectation": broken_stick_expectation,
    }
