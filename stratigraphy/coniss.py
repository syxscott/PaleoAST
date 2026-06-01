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
    ) -> CONISSResult:
        """
        Perform CONISS zonation.

        Parameters:
            data: Abundance matrix (n_levels x n_variables)
            n_zones: Number of zones to extract
            depths: Depth/age for each level (optional)

        Returns:
            CONISSResult
        """
        n_levels, n_vars = data.shape

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

        return CONISSResult(
            linkage_matrix=linkage_matrix,
            zone_labels=labels,
            n_zones=len(set(labels)),
            iss_total=float(iss_total),
            iss_by_zone=iss_by_zone,
            depth_levels=depths,
        )

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

            # Merge the best pair
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
            cluster_order = cluster_order[:k] + [new_id] + cluster_order[k + 2:]

            current_iss += best_increase

        return linkage
