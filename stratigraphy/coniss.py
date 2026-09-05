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
    The broken-stick model (MacArthur 1957):

    1. A stick of unit length is broken at n-1 random points into n
       segments; the expected length of the k-th largest segment is
       E[k] = (1/n) * sum_{i=k..n} (1/i).
    2. Observed BD values (sorted descending, normalised to sum 1) are
       compared with the expectation one-to-one.
    3. p-values come from Monte-Carlo simulation of true random
       breakages (Dirichlet(1,...,1)); a zone is significant when its
       normalised BD exceeds E[k] with p < 0.05, counted as a
       contiguous prefix from the largest zone (Bennett 1996).

    References
    ----------
    Bennett, K.D. (1996). "Determination of the number of zones in a
        biostratigraphical sequence." New Phytologist, 132: 155-170.

    Grimm, E.C. (1987). "CONISS: A FORTRAN 78 program for
        stratigraphically constrained cluster analysis." Computers &
        Geosciences, 13: 13-35.
    """
    bd_values = np.asarray(bd_values, dtype=float)
    n_segments = len(bd_values)
    if n_segments == 0:
        return {
            "significant_zones": 0,
            "p_values": [],
            "broken_stick_expectation": [],
        }

    # 规范 broken-stick 期望 (MacArthur 1957; Bennett 1996;
    # rioja::bstick 同式): 把单位长度随机折成 n 段, 第 k 大段的期望为
    #     E[k] = (1/n) * Σ_{i=k..n} (1/i)
    # 此前的实现用调和级数项 1/(n_levels - i) (升序) 归一化后与降序 BD
    # 配对, 不是 broken-stick 分布 (n=5 时最大份额 0.12 vs 规范 0.457)。
    n = n_segments
    harmonic_suffix = np.cumsum(1.0 / np.arange(n, 0, -1))[::-1]  # Σ_{i=k..n} 1/i, k=1..n
    broken_stick_expectation = harmonic_suffix / n  # 和为 1

    # 观测 BD 按降序排列 (最大惯性增量 = 最早/最重要的分带), 与
    # E[1..n] 一一对应; 归一化到总惯量为 1。
    sorted_bd = np.sort(bd_values)[::-1]
    total_bd = np.sum(bd_values)
    if total_bd <= 0:
        return {
            "significant_zones": 0,
            "p_values": [1.0] * n,
            "broken_stick_expectation": broken_stick_expectation.tolist(),
        }
    normalized_bd = sorted_bd / total_bd

    # 显著性: 逐带比较观测与期望, 并用 Monte Carlo 模拟真实
    # broken-stick 随机分割 (Dirichlet(1,...,1) 与随机折棒等价)
    # 估计每个秩次的 p 值。Bennett (1996) 的判据是"自最大带起,
    # 观测 > 期望 的连续前缀长度"为显著带数。
    n_permutations = max(int(n_permutations), 99)
    rng = np.random.default_rng()
    perm_sorted = np.sort(rng.dirichlet(np.ones(n), size=n_permutations), axis=1)[:, ::-1]
    # p_k = P(随机分割的第 k 大段 >= 观测第 k 大段) (add-one)
    exceed = perm_sorted >= normalized_bd[np.newaxis, :]
    p_values = (np.sum(exceed, axis=0) + 1.0) / (n_permutations + 1.0)

    significant_zones = 0
    for k in range(n):
        if normalized_bd[k] > broken_stick_expectation[k] and p_values[k] < 0.05:
            significant_zones = k + 1
        else:
            break

    return {
        "significant_zones": significant_zones,
        "p_values": [float(p) for p in p_values],
        "broken_stick_expectation": broken_stick_expectation.tolist(),
    }
