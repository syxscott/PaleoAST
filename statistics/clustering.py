# =============================================================================
# FILE: statistics/clustering.py
# =============================================================================
"""
Hierarchical Clustering Module for PaleoAST

Provides agglomerative hierarchical clustering with dendrogram generation
and cophenetic correlation coefficient.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """
    Container for hierarchical clustering results.

    Attributes:
        linkage_matrix: scipy linkage matrix (n-1 x 4)
        cophenetic_corr: Cophenetic correlation coefficient
        labels: Cluster assignments for each sample (at given threshold)
        n_clusters: Number of clusters found
        distance_matrix: Original distance matrix
        method: Linkage method used
        metric: Distance metric used
    """

    linkage_matrix: npt.NDArray
    cophenetic_corr: float
    labels: npt.NDArray
    n_clusters: int
    distance_matrix: npt.NDArray
    method: str
    metric: str

    def summary(self) -> str:
        lines = [
            _("Hierarchical Clustering"),
            "=" * 45,
            f"{_('Method')}: {self.method}",
            f"{_('Distance metric')}: {self.metric}",
            f"{_('Cophenetic correlation')}: {self.cophenetic_corr:.4f}",
            f"{_('Clusters found')}: {self.n_clusters}",
        ]
        return "\n".join(lines)


LINKAGE_METHODS = ["ward", "complete", "average", "single"]
DISTANCE_METRICS = [
    "euclidean", "braycurtis", "canberra", "cityblock",
    "jaccard", "hamming", "cosine", "correlation",
]


class ClusteringAnalyzer:
    """Hierarchical clustering engine."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.ClusteringAnalyzer")
        self._lock = threading.RLock()
        self._last_result: ClusteringResult | None = None

    def analyze(
        self,
        data: npt.NDArray,
        method: str = "ward",
        metric: str = "euclidean",
        n_clusters: int | None = None,
        threshold: float | None = None,
    ) -> ClusteringResult:
        """
        Perform hierarchical clustering.

        Parameters:
            data: Data matrix (n_samples x n_variables) or precomputed distance matrix
            method: Linkage method ('ward', 'complete', 'average', 'single')
            metric: Distance metric (ignored if data is a distance matrix)
            n_clusters: Number of clusters to extract (default: 2)
            threshold: Distance threshold for cutting dendrogram

        Returns:
            ClusteringResult
        """
        with self._lock:
            data = validate_data_array(data, name="data")

            if method == "ward" and metric not in ("euclidean", None):
                self._logger.warning("Ward linkage requires Euclidean distance; overriding metric")
                metric = "euclidean"

            # Compute distance matrix if needed
            if data.shape[0] == data.shape[1] and self._is_distance_matrix(data):
                dm = data
                dist_condensed = squareform(dm, checks=False)
            else:
                dist_condensed = pdist(data, metric=metric)
                dm = squareform(dist_condensed)

            # Perform linkage
            Z = linkage(dist_condensed, method=method)

            # Cophenetic correlation
            coph_corr, _ = cophenet(Z, dist_condensed)

            # Extract clusters
            if n_clusters is not None:
                labels = fcluster(Z, n_clusters, criterion="maxclust")
            elif threshold is not None:
                labels = fcluster(Z, threshold, criterion="distance")
            else:
                n_clusters = 2
                labels = fcluster(Z, n_clusters, criterion="maxclust")

            n_found = len(set(labels))

            result = ClusteringResult(
                linkage_matrix=Z,
                cophenetic_corr=float(coph_corr),
                labels=labels,
                n_clusters=n_found,
                distance_matrix=dm,
                method=method,
                metric=metric,
            )

            self._last_result = result
            self._logger.info(
                f"Clustering complete: {n_found} clusters, cophenetic r={coph_corr:.4f}"
            )
            return result

    def _is_distance_matrix(self, data: npt.NDArray) -> bool:
        """Heuristic check if a matrix is a distance matrix."""
        if data.shape[0] != data.shape[1]:
            return False
        diag = np.diag(data)
        return np.allclose(diag, 0, atol=1e-10)

    @property
    def last_result(self) -> ClusteringResult | None:
        with self._lock:
            return self._last_result
