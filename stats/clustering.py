# =============================================================================
# FILE: stats/clustering.py
# =============================================================================
"""
Clustering Module for PaleoAST

Provides two families:

* **Agglomerative hierarchical clustering** with dendrogram generation
  and cophenetic correlation coefficient.
* **K-means partitioning**, with the elbow curve and silhouette score
  needed to choose k rather than have it guessed.

The two are complementary rather than competing: hierarchical clustering
answers "how many groups does the structure support" without being told
how many to look for, while k-means partitions cleanly once k is known.
PAST3 offers both (its Cluster menu has K-means alongside neighbour
joining), and the choice between them is a real modelling decision
rather than a preference.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError
from utils.statistics_core import make_rng
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


def _kmeans_backend():
    """Import scikit-learn lazily, or explain that it is missing.

    A module-level import would make every package that reaches
    ``stats.clustering`` -- and that is ``controllers``, and therefore
    ``views`` -- fail to import on a base install, because
    scikit-learn lives in the ``full`` extra rather than in the base
    dependencies. The Wheel Build & Import Smoke job installs base
    dependencies only, so it caught this where a local environment with
    the full extra set could not. The hierarchical path here needs only
    scipy and must keep working without scikit-learn.

    Follows the convention already used in stats/lda.py: import inside
    the function, raise the project's own error naming the package.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ComputationError(
            "k-means requires scikit-learn. Install the full extra: "
            "pip install 'PaleoAST[full]', or pip install scikit-learn",
            original_exception=exc,
        ) from exc
    return KMeans, silhouette_score


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
    "euclidean",
    "braycurtis",
    "canberra",
    "cityblock",
    "jaccard",
    "hamming",
    "cosine",
    "correlation",
]


@dataclass
class KMeansResult:
    """
    Container for a k-means partition.

    Attributes:
        labels: Cluster index (0-based) per sample.
        centroids: Cluster centres, one row per cluster.
        inertia: Within-cluster sum of squares (WCSS).
        silhouette: Mean silhouette coefficient, or NaN when undefined
            (fewer than 2 clusters, or k equal to the sample count).
        n_clusters: Number of clusters.
        n_iter: Iterations the solver used.
        n_samples: Number of samples clustered.
        seed: Seed the solver ran with, so a run can be reproduced.
    """

    labels: npt.NDArray
    centroids: npt.NDArray
    inertia: float
    silhouette: float
    n_clusters: int
    n_iter: int
    n_samples: int
    seed: int | None

    def cluster_sizes(self) -> dict[int, int]:
        """Number of samples in each cluster."""
        return {int(label): int(np.sum(self.labels == label)) for label in np.unique(self.labels)}

    def summary(self) -> str:
        lines = [
            _("K-Means Clustering"),
            "=" * 45,
            f"{_('Clusters')}: {self.n_clusters}",
            f"{_('Samples')}: {self.n_samples}",
            f"{_('Within-cluster sum of squares')}: {self.inertia:.4f}",
            f"{_('Silhouette')}: " + (f"{self.silhouette:.4f}" if np.isfinite(self.silhouette) else _("undefined")),
        ]
        for label, size in sorted(self.cluster_sizes().items()):
            lines.append(f"  {_('Cluster {0}').format(label + 1)}: {size}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "labels": [int(x) for x in self.labels],
            "centroids": [[float(v) for v in row] for row in self.centroids],
            "inertia": float(self.inertia),
            "silhouette": float(self.silhouette),
            "n_clusters": int(self.n_clusters),
            "n_iter": int(self.n_iter),
            "n_samples": int(self.n_samples),
            "seed": self.seed,
            "cluster_sizes": self.cluster_sizes(),
        }


@dataclass
class ElbowCurve:
    """
    Within-cluster sum of squares against k, for choosing k.

    Attributes:
        k_values: The k values evaluated.
        inertia: WCSS at each k, in the same order.
        recommended_k: The k at the largest drop in WCSS, chosen by the
            maximum second difference (the "elbow" of the curve). A
            heuristic, and documented as such: it is a starting point for
            inspection, not a decision.
    """

    k_values: npt.NDArray
    inertia: npt.NDArray
    recommended_k: int | None

    def summary(self) -> str:
        lines = [
            _("Elbow Curve"),
            "=" * 45,
            f"{_('k')}\t{_('Within-cluster sum of squares')} (lower is tighter)",
        ]
        for k, w in zip(self.k_values, self.inertia, strict=True):
            marker = "  <-- " if int(k) == self.recommended_k else ""
            lines.append(f"{int(k)}\t{w:.4f}{marker}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "k_values": [int(k) for k in self.k_values],
            "inertia": [float(w) for w in self.inertia],
            "recommended_k": self.recommended_k,
        }


class ClusteringAnalyzer:
    """Hierarchical clustering engine."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.ClusteringAnalyzer")
        self._lock = threading.RLock()
        self._last_result: ClusteringResult | None = None
        self._last_kmeans: KMeansResult | None = None

    def analyze(
        self,
        data: npt.NDArray,
        method: str = "ward",
        metric: str = "euclidean",
        n_clusters: int | None = None,
        threshold: float | None = None,
        precomputed: bool = False,
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
            if precomputed:
                if data.shape[0] != data.shape[1]:
                    raise MatrixDimensionError("Precomputed distance matrix must be square")
                if not self._is_distance_matrix(data):
                    raise MatrixDimensionError("Precomputed distance matrix must be symmetric with a zero diagonal")
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
            self._logger.info(f"Clustering complete: {n_found} clusters, cophenetic r={coph_corr:.4f}")
            return result

    def analyze_kmeans(
        self,
        data: npt.NDArray,
        n_clusters: int = 2,
        n_init: int = 10,
        max_iter: int = 300,
        random_seed: int | None = None,
        compute_silhouette: bool = True,
    ) -> KMeansResult:
        """
        Partition the data into k clusters by k-means.

        K-means minimises within-cluster sum of squares under Euclidean
        distance, so it assumes roughly spherical, similarly sized
        clusters. Where hierarchical clustering was the better choice --
        nested or elongated groups, or a dendrogram that shows the
        structure rather than assuming it -- this is the wrong tool, and
        the silhouette score in the result is the quickest way to notice
        when it has been used on the wrong data.

        Parameters:
            data: Data matrix (n_samples x n_variables)
            n_clusters: Number of clusters k
            n_init: Restarts per fit; k-means converges to a local
                optimum, so restarts are how the global one is approached
            max_iter: Iteration cap per restart
            random_seed: Seed for reproducibility
            compute_silhouette: Report the silhouette coefficient

        Returns:
            KMeansResult
        """
        with self._lock:
            X = validate_data_array(data, name="data")
            n_samples = X.shape[0]

            if n_clusters < 1:
                raise ValidationError(
                    f"n_clusters must be at least 1, got {n_clusters}",
                    details={"n_clusters": n_clusters},
                )
            if n_clusters >= n_samples:
                # Every point its own cluster: inertia 0, silhouette
                # undefined. Returning that would be a number with no
                # meaning behind it.
                raise ValidationError(
                    f"n_clusters ({n_clusters}) must be smaller than the number of samples ({n_samples})",
                    details={
                        "n_clusters": n_clusters,
                        "n_samples": n_samples,
                    },
                )
            if n_init < 1:
                raise ValidationError(
                    f"n_init must be at least 1, got {n_init}",
                    details={"n_init": n_init},
                )

            # The caller's seed IS the solver's seed. Drawing a sub-seed out
            # of default_rng(seed) instead would make the reported seed
            # unreplayable, which defeats the point of reporting it. The
            # unseeded case still draws one, from an isolated generator, so
            # that run too can be replayed from the seed it returns.
            if random_seed is None:
                rng = make_rng(None, context="K-means")
                seed = int(rng.integers(0, 2**31 - 1))
            else:
                seed = int(random_seed)

            if not np.all(np.isfinite(X)):
                raise ComputationError(
                    "K-means requires finite values; the input contains NaN or Inf. Impute or drop those rows first."
                )
            if np.allclose(X.std(axis=0), 0):
                raise ComputationError("K-means cannot partition data with no variation: every variable is constant.")

            kmeans_cls, silhouette_fn = _kmeans_backend()
            km = kmeans_cls(
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                random_state=seed,
            )
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_

            silhouette = float("nan")
            if compute_silhouette and 1 < n_clusters < n_samples:
                try:
                    silhouette = float(silhouette_fn(X, labels))
                except ValueError:
                    # sklearn raises when a cluster ends up empty after
                    # relabelling, which is a degenerate partition rather
                    # than a failed fit. NaN says "not meaningful here";
                    # zero would be read as "clusters overlap completely".
                    silhouette = float("nan")

            result = KMeansResult(
                labels=np.asarray(labels, dtype=int),
                centroids=np.asarray(centroids, dtype=float),
                inertia=float(km.inertia_),
                silhouette=silhouette,
                n_clusters=n_clusters,
                n_iter=int(km.n_iter_),
                n_samples=n_samples,
                seed=seed,
            )
            self._last_kmeans = result
            self._logger.info(
                "K-means complete: k=%d, WCSS=%.4f, silhouette=%s",
                n_clusters,
                result.inertia,
                f"{result.silhouette:.4f}" if np.isfinite(result.silhouette) else "n/a",
            )
            return result

    def elbow_curve(
        self,
        data: npt.NDArray,
        k_min: int = 2,
        k_max: int = 8,
        n_init: int = 10,
        random_seed: int | None = None,
    ) -> ElbowCurve:
        """
        Compute the within-cluster sum of squares for a range of k.

        The point of this is to avoid making the user guess k. WCSS falls
        monotonically with k, so the informative quantity is where it
        stops falling fast; ``recommended_k`` is the largest drop in the
        slope. That is a heuristic and is labelled as one in the result --
        the curve itself is the evidence, and the recommendation is a
        starting point to argue with.

        Parameters:
            data: Data matrix (n_samples x n_variables)
            k_min: Smallest k to evaluate
            k_max: Largest k to evaluate
            n_init: Restarts per fit
            random_seed: Seed for reproducibility

        Returns:
            ElbowCurve
        """
        with self._lock:
            X = validate_data_array(data, name="data")
            n_samples = X.shape[0]
            if k_min < 1 or k_max < k_min:
                raise ValidationError(
                    f"need 1 <= k_min <= k_max, got k_min={k_min}, k_max={k_max}",
                    details={"k_min": k_min, "k_max": k_max},
                )
            if k_max >= n_samples:
                raise ValidationError(
                    f"k_max ({k_max}) must be smaller than the number of samples ({n_samples})",
                    details={"k_max": k_max, "n_samples": n_samples},
                )

            rng = make_rng(random_seed, context="K-means elbow curve")
            k_values = np.arange(k_min, k_max + 1)
            inertia = np.empty(len(k_values), dtype=float)
            for i, k in enumerate(k_values):
                # A distinct seed per k, all drawn from the one generator,
                # so the whole curve is reproducible and each k is
                # independently reproducible.
                seed = int(rng.integers(0, 2**31 - 1))
                kmeans_cls, _silhouette_fn = _kmeans_backend()
                km = kmeans_cls(n_clusters=int(k), n_init=n_init, random_state=seed)
                km.fit(X)
                inertia[i] = float(km.inertia_)

            # Largest drop in WCSS between consecutive k.
            recommended: int | None = None
            if len(k_values) >= 3:
                drops = inertia[:-2] - 2 * inertia[1:-1] + inertia[2:]
                recommended = int(k_values[1:][int(np.argmax(drops))])
            elif len(k_values) == 2:
                recommended = int(k_values[0])

            return ElbowCurve(
                k_values=k_values,
                inertia=inertia,
                recommended_k=recommended,
            )

    def _is_distance_matrix(self, data: npt.NDArray) -> bool:
        """Heuristic check if a matrix is a distance matrix."""
        if data.shape[0] != data.shape[1]:
            return False
        diag = np.diag(data)
        return np.allclose(diag, 0, atol=1e-10) and np.allclose(data, data.T, atol=1e-10)

    @property
    def last_result(self) -> ClusteringResult | None:
        with self._lock:
            return self._last_result

    @property
    def last_kmeans_result(self) -> KMeansResult | None:
        """Most recent k-means partition, or None."""
        with self._lock:
            return self._last_kmeans
