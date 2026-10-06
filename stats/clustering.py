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

PAST4 additionally offers three methods this file now covers, each
filling a gap the first two leave:

* **K-medoids (PAM)**, where every cluster centre is an actual observed
  sample rather than an average that may not correspond to any specimen.
* **DBSCAN**, which finds density-connected clusters and reports
  outliers as such instead of forcing every point into a group.
* **K-nearest neighbours classification**, which assigns cases to the
  class their nearest observed neighbours belong to.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from config.i18n import _
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError
from utils.statistics_core import make_rng
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)

__all__ = [
    "DISTANCE_METRICS",
    "LINKAGE_METHODS",
    "ClusteringAnalyzer",
    "ClusteringResult",
    "DBSCANResult",
    "ElbowCurve",
    "KMeansResult",
    "KMedoidsResult",
    "KNNResult",
]


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


class _SklearnBackend(NamedTuple):
    """The scikit-learn entry points this module needs, by name.

    Returned as a NamedTuple rather than a bare tuple because three
    call sites each want one of these and one of them wanted the wrong
    one: positional unpacking picked up ``DBSCAN`` where the kNN
    classifier was meant. Naming them makes that a NameError instead.
    """

    dbscan: type
    knn: type
    silhouette_score: Callable[..., float]


def _sklearn_backend() -> _SklearnBackend:
    """Import the scikit-learn pieces the PAST4 methods need, lazily.

    Same rationale as ``_kmeans_backend`` above, extended to DBSCAN and
    k-nearest neighbours: scikit-learn is in the ``full`` extra, not the
    base dependencies, so a module-level import would break the base
    install. DBSCAN and KNeighborsClassifier live in
    ``sklearn.cluster`` and ``sklearn.neighbors`` respectively.

    K-medoids deliberately does *not* come through here. It is
    implemented on numpy and scipy alone (see ``_pam``), because
    scikit-learn's KMedoids was deprecated in 1.3 and then removed, and
    the version floor in pyproject.toml (>=1.3.0) does not guarantee
    it exists in the environment actually running the code.
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError as exc:
        raise ComputationError(
            "This clustering method requires scikit-learn. Install the full extra: "
            "pip install 'PaleoAST[full]', or pip install scikit-learn",
            original_exception=exc,
        ) from exc
    return _SklearnBackend(dbscan=DBSCAN, knn=KNeighborsClassifier, silhouette_score=silhouette_score)


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


def _total_cost(dm: npt.NDArray, medoids: list[int]) -> float:
    """PAM objective: summed distance from every sample to its medoid.

    This is the quantity BUILD and SWAP both minimise, so it is defined
    once here rather than twice inside the two steps.
    """
    idx = np.asarray(medoids, dtype=int)
    return float(dm[:, idx].min(axis=1).sum())


def _pam(dm: npt.NDArray, n_clusters: int, max_iter: int = 100) -> tuple[list[int], float, int]:
    """Partitioning Around Medoids: BUILD then SWAP.

    Implemented here rather than taken from scikit-learn, because
    scikit-learn's ``KMedoids`` was deprecated in 1.3 and removed
    afterwards, and this project's floor (>=1.3.0) does not guarantee the
    class is present. The steps below are the original Kaufman &
    Rousseeuw formulation, not a k-means substitute:

    * **BUILD** grows the medoid set one point at a time, each time
      taking the point that most reduces the total cost above.
    * **SWAP** then repeatedly tries every (medoid, non-medoid)
      exchange and applies the one with the largest cost reduction,
      stopping when no exchange improves the objective or ``max_iter``
      iterations have run.

    A k-means-like "assign, then replace a centre by a mean" loop is
    *not* what this does: the centres here are always rows of ``dm``.

    Tie-breaking is by lowest index in both steps, so the result is
    deterministic and needs no random seed -- unlike k-means, which is
    why the k-means result reports one and this one does not.

    Returns:
        The medoid row indices, the final total cost, and the number of
        SWAP iterations actually performed.
    """
    n = dm.shape[0]

    # BUILD: the first medoid is the point closest to everything in
    # total, which is the PAM initialisation rather than an arbitrary one.
    medoids = [int(np.argmin(dm.sum(axis=1)))]
    while len(medoids) < n_clusters:
        current = _total_cost(dm, medoids)
        best_cost = current
        best_candidate: int | None = None
        for candidate in range(n):
            if candidate in medoids:
                continue
            cost = _total_cost(dm, [*medoids, candidate])
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate
        if best_candidate is None:
            # No candidate improved the cost: the remaining points are
            # duplicates of existing medoids. Fill with unused indices so
            # the caller still gets exactly n_clusters *distinct* medoids.
            # Distinctness is part of what k-medoids means, so it is not
            # left to chance.
            for candidate in range(n):
                if candidate not in medoids:
                    medoids.append(candidate)
                if len(medoids) == n_clusters:
                    break
            break
        medoids.append(best_candidate)

    # SWAP: hill-climb on single medoid/non-medoid exchanges.
    n_iter = 0
    cost = _total_cost(dm, medoids)
    for _ in range(max_iter):
        best_cost = cost
        best_swap: tuple[int, int] | None = None
        for position, _current in enumerate(medoids):
            for candidate in range(n):
                if candidate in medoids:
                    continue
                trial = list(medoids)
                trial[position] = candidate
                trial_cost = _total_cost(dm, trial)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_swap = (position, candidate)
        if best_swap is None:
            break
        position, candidate = best_swap
        medoids[position] = candidate
        cost = best_cost
        n_iter += 1

    return medoids, cost, n_iter


@dataclass
class KMedoidsResult:
    """
    Container for a k-medoids (PAM) partition.

    Attributes:
        labels: Cluster index (0-based, contiguous from zero) per sample.
        medoid_indices: Row indices of the medoids in the input data.
            Each is an actual observed sample, not an average.
        medoid_coordinates: The medoids themselves, one row per cluster,
            in the same order as ``labels`` values 0..n_clusters-1.
        cost: PAM objective, the summed distance from each sample to
            the medoid of its cluster.
        silhouette: Mean silhouette coefficient, or NaN when undefined.
        n_clusters: Number of clusters found.
        n_iter: SWAP iterations performed (0 means BUILD alone sufficed).
        n_samples: Number of samples clustered.
        metric: Distance metric used.
    """

    labels: npt.NDArray
    medoid_indices: npt.NDArray
    medoid_coordinates: npt.NDArray
    cost: float
    silhouette: float
    n_clusters: int
    n_iter: int
    n_samples: int
    metric: str

    def cluster_sizes(self) -> dict[int, int]:
        """Number of samples in each cluster."""
        return {int(label): int(np.sum(self.labels == label)) for label in np.unique(self.labels)}

    def summary(self) -> str:
        lines = [
            _("K-Medoids Clustering (PAM)"),
            "=" * 45,
            f"{_('Clusters')}: {self.n_clusters}",
            f"{_('Samples')}: {self.n_samples}",
            f"{_('Distance metric')}: {self.metric}",
            f"{_('Total distance to medoids')}: {self.cost:.4f}",
            f"{_('Silhouette')}: " + (f"{self.silhouette:.4f}" if np.isfinite(self.silhouette) else _("undefined")),
            f"{_('Swap iterations')}: {self.n_iter}",
        ]
        for label, size in sorted(self.cluster_sizes().items()):
            lines.append(f"  {_('Cluster {0}').format(label + 1)}: {size}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "labels": [int(x) for x in self.labels],
            "medoid_indices": [int(i) for i in self.medoid_indices],
            "medoid_coordinates": [[float(v) for v in row] for row in self.medoid_coordinates],
            "cost": float(self.cost),
            "silhouette": float(self.silhouette),
            "n_clusters": int(self.n_clusters),
            "n_iter": int(self.n_iter),
            "n_samples": int(self.n_samples),
            "metric": self.metric,
            "cluster_sizes": self.cluster_sizes(),
        }


@dataclass
class DBSCANResult:
    """
    Container for a DBSCAN partition.

    Attributes:
        labels: Cluster index per sample, where **-1 marks noise**.
            Unlike every other method here, a point may be left
            unassigned on purpose.
        n_clusters: Number of density-connected clusters found. 0 means
            every sample was noise at this ``eps``.
        n_noise: How many samples were labelled noise.
        core_indices: Indices of the core samples, i.e. those with at
            least ``min_samples`` points within ``eps`` of themselves.
        silhouette: Mean silhouette over the clustered samples only, or
            NaN when undefined. Noise is excluded rather than treated as
            its own class, which would flatter the score.
        eps: Neighbourhood radius used.
        min_samples: Minimum neighbourhood size for a core point,
            **including the point itself**.
        n_samples: Number of samples clustered.
        metric: Distance metric used, or 'precomputed'.
    """

    labels: npt.NDArray
    n_clusters: int
    n_noise: int
    core_indices: npt.NDArray
    silhouette: float
    eps: float
    min_samples: int
    n_samples: int
    metric: str

    @property
    def noise_indices(self) -> npt.NDArray:
        """Indices of the samples DBSCAN refused to cluster."""
        return np.flatnonzero(self.labels == -1)

    def cluster_sizes(self) -> dict[int, int]:
        """Number of samples in each cluster. Noise (-1) is excluded."""
        return {int(label): int(np.sum(self.labels == label)) for label in np.unique(self.labels) if label != -1}

    def summary(self) -> str:
        lines = [
            _("DBSCAN Clustering"),
            "=" * 45,
            f"{_('Clusters')}: {self.n_clusters}",
            f"{_('Samples')}: {self.n_samples}",
            f"{_('Noise points')}: {self.n_noise}",
            f"{_('eps')}: {self.eps:g}",
            f"{_('min_samples')}: {self.min_samples}",
            f"{_('Distance metric')}: {self.metric}",
            f"{_('Core points')}: {len(self.core_indices)}",
            f"{_('Silhouette')}: " + (f"{self.silhouette:.4f}" if np.isfinite(self.silhouette) else _("undefined")),
        ]
        for label, size in sorted(self.cluster_sizes().items()):
            lines.append(f"  {_('Cluster {0}').format(label + 1)}: {size}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "labels": [int(x) for x in self.labels],
            "n_clusters": int(self.n_clusters),
            "n_noise": int(self.n_noise),
            "core_indices": [int(i) for i in self.core_indices],
            "noise_indices": [int(i) for i in self.noise_indices],
            "silhouette": float(self.silhouette),
            "eps": float(self.eps),
            "min_samples": int(self.min_samples),
            "n_samples": int(self.n_samples),
            "metric": self.metric,
            "cluster_sizes": self.cluster_sizes(),
        }


@dataclass
class KNNResult:
    """
    Container for a k-nearest neighbours classification.

    Attributes:
        predictions: Predicted class per query sample, as plain Python
            values so that string class labels survive intact.
        accuracy: Fraction of query samples classified correctly, or NaN
            when the caller supplied no known labels to score against.
        confusion: Rows are true classes, columns predicted, ordered by
            ``classes``. Empty when no labels were supplied.
        classes: The class labels seen, sorted.
        neighbor_counts: How many of each sample's neighbours carried
            each class. Same shape as ``confusion``.
        n_neighbors: k used.
        n_train: Training samples the classifier was fitted on.
        n_queries: Samples classified.
        metric: Distance metric used.
        weights: Neighbour weighting, 'uniform' or 'distance'.
    """

    predictions: list
    accuracy: float
    confusion: npt.NDArray
    classes: list
    neighbor_counts: npt.NDArray
    n_neighbors: int
    n_train: int
    n_queries: int
    metric: str
    weights: str

    def summary(self) -> str:
        lines = [
            _("K-Nearest Neighbours Classification"),
            "=" * 45,
            f"{_('Neighbours (k)')}: {self.n_neighbors}",
            f"{_('Training samples')}: {self.n_train}",
            f"{_('Classified samples')}: {self.n_queries}",
            f"{_('Distance metric')}: {self.metric}",
            f"{_('Neighbour weights')}: {self.weights}",
            f"{_('Accuracy')}: " + (f"{self.accuracy:.4f}" if np.isfinite(self.accuracy) else _("not scored")),
        ]
        if not self.confusion.size:
            # Nothing to tabulate: the caller supplied no labels, so there
            # is no confusion matrix and no per-class accuracy to report.
            return "\n".join(lines)
        for i, true_class in enumerate(self.classes):
            total = int(self.confusion[i].sum())
            if total == 0:
                # A class with no queries here -- e.g. one that only
                # appears among the training labels. Reporting "0/0" would
                # read as a measurement of nothing.
                continue
            lines.append(f"  {true_class!s}: {int(self.confusion[i, i])}/{total}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "predictions": [_jsonable(p) for p in self.predictions],
            "accuracy": float(self.accuracy),
            "confusion": [[int(v) for v in row] for row in self.confusion],
            "classes": [_jsonable(c) for c in self.classes],
            "neighbor_counts": [[int(v) for v in row] for row in self.neighbor_counts],
            "n_neighbors": int(self.n_neighbors),
            "n_train": int(self.n_train),
            "n_queries": int(self.n_queries),
            "metric": self.metric,
            "weights": self.weights,
        }


def _jsonable(value):
    """Reduce a numpy scalar to a plain Python value for JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class ClusteringAnalyzer:
    """Clustering engine."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.ClusteringAnalyzer")
        self._lock = threading.RLock()
        self._last_result: ClusteringResult | None = None
        self._last_kmeans: KMeansResult | None = None
        self._last_kmedoids: KMedoidsResult | None = None
        self._last_dbscan: DBSCANResult | None = None
        self._last_knn: KNNResult | None = None

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

    def _resolve_distances(
        self,
        data: npt.NDArray,
        metric: str,
        precomputed: bool,
        context: str,
    ) -> npt.NDArray:
        """Return a square distance matrix, from data or from input.

        PAST accepts a distance matrix wherever it accepts an observation
        matrix, and the rest of this project does the same (PERMANOVA,
        PCOA), so k-medoids and DBSCAN do too. The checks mirror the ones
        ``analyze`` already applies, so the failure modes read the same
        whichever entry point the caller used.
        """
        if not precomputed:
            return squareform(pdist(data, metric=metric))
        if data.shape[0] != data.shape[1]:
            raise MatrixDimensionError("Precomputed distance matrix must be square")
        if not self._is_distance_matrix(data):
            raise MatrixDimensionError("Precomputed distance matrix must be symmetric with a zero diagonal")
        dm = np.asarray(data, dtype=float)
        if not np.all(np.isfinite(dm)):
            raise ComputationError(
                f"{context} cannot use a distance matrix containing NaN or Inf. Check it for missing values first."
            )
        if np.any(dm < 0):
            raise ComputationError("Precomputed distance matrix contains negative entries, which are not distances.")
        return dm

    def analyze_kmedoids(
        self,
        data: npt.NDArray,
        n_clusters: int = 2,
        metric: str = "euclidean",
        precomputed: bool = False,
        max_iter: int = 100,
        compute_silhouette: bool = True,
    ) -> KMedoidsResult:
        """
        Partition the data into k clusters with k-medoids (PAM).

        Each cluster centre is an actual observed sample, so unlike
        k-means nothing here is an average of specimens. That is the
        whole reason to prefer it: on morphometric or taxonomic data a
        centroid may sit in the middle of empty space, or between two
        species, and a centre that no real specimen occupies is harder to
        defend in a paper than one that can be pointed at.

        The fit is genuine PAM -- BUILD then SWAP, see ``_pam`` -- and
        needs no random seed, because both steps break ties by lowest
        index and the result is therefore reproducible.

        Parameters:
            data: Data matrix (n_samples x n_variables) or precomputed
                distance matrix when ``precomputed`` is True
            n_clusters: Number of clusters k
            metric: Distance metric (ignored for a precomputed matrix)
            precomputed: Treat ``data`` as a distance matrix
            max_iter: Cap on SWAP iterations
            compute_silhouette: Report the silhouette coefficient

        Returns:
            KMedoidsResult
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
                # Every sample its own medoid: cost 0, and a "cluster" of
                # one specimen says nothing. Same reasoning as k-means.
                raise ValidationError(
                    f"n_clusters ({n_clusters}) must be smaller than the number of samples ({n_samples})",
                    details={"n_clusters": n_clusters, "n_samples": n_samples},
                )
            if max_iter < 1:
                raise ValidationError(
                    f"max_iter must be at least 1, got {max_iter}",
                    details={"max_iter": max_iter},
                )

            effective_metric = "precomputed" if precomputed else metric
            dm = self._resolve_distances(X, metric, precomputed, "K-medoids")

            # PAM cannot tell a real partition from noise here: with every
            # distance zero the cost is zero for any choice of medoids and
            # the answer would be an artefact of index order.
            if np.allclose(dm, 0):
                raise ComputationError(
                    "K-medoids cannot partition data with no variation: every sample is identical to every other."
                )

            medoids, cost, n_iter = _pam(dm, n_clusters, max_iter=max_iter)
            medoid_idx = np.asarray(medoids, dtype=int)

            # Each sample joins its closest medoid, so cluster i is exactly
            # the one the i-th medoid owns. That keeps labels and
            # medoid_coordinates aligned row for row. With duplicate
            # medoid rows a cluster can come out empty (two identical
            # medoids tie, and argmin keeps only the first); that is
            # reported honestly by cluster_sizes() rather than hidden.
            labels = np.argmin(dm[:, medoid_idx], axis=1).astype(int)

            coordinates = dm[medoid_idx] if precomputed else X[medoid_idx]

            silhouette = float("nan")
            if compute_silhouette and 1 < n_clusters < n_samples:
                silhouette_fn = _sklearn_backend().silhouette_score
                try:
                    silhouette = float(
                        silhouette_fn(dm, labels) if precomputed else silhouette_fn(X, labels, metric=metric)
                    )
                except ValueError:
                    silhouette = float("nan")

            result = KMedoidsResult(
                labels=labels,
                medoid_indices=medoid_idx,
                medoid_coordinates=np.asarray(coordinates, dtype=float),
                cost=float(cost),
                silhouette=silhouette,
                n_clusters=n_clusters,
                n_iter=int(n_iter),
                n_samples=n_samples,
                metric=effective_metric,
            )
            self._last_kmedoids = result
            self._logger.info(
                "K-medoids complete: k=%d, cost=%.4f, %d swap iterations",
                n_clusters,
                result.cost,
                result.n_iter,
            )
            return result

    def analyze_dbscan(
        self,
        data: npt.NDArray,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        precomputed: bool = False,
        compute_silhouette: bool = True,
    ) -> DBSCANResult:
        """
        Cluster by density with DBSCAN.

        The difference from k-means and k-medoids is that neither needs
        k, and that a point can be left unassigned: samples in no dense
        region come back as noise (-1) instead of being forced into
        whichever cluster is least wrong. On fossil data that is usually
        the point -- an odd specimen may genuinely not belong to any
        group -- and PAST reports it as its own category for that reason.

        ``min_samples`` counts the point itself, so the smallest possible
        core point has ``min_samples=1`` (its own zero distance) and the
        default of 5 means "this point plus four others within ``eps``".
        ``eps`` is in the units of the data, which is the parameter to
        vary when the answer looks wrong: too small and everything is
        noise, too large and everything is one cluster.

        Parameters:
            data: Data matrix (n_samples x n_variables) or precomputed
                distance matrix when ``precomputed`` is True
            eps: Neighbourhood radius
            min_samples: Points within ``eps`` (including the point
                itself) required for a core point
            metric: Distance metric (ignored for a precomputed matrix)
            precomputed: Treat ``data`` as a distance matrix
            compute_silhouette: Report the silhouette over clustered
                samples only

        Returns:
            DBSCANResult
        """
        with self._lock:
            X = validate_data_array(data, name="data")
            n_samples = X.shape[0]

            if eps <= 0:
                raise ValidationError(
                    f"eps must be greater than 0, got {eps}. A radius of 0 or less cannot define a neighbourhood.",
                    details={"eps": eps},
                )
            if min_samples < 1:
                raise ValidationError(
                    f"min_samples must be at least 1, got {min_samples}. The value counts the point itself.",
                    details={"min_samples": min_samples},
                )
            if n_samples < 2:
                # DBSCAN cannot grow a cluster from a single sample: with
                # one point every sample is its own only neighbour, so the
                # result would always be 1 noise point.
                raise ValidationError(
                    f"DBSCAN needs at least 2 samples, got {n_samples}",
                    details={"n_samples": n_samples},
                )

            effective_metric = "precomputed" if precomputed else metric
            dbscan_cls = _sklearn_backend().dbscan

            # sklearn consumes the condensed form when given raw data and
            # the square form when given 'precomputed'; feeding it the
            # wrong one silently changes the neighbourhoods, so the
            # matrix is resolved here rather than passed straight through.
            X_in = self._resolve_distances(X, metric, precomputed, "DBSCAN") if precomputed else X
            model = dbscan_cls(eps=float(eps), min_samples=int(min_samples), metric=effective_metric)
            labels = np.asarray(model.fit_predict(X_in), dtype=int)
            core_indices = np.asarray(model.core_sample_indices_, dtype=int)

            n_clusters = len(set(labels.tolist()) - {-1})
            n_noise = int(np.sum(labels == -1))

            silhouette = float("nan")
            clustered = labels != -1
            # Silhouette is undefined with one cluster, and mixing noise
            # in as a class of its own would reward an arbitrary radius,
            # so it is measured over the clustered samples only.
            if compute_silhouette and n_clusters > 1 and n_noise < n_samples:
                silhouette_fn = _sklearn_backend().silhouette_score
                try:
                    silhouette = float(
                        silhouette_fn(X_in[clustered], labels[clustered])
                        if precomputed
                        else silhouette_fn(X[clustered], labels[clustered], metric=metric)
                    )
                except ValueError:
                    silhouette = float("nan")

            result = DBSCANResult(
                labels=labels,
                n_clusters=n_clusters,
                n_noise=n_noise,
                core_indices=core_indices,
                silhouette=silhouette,
                eps=float(eps),
                min_samples=int(min_samples),
                n_samples=n_samples,
                metric=effective_metric,
            )
            self._last_dbscan = result
            self._logger.info(
                "DBSCAN complete: %d clusters, %d/%d noise at eps=%g",
                n_clusters,
                n_noise,
                n_samples,
                eps,
            )
            return result

    def classify_knn(
        self,
        train_data: npt.NDArray,
        train_labels,
        query_data: npt.NDArray | None = None,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
        query_labels=None,
    ) -> KNNResult:
        """
        Classify samples by their k nearest labelled neighbours.

        No model is fitted beyond storing the training set: a query is
        assigned the majority class among its k nearest training samples.
        That makes it the right tool for small, well-labelled reference
        sets -- assigning a handful of new specimens to known species --
        and the wrong tool for many variables or a weak class boundary,
        where every dimension counts equally and a single outlier
        variable can decide the answer.

        Scoring is optional. Passing ``query_labels`` computes accuracy
        and a confusion matrix; omitting it leaves accuracy NaN rather
        than inventing a score from the training fit. Note that
        classifying the training rows themselves is in-sample -- a point
        is always its own nearest neighbour -- so the accuracy would be
        1.0 by construction. Score on held-out data.

        Parameters:
            train_data: Labelled reference matrix (n_train x n_variables)
            train_labels: Class of each training sample; numeric or string
            query_data: Rows to classify. Defaults to ``train_data``,
                which is in-sample and scores optimistically.
            n_neighbors: k, must be at least 1 and no larger than the
                number of training samples
            metric: Distance metric
            weights: 'uniform' (each neighbour one vote) or 'distance'
                (closer neighbours count more)
            query_labels: True classes for the queries, for scoring

        Returns:
            KNNResult
        """
        with self._lock:
            X_train = validate_data_array(train_data, name="train_data")
            n_train = X_train.shape[0]

            if n_train < 2:
                raise ValidationError(
                    f"k-nearest neighbours needs at least 2 training samples, got {n_train}",
                    details={"n_train": n_train},
                )
            y_train = np.asarray(train_labels)
            if y_train.ndim != 1 or len(y_train) != n_train:
                raise ValidationError(
                    f"train_labels must have one entry per training sample: expected {n_train}, got {len(y_train)}",
                    details={"n_train": n_train, "n_labels": len(y_train)},
                )
            if n_neighbors < 1:
                raise ValidationError(
                    f"n_neighbors must be at least 1, got {n_neighbors}",
                    details={"n_neighbors": n_neighbors},
                )
            if n_neighbors > n_train:
                raise ValidationError(
                    f"n_neighbors ({n_neighbors}) cannot exceed the number of training samples ({n_train})",
                    details={"n_neighbors": n_neighbors, "n_train": n_train},
                )
            if weights not in ("uniform", "distance"):
                raise ValidationError(
                    f"weights must be 'uniform' or 'distance', got {weights!r}",
                    details={"weights": weights},
                )

            if query_data is None:
                X_query = X_train
            else:
                X_query = validate_data_array(query_data, name="query_data")
                if X_query.shape[1] != X_train.shape[1]:
                    raise MatrixDimensionError(
                        f"query_data has {X_query.shape[1]} variables but train_data has {X_train.shape[1]}",
                        details={
                            "expected": f"(n, {X_train.shape[1]})",
                            "actual": str(X_query.shape),
                            "operation": "k-nearest neighbours classification",
                        },
                    )

            knn_cls = _sklearn_backend().knn
            model = knn_cls(n_neighbors=int(n_neighbors), metric=metric, weights=weights)
            model.fit(X_train, y_train)

            predictions = model.predict(X_query)
            n_queries = len(predictions)

            accuracy = float("nan")
            confusion = np.zeros((0, 0), dtype=int)
            neighbor_counts = np.zeros((0, 0), dtype=int)
            if query_labels is not None:
                y_query = np.asarray(query_labels)
                if len(y_query) != n_queries:
                    raise ValidationError(
                        f"query_labels must have one entry per query: expected {n_queries}, got {len(y_query)}",
                        details={"n_queries": n_queries, "n_labels": len(y_query)},
                    )
                accuracy = float(np.mean(predictions == y_query))
                classes = np.unique(np.concatenate([y_train, y_query]))
                confusion = np.zeros((len(classes), len(classes)), dtype=int)
                for true, pred in zip(y_query, predictions, strict=True):
                    i = int(np.searchsorted(classes, true))
                    j = int(np.searchsorted(classes, pred))
                    confusion[i, j] += 1
            else:
                classes = np.unique(y_train)

            # Per-class neighbour votes, laid out on the same ``classes``
            # axis as the confusion matrix. Counted from the actual
            # neighbour indices rather than read back out of
            # predict_proba, because under weights='distance' the
            # probabilities are distance-weighted and multiplying them by
            # k would not recover vote counts.
            neighbor_counts = np.zeros((n_queries, len(classes)), dtype=int)
            _, neighbor_idx = model.kneighbors(X_query, n_neighbors=int(n_neighbors))
            for row, indices in enumerate(neighbor_idx):
                for neighbor in indices:
                    class_pos = int(np.searchsorted(classes, y_train[neighbor]))
                    neighbor_counts[row, class_pos] += 1

            result = KNNResult(
                predictions=list(predictions),
                accuracy=accuracy,
                confusion=confusion,
                classes=[_jsonable(c) for c in classes.tolist()],
                neighbor_counts=neighbor_counts,
                n_neighbors=int(n_neighbors),
                n_train=n_train,
                n_queries=n_queries,
                metric=metric,
                weights=weights,
            )
            self._last_knn = result
            self._logger.info(
                "k-NN classified %d samples using k=%d, accuracy=%s",
                n_queries,
                n_neighbors,
                "n/a" if not np.isfinite(accuracy) else f"{accuracy:.4f}",
            )
            return result

    @property
    def last_result(self) -> ClusteringResult | None:
        with self._lock:
            return self._last_result

    @property
    def last_kmeans_result(self) -> KMeansResult | None:
        """Most recent k-means partition, or None."""
        with self._lock:
            return self._last_kmeans

    @property
    def last_kmedoids_result(self) -> KMedoidsResult | None:
        """Most recent k-medoids (PAM) partition, or None."""
        with self._lock:
            return self._last_kmedoids

    @property
    def last_dbscan_result(self) -> DBSCANResult | None:
        """Most recent DBSCAN partition, or None."""
        with self._lock:
            return self._last_dbscan

    @property
    def last_knn_result(self) -> KNNResult | None:
        """Most recent k-nearest neighbours classification, or None."""
        with self._lock:
            return self._last_knn
