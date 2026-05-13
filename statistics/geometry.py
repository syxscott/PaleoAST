# =============================================================================
# FILE: statistics/geometry.py
# =============================================================================
"""
Geometric Analysis Module for PaleoAST

Provides minimum spanning tree, convex hull, and morphospace
disparity analysis for paleontological data.

Mathematical Foundation:

1. Minimum Spanning Tree (MST):
    The MST of a weighted graph is a spanning tree with
    minimum total edge weight. For points in ℝ^n with
    Euclidean distance, the MST connects all points using
    the shortest possible total distance.

    Prim's Algorithm: O(n²) for dense graphs
    Kruskal's Algorithm: O(m log m) for sparse graphs

2. Convex Hull Volume:
    The convex hull of a point set is the smallest convex
    set containing all points. In ℝ^n, the hypervolume
    of the convex hull measures the morphospace occupation.

    For 2D: Area of polygon
    For 3D: Volume of polyhedron
    For nD: Hypervolume via QHull (scipy.spatial)

3. Morphospace Disparity:
    Disparity measures how spread out specimens are in
    morphospace. Two common metrics:

    Dispersion: Mean pairwise distance
        D = (2/n(n-1)) * Σ_{i<j} d_ij

    Range: Maximum pairwise distance
        R = max_{i<j} d_ij

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, distance_matrix

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class MSTResult:
    """
    Container for Minimum Spanning Tree results.

    Attributes:
        edges: List of (i, j, weight) tuples for MST edges
        total_length: Total weight of MST edges
        n_points: Number of points
        labels: Point labels
    """

    edges: list[tuple[int, int, float]]
    total_length: float
    n_points: int
    labels: list[str]

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Minimum Spanning Tree (MST)')}\n"
            f"{'=' * 45}\n"
            f"{_('Points: {0}').format(self.n_points)}\n"
            f"{_('Edges: {0}').format(len(self.edges))}\n"
            f"{_('Total length: {0:.4f}').format(self.total_length)}"
        )


@dataclass
class DisparityResult:
    """
    Container for morphospace disparity analysis results.

    Attributes:
        dispersion: Mean pairwise distance
        range: Maximum pairwise distance
        variance: Variance of pairwise distances
        median: Median pairwise distance
        n_points: Number of points
        n_dimensions: Number of dimensions
    """

    dispersion: float
    range_val: float
    variance: float
    median: float
    n_points: int
    n_dimensions: int

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Morphospace Disparity Analysis')}\n"
            f"{'=' * 45}\n"
            f"{_('Points: {0}').format(self.n_points)}\n"
            f"{_('Dimensions: {0}').format(self.n_dimensions)}\n"
            f"{_('Dispersion (mean pairwise dist): {0:.4f}').format(self.dispersion)}\n"
            f"{_('Range (max pairwise dist): {0:.4f}').format(self.range_val)}\n"
            f"{_('Variance: {0:.4f}').format(self.variance)}\n"
            f"{_('Median: {0:.4f}').format(self.median)}"
        )


class GeometryAnalyzer:
    """
    Geometric analysis utilities for paleontological data.

    Provides MST, convex hull, and disparity analysis
    for morphometric and spatial data.
    """

    def __init__(self) -> None:
        """Initialize the geometry analyzer."""
        self._logger = logging.getLogger(f"{__name__}.GeometryAnalyzer")
        self._lock = threading.RLock()
        self._last_mst_result: MSTResult | None = None
        self._last_disparity_result: DisparityResult | None = None
        self._logger.info("GeometryAnalyzer initialized")

    def minimum_spanning_tree(
        self,
        points: npt.NDArray,
        labels: list[str] | None = None,
    ) -> MSTResult:
        """
        Compute the Minimum Spanning Tree using Prim's algorithm.

        Parameters:
            points: Point coordinates (n_points, n_dims)
            labels: Optional labels for each point

        Returns:
            MSTResult containing edges and total length

        Mathematical Background:
            Prim's algorithm grows the MST from a starting node:
            1. Initialize tree with single arbitrary node
            2. At each step, add the nearest node not in tree
            3. Repeat until all nodes are included

            Time complexity: O(n²) for dense graphs
        """
        with self._lock:
            pts = validate_data_array(points, allow_nan=False, name="points")

            if pts.ndim != 2:
                raise ComputationError(
                    "Points must be 2D array (n_points, n_dims)"
                )

            n = pts.shape[0]

            if labels is None:
                labels = [f"P{i}" for i in range(n)]
            elif len(labels) != n:
                raise ComputationError(
                    f"Labels length ({len(labels)}) must match n_points ({n})"
                )

            self._logger.info(f"Computing MST for {n} points in {pts.shape[1]}D")

            # Compute pairwise distance matrix
            dist_mat = distance_matrix(pts, pts)

            # Prim's algorithm
            in_tree = np.zeros(n, dtype=bool)
            min_dist = np.full(n, np.inf)
            parent = np.full(n, -1, dtype=int)

            # Start from first node
            min_dist[0] = 0.0
            in_tree[0] = True

            for _ in range(n - 1):
                # Find node with minimum distance to tree
                u = np.argmin(np.where(in_tree, np.inf, min_dist))
                in_tree[u] = True

                # Update distances
                for v in range(n):
                    if not in_tree[v] and dist_mat[u, v] < min_dist[v]:
                        min_dist[v] = dist_mat[u, v]
                        parent[v] = u

            # Build edge list
            edges = []
            total_length = 0.0
            for i in range(1, n):
                if parent[i] >= 0:
                    w = dist_mat[i, parent[i]]
                    edges.append((parent[i], i, float(w)))
                    total_length += w

            result = MSTResult(
                edges=edges,
                total_length=float(total_length),
                n_points=n,
                labels=list(labels),
            )

            self._last_mst_result = result
            self._logger.info(f"MST computed: {len(edges)} edges, total length = {total_length:.4f}")
            return result

    def convex_hull_volume(self, points: npt.NDArray) -> float:
        """
        Compute the hypervolume of the convex hull.

        Parameters:
            points: Point coordinates (n_points, n_dims)

        Returns:
            Hypervolume of convex hull

        Note:
            For n_dims > len(points) - 1, returns inf
            For n_dims = 1, returns range (length)
            For n_dims = 2, returns area
            For n_dims >= 3, returns nD hypervolume via QHull
        """
        pts = validate_data_array(points, allow_nan=False, name="points")

        if pts.ndim != 2:
            raise ComputationError(
                "Points must be 2D array (n_points, n_dims)"
            )

        n, dim = pts.shape

        if n <= dim:
            self._logger.warning(
                f"Cannot compute convex hull: {n} points in {dim}D "
                f"(need n > dims for non-degenerate hull)"
            )
            return np.inf

        try:
            hull = ConvexHull(pts)
            volume = hull.volume
            self._logger.info(f"Convex hull volume: {volume:.4f}")
            return float(volume)
        except Exception as e:
            self._logger.error(f"Convex hull computation failed: {e}")
            raise ComputationError(f"Convex hull computation failed: {e}")

    def morphospace_disparity(self, procrustes_coords: npt.NDArray) -> DisparityResult:
        """
        Compute morphospace disparity metrics.

        Disparity measures how spread out specimens are in
        morphospace, commonly used in macroevolution and
        morphometrics studies.

        Parameters:
            procrustes_coords: Procrustes-aligned coordinates
                              (n_specimens, n_dims) or flattened
                              (n_specimens, n_landmarks * n_dims)

        Returns:
            DisparityResult with dispersion, range, variance, median

        Mathematical Background:
            Dispersion (Foote 1993):
                D = (1/N) * Σ_{i=1}^{N} ||x_i - x̄||²

            Mean Pairwise Distance:
                D_pair = (2/N(N-1)) * Σ_{i<j} ||x_i - x_j||

            Range:
                R = max_{i<j} ||x_i - x_j||

        Reference:
            Foote, M. (1993). Contribution of the fossil record
            to the study of morphological evolution.
            Science, 260, 971-974.
        """
        with self._lock:
            coords = validate_data_array(
                procrustes_coords, allow_nan=False, name="procrustes_coords"
            )

            # Handle flattened input
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            elif coords.ndim > 2:
                n_specs = coords.shape[0]
                coords = coords.reshape(n_specs, -1)

            n = coords.shape[0]
            dim = coords.shape[1]

            if n < 2:
                raise ComputationError(
                    f"Need at least 2 specimens for disparity, got {n}"
                )

            self._logger.info(
                f"Computing morphospace disparity for {n} specimens in {dim}D"
            )

            # Compute pairwise distance matrix
            dist_mat = distance_matrix(coords, coords)

            # Extract upper triangle (i < j)
            i_upper = np.triu_indices(n, k=1)
            distances = dist_mat[i_upper]

            # Compute metrics
            dispersion = float(np.mean(distances))
            range_val = float(np.max(distances))
            variance = float(np.var(distances))
            median = float(np.median(distances))

            result = DisparityResult(
                dispersion=dispersion,
                range_val=range_val,
                variance=variance,
                median=median,
                n_points=n,
                n_dimensions=dim,
            )

            self._last_disparity_result = result
            self._logger.info(
                f"Disparity computed: dispersion={dispersion:.4f}, range={range_val:.4f}"
            )
            return result

    def pairwise_distances(
        self,
        points: npt.NDArray,
        metric: str = "euclidean",
    ) -> npt.NDArray:
        """
        Compute pairwise distance matrix.

        Parameters:
            points: Point coordinates (n_points, n_dims)
            metric: Distance metric ('euclidean', 'cityblock', 'cosine')

        Returns:
            Distance matrix (n_points, n_points)
        """
        pts = validate_data_array(points, allow_nan=False, name="points")

        if pts.ndim != 2:
            raise ComputationError(
                "Points must be 2D array (n_points, n_dims)"
            )

        self._logger.info(
            f"Computing {metric} pairwise distances for {pts.shape[0]} points"
        )

        from scipy.spatial.distance import cdist

        dist_mat = cdist(pts, pts, metric=metric)

        return dist_mat

    @property
    def last_mst_result(self) -> MSTResult | None:
        """Get the last MST result."""
        with self._lock:
            return self._last_mst_result

    @property
    def last_disparity_result(self) -> DisparityResult | None:
        """Get the last disparity result."""
        with self._lock:
            return self._last_disparity_result
