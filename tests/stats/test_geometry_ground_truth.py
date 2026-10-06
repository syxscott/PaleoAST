"""
Ground truth for ``stats/geometry.py`` (27.2% covered).

Convex hull hypervolume, minimum spanning tree weight and pairwise distances
all have exact answers on small inputs, so a wrong implementation is
decidable rather than merely suspicious.

One defect was found this way: ``convex_hull_volume`` returned ``inf`` for
fewer than ``n_dims + 1`` points while raising ``ComputationError`` for a
coplanar cloud. Two different answers to the same question, and neither was
right -- an infinite morphospace is not a measurement, and it feeds straight
into every disparity statistic computed from it. Degenerate input is now
refused on every path.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull as SciPyHull

from stats.geometry import GeometryAnalyzer
from utils.exceptions import ComputationError

SEED = 11


@pytest.fixture(scope="module")
def analyzer() -> GeometryAnalyzer:
    return GeometryAnalyzer()


def _brute_force_prim(points: np.ndarray) -> float:
    """Prim's algorithm, written out, as an independent MST reference."""
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    n = len(points)
    inside = {0}
    total = 0.0
    while len(inside) < n:
        best, pick = np.inf, None
        for a in inside:
            for b in range(n):
                if b not in inside and distance[a, b] < best:
                    best, pick = distance[a, b], b
        inside.add(pick)
        total += best
    return total


class TestConvexHull:
    def test_matches_scipy_on_random_clouds(self, analyzer):
        rng = np.random.default_rng(SEED)
        for _ in range(12):
            points = rng.normal(size=(40, 3))
            assert analyzer.convex_hull_volume(points) == pytest.approx(float(SciPyHull(points).volume), rel=1e-9)

    def test_unit_cube_has_volume_one(self, analyzer):
        cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
        assert analyzer.convex_hull_volume(cube) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        "label,points",
        [
            ("coplanar", np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])),
            ("collinear", np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2]])),
            ("too few", np.array([[0.0, 0, 0], [1, 0, 0]])),
            ("single", np.array([[0.0, 0, 0]])),
        ],
    )
    def test_degenerate_hull_is_refused_not_answered(self, analyzer, label, points):
        """Every degenerate shape takes the same route: a clear refusal.

        Returning ``inf`` (the old behaviour for some shapes) or 0 is a number
        that looks like a measurement. inf in particular says "this morphospace
        is infinitely large", which would be reported as a finding.
        """
        with pytest.raises(ComputationError):
            analyzer.convex_hull_volume(points)

    def test_one_dimensional_input_is_refused(self, analyzer):
        with pytest.raises(ComputationError):
            analyzer.convex_hull_volume(np.arange(10.0).reshape(-1, 1))


class TestMinimumSpanningTree:
    def test_matches_brute_force_prim(self, analyzer):
        rng = np.random.default_rng(SEED)
        points = rng.normal(size=(12, 3))
        tree = analyzer.minimum_spanning_tree(points, [f"t{i}" for i in range(12)])
        total = float(getattr(tree, "total_length", np.sum([e[2] for e in tree.edges])))
        assert total == pytest.approx(_brute_force_prim(points), rel=1e-9)

    def test_collinear_points_span_exactly_their_range(self, analyzer):
        """A closed form: the MST of points on a line is just the span."""
        line = np.array([[float(i), 0.0, 0.0] for i in range(6)])
        tree = analyzer.minimum_spanning_tree(line, [str(i) for i in range(6)])
        total = float(getattr(tree, "total_length", np.sum([e[2] for e in tree.edges])))
        assert total == pytest.approx(5.0, abs=1e-9)

    def test_single_point_has_zero_length(self, analyzer):
        tree = analyzer.minimum_spanning_tree(np.zeros((1, 3)), ["only"])
        total = float(getattr(tree, "total_length", np.sum([e[2] for e in tree.edges], dtype=float)))
        assert total == pytest.approx(0.0, abs=1e-9)


class TestPairwiseDistances:
    def test_matches_the_explicit_computation(self, analyzer):
        rng = np.random.default_rng(SEED)
        points = rng.normal(size=(6, 3))
        expected = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        assert np.allclose(analyzer.pairwise_distances(points), expected, atol=1e-12)

    def test_symmetric_with_zero_diagonal(self, analyzer):
        rng = np.random.default_rng(SEED)
        matrix = analyzer.pairwise_distances(rng.normal(size=(6, 3)))
        assert np.allclose(matrix, matrix.T, atol=1e-12)
        assert np.allclose(np.diag(matrix), 0.0, atol=1e-12)


class TestDisparity:
    def test_every_reported_statistic_is_finite(self, analyzer):
        rng = np.random.default_rng(SEED)
        disparity = analyzer.morphospace_disparity(rng.normal(size=(8, 6)))
        for name in dir(disparity):
            if name.startswith("_"):
                continue
            value = getattr(disparity, name)
            if isinstance(value, (int, float, np.floating)):
                assert np.isfinite(value), f"{name} = {value}"
            elif isinstance(value, np.ndarray) and value.size:
                assert np.all(np.isfinite(value)), f"{name} has NaN"

    def test_range_is_at_least_the_median(self, analyzer):
        """A cheap consistency check between two reported statistics."""
        rng = np.random.default_rng(SEED)
        disparity = analyzer.morphospace_disparity(rng.normal(size=(8, 6)))
        assert disparity.range_val >= disparity.median - 1e-9
