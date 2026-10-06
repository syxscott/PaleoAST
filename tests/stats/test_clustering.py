# =============================================================================
# FILE: tests/stats/test_clustering.py
# =============================================================================
"""
Tests for hierarchical and k-means clustering.

stats/clustering.py had no test file at all, so the hierarchical path is
covered here too rather than only the new one.

The k-means assertions are built on separated blobs, where the correct
answer is known rather than merely self-consistent: three clusters far
apart in a 2-D plane must come back as three clusters with one blob each,
so a solver that mixed two blobs together, or split one, fails instead of
returning a different-but-plausible partition.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from stats.clustering import ClusteringAnalyzer
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError


@pytest.fixture
def analyzer() -> ClusteringAnalyzer:
    """A fresh analyzer."""
    return ClusteringAnalyzer()


def three_blobs(
    centers: npt.NDArray = None,  # type: ignore[assignment]
    per_blob: int = 20,
    spread: float = 0.35,
    seed: int = 0,
) -> npt.NDArray:
    """Three well-separated Gaussian blobs, one true cluster each."""
    if centers is None:
        centers = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    rng = np.random.default_rng(seed)
    return np.vstack([c + rng.normal(0, spread, (per_blob, centers.shape[1])) for c in centers])


# ---------------------------------------------------------------------------
# K-means on data with a known answer
# ---------------------------------------------------------------------------


def test_kmeans_recovers_separated_blobs(analyzer: ClusteringAnalyzer) -> None:
    """Three far-apart blobs must come back as three clusters, one each."""
    result = analyzer.analyze_kmeans(three_blobs(), n_clusters=3, random_seed=0, n_init=10)
    assert result.n_clusters == 3
    assert sorted(result.cluster_sizes().values()) == [20, 20, 20]
    # Each blob internally tight, so the silhouette is close to its maximum.
    assert result.silhouette > 0.9
    # Centroids should land on the blob centres. Matched by nearest centre
    # rather than by sorting a column: the true centres are
    # [[0,0],[8,0],[0,8]], whose order after a lexicographic sort is
    # [[0,0],[0,8],[8,0]], not the order they were written in. The noise
    # is sd=0.35 over 20 points, so a centroid's own sd is ~0.078; 0.5 is
    # a wide margin that still fails if a blob was split or merged.
    true_centers = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    for center in true_centers:
        nearest = result.centroids[np.argmin(np.linalg.norm(result.centroids - center, axis=1))]
        assert np.allclose(nearest, center, atol=0.5)


def test_kmeans_labels_are_contiguous_from_zero(analyzer: ClusteringAnalyzer) -> None:
    """Labels are 0-based and every index 0..k-1 is used.

    sklearn's own labelling is 0-based, but a wrapper that renumbered
    would still produce a valid-looking result; checking the exact set
    pins the convention down.
    """
    result = analyzer.analyze_kmeans(three_blobs(), n_clusters=3, random_seed=0)
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert result.labels.min() == 0
    assert result.labels.max() == 2


def test_kmeans_wrong_k_is_visible_in_inertia(analyzer: ClusteringAnalyzer) -> None:
    """Over-clustering inflates the silhouette and shrinks inertia.

    This is the failure mode a user actually hits -- asking for 8
    clusters on 3 blobs and seeing a high score -- so the test records
    it rather than pretending it cannot happen.
    """
    data = three_blobs()
    good = analyzer.analyze_kmeans(data, n_clusters=3, random_seed=0)
    over = analyzer.analyze_kmeans(data, n_clusters=8, random_seed=0)
    assert over.inertia < good.inertia
    # Over-splitting a blob produces near-duplicate centroids, which the
    # silhouette penalises: the two points in a pair are each other's
    # nearest neighbour with distance ~0, so s(a) ~ 0 for both.
    assert over.silhouette < good.silhouette


def test_kmeans_is_deterministic_under_a_seed(analyzer: ClusteringAnalyzer) -> None:
    """Same seed, same labels, same inertia."""
    data = three_blobs(seed=1)
    first = analyzer.analyze_kmeans(data, n_clusters=3, random_seed=7)
    second = analyzer.analyze_kmeans(data, n_clusters=3, random_seed=7)
    assert np.array_equal(first.labels, second.labels)
    assert first.inertia == pytest.approx(second.inertia, abs=1e-9)
    assert first.seed == second.seed


def test_kmeans_reports_the_seed_it_used(analyzer: ClusteringAnalyzer) -> None:
    """An unseeded run still reports a seed, so it can be replayed."""
    data = three_blobs(seed=2)
    result = analyzer.analyze_kmeans(data, n_clusters=3)
    assert isinstance(result.seed, int)
    replay = analyzer.analyze_kmeans(data, n_clusters=3, random_seed=result.seed)
    assert np.array_equal(result.labels, replay.labels)


def test_kmeans_does_not_touch_the_global_random_stream(
    analyzer: ClusteringAnalyzer,
) -> None:
    """An unseeded run must not advance the caller's random state."""
    data = three_blobs(seed=3)
    np.random.seed(4242)
    control = np.random.rand(4)
    np.random.seed(4242)
    with pytest.warns(RuntimeWarning):
        analyzer.analyze_kmeans(data, n_clusters=3)
    np.testing.assert_allclose(np.random.rand(4), control)


# ---------------------------------------------------------------------------
# Elbow curve
# ---------------------------------------------------------------------------


def test_elbow_curve_is_monotone_and_finds_three_blobs(
    analyzer: ClusteringAnalyzer,
) -> None:
    """WCSS falls with k, and the elbow lands near the true 3."""
    curve = analyzer.elbow_curve(three_blobs(), k_min=1, k_max=8, random_seed=0)
    assert list(curve.k_values) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert np.all(np.diff(curve.inertia) < 0)
    assert curve.recommended_k == 3
    # The drop into k=3 is far larger than the drop after it.
    assert curve.inertia[1] - curve.inertia[2] > 5 * (curve.inertia[2] - curve.inertia[3])


def test_elbow_curve_is_deterministic(analyzer: ClusteringAnalyzer) -> None:
    """Same seed, same curve."""
    data = three_blobs(seed=4)
    a = analyzer.elbow_curve(data, k_min=2, k_max=6, random_seed=5)
    b = analyzer.elbow_curve(data, k_min=2, k_max=6, random_seed=5)
    np.testing.assert_allclose(a.inertia, b.inertia)


def test_elbow_curve_serialises(analyzer: ClusteringAnalyzer) -> None:
    """to_dict and summary stay JSON-friendly."""
    curve = analyzer.elbow_curve(three_blobs(), k_min=2, k_max=5, random_seed=0)
    payload = curve.to_dict()
    assert payload["k_values"] == [2, 3, 4, 5]
    assert len(payload["inertia"]) == 4
    assert "Elbow" in curve.summary()


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


def test_k_at_or_above_sample_count_is_rejected(
    analyzer: ClusteringAnalyzer,
) -> None:
    """k = n means one point per cluster: inertia 0, silhouette undefined."""
    data = np.random.default_rng(0).normal(size=(5, 3))
    with pytest.raises(ValidationError):
        analyzer.analyze_kmeans(data, n_clusters=5, random_seed=1)
    with pytest.raises(ValidationError):
        analyzer.analyze_kmeans(data, n_clusters=9, random_seed=1)


def test_non_positive_k_is_rejected(analyzer: ClusteringAnalyzer) -> None:
    """k must be at least 1."""
    data = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValidationError):
        analyzer.analyze_kmeans(data, n_clusters=0, random_seed=1)


def test_constant_data_is_rejected(analyzer: ClusteringAnalyzer) -> None:
    """No variation means no partition worth reporting."""
    with pytest.raises((ComputationError, ValidationError)):
        analyzer.analyze_kmeans(np.ones((12, 3)), n_clusters=2, random_seed=1)


def test_non_finite_data_is_rejected(analyzer: ClusteringAnalyzer) -> None:
    """NaN has to be dealt with before k-means, not inside it."""
    data = three_blobs()
    data[3, 0] = np.nan
    with pytest.raises((ComputationError, ValidationError)):
        analyzer.analyze_kmeans(data, n_clusters=3, random_seed=1)


def test_bad_elbow_range_is_rejected(analyzer: ClusteringAnalyzer) -> None:
    """k_min must be positive and not exceed k_max."""
    data = three_blobs()
    with pytest.raises(ValidationError):
        analyzer.elbow_curve(data, k_min=0, k_max=5)
    with pytest.raises(ValidationError):
        analyzer.elbow_curve(data, k_min=5, k_max=2)
    with pytest.raises(ValidationError):
        analyzer.elbow_curve(data, k_min=2, k_max=500)


# ---------------------------------------------------------------------------
# Results and the pre-existing hierarchical path
# ---------------------------------------------------------------------------


def test_kmeans_result_serialises(analyzer: ClusteringAnalyzer) -> None:
    """to_dict and summary both work."""
    result = analyzer.analyze_kmeans(three_blobs(), n_clusters=3, random_seed=0)
    payload = result.to_dict()
    assert payload["n_clusters"] == 3
    assert len(payload["labels"]) == 60
    assert sum(payload["cluster_sizes"].values()) == 60
    assert "K-Means" in result.summary()


def test_last_kmeans_result_is_retained(analyzer: ClusteringAnalyzer) -> None:
    """The analyzer remembers its most recent k-means run."""
    assert analyzer.last_kmeans_result is None
    result = analyzer.analyze_kmeans(three_blobs(), n_clusters=3, random_seed=0)
    assert analyzer.last_kmeans_result is result


def test_hierarchical_still_works(analyzer: ClusteringAnalyzer) -> None:
    """The pre-existing path is untouched by the k-means addition."""
    result = analyzer.analyze(three_blobs(), method="ward", n_clusters=3)
    assert result.n_clusters == 3
    assert result.linkage_matrix.shape == (len(np.unique(three_blobs(), axis=0)) - 1, 4)
    assert -1.0 <= result.cophenetic_corr <= 1.0
    assert "Hierarchical Clustering" in result.summary()


def test_hierarchical_rejects_a_non_square_precomputed_matrix(
    analyzer: ClusteringAnalyzer,
) -> None:
    """A precomputed distance matrix must be square."""
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze(np.zeros((5, 4)), precomputed=True)


def test_hierarchical_overrides_ward_to_euclidean(
    analyzer: ClusteringAnalyzer,
) -> None:
    """Ward linkage is only defined for Euclidean distance."""
    result = analyzer.analyze(three_blobs(), method="ward", metric="manhattan")
    assert result.metric == "euclidean"


def test_kmeans_does_not_break_import_without_scikit_learn() -> None:
    """stats.clustering must import on a base install.

    scikit-learn is in the ``full`` extra, not in the base dependencies,
    so a module-level import made every package that reaches this one --
    ``controllers``, and therefore ``views`` -- fail to import with only
    the base dependencies installed. The Wheel Build & Import Smoke job
    installs base dependencies only, so CI caught this and a local
    environment with the full extra set could not.
    """
    import builtins
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name == 'sklearn' or name.startswith('sklearn.'):\n"
        "        raise ImportError(name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import stats.clustering as C\n"
        "import numpy as np\n"
        "r = C.ClusteringAnalyzer().analyze("
        "np.random.default_rng(0).normal(size=(12, 3)), n_clusters=2)\n"
        "assert r.n_clusters == 2, 'hierarchical path must not need sklearn'\n"
        "try:\n"
        "    C.ClusteringAnalyzer().analyze_kmeans(\n"
        "        np.random.default_rng(0).normal(size=(12, 3)), n_clusters=2)\n"
        "except Exception as exc:\n"
        "    assert 'scikit-learn' in str(exc), str(exc)\n"
        "    print('OK')\n"
        "else:\n"
        "    raise AssertionError('k-means should have refused')\n"
    )
    root = Path(__file__).resolve().parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout
    assert builtins is not None  # keep the import referenced for linters
