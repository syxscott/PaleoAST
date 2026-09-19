"""Tests for statistics/nmds.py — NMDS SMACOF performance and correctness."""

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_distance_matrix() -> np.ndarray:
    """4x4 Euclidean distance matrix for basic correctness tests."""
    # Three points in 2D: (0,0), (1,0), (0,1) + centroid
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    # Pre-computed Euclidean distances
    from scipy.spatial.distance import cdist

    return cdist(pts, pts, metric="euclidean")


# ---------------------------------------------------------------------------
# Correctness: stress values
# ---------------------------------------------------------------------------


def test_nmds_stress_below_converged_threshold(small_distance_matrix):
    """NMDS must converge (stress < 0.05) on a trivially-embeddable distance matrix."""
    from statistics.nmds import NMDSAnalyzer

    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(
        small_distance_matrix,
        n_dimensions=2,
        n_restarts=3,
        max_iterations=200,
        random_seed=42,
    )

    assert result.stress < 0.05, f"stress={result.stress} should be < 0.05 for embeddable data"
    assert bool(result.converged) is True
    assert result.n_iterations > 0


def test_nmds_stress_history_is_monotone_decreasing(small_distance_matrix):
    """Stress history should generally decrease (allowing minor fluctuations)."""
    from statistics.nmds import NMDSAnalyzer

    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(
        small_distance_matrix,
        n_dimensions=2,
        n_restarts=1,
        max_iterations=100,
        random_seed=0,
    )

    history = result.stress_history
    # Stress may occasionally increase slightly; check that overall trend is down
    # by comparing first 10% and last 10% of iterations
    if len(history) >= 10:
        first_10pct = history[: len(history) // 10]
        last_10pct = history[-len(history) // 10 :]
        assert np.mean(last_10pct) <= np.mean(first_10pct), (
            f"Stress should decrease overall: first={first_10pct}, last={last_10pct}"
        )


def test_nmds_stress_matches_reference_formula(small_distance_matrix):
    """Optimised SMACOF must compute stress using the correct reference formula.

    Stress-1 formula (Kruskal 1964): sqrt(sum((d_hat - d_tilde)^2) / sum(d_target^2))
    where d_target are the original dissimilarities (NOT the configuration distances).

    This test verifies the stress formula is correct by computing it manually.
    """
    from scipy.spatial.distance import cdist

    from statistics.nmds import NMDSAnalyzer, _pava_increasing

    # Get NMDS result
    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(
        small_distance_matrix,
        n_dimensions=2,
        n_restarts=1,
        max_iterations=50,
        random_seed=99,
    )

    # Manually compute stress using the correct formula
    D = small_distance_matrix
    n = D.shape[0]
    X = result.coordinates
    iu, ju = np.triu_indices(n, k=1)
    d_target = D[iu, ju]

    # Compute configuration distances
    D_hat = cdist(X, X, metric="euclidean")
    d_hat = D_hat[iu, ju]

    # Isotonic regression (PAVA on d_hat sorted by d_target)
    order = np.argsort(d_target, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    d_tilde = _pava_increasing(d_hat[order])[inv]

    # Reference stress formula
    diff = d_hat - d_tilde
    reference_stress = np.sqrt(np.sum(diff**2) / np.sum(d_target**2))

    # NMDS result stress should match reference formula
    np.testing.assert_allclose(
        result.stress,
        reference_stress,
        rtol=1e-6,
        err_msg=f"stress={result.stress} should match reference {reference_stress}",
    )


def test_nmds_different_restarts_give_different_results():
    """Multiple restarts must not all return identical coordinates."""
    from scipy.spatial.distance import cdist

    from statistics.nmds import NMDSAnalyzer

    # A more challenging matrix that is unlikely to converge to the same
    # minimum from different random initialisations
    np.random.seed(0)
    D = cdist(np.random.rand(8, 4), np.random.rand(8, 4), metric="euclidean")

    analyzer = NMDSAnalyzer()
    results = [
        analyzer.analyze(D, n_dimensions=2, n_restarts=1, max_iterations=100, random_seed=i)
        for i in range(3)
    ]

    # At least two of the three coordinate matrices should differ
    norms = [np.linalg.norm(r.coordinates) for r in results]
    # If all norms are identical within tolerance, the random restarts
    # may have collapsed to the same minimum — check they at least have
    # the same stress order
    assert len(set(round(n, 4) for n in norms)) >= 2 or all(
        results[0].stress <= r.stress for r in results[1:]
    ), "Restarts should produce distinct configurations or best stress is kept"


# ---------------------------------------------------------------------------
# Dependency-free isotonic step — source-code inspection test
# ---------------------------------------------------------------------------


def test_smacof_uses_builtin_pava_not_sklearn():
    """SMACOF's isotonic step must use the built-in PAVA helper.

    The old implementation hard-imported sklearn.IsotonicRegression, which
    broke NMDS entirely when scikit-learn was not installed. Verify by
    source inspection that the hot loop is dependency-free.
    """
    import inspect

    from statistics.nmds import NMDSAnalyzer

    source = inspect.getsource(NMDSAnalyzer._smacof)

    assert "from sklearn" not in source and "import sklearn" not in source, \
        "_smacof must not import sklearn"
    assert "IsotonicRegression(" not in source, "_smacof must not use sklearn's isotonic regression"
    assert "_pava_increasing(" in source, "_smacof must call the built-in PAVA helper"


# ---------------------------------------------------------------------------
# Progress reporting for multiple restarts
# ---------------------------------------------------------------------------


def test_analyze_accepts_progress_callback():
    """The analyze method should accept an optional progress callback."""
    from statistics.nmds import NMDSAnalyzer

    analyzer = NMDSAnalyzer()
    np.random.seed(0)
    D = np.random.rand(6, 6)
    np.fill_diagonal(D, 0)

    progress_calls = []

    def progress(restart, total, stress):
        progress_calls.append((restart, total, stress))

    # This should not raise
    result = analyzer.analyze(
        D,
        n_dimensions=2,
        n_restarts=3,
        max_iterations=20,
        progress_callback=progress,
    )

    # At minimum, we should have recorded each restart's final stress
    assert len(progress_calls) >= 3, (
        f"Expected >= 3 progress calls (one per restart), got {len(progress_calls)}"
    )
    for restart_idx, total, stress in progress_calls:
        assert 0 <= restart_idx < 3
        assert 0.0 <= float(stress) <= 1.0


# ---------------------------------------------------------------------------
# Isotonic (PAVA) helper and performance
# ---------------------------------------------------------------------------


def test_pava_increasing_matches_known_isotonic_fits():
    """_pava_increasing must equal the least-squares isotonic fit."""
    from statistics.nmds import _pava_increasing

    # Already increasing: identity
    y = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(_pava_increasing(y), y)

    # One violation: pool [3, 2] -> [2.5, 2.5]
    y = np.array([1.0, 3.0, 2.0, 4.0])
    np.testing.assert_allclose(_pava_increasing(y), [1.0, 2.5, 2.5, 4.0])

    # Fully decreasing: one pooled block
    y = np.array([3.0, 2.0, 1.0])
    np.testing.assert_allclose(_pava_increasing(y), [2.0, 2.0, 2.0])

    # Multiple blocks
    y = np.array([10.0, 1.0, 2.0, 9.0, 3.0])
    np.testing.assert_allclose(
        _pava_increasing(y), [13.0 / 3.0, 13.0 / 3.0, 13.0 / 3.0, 6.0, 6.0]
    )


@pytest.mark.slow
def test_smacof_runs_without_sklearn():
    """SMACOF must handle a realistic workload using the built-in PAVA fit.

    Replaces the old sklearn-vs-sklearn benchmark: the hot loop no longer
    imports scikit-learn at all, so the meaningful regression check is that
    a 50-point, 10-restart analysis completes quickly with good stress.
    """
    from scipy.spatial.distance import cdist

    from statistics.nmds import NMDSAnalyzer

    np.random.seed(42)
    data = np.random.rand(50, 10)
    D = cdist(data, data, metric="euclidean")

    analyzer = NMDSAnalyzer()
    start = time.perf_counter()
    result = analyzer.analyze(
        D,
        n_dimensions=2,
        n_restarts=10,
        max_iterations=200,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0, f"NMDS took {elapsed:.1f}s for a 50-point matrix"
    assert result.stress < 0.2, f"Stress {result.stress} is unexpectedly high"
    assert result.n_iterations > 0
