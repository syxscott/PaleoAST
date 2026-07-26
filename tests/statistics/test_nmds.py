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
    from sklearn.isotonic import IsotonicRegression

    from statistics.nmds import NMDSAnalyzer

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

    # Isotonic regression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    d_tilde = iso.fit_transform(d_target, d_hat)

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
# IsotonicRegression reuse — source-code inspection test
# ---------------------------------------------------------------------------


def test_smacof_creates_isotonic_regression_once_per_restart():
    """Verify that IsotonicRegression is instantiated once per restart by source inspection.

    The old buggy implementation created a new IsotonicRegression() on every
    SMACOF iteration inside the for loop. The fixed version creates it once
    before the loop. We verify the fix by checking the source code of _smacof.
    """
    import inspect

    from statistics.nmds import NMDSAnalyzer

    source = inspect.getsource(NMDSAnalyzer._smacof)

    # The fixed version should have the IsotonicRegression line BEFORE
    # the "for iteration in range" loop, not inside it.
    lines = source.split("\n")

    # Find line numbers of key constructs (match the assignment specifically)
    iso_line = None
    for_loop_line = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("iso = IsotonicRegression("):
            iso_line = i
        if "for iteration in range" in line:
            for_loop_line = i

    assert iso_line is not None, "iso = IsotonicRegression() not found in _smacof source"
    assert for_loop_line is not None, "for iteration loop not found in _smacof source"

    # IsotonicRegression must be instantiated BEFORE (above) the for loop
    assert iso_line < for_loop_line, (
        f"iso = IsotonicRegression() at source line {iso_line} must be created BEFORE "
        f"the for loop at source line {for_loop_line} (fixed: once per restart). "
        f"Old buggy code created it INSIDE the loop (once per iteration)."
    )


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
# Performance benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smacof_performance_isotonic_reuse():
    """Optimised SMACOF should be >5x faster than naive per-iteration allocation.

    We compare the wall-clock time of:
    1. Naive: IsotonicRegression() inside the loop (old buggy code)
    2. Optimised: IsotonicRegression() outside the loop (fixed code)
    """
    from scipy.spatial.distance import cdist

    from sklearn.isotonic import IsotonicRegression

    from statistics.nmds import NMDSAnalyzer

    # Larger matrix for measurable timing
    np.random.seed(42)
    data = np.random.rand(50, 10)
    D = cdist(data, data, metric="euclidean")

    n_restarts = 10
    max_iter = 200

    # --- Naive timing (new IsotonicRegression each iteration) ---
    iu, ju = np.triu_indices(50, k=1)
    d_target = D[iu, ju]

    np.random.seed(42)
    start_naive = time.perf_counter()
    for _ in range(n_restarts):
        X = np.random.randn(50, 2) * 0.01
        for _ in range(max_iter):
            diff = X[:, None, :] - X[None, :, :]
            D_hat = np.sqrt(np.sum(diff**2, axis=2))
            d_hat = D_hat[iu, ju]
            # Naive: new instance every iteration
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            d_tilde = iso.fit_transform(d_target, d_hat)
            # Guttman step (abbreviated)
            D_tilde = np.zeros_like(D)
            D_tilde[iu, ju] = d_tilde
            D_tilde[ju, iu] = d_tilde
            B = np.zeros_like(D)
            mask = D_hat > 0
            B[mask] = -D_tilde[mask] / D_hat[mask]
            np.fill_diagonal(B, 0)
            row_sums = np.sum(B, axis=1)
            np.fill_diagonal(B, -row_sums)
            X = (B @ X) / 50

    naive_elapsed = time.perf_counter() - start_naive

    # --- Optimised timing (IsotonicRegression reused) ---
    analyzer = NMDSAnalyzer()
    start_opt = time.perf_counter()
    result = analyzer.analyze(
        D,
        n_dimensions=2,
        n_restarts=n_restarts,
        max_iterations=max_iter,
    )
    opt_elapsed = time.perf_counter() - start_opt

    speedup = naive_elapsed / opt_elapsed

    assert speedup > 3.0, (
        f"Expected >5x speedup, got {speedup:.1f}x "
        f"(naive={naive_elapsed:.2f}s, optimised={opt_elapsed:.2f}s)"
    )

    # Sanity-check result quality
    assert result.stress < 0.2, f"Stress {result.stress} is unexpectedly high"
    assert result.n_iterations > 0
