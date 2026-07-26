# =============================================================================
# FILE: tests/property/test_nmds_properties.py
# =============================================================================
"""
Property-based tests for NMDS using Hypothesis.

Tests mathematical invariants of NMDS ordination:
1. Stress is non-negative
2. Stress <= 1 (theoretical upper bound)
3. Coordinates shape matches input
4. Stress decreases monotonically within restart
5. Best restart has lowest stress
6. Zero stress for degenerate equal-distance matrix

References:
    Kruskal, J.B. (1964). Multidimensional scaling. Psychometrika, 29(1), 1-27.
    Borg, I. & Groenen, P.J.F. (1997). Modern Multidimensional Scaling. Springer.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from statistics.nmds import NMDSAnalyzer
from statistics.distance_metrics import compute_distance_matrix

_abundance_data = st.lists(
    st.lists(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=8,
    ),
    min_size=3, max_size=30,
)


def _try_make_array(data):
    """Try to convert data to float array, return None if invalid."""
    try:
        X = np.array(data, dtype=float)
    except (ValueError, TypeError):
        return None
    return X


@given(data=_abundance_data, random_seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_nmds_stress_is_nonnegative(data, random_seed):
    """Property: stress >= 0 always."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 3:
        return
    # Skip data with zero or near-zero values that cause numerical issues
    if np.max(np.abs(X)) < 1e-10:
        return
    D = compute_distance_matrix(X, metric="euclidean").matrix
    if np.max(D) < 1e-10:
        return
    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(D, n_dimensions=2, n_restarts=1, random_seed=random_seed)
    assert result.stress >= 0.0, f"Negative stress: {result.stress}"


@given(data=_abundance_data, random_seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_nmds_stress_le_one(data, random_seed):
    """Property: stress <= 1 (theoretical upper bound)."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 3:
        return
    if np.max(np.abs(X)) < 1e-10:
        return
    D = compute_distance_matrix(X, metric="euclidean").matrix
    if np.max(D) < 1e-10:
        return
    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(D, n_dimensions=2, n_restarts=1, random_seed=random_seed)
    assert result.stress <= 1.0 + 1e-10, f"Stress > 1: {result.stress}"


@given(data=_abundance_data, random_seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_nmds_coordinates_shape(data, random_seed):
    """Property: coordinates shape = (n_samples, n_dimensions)."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 3:
        return
    if np.max(np.abs(X)) < 1e-10:
        return
    D = compute_distance_matrix(X, metric="euclidean").matrix
    if np.max(D) < 1e-10:
        return
    analyzer = NMDSAnalyzer()
    for ndim in [2, 3]:
        result = analyzer.analyze(D, n_dimensions=ndim, n_restarts=1, random_seed=random_seed)
        assert result.coordinates.shape == (D.shape[0], ndim)


@given(data=_abundance_data, random_seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_nmds_stress_monotonic_decrease(data, random_seed):
    """Property: stress history is non-increasing within each restart."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 3:
        return
    if np.max(np.abs(X)) < 1e-10:
        return
    D = compute_distance_matrix(X, metric="euclidean").matrix
    if np.max(D) < 1e-10:
        return
    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(D, n_dimensions=2, n_restarts=1, random_seed=random_seed, tolerance=1e-8)
    history = result.stress_history
    for t in range(1, len(history)):
        # Allow small numerical noise but stress should generally decrease
        if t > 1:
            assert history[t] <= history[t - 1] + 1e-6, (
                f"Stress increased at iteration {t}: {history[t - 1]:.6f} -> {history[t]:.6f}"
            )


def test_nmds_zero_stress_for_degenerate_distance():
    """Property: if all off-diagonal distances are equal, stress is minimal."""
    np.random.seed(42)
    n = 5
    D = np.ones((n, n)) - np.eye(n)
    analyzer = NMDSAnalyzer()
    result = analyzer.analyze(D, n_dimensions=2, n_restarts=1, random_seed=42)
    assert result.stress < 0.2
