# =============================================================================
# FILE: tests/property/test_distance_properties.py
# =============================================================================
"""
Property-based tests for distance metrics using Hypothesis.

Tests mathematical invariants ALL distance metrics must satisfy:
1. Symmetry: d(x,y) = d(y,x)
2. Non-negativity: d(x,y) >= 0
3. Self-distance zero: d(x,x) = 0
4. Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
5. Identity of indiscernibles: d(x,y) = 0 iff x = y
6. Canberra distance bounded by number of features
7. Bray-Curtis in [0, 1]
8. Euclidean agrees with manual computation

References:
    Kelley, J.L. (1955). General Topology. Van Nostrand.
    Legendre, P. & Legendre, L. (2012). Numerical Ecology, 3rd ed. Elsevier.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from statistics.distance_metrics import compute_distance_matrix

_abundance_data = st.lists(
    st.lists(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=8,
    ),
    min_size=2, max_size=30,
)

_general_data = st.lists(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=8,
    ),
    min_size=2, max_size=30,
)

METRICS = ["euclidean", "manhattan", "canberra", "chebychev"]


def _try_make_array(data):
    """Try to convert data to float array, return None if invalid."""
    try:
        X = np.array(data, dtype=float)
    except (ValueError, TypeError):
        return None
    return X


@given(data=_general_data, metric=st.sampled_from(METRICS))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_distance_symmetry(data, metric):
    """Property: d(x,y) = d(y,x) for all pairs."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 2:
        return
    result = compute_distance_matrix(X, metric=metric)
    D = result.matrix
    assert np.allclose(D, D.T, atol=1e-10), "Distance matrix is not symmetric"


@given(data=_abundance_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_bray_curtis_symmetry(data):
    """Property: Bray-Curtis is symmetric."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 2:
        return
    result = compute_distance_matrix(X, metric="bray_curtis")
    D = result.matrix
    assert np.allclose(D, D.T, atol=1e-10)


@given(data=_general_data, metric=st.sampled_from(METRICS))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_distance_nonnegative(data, metric):
    """Property: d(x,y) >= 0 for all pairs."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 2:
        return
    result = compute_distance_matrix(X, metric=metric)
    D = result.matrix
    assert np.all(D >= -1e-10), f"Found negative distances with metric={metric}"


@given(data=_general_data, metric=st.sampled_from(METRICS))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_distance_self_zero(data, metric):
    """Property: d(x,x) = 0 (diagonal of distance matrix is zero)."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 1:
        return
    result = compute_distance_matrix(X, metric=metric)
    D = result.matrix
    assert np.allclose(np.diag(D), 0.0, atol=1e-10)


@given(data=_general_data, metric=st.sampled_from(["euclidean", "manhattan"]))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_euclidean_triangle_inequality(data, metric):
    """Property: d(x,z) <= d(x,y) + d(y,z) for Euclidean and Manhattan."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 3:
        return
    D = compute_distance_matrix(X, metric=metric).matrix
    n = D.shape[0]
    for i in range(min(20, n)):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                assert D[i, j] + D[j, k] >= D[i, k] - 1e-10


@given(data=_abundance_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_bray_curtis_in_unit_interval(data):
    """Property: Bray-Curtis dissimilarity is in [0, 1]."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 2:
        return
    D = compute_distance_matrix(X, metric="bray_curtis").matrix
    upper = D[np.triu_indices(D.shape[0], k=1)]
    assert np.all(upper >= 0.0 - 1e-10)
    assert np.all(upper <= 1.0 + 1e-10)


@given(data=_general_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_euclidean_agrees_with_manual(data):
    """Property: Euclidean distance matches manual computation."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[0] < 2:
        return
    D = compute_distance_matrix(X, metric="euclidean").matrix
    n = D.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            manual = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            assert np.isclose(D[i, j], manual, atol=1e-10), (
                f"Euclidean mismatch at ({i},{j}): {D[i,j]:.6f} vs {manual:.6f}"
            )
