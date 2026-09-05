# =============================================================================
# FILE: tests/property/test_pca_properties.py
# =============================================================================
"""
Property-based tests for PCA using Hypothesis.

Tests mathematical invariants of PCA:
1. Explained variance sums to 100%
2. Eigenvalues are non-negative
3. Loadings orthogonality (L.T @ L = diagonal)
4. Covariance reconstruction
5. Scores are centered
6. Rotation invariance (covariance PCA)
7. Correlation PCA total variance = p
8. Singular values eigenvalue relationship

References:
    Jolliffe, I.T. (2002). Principal Component Analysis, 2nd ed. Springer.
    Hotelling, H. (1933). Journal of Educational Psychology, 24(6), 417-441.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from statistics.pca import PCAAnalyzer

_2d_finite_data = st.lists(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=20,
    ),
    min_size=3, max_size=100,
)

_2d_small_data = st.lists(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=5,
    ),
    min_size=3, max_size=20,
)


def _try_make_array(data):
    """Try to convert data to float array, return None if invalid."""
    try:
        X = np.array(data, dtype=float)
    except (ValueError, TypeError):
        return None
    return X


@given(data=_2d_finite_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_explained_variance_sums_to_100(data):
    """Property: explained_variance percentages sum to 100%."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    col_stds = np.std(X, axis=0)
    if np.any(col_stds < 1e-12):
        return  # Skip if any column has zero variance
    result = PCAAnalyzer().analyze(X, method="covariance")
    total_explained = np.sum(result.explained_variance)
    assert abs(total_explained - 100.0) < 0.1, (
        f"Explained variance sum = {total_explained:.6f}%, expected 100%"
    )


@given(data=_2d_finite_data)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_eigenvalues_nonnegative(data):
    """Property: all eigenvalues >= 0 (covariance matrices are PSD)."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    result = PCAAnalyzer().analyze(X, method="covariance")
    assert np.all(result.eigenvalues >= -1e-10), (
        f"Found negative eigenvalues: {result.eigenvalues[result.eigenvalues < 0]}"
    )


@given(data=_2d_small_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_loadings_diagonal_gramian(data):
    """Property: loadings have diagonal Gramian (L.T @ L = diag(eigenvalues))."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    rank = np.linalg.matrix_rank(X)
    if rank < min(2, X.shape[1]):
        return
    result = PCAAnalyzer().analyze(X, method="covariance")
    L = result.loadings
    gramian = L.T @ L
    # gramian should be diagonal with eigenvalues on diagonal
    n = result.n_components
    expected_diag = np.diag(result.eigenvalues)
    diag_str = np.array2string(np.diag(result.eigenvalues), precision=4)
    gramian_str = np.array2string(gramian, precision=4)
    # 量级感知容差: 特征值大时 (如 5.9e5) 浮点舍入的绝对残差可达
    # 1e-10 以上, 固定 atol=1e-10 会随数据量级偶然失败。
    tol = 1e-10 + 1e-9 * float(np.max(np.abs(expected_diag)))
    assert np.allclose(gramian, expected_diag, atol=tol), (
        f"Loadings Gramian is not diagonal. Gramian: {gramian_str}, Expected diagonal: {diag_str}"
    )


@given(data=_2d_small_data)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_covariance_reconstruction(data):
    """Property: V @ Lambda @ V.T approximates the original covariance matrix."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    if np.std(X) < 1e-12:
        return
    rank = np.linalg.matrix_rank(X)
    if rank < min(2, X.shape[1]):
        return
    # Skip ill-conditioned data (high condition number)
    try:
        cond = np.linalg.cond(X)
        if cond > 1e6:
            return
    except Exception:
        return
    analyzer = PCAAnalyzer()
    n_samples, n_features = X.shape
    # 显式请求全部主成分: 全成分下 V·Λ·Vᵀ 必须精确重建协方差矩阵。
    # 旧测试用默认截断的 n_components, 丢弃尾部成分后重建误差可以
    # 任意大, "<2.0" 的宽容差掩盖不了病态样例 (稀疏 4x3 矩阵即失败)。
    result = analyzer.analyze(X, n_components=min(n_samples - 1, n_features), method="covariance")
    X_centered = X - result.mean_vector
    S_original = (X_centered.T @ X_centered) / (n_samples - 1)
    # 本代码的 loadings 约定为 U·√Λ (见 pca.py: loadings[:,k] =
    # eigenvector_k * sqrt(eigenvalue_k)), 因此 S = L·Lᵀ;
    # 旧写法 L·Λ·Lᵀ 隐含 loadings 为单位特征向量, 与约定不符。
    L = result.loadings
    S_reconstructed = L @ L.T
    # 全成分重建: 相对 Frobenius 误差应接近机器精度
    denom = np.linalg.norm(S_original, "fro")
    if denom < 1e-12:
        return
    relative_error = np.linalg.norm(S_original - S_reconstructed, "fro") / denom
    assert relative_error < 1e-8, (
        f"Full-component covariance reconstruction failed: relative_error={relative_error}"
    )


@given(data=_2d_small_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_scores_are_centered(data):
    """Property: PCA scores have zero mean."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    result = PCAAnalyzer().analyze(X, method="covariance")
    score_means = np.mean(result.scores, axis=0)
    assert np.allclose(score_means, 0.0, atol=1e-10)


@given(
    data=_2d_small_data,
    angle=st.floats(min_value=0.0, max_value=6.283185307),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_rotation_invariance(data, angle):
    """Property: eigenvalues are invariant to orthogonal rotation for covariance PCA."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    skip_angles = [0, 1.570796327, 3.141592654, 4.71238898, 6.283185307]
    if any(abs(angle - a) < 1e-6 for a in skip_angles):
        return
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    X_rotated = X.copy()
    X_rotated[:, :2] = X[:, :2] @ R.T
    r_orig = PCAAnalyzer().analyze(X, method="covariance")
    r_rot = PCAAnalyzer().analyze(X_rotated, method="covariance")
    assert np.allclose(np.sort(r_orig.eigenvalues), np.sort(r_rot.eigenvalues), rtol=1e-8)


@given(data=_2d_small_data)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_correlation_pca_total_variance_equals_p(data):
    """Property: for correlation PCA, sum of eigenvalues = number of non-zero-variance columns."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    col_stds = np.std(X, axis=0)
    if np.any(col_stds < 1e-12):
        return  # Skip columns with zero variance
    result = PCAAnalyzer().analyze(X, method="correlation")
    total_var = np.sum(result.eigenvalues)
    n_vars = X.shape[1]
    assert abs(total_var - n_vars) < 0.1


@given(data=_2d_small_data)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_pca_singular_values_eigenvalue_relationship(data):
    """Property: eigenvalues = singular_values^2 / (n - 1)."""
    X = _try_make_array(data)
    if X is None or X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 3:
        return
    result = PCAAnalyzer().analyze(X, method="covariance")
    n = X.shape[0]
    expected_ev = (result.singular_values**2) / (n - 1)
    ev_sorted = np.sort(result.eigenvalues_raw[:result.n_components])
    sv_sorted = np.sort(expected_ev[:result.n_components])
    assert np.allclose(ev_sorted, sv_sorted, rtol=1e-8)
