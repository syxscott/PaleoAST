"""
Regression tests for defect 8: allometry isometry F-test used the PCA-
reduced shape dimension for df1, which silently truncated the test to the
retained PCs even though the function reports the p-value as a "shape
isometry" test.

The fix computes the F-test in the FULL shape space (using
``predicted_full`` and the corresponding full-space residuals).  When the
caller does NOT request PCA reduction, the new and old F-statistics
agree to fp64 noise.  When PCA reduction is requested, the new
F-statistic uses df1 = full shape dimension and SS computed in the full
shape space.

The bug was that ``df1 = shape_reduced.shape[1]`` made the F-test a
"shape-isometry test restricted to the retained PCs", which is a much
weaker claim than the function's docstring suggests.
"""

import numpy as np
import pytest

from morphometrics.allometry import AllometryAnalyzer


def _aligned_with_allometry(seed=7, n=25, p=5, k=2):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(p, k)) * 0.3
    cfgs = np.empty((n, p, k))
    log_cs = np.linspace(0.0, 1.5, n)
    slope = rng.normal(size=(p, k)) * 0.4
    for i in range(n):
        cfgs[i] = base + log_cs[i] * slope + 0.05 * rng.normal(size=(p, k))
    return cfgs, log_cs


def _expected_full_shape_f(cfgs, n_components):
    """Reference F-test computed on the FULL shape space."""
    from scipy import stats

    n, p, k = cfgs.shape
    full_dim = p * k
    flat = cfgs.reshape(n, full_dim)
    mean = flat.mean(axis=0)
    centred = flat - mean
    cs = np.sqrt(np.sum((cfgs - cfgs.mean(axis=1, keepdims=True)) ** 2, axis=(1, 2)))
    log_cs = np.log(cs)

    evecs = None
    if n_components is not None and n_components < full_dim:
        pca_mat = centred.T @ centred / (n - 1)
        evals, vecs = np.linalg.eigh(pca_mat)
        order = np.argsort(evals)[::-1]
        evecs = vecs[:, order[:n_components]]
        shape_reduced = centred @ evecs
    else:
        shape_reduced = centred

    X = np.column_stack([np.ones(n), log_cs])
    beta, *_ = np.linalg.lstsq(X, shape_reduced, rcond=None)
    predicted_reduced = X @ beta
    if evecs is not None:
        predicted_full = predicted_reduced @ evecs.T + mean
    else:
        predicted_full = predicted_reduced + mean

    ss_null = float(np.sum((flat - mean) ** 2))
    ss_res = float(np.sum((flat - predicted_full) ** 2))
    df1 = full_dim
    df2 = n - 2
    if ss_res > 0 and df2 > 0:
        f_stat = ((ss_null - ss_res) / df1) / (ss_res / df2)
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
    else:
        f_stat = 0.0
        p_value = 1.0
    return f_stat, p_value


def _expected_reduced_shape_f(cfgs, n_components):
    """Reference F-test computed on the REDUCED shape space
    (the legacy behaviour)."""
    from scipy import stats

    n, p, k = cfgs.shape
    full_dim = p * k
    flat = cfgs.reshape(n, full_dim)
    mean = flat.mean(axis=0)
    centred = flat - mean
    cs = np.sqrt(np.sum((cfgs - cfgs.mean(axis=1, keepdims=True)) ** 2, axis=(1, 2)))
    log_cs = np.log(cs)

    if n_components is not None and n_components < full_dim:
        pca_mat = centred.T @ centred / (n - 1)
        evals, vecs = np.linalg.eigh(pca_mat)
        order = np.argsort(evals)[::-1]
        evecs = vecs[:, order[:n_components]]
        shape_reduced = centred @ evecs
    else:
        shape_reduced = centred
        evecs = None

    X = np.column_stack([np.ones(n), log_cs])
    beta, *_ = np.linalg.lstsq(X, shape_reduced, rcond=None)
    pred = X @ beta
    ss_null = float(np.sum((shape_reduced - shape_reduced.mean(0)) ** 2))
    ss_res = float(np.sum((shape_reduced - pred) ** 2))
    df1 = shape_reduced.shape[1]
    df2 = n - 2
    if ss_res > 0 and df2 > 0:
        f_stat = ((ss_null - ss_res) / df1) / (ss_res / df2)
        p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
    else:
        f_stat = 0.0
        p_value = 1.0
    return f_stat, p_value


class TestAllometryFTestUsesFullShapeSpace:
    def test_no_pca_matches_full_shape_reference(self):
        cfgs, _ = _aligned_with_allometry()
        analyzer = AllometryAnalyzer()
        result = analyzer.analyze_allometry(cfgs, n_components=None)
        ref_f, ref_p = _expected_full_shape_f(cfgs, n_components=None)
        assert result.f_statistic == pytest.approx(ref_f, abs=1e-9)
        assert result.isometry_pvalue == pytest.approx(ref_p, abs=1e-12)

    def test_pca_reduction_still_uses_full_shape_reference(self):
        cfgs, _ = _aligned_with_allometry(seed=42, n=25, p=8, k=2)
        analyzer = AllometryAnalyzer()
        result = analyzer.analyze_allometry(cfgs, n_components=2)
        ref_f, ref_p = _expected_full_shape_f(cfgs, n_components=2)
        assert result.f_statistic == pytest.approx(ref_f, abs=1e-9)
        assert result.isometry_pvalue == pytest.approx(ref_p, abs=1e-12)

    def test_pca_reduction_differs_from_legacy_reduced(self):
        """The bug made the F-test identical to a reduced-space test.
        After the fix the analyzer's F-statistic should differ from the
        legacy (reduced-space) computation."""
        cfgs, _ = _aligned_with_allometry(seed=42, n=25, p=8, k=2)
        analyzer = AllometryAnalyzer()
        result = analyzer.analyze_allometry(cfgs, n_components=2)
        legacy_f, _ = _expected_reduced_shape_f(cfgs, n_components=2)
        # the new F-statistic must NOT equal the legacy one.
        assert result.f_statistic != pytest.approx(legacy_f, abs=1e-09), (
            "Bug not fixed: analyzer's F-statistic matches the legacy reduced-space computation."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
