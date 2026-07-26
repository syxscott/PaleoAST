# tests/stratigraphy/test_pyper_peterman.py
"""
Tests for pyper_peterman_correction function.

Validates:
1. White noise data: n_eff ≈ n (no autocorrelation)
2. AR(1) data with phi=0.9: n_eff << n (strong autocorrelation)
3. Comparison with uncorrected pearsonr p-values
"""

import numpy as np
from numpy.typing import ArrayLike


def _generate_ar1(n: int, phi: float, innovations_std: float = 1.0) -> np.ndarray:
    """
    Generate AR(1) process: x[t] = phi * x[t-1] + eps[t]
    """
    eps = np.random.randn(n) * innovations_std
    x = np.zeros(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_pyper_peterman_white_noise():
    """
    White noise data: effective sample size should be close to n.
    """
    from stratigraphy.correlation import pyper_peterman_correction
    from scipy import stats

    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)

    r, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    # For white noise, n_eff should be close to n
    assert n_orig == n, f"Original n should be {n}, got {n_orig}"
    assert 0.5 * n <= n_eff <= n, (
        f"For white noise, n_eff should be between 0.5*n and n, got {n_eff}"
    )


def test_pyper_peterman_ar1_high_autocorrelation():
    """
    AR(1) data with phi=0.9: n_eff should be substantially smaller than n.
    """
    from stratigraphy.correlation import pyper_peterman_correction

    np.random.seed(42)
    n = 100
    phi = 0.9

    # Generate AR(1) series with strong autocorrelation
    x = _generate_ar1(n, phi)
    y = _generate_ar1(n, phi)

    r, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    assert n_orig == n
    # With phi=0.9, effective sample size should be much smaller
    assert n_eff < 0.5 * n, (
        f"With phi={phi}, n_eff should be < 0.5*n={0.5*n}, got {n_eff}"
    )


def test_pyper_peterman_ar1_low_autocorrelation():
    """
    AR(1) data with phi=0.3: n_eff moderate reduction.
    """
    from stratigraphy.correlation import pyper_peterman_correction

    np.random.seed(42)
    n = 100
    phi = 0.3

    x = _generate_ar1(n, phi)
    y = _generate_ar1(n, phi)

    r, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    assert n_orig == n
    # With phi=0.3, n_eff should be somewhat smaller but not as dramatic
    assert n_eff < n, f"n_eff should be < n, got {n_eff}"


def test_pyper_peterman_vs_pearsonr():
    """
    Compare corrected p-value with uncorrected pearsonr p-value.
    The corrected p-value should generally be larger (less significant)
    for autocorrelated data.
    """
    from stratigraphy.correlation import pyper_peterman_correction
    from scipy import stats

    np.random.seed(123)
    n = 50
    phi = 0.8

    x = _generate_ar1(n, phi)
    y = x + 0.5 * np.random.randn(n)  # y correlated with x

    r_uncorr, p_uncorr = stats.pearsonr(x, y)
    r_corr, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    # The corrected p-value should be larger than uncorrected
    # for autocorrelated data (more conservative)
    assert p_corr >= p_uncorr - 1e-10, (
        f"Corrected p={p_corr} should be >= uncorrected p={p_uncorr}"
    )
    assert n_eff <= n_orig


def test_pyper_peterman_nan_handling():
    """
    Test handling of NaN values.
    """
    from stratigraphy.correlation import pyper_peterman_correction

    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    y = np.random.randn(n)

    # Insert NaNs
    x[10] = np.nan
    x[25] = np.nan
    y[30] = np.nan

    r, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    # Should handle NaNs gracefully
    assert n_orig == n - 3, f"Expected n_orig={n-3}, got {n_orig}"
    assert not np.isnan(r)
    assert not np.isnan(p_corr)


def test_pyper_peterman_short_series():
    """
    Test behavior with very short series (less than 4 points).
    """
    from stratigraphy.correlation import pyper_peterman_correction

    np.random.seed(42)
    x = np.random.randn(3)
    y = np.random.randn(3)

    r, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    # Should return nan for very short series
    assert np.isnan(r) and np.isnan(p_corr)


def test_pyper_peterman_returns_float():
    """
    Test that return values are proper floats (not numpy scalars).
    """
    from stratigraphy.correlation import pyper_peterman_correction

    np.random.seed(42)
    x = np.random.randn(50)
    y = np.random.randn(50)

    r, p_corr, n_eff, n_orig = pyper_peterman_correction(x, y)

    assert isinstance(r, float)
    assert isinstance(p_corr, float)
    assert isinstance(n_eff, int)
    assert isinstance(n_orig, int)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
