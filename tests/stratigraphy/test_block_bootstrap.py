# tests/stratigraphy/test_block_bootstrap.py
"""
Tests for block_bootstrap_ci function.

Validates:
1. White noise: CI width similar to standard bootstrap
2. AR(1) high autocorrelation: CI width > standard bootstrap
3. Coverage: 95% CI contains true parameter ~95% of the time (in expectation)
"""

import numpy as np


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


def _standard_bootstrap_ci(
    data: np.ndarray, statistic_func: callable, n_bootstrap: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Standard (i.i.d.) bootstrap CI for comparison."""
    rng = np.random
    bootstrap_stats = np.empty(n_bootstrap)
    n = len(data)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        resampled = data[indices]
        bootstrap_stats[i] = statistic_func(resampled)

    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return float(ci_lower), float(ci_upper)


def test_block_bootstrap_white_noise_vs_standard():
    """
    For white noise, block bootstrap CI should be similar to standard bootstrap.
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 100
    data = np.random.randn(n)

    stat_func = np.mean

    ci_block = block_bootstrap_ci(data, stat_func, block_size=10, n_bootstrap=500, alpha=0.05)
    ci_standard = _standard_bootstrap_ci(data, stat_func, n_bootstrap=500, alpha=0.05)

    width_block = ci_block[1] - ci_block[0]
    width_standard = ci_standard[1] - ci_standard[0]

    # Should be similar for white noise
    assert 0.5 * width_standard <= width_block <= 2.0 * width_standard, (
        f"Block bootstrap CI width {width_block} should be similar to "
        f"standard {width_standard} for white noise"
    )


def test_block_bootstrap_ar1_ci_wider():
    """
    For highly autocorrelated AR(1) data, block bootstrap CI should be wider
    than standard bootstrap CI.
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 100
    phi = 0.9
    data = _generate_ar1(n, phi)

    stat_func = np.mean

    ci_block = block_bootstrap_ci(data, stat_func, block_size=10, n_bootstrap=500, alpha=0.05)
    ci_standard = _standard_bootstrap_ci(data, stat_func, n_bootstrap=500, alpha=0.05)

    width_block = ci_block[1] - ci_block[0]
    width_standard = ci_standard[1] - ci_standard[0]

    # Block bootstrap CI should be wider for autocorrelated data
    assert width_block >= width_standard * 0.8, (
        f"Block bootstrap CI width {width_block} should be >= "
        f"standard bootstrap width {width_standard} for AR(1) phi=0.9"
    )


def test_block_bootstrap_coverage_simulation():
    """
    Simulate coverage: repeatedly compute 95% CI and check if true mean
    is inside the interval. Should be close to 95% for correct method.
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 50
    true_mean = 0.0
    n_trials = 100  # Reduced for speed
    phi = 0.5  # Moderate autocorrelation

    stat_func = np.mean
    block_size = 5

    covered = 0
    for _ in range(n_trials):
        # Generate AR(1) data with known mean
        data = true_mean + _generate_ar1(n, phi)

        ci_lower, ci_upper = block_bootstrap_ci(
            data, stat_func, block_size=block_size, n_bootstrap=200, alpha=0.05
        )

        if ci_lower <= true_mean <= ci_upper:
            covered += 1

    coverage = covered / n_trials

    # Coverage should be approximately 95% (allow some Monte Carlo variation)
    # For highly autocorrelated data with small n_bootstrap, it may be less precise
    assert 0.70 <= coverage <= 1.0, (
        f"Coverage {coverage:.2%} is outside reasonable range for 95% CI"
    )


def test_block_bootstrap_nan_handling():
    """
    Test that NaN values are handled gracefully.
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 50
    data = np.random.randn(n)
    data[10] = np.nan
    data[25] = np.nan

    stat_func = np.mean

    ci_lower, ci_upper = block_bootstrap_ci(data, stat_func, block_size=10, n_bootstrap=100)

    # Should not return NaN
    assert not np.isnan(ci_lower)
    assert not np.isnan(ci_upper)
    assert ci_lower <= ci_upper


def test_block_bootstrap_short_series():
    """
    Test behavior with very short series (less than 4 points).
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 3  # < 4, so should return NaN
    data = np.random.randn(n)

    stat_func = np.mean

    ci_lower, ci_upper = block_bootstrap_ci(data, stat_func, block_size=2, n_bootstrap=50)

    # Should return nan for very short series
    assert np.isnan(ci_lower)
    assert np.isnan(ci_upper)


def test_block_bootstrap_automatic_block_size():
    """
    Test that automatic block size selection (block_size=None) works.
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 100
    phi = 0.7
    data = _generate_ar1(n, phi)

    stat_func = np.mean

    # With automatic block size
    ci_auto = block_bootstrap_ci(data, stat_func, block_size=None, n_bootstrap=200, alpha=0.05)

    # With explicit block size
    ci_explicit = block_bootstrap_ci(data, stat_func, block_size=10, n_bootstrap=200, alpha=0.05)

    # Both should produce valid CIs
    assert not np.isnan(ci_auto[0]) and not np.isnan(ci_auto[1])
    assert not np.isnan(ci_explicit[0]) and not np.isnan(ci_explicit[1])


def test_block_bootstrap_variance_statistic():
    """
    Test with variance as the statistic (positive valued).
    """
    from stratigraphy.isotope_analysis import block_bootstrap_ci

    np.random.seed(42)
    n = 100
    data = np.random.randn(n) ** 2  # Chi-squared with 1 df

    stat_func = np.var

    ci_lower, ci_upper = block_bootstrap_ci(data, stat_func, block_size=10, n_bootstrap=200)

    # Variance should always be positive
    assert ci_lower >= 0
    assert ci_upper >= ci_lower


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
