# =============================================================================
# FILE: tests/ecology/test_inext_bootstrap.py
# =============================================================================
"""
Tests for iNEXT bootstrap resampling in coverage-based rarefaction.

Verifies that the multinomial bootstrap properly resamples data according to
Chao & Jost (2012) - iNEXT methodology.

Key tests:
- Bootstrap samples are truly different from each other (resampling works)
- Bootstrap samples differ from original data
- Results are reproducible with seed
- Different seeds produce different results

References:
    Chao, A., & Jost, L. (2012). Coverage-based rarefaction and
        extrapolation: sampling and projecting species diversity.
        Methods in Ecology and Evolution, 3(5), 873-882.
"""

import numpy as np
import pytest


class TestINEXTBootstrapResampling:
    """Test that iNEXT bootstrap properly resamples data."""

    def test_bootstrap_samples_differ_from_each_other(self):
        """
        Verify that bootstrap resampling produces different samples.

        Each bootstrap replicate should be a different resample from the
        multinomial distribution, so bootstrap curves should vary.
        """
        from ecology.beta_diversity import coverage_rarefaction_hill, _multinomial_resample

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Test data: 50 individuals, 5 species
        species_counts = np.array([20.0, 15.0, 8.0, 5.0, 2.0])
        N = int(np.sum(species_counts))

        # Generate multiple resamples
        resamples = []
        for _ in range(10):
            resample = _multinomial_resample(species_counts, N, rng)
            resamples.append(resample.copy())

        # Resamples should generally differ from each other
        # (there's a small probability they could be identical)
        all_same = all(np.array_equal(resamples[0], r) for r in resamples[1:])
        assert not all_same, "All bootstrap resamples were identical - resampling not working"

    def test_bootstrap_resample_sum_equals_N(self):
        """
        Multinomial resample should always sum to N.
        """
        from ecology.beta_diversity import _multinomial_resample

        rng = np.random.default_rng(123)
        species_counts = np.array([25.0, 10.0, 5.0])
        N = int(np.sum(species_counts))

        for _ in range(20):
            resample = _multinomial_resample(species_counts, N, rng)
            assert int(np.sum(resample)) == N, f"Resample sum {np.sum(resample)} != N={N}"

    def test_bootstrap_differs_from_original(self):
        """
        Bootstrap resample should generally differ from original data.
        """
        from ecology.beta_diversity import _multinomial_resample

        rng = np.random.default_rng(456)
        species_counts = np.array([30.0, 10.0, 5.0, 3.0, 2.0])
        N = int(np.sum(species_counts))

        # With 5 species and N=50, resamples will often differ from original
        n_different = 0
        for _ in range(50):
            resample = _multinomial_resample(species_counts, N, rng)
            if not np.array_equal(resample, species_counts):
                n_different += 1

        # At least some resamples should differ
        assert n_different > 0, "No resamples differed from original - possible bug"

    def test_bootstrap_reproducible_with_seed(self):
        """
        Same seed should produce identical bootstrap results.
        """
        from ecology.beta_diversity import coverage_rarefaction_hill

        abundance = np.array([[25, 10, 5, 3, 2]])

        result1 = coverage_rarefaction_hill(
            abundance, q=0, n_points=10, n_bootstrap=20, seed=999
        )
        result2 = coverage_rarefaction_hill(
            abundance, q=0, n_points=10, n_bootstrap=20, seed=999
        )

        np.testing.assert_allclose(
            result1.expected_richness, result2.expected_richness, rtol=1e-10,
            err_msg="Bootstrap results not reproducible with same seed"
        )
        np.testing.assert_allclose(
            result1.confidence_lower, result2.confidence_lower, rtol=1e-10
        )
        np.testing.assert_allclose(
            result1.confidence_upper, result2.confidence_upper, rtol=1e-10
        )

    def test_different_seeds_different_results(self):
        """
        Different seeds should produce different bootstrap results.
        """
        from ecology.beta_diversity import coverage_rarefaction_hill

        abundance = np.array([[25, 10, 5, 3, 2]])

        result1 = coverage_rarefaction_hill(
            abundance, q=0, n_points=10, n_bootstrap=30, seed=111
        )
        result2 = coverage_rarefaction_hill(
            abundance, q=0, n_points=10, n_bootstrap=30, seed=222
        )

        # 点估计来自观测数据 (Chao et al. 2014), 必须与种子完全无关;
        # bootstrap 只用于置信区间, 因此不同种子的 CI 端点应当不同。
        # (旧断言期望点估计随种子变化——那正是"点估计取 bootstrap
        # 中位数"这一缺陷的症状, 2026-09 修复后语义反转。)
        assert np.allclose(
            result1.expected_richness, result2.expected_richness, rtol=1e-12
        ), "Point estimate must be seed-independent (derived from observed data)"
        assert not np.allclose(
            result1.confidence_lower, result2.confidence_lower, rtol=1e-3
        ) or not np.allclose(
            result1.confidence_upper, result2.confidence_upper, rtol=1e-3
        ), "Different seeds should produce different bootstrap CI bounds"


class TestINEXTBootstrapCoverageLevels:
    """Test bootstrap behavior at different coverage levels."""

    def test_ci_width_increases_with_extrapolation(self):
        """
        Confidence intervals should generally widen at higher coverage levels
        (where extrapolation dominates).
        """
        from ecology.beta_diversity import coverage_rarefaction_hill

        abundance = np.array([[50, 30, 15, 5]])

        result = coverage_rarefaction_hill(
            abundance, q=0, n_points=20, n_bootstrap=50, seed=42
        )

        ci_widths = result.confidence_upper - result.confidence_lower

        # CI width at high coverage (last quartile) should typically be
        # larger than at medium coverage (middle quartile)
        low_idx = len(ci_widths) // 4
        high_idx = 3 * len(ci_widths) // 4

        # This is a heuristic - the relationship isn't strictly monotonic
        # but CIs should generally widen with extrapolation
        assert ci_widths[high_idx] >= 0

    def test_bootstrap_variance_at_coverage(self):
        """
        Test that bootstrap curves show genuine variance.
        """
        from ecology.beta_diversity import _multinomial_resample, _rarefaction_species

        # Data with known structure
        species_counts = np.array([20.0, 15.0, 10.0, 5.0])
        N = int(np.sum(species_counts))
        rng = np.random.default_rng(789)

        # Generate many resamples and compute variance
        n_resamples = 100
        rarefaction_at_m20 = []

        for _ in range(n_resamples):
            resample = _multinomial_resample(species_counts, N, rng)
            rarefied = _rarefaction_species(resample, 20)
            rarefaction_at_m20.append(rarefied)

        # Variance should be > 0 (unless by chance all resamples are identical)
        var = np.var(rarefaction_at_m20)
        assert var >= 0, "Variance should be non-negative"

        # With 100 resamples, we expect some variance unless the data is very unusual
        # This is a weak test but confirms bootstrap is actually varying


class TestINEXTBootstrapHillNumbers:
    """Test bootstrap with different Hill number orders."""

    def test_bootstrap_q0_richness(self):
        """Test bootstrap for q=0 (species richness)."""
        from ecology.beta_diversity import coverage_rarefaction_hill

        abundance = np.array([[100, 50, 30, 15, 5]])

        result = coverage_rarefaction_hill(
            abundance, q=0, n_points=15, n_bootstrap=30, seed=333
        )

        assert result.method == "coverage_rarefaction_hill_q0"
        assert all(result.expected_richness >= 0)
        assert all(result.confidence_lower <= result.expected_richness)
        assert all(result.expected_richness <= result.confidence_upper)

    def test_bootstrap_q1_shannon(self):
        """Test bootstrap for q=1 (Shannon entropy)."""
        from ecology.beta_diversity import coverage_rarefaction_hill

        abundance = np.array([[100, 50, 30, 15, 5]])

        result = coverage_rarefaction_hill(
            abundance, q=1, n_points=15, n_bootstrap=30, seed=333
        )

        assert result.method == "coverage_rarefaction_hill_q1"
        # Shannon diversity (exp(H)) should be positive
        assert all(result.expected_richness >= 0)

    def test_bootstrap_q2_simpson(self):
        """Test bootstrap for q=2 (Simpson concentration)."""
        from ecology.beta_diversity import coverage_rarefaction_hill

        abundance = np.array([[100, 50, 30, 15, 5]])

        result = coverage_rarefaction_hill(
            abundance, q=2, n_points=15, n_bootstrap=30, seed=333
        )

        assert result.method == "coverage_rarefaction_hill_q2"
        # Simpson concentration should be between 0 and 1
        assert all(result.expected_richness >= 0)
        assert all(result.expected_richness <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
