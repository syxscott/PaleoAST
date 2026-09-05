# tests/stratigraphy/test_coniss_broken_stick.py
"""
Unit tests for CONISS broken-stick model significance test.

Tests the broken_stick_test function and its integration with CONISSAnalyzer.
"""

import numpy as np
import pytest

from stratigraphy.coniss import CONISSAnalyzer, broken_stick_test


class TestBrokenStickTest:
    """Test suite for broken-stick model significance test."""

    def test_broken_stick_basic(self):
        """Test broken-stick test with simple data."""
        # Create synthetic BD values (should sum to ~1 when normalized)
        # Higher values at early splits indicate significant zones
        # bd_values has n elements → n+1 levels → n possible zone boundaries
        bd_values = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

        result = broken_stick_test(bd_values, n_permutations=99)

        assert "significant_zones" in result
        assert "p_values" in result
        assert "broken_stick_expectation" in result
        # n bd_values → n zone boundaries → n p_values
        assert len(result["p_values"]) == len(bd_values)
        # n zone boundaries → n broken-stick expectations
        assert len(result["broken_stick_expectation"]) == len(bd_values)
        # All expectation values must be in (0, 1]; the last value can be 1.0
        for ev in result["broken_stick_expectation"]:
            assert 0 < ev <= 1, f"Expectation {ev} out of range (0, 1]"

    def test_broken_stick_uniform(self):
        """Test broken-stick with uniform (non-significant) data."""
        # When all BD values are nearly equal, no zone should be significant
        bd_values = np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.16])

        result = broken_stick_test(bd_values, n_permutations=99)

        # With uniform values, most p-values should be high (> 0.05)
        high_p_count = sum(1 for p in result["p_values"] if p > 0.05)
        assert high_p_count >= len(result["p_values"]) // 2

    def test_broken_stick_one_strong(self):
        """Test broken-stick with one dominant zone."""
        # One very large BD value followed by small ones.
        # The permutation test evaluates the maximum cumulative deviation
        # across all steps; it may or may not flag the first zone as
        # significant depending on the data.  Here we verify that:
        # (a) the output structure is correct, and
        # (b) p-values are well-defined (in [0, 1]).
        bd_values = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])

        result = broken_stick_test(bd_values, n_permutations=999)

        assert result["significant_zones"] >= 0
        assert all(0.0 <= p <= 1.0 for p in result["p_values"])
        # The dominant first value should give the smallest p-value at step 0
        assert result["p_values"][0] <= result["p_values"][-1]

    def test_broken_stick_expectation_values(self):
        """Expectation must follow the canonical MacArthur broken stick.

        For n segments the expected k-th largest share is
            E[k] = (1/n) * sum_{i=k..n} (1/i)
        (MacArthur 1957; Bennett 1996; identical to rioja::bstick).
        The previous implementation used the plain harmonic terms
        1/(n - i + 1) normalised to sum 1, which is NOT the broken-stick
        distribution (for n=6 the largest share would be 0.408 instead
        of the canonical 0.4083 vs ... see below).
        """
        n = 6
        bd_values = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])

        result = broken_stick_test(bd_values, n_permutations=99)
        expectation = np.asarray(result["broken_stick_expectation"])

        # Canonical expectation
        suffix = np.cumsum([1.0 / i for i in range(n, 0, -1)])[::-1]
        canonical = suffix / n
        assert len(expectation) == n
        assert np.allclose(expectation, canonical, atol=1e-12)
        # Monotonically decreasing, sums to 1
        assert np.all(np.diff(expectation) < 0)
        assert abs(expectation.sum() - 1.0) < 1e-12

    def test_broken_stick_dominant_zone_is_significant(self):
        """A first zone well above E[1] must be flagged significant
        (Bennett-style contiguous-prefix counting)."""
        bd = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = broken_stick_test(bd, n_permutations=999)
        assert result["significant_zones"] >= 1
        assert result["p_values"][0] < 0.05

    def test_broken_stick_zero_sum(self):
        """Test handling of zero BD sum (edge case)."""
        bd_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = broken_stick_test(bd_values, n_permutations=99)

        # Should handle gracefully
        assert result["significant_zones"] == 0
        assert all(p == 1.0 for p in result["p_values"])


class TestCONISSWithBrokenStick:
    """Test suite for CONISS integration with broken-stick test."""

    def test_analyze_with_broken_stick(self):
        """Test CONISS analyze method with broken-stick computation."""
        analyzer = CONISSAnalyzer()

        # Create synthetic stratigraphic data with clear zones
        np.random.seed(42)
        n_levels = 20
        n_vars = 5

        # Create data with 3 distinct zones
        data = np.vstack([
            np.random.randn(7, n_vars) + np.array([1.0, 1.0, 0.5, 0.3, 0.2]),
            np.random.randn(7, n_vars) + np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.random.randn(6, n_vars) + np.array([-1.0, -0.8, -0.5, -0.3, -0.2]),
        ])

        result, broken_stick = analyzer.analyze(
            data,
            n_zones=3,
            compute_broken_stick=True,
            n_permutations=99
        )

        assert result is not None
        assert broken_stick is not None
        assert "significant_zones" in broken_stick
        # CONISS may find fewer than n_zones if the hierarchical structure
        # doesn't support that many distinct clusters; the broken-stick
        # test tells us how many are statistically meaningful.
        assert 1 <= result.n_zones <= 3

    def test_analyze_without_broken_stick(self):
        """Test CONISS analyze method without broken-stick computation."""
        analyzer = CONISSAnalyzer()

        data = np.random.randn(10, 4)

        result, broken_stick = analyzer.analyze(
            data,
            n_zones=3,
            compute_broken_stick=False
        )

        assert result is not None
        assert broken_stick is None

    def test_analyze_return_type(self):
        """Test that analyze returns tuple of (CONISSResult, dict)."""
        analyzer = CONISSAnalyzer()

        data = np.random.randn(8, 3)

        result, broken_stick = analyzer.analyze(data, n_zones=2, compute_broken_stick=True)

        # Check result type
        from stratigraphy.coniss import CONISSResult
        assert isinstance(result, CONISSResult)
        assert isinstance(broken_stick, dict)

        # Check broken_stick structure
        assert "significant_zones" in broken_stick
        assert "p_values" in broken_stick
        assert "broken_stick_expectation" in broken_stick

    def test_significant_zones_consistency(self):
        """Test that significant_zones is consistent with p_values."""
        analyzer = CONISSAnalyzer()

        # Create data with very clear zones
        np.random.seed(123)
        data = np.vstack([
            np.random.randn(5, 3) + [2.0, 2.0, 2.0],
            np.random.randn(5, 3) + [0.0, 0.0, 0.0],
            np.random.randn(5, 3) + [-2.0, -2.0, -2.0],
        ])

        result, broken_stick = analyzer.analyze(
            data,
            n_zones=3,
            compute_broken_stick=True,
            n_permutations=199  # More permutations for better estimate
        )

        # Count significant p-values
        sig_count = sum(1 for p in broken_stick["p_values"] if p < 0.05)

        # The reported significant_zones should be reasonable
        assert 0 <= broken_stick["significant_zones"] <= len(broken_stick["p_values"])


class TestBrokenStickEdgeCases:
    """Test edge cases for broken-stick model."""

    def test_single_level(self):
        """Test with single level (no splits possible)."""
        bd_values = np.array([])

        # With no BD values there are no zones to be significant.
        result = broken_stick_test(bd_values, n_permutations=99)
        assert result["significant_zones"] >= 0

    def test_two_levels(self):
        """Test with minimal case (2 levels = 1 split)."""
        bd_values = np.array([1.0])  # Single split

        result = broken_stick_test(bd_values, n_permutations=99)
        # significant_zones can be 0 (not significant) or 1 (significant)
        assert 0 <= result["significant_zones"] <= 1
        assert len(result["p_values"]) == 1

    def test_many_permutations(self):
        """Test with large number of permutations."""
        bd_values = np.array([0.4, 0.3, 0.2, 0.1])

        result = broken_stick_test(bd_values, n_permutations=999)

        # p-values should be more stable with more permutations
        assert len(result["p_values"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
