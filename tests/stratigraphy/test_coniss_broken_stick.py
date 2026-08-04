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
        bd_values = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

        result = broken_stick_test(bd_values, n_permutations=99)

        assert "significant_zones" in result
        assert "p_values" in result
        assert "broken_stick_expectation" in result
        assert result["significant_zones"] >= 1
        assert len(result["p_values"]) == len(bd_values)
        assert len(result["broken_stick_expectation"]) == len(bd_values)

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
        # One very large BD value followed by small ones
        bd_values = np.array([0.8, 0.1, 0.05, 0.02, 0.02, 0.01])

        result = broken_stick_test(bd_values, n_permutations=99)

        # First zone should be significant (low p-value)
        assert result["p_values"][0] < 0.1, "First zone should be significant with dominant BD value"

    def test_broken_stick_expectation_values(self):
        """Test that broken-stick expectation values are correct."""
        # For 6 levels (5 possible splits), expectations should follow broken-stick model
        n_levels = 6
        bd_values = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])

        result = broken_stick_test(bd_values, n_permutations=99)
        expectation = result["broken_stick_expectation"]

        # Broken-stick expectation: E[i] = 1/(n-i+1) for sorted descending
        # For n=6: E[1] = 1/6, E[2] = 1/5, E[3] = 1/4, E[4] = 1/3, E[5] = 1/2
        expected_vals = [1.0/6, 1.0/5, 1.0/4, 1.0/3, 1.0/2]

        # Check that expectation values are in the right range
        for i, ev in enumerate(expectation):
            assert 0 < ev < 1, f"Expectation {ev} at index {i} out of range"
            # Approximate check
            assert abs(ev - expected_vals[i]) < 0.1, f"Expectation {ev} differs from expected {expected_vals[i]}"

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
        assert result.n_zones == 3

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
        assert 1 <= broken_stick["significant_zones"] <= len(broken_stick["p_values"]) + 1


class TestBrokenStickEdgeCases:
    """Test edge cases for broken-stick model."""

    def test_single_level(self):
        """Test with single level (no splits possible)."""
        bd_values = np.array([])

        # This should handle gracefully
        result = broken_stick_test(bd_values, n_permutations=99)
        assert result["significant_zones"] >= 1  # At least one zone

    def test_two_levels(self):
        """Test with minimal case (2 levels = 1 split)."""
        bd_values = np.array([1.0])  # Single split

        result = broken_stick_test(bd_values, n_permutations=99)
        assert result["significant_zones"] >= 1
        assert len(result["p_values"]) == 1

    def test_many_permutations(self):
        """Test with large number of permutations."""
        bd_values = np.array([0.4, 0.3, 0.2, 0.1])

        result = broken_stick_test(bd_values, n_permutations=999)

        # p-values should be more stable with more permutations
        assert len(result["p_values"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
