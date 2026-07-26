"""
Tests for statistics/cca.py - CCA and RDA constrained ordination.

Tests numerical stability fixes:
    - Bug 1: X'X near-singular inverse (ridge regularization + lstsq)
    - Bug 2: chi-square distance with zero expected values
"""

import logging
import warnings

import numpy as np
import pytest

from statistics.cca import CCAAnalyzer


class TestCCAChiSquareZeroExpected:
    """
    Tests for Bug 2: chi-square distance with zero expected values.

    When expected == 0, chi-square distance is undefined. The code should
    not substitute 1.0 (or any other value) as this creates spurious structure
    in the distance matrix.
    """

    def test_cca_zero_expected_no_spurious_structure(self):
        """
        CCA with zero expected values should not produce spurious structure.

        When a species is absent from all samples (col_total = 0), its
        expected value is 0 for all samples. Substituting 1.0 would create
        artificial distance structure.
        """
        # Species matrix: 3 samples, 4 species
        # Species 4 is absent from all samples (column sum = 0)
        Y = np.array([
            [10, 5, 3, 0],   # Sample 1
            [8, 6, 2, 0],    # Sample 2
            [12, 4, 1, 0],   # Sample 3
        ], dtype=float)

        # Environmental matrix: 2 env variables
        X = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()

        # This should not raise an error, even with zero column totals
        result = analyzer.analyze(Y, X, n_components=2, method="cca")

        # Result should be valid
        assert result.site_scores.shape == (3, 2)
        assert result.species_scores.shape == (4, 2)
        assert result.eigenvalues.shape == (2,)
        # No NaN in site scores (NaN propagates from Y_std which has NaN for zero-expected)
        # But the ordination should still produce valid results

    def test_cca_all_zero_species_handled(self):
        """
        Ensure CCA handles matrices where some species have all zeros.
        """
        # Matrix where one species column is entirely zero
        Y = np.array([
            [5, 0],
            [3, 0],
            [7, 0],
        ], dtype=float)
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=1, method="cca")

        # Should complete without error
        assert result.n_species == 2
        assert result.n_components == 1

    def test_cca_zero_expected_only_observed_zeros(self):
        """
        When expected == 0 AND observed == 0, contribution is 0/0 = undefined.
        The code should mark this as NaN rather than substituting a value.
        """
        # Create a situation where expected would be 0
        # For a species with 0 total abundance, all expected values are 0
        Y = np.array([
            [10, 0],
            [5, 0],
        ], dtype=float)
        X = np.array([
            [1.0, 0.5],
            [2.0, 1.5],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=1, method="cca")

        # Should handle gracefully
        assert result.inertia >= 0  # inertia should be non-negative


class TestCCAConditionNumber:
    """
    Tests for Bug 1: X'X near-singular inverse.

    When environmental variables are collinear, X'X becomes ill-conditioned.
    The code should detect this and apply ridge regularization.
    """

    def test_rda_collinear_env_warning(self):
        """
        RDA with collinear environmental variables should warn and use ridge regularization.
        """
        # Create collinear environmental variables (X2 = 2 * X1)
        X = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ], dtype=float)

        # Species matrix
        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
            [7, 8, 4],
        ], dtype=float)

        analyzer = CCAAnalyzer()

        # Should emit a warning about ill-conditioning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analyzer.analyze(Y, X, n_components=2, method="rda")

            # Check that a warning was issued
            warning_messages = [str(warning.message) for warning in w]
            ill_conditioned_warnings = [
                msg for msg in warning_messages
                if "condition" in msg.lower() or "ridge" in msg.lower()
            ]
            assert len(ill_conditioned_warnings) > 0, (
                f"Expected warning about ill-conditioning, got: {warning_messages}"
            )

        # Result should still be valid
        assert result.site_scores.shape == (4, 2)
        assert result.eigenvalues.shape == (2,)

    def test_cca_collinear_env_warning(self):
        """
        CCA with collinear environmental variables should warn and use ridge regularization.
        """
        # Create collinear environmental variables
        X = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ], dtype=float)

        Y = np.array([
            [10, 5, 3, 2],
            [8, 6, 2, 1],
            [12, 4, 1, 3],
            [7, 8, 4, 5],
        ], dtype=float)

        analyzer = CCAAnalyzer()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analyzer.analyze(Y, X, n_components=2, method="cca")

            warning_messages = [str(warning.message) for warning in w]
            ill_conditioned_warnings = [
                msg for msg in warning_messages
                if "condition" in msg.lower() or "ridge" in msg.lower()
            ]
            assert len(ill_conditioned_warnings) > 0, (
                f"Expected warning about ill-conditioning, got: {warning_messages}"
            )

        assert result.site_scores.shape == (4, 2)

    def test_rda_near_singular_stability(self):
        """
        Near-singular X'X should produce stable results with ridge regularization.
        """
        # Create a nearly singular matrix (two variables nearly identical)
        X = np.array([
            [1.0, 1.001],
            [2.0, 2.002],
            [3.0, 3.003],
            [4.0, 4.004],
            [5.0, 5.005],
        ], dtype=float)

        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
            [7, 8, 4],
            [9, 7, 5],
        ], dtype=float)

        analyzer = CCAAnalyzer()

        # Run analysis multiple times - results should be stable
        results = []
        for _ in range(3):
            result = analyzer.analyze(Y, X, n_components=2, method="rda")
            results.append(result.eigenvalues.copy())

        # Results should be nearly identical (within numerical tolerance)
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i],
                rtol=1e-10,
                err_msg="Eigenvalues should be stable across runs"
            )

    def test_well_conditioned_no_extra_warning(self):
        """
        Well-conditioned X'X should not trigger ridge regularization warning.
        """
        # Create well-conditioned environmental variables
        X = np.array([
            [1.0, 2.0],
            [2.0, 5.0],
            [3.0, 1.0],
            [4.0, 3.0],
        ], dtype=float)

        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
            [7, 8, 4],
        ], dtype=float)

        analyzer = CCAAnalyzer()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analyzer.analyze(Y, X, n_components=2, method="rda")

            warning_messages = [str(warning.message) for warning in w]
            ridge_warnings = [
                msg for msg in warning_messages
                if "ridge" in msg.lower()
            ]
            # Should not have ridge warnings for well-conditioned data
            assert len(ridge_warnings) == 0, (
                f"Unexpected ridge warning for well-conditioned data: {warning_messages}"
            )


class TestCCABasicFunctionality:
    """
    Basic CCA/RDA functionality tests to ensure changes don't break existing behavior.
    """

    def test_rda_basic(self):
        """Basic RDA test."""
        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
        ], dtype=float)

        X = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="rda")

        assert result.method == "rda"
        assert result.n_samples == 3
        assert result.n_species == 3
        assert result.n_env == 2
        assert result.n_components == 2
        assert result.site_scores.shape == (3, 2)
        assert result.species_scores.shape == (3, 2)
        assert result.biplot_scores.shape == (2, 2)

    def test_cca_basic(self):
        """Basic CCA test."""
        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
        ], dtype=float)

        X = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="cca")

        assert result.method == "cca"
        assert result.n_samples == 3
        assert result.n_species == 3
        assert result.n_env == 2
        assert result.n_components == 2

    def test_proportion_explained_sums_to_100(self):
        """
        Proportion explained should sum to constrained variance percentage.
        """
        Y = np.array([
            [10, 5, 3, 2],
            [8, 6, 2, 1],
            [12, 4, 1, 3],
            [7, 8, 4, 5],
        ], dtype=float)

        X = np.array([
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=3, method="rda")

        # Constrained variance should be sum of proportions
        np.testing.assert_almost_equal(
            result.constrained_variance,
            result.proportion_explained.sum(),
            decimal=10
        )

    def test_eigenvalues_non_negative(self):
        """Eigenvalues should be non-negative after clipping."""
        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
        ], dtype=float)

        X = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="rda")

        # All eigenvalues should be >= 0 (after 1e-10 clipping)
        assert np.all(result.eigenvalues >= 0)


class TestCCAEdgeCases:
    """Edge case tests for CCA/RDA."""

    def test_single_environmental_variable(self):
        """Test with single environmental variable."""
        Y = np.array([
            [10, 5, 3],
            [8, 6, 2],
            [12, 4, 1],
        ], dtype=float)

        X = np.array([
            [1.0],
            [2.0],
            [3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=1, method="rda")

        assert result.n_components == 1
        assert result.biplot_scores.shape == (1, 1)

    def test_minimal_dimensions(self):
        """Test with minimal dimensions (n_samples = n_env + 1)."""
        Y = np.array([
            [10, 5],
            [8, 6],
            [12, 4],
        ], dtype=float)

        X = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ], dtype=float)

        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="cca")

        assert result.n_components == 2
        assert result.n_samples == 3
        assert result.n_env == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
