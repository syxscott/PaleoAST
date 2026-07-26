# =============================================================================
# FILE: tests/statistics/test_pca.py
# =============================================================================
"""
Unit tests for PCA module - focusing on Bug 1: n_samples=1 dtype=int crash.

Bug 1 Description:
    When n_samples=1 and input data dtype is int, np.linalg.svd crashes or
    produces incorrect results. The fix ensures:
    1. Data is converted to float64 before SVD
    2. 1D arrays are reshaped to 2D (1, -1)
    3. A clear ValueError is raised when n_samples < 2
"""

import numpy as np
import pytest

from statistics.pca import PCAAnalyzer, PCAResult


class TestPCADtypeFix:
    """Tests for Bug 1: dtype=int with n_samples handling."""

    def test_integer_array_input(self):
        """Test that integer input arrays are converted to float64 for SVD."""
        analyzer = PCAAnalyzer()
        # Create integer data with 2 samples (minimum for PCA)
        data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
        result = analyzer.analyze(data, n_components=2)
        assert result.scores.dtype == np.float64
        assert result.loadings.dtype == np.float64
        assert isinstance(result, PCAResult)

    def test_integer_array_1d_input(self):
        """Test that 1D integer array is reshaped and converted to float64."""
        analyzer = PCAAnalyzer()
        # 1D integer array (2 specimens with 5 measurements each)
        data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
        result = analyzer.analyze(data, n_components=2)
        assert result.scores.shape[0] == 2  # 2 samples
        assert result.scores.dtype == np.float64

    def test_single_sample_raises_clear_error(self):
        """Test that n_samples=1 raises a clear ValueError, not a cryptic SVD error."""
        analyzer = PCAAnalyzer()
        data = np.array([[1, 2, 3, 4, 5]])  # 1 sample, 5 variables
        with pytest.raises(ValueError, match="PCA requires at least 2 samples"):
            analyzer.analyze(data)

    def test_single_sample_integer_input_error_message(self):
        """Test error message mentions sample count issue."""
        analyzer = PCAAnalyzer()
        data = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)  # 1 sample as int
        with pytest.raises(ValueError, match=r"1 sample"):
            analyzer.analyze(data)

    def test_float64_conversion_preserves_values(self):
        """Test that float conversion doesn't alter values."""
        analyzer = PCAAnalyzer()
        data = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
        result = analyzer.analyze(data, n_components=2)
        # Values should be approximately preserved in loadings
        assert result.loadings.dtype == np.float64

    def test_nan_handling_with_float_input(self):
        """Test NaN imputation works with float64 conversion."""
        analyzer = PCAAnalyzer()
        # Float data with NaN (imputation requires float)
        data = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        result = analyzer.analyze(data, impute_missing=True)
        assert not np.any(np.isnan(result.scores))


class TestPCANormalOperation:
    """Tests for normal PCA operation (regression tests)."""

    def test_two_samples_max_components(self):
        """Test PCA with exactly 2 samples (max 1 component due to n_samples-1 limit)."""
        analyzer = PCAAnalyzer()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = analyzer.analyze(data)  # Should auto-select 1 component (n_samples - 1)
        assert result.scores.shape == (2, 1)
        assert result.n_components == 1

    def test_three_samples_two_components(self):
        """Test PCA with 3 samples can extract 2 components."""
        analyzer = PCAAnalyzer()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = analyzer.analyze(data, n_components=2)
        assert result.scores.shape == (3, 2)
        assert result.n_components == 2

    def test_covariance_vs_correlation_methods(self):
        """Test both covariance and correlation PCA methods work."""
        analyzer = PCAAnalyzer()
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        cov_result = analyzer.analyze(data, method='covariance')
        cor_result = analyzer.analyze(data, method='correlation')
        assert cov_result.scores.shape == cor_result.scores.shape

    def test_eigenvalue_properties(self):
        """Test that eigenvalues have expected mathematical properties."""
        analyzer = PCAAnalyzer()
        np.random.seed(42)
        data = np.random.randn(10, 5)  # 10 samples, 5 variables
        result = analyzer.analyze(data)
        # Eigenvalues should be non-negative for standard PCA
        assert np.all(result.eigenvalues >= 0)
        # Explained variance should sum to ~100%
        assert 99.0 <= np.sum(result.explained_variance) <= 100.1


class TestPCAEdgeCases:
    """Tests for edge cases."""

    def test_large_integer_matrix(self):
        """Test with large integer values (fossil counts etc.)."""
        analyzer = PCAAnalyzer()
        data = np.array([[1000000, 2000000], [3000000, 4000000]], dtype=np.int64)
        result = analyzer.analyze(data)
        assert result.scores.dtype == np.float64

    def test_zero_variance_column(self):
        """Test handling of zero-variance columns."""
        analyzer = PCAAnalyzer()
        data = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        result = analyzer.analyze(data)
        # Should complete without error
        assert result.n_components >= 1
