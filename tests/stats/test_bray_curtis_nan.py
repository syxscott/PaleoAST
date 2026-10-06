# =============================================================================
# FILE: tests/stats/test_bray_curtis_nan.py
# =============================================================================
"""
Tests for Bray-Curtis and Jaccard edge cases.

Bray-Curtis edge cases (defect 5):
  - Two all-zero samples: denominator = 0, numerator = 0; the previous
    implementation returned 0 (the artificial "identical" answer) which
    silently biases downstream PCoA / NMDS. We now return NaN.
  - Negative abundances: Bray-Curtis requires non-negative data; the
    previous implementation silently propagated negatives. We now raise.

Jaccard edge cases (defect 6):
  - Quantitative (non-binary) inputs are silently binarised by scipy. We
    now warn the user.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings

import numpy as np
from numpy.testing import assert_allclose

from stats.distance_metrics import (
    _bray_curtis_distance_matrix,
    compute_distance_matrix,
)


class TestBrayCurtisZeroZeroPair:
    """When both samples are all zeros, Bray-Curtis must return NaN."""

    def test_two_zero_rows_return_nan(self):
        X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        D = _bray_curtis_distance_matrix(X)
        assert np.isnan(D[0, 1])
        assert np.isnan(D[1, 0])
        assert D[0, 0] == 0.0  # diagonal stays zero

    def test_three_rows_with_two_zero(self):
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
            ]
        )
        D = _bray_curtis_distance_matrix(X)
        assert np.isnan(D[0, 1])
        assert np.isnan(D[1, 0])
        # Non-zero pairs are still defined. d_BC(0, 2) where x_0 is all
        # zeros and x_2 = [1, 2, 3]: numerator = 1+2+3 = 6, denominator
        # = 1+2+3 = 6 (the all-zero row contributes 0 to both sums), so
        # d_BC = 1.0 -- the two samples have no overlap (which is the
        # ecologically correct interpretation: an empty sample is fully
        # dissimilar to a non-empty one).
        assert_allclose(D[0, 2], 1.0, atol=1e-6)
        assert_allclose(D[1, 2], 1.0, atol=1e-6)

    def test_mixed_zeros_returns_correct_value(self):
        """Mixed-zero pairs (some zeros, some non-zeros) still compute normally."""
        X = np.array([[5.0, 0.0, 0.0], [2.0, 0.0, 3.0]])
        D = _bray_curtis_distance_matrix(X)
        # |5-2|+|0-0|+|0-3| = 3+0+3=6; sum=10; d=0.6
        assert_allclose(D[0, 1], 0.6, atol=1e-6)


class TestBrayCurtisRejectsNegative:
    """Bray-Curtis must raise on negative abundances."""

    def test_negative_raises(self):
        X = np.array([[1.0, -2.0, 3.0], [0.0, 1.0, 2.0]])
        with pytest_raises_value_error():
            _bray_curtis_distance_matrix(X)

    def test_single_negative_raises(self):
        X = np.array([[0.0, 0.0, 0.0], [-1e-9, 0.0, 0.0]])
        with pytest_raises_value_error():
            _bray_curtis_distance_matrix(X)


class TestJaccardBinarizationWarning:
    """Jaccard on quantitative input must warn the user."""

    def test_quantitative_input_warns(self):
        """[1.5, 0, 2.7] should trigger the binarisation warning."""
        X = np.array(
            [
                [1.5, 0.0, 2.7],
                [0.5, 1.0, 0.0],
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_distance_matrix(X, metric="jaccard")
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any(
                "Jaccard" in str(x.message) for x in user_warnings
            ), f"Expected a Jaccard-related UserWarning, got: {[str(x.message) for x in w]}"

    def test_binary_input_no_warning(self):
        """A pure 0/1 matrix must NOT trigger the warning."""
        X = np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_distance_matrix(X, metric="jaccard")
            jaccard_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning) and "Jaccard" in str(x.message)
            ]
            assert len(jaccard_warnings) == 0, (
                f"Unexpected Jaccard warning on binary input: "
                f"{[str(x.message) for x in jaccard_warnings]}"
            )

    def test_count_like_input_warns_and_suggests_bray_curtis(self):
        """Integer counts (>=2) must trigger the warning and mention Bray-Curtis."""
        X = np.array(
            [
                [10, 0, 5],
                [0, 3, 7],
            ]
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_distance_matrix(X, metric="jaccard")
            jaccard_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning) and "Jaccard" in str(x.message)
            ]
            assert len(jaccard_warnings) == 1
            assert "bray_curtis" in str(jaccard_warnings[0].message).lower()


# Tiny helper so we don't pull in pytest just for this file
def pytest_raises_value_error():
    """Context manager for ``with pytest_raises_value_error():``."""
    import contextlib

    return contextlib.suppress() if False else _Raises()


class _Raises:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("Expected ValueError but no exception was raised")
        if not issubclass(exc_type, ValueError):
            raise AssertionError(
                f"Expected ValueError but got {exc_type.__name__}: {exc}"
            )
        return True
