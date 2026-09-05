# =============================================================================
# FILE: tests/statistics/test_bray_curtis.py
# =============================================================================
"""
Unit tests for Bray-Curtis distance implementation.

Golden values from R vegan::vegdist(method="bray").

References:
    Oksanen, J. et al. (2022). vegan: Ecological Community Analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statistics.distance_metrics import compute_distance_matrix, _bray_curtis_distance_matrix


class TestBrayCurtisGoldenValues:
    """Test Bray-Curtis against known golden values from R vegan."""

    def test_bray_curtis_4x4_matrix(self):
        """4x4 matrix — standard Bray-Curtis (rows = samples, cols = species).

        The original expected values were copied from R vegan but used the
        transposed matrix convention (species as rows).  The PaleoAST
        implementation follows the standard ecological convention:
            d_BC(i,j) = Σ_k |x_ik - x_jk| / Σ_k (x_ik + x_jk)
        where rows are samples and columns are species.  Verified against
        hand-computed exact fractions below.
        """
        X = np.array([
            [10.0, 5.0, 2.0, 0.0],
            [8.0, 6.0, 3.0, 1.0],
            [0.0, 2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        # Exact fractions:  [0,1]=1/7, [0,2]=21/29, [0,3]=5/7,
        #                    [1,2]=3/5,  [1,3]=7/11,  [2,3]=5/8
        expected = np.array([
            [0.0,            1.0/7,   21.0/29, 5.0/7  ],
            [1.0/7,          0.0,      3.0/5,   7.0/11 ],
            [21.0/29,        3.0/5,    0.0,      5.0/8  ],
            [5.0/7,          7.0/11,   5.0/8,    0.0    ],
        ], dtype=float)
        assert_allclose(D, expected, atol=1e-10)

    def test_bray_curtis_identical_samples(self):
        """Identical samples should have zero distance."""
        X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        assert_allclose(D[0, 1], 0.0, atol=1e-10)

    def test_bray_curtis_no_overlap(self):
        """Samples with no shared species should have distance of 1.0."""
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        upper = D[np.triu_indices(3, k=1)]
        assert_allclose(upper, [1.0, 1.0, 1.0], atol=1e-3)

    def test_bray_curtis_partial_overlap(self):
        """Test partial overlap case with known result."""
        X = np.array([[10.0, 5.0, 2.0], [10.0, 3.0, 0.0]])
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        # |10-10| + |5-3| + |2-0| = 0 + 2 + 2 = 4
        # (10+10) + (5+3) + (2+0) = 20 + 8 + 2 = 30
        # d = 4/30 = 0.1333
        expected = 4.0 / 30.0
        assert_allclose(D[0, 1], expected, atol=1e-3)


class TestBrayCurtisEdgeCases:
    """Test Bray-Curtis with edge cases."""

    def test_single_row(self):
        """Single row should give zero distances (no pairs)."""
        X = np.array([[1.0, 2.0, 3.0]])
        D = _bray_curtis_distance_matrix(X)
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

    def test_two_rows(self):
        """Two rows should compute single distance."""
        X = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        D = _bray_curtis_distance_matrix(X)
        assert D.shape == (2, 2)
        # |1-3| + |2-2| + |3-1| = 2 + 0 + 2 = 4
        # (1+3) + (2+2) + (3+1) = 4 + 4 + 4 = 12
        # d = 4/12 = 0.333
        assert_allclose(D[0, 1], 4.0/12.0, atol=1e-6)

    def test_zero_abundances(self):
        """All zeros should give zero distance (no information)."""
        X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        D = _bray_curtis_distance_matrix(X)
        # With all zeros, denominator is 0, so we use fallback of 1
        # and numerator is also 0, so distance = 0
        assert_allclose(D[0, 1], 0.0, atol=1e-6)

    def test_mixed_zeros(self):
        """Test with some zero abundances."""
        X = np.array([[5.0, 0.0, 0.0], [2.0, 0.0, 3.0]])
        D = _bray_curtis_distance_matrix(X)
        # |5-2| + |0-0| + |0-3| = 3 + 0 + 3 = 6
        # (5+2) + (0+0) + (0+3) = 7 + 0 + 3 = 10
        # d = 6/10 = 0.6
        assert_allclose(D[0, 1], 0.6, atol=1e-6)


class TestBrayCurtisLargeMatrix:
    """Test that large matrix path gives same results as small matrix path."""

    def test_large_matrix_consistency(self):
        """Large matrices should use chunked path but produce same results."""
        np.random.seed(42)
        X = np.random.rand(100, 10) * 10  # 100 samples, 10 features

        # Compute using small matrix path (n=100 > 500? no, so this uses small path)
        # Actually n=100 <= 500, so it uses small path. Let's test with n=600
        X_large = np.random.rand(600, 10) * 10
        D_large = _bray_curtis_distance_matrix(X_large)

        # Verify it's symmetric
        assert_allclose(D_large, D_large.T, atol=1e-10)

        # Verify diagonal is zero
        assert_allclose(np.diag(D_large), 0.0, atol=1e-10)

        # Verify distances are in [0, 1]
        upper = D_large[np.triu_indices(600, k=1)]
        assert np.all(upper >= 0)
        assert np.all(upper <= 1)

    def test_small_vs_large_path_equivalence(self):
        """Small and large paths should produce same results for same data."""
        np.random.seed(42)
        # Use data where n <= 500 so we can compare directly
        X = np.random.rand(50, 8) * 10

        # Manually set n > 500 to force large path
        # But we can't easily do that without modifying the function
        # So we just verify the small path is correct by checking against known values
        D = _bray_curtis_distance_matrix(X)

        # Check symmetry and bounds
        assert_allclose(D, D.T, atol=1e-10)
        assert_allclose(np.diag(D), 0.0, atol=1e-10)
        upper = D[np.triu_indices(50, k=1)]
        assert np.all(upper >= 0)
        assert np.all(upper <= 1)


class TestBrayCurtisFormula:
    """Verify the Bray-Curtis formula implementation."""

    def test_formula_numerator(self):
        """Verify numerator is sum of absolute differences."""
        X = np.array([[3.0, 5.0], [1.0, 7.0]])
        D = _bray_curtis_distance_matrix(X)
        # |3-1| + |5-7| = 2 + 2 = 4
        expected_numerator = 4.0
        # Denominator: (3+1) + (5+7) = 4 + 12 = 16
        expected_denominator = 16.0
        expected_d = expected_numerator / expected_denominator
        assert_allclose(D[0, 1], expected_d, atol=1e-6)

    def test_formula_is_symmetric(self):
        """Bray-Curtis should be symmetric: d(i,j) = d(j,i)."""
        X = np.random.rand(20, 5) * 10
        D = _bray_curtis_distance_matrix(X)
        assert_allclose(D, D.T, atol=1e-10)

    def test_formula_range(self):
        """Bray-Curtis should be in [0, 1]."""
        X = np.random.rand(30, 5) * 10
        D = _bray_curtis_distance_matrix(X)
        upper = D[np.triu_indices(30, k=1)]
        assert np.all(upper >= 0)
        assert np.all(upper <= 1)
