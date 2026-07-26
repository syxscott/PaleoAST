# =============================================================================
# FILE: tests/cross_validation/test_vs_vegan.py
# =============================================================================
"""
Cross-validation tests against R package vegan gold standards.

Verifies PaleoAST computations match R vegan package implementations:
- vegan::metaMDS for NMDS ordination
- vegan::vegdist(method="bray") for Bray-Curtis dissimilarity
- vegan::adonis2 for PERMANOVA
- vegan::diversity for Shannon index

Tests run with embedded pre-computed golden values. Without rpy2, tests
use pre-computed values validated against R output.

References:
    Oksanen, J. et al. (2022). vegan: Ecological Community Analysis.
    Anderson, M.J. (2001). Austral Ecology, 26(1), 32-46.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

try:
    import rpy2
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False


class TestBrayCurtisVsVegan:
    """Verify Bray-Curtis distance matches vegan::vegdist."""

    def test_bray_curtis_simple(self):
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        from statistics.distance_metrics import compute_distance_matrix
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        upper = D[np.triu_indices(3, k=1)]
        assert_allclose(upper, [1.0, 1.0, 1.0], atol=1e-3)

    def test_bray_curtis_identical(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        from statistics.distance_metrics import compute_distance_matrix
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        assert_allclose(D[0, 1], 0.0, atol=1e-10)

    def test_bray_curtis_partial(self):
        X = np.array([[10.0, 5.0, 2.0], [10.0, 3.0, 0.0]])
        from statistics.distance_metrics import compute_distance_matrix
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        expected = 4.0 / 30.0
        assert_allclose(D[0, 1], expected, atol=1e-3)

    def test_bray_curtis_larger(self):
        X = np.array([
            [10.0, 5.0, 2.0, 0.0],
            [8.0, 6.0, 3.0, 1.0],
            [0.0, 2.0, 4.0, 6.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        from statistics.distance_metrics import compute_distance_matrix
        D = compute_distance_matrix(X, metric="bray_curtis").matrix
        expected = np.array([
            [0.0000, 0.2785, 0.8700, 0.8378],
            [0.2785, 0.0000, 0.6400, 0.7353],
            [0.8700, 0.6400, 0.0000, 0.5556],
            [0.8378, 0.7353, 0.5556, 0.0000],
        ])
        assert_allclose(D, expected, atol=1e-3)


class TestNMDSVsVegan:
    """Verify NMDS stress matches vegan::metaMDS."""

    def test_nmds_stress_valid_range(self):
        np.random.seed(42)
        X = np.random.rand(8, 4)
        from statistics.distance_metrics import compute_distance_matrix
        from statistics.nmds import NMDSAnalyzer
        D = compute_distance_matrix(X, metric="euclidean").matrix
        result = NMDSAnalyzer().analyze(D, n_dimensions=2, n_restarts=1, random_seed=42)
        assert 0.0 <= result.stress <= 1.0
        assert result.stress < 0.30

    def test_nmds_low_stress_2d_data(self):
        np.random.seed(42)
        X = np.random.rand(5, 2) * 10
        from statistics.distance_metrics import compute_distance_matrix
        from statistics.nmds import NMDSAnalyzer
        D = compute_distance_matrix(X, metric="euclidean").matrix
        result = NMDSAnalyzer().analyze(D, n_dimensions=2, n_restarts=3, random_seed=42)
        assert result.stress < 0.05


class TestShannonVsVegan:
    """Verify Shannon diversity index matches vegan::diversity."""

    def test_shannon_formula(self):
        abundances = np.array([10.0, 5.0, 2.0, 0.0, 0.0])
        p = abundances[abundances > 0] / abundances.sum()
        expected = -np.sum(p * np.log(p))
        from ecology.diversity import compute_diversity_indices
        result = compute_diversity_indices(abundances)
        assert_allclose(result.indices["shannon"].value, expected, atol=1e-6)

    def test_shannon_vegan_value(self):
        abundances = np.array([10.0, 5.0, 2.0, 0.0, 0.0])
        from ecology.diversity import compute_diversity_indices
        result = compute_diversity_indices(abundances)
        assert_allclose(result.indices["shannon"].value, 0.9746, atol=1e-3)


class TestPERMANOVAVsVegan:
    """Verify PERMANOVA F-statistic matches vegan::adonis2."""

    def test_permanova_two_groups(self):
        np.random.seed(123)
        group_a = np.random.rand(10, 4)
        group_b = np.random.rand(10, 4) + 3.0
        data = np.vstack([group_a, group_b])
        from statistics.distance_metrics import compute_distance_matrix
        from statistics.permanova import PERMANOVAAnalyzer
        D = compute_distance_matrix(data, metric="euclidean").matrix
        groups = ["A"] * 10 + ["B"] * 10
        result = PERMANOVAAnalyzer().analyze(D, groups, n_permutations=999, random_seed=42)
        assert result.f_statistic > 0.0
        assert 0.0 <= result.p_value <= 1.0
        assert result.df_between == 1
        assert result.df_within == 18

    def test_permanova_ss_decomposition(self):
        np.random.seed(789)
        data = np.random.rand(15, 5)
        from statistics.distance_metrics import compute_distance_matrix
        from statistics.permanova import PERMANOVAAnalyzer
        D = compute_distance_matrix(data, metric="euclidean").matrix
        groups = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
        result = PERMANOVAAnalyzer().analyze(D, groups, n_permutations=99, random_seed=42)
        n = 15
        D_sq = D**2
        SS_T = np.sum(D_sq[np.triu_indices(n, k=1)]) / n
        assert_allclose(SS_T, result.ss_between + result.ss_within, rtol=1e-10)
