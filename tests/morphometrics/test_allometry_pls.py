"""Tests for W5 allometry/2B-PLS upgrades: CAC/RSC, PLS permutation + Z + RV, field rename."""

import numpy as np
import pytest

from morphometrics.allometry import AllometryAnalyzer, IntegrationAnalyzer


def _allometric_configs(n=30, seed=5):
    """Configurations whose shape shifts linearly with centroid size."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    base = rng.normal(size=(6, 2))
    pattern = rng.normal(size=(6, 2))
    configs = []
    for xi in x:
        cs_factor = np.exp(0.15 * xi)  # size varies with x
        shape = base + 0.4 * xi * pattern + rng.normal(scale=0.01, size=(6, 2))
        shape = shape - shape.mean(axis=0)
        configs.append(shape * cs_factor)
    return np.array(configs), x


class TestAllometryCAC:
    def test_centroid_size_translation_invariance(self):
        configs, _ = _allometric_configs()
        analyzer = AllometryAnalyzer()
        r1 = analyzer.analyze_allometry(configs)
        moved = configs + np.array([500.0, -321.0])
        r2 = analyzer.analyze_allometry(moved)
        np.testing.assert_allclose(r1.centroid_sizes, r2.centroid_sizes, rtol=1e-8)
        assert r1.r_squared == pytest.approx(r2.r_squared, rel=1e-6)

    def test_strong_allometry_detected(self):
        configs, _ = _allometric_configs()
        result = AllometryAnalyzer().analyze_allometry(configs)
        assert result.r_squared > 0.8
        assert result.isometry_pvalue < 0.001

    def test_cac_fields_present_and_bounded(self):
        configs, _ = _allometric_configs()
        result = AllometryAnalyzer().analyze_allometry(configs)
        assert result.cac_scores is not None
        assert result.cac_scores.shape == (configs.shape[0],)
        assert 0.0 < result.cac_variance < 1.0

    def test_cac_tracks_size(self):
        configs, _ = _allometric_configs()
        result = AllometryAnalyzer().analyze_allometry(configs)
        r = np.corrcoef(result.cac_scores, result.log_centroid_sizes)[0, 1]
        assert abs(r) > 0.95

    def test_rsc_axes_uncorrelated(self):
        configs, _ = _allometric_configs()
        result = AllometryAnalyzer().analyze_allometry(configs)
        assert result.rsc_scores is not None
        # RSC scores come from a PCA, so axes are mutually uncorrelated
        # and centred
        S = result.rsc_scores
        np.testing.assert_allclose(S.mean(axis=0), 0.0, atol=1e-9)
        C = np.corrcoef(S.T)
        off = C - np.eye(C.shape[0])
        assert np.max(np.abs(off)) < 1e-8
        assert np.all(result.rsc_proportion >= 0.0)
        assert result.rsc_proportion.sum() <= 1.0 + 1e-9

    def test_variance_decomposition(self):
        configs, _ = _allometric_configs()
        result = AllometryAnalyzer().analyze_allometry(configs)
        # CAC + residual shape account for all centred shape variance
        n = configs.shape[0]
        Yc = configs.reshape(n, -1) - configs.reshape(n, -1).mean(axis=0)
        total = np.sum(Yc**2)
        cac_var = np.sum(result.cac_scores**2) / total
        assert cac_var == pytest.approx(result.cac_variance, rel=1e-8)
        assert 0.0 <= result.cac_variance <= 1.0

    def test_no_allometry_gives_small_cac_variance(self):
        rng = np.random.default_rng(17)
        configs = rng.normal(size=(25, 6, 2))
        result = AllometryAnalyzer().analyze_allometry(configs)
        assert result.cac_variance < 0.4


class TestPLSIntegration:
    @staticmethod
    def _blocks(seed=3, n=30, coupled=True):
        rng = np.random.default_rng(seed)
        f = rng.normal(size=n)
        u = rng.normal(size=4)
        v = rng.normal(size=4)
        A = np.outer(f, u) + rng.normal(scale=0.1, size=(n, 4))
        if coupled:
            B = np.outer(f, v) + rng.normal(scale=0.1, size=(n, 4))
        else:
            B = rng.normal(size=(n, 4))
        return A, B

    def test_field_renamed(self):
        A, B = self._blocks()
        result = IntegrationAnalyzer().analyze_pls(A, B)
        assert hasattr(result, "pls_correlations")
        assert not hasattr(result, "rv_coefficients")

    def test_coupled_blocks_high_r1(self):
        A, B = self._blocks()
        result = IntegrationAnalyzer().analyze_pls(A, B)
        assert result.integration_index > 0.9
        assert result.integration_index == pytest.approx(result.pls_correlations[0], rel=1e-12)

    def test_permutation_significant_for_coupled(self):
        A, B = self._blocks()
        result = IntegrationAnalyzer().analyze_pls(A, B, permutations=199, seed=1)
        assert result.pls1_pvalue is not None and result.pls1_pvalue <= 0.05
        assert result.pls1_z is not None and result.pls1_z > 2.0
        assert len(result.random_correlations) == 199
        # permutation null is centred near zero for coupled data
        assert abs(np.mean(result.random_correlations)) < 0.5

    def test_permutation_null_for_independent(self):
        A, B = self._blocks(seed=4, coupled=False)
        result = IntegrationAnalyzer().analyze_pls(A, B, permutations=199, seed=4)
        assert result.pls1_pvalue > 0.05

    def test_no_permutation_by_default(self):
        A, B = self._blocks()
        result = IntegrationAnalyzer().analyze_pls(A, B)
        assert result.pls1_pvalue is None
        assert result.pls1_z is None
        assert result.random_correlations is None

    def test_rv_coefficient_bounds(self):
        A, _ = self._blocks()
        same = IntegrationAnalyzer().analyze_pls(A, A.copy())
        assert same.rv_coefficient == pytest.approx(1.0, abs=1e-8)
        B = self._blocks(seed=4, coupled=False)[1]
        indep = IntegrationAnalyzer().analyze_pls(A, B)
        assert 0.0 <= indep.rv_coefficient < 0.9
        coupled_B = self._blocks()[1]
        coup = IntegrationAnalyzer().analyze_pls(A, coupled_B)
        assert coup.rv_coefficient > indep.rv_coefficient

    def test_to_dict_keys(self):
        A, B = self._blocks()
        d = IntegrationAnalyzer().analyze_pls(A, B, permutations=29, seed=2).to_dict()
        assert "pls_correlations" in d and "rv_coefficient" in d and "pls1_pvalue" in d
        assert "rv_coefficients" not in d
        assert d["random_correlations"] is not None

    def test_summary_runs(self):
        A, B = self._blocks()
        text = IntegrationAnalyzer().analyze_pls(A, B, permutations=29, seed=2).summary()
        assert "PLS1 correlation r1" in text
        assert "Escoufier RV" in text
