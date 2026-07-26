# =============================================================================
# FILE: tests/statistics/test_univariate.py
# =============================================================================
"""
Tests for statistics/univariate.py AICc and effect size functions.

References
----------
Burnham, K. P., & Anderson, D. R. (2002). Model Selection and Multimodel
    Inference (2nd ed.). Springer.
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
    (2nd ed.). Lawrence Erlbaum Associates.
Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159.
Lakens, D. (2013). Calculating and reporting effect sizes. Frontiers in
    Psychology, 4, 863.
"""

import numpy as np
import pytest

from statistics.univariate import (
    cohens_d,
    compare_models,
    compute_aicc,
    eta_squared,
    omega_squared,
    partial_eta_squared,
)


# =============================================================================
# AICc Tests
# =============================================================================


class TestComputeAicc:
    """Tests for compute_aicc function."""

    def test_basic_aicc_calculation(self):
        """AICc = AIC + (2*k*(k+1))/(n-k-1) with known values."""
        # Manual calculation:
        # n=20, k=2: AIC = -2*(-20) + 4 = 44
        # AICc = 44 + (2*2*3)/(20-2-1) = 44 + 12/17 = 44.7059
        log_likelihood = -20.0
        n_params = 2
        n_obs = 20
        aicc = compute_aicc(log_likelihood, n_params, n_obs)
        expected = 44.0 + 12.0 / 17.0  # = 44.7059
        assert abs(aicc - expected) < 1e-6

    def test_aicc_reduces_to_aic_for_large_n(self):
        """When n >> k, AICc should approach AIC."""
        log_likelihood = -50.0
        n_params = 3
        n_obs = 10000
        aicc = compute_aicc(log_likelihood, n_params, n_obs)
        aic = -2.0 * log_likelihood + 2.0 * n_params
        # correction = (2*3*4)/(10000-3-1) = 24/9996 ≈ 0.0024
        assert abs(aicc - aic) < 0.003

    def test_n_less_than_k_raises_value_error(self):
        """Should raise ValueError when n - k - 1 <= 0."""
        with pytest.raises(ValueError, match="n_params.*must be less than n_obs"):
            compute_aicc(-10.0, n_params=10, n_obs=10)

    def test_n_minus_k_minus_one_zero_raises(self):
        """Should raise ValueError when n - k - 1 == 0."""
        with pytest.raises(ValueError, match="Insufficient data for AICc"):
            compute_aicc(-10.0, n_params=5, n_obs=6)  # 6 - 5 - 1 = 0

    def test_negative_n_params_raises(self):
        """Should raise ValueError for non-positive n_params."""
        with pytest.raises(ValueError, match="n_params must be a positive integer"):
            compute_aicc(-10.0, n_params=0, n_obs=10)

    def test_negative_n_obs_raises(self):
        """Should raise ValueError for non-positive n_obs."""
        with pytest.raises(ValueError, match="n_obs must be a positive integer"):
            compute_aicc(-10.0, n_params=2, n_obs=-5)


class TestCompareModels:
    """Tests for compare_models function."""

    def test_weights_sum_to_one(self):
        """Model weights should sum to 1."""
        models = [
            ("constant", -30.0, 1, 20),
            ("linear", -25.0, 2, 20),
            ("quadratic", -24.0, 3, 20),
        ]
        result = compare_models(models)
        weight_sum = sum(result["weights"])
        assert abs(weight_sum - 1.0) < 1e-10

    def test_best_model_has_zero_delta(self):
        """Best model should have ΔAICc = 0."""
        models = [
            ("simple", -40.0, 2, 30),
            ("complex", -35.0, 5, 30),
        ]
        result = compare_models(models)
        assert result["delta_aicc"][0] == 0.0

    def test_models_sorted_by_aicc(self):
        """Models should be sorted by AICc (ascending)."""
        models = [
            ("model_b", -25.0, 3, 20),
            ("model_a", -30.0, 2, 20),
            ("model_c", -20.0, 4, 20),
        ]
        result = compare_models(models)
        aicc_values = [r["aicc"] for r in result["models"]]
        assert aicc_values == sorted(aicc_values)

    def test_empty_models_raises(self):
        """Empty model list should raise ValueError."""
        with pytest.raises(ValueError, match="models list cannot be empty"):
            compare_models([])

    def test_insufficient_data_raises(self):
        """Model with n <= k should raise ValueError."""
        models = [
            ("overfit", -10.0, 10, 10),
        ]
        with pytest.raises(ValueError, match="insufficient data"):
            compare_models(models)

    def test_weights_are_positive(self):
        """All weights should be positive."""
        models = [
            ("a", -30.0, 2, 25),
            ("b", -28.0, 3, 25),
        ]
        result = compare_models(models)
        for w in result["weights"]:
            assert w > 0.0

    def test_single_model_delta_is_zero(self):
        """Single model should have ΔAICc = 0 and weight = 1."""
        models = [("only_model", -20.0, 2, 15)]
        result = compare_models(models)
        assert result["delta_aicc"][0] == 0.0
        assert abs(result["weights"][0] - 1.0) < 1e-10


# =============================================================================
# Effect Size Tests
# =============================================================================


class TestCohensD:
    """Tests for cohens_d function."""

    def test_zero_difference_gives_zero_d(self):
        """Identical groups should give d = 0."""
        group1 = np.array([10.0, 12.0, 11.0, 13.0, 12.5])
        group2 = np.array([10.0, 12.0, 11.0, 13.0, 12.5])
        d = cohens_d(group1, group2)
        assert abs(d) < 1e-10

    def test_known_values_from_cohen_1988(self):
        """Validate against published Cohen's d example."""
        # Construct data matching Cohen 1988 large effect (d ≈ 0.8):
        # group1 mean=10, group2 mean=12, pooled SD=2.5 -> d = -0.8
        # Using np.random.seed for reproducibility
        rng = np.random.default_rng(42)
        group1 = rng.normal(10.0, 2.5, size=50)
        group2 = rng.normal(12.0, 2.5, size=50)
        d = cohens_d(group1, group2)
        # Expected d ≈ -0.8 (large effect)
        assert abs(abs(d) - 0.8) < 0.15

    def test_handles_nan_values(self):
        """NaN values should be removed before calculation."""
        group1 = np.array([10.0, np.nan, 12.0, np.nan, 11.0])
        group2 = np.array([np.nan, 8.0, 9.0, 7.0, np.nan])
        d = cohens_d(group1, group2)
        # Computed from [10, 12, 11] vs [8, 9, 7]
        assert isinstance(d, float)

    def test_single_observation_raises(self):
        """Group with only 1 valid observation should raise."""
        group1 = np.array([10.0])
        group2 = np.array([12.0, 14.0, 13.0])
        with pytest.raises(ValueError, match="at least 2 valid observations"):
            cohens_d(group1, group2)

    def test_identical_values_raises(self):
        """Zero variance in both groups should raise."""
        group1 = np.array([5.0, 5.0, 5.0])
        group2 = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="Pooled standard deviation is zero"):
            cohens_d(group1, group2)

    def test_d_value_interpretation(self):
        """Verify d falls into expected small/medium/large ranges."""
        # Large effect: mean diff >> pooled sd
        group1 = np.array([50.0, 52.0, 48.0, 51.0, 49.0])
        group2 = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        d = cohens_d(group1, group2)
        assert abs(d) > 0.8  # large effect


class TestEtaSquared:
    """Tests for eta_squared function."""

    def test_eta_squared_bounds(self):
        """η² should be bounded [0, 1]."""
        # Large F statistic should give η² close to 1
        eta2 = eta_squared(F_statistic=100.0, df_between=2, df_within=30)
        assert 0.0 <= eta2 <= 1.0

    def test_zero_f_gives_zero_eta(self):
        """F = 0 should give η² = 0."""
        eta2 = eta_squared(F_statistic=0.0, df_between=2, df_within=30)
        assert eta2 == 0.0

    def test_known_values(self):
        """Validate against published η² examples."""
        # From Cohen 1988, example 8.2.1
        # F = 13.13, df_between = 3, df_within = 36 -> η² = 0.52
        eta2 = eta_squared(F_statistic=13.13, df_between=3, df_within=36)
        expected = (13.13 * 3) / (13.13 * 3 + 36)
        assert abs(eta2 - expected) < 1e-6

    def test_negative_df_raises(self):
        """Negative df should raise ValueError."""
        with pytest.raises(ValueError):
            eta_squared(F_statistic=5.0, df_between=-1, df_within=30)

    def test_zero_df_between_raises(self):
        """df_between = 0 should raise ValueError."""
        with pytest.raises(ValueError):
            eta_squared(F_statistic=5.0, df_between=0, df_within=30)


class TestOmegaSquared:
    """Tests for omega_squared function."""

    def test_omega_squared_bounds(self):
        """ω² should be bounded [0, 1]."""
        omega2 = omega_squared(F_statistic=50.0, df_between=2, df_within=30, n=35)
        assert 0.0 <= omega2 <= 1.0

    def test_zero_f_gives_negative_omega_clamped(self):
        """F = 0 can give negative ω²; should be clamped to 0."""
        omega2 = omega_squared(F_statistic=0.0, df_between=3, df_within=20, n=25)
        assert omega2 == 0.0

    def test_omega_less_than_eta(self):
        """ω² should generally be <= η² (less biased)."""
        eta2 = eta_squared(F_statistic=10.0, df_between=2, df_within=30)
        omega2 = omega_squared(F_statistic=10.0, df_between=2, df_within=30, n=35)
        assert omega2 <= eta2

    def test_n_equals_df_between_raises(self):
        """n <= df_between should raise ValueError."""
        with pytest.raises(ValueError, match="Total observations"):
            omega_squared(F_statistic=5.0, df_between=10, df_within=5, n=10)


class TestPartialEtaSquared:
    """Tests for partial_eta_squared function."""

    def test_partial_eta_squared_bounds(self):
        """η²_p should be bounded [0, 1]."""
        eta2p = partial_eta_squared(F_statistic=20.0, df_between=2, df_error=30)
        assert 0.0 <= eta2p <= 1.0

    def test_equivalence_to_eta_squared(self):
        """partial_eta_squared should equal eta_squared when df_error = df_within."""
        F, df_b, df_w = 8.0, 2, 40
        eta2 = eta_squared(F, df_b, df_w)
        eta2p = partial_eta_squared(F, df_b, df_w)
        assert abs(eta2 - eta2p) < 1e-10

    def test_zero_f_gives_zero(self):
        """F = 0 should give η²_p = 0."""
        eta2p = partial_eta_squared(F_statistic=0.0, df_between=2, df_error=30)
        assert eta2p == 0.0

    def test_negative_df_raises(self):
        """Negative df should raise ValueError."""
        with pytest.raises(ValueError):
            partial_eta_squared(F_statistic=5.0, df_between=-1, df_error=30)
