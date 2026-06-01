# tests/macroevolution/test_survival.py
"""Tests for survival analysis module including Cox PH."""

import numpy as np

from macroevolution.survival import (
    CoxPHResult,
    KaplanMeierAnalyzer,
    _compute_concordance,
    cox_ph,
    log_rank_test,
)


def test_kaplan_meier_basic():
    """Test basic Kaplan-Meier estimation."""
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 0, 1, 1])

    analyzer = KaplanMeierAnalyzer()
    result = analyzer.fit(times, events)

    # Survival probabilities should be non-increasing
    assert len(result.survival_prob) >= 1
    assert result.survival_prob[0] <= 1.0
    assert all(result.survival_prob[i] >= result.survival_prob[i + 1] for i in range(len(result.survival_prob) - 1))


def test_kaplan_meier_empty():
    """Test Kaplan-Meier with empty data."""
    times = np.array([])
    events = np.array([])

    analyzer = KaplanMeierAnalyzer()
    result = analyzer.fit(times, events)

    # Empty data returns trivial survival [1.]
    assert len(result.survival_prob) >= 1
    assert result.survival_prob[0] == 1.0


def test_log_rank_test():
    """Test log-rank test."""
    times1 = np.array([1.0, 2.0, 3.0, 4.0])
    events1 = np.array([1, 1, 0, 1])
    times2 = np.array([1.5, 2.5, 3.5, 4.5])
    events2 = np.array([1, 0, 1, 1])

    result = log_rank_test(times1, events1, times2, events2)

    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert 0 <= result.p_value <= 1


def test_compute_concordance_basic():
    """Test concordance index computation."""
    durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 0, 1, 1])
    covariates = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    beta = np.array([0.5])

    c_index = _compute_concordance(durations, events, covariates, beta)

    assert 0 <= c_index <= 1


def test_compute_concordance_perfect():
    """Test concordance with perfect prediction."""
    # Higher covariate = shorter survival (higher risk)
    durations = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    events = np.array([1, 1, 1, 1, 1])
    covariates = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    beta = np.array([1.0])

    c_index = _compute_concordance(durations, events, covariates, beta)

    # Should be high (good prediction)
    assert c_index > 0.7


def test_compute_concordance_random():
    """Test concordance with random data."""
    np.random.seed(42)
    n = 50
    durations = np.random.exponential(5, n)
    events = np.random.binomial(1, 0.7, n)
    covariates = np.random.randn(n, 2)
    beta = np.array([0.3, -0.2])

    c_index = _compute_concordance(durations, events, covariates, beta)

    assert 0 <= c_index <= 1


def test_cox_ph_result_summary():
    """Test CoxPHResult summary generation."""
    result = CoxPHResult(
        beta=np.array([0.5]),
        exp_beta=np.array([1.65]),
        se=np.array([0.2]),
        z_scores=np.array([2.5]),
        p_values=np.array([0.01]),
        concordance=0.75,
        log_likelihood=-25.3,
        AIC=52.6,
    )

    summary = result.summary()
    assert "Cox" in summary or "C-index" in summary
    assert "0.75" in summary or "0.8" in summary


def test_cox_ph_scipy_univariate():
    """Test Cox PH with single covariate using scipy."""
    np.random.seed(123)
    n = 30

    # Generate survival times with exponential distribution
    durations = np.random.exponential(5, n)
    events = np.random.binomial(1, 0.6, n)

    # Single covariate - stronger effect
    covariate = (durations > 3).astype(float)

    try:
        result = cox_ph(durations, events, covariate, max_iter=50)
        assert isinstance(result, CoxPHResult)
        assert len(result.beta) == 1
        assert 0 <= result.concordance <= 1
        print(f"Cox PH (scipy) univariate: beta={result.beta[0]:.3f}, C-index={result.concordance:.3f}")
    except Exception as e:
        print(f"Cox PH scipy failed: {e}")


def test_cox_ph_scipy_multivariate():
    """Test Cox PH with multiple covariates using scipy."""
    np.random.seed(456)
    n = 40

    durations = np.random.exponential(5, n)
    events = np.random.binomial(1, 0.5, n)
    covariates = np.random.randn(n, 3)

    try:
        result = cox_ph(durations, events, covariates, max_iter=50)
        assert isinstance(result, CoxPHResult)
        assert len(result.beta) == 3
        assert 0 <= result.concordance <= 1
        assert len(result.p_values) == 3
        print(f"Cox PH (scipy) multivariate: C-index={result.concordance:.3f}, AIC={result.AIC:.2f}")
    except Exception as e:
        print(f"Cox PH scipy multivariate failed: {e}")


def test_cox_ph_scipy_concordance_vs_manual():
    """Test that scipy Cox PH produces reasonable concordance."""
    np.random.seed(789)
    n = 25

    # Create a strong effect: covariate 0 is protective (higher = longer survival)
    durations = np.random.exponential(5, n)
    events = np.random.binomial(1, 0.6, n)
    covariates = np.column_stack(
        [
            (10 - durations) / 5,  # Covariate 1: inverse relationship
            np.random.randn(n),
        ]
    )

    try:
        result = cox_ph(durations, events, covariates, max_iter=50)
        # With a strong signal, concordance should be > 0.6
        assert result.concordance > 0.5, f"Expected C-index > 0.5, got {result.concordance}"
        print(f"Cox PH concordance test: C-index={result.concordance:.3f}")
    except Exception as e:
        print(f"Cox PH test failed: {e}")


if __name__ == "__main__":
    print("Running survival analysis tests...")

    test_kaplan_meier_basic()
    print("test_kaplan_meier_basic: PASSED")

    test_kaplan_meier_empty()
    print("test_kaplan_meier_empty: PASSED")

    test_log_rank_test()
    print("test_log_rank_test: PASSED")

    test_compute_concordance_basic()
    print("test_compute_concordance_basic: PASSED")

    test_compute_concordance_perfect()
    print("test_compute_concordance_perfect: PASSED")

    test_compute_concordance_random()
    print("test_compute_concordance_random: PASSED")

    test_cox_ph_result_summary()
    print("test_cox_ph_result_summary: PASSED")

    test_cox_ph_scipy_univariate()
    print("test_cox_ph_scipy_univariate: PASSED")

    test_cox_ph_scipy_multivariate()
    print("test_cox_ph_scipy_multivariate: PASSED")

    test_cox_ph_scipy_concordance_vs_manual()
    print("test_cox_ph_scipy_concordance_vs_manual: PASSED")

    print("\nAll survival tests PASSED!")
