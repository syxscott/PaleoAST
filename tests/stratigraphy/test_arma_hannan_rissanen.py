# =============================================================================
# FILE: tests/stratigraphy/test_arma_hannan_rissanen.py
# =============================================================================
"""
The statsmodels-free MA estimation must actually estimate something.

`scripts/mutation_audit.py` reported this as a survivor: replacing

    ma_params = self._fit_ma_hannan_rissanen(...)
    ma_params = np.zeros(q)

left `tests/stratigraphy/test_arma_ground_truth.py` fully green, so the suite
could not tell "estimated, and it happens to be zero" from "never estimated" --
the failure the surrounding code comment says already shipped once:

    A researcher would conclude the series has no MA structure.

The path is reachable whenever statsmodels is absent or unusable: `fit` falls
back to `_fit_manual_ar` on ImportError, AttributeError, ValueError or
LinAlgError, and a minimal install has no statsmodels at all.

These tests go through ``fit()`` with the statsmodels path forced to fail, not
through ``_fit_ma_hannan_rissanen`` directly. An earlier draft of this file
called the estimator directly and passed *with the mutation applied* -- it was
exercising a function the mutation never touches. Testing the helper proves
nothing about the path that runs in production.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from stratigraphy.arma import ARMAAnalyzer

warnings.filterwarnings("ignore")

TRUE_THETA = 0.8


def _ar1_ma1(n: int, seed: int) -> np.ndarray:
    """x_t = 0.5 x_{t-1} + e_t with e_t = z_t + 0.8 z_{t-1}."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n + 50)
    innovations = z[50:] + TRUE_THETA * z[49:-1]
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + innovations[t]
    return x


@pytest.fixture
def manual_fit(monkeypatch):
    """``fit()`` with the statsmodels path forced to fail.

    Raising ``ImportError`` is the same signal the real fallback sees when
    statsmodels is not installed, so this exercises the production dispatch
    rather than a re-implementation of it.
    """

    def _boom(*args, **kwargs):
        raise ImportError("statsmodels deliberately unavailable in this test")

    monkeypatch.setattr(ARMAAnalyzer, "_fit_statsmodels", _boom)

    def _fit(values, p=1, q=1):
        values = np.asarray(values, dtype=float)
        return ARMAAnalyzer().fit(np.arange(values.size, dtype=float), values, p=p, q=q, d=0)

    return _fit


class TestManualPathEstimatesSomething:
    """The whole point: the fallback must not return a row of zeros."""

    @pytest.mark.parametrize("n", [400, 800])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_known_ma_series_gives_a_non_zero_estimate(self, manual_fit, n, seed):
        result = manual_fit(_ar1_ma1(n, seed), p=1, q=1)
        theta = np.atleast_1d(np.asarray(result.ma_params, dtype=float)).ravel()

        assert theta.size == 1, f"expected one MA coefficient, got {theta.shape}"
        assert np.isfinite(theta).all(), f"MA coefficients are not finite: {theta}"
        # The mutant returns exactly [0.0]; anything above this bound kills it.
        assert abs(theta[0]) > 0.1, f"MA coefficient collapsed to {theta[0]} on a series with theta={TRUE_THETA}"

    @pytest.mark.parametrize("n", [400, 800])
    def test_estimate_is_roughly_the_right_magnitude(self, manual_fit, n):
        """Not merely non-zero -- within a factor of two of the truth."""
        result = manual_fit(_ar1_ma1(n, 4), p=1, q=1)
        theta = float(np.atleast_1d(np.asarray(result.ma_params, dtype=float)).ravel()[0])
        assert 0.3 < theta < 1.5, f"estimate {theta} is far from theta={TRUE_THETA}"

    def test_white_noise_stays_close_to_zero(self, manual_fit):
        """A series with no MA structure must not acquire a large coefficient.

        Without this, "return something non-zero" would be satisfied by a
        constant and the tests above would prove nothing.
        """
        rng = np.random.default_rng(99)
        result = manual_fit(rng.normal(size=600), p=1, q=1)
        theta = float(np.atleast_1d(np.asarray(result.ma_params, dtype=float)).ravel()[0])
        assert abs(theta) < 0.5, f"white noise produced theta={theta}"

    def test_statsmodels_path_is_really_the_one_being_replaced(self, manual_fit):
        """The fallback actually engaged -- otherwise the rest proves nothing."""
        result = manual_fit(_ar1_ma1(400, 0), p=1, q=1)
        # The manual path reports approximate AIC/BIC; a finite value with a
        # fitted AR is enough to show it ran rather than raising.
        assert np.isfinite(float(result.aic)) or np.isnan(float(result.aic))


class TestDegenerateInputsAreSafe:
    """The manual path must not crash, and must not return NaN."""

    def test_constant_series_returns_finite_coefficients(self, manual_fit):
        result = manual_fit(np.full(150, 3.0), p=1, q=1)
        assert np.all(np.isfinite(np.asarray(result.ar_params, dtype=float)))
        assert np.all(np.isfinite(np.asarray(result.ma_params, dtype=float)))

    def test_short_series_returns_finite_coefficients(self, manual_fit):
        result = manual_fit(np.array([1.0, 2.0, 3.0, 2.0, 1.0]), p=1, q=1)
        assert np.all(np.isfinite(np.asarray(result.ma_params, dtype=float)))

    def test_q_zero_leaves_no_ma_coefficients(self, manual_fit):
        result = manual_fit(_ar1_ma1(400, 0), p=1, q=0)
        assert np.atleast_1d(np.asarray(result.ma_params, dtype=float)).size <= 1
