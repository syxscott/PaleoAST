# =============================================================================
# FILE: tests/stratigraphy/test_arma_forecast_mean.py
# =============================================================================
"""
ARMA forecasts must stay on the scale of the data they were fitted to.

The forecast recursion ran on raw values:

    forecast = np.dot(ar_params, recent[::-1])

with no mean term. A stationary AR(p) is

    x_t - mu = phi_1 (x_{t-1} - mu) + ... + phi_p (x_{t-p} - mu) + e_t

so the h-step forecast is ``mu + sum_i phi_i (x_{t+h-i} - mu)``. Dropping
``mu`` is only harmless for a zero-mean series; for anything else every forecast
is dragged toward zero. Concretely, a constant series at 3.25 forecast to
~3e-17, and an AR(1) with phi = 0.7 fitted to data centred at 100 forecast to
~30 instead of ~100.

The second property below is the general form of the bug: an AR model with a
mean is shift-equivariant, so translating the data must translate the forecast
by the same amount. A hand-written "does 3.25 forecast to 3.25" test would not
have caught the general case; the invariance does.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from stratigraphy.arma import ARMAAnalyzer

warnings.filterwarnings("ignore")


def _forecast(values, n_steps=3, p=2, q=2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    analyzer = ARMAAnalyzer()
    result = analyzer.fit(np.arange(values.size, dtype=float), values, p=p, q=q, d=0)
    return np.asarray(analyzer.predict(result, n_steps).forecasts, dtype=float)


def _ar1(n: int, phi: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(scale=2.0, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


class TestConstantSeriesForecastsAtItsLevel:
    """A series that never varies must forecast at that same value."""

    @pytest.mark.parametrize("level", [3.25, 50.0, -12.0])
    def test_constant_series(self, level):
        forecasts = _forecast(np.full(150, level))
        assert np.allclose(forecasts, level, atol=1e-6), (
            f"a constant series at {level} forecast to {forecasts.tolist()}"
        )


class TestForecastIsShiftEquivariant:
    """The general form of the bug: translating the data translates the forecast."""

    def test_shift_by_100_moves_the_forecast_by_100(self):
        base = _ar1(300, 0.7)
        shift = 100.0

        unshifted = _forecast(base, n_steps=4, p=1, q=1)
        shifted = _forecast(base + shift, n_steps=4, p=1, q=1)
        delta = shifted - unshifted

        assert np.allclose(delta, shift, rtol=1e-3, atol=1e-3), (
            f"shifting the data by {shift} moved the forecast by {np.round(delta, 4).tolist()}"
        )

    def test_a_forecast_stays_near_its_data(self):
        """A strongly autocorrelated series must not decay toward zero."""
        x = _ar1(400, 0.95, seed=11) + 80.0
        forecasts = _forecast(x, n_steps=5)

        assert np.all(np.abs(forecasts) > 10.0), (
            f"data sits around {x.mean():.1f} but the forecast collapsed to {np.round(forecasts, 4).tolist()}"
        )
        assert np.all(np.abs(forecasts - x.mean()) < 20.0), (
            f"forecast is far from the data mean {x.mean():.2f}: {np.round(forecasts, 4).tolist()}"
        )
