# =============================================================================
# FILE: stratigraphy/arma.py
# =============================================================================
"""
ARMA/ARIMA Time Series Analysis Module for PaleoAST

Autoregressive Moving Average models for analyzing temporal
patterns in stratigraphic and paleontological sequences.

Mathematical Foundation:

ARMA(p,q) Model:
    X_t = φ₁X_{t-1} + ... + φ_pX_{t-p}
        + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}

where:
    φ = autoregressive coefficients
    θ = moving average coefficients
    ε = white noise innovation

ARIMA(p,d,q) adds differencing (d) for non-stationary series:
    Δ^d X_t = ARMA(p,q)

Model Selection:
    AIC = -2*log(L) + 2k
    BIC = -2*log(L) + k*log(n)

where L is likelihood and k is number of parameters.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class ARMAResult:
    """
    Container for ARMA model results.

    Attributes:
        ar_params: Autoregressive coefficients (p,)
        ma_params: Moving average coefficients (q,)
        residuals: Model residuals (n,)
        predicted: In-sample predictions (n,)
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        p: AR order
        q: MA order
        d: Differencing order (0 for ARMA)
        n_params: Number of estimated parameters
        times: Time indices
        values: Original values
    """

    ar_params: npt.NDArray
    ma_params: npt.NDArray
    residuals: npt.NDArray
    predicted: npt.NDArray
    aic: float
    bic: float
    p: int
    q: int
    d: int
    n_params: int
    times: npt.NDArray
    values: npt.NDArray

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"ARMA({self.p},{self.q}) Model Results\n"
            f"{'=' * 50}\n"
            f"AIC: {self.aic:.4f}\n"
            f"BIC: {self.bic:.4f}\n"
            f"Parameters: {self.n_params}\n"
            f"AR coefficients: {list(self.ar_params.round(4))}\n"
            f"MA coefficients: {list(self.ma_params.round(4))}"
        )


@dataclass
class ForecastResult:
    """
    Container for ARMA forecast results.

    Attributes:
        forecasts: Forecasted values (n_steps,)
        lower_ci: Lower confidence interval (n_steps,)
        upper_ci: Upper confidence interval (n_steps,)
        std_error: Forecast standard errors (n_steps,)
        n_steps: Number of forecast steps
    """

    forecasts: npt.NDArray
    lower_ci: npt.NDArray
    upper_ci: npt.NDArray
    std_error: npt.NDArray
    n_steps: int


class ARMAAnalyzer:
    """
    ARMA/ARIMA time series analyzer.

    Fits Autoregressive Moving Average models to stratigraphic
    and paleontological time series data.
    """

    def __init__(self) -> None:
        """Initialize the ARMA analyzer."""
        self._logger = logging.getLogger(f"{__name__}.ARMAAnalyzer")
        self._lock = threading.RLock()
        self._last_result: ARMAResult | None = None
        self._logger.info("ARMAAnalyzer initialized")

    def fit(
        self,
        times: npt.NDArray,
        values: npt.NDArray,
        p: int = 2,
        q: int = 2,
        d: int = 0,
        include_intercept: bool = True,
    ) -> ARMAResult:
        """
        Fit an ARMA(p,q) or ARIMA(p,d,q) model.

        Parameters:
            times: Time indices (n,)
            values: Observed values (n,)
            p: Autoregressive order
            q: Moving average order
            d: Differencing order (0 for ARMA, >=1 for ARIMA)
            include_intercept: Whether to include intercept term

        Returns:
            ARMAResult with model parameters and diagnostics

        Note:
            Uses statsmodels ARIMA if available, otherwise
            falls back to Yule-Walker (AR) and innovation algorithm (MA).
        """
        with self._lock:
            t = validate_data_array(times, allow_nan=False, name="times")
            y = validate_data_array(values, allow_nan=False, name="values")

            if t.shape != y.shape:
                raise ComputationError(f"Times and values must have same shape: {t.shape} vs {y.shape}")

            if len(t) < max(p, q) * 3:
                raise ComputationError(f"Need at least {max(p, q) * 3} observations, got {len(t)}")

            self._logger.info(f"Fitting ARMA({p},{q}) model to {len(t)} observations")

            # Apply differencing if needed
            if d > 0:
                y_diff = self._difference(y, d)
            else:
                y_diff = y.copy()

            # Try statsmodels first, otherwise use manual implementation
            try:
                result = self._fit_statsmodels(y_diff, p, q, include_intercept)
            except ImportError:
                self._logger.info("statsmodels not available, using manual ARMA")
                result = self._fit_manual_ar(y_diff, p, q, include_intercept)

            # Compute predictions (back-transform if differenced)
            if d > 0:
                predicted = self._inverse_difference(y, result["predicted"], d)
                residuals = y - predicted
            else:
                predicted = result["predicted"]
                residuals = result["residuals"]

            # Compute information criteria
            n = len(y)
            k = result["n_params"]
            sigma2 = np.var(residuals)
            aic = float(n * np.log(sigma2) + 2 * k)
            bic = float(n * np.log(sigma2) + k * np.log(n))

            armaresult = ARMAResult(
                ar_params=result["ar_params"],
                ma_params=result["ma_params"],
                residuals=residuals,
                predicted=predicted,
                aic=aic,
                bic=bic,
                p=p,
                q=q,
                d=d,
                n_params=k,
                times=t,
                values=y,
            )

            self._last_result = armaresult
            self._logger.info(f"ARMA({p},{q}) fit complete: AIC={aic:.4f}, BIC={bic:.4f}")
            return armaresult

    def _fit_statsmodels(
        self,
        y: npt.NDArray,
        p: int,
        q: int,
        include_intercept: bool,
    ) -> dict[str, Any]:
        """Fit using statsmodels ARIMA.

        statsmodels ARIMA parameter ordering depends on whether an
        intercept is included:
            - With intercept:    [intercept, ar.L1, ..., ar.Lp, ma.L1, ..., ma.Lq, sigma2]
            - Without intercept: [ar.L1, ..., ar.Lp, ma.L1, ..., ma.Lq, sigma2]
        We therefore offset the slice by 1 when ``include_intercept``.
        """
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(y, order=(p, 0, q))
        fit = model.fit()

        offset = 1 if include_intercept else 0
        params = fit.params.values
        return {
            "ar_params": params[offset : offset + p],
            "ma_params": params[offset + p : offset + p + q],
            "predicted": fit.fittedvalues,
            "residuals": fit.resid,
            "n_params": p + q + (1 if include_intercept else 0),
        }

    def _fit_manual_ar(
        self,
        y: npt.NDArray,
        p: int,
        q: int,
        include_intercept: bool,
    ) -> dict[str, Any]:
        """Manual AR fit using Yule-Walker for AR part."""
        n = len(y)

        # Yule-Walker for AR coefficients
        ar_params = np.zeros(p)
        if p > 0:
            # Compute autocovariance
            gamma = np.correlate(y, y, mode="full")
            gamma = gamma[n - 1 : n + p]

            # Build Toeplitz matrix
            R = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    lag = abs(i - j)
                    if lag < len(gamma):
                        R[i, j] = gamma[lag]

            try:
                ar_params = np.linalg.solve(R, gamma)
            except np.linalg.LinAlgError:
                ar_params = np.zeros(p)

        # Innovation algorithm for MA part (simplified)
        ma_params = np.zeros(q)

        # Compute residuals
        predicted = np.zeros(n)
        residuals = np.zeros(n)

        for t in range(p, n):
            pred = np.dot(ar_params, y[t - p : t][::-1])
            if include_intercept:
                pred += np.mean(y[:p])
            predicted[t] = pred
            residuals[t] = y[t] - pred

        return {
            "ar_params": ar_params,
            "ma_params": ma_params,
            "predicted": predicted,
            "residuals": residuals,
            "n_params": p + q + (1 if include_intercept else 0),
        }

    def _difference(self, y: npt.NDArray, d: int) -> npt.NDArray:
        """Apply differencing."""
        result = y.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _inverse_difference(self, original: npt.NDArray, differenced: npt.NDArray, d: int) -> npt.NDArray:
        """Reverse differencing for predictions."""
        result = differenced.copy()
        for _ in range(d):
            result = np.cumsum(result)
            result += original[0]
        return result

    def predict(
        self,
        result: ARMAResult | None = None,
        n_steps: int = 10,
        alpha: float = 0.05,
    ) -> ForecastResult:
        """
        Generate forecasts from fitted ARMA model.

        Parameters:
            result: ARMA result from fit(). If None, uses last result.
            n_steps: Number of steps to forecast
            alpha: Significance level for confidence interval

        Returns:
            ForecastResult with forecasts and confidence intervals
        """
        with self._lock:
            if result is None:
                result = self._last_result

            if result is None:
                raise ComputationError("No ARMA result available. Call fit() first.")

            self._logger.info(f"Generating {n_steps}-step ahead forecast")

            # Simple AR forecast using last observations
            forecasts = np.zeros(n_steps)
            stderr = np.zeros(n_steps)
            scale = np.std(result.residuals)

            # Use last p values as starting point
            recent = result.values[-result.p :] if result.p > 0 else result.values[-1:]

            for h in range(n_steps):
                # Point forecast
                if result.d > 0:
                    # For differenced model, integrate
                    forecast = np.mean(recent)
                else:
                    forecast = np.dot(result.ar_params, recent[::-1])

                forecasts[h] = forecast

                # Recursive prediction variance. The previous
                # implementation used an ad-hoc ``scale *
                # sqrt(1 + h * 0.1)`` heuristic with an arbitrary
                # growth factor of 0.1, which produced wildly
                # inaccurate confidence intervals for moderate
                # horizons. Use the standard AR(p) prediction
                # variance recursion:
                #     Var(forecast at h) = σ² * (1 + Σ ψ_i²)
                # where ψ are the MA(∞) coefficients. We
                # approximate ψ_i by iterating the AR recursion
                # up to ``min(h, max(p, 5))``.
                variance_h = scale**2
                if h > 0:
                    max_lag = min(h, max(len(result.ar_params), 5))
                    psi_sum = 0.0
                    # Initialise ψ with ψ_0 = 1, ψ_i = AR_i for i >= 1
                    psi = np.zeros(max_lag + 1)
                    psi[0] = 1.0
                    for i in range(1, max_lag + 1):
                        val = 0.0
                        for j in range(1, min(i, len(result.ar_params)) + 1):
                            val += result.ar_params[j - 1] * psi[i - j]
                        psi[i] = val
                        psi_sum += val**2
                    variance_h = scale**2 * (1.0 + psi_sum)
                stderr[h] = np.sqrt(max(variance_h, 0.0))

                # Update recent values for next step
                recent = np.append(recent[1:], forecast)

            # Confidence intervals
            z = 1.96  # ~95% CI
            lower = forecasts - z * stderr
            upper = forecasts + z * stderr

            return ForecastResult(
                forecasts=forecasts,
                lower_ci=lower,
                upper_ci=upper,
                std_error=stderr,
                n_steps=n_steps,
            )

    def cross_validate(
        self,
        times: npt.NDArray,
        values: npt.NDArray,
        max_p: int = 5,
        max_q: int = 5,
        d: int = 0,
    ) -> dict[str, Any]:
        """
        Find optimal ARMA order using cross-validation or AIC.

        Parameters:
            times: Time indices
            values: Observed values
            max_p: Maximum AR order to try
            max_q: Maximum MA order to try
            d: Differencing order

        Returns:
            Dict with best_order, aic_table, bic_table
        """
        best_aic = np.inf
        best_order = (2, 2)

        aic_table = {}
        bic_table = {}

        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    result = self.fit(times, values, p=p, q=q, d=d)
                    aic_table[(p, q)] = result.aic
                    bic_table[(p, q)] = result.bic

                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, q)
                except Exception as e:
                    self._logger.debug(f"ARMA({p},{q}) failed: {e}")

        self._logger.info(f"Best ARMA order by AIC: ({best_order[0]},{best_order[1]})")

        return {
            "best_order": best_order,
            "best_aic": best_aic,
            "aic_table": aic_table,
            "bic_table": bic_table,
        }

    @property
    def last_result(self) -> ARMAResult | None:
        """Get the last ARMA result."""
        with self._lock:
            return self._last_result
