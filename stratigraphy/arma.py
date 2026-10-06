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
version: 1.1.0
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
            # ``validate_data_array`` reshapes 1-D input to a column vector
            # (n, 1).  Every downstream routine (np.diff, np.correlate,
            # statsmodels ARIMA, cumsum-based back-transformation) assumes a
            # *flat* series, so the arrays are flattened right here; otherwise
            # ``np.diff`` runs along the singleton axis and silently returns an
            # (n, 0) array (2026-09 review).
            t = np.asarray(validate_data_array(times, allow_nan=False, name="times")).reshape(-1)
            y = np.asarray(validate_data_array(values, allow_nan=False, name="values")).reshape(-1)

            if t.shape != y.shape:
                raise ComputationError(f"Times and values must have same shape: {t.shape} vs {y.shape}")

            if len(t) < max(p, q) * 3:
                raise ComputationError(f"Need at least {max(p, q) * 3} observations, got {len(t)}")

            self._logger.info(f"Fitting ARMA({p},{q}) model to {len(t)} observations")

            # Apply differencing if needed
            if d > 0:
                y_diff = self._difference(y, d)
                if len(y_diff) <= max(p, q) + 1:
                    raise ComputationError(
                        f"Series too short for ARIMA({p},{d},{q}): {len(y_diff)} differenced "
                        f"observations remain, need more than {max(p, q) + 1}"
                    )
            else:
                y_diff = y.copy()

            # Try statsmodels first, otherwise use manual implementation.
            # The fallback used to catch ImportError only, so a statsmodels
            # version mismatch (AttributeError on ``.values``, ValueError from
            # a failed optimizer, LinAlgError from the Kalman filter) crashed
            # the whole analysis instead of degrading to the manual AR fit.
            try:
                result = self._fit_statsmodels(y_diff, p, q, include_intercept)
            except (ImportError, AttributeError, ValueError, np.linalg.LinAlgError) as e:
                self._logger.info(f"statsmodels ARMA path unavailable ({type(e).__name__}: {e}); using manual AR")
                result = self._fit_manual_ar(y_diff, p, q, include_intercept)

            # Compute predictions (back-transform if differenced)
            if d > 0:
                predicted = self._inverse_difference(y, result["predicted"], d)
                residuals = y - predicted
            else:
                predicted = result["predicted"]
                residuals = result["residuals"]

            # Compute information criteria.
            #
            # AIC/BIC are n*ln(sigma^2) + penalty. A constant series fits
            # exactly, so the residuals are all zero and sigma^2 is 0:
            # ln(0) is -inf, and the old code produced an AIC and BIC of
            # -inf, which is not a score -- it is a sign the model
            # ordering is meaningless (every other model looks infinitely
            # worse). Guard it, and say why.
            n = len(y)
            k = result["n_params"]
            sigma2 = float(np.var(residuals))
            if not np.isfinite(sigma2) or sigma2 <= 0.0:
                # A perfect (or numerically degenerate) fit: information
                # criteria are undefined, so report NaN rather than -inf.
                self._logger.warning(
                    "Residual variance is %s; AIC/BIC are undefined for a degenerate fit and are reported as NaN.",
                    sigma2,
                )
                aic = float("nan")
                bic = float("nan")
            else:
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

        Two further caveats (2026-09 review):
            * ``fit.params`` is a plain ``numpy.ndarray`` for statsmodels
              >= 0.13, so ``fit.params.values`` raised ``AttributeError``.
              ``np.asarray(fit.params)`` works for both the ndarray and the
              legacy pandas-Series result.
            * ``trend`` must be pinned down explicitly: the statsmodels
              default is ``trend='c'``, which keeps fitting a constant even
              when ``include_intercept`` is False, and that constant is then
              reported as an AR coefficient (the parameter vector is offset
              by one).
        """
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(y, order=(p, 0, q), trend="n" if not include_intercept else "c")
        fit = model.fit()

        offset = 1 if include_intercept else 0
        params = np.asarray(fit.params)
        return {
            "ar_params": params[offset : offset + p],
            "ma_params": params[offset + p : offset + p + q],
            "predicted": np.asarray(fit.fittedvalues),
            "residuals": np.asarray(fit.resid),
            "n_params": p + q + (1 if include_intercept else 0),
        }

    def _fit_ma_hannan_rissanen(
        self,
        y_c: npt.NDArray,
        p: int,
        q: int,
    ) -> npt.NDArray:
        """Estimate MA coefficients by the Hannan-Rissanen innovation method.

        Hannan, E. J. & Rissanen, J. (1972), "A level of approximation for
        estimating the power spectrum of a process", IEEE Trans. Acoust.
        Speech Signal Process. 20:323-332.

        1. Fit a LONG AR filter (longer than the model's own p) to the series.
           Its residuals are a proxy for the unobserved innovations; without
           them an MA term is unidentifiable from the data alone.
        2. Regress the series on its own AR lags AND on the lagged proxy
           innovations. The coefficients on the innovations are theta.

        Returns an empty array when q == 0, or when the design matrix is too
        small / rank-deficient to solve.
        """
        if q <= 0:
            return np.zeros(0)

        n = len(y_c)
        # A long filter, but never so long that there is no data left to
        # regress. p_long is deliberately allowed to exceed p: that is the
        # whole point of the innovation step.
        p_long = min(max(p + q, 5), max(1, n // 8))
        start = max(p, p_long, q, 2)

        if n - start < max(p + q, 1):
            self._logger.warning(
                "Hannan-Rissanen MA estimation needs more than %d observations for "
                "p=%d q=%d; returning zero MA coefficients",
                n - start,
                p,
                q,
            )
            return np.zeros(q)

        # Step 1 -- long AR filter by Yule-Walker, for the innovation proxy.
        innovation = np.zeros(n)
        if p_long > 0:
            gamma = np.correlate(y_c, y_c, mode="full")[n - 1 :]
            if gamma[0] <= 0:
                return np.zeros(q)
            r_matrix = np.array([[gamma[abs(i - j)] for j in range(p_long)] for i in range(p_long)])
            try:
                long_ar = np.linalg.solve(r_matrix, gamma[1 : p_long + 1])
            except np.linalg.LinAlgError:
                long_ar = np.linalg.lstsq(r_matrix, gamma[1 : p_long + 1], rcond=None)[0]
            for t in range(p_long, n):
                innovation[t] = y_c[t] - float(np.dot(long_ar, y_c[t - p_long : t][::-1]))

        # Step 2 -- OLS of the series on its AR lags and the innovation lags.
        rows = []
        target = []
        for t in range(start, n):
            ar_block = y_c[t - p : t][::-1] if p > 0 else np.zeros(0)
            ma_block = innovation[t - q : t][::-1]
            rows.append(np.concatenate([ar_block, ma_block]))
            target.append(y_c[t])

        design = np.asarray(rows, dtype=float)
        if design.shape[0] <= design.shape[1]:
            return np.zeros(q)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            self._logger.warning(
                "Hannan-Rissanen design matrix is rank deficient for p=%d q=%d; "
                "MA coefficients are not identifiable from this series",
                p,
                q,
            )
            return np.zeros(q)

        try:
            beta = np.linalg.lstsq(design, np.asarray(target, dtype=float), rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.zeros(q)

        return np.asarray(beta[-q:], dtype=float)

    def _fit_manual_ar(
        self,
        y: npt.NDArray,
        p: int,
        q: int,
        include_intercept: bool,
    ) -> dict[str, Any]:
        """Manual AR fit using Yule-Walker for AR part.

        The MA coefficients are *not* estimated on this path (no innovation
        algorithm is implemented), so ``q`` must not be counted in the
        number of fitted parameters: inflating ``k`` biases AIC/BIC.

        Yule-Walker normal equations (Box & Jenkins):

            Σ_j φ_j γ(|i-j|) = γ(i),  i = 1..p

        i.e. the right-hand side is the autocovariance vector γ₁..γ_p, *not*
        γ₀..γ_{p-1} (whose first entry is the variance). The previous
        implementation passed the whole ``gamma`` array (length p+1) to
        ``np.linalg.solve`` with a p×p matrix, which raised
        ``ValueError: ... object too deep``/shape mismatch and made the
        fallback path — the only path available without statsmodels — fail
        outright.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)
        mu = float(np.mean(y))

        # Yule-Walker for AR coefficients, on the mean-centred series
        ar_params = np.zeros(p)
        if p > 0:
            y_centered = y - mu
            # Biased autocovariance estimates γ_0 .. γ_p (dividing by n keeps
            # the Toeplitz system positive semi-definite).
            autocov = np.correlate(y_centered, y_centered, mode="full")
            autocov = autocov[n - 1 : n + p] / n
            gamma0 = float(autocov[0])
            if gamma0 <= 0 or not np.isfinite(gamma0):
                # Constant (or degenerate) series: no AR structure to fit.
                ar_params = np.zeros(p)
            else:
                R = np.empty((p, p), dtype=float)
                for i in range(p):
                    for j in range(p):
                        R[i, j] = autocov[abs(i - j)]
                rhs = autocov[1 : p + 1]
                try:
                    ar_params = np.linalg.solve(R, rhs)
                except np.linalg.LinAlgError:
                    self._logger.warning("Yule-Walker system is singular; falling back to least squares")
                    ar_params = np.linalg.lstsq(R, rhs, rcond=None)[0]

        # MA coefficients: Hannan-Rissanen.
        #
        # The previous code left them at zero and merely logged a warning. That
        # is worse than useless for a cyclostratigraphy workflow: MA(1) on a
        # series with a real moving-average term came back as ``[0.0]``, and
        # the summary printed "MA coefficients: [0.0]" -- which reads as
        # "measured, and it is zero" rather than "never estimated". A
        # researcher would conclude the series has no MA structure.
        #
        # Hannan & Rissanen (1972): fit a LONG AR filter first to get an
        # innovation proxy, then regress the data on the AR lags AND the
        # innovation lags; the coefficients on the innovations are the MA part.
        ma_params = self._fit_ma_hannan_rissanen(y_c=y_centered if p > 0 else (y - mu), p=p, q=q)
        if q > 0:
            self._logger.info(
                "statsmodels unavailable: MA(%d) coefficients estimated by the "
                "Hannan-Rissanen innovation algorithm rather than MLE; AIC/BIC "
                "are therefore approximate.",
                q,
            )

        # Constant term: for a stationary AR(p) with mean mu the intercept is
        # c = (1 - Σφ) * mu. The previous code used mean(y[:p]) (the mean of
        # the burn-in window), which is unrelated to the process mean.
        intercept = (1.0 - float(np.sum(ar_params))) * mu if include_intercept else 0.0

        predicted = np.zeros(n)
        residuals = np.zeros(n)

        for t in range(p, n):
            pred = float(np.dot(ar_params, y[t - p : t][::-1])) + intercept
            predicted[t] = pred
            residuals[t] = y[t] - pred

        return {
            "ar_params": ar_params,
            "ma_params": ma_params,
            "predicted": predicted,
            "residuals": residuals,
            "n_params": p + (1 if include_intercept else 0),
        }

    def _difference(self, y: npt.NDArray, d: int) -> npt.NDArray:
        """Apply differencing.

        ``y`` is flattened first: ``np.diff`` works along the *last* axis by
        default, so a (n, 1) column vector would be turned into an (n, 0)
        array.
        """
        result = np.asarray(y, dtype=float).reshape(-1).copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _inverse_difference(self, original: npt.NDArray, differenced: npt.NDArray, d: int) -> npt.NDArray:
        """Reverse differencing for predictions.

        Integrating a d-th difference back to the observed scale consumes the
        first value of each difference order, so the initial conditions are
        taken from ``original`` one order at a time. The previous
        implementation added ``original[0]`` to the whole cumsum at every
        order, which returned a curve shorter than ``original`` (breaking
        ``residuals = y - predicted``) and with the wrong offset.
        """
        original = np.asarray(original, dtype=float).reshape(-1)
        result = np.asarray(differenced, dtype=float).reshape(-1).copy()

        if result.shape[0] != original.shape[0] - d:
            # Unexpected geometry (e.g. a predictor that returned the
            # undifferenced length): fall back to the caller's series so the
            # residual computation stays well-defined.
            self._logger.warning(
                "Cannot back-transform %d predictions against %d observations (d=%d); returning predictions unchanged",
                result.shape[0],
                original.shape[0],
                d,
            )
            return result

        for order in range(d, 0, -1):
            # ``init`` is the first element of the (order-1)-th difference of
            # the observed series; integrating adds it to the running cumsum.
            init = float(np.diff(original, n=order - 1)[0])
            result = np.concatenate(([init], init + np.cumsum(result)))

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

            # AR(p) forecasts must be taken about the series mean.
            #
            # A stationary AR(p) says
            #     x_t - mu = phi_1 (x_{t-1} - mu) + ... + phi_p (x_{t-p} - mu) + e_t
            # so the h-step forecast is
            #     x_{t+h} = mu + sum_i phi_i (x_{t+h-i} - mu)
            # and omitting `mu` was not a small slip: the previous
            # implementation recursed on the raw values,
            #     forecast = dot(ar_params, recent[::-1])
            # which drives every forecast toward zero instead of toward the
            # series mean. A constant series at 3.25 forecast to ~0, and an
            # AR(1) with phi=0.6 on data centred at 100 forecast to ~60.
            # The AR coefficients describe dynamics *about* the mean, so the
            # recursion runs on centred values and the mean is added back at
            # each step.
            mu = float(np.mean(result.values)) if result.values.size else 0.0

            forecasts = np.zeros(n_steps)
            stderr = np.zeros(n_steps)
            scale = np.std(result.residuals)

            # Use last p values as starting point, centred
            recent = result.values[-result.p :] if result.p > 0 else result.values[-1:]
            recent = np.asarray(recent, dtype=float) - mu

            for h in range(n_steps):
                # Point forecast
                if result.d > 0:
                    # For differenced model, integrate
                    forecast = np.mean(recent)
                else:
                    forecast = np.dot(result.ar_params, recent[::-1])

                # Back to the original scale for the reported value...
                forecasts[h] = forecast + mu
                # ...but the recursion continues on centred values.
                centred = forecast

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

                # Update recent values for next step (still centred)
                recent = np.append(recent[1:], centred)

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
        best_order = (0, 0)

        aic_table = {}
        bic_table = {}

        # Start BOTH loops at 0. The previous range(1, ...) meant p=0 and
        # q=0 were never fitted, so:
        #   * a pure AR series could never win -- the search was FORCED to add
        #     a moving-average term that is not there (an AR(0.9) series came
        #     back as (1,1) instead of (1,0));
        #   * a pure MA series, and white noise, had no reachable order at all.
        # p=0 is well defined: a blank AR vector, and the MA innovation
        # estimator handles a pure MA model.
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
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
