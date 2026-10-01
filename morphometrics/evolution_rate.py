# =============================================================================
# FILE: morphometrics/evolution_rate.py
# =============================================================================
"""
Rate of Morphological Evolution Analysis for PaleoAST

Analyzes the mode and rate of morphological evolution using trait data
ordered stratigraphically (or by time).

Implements three evolutionary models:
    1. Random Walk (Brownian Motion)
    2. Directional Evolution (trend + random walk)
    3. Stasis (Ornstein-Uhlenbeck process)

Uses AIC (Akaike Information Criterion) for model selection.

Mathematical Foundation:
    Foote, R. (1997). The missing science of the P-Tr extinction.
    Evolutionary Paleobiology (Chicago Press).

    Pagel, M. (1994). Detecting correlated evolution on phylogenies.
    Evolution, 48(1), 173-190.

Models:
==============================================================================

1. Random Walk (Brownian Motion):
    x(t) = x(0) + sum_{i=1}^{t} epsilon_i
    where epsilon_i ~ N(0, sigma^2 * dt_i)

    Variance grows linearly with time: Var[x(t)] = sigma^2 * t

2. Directional:
    x(t) = x(0) + beta * t + Brownian motion
    Detects directional trends in the data

3. Stasis (Ornstein-Uhlenbeck):
    dx = -alpha * (theta - x) * dt + sigma * dW
    Mean-reverting process with equilibrium theta

Model Selection:
    AIC = -2 * log(L) + 2 * k
    AIC_weights = exp(-0.5 * delta_AIC) / sum(exp(-0.5 * delta_AIC))

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Result Classes
# =============================================================================


class EvolutionModel(Enum):
    """Evolutionary model types."""

    RANDOM_WALK = "random_walk"
    DIRECTIONAL = "directional"
    STASIS = "stasis"


@dataclass
class EvolutionRateResult:
    """
    Container for morphological evolution rate analysis results.

    Attributes:
        best_model: Name of the best-fit model
        aic_values: AIC values for each model
        aic_weights: AIC weights for each model (normalized)
        log_likelihoods: Log-likelihoods for each model
        model_probabilities: Posterior model probabilities
        rate_estimate: Evolution rate estimate
        rate_ci_lower: Lower confidence bound
        rate_ci_upper: Upper confidence bound
        trend_estimate: Directional trend (if applicable)
        trend_significance: P-value for trend
        optimum: OU process optimum (if applicable)
        attraction_strength: OU alpha parameter (if applicable)
        n_measurements: Number of data points
        trait_mean: Mean trait value
        trait_variance: Trait variance
        trait_series: Original trait values used for analysis
    """

    best_model: str
    aic_values: dict[str, float]
    aic_weights: dict[str, float]
    log_likelihoods: dict[str, float]
    model_probabilities: dict[str, float]
    rate_estimate: float
    rate_ci_lower: float | None = None
    rate_ci_upper: float | None = None
    trend_estimate: float | None = None
    trend_significance: float | None = None
    optimum: float | None = None
    attraction_strength: float | None = None
    n_measurements: int = 0
    trait_mean: float = 0.0
    trait_variance: float = 0.0
    trait_series: npt.NDArray[np.float64] | None = None

    def summary(self) -> str:
        """Generate summary text."""
        max(self.aic_weights, key=self.aic_weights.get)
        lines = [
            f"{_('Rate of Morphological Evolution')}\n",
            f"{'=' * 50}\n",
            f"{_('Best model: {0}').format(self.best_model.upper())}\n",
            f"{_('Measurements: {0}').format(self.n_measurements)}\n",
            f"{_('Trait mean: {0:.4f}, variance: {1:.4f}').format(self.trait_mean, self.trait_variance)}\n",
            "",
            f"{_('Model Comparison (AIC weights):')}\n",
        ]
        for model, weight in sorted(self.aic_weights.items(), key=lambda x: -x[1]):
            marker = " ***" if model == self.best_model else ""
            lines.append(f"  {model.upper()}: {weight:.4f}{marker}")

        lines.append("")
        lines.append(f"{_('Evolution rate: {0:.6f}').format(self.rate_estimate)}")
        if self.rate_ci_lower is not None and self.rate_ci_upper is not None:
            lines.append(f"  95% CI: [{self.rate_ci_lower:.6f}, {self.rate_ci_upper:.6f}]")

        if self.trend_estimate is not None:
            lines.append(f"{_('Directional trend: {0:.6f}').format(self.trend_estimate)}")
            if self.trend_significance is not None:
                sig = (
                    "***"
                    if self.trend_significance < 0.001
                    else ("**" if self.trend_significance < 0.01 else ("*" if self.trend_significance < 0.05 else ""))
                )
                lines.append(f"  p = {self.trend_significance:.4f} {sig}")

        if self.optimum is not None:
            lines.append(f"{_('OU optimum: {0:.4f}').format(self.optimum)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_model": self.best_model,
            "aic_values": self.aic_values,
            "aic_weights": self.aic_weights,
            "log_likelihoods": self.log_likelihoods,
            "model_probabilities": self.model_probabilities,
            "rate_estimate": self.rate_estimate,
            "rate_ci_lower": self.rate_ci_lower,
            "rate_ci_upper": self.rate_ci_upper,
            "trend_estimate": self.trend_estimate,
            "trend_significance": self.trend_significance,
            "optimum": self.optimum,
            "attraction_strength": self.attraction_strength,
            "n_measurements": self.n_measurements,
            "trait_mean": self.trait_mean,
            "trait_variance": self.trait_variance,
            "trait_series": self.trait_series.tolist() if self.trait_series is not None else None,
            "summary": self.summary(),
        }


# =============================================================================
# Main Analyzer Class
# =============================================================================


class EvolutionRateAnalyzer:
    """
    Analyzes rate of morphological evolution using trait data ordered
    stratigraphically or by time.

    Supports three evolutionary models and uses AIC for model selection.

    Example:
        >>> analyzer = EvolutionRateAnalyzer()
        >>> # Trait values in stratigraphic order
        >>> traits = np.array([2.1, 2.3, 2.5, 2.8, 3.0, 3.2, 3.1])
        >>> # Time/depth intervals
        >>> intervals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        >>> result = analyzer.analyze(traits, time_intervals=intervals)
        >>> print(result.summary())
    """

    def __init__(self) -> None:
        """Initialize evolution rate analyzer."""
        self._logger = logging.getLogger(f"{__name__}.EvolutionRateAnalyzer")
        self._lock = threading.RLock()
        self._last_result: EvolutionRateResult | None = None

    @property
    def last_result(self) -> EvolutionRateResult | None:
        """Get last computed result."""
        with self._lock:
            return self._last_result

    def analyze(
        self,
        trait_series: npt.NDArray,
        time_intervals: npt.NDArray | None = None,
        models: list[str] | None = None,
        confidence_level: float = 0.95,
        seed: int | None = None,
        n_bootstrap: int = 199,
    ) -> EvolutionRateResult:
        """
        Analyze morphological evolution rates.

        Parameters:
            trait_series: Array of trait values (n_measurements,)
                        in stratigraphic/time order
            time_intervals: Time/depth intervals between measurements
                          If None, assumes unit intervals
            models: List of models to fit, using the
                    :class:`EvolutionModel` values ``["random_walk",
                    "directional", "stasis"]`` (default: all three).
                    Any other name -- including the legacy labels
                    ``"BM"`` / ``"OU"`` and misspelled variants -- is
                    rejected rather than silently ignored.
            confidence_level: Confidence level for rate CI
            seed: Optional RNG seed for the bootstrap CI.  When ``None``
                the CI is non-reproducible across runs (the previous
                behaviour was to hit the global numpy state).
            n_bootstrap: Number of bootstrap replicates (default 199).

        Returns:
            EvolutionRateResult with best model and statistics

        Raises:
            ValidationError: If input data is invalid, or if ``models``
                contains a name that is not an :class:`EvolutionModel` value
        """
        with self._lock:
            self._logger.info(f"Analyzing evolution rate: n={len(trait_series)}")

            # Validate input
            trait_series = np.asarray(trait_series, dtype=np.float64)
            if len(trait_series) < 3:
                raise ValidationError(_("Need at least 3 measurements for evolution rate analysis"))

            if time_intervals is None:
                time_intervals = np.ones(len(trait_series) - 1)
            else:
                time_intervals = np.asarray(time_intervals, dtype=np.float64)

            if len(time_intervals) != len(trait_series) - 1:
                raise ValidationError(
                    _("Number of intervals ({0}) must equal measurements - 1 ({1})").format(
                        len(time_intervals), len(trait_series) - 1
                    )
                )

            # ``EvolutionModel`` is the single source of truth for the model
            # names.  The three ``if ... in models`` guards below silently
            # skipped every unknown name, so a legacy label ("BM", "OU") or
            # a typo either blew up later at ``min()`` on an empty
            # sequence, or was dropped without a word while still being
            # reported as a successful fit.  Reject unknown names up front.
            supported_models = tuple(model.value for model in EvolutionModel)

            if models is None:
                models = list(supported_models)

            unknown_models = [name for name in models if name not in supported_models]
            if unknown_models:
                raise ValidationError(
                    _("Unknown evolution model(s): {0}. Valid models are: {1}").format(
                        ", ".join(repr(name) for name in unknown_models),
                        ", ".join(supported_models),
                    )
                )

            if not models:
                raise ValidationError(
                    _("At least one model must be selected. Valid models are: {0}").format(", ".join(supported_models))
                )

            # First differences
            dx = np.diff(trait_series)
            dt = time_intervals

            # Trait statistics
            trait_mean = float(np.mean(trait_series))
            trait_variance = float(np.var(trait_series))

            # Fit each model
            log_liks: dict[str, float] = {}
            params: dict[str, dict[str, float]] = {}

            if EvolutionModel.RANDOM_WALK.value in models:
                ll, rate = self._fit_random_walk(dx, dt)
                log_liks[EvolutionModel.RANDOM_WALK.value] = ll
                params[EvolutionModel.RANDOM_WALK.value] = {"rate": rate}

            if EvolutionModel.DIRECTIONAL.value in models:
                ll, rate, trend, trend_se, trend_p = self._fit_directional(trait_series, dx, dt)
                log_liks[EvolutionModel.DIRECTIONAL.value] = ll
                params[EvolutionModel.DIRECTIONAL.value] = {
                    "rate": rate,
                    "trend": trend,
                    "trend_se": trend_se,
                    "trend_p": trend_p,
                }

            if EvolutionModel.STASIS.value in models:
                ll, rate, theta, alpha = self._fit_stasis(trait_series, dx, dt)
                log_liks[EvolutionModel.STASIS.value] = ll
                params[EvolutionModel.STASIS.value] = {
                    "rate": rate,
                    "optimum": theta,
                    "alpha": alpha,
                }

            # Compute AIC values
            len(trait_series)
            k = {
                EvolutionModel.RANDOM_WALK.value: 1,
                EvolutionModel.DIRECTIONAL.value: 2,
                EvolutionModel.STASIS.value: 3,
            }

            aic_values = {}
            for model in log_liks:
                if model in k:
                    aic_values[model] = -2 * log_liks[model] + 2 * k[model]
                else:
                    aic_values[model] = -2 * log_liks[model] + 2

            # Compute AIC weights
            min_aic = min(aic_values.values())
            aic_weights = {m: math.exp(-0.5 * (aic_values[m] - min_aic)) for m in aic_values}
            total_weight = sum(aic_weights.values())
            aic_weights = {m: w / total_weight for m, w in aic_weights.items()}

            # Best model
            best_model = max(aic_weights, key=aic_weights.get)

            # Get parameters for best model
            best_params = params.get(best_model, {})

            # Compute rate CI via bootstrap
            rate_ci_lower, rate_ci_upper = self._bootstrap_rate_ci(
                trait_series,
                time_intervals,
                best_model,
                confidence_level,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            result = EvolutionRateResult(
                best_model=best_model,
                aic_values=aic_values,
                aic_weights=aic_weights,
                log_likelihoods=log_liks,
                model_probabilities=aic_weights,
                rate_estimate=best_params.get("rate", 0.0),
                rate_ci_lower=rate_ci_lower,
                rate_ci_upper=rate_ci_upper,
                trend_estimate=best_params.get("trend"),
                trend_significance=best_params.get("trend_p"),
                optimum=best_params.get("optimum"),
                attraction_strength=best_params.get("alpha"),
                n_measurements=len(trait_series),
                trait_mean=trait_mean,
                trait_variance=trait_variance,
                trait_series=trait_series,
            )

            self._last_result = result
            self._logger.info(f"Evolution rate: best={best_model}, rate={result.rate_estimate:.6f}")
            return result

    def _fit_random_walk(
        self,
        dx: npt.NDArray,
        dt: npt.NDArray,
    ) -> tuple[float, float]:
        """
        Fit pure random walk model.

        For RW: variance increments = rate * dt
        MLE rate = sum(dx^2) / sum(dt)

        Returns:
            (log_likelihood, rate)
        """
        if len(dx) == 0:
            return 0.0, 0.0

        # Rate estimate (variance of increments per unit time)
        dt_sum = float(np.sum(dt))
        rate = float(np.sum(dx**2)) / dt_sum if dt_sum > 0 else 0.0

        # Log-likelihood under normal distribution
        var = rate * dt
        # Handle zero variance
        var = np.maximum(var, 1e-10)

        ll = (
            -0.5 * len(dx) * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(var))) - 0.5 * float(np.sum(dx**2 / var))
        )

        return ll, float(rate)

    def _fit_directional(
        self,
        trait_series: npt.NDArray,
        dx: npt.NDArray,
        dt: npt.NDArray,
    ) -> tuple[float, float, float, float, float]:
        """
        Fit directional (trend + RW) model.

        Trait ~ trend * time + RW

        Returns:
            (log_likelihood, rate, trend, trend_se, trend_pvalue)
        """
        if len(dx) == 0:
            return 0.0, 0.0, 0.0, 0.0, 1.0

        n = len(trait_series)

        # Cumulative time from start
        t = np.zeros(n)
        t[1:] = np.cumsum(dt)

        # Regress dx on dt to find trend
        # E[dx] = beta * dt
        dt2_sum = float(np.sum(dt**2))
        if dt2_sum > 0:
            beta = float(np.sum(dx * dt)) / dt2_sum
        else:
            beta = 0.0

        # Residuals after removing trend
        residuals = dx - beta * dt

        # Rate from residuals
        dt_sum = float(np.sum(dt))
        if dt_sum > 0:
            rate = float(np.sum(residuals**2)) / dt_sum
        else:
            rate = 0.0

        # Standard error of beta
        ss_res = float(np.sum(residuals**2))
        if dt2_sum > 0 and n > 2:
            se_beta = math.sqrt(ss_res / ((n - 2) * dt2_sum))
        else:
            se_beta = 0.0

        # T-statistic and p-value
        if se_beta > 0:
            t_stat = beta / se_beta
            # Two-tailed p-value
            p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            t_stat = 0.0
            p_value = 1.0

        # Log-likelihood
        var = rate * dt
        var = np.maximum(var, 1e-10)
        ll = (
            -0.5 * len(dx) * math.log(2 * math.pi)
            - 0.5 * float(np.sum(np.log(var)))
            - 0.5 * float(np.sum(residuals**2 / var))
        )

        return ll, float(rate), float(beta), float(se_beta), float(p_value)

    def _fit_stasis(
        self,
        trait_series: npt.NDArray,
        dx: npt.NDArray,
        dt: npt.NDArray,
    ) -> tuple[float, float, float, float]:
        """
        Fit Ornstein-Uhlenbeck (stasis) model.

        dx = -alpha*(theta - x_prev)*dt + sigma*dW

        Note: alpha = -ln(lag-1 autocorrelation) assumes even sampling
        spacing (constant dt); for unevenly spaced series the estimate
        is only approximate.

        Returns:
            (log_likelihood, sigma, theta, alpha)
        """
        if len(dx) == 0:
            return 0.0, 0.0, 0.0, 0.0

        n = len(trait_series)

        # Theta (optimum) estimate - mean of the series
        theta = float(np.mean(trait_series))
        trait_var = float(np.var(trait_series))

        # Alpha (selection strength) - estimate from autocorrelation
        if n > 2 and trait_var > 0:
            autocorr = 0.0
            for i in range(n - 1):
                autocorr += (trait_series[i] - theta) * (trait_series[i + 1] - theta)
            autocorr = autocorr / ((n - 1) * trait_var)
            # alpha = -ln(autocorr) if autocorr > 0
            alpha = -math.log(max(0.01, min(0.99, autocorr))) if autocorr > 0 else 0.1
        else:
            alpha = 0.1

        # Sigma (rate) estimate. Under the stasis model the residuals are
        # heteroscedastic, res_i ~ N(0, sigma^2 * dt_i), so the weighted
        # estimate of sigma^2 divides by sum(dt_i) (not sum(dt_i^2)).
        residuals = dx + alpha * (trait_series[:-1] - theta) * dt
        dt_sum = float(np.sum(dt))
        sigma_sq = float(np.sum(residuals**2)) / dt_sum if dt_sum > 0 else 0.0
        sigma = math.sqrt(max(sigma_sq, 1e-10))

        # Log-likelihood (Gaussian approximation)
        var = sigma**2 * dt
        var = np.maximum(var, 1e-10)
        ll = (
            -0.5 * len(dx) * math.log(2 * math.pi)
            - 0.5 * float(np.sum(np.log(var)))
            - 0.5 * float(np.sum(residuals**2 / var))
        )

        return ll, float(sigma), float(theta), float(alpha)

    def _bootstrap_rate_ci(
        self,
        trait_series: npt.NDArray,
        time_intervals: npt.NDArray,
        model: str,
        confidence_level: float,
        n_bootstrap: int = 199,
        seed: int | None = None,
    ) -> tuple[float | None, float | None]:
        """
        Bootstrap confidence interval for the rate estimate.

        The residuals are resampled with replacement and the bootstrap
        trait series is reconstructed by integrating the resampled
        residuals.  The residuals MUST be model-specific (the previous
        implementation used the random-walk first-difference residuals
        for every model, which silently broke the CI for the directional
        and stasis models — its own comment admitted this).

        Parameters:
            trait_series: (n,) trait values in stratigraphic / time order.
            time_intervals: (n-1,) inter-sample intervals.
            model: ``"random_walk"``, ``"directional"``, or ``"stasis"``.
            confidence_level: e.g. 0.95 for a 95% percentile CI.
            n_bootstrap: number of bootstrap replicates.
            seed: optional RNG seed for reproducibility.  When ``None``,
                a fresh ``np.random.default_rng`` is used; the previous
                ``np.random.choice`` call hit the global numpy state and
                made the CIs non-reproducible across runs.

        Returns:
            (ci_lower, ci_upper)
        """
        n = len(trait_series)
        if n < 5:
            return None, None

        rng = np.random.default_rng(seed)
        dx = np.diff(trait_series)

        # Model-specific residuals (centred so the bootstrap surrogate has
        # the same mean drift as the data).  Random-walk uses the
        # first-difference residuals; directional uses the residuals
        # around the fitted linear trend (same expression as in
        # ``_fit_directional``); stasis uses the residuals around the OU
        # optimum (same expression as in ``_fit_stasis``).
        if model == EvolutionModel.RANDOM_WALK.value:
            residuals = dx - np.mean(dx)
        elif model == EvolutionModel.DIRECTIONAL.value:
            dt2_sum = float(np.sum(time_intervals**2))
            if dt2_sum > 0:
                beta = float(np.sum(dx * time_intervals)) / dt2_sum
            else:
                beta = 0.0
            residuals = dx - beta * time_intervals
        elif model == EvolutionModel.STASIS.value:
            theta = float(np.mean(trait_series))
            trait_var = float(np.var(trait_series))
            if n > 2 and trait_var > 0:
                autocorr = 0.0
                for i in range(n - 1):
                    autocorr += (trait_series[i] - theta) * (trait_series[i + 1] - theta)
                autocorr = autocorr / ((n - 1) * trait_var)
                alpha = -math.log(max(0.01, min(0.99, autocorr))) if autocorr > 0 else 0.1
            else:
                alpha = 0.1
            residuals = dx + alpha * (trait_series[:-1] - theta) * time_intervals
        else:
            raise ValidationError(_("Unknown bootstrap model '{0}'").format(model))

        rates = []
        for _draw in range(n_bootstrap):
            boot_residuals = rng.choice(residuals, size=len(residuals), replace=True)

            # Reconstruct bootstrap trait series
            boot_trait = np.zeros(n)
            boot_trait[0] = trait_series[0]
            for i in range(len(boot_residuals)):
                boot_trait[i + 1] = boot_trait[i] + boot_residuals[i]

            # Fit model
            boot_dx = np.diff(boot_trait)
            if model == EvolutionModel.RANDOM_WALK.value:
                _, rate = self._fit_random_walk(boot_dx, time_intervals)
                rates.append(rate)
            elif model == EvolutionModel.DIRECTIONAL.value:
                _, rate, _, _, _ = self._fit_directional(boot_trait, boot_dx, time_intervals)
                rates.append(rate)
            elif model == EvolutionModel.STASIS.value:
                _, rate, _, _ = self._fit_stasis(boot_trait, boot_dx, time_intervals)
                rates.append(rate)

        if not rates:
            return None, None

        rates = np.array(rates)
        alpha_tail = 1 - confidence_level
        ci_lower = float(np.percentile(rates, alpha_tail / 2 * 100))
        ci_upper = float(np.percentile(rates, (1 - alpha_tail / 2) * 100))

        return ci_lower, ci_upper
