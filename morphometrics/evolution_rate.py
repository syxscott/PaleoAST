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
Version: 1.0.0
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from config.i18n import _
from utils.exceptions import ComputationError, ValidationError

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
        best_idx = max(self.aic_weights, key=self.aic_weights.get)
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
            lines.append(
                f"  95% CI: [{self.rate_ci_lower:.6f}, {self.rate_ci_upper:.6f}]"
            )

        if self.trend_estimate is not None:
            lines.append(f"{_('Directional trend: {0:.6f}').format(self.trend_estimate)}")
            if self.trend_significance is not None:
                sig = "***" if self.trend_significance < 0.001 else (
                    "**" if self.trend_significance < 0.01 else (
                        "*" if self.trend_significance < 0.05 else ""))
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
    ) -> EvolutionRateResult:
        """
        Analyze morphological evolution rates.

        Parameters:
            trait_series: Array of trait values (n_measurements,)
                        in stratigraphic/time order
            time_intervals: Time/depth intervals between measurements
                          If None, assumes unit intervals
            models: List of models to fit ["random_walk", "directional", "stasis"]
            confidence_level: Confidence level for rate CI

        Returns:
            EvolutionRateResult with best model and statistics

        Raises:
            ValidationError: If input data is invalid
        """
        with self._lock:
            self._logger.info(
                f"Analyzing evolution rate: n={len(trait_series)}"
            )

            # Validate input
            trait_series = np.asarray(trait_series, dtype=np.float64)
            if len(trait_series) < 3:
                raise ValidationError(
                    _("Need at least 3 measurements for evolution rate analysis")
                )

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

            if models is None:
                models = ["random_walk", "directional", "stasis"]

            # First differences
            dx = np.diff(trait_series)
            dt = time_intervals

            # Trait statistics
            trait_mean = float(np.mean(trait_series))
            trait_variance = float(np.var(trait_series))

            # Fit each model
            log_liks: dict[str, float] = {}
            params: dict[str, dict[str, float]] = {}

            if "random_walk" in models:
                ll, rate = self._fit_random_walk(dx, dt)
                log_liks["random_walk"] = ll
                params["random_walk"] = {"rate": rate}

            if "directional" in models:
                ll, rate, trend, trend_se, trend_p = self._fit_directional(
                    trait_series, dx, dt
                )
                log_liks["directional"] = ll
                params["directional"] = {
                    "rate": rate,
                    "trend": trend,
                    "trend_se": trend_se,
                    "trend_p": trend_p,
                }

            if "stasis" in models:
                ll, rate, theta, alpha = self._fit_stasis(trait_series, dx, dt)
                log_liks["stasis"] = ll
                params["stasis"] = {
                    "rate": rate,
                    "optimum": theta,
                    "alpha": alpha,
                }

            # Compute AIC values
            n = len(trait_series)
            k = {"random_walk": 1, "directional": 2, "stasis": 3}

            aic_values = {}
            for model in log_liks:
                if model in k:
                    aic_values[model] = -2 * log_liks[model] + 2 * k[model]
                else:
                    aic_values[model] = -2 * log_liks[model] + 2

            # Compute AIC weights
            min_aic = min(aic_values.values())
            aic_weights = {
                m: math.exp(-0.5 * (aic_values[m] - min_aic))
                for m in aic_values
            }
            total_weight = sum(aic_weights.values())
            aic_weights = {m: w / total_weight for m, w in aic_weights.items()}

            # Best model
            best_model = max(aic_weights, key=aic_weights.get)

            # Get parameters for best model
            best_params = params.get(best_model, {})

            # Compute rate CI via bootstrap
            rate_ci_lower, rate_ci_upper = self._bootstrap_rate_ci(
                trait_series, time_intervals, best_model, confidence_level
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
            self._logger.info(
                f"Evolution rate: best={best_model}, rate={result.rate_estimate:.6f}"
            )
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

        ll = -0.5 * len(dx) * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(var))) - 0.5 * float(np.sum(dx**2 / var))

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
        ll = -0.5 * len(dx) * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(var))) - 0.5 * float(np.sum(residuals**2 / var))

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

        # Sigma (rate) estimate
        residuals = dx + alpha * (trait_series[:-1] - theta) * dt
        dt2_sum = float(np.sum(dt**2))
        sigma_sq = float(np.sum(residuals**2)) / dt2_sum if dt2_sum > 0 else 0.0
        sigma = math.sqrt(max(sigma_sq, 1e-10))

        # Log-likelihood (Gaussian approximation)
        var = sigma**2 * dt
        var = np.maximum(var, 1e-10)
        ll = -0.5 * len(dx) * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(var))) - 0.5 * float(np.sum(residuals**2 / var))

        return ll, float(sigma), float(theta), float(alpha)

    def _bootstrap_rate_ci(
        self,
        trait_series: npt.NDArray,
        time_intervals: npt.NDArray,
        model: str,
        confidence_level: float,
        n_bootstrap: int = 199,
    ) -> tuple[float | None, float | None]:
        """
        Bootstrap confidence interval for rate estimate.

        Returns:
            (ci_lower, ci_upper)
        """
        n = len(trait_series)
        if n < 5:
            return None, None

        rates = []
        for _ in range(n_bootstrap):
            # Resample residuals
            dx = np.diff(trait_series)
            residuals = dx - np.mean(dx)
            np.random.seed(None)  # Use random seed for each bootstrap
            boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)

            # Reconstruct bootstrap trait series
            boot_trait = np.zeros(n)
            boot_trait[0] = trait_series[0]
            for i in range(len(boot_residuals)):
                boot_trait[i + 1] = boot_trait[i] + boot_residuals[i]

            # Fit model
            boot_dx = np.diff(boot_trait)
            if model == "random_walk":
                _, rate = self._fit_random_walk(boot_dx, time_intervals)
                rates.append(rate)
            elif model == "directional":
                _, rate, _, _, _ = self._fit_directional(boot_trait, boot_dx, time_intervals)
                rates.append(rate)
            elif model == "stasis":
                _, rate, _, _ = self._fit_stasis(boot_trait, boot_dx, time_intervals)
                rates.append(rate)

        if not rates:
            return None, None

        rates = np.array(rates)
        alpha = 1 - confidence_level
        ci_lower = float(np.percentile(rates, alpha / 2 * 100))
        ci_upper = float(np.percentile(rates, (1 - alpha / 2) * 100))

        return ci_lower, ci_upper
