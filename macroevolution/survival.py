# =============================================================================
# FILE: macroevolution/survival.py
# =============================================================================
"""
Survival Analysis Module for PaleoAST

Kaplan-Meier survival curves and Log-rank tests for analyzing
fossil taxon longevity and extinction risk.

Mathematical Foundation:

1. Kaplan-Meier Estimator:
    S(t) = Π_{t_i < t} (1 - d_i / n_i)

where:
    t_i = time point i
    d_i = number of events (deaths/extinctions) at t_i
    n_i = number at risk just before t_i

2. Nelson-Aalen Estimator (Cumulative Hazard):
    H(t) = Σ_{t_i < t} (d_i / n_i)

3. Log-Rank Test:
    H₀: S₁(t) = S₂(t) for all t
    H₁: S₁(t) ≠ S₂(t) for some t

    Q = Σ (O_i - E_i)² / Var(O_i - E_i)

where O = observed, E = expected under null.

Reference:
    Kaplan, E.L. and Meier, P. (1958). Nonparametric estimation
    from incomplete observations. JASA, 53(282), 457-481.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class SurvivalResult:
    """
    Container for Kaplan-Meier survival analysis results.

    Attributes:
        times: Unique event times
        survival_prob: Survival probability at each time
        std_error: Standard error of survival estimate
        lower_ci: Lower 95% confidence interval
        upper_ci: Upper 95% confidence interval
        n_at_risk: Number at risk at each time
        n_events: Number of events at each time
        median_survival: Median survival time (if reached)
        events: Event indicator (1=event, 0=censored)
        times_full: Full time vector (including censored)
    """

    times: npt.NDArray
    survival_prob: npt.NDArray
    std_error: npt.NDArray
    lower_ci: npt.NDArray
    upper_ci: npt.NDArray
    n_at_risk: npt.NDArray
    n_events: npt.NDArray
    median_survival: float | None
    events: npt.NDArray
    times_full: npt.NDArray

    def summary(self) -> str:
        """Generate summary text."""
        median_str = (
            f"{self.median_survival:.4f}"
            if self.median_survival is not None
            else "Not reached"
        )
        return (
            f"{_('Kaplan-Meier Survival Analysis')}\n"
            f"{'=' * 50}\n"
            f"{_('Median survival time: {0}').format(median_str)}\n"
            f"{_('Final survival probability: {0:.4f}').format(self.survival_prob[-1] if len(self.survival_prob) > 0 else float('nan'))}\n"
            f"{_('Time points: {0}').format(len(self.times))}"
        )


@dataclass
class LogRankResult:
    """
    Container for Log-rank test results.

    Attributes:
        statistic: Chi-square test statistic
        p_value: P-value
        degrees_of_freedom: Degrees of freedom (always 1 for log-rank)
        group1_events: Observed events in group 1
        group1_expected: Expected events in group 1
        group2_events: Observed events in group 2
        group2_expected: Expected events in group 2
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    group1_events: float
    group1_expected: float
    group2_events: float
    group2_expected: float

    def summary(self) -> str:
        """Generate summary text."""
        sig = "**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "")
        return (
            f"{_('Log-Rank Test Results')}\n"
            f"{'=' * 50}\n"
            f"{_('Test statistic (χ²): {0:.4f}').format(self.statistic)}\n"
            f"{_('P-value: {0:.4f} {1}').format(self.p_value, sig)}\n"
            f"{_('Degrees of freedom: {0}').format(self.degrees_of_freedom)}\n"
            f"{_('Group 1: O={0:.1f}, E={1:.1f}').format(self.group1_events, self.group1_expected)}\n"
            f"{_('Group 2: O={0:.1f}, E={1:.1f}').format(self.group2_events, self.group2_expected)}"
        )


class KaplanMeierAnalyzer:
    """
    Kaplan-Meier survival curve estimator.

    Analyzes time-to-event data commonly used in paleontology
    for studying taxon longevity, species duration, and extinction timing.
    """

    def __init__(self) -> None:
        """Initialize the Kaplan-Meier analyzer."""
        self._logger = logging.getLogger(f"{__name__}.KaplanMeierAnalyzer")
        self._last_result: SurvivalResult | None = None
        self._logger.info("KaplanMeierAnalyzer initialized")

    def fit(
        self,
        times: npt.NDArray,
        events: npt.NDArray,
        confidence_level: float = 0.95,
    ) -> SurvivalResult:
        """
        Compute Kaplan-Meier survival curve.

        Parameters:
            times: Time to event or censoring (n,)
            events: Event indicator (1=event/death, 0=censored) (n,)
            confidence_level: Confidence interval level (default 0.95)

        Returns:
            SurvivalResult with survival probabilities and CIs

        Note:
            Times should be non-negative.
            Events should be 0 (censored) or 1 (event occurred).
        """
        t = validate_data_array(times, allow_nan=False, name="times")
        e = validate_data_array(events, allow_nan=False, name="events")

        if t.shape != e.shape:
            raise ComputationError(
                f"Times and events must have same shape: {t.shape} vs {e.shape}"
            )

        if np.any(t < 0):
            raise ComputationError("Times must be non-negative")

        # Ensure binary events (0 or 1)
        e = (e > 0).astype(int)

        n = len(t)
        self._logger.info(f"Computing Kaplan-Meier survival curve for {n} observations")

        # Sort by time
        sort_idx = np.argsort(t)
        t_sorted = t[sort_idx]
        e_sorted = e[sort_idx]

        # Get unique event times (where event occurred)
        unique_times, first_idx = np.unique(t_sorted, return_index=True)
        event_mask = e_sorted[first_idx] == 1

        event_times = unique_times[event_mask]
        n_unique = len(event_times)

        if n_unique == 0:
            self._logger.warning("No events observed, returning trivial survival curve")
            survival_prob = np.ones(1)
            std_error = np.zeros(1)
            lower_ci = np.ones(1)
            upper_ci = np.ones(1)
            n_at_risk = np.array([n])
            n_events = np.zeros(1)
            median_survival = None
        else:
            n_at_risk = np.zeros(n_unique)
            n_events = np.zeros(n_unique, dtype=int)
            survival_prob = np.ones(n_unique)
            std_error = np.zeros(n_unique)

            # Greenwood's formula for standard error
            for i, t_i in enumerate(event_times):
                # Number at risk just before this time
                n_at_risk[i] = np.sum(t_sorted >= t_i)

                # Number of events at this time
                n_events[i] = np.sum((t_sorted == t_i) & (e_sorted == 1))

                if n_at_risk[i] > 0:
                    # Kaplan-Meier product limit estimator
                    survival_prob[i] = survival_prob[i - 1] if i > 0 else 1.0
                    survival_prob[i] *= (1 - n_events[i] / n_at_risk[i])

                    # Greenwood's formula
                    if n_at_risk[i] > n_events[i]:
                        denom = n_at_risk[i] * (n_at_risk[i] - n_events[i])
                        std_error[i] = (
                            std_error[i - 1] ** 2
                            if i > 0
                            else survival_prob[i] ** 2
                        ) * (n_events[i] / denom)
                    else:
                        std_error[i] = std_error[i - 1] ** 2 if i > 0 else 0
                else:
                    std_error[i] = std_error[i - 1] ** 2 if i > 0 else 0

            std_error = np.sqrt(std_error)

            # Confidence intervals (log-log transform for bounded CIs)
            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            log_log_surv = np.log(-np.log(survival_prob + 1e-10))
            log_log_se = z * std_error / ((survival_prob + 1e-10) * np.log(survival_prob + 1e-10))

            with np.errstate(divide="ignore", invalid="ignore"):
                lower_ci = np.exp(-np.exp(log_log_surv + log_log_se))
                upper_ci = np.exp(-np.exp(log_log_surv - log_log_se))

            # Clip to valid range
            lower_ci = np.clip(lower_ci, 0, 1)
            upper_ci = np.clip(upper_ci, 0, 1)

            # Find median survival time
            if survival_prob[-1] <= 0.5:
                idx = np.searchsorted(survival_prob, 0.5, side="left")
                if idx < len(event_times):
                    median_survival = float(event_times[idx])
                else:
                    median_survival = None
            else:
                median_survival = None

        result = SurvivalResult(
            times=event_times,
            survival_prob=survival_prob,
            std_error=std_error,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            n_at_risk=n_at_risk,
            n_events=n_events,
            median_survival=median_survival,
            events=e,
            times_full=t,
        )

        self._last_result = result
        self._logger.info(
            f"Kaplan-Meier complete: median survival = {median_survival}, "
            f"n_events = {np.sum(e)}"
        )
        return result

    @property
    def last_result(self) -> SurvivalResult | None:
        """Get the last survival result."""
        return self._last_result


def log_rank_test(
    times1: npt.NDArray,
    events1: npt.NDArray,
    times2: npt.NDArray,
    events2: npt.NDArray,
) -> LogRankResult:
    """
    Log-rank test for comparing two survival curves.

    Tests whether two groups have statistically different
    survival experiences.

    Parameters:
        times1: Time to event for group 1
        events1: Event indicator for group 1
        times2: Time to event for group 2
        events2: Event indicator for group 2

    Returns:
        LogRankResult with test statistic and p-value

    Mathematical Background:
        The log-rank test compares observed vs expected events
        at each time point across groups:

        Q = Σ (O₁i - E₁i)² / Var(O₁i - E₁i)

        Under H₀, Q follows χ²(1) distribution.
    """
    t1 = validate_data_array(times1, allow_nan=False, name="times1")
    e1 = validate_data_array(events1, allow_nan=False, name="events1")
    t2 = validate_data_array(times2, allow_nan=False, name="times2")
    e2 = validate_data_array(events2, allow_nan=False, name="events2")

    if t1.shape != e1.shape:
        raise ComputationError(f"Group 1 times and events shape mismatch")
    if t2.shape != e2.shape:
        raise ComputationError(f"Group 2 times and events shape mismatch")

    e1 = (e1 > 0).astype(int)
    e2 = (e2 > 0).astype(int)

    logger.info(f"Computing log-rank test: n1={len(t1)}, n2={len(t2)}")

    # Combine and sort
    combined_t = np.concatenate([t1, t2])
    combined_e = np.concatenate([e1, e2])
    group_indicator = np.concatenate([np.ones(len(t1)), 2 * np.ones(len(t2))])

    sort_idx = np.argsort(combined_t)
    t_sorted = combined_t[sort_idx]
    e_sorted = combined_e[sort_idx]
    g_sorted = group_indicator[sort_idx]

    # Get unique event times
    unique_times = np.unique(t_sorted[e_sorted == 1])

    O1_total = 0.0
    E1_total = 0.0
    var_total = 0.0

    for t_i in unique_times:
        # Number at risk in each group
        n1_i = np.sum((t1 >= t_i) | ((t1 == t_i) & (e1 == 1)))
        n2_i = np.sum((t2 >= t_i) | ((t2 == t_i) & (e2 == 1)))
        n_i = n1_i + n2_i

        # Events at this time
        d1_i = np.sum((t_sorted == t_i) & (e_sorted == 1) & (g_sorted == 1))
        d2_i = np.sum((t_sorted == t_i) & (e_sorted == 1) & (g_sorted == 2))
        d_i = d1_i + d2_i

        if n_i > 0:
            # Expected events in group 1
            e1_i = n1_i * d_i / n_i

            # Variance
            var_i = (n1_i * n2_i * d_i * (n_i - d_i)) / (n_i**2 * (n_i - 1)) if n_i > 1 else 0

            O1_total += d1_i
            E1_total += e1_i
            var_total += var_i

    # Chi-square statistic
    if var_total > 0:
        Q = (O1_total - E1_total) ** 2 / var_total
    else:
        Q = 0.0

    # P-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(Q, df=1)

    result = LogRankResult(
        statistic=float(Q),
        p_value=float(p_value),
        degrees_of_freedom=1,
        group1_events=float(O1_total),
        group1_expected=float(E1_total),
        group2_events=float(d1_i) if "d1_i" in dir() else 0.0,
        group2_expected=float(d_i - e1_i) if "d_i" in dir() and "e1_i" in dir() else 0.0,
    )

    logger.info(f"Log-rank test complete: χ²={Q:.4f}, p={p_value:.4f}")
    return result
