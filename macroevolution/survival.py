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
version: 1.0.1
"""

import logging
from dataclasses import dataclass

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
        median_str = f"{self.median_survival:.4f}" if self.median_survival is not None else "Not reached"
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
        t = validate_data_array(times, allow_nan=False, name="times", preserve_dimensions=True)
        e = validate_data_array(events, allow_nan=False, name="events", preserve_dimensions=True)

        # Flatten to 1D (preserve_dimensions may return 2D)
        t = np.asarray(t).ravel()
        e = np.asarray(e).ravel()

        if t.shape != e.shape:
            raise ComputationError(f"Times and events must have same shape: {t.shape} vs {e.shape}")

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
            greenwood_sum = 0.0
            for i, t_i in enumerate(event_times):
                # Number at risk just before this time
                n_at_risk[i] = np.sum(t_sorted >= t_i)

                # Number of events at this time
                n_events[i] = np.sum((t_sorted == t_i) & (e_sorted == 1))

                if n_at_risk[i] > 0:
                    # Kaplan-Meier product limit estimator
                    survival_prob[i] = survival_prob[i - 1] if i > 0 else 1.0
                    survival_prob[i] *= 1 - n_events[i] / n_at_risk[i]

                    # Greenwood's formula:
                    # Var(S_i) = S_i^2 * sum_j d_j / (n_j * (n_j - d_j))
                    if n_events[i] > 0 and n_at_risk[i] > n_events[i]:
                        greenwood_sum += n_events[i] / (n_at_risk[i] * (n_at_risk[i] - n_events[i]))
                    std_error[i] = survival_prob[i] * np.sqrt(greenwood_sum)
                else:
                    std_error[i] = std_error[i - 1] if i > 0 else 0

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
                below = np.where(survival_prob <= 0.5)[0]
                median_survival = float(event_times[below[0]]) if len(below) > 0 else None
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
        self._logger.info(f"Kaplan-Meier complete: median survival = {median_survival}, n_events = {np.sum(e)}")
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
        raise ComputationError("Group 1 times and events shape mismatch")
    if t2.shape != e2.shape:
        raise ComputationError("Group 2 times and events shape mismatch")

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


@dataclass
class CoxPHResult:
    """
    Container for Cox proportional hazards model results.

    Attributes:
        beta: Regression coefficients
        exp_beta: Hazard ratios (exp(beta))
        se: Standard errors
        z_scores: Z statistics
        p_values: P-values
        concordance: C-index (concordance index)
        log_likelihood: Log-likelihood
        AIC: Akaike Information Criterion
    """

    beta: npt.NDArray
    exp_beta: npt.NDArray
    se: npt.NDArray
    z_scores: npt.NDArray
    p_values: npt.NDArray
    concordance: float
    log_likelihood: float
    AIC: float

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Cox Proportional Hazards Model"),
            "=" * 50,
            f"{'Covariate':<12} {'Beta':>10} {'HR':>10} {'SE':>10} {'Z':>10} {'P':>10}",
            "-" * 65,
        ]
        for i in range(len(self.beta)):
            sig = "**" if self.p_values[i] < 0.01 else ("*" if self.p_values[i] < 0.05 else "")
            lines.append(
                f"{i:<12} {self.beta[i]:>10.4f} {self.exp_beta[i]:>10.4f} "
                f"{self.se[i]:>10.4f} {self.z_scores[i]:>10.4f} {self.p_values[i]:>10.4f}{sig}"
            )
        lines.append("-" * 65)
        lines.append(f"C-index: {self.concordance:.4f}")
        lines.append(f"Log-likelihood: {self.log_likelihood:.4f}")
        lines.append(f"AIC: {self.AIC:.4f}")
        return "\n".join(lines)


def _compute_concordance(
    durations: npt.NDArray,
    events: npt.NDArray,
    covariates: npt.NDArray,
    beta: npt.NDArray,
) -> float:
    """
    Compute concordance index for Cox model (vectorized implementation).

    Parameters:
        durations: Survival times
        events: Event indicators
        covariates: Covariate matrix
        beta: Coefficients

    Returns:
        Concordance index (C-index)
    """
    # Handle 2D inputs (from validate_data_array with preserve_dimensions=True)
    durations_arr = np.asarray(durations).ravel()
    events_arr = np.asarray(events).ravel()
    covariates_arr = np.asarray(covariates)
    beta_arr = np.asarray(beta)

    n = len(durations_arr)
    if n < 2:
        return 0.5

    # Linear predictor
    lp = np.asarray(covariates_arr @ beta_arr).ravel()

    # For large datasets, sample pairs to avoid O(n²) computation
    max_pairs = 10000
    n_pairs = n * (n - 1) // 2

    if n_pairs > max_pairs:
        # Sample random pairs using a local Generator so the global
        # numpy RNG is not reseeded as a side-effect.  The previous
        # ``np.random.seed(42)`` silently contaminated every downstream
        # stochastic operation in the same process.
        rng = np.random.default_rng(42)
        idx1 = rng.integers(0, n, max_pairs)
        idx2 = rng.integers(0, n, max_pairs)
        # Ensure idx2 > idx1 to avoid duplicates
        mask = idx2 > idx1
        idx1, idx2 = idx1[mask], idx2[mask]
    else:
        # Use all pairs
        idx1, idx2 = np.triu_indices(n, k=1)

    # Extract pairs
    t1, t2 = durations_arr[idx1], durations_arr[idx2]
    e1, e2 = events_arr[idx1], events_arr[idx2]
    lp1, lp2 = lp[idx1], lp[idx2]

    # Determine which pairs are evaluable (at least one event)
    has_event = (e1 == 1) | (e2 == 1)

    # Initialize counts
    concordant = 0.0
    tied_risk = 0.0
    total_pairs = 0

    # Case 1: Both have events - compare times, higher risk (lower t) should have higher lp
    both_events = has_event & (e1 == 1) & (e2 == 1)
    if np.any(both_events):
        t1_ev, t2_ev = t1[both_events], t2[both_events]
        lp1_ev, lp2_ev = lp1[both_events], lp2[both_events]
        # Lower time = higher risk = should have higher lp
        risk_higher_1 = lp1_ev > lp2_ev
        risk_higher_2 = lp2_ev > lp1_ev
        time_lower_1 = t1_ev < t2_ev
        time_lower_2 = t2_ev < t1_ev
        concordant += np.sum((time_lower_1 & risk_higher_1) | (time_lower_2 & risk_higher_2))
        tied_risk += np.sum((lp1_ev == lp2_ev) | ((t1_ev == t2_ev) & (lp1_ev == lp2_ev)))

    # Case 2: Only i has event - if t_i <= t_j, i is at higher risk
    i_only_event = has_event & (e1 == 1) & (e2 == 0)
    if np.any(i_only_event):
        t1_i, t2_i = t1[i_only_event], t2[i_only_event]
        lp1_i, lp2_i = lp1[i_only_event], lp2[i_only_event]
        i_higher_risk = lp1_i > lp2_i
        i_earlier_or_same = t1_i <= t2_i
        concordant += np.sum(i_earlier_or_same & i_higher_risk)
        tied_risk += np.sum((lp1_i == lp2_i) & i_earlier_or_same)

    # Case 3: Only j has event - if t_j <= t_i, j is at higher risk
    j_only_event = has_event & (e1 == 0) & (e2 == 1)
    if np.any(j_only_event):
        t1_j, t2_j = t1[j_only_event], t2[j_only_event]
        lp1_j, lp2_j = lp1[j_only_event], lp2[j_only_event]
        j_higher_risk = lp2_j > lp1_j
        j_earlier_or_same = t2_j <= t1_j
        concordant += np.sum(j_earlier_or_same & j_higher_risk)
        tied_risk += np.sum((lp1_j == lp2_j) & j_earlier_or_same)

    total_pairs = np.sum(has_event)
    if total_pairs == 0:
        return 0.5

    c_index = (concordant + tied_risk) / total_pairs
    return float(np.clip(c_index, 0, 1))


def cox_ph(
    durations: npt.NDArray,
    events: npt.NDArray,
    covariates: npt.NDArray,
    max_iter: int = 100,
) -> CoxPHResult:
    """
    Cox proportional hazards model using partial likelihood.

    Tries to use lifelines library if available, otherwise falls back
    to scipy optimization.
    """
    try:
        import lifelines  # noqa: F401  # availability check

        return _cox_ph_lifelines(durations, events, covariates)
    except ImportError:
        logger.info("lifelines not available, using scipy optimization")
        return _cox_ph_scipy(durations, events, covariates, max_iter)


def _cox_ph_lifelines(
    durations: npt.NDArray,
    events: npt.NDArray,
    covariates: npt.NDArray,
) -> CoxPHResult:
    """Cox PH using lifelines library."""
    from lifelines import CoxPHFitter

    t = validate_data_array(durations, allow_nan=False, name="durations")
    e = validate_data_array(events, allow_nan=False, name="events").astype(int)
    X = np.asarray(covariates)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    _n, p = X.shape

    # Create DataFrame for lifelines
    import pandas as pd

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["duration"] = t
    df["event"] = e

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    # Extract results
    summary = cph.summary

    beta = summary["coef"].values
    se = summary["se(coef)"].values
    z_scores = summary["z"].values
    p_values = summary["p"].values
    exp_beta = np.exp(beta)

    # Concordance index
    concordance = cph.concordance_index_

    # Log-likelihood
    log_likelihood = cph.log_likelihood_

    # AIC
    aic = cph.AIC_partial_

    logger.info(f"Cox PH (lifelines) complete: {p} covariates, C-index={concordance:.4f}")

    return CoxPHResult(
        beta=beta,
        exp_beta=exp_beta,
        se=se,
        z_scores=z_scores,
        p_values=p_values,
        concordance=concordance,
        log_likelihood=log_likelihood,
        AIC=aic,
    )


def _cox_ph_scipy(
    durations: npt.NDArray,
    events: npt.NDArray,
    covariates: npt.NDArray,
    max_iter: int = 100,
) -> CoxPHResult:
    """Cox PH using scipy optimization (fallback when lifelines unavailable)."""
    from scipy import optimize, stats

    t = validate_data_array(durations, allow_nan=False, name="durations")
    e = validate_data_array(events, allow_nan=False, name="events").astype(int)
    X = np.asarray(covariates)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    if len(t) != n or len(e) != n:
        raise ComputationError(f"Shape mismatch: durations={len(t)}, events={len(e)}, covariates={X.shape}")

    if np.any(t < 0):
        raise ComputationError("Durations must be non-negative")

    logger.info(f"Fitting Cox PH model: n={n}, p={p}")

    # Standardize covariates
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std

    # Precompute unique event times (outside optimization loop)
    event_mask = e == 1
    unique_times = np.unique(t[event_mask])

    # Precompute at-risk indices and event indices for each unique time
    risk_sets = {}
    event_indices = {}
    for t_i in unique_times:
        risk_sets[t_i] = np.where(t >= t_i)[0]
        event_indices[t_i] = np.where((t == t_i) & (e == 1))[0]

    # Partial log-likelihood
    def neg_partial_log_likelihood(beta: np.ndarray) -> float:
        risk_scores = np.exp(X_scaled @ beta)
        log_lik = 0.0

        for t_i in unique_times:
            at_risk_idx = risk_sets[t_i]
            if len(at_risk_idx) == 0:
                continue

            risk_set = risk_scores[at_risk_idx]
            denom = np.sum(risk_set)

            for idx in event_indices[t_i]:
                numer = risk_scores[idx]
                if denom > 0 and numer > 0:
                    log_lik += np.log(numer / denom)

        return -log_lik

    # Better initial values using univariate CoxPH estimates per covariate
    beta_init = np.zeros(p)
    for j in range(p):
        try:
            # Use 0.1 * sign of correlation as rough initial guess
            if len(t) > 2:
                corr = np.corrcoef(t[event_mask], X_scaled[event_mask, j])[0, 1]
                beta_init[j] = 0.1 * np.sign(corr) if not np.isnan(corr) else 0.0
        except Exception:
            beta_init[j] = 0.0

    # Optimize using L-BFGS-B (better for smooth problems)
    result = optimize.minimize(
        neg_partial_log_likelihood,
        beta_init,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-6},
    )

    beta = result.x

    # Compute standard errors via numerical Hessian
    eps = 1e-5
    hessian = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            beta_ij = beta.copy()
            beta_ij[i] += eps
            beta_ij[j] += eps
            hessian[i, j] = (neg_partial_log_likelihood(beta_ij) - neg_partial_log_likelihood(beta)) / eps
            hessian[j, i] = hessian[i, j]

    try:
        info_inv = np.linalg.inv(hessian)
        se = np.sqrt(np.abs(np.diag(info_inv)))
    except np.linalg.LinAlgError:
        se = np.ones(p) * 0.5

    z_scores = beta / np.maximum(se, 1e-10)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    concordance = _compute_concordance(t, e, X_scaled, beta)
    log_likelihood = -result.fun
    aic = 2 * p - 2 * log_likelihood

    logger.info(f"Cox PH (scipy) complete: C-index={concordance:.4f}")

    return CoxPHResult(
        beta=beta,
        exp_beta=np.exp(beta),
        se=se,
        z_scores=z_scores,
        p_values=p_values,
        concordance=concordance,
        log_likelihood=log_likelihood,
        AIC=aic,
    )


class CoxPHAnalyzer:
    """
    Cox Proportional Hazards Model analyzer.

    Analyzes survival data with covariates to identify risk factors.
    """

    def __init__(self) -> None:
        """Initialize the Cox PH analyzer."""
        self._logger = logging.getLogger(f"{__name__}.CoxPHAnalyzer")
        self._last_result: CoxPHResult | None = None
        self._logger.info("CoxPHAnalyzer initialized")

    def fit(
        self,
        durations: npt.NDArray,
        events: npt.NDArray,
        covariates: npt.NDArray,
    ) -> CoxPHResult:
        """
        Fit Cox proportional hazards model.

        Parameters:
            durations: Time to event or censoring
            events: Event indicator (1=event, 0=censored)
            covariates: Covariate matrix (n_samples, n_covariates)

        Returns:
            CoxPHResult with model coefficients
        """
        result = cox_ph(durations, events, covariates)
        self._last_result = result
        return result

    @property
    def last_result(self) -> CoxPHResult | None:
        """Get the last Cox PH result."""
        return self._last_result
