# =============================================================================
# FILE: statistics/univariate.py
# =============================================================================
"""
Univariate Statistics & Hypothesis Testing Module for PaleoAST

Provides descriptive statistics, normality tests, t-tests, ANOVA,
and non-parametric alternatives for paleontological data exploration.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


@dataclass
class ColumnStats:
    """Descriptive statistics for a single variable."""

    name: str
    n: int
    mean: float
    std: float
    variance: float
    min_val: float
    max_val: float
    median: float
    skewness: float
    kurtosis: float
    se: float
    ci_95: tuple[float, float]


@dataclass
class SummaryResult:
    """Summary statistics for multiple variables."""

    columns: list[ColumnStats]

    def summary(self) -> str:
        lines = [
            _("Descriptive Statistics"),
            "=" * 80,
            f"{'Variable':<15} {'N':>5} {'Mean':>10} {'Std':>10} {'Skew':>8} {'Kurt':>8} {'SE':>10}",
            "-" * 80,
        ]
        for c in self.columns:
            lines.append(
                f"{c.name:<15} {c.n:>5} {c.mean:>10.4f} {c.std:>10.4f} "
                f"{c.skewness:>8.3f} {c.kurtosis:>8.3f} {c.se:>10.4f}"
            )
        return "\n".join(lines)


@dataclass
class NormalityResult:
    """Results of normality tests."""

    shapiro_stat: float
    shapiro_p: float
    anderson_stat: float
    anderson_critical: dict[float, float]
    is_normal_shapiro: bool
    is_normal_anderson: bool


@dataclass
class TTestResult:
    """Result of a t-test."""

    statistic: float
    p_value: float
    test_type: str
    n1: int
    n2: int
    mean1: float
    mean2: float
    significant: bool


@dataclass
class ANOVAResult:
    """Result of one-way ANOVA with optional post-hoc."""

    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    n_groups: int
    significant: bool
    tukey_results: list[dict] | None = None

    def summary(self) -> str:
        sig = "**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "ns")
        lines = [
            _("One-Way ANOVA"),
            "=" * 50,
            f"F = {self.f_statistic:.4f}, p = {self.p_value:.4f} {sig}",
            f"df = ({self.df_between}, {self.df_within})",
            f"SS_between = {self.ss_between:.4f}, SS_within = {self.ss_within:.4f}",
        ]
        if self.tukey_results:
            lines.append("")
            lines.append(_("Tukey HSD Post-Hoc:"))
            lines.append(f"{'Group A':<10} {'Group B':<10} {'Diff':>10} {'p-adj':>10} {'Sig':>5}")
            lines.append("-" * 50)
            for r in self.tukey_results:
                sig_mark = "**" if r["p_adj"] < 0.01 else ("*" if r["p_adj"] < 0.05 else "ns")
                lines.append(
                    f"{r['group_a']:<10} {r['group_b']:<10} {r['diff']:>10.4f} {r['p_adj']:>10.4f} {sig_mark:>5}"
                )
        return "\n".join(lines)


@dataclass
class KruskalResult:
    """Result of Kruskal-Wallis test."""

    statistic: float
    p_value: float
    n_groups: int
    significant: bool


class UnivariateAnalyzer:
    """Engine for univariate statistics and hypothesis testing."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.UnivariateAnalyzer")
        self._lock = threading.RLock()

    def summary_statistics(
        self, data: npt.NDArray, column_names: list[str] | None = None, columns: list[int] | None = None
    ) -> SummaryResult:
        """
        Compute descriptive statistics for selected columns.

        Parameters:
            data: Data matrix (n_samples x n_variables)
            column_names: Names for each column
            columns: Indices of columns to analyze (None = all)

        Returns:
            SummaryResult
        """
        with self._lock:
            data = validate_data_array(data, name="data")
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_cols = data.shape[1]
            if column_names is None:
                column_names = [f"Var_{i + 1}" for i in range(n_cols)]
            if columns is None:
                columns = list(range(n_cols))

            results = []
            for idx in columns:
                col = data[:, idx]
                valid = col[~np.isnan(col)]
                n = len(valid)
                if n < 2:
                    results.append(
                        ColumnStats(
                            name=column_names[idx],
                            n=n,
                            mean=np.nan,
                            std=np.nan,
                            variance=np.nan,
                            min_val=np.nan,
                            max_val=np.nan,
                            median=np.nan,
                            skewness=np.nan,
                            kurtosis=np.nan,
                            se=np.nan,
                            ci_95=(np.nan, np.nan),
                        )
                    )
                    continue

                m = np.mean(valid)
                s = np.std(valid, ddof=1)
                se = s / np.sqrt(n)
                t_crit = sp_stats.t.ppf(0.975, df=n - 1)

                results.append(
                    ColumnStats(
                        name=column_names[idx],
                        n=n,
                        mean=m,
                        std=s,
                        variance=np.var(valid, ddof=1),
                        min_val=np.min(valid),
                        max_val=np.max(valid),
                        median=np.median(valid),
                        skewness=float(sp_stats.skew(valid)),
                        kurtosis=float(sp_stats.kurtosis(valid)),
                        se=se,
                        ci_95=(m - t_crit * se, m + t_crit * se),
                    )
                )

            self._logger.info(f"Summary statistics computed for {len(results)} columns")
            return SummaryResult(columns=results)

    def normality_test(self, data: npt.NDArray, column: int = 0) -> NormalityResult:
        """
        Test normality of a variable using Shapiro-Wilk and Anderson-Darling.

        Parameters:
            data: Data matrix or 1D array
            column: Column index to test

        Returns:
            NormalityResult
        """
        with self._lock:
            if data.ndim == 2:
                col = data[:, column]
            else:
                col = data

            valid = col[~np.isnan(col)]
            if len(valid) < 3:
                raise ComputationError("Need at least 3 non-NaN values for normality test")

            # Shapiro-Wilk (best for n < 5000)
            if len(valid) <= 5000:
                sw_stat, sw_p = sp_stats.shapiro(valid)
            else:
                sw_stat, sw_p = sp_stats.shapiro(valid[:5000])

            # Anderson-Darling
            ad_result = sp_stats.anderson(valid, dist="norm")
            ad_stat = ad_result.statistic
            ad_critical = dict(zip(ad_result.significance_level, ad_result.critical_values, strict=False))

            # Check at 5% level
            is_normal_sw = sw_p > 0.05
            is_normal_ad = ad_stat < ad_critical.get(5.0, 0)

            return NormalityResult(
                shapiro_stat=float(sw_stat),
                shapiro_p=float(sw_p),
                anderson_stat=float(ad_stat),
                anderson_critical=ad_critical,
                is_normal_shapiro=is_normal_sw,
                is_normal_anderson=is_normal_ad,
            )

    def t_test(
        self, data: npt.NDArray, column: int = 0, groups: list[int] | None = None, paired: bool = False
    ) -> TTestResult:
        """
        Perform t-test between two groups.

        Parameters:
            data: Data matrix
            column: Column to test
            groups: Group labels (must have exactly 2 unique values)
            paired: If True, perform paired t-test

        Returns:
            TTestResult
        """
        with self._lock:
            if groups is None:
                raise ComputationError("Groups required for t-test")

            unique_groups = sorted(set(groups))
            if len(unique_groups) != 2:
                raise ComputationError(f"t-test requires exactly 2 groups, got {len(unique_groups)}")

            g0, g1 = unique_groups
            if data.ndim == 2:
                vals_0 = data[[i for i, g in enumerate(groups) if g == g0], column]
                vals_1 = data[[i for i, g in enumerate(groups) if g == g1], column]
            else:
                vals_0 = data[[i for i, g in enumerate(groups) if g == g0]]
                vals_1 = data[[i for i, g in enumerate(groups) if g == g1]]

            vals_0 = vals_0[~np.isnan(vals_0)]
            vals_1 = vals_1[~np.isnan(vals_1)]

            if len(vals_0) < 2 or len(vals_1) < 2:
                raise ComputationError(
                    f"Not enough valid data after NaN filtering: group sizes {len(vals_0)}, {len(vals_1)}"
                )

            if paired:
                if len(vals_0) != len(vals_1):
                    raise ComputationError("Paired t-test requires equal sample sizes")
                t_stat, p_val = sp_stats.ttest_rel(vals_0, vals_1)
                test_type = "paired"
            else:
                t_stat, p_val = sp_stats.ttest_ind(vals_0, vals_1)
                test_type = "independent"

            return TTestResult(
                statistic=float(t_stat),
                p_value=float(p_val),
                test_type=test_type,
                n1=len(vals_0),
                n2=len(vals_1),
                mean1=float(np.mean(vals_0)),
                mean2=float(np.mean(vals_1)),
                significant=p_val < 0.05,
            )

    def one_way_anova(self, data: npt.NDArray, groups: list[int], column: int = 0, tukey: bool = True) -> ANOVAResult:
        """
        Perform one-way ANOVA with optional Tukey HSD post-hoc.

        Parameters:
            data: Data matrix or 1D array
            groups: Group labels for each sample
            column: Column to analyze (if data is 2D)
            tukey: Whether to run Tukey HSD if ANOVA is significant

        Returns:
            ANOVAResult
        """
        with self._lock:
            if data.ndim == 2:
                values = data[:, column]
            else:
                values = data

            unique_groups = sorted(set(groups))
            n_groups = len(unique_groups)

            if n_groups < 2:
                raise ComputationError("ANOVA requires at least 2 groups")

            # Collect group data, filtering NaN
            group_data = []
            group_labels = []
            for g in unique_groups:
                g_vals = values[[i for i, gr in enumerate(groups) if gr == g]]
                g_vals = g_vals[~np.isnan(g_vals)]
                if len(g_vals) > 0:
                    group_data.append(g_vals)
                    group_labels.append(g)

            if len(group_data) < 2:
                raise ComputationError("Need at least 2 non-empty groups for ANOVA")
            # scipy.stats.f_oneway raises a confusing ValueError when any
            # group has fewer than 2 observations (it cannot estimate the
            # within-group variance). Validate up front and give the user
            # an actionable message.
            single_sample_groups = [
                group_labels[i] for i, g in enumerate(group_data) if len(g) < 2
            ]
            if single_sample_groups:
                raise ComputationError(
                    "One-way ANOVA requires at least 2 observations per group; "
                    f"group(s) {single_sample_groups} have only 1."
                )

            # One-way ANOVA
            f_stat, p_val = sp_stats.f_oneway(*group_data)

            # Compute SS manually for result details
            all_vals = np.concatenate(group_data)
            grand_mean = np.mean(all_vals)

            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
            ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in group_data)

            df_between = len(group_data) - 1
            df_within = len(all_vals) - len(group_data)

            ms_between = ss_between / df_between if df_between > 0 else 0
            ms_within = ss_within / df_within if df_within > 0 else 0

            # Tukey HSD post-hoc
            tukey_results = None
            if tukey and p_val < 0.05 and len(group_data) >= 2:
                tukey_results = self._tukey_hsd(group_data, group_labels)

            return ANOVAResult(
                f_statistic=float(f_stat),
                p_value=float(p_val),
                df_between=df_between,
                df_within=df_within,
                ss_between=float(ss_between),
                ss_within=float(ss_within),
                ms_between=float(ms_between),
                ms_within=float(ms_within),
                n_groups=len(group_data),
                significant=p_val < 0.05,
                tukey_results=tukey_results,
            )

    def kruskal_wallis(self, data: npt.NDArray, groups: list[int], column: int = 0) -> KruskalResult:
        """Non-parametric alternative to one-way ANOVA."""
        with self._lock:
            if data.ndim == 2:
                values = data[:, column]
            else:
                values = data

            unique_groups = sorted(set(groups))
            group_data = []
            for g in unique_groups:
                g_vals = values[[i for i, gr in enumerate(groups) if gr == g]]
                g_vals = g_vals[~np.isnan(g_vals)]
                if len(g_vals) > 0:
                    group_data.append(g_vals)

            if len(group_data) < 2:
                raise ComputationError("Need at least 2 non-empty groups")

            stat, p_val = sp_stats.kruskal(*group_data)

            return KruskalResult(
                statistic=float(stat),
                p_value=float(p_val),
                n_groups=len(group_data),
                significant=p_val < 0.05,
            )

    def mann_whitney(self, data: npt.NDArray, groups: list[int], column: int = 0) -> TTestResult:
        """Non-parametric two-sample test (Mann-Whitney U)."""
        with self._lock:
            if data.ndim == 2:
                values = data[:, column]
            else:
                values = data

            unique_groups = sorted(set(groups))
            if len(unique_groups) != 2:
                raise ComputationError("Mann-Whitney requires exactly 2 groups")

            g0, g1 = unique_groups
            vals_0 = values[[i for i, g in enumerate(groups) if g == g0]]
            vals_1 = values[[i for i, g in enumerate(groups) if g == g1]]
            vals_0 = vals_0[~np.isnan(vals_0)]
            vals_1 = vals_1[~np.isnan(vals_1)]

            u_stat, p_val = sp_stats.mannwhitneyu(vals_0, vals_1, alternative="two-sided")

            return TTestResult(
                statistic=float(u_stat),
                p_value=float(p_val),
                test_type="mann_whitney",
                n1=len(vals_0),
                n2=len(vals_1),
                mean1=float(np.mean(vals_0)),
                mean2=float(np.mean(vals_1)),
                significant=p_val < 0.05,
            )

    def _tukey_hsd(self, group_data: list[npt.NDArray], group_labels: list) -> list[dict]:
        """Perform Tukey HSD post-hoc pairwise comparisons."""
        results = []
        n_groups = len(group_data)

        try:
            tukey_result = sp_stats.tukey_hsd(*group_data)
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    mean_diff = float(np.mean(group_data[i]) - np.mean(group_data[j]))
                    p_adj = float(tukey_result.pvalue[i, j])
                    # Compute q-statistic from p-value approximation
                    all_vals = np.concatenate(group_data)
                    df_within = len(all_vals) - n_groups
                    ms_within = np.sum([(g - np.mean(g)) ** 2 for g in group_data]) / df_within
                    # scipy's tukey_hsd uses stand_err = sqrt(MSE * (1/ni + 1/nj) / 2)
                    # and the statistic is |mean_diff| / stand_err. Match the
                    # fallback path below exactly so the reported q_stat is on
                    # the studentized-range scale.
                    se = np.sqrt(ms_within * (1.0 / len(group_data[i]) + 1.0 / len(group_data[j])) / 2.0)
                    q_stat = abs(mean_diff) / se if se > 0 else 0.0

                    results.append(
                        {
                            "group_a": group_labels[i],
                            "group_b": group_labels[j],
                            "diff": mean_diff,
                            "q_stat": float(q_stat),
                            "p_value": p_adj,
                            "significant": p_adj < 0.05,
                        }
                    )
        except (AttributeError, TypeError):
            # Fallback for older scipy without tukey_hsd.
            # The previous implementation multiplied the raw two-sided t p-value
            # by the number of comparisons (Bonferroni). That is a
            # conservative correction that is *not* the same as Tukey's HSD
            # and routinely over-corrects.
            #
            # Here we use the Studentized range (q) distribution directly.
            # This is the same statistic scipy.stats.tukey_hsd uses internally,
            # so the resulting p-values agree to numerical precision when
            # sp_stats.studentized_range is available (scipy >= 1.7).
            all_vals = np.concatenate(group_data)
            n_total = len(all_vals)
            df_within = n_total - n_groups
            ms_within = np.sum([(g - np.mean(g)) ** 2 for g in group_data]) / df_within

            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    ni = len(group_data[i])
                    nj = len(group_data[j])
                    mean_diff = np.mean(group_data[i]) - np.mean(group_data[j])
                    # scipy's tukey_hsd uses stand_err = sqrt(MSE * (1/ni + 1/nj) / 2)
                    # and the statistic is |mean_diff| / stand_err. We use
                    # exactly the same formula here so the fallback agrees
                    # with sp_stats.tukey_hsd to numerical precision.
                    se = np.sqrt(ms_within * (1.0 / ni + 1.0 / nj) / 2.0)
                    q_stat = abs(mean_diff) / se if se > 0 else 0.0
                    try:
                        if hasattr(sp_stats, "studentized_range"):
                            p_adj = float(sp_stats.studentized_range.sf(q_stat, n_groups, df_within))
                        else:
                            # Old scipy without studentized_range. We fall
                            # back to the t-distribution as a coarse
                            # approximation; this is closer to Tukey HSD than
                            # Bonferroni was, but still only approximate.
                            p_raw = 2.0 * (1.0 - sp_stats.t.cdf(q_stat, df=max(df_within, 1)))
                            p_adj = float(min(max(p_raw, 0.0), 1.0))
                    except Exception:
                        p_adj = 1.0

                    results.append(
                        {
                            "group_a": group_labels[i],
                            "group_b": group_labels[j],
                            "diff": float(mean_diff),
                            "q_stat": float(q_stat),
                            "p_value": p_adj,
                            "significant": p_adj < 0.05,
                        }
                    )

        return results

    @property
    def last_result(self):
        return None


# =============================================================================
# AICc Model Selection (Burnham & Anderson 2002)
# =============================================================================


def compute_aicc(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Compute the small-sample corrected Akaike Information Criterion (AICc).

    AICc corrects for finite sample size and is preferred over AIC when
    the sample size is small relative to the number of parameters
    (typically when n / k < 40).

    Formula (Burnham & Anderson 2002, eq. 2.2.2):
        AIC  = -2 * LL + 2 * k
        AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
    n_params : int
        Number of estimated parameters (k).
    n_obs : int
        Number of observations (n).

    Returns
    -------
    float
        The AICc value.

    Raises
    ------
    ValueError
        If n - k - 1 <= 0 (insufficient data for correction).

    References
    ----------
    Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and
    Multimodel Inference: A Practical Information-Theoretic Approach*
    (2nd ed.). Springer.
    """
    if n_params <= 0:
        raise ValueError("n_params must be a positive integer")
    if n_obs <= 0:
        raise ValueError("n_obs must be a positive integer")
    if n_params >= n_obs:
        raise ValueError(
            f"n_params ({n_params}) must be less than n_obs ({n_obs}) "
            "for AICc computation"
        )
    if n_obs - n_params - 1 <= 0:
        raise ValueError(
            f"Insufficient data for AICc correction: n ({n_obs}) - k ({n_params}) - 1 must be > 0"
        )

    k = n_params
    n = n_obs
    aic = -2.0 * log_likelihood + 2.0 * k
    correction = (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    return aic + correction


def compare_models(models: list[tuple[str, float, int, int]]) -> dict:
    """
    Compare candidate models using AICc and compute model weights.

    Parameters
    ----------
    models : list[tuple[name, log_likelihood, n_params, n_obs]]
        List of candidate models. Each entry contains:
        - name: Model identifier (str)
        - log_likelihood: Log-likelihood of the fitted model (float)
        - n_params: Number of estimated parameters (int)
        - n_obs: Number of observations (int)

    Returns
    -------
    dict
        Dictionary with keys:
        - "models": List of dicts with model results sorted by AICc
        - "best_model": Name of the best (lowest AICc) model
        - "delta_aicc": List of ΔAICc values (relative to best)
        - "weights": List of AICc weights (summing to 1)

    Raises
    ------
    ValueError
        If any model has insufficient data (n - k - 1 <= 0).

    Notes
    -----
    AICc weight for model i:
        w_i = exp(-0.5 * ΔAICc_i) / Σ_j exp(-0.5 * ΔAICc_j)

    Weights represent the relative probability that each model is the
    best among the candidate set, given the data (Burnham & Anderson 2002).

    References
    ----------
    Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and
    Multimodel Inference: A Practical Information-Theoretic Approach*
    (2nd ed.). Springer.
    """
    if not models:
        raise ValueError("models list cannot be empty")

    results = []
    for name, ll, k, n in models:
        try:
            aicc = compute_aicc(ll, k, n)
        except ValueError:
            raise ValueError(
                f"Model '{name}' has insufficient data for AICc "
                f"(n_params={k}, n_obs={n})"
            )
        results.append({"name": name, "aicc": aicc, "log_likelihood": ll, "n_params": k, "n_obs": n})

    # Sort by AICc
    results.sort(key=lambda x: x["aicc"])
    best_aicc = results[0]["aicc"]

    delta_aicc = [r["aicc"] - best_aicc for r in results]

    # Compute weights
    log_weights = [-0.5 * d for d in delta_aicc]
    max_log_w = max(log_weights)  # for numerical stability
    weights = [np.exp(lw - max_log_w) for lw in log_weights]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    for i, r in enumerate(results):
        r["delta_aicc"] = delta_aicc[i]
        r["weight"] = weights[i]

    return {
        "models": results,
        "best_model": results[0]["name"],
        "delta_aicc": delta_aicc,
        "weights": weights,
    }


# =============================================================================
# Effect Size Measures (Cohen 1988, Cohen 1992, Lakens 2013)
# =============================================================================


def cohens_d(group1: npt.NDArray, group2: npt.NDArray) -> float:
    """
    Compute Cohen's d for the difference between two independent groups.

    Cohen's d is a standardised measure of effect size representing the
    difference between two means in units of the pooled standard deviation.

    Formula (Cohen 1988, eq. 3.6):
        d = (mean_1 - mean_2) / s_pooled
        s_pooled = sqrt(((n_1-1)*s_1^2 + (n_2-1)*s_2^2) / (n_1 + n_2 - 2))

    Parameters
    ----------
    group1, group2 : array-like
        Data values for each group. NaN values are removed.

    Returns
    -------
    float
        Cohen's d effect size.

    Raises
    ------
    ValueError
        If either group has fewer than 2 valid observations after
        removing NaN.

    Notes
    -----
    Interpretation guidelines (Cohen 1988):
        |d| ≈ 0.2  -> small
        |d| ≈ 0.5  -> medium
        |d| ≈ 0.8  -> large

    For paired samples, use `cohens_d_paired` or the `effsize` library.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Lawrence Erlbaum Associates.
    Cohen, J. (1992). A power primer. *Psychological Bulletin*, 112(1), 155–159.
    Lakens, D. (2013). Calculating and reporting effect sizes to
    facilitate cumulative science. *Frontiers in Psychology*, 4, 863.
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1 = len(g1)
    n2 = len(g2)

    if n1 < 2:
        raise ValueError(f"group1 must have at least 2 valid observations, got {n1}")
    if n2 < 2:
        raise ValueError(f"group2 must have at least 2 valid observations, got {n2}")

    mean1 = np.mean(g1)
    mean2 = np.mean(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)

    s_pooled = np.sqrt(((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0))

    if s_pooled == 0.0:
        raise ValueError("Pooled standard deviation is zero (all values identical)")

    return (mean1 - mean2) / s_pooled


def eta_squared(F_statistic: float, df_between: int, df_within: int) -> float:
    """
    Compute partial eta-squared (η²) from ANOVA results.

    η² represents the proportion of variance in the dependent variable
    that is explained by the independent variable.

    Formula (Cohen 1988, eq. 8.2.3):
        η² = (F * df_between) / (F * df_between + df_within)

    Parameters
    ----------
    F_statistic : float
        Observed F-statistic from ANOVA.
    df_between : int
        Degrees of freedom between groups.
    df_within : int
        Degrees of freedom within groups (error df).

    Returns
    -------
    float
        Partial eta-squared (bounded [0, 1]).

    Raises
    ------
    ValueError
        If df_between <= 0 or df_within < 0.

    Notes
    -----
    Interpretation guidelines (Cohen 1988):
        η² ≈ 0.01  -> small
        η² ≈ 0.06  -> medium
        η² ≈ 0.14  -> large

    Warning: η² is a biased overestimate of the true effect size,
    especially for small samples. Consider using omega-squared (ω²)
    for more accurate estimates.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Lawrence Erlbaum Associates.
    Lakens, D. (2013). Calculating and reporting effect sizes to
    facilitate cumulative science. *Frontiers in Psychology*, 4, 863.
    """
    if df_between <= 0:
        raise ValueError("df_between must be a positive integer")
    if df_within < 0:
        raise ValueError("df_within must be a non-negative integer")

    numerator = F_statistic * df_between
    denominator = numerator + df_within

    if denominator == 0.0:
        return 0.0

    eta2 = numerator / denominator

    # Bound to [0, 1] for numerical stability
    return float(np.clip(eta2, 0.0, 1.0))


def omega_squared(F_statistic: float, df_between: int, df_within: int, n: int) -> float:
    """
    Compute omega-squared (ω²) from ANOVA results.

    ω² is a less biased alternative to η² that better estimates the
    population effect size, especially for small samples.

    Formula (Cohen 1988, eq. 8.2.4; adapted from Lakens 2013):
        ω² = (F*df_between - df_between) / (F*df_between + df_within + 1)

    Parameters
    ----------
    F_statistic : float
        Observed F-statistic from ANOVA.
    df_between : int
        Degrees of freedom between groups.
    df_within : int
        Degrees of freedom within groups (error df).
    n : int
        Total number of observations.

    Returns
    -------
    float
        Omega-squared (bounded [0, 1]).

    Raises
    ------
    ValueError
        If df_between <= 0, df_within < 0, or n <= df_between.

    Notes
    -----
    Interpretation guidelines (Cohen 1988):
        ω² ≈ 0.01  -> small
        ω² ≈ 0.06  -> medium
        ω² ≈ 0.14  -> large

    ω² is generally preferred over η² because it provides a less
    biased estimate of the population effect size.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Lawrence Erlbaum Associates.
    Lakens, D. (2013). Calculating and reporting effect sizes to
    facilitate cumulative science. *Frontiers in Psychology*, 4, 863.
    """
    if df_between <= 0:
        raise ValueError("df_between must be a positive integer")
    if df_within < 0:
        raise ValueError("df_within must be a non-negative integer")
    if n <= df_between:
        raise ValueError(f"Total observations n ({n}) must exceed df_between ({df_between})")

    numerator = F_statistic * df_between - df_between
    denominator = F_statistic * df_between + df_within + 1.0

    if denominator == 0.0:
        return 0.0

    omega2 = numerator / denominator

    return float(np.clip(omega2, 0.0, 1.0))


def partial_eta_squared(F_statistic: float, df_between: int, df_error: int) -> float:
    """
    Compute partial eta-squared (η²_p) from ANOVA results.

    Partial η²_p represents the proportion of variance explained by a
    factor controlling for other factors, commonly used in ANCOVA.

    Formula (Cohen 1992):
        η²_p = (F * df_between) / (F * df_between + df_error)

    Parameters
    ----------
    F_statistic : float
        Observed F-statistic from ANOVA/ANCOVA.
    df_between : int
        Degrees of freedom for the effect (numerator df).
    df_error : int
        Degrees of freedom for the error (denominator df).

    Returns
    -------
    float
        Partial eta-squared (bounded [0, 1]).

    Raises
    ------
    ValueError
        If df_between <= 0 or df_error < 0.

    Notes
    -----
    This function is equivalent to `eta_squared` when there is only
    one factor (no covariates). The distinction is conceptual: partial
    η² is the preferred measure when there are multiple factors or
    covariates.

    References
    ----------
    Cohen, J. (1992). A power primer. *Psychological Bulletin*, 112(1), 155–159.
    Lakens, D. (2013). Calculating and reporting effect sizes to
    facilitate cumulative science. *Frontiers in Psychology*, 4, 863.
    """
    if df_between <= 0:
        raise ValueError("df_between must be a positive integer")
    if df_error < 0:
        raise ValueError("df_error must be a non-negative integer")

    numerator = F_statistic * df_between
    denominator = numerator + df_error

    if denominator == 0.0:
        return 0.0

    eta2p = numerator / denominator

    return float(np.clip(eta2p, 0.0, 1.0))
