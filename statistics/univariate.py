# =============================================================================
# FILE: statistics/univariate.py
# =============================================================================
"""
Univariate Statistics & Hypothesis Testing Module for PaleoAST

Provides descriptive statistics, normality tests, t-tests, ANOVA,
and non-parametric alternatives for paleontological data exploration.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass, field

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
                    f"{r['group_a']:<10} {r['group_b']:<10} {r['diff']:>10.4f} "
                    f"{r['p_adj']:>10.4f} {sig_mark:>5}"
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
                    results.append(ColumnStats(
                        name=column_names[idx], n=n, mean=np.nan, std=np.nan,
                        variance=np.nan, min_val=np.nan, max_val=np.nan,
                        median=np.nan, skewness=np.nan, kurtosis=np.nan,
                        se=np.nan, ci_95=(np.nan, np.nan),
                    ))
                    continue

                m = np.mean(valid)
                s = np.std(valid, ddof=1)
                se = s / np.sqrt(n)
                t_crit = sp_stats.t.ppf(0.975, df=n - 1)

                results.append(ColumnStats(
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
                ))

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
            ad_critical = dict(zip(ad_result.significance_level, ad_result.critical_values))

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

    def one_way_anova(
        self, data: npt.NDArray, groups: list[int], column: int = 0, tukey: bool = True
    ) -> ANOVAResult:
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
        all_vals = np.concatenate(group_data)
        n_total = len(all_vals)
        n_group_means = len(group_data)
        df_within = n_total - n_group_means

        ms_within = np.sum([(g - np.mean(g)) ** 2 for g in group_data]) / df_within

        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                ni = len(group_data[i])
                nj = len(group_data[j])
                mean_diff = np.mean(group_data[i]) - np.mean(group_data[j])

                se = np.sqrt(ms_within * (1 / ni + 1 / nj))
                if se == 0:
                    q_stat = 0.0
                else:
                    q_stat = abs(mean_diff) / se

                # Approximate p-value using studentized range distribution
                # Use scipy's tukey if available, else approximate
                try:
                    # Approximate: p from t-distribution adjusted for multiple comparisons
                    t_stat = q_stat / np.sqrt(2)
                    p_raw = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(df_within, 1)))
                    # Bonferroni-like correction for pairwise comparisons
                    n_comparisons = n_groups * (n_groups - 1) / 2
                    p_adj = min(p_raw * n_comparisons, 1.0)
                except Exception:
                    p_adj = 1.0

                results.append({
                    "group_a": group_labels[i],
                    "group_b": group_labels[j],
                    "diff": float(mean_diff),
                    "q_stat": float(q_stat),
                    "p_adj": float(p_adj),
                    "significant": p_adj < 0.05,
                })

        return results

    @property
    def last_result(self):
        return None
