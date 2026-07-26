# stratigraphy/correlation.py
"""
Stratigraphic Correlation and Age Modeling for PaleoAST

Provides tools for:
- Correlating multiple stratigraphic sections using DTW
- Building geological age models from biostratigraphic constraints
- Computing sedimentation rates

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StratigraphicSection:
    """单条地层剖面数据"""

    name: str
    heights: np.ndarray  # 层位高度 (m)
    thicknesses: np.ndarray  # 层厚 (m)
    lithologies: list[str]  # 岩性描述
    ages: np.ndarray | None = None  # 绝对年龄 (如果有)
    age_errors: np.ndarray | None = None  # 年龄误差
    notes: list[str] | None = None  # 备注

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "heights": self.heights.tolist(),
            "thicknesses": self.thicknesses.tolist(),
            "lithologies": self.lithologies,
            "ages": self.ages.tolist() if self.ages is not None else None,
            "age_errors": self.age_errors.tolist() if self.age_errors is not None else None,
        }


@dataclass
class StratigraphicCorrelationResult:
    """相关性分析结果"""

    sections: list[StratigraphicSection]
    correlation_matrix: np.ndarray
    best_matches: list[tuple[int, int, float]]

    def summary(self) -> str:
        n = len(self.sections)
        triu_indices = np.triu_indices(n, k=1)
        if len(triu_indices[0]) > 0:
            avg_corr = np.mean(self.correlation_matrix[triu_indices])
        else:
            avg_corr = 1.0
        lines = [
            f"地层相关性分析结果 ({n} 个剖面)",
            f"平均相关性: {avg_corr:.3f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "correlation_matrix": self.correlation_matrix.tolist(),
        }


@dataclass
class AgeModelResult:
    """年龄模型结果"""

    section: StratigraphicSection
    modeled_ages: np.ndarray
    confidence_intervals: tuple[np.ndarray, np.ndarray]
    sedimentation_rates: np.ndarray

    def summary(self) -> str:
        lines = [
            f"年龄模型: {self.section.name}",
            f"年龄范围: {self.modeled_ages.min():.2f} - {self.modeled_ages.max():.2f} Ma",
            f"平均沉积速率: {np.mean(self.sedimentation_rates):.2f} m/Myr",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "section": self.section.to_dict(),
            "modeled_ages": self.modeled_ages.tolist(),
            "ci_lower": self.confidence_intervals[0].tolist(),
            "ci_upper": self.confidence_intervals[1].tolist(),
            "sedimentation_rates": self.sedimentation_rates.tolist(),
        }


class StratigraphicCorrelationAnalyzer:
    """地层相关性分析器 (使用动态时间规整 DTW)"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.StratigraphicCorrelationAnalyzer")

    def analyze(
        self, sections: list[StratigraphicSection], method: str = "dtw", max_matches: int = 5
    ) -> StratigraphicCorrelationResult:
        """
        分析多个地层剖面的相关性

        Parameters:
            sections: 地层剖面列表
            method: 'dtw' (动态时间规整) 或 'euclidean' (欧氏距离)
            max_matches: 最大返回的匹配对数量

        Returns:
            StratigraphicCorrelationResult
        """
        n = len(sections)
        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = self._compute_correlation(sections[i], sections[j], method)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        best_matches = self._find_best_matches(corr_matrix, sections, max_matches)

        self._logger.info(f"Correlation analysis complete: {n} sections, avg corr={np.mean(corr_matrix):.3f}")

        return StratigraphicCorrelationResult(
            sections=sections, correlation_matrix=corr_matrix, best_matches=best_matches
        )

    def _compute_correlation(self, sec_a: StratigraphicSection, sec_b: StratigraphicSection, method: str) -> float:
        if method == "dtw":
            return self._dtw_correlation(sec_a, sec_b)
        elif method == "euclidean":
            return self._euclidean_correlation(sec_a, sec_b)
        else:
            return self._dtw_correlation(sec_a, sec_b)

    def _dtw_correlation(self, sec_a: StratigraphicSection, sec_b: StratigraphicSection) -> float:
        """动态时间规整相关性"""
        h_a = sec_a.heights
        h_b = sec_b.heights

        n, m = len(h_a), len(h_b)
        if n + m == 0:
            return 1.0

        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(h_a[i - 1] - h_b[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        similarity = 1 / (1 + dtw[n, m] / (n + m))
        return similarity

    def _euclidean_correlation(self, sec_a: StratigraphicSection, sec_b: StratigraphicSection) -> float:
        """欧氏距离相关性"""
        h_a = sec_a.heights
        h_b = sec_b.heights

        min_len = min(len(h_a), len(h_b))
        if min_len == 0:
            return 0.0
        diff = np.sum((h_a[:min_len] - h_b[:min_len]) ** 2)
        similarity = 1 / (1 + np.sqrt(diff) / min_len)
        return similarity

    def _find_best_matches(
        self, corr_matrix: np.ndarray, sections: list[StratigraphicSection], max_matches: int = 5
    ) -> list[tuple[int, int, float]]:
        """找相关性最高的剖面对"""
        matches = []
        n = len(sections)
        for i in range(n):
            for j in range(i + 1, n):
                matches.append((i, j, corr_matrix[i, j]))
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:max_matches]


class AgeModelAnalyzer:
    """地质年龄模型构建器"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.AgeModelAnalyzer")

    def build_model(
        self,
        section: StratigraphicSection,
        age_constraints: list[tuple[float, float, float]],
        model_type: str = "linear",
    ) -> AgeModelResult:
        """
        构建地层年龄模型

        Parameters:
            section: 地层剖面
            age_constraints: 年龄约束点 [(height, age, error), ...]
            model_type: 'linear' (线性插值) 或 'spline' (样条插值)

        Returns:
            AgeModelResult
        """
        from scipy.interpolate import interp1d

        constraint_heights = np.array([c[0] for c in age_constraints])
        constraint_ages = np.array([c[1] for c in age_constraints])
        constraint_errors = np.array([c[2] for c in age_constraints])

        kind = "linear" if model_type == "linear" else "cubic"
        age_func = interp1d(constraint_heights, constraint_ages, kind=kind, fill_value="extrapolate")

        modeled_ages = age_func(section.heights)

        rates = np.zeros(len(section.heights))
        if len(section.heights) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                # Note: In geological convention, age decreases with height
                # so dHeight/dAge < 0, hence the negation to get positive rates
                rates = np.diff(section.heights) / np.diff(modeled_ages)
                rates = -rates  # Make positive (sedimentation rate)
                rates = np.concatenate([[rates[0]], rates])

        ci_lower = modeled_ages - 1.96 * constraint_errors.mean()
        ci_upper = modeled_ages + 1.96 * constraint_errors.mean()

        self._logger.info(
            f"Age model built: {len(section.heights)} points, "
            f"range {modeled_ages.min():.2f}-{modeled_ages.max():.2f} Ma"
        )

        return AgeModelResult(
            section=section,
            modeled_ages=modeled_ages,
            confidence_intervals=(ci_lower, ci_upper),
            sedimentation_rates=rates,
        )


def pyper_peterman_correction(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05, max_lag: int | None = None
) -> tuple[float, float, float, int]:
    """
    Pyper & Peterman 1998 有效自由度修正的 Pearson 相关检验。

    地层和沉积记录天然具有时间自相关（成岩作用、沉积速率变化等），
    传统 Pearson 相关假设独立样本，会显著高估显著性。
    本函数使用 Pyper & Peterman (1998) 的方法估计有效样本量并修正 p 值。

    Parameters
    ----------
    x, y : array-like
        两个等长的时间序列。
    alpha : float, default 0.05
        显著性水平（未使用，保留 API 兼容）。
    max_lag : int, optional
        最大滞后阶数。默认为 n//2。

    Returns
    -------
    r : float
        Pearson 相关系数。
    p_corrected : float
        基于有效样本量的修正 p 值。
    n_eff : int
        有效样本量。
    n_original : int
        原始样本量（移除 NaN 后）。

    Notes
    -----
    有效样本量估计（Pyper & Peterman 1998, Canadian Journal of Fisheries
    and Aquatic Sciences）::

        n_eff = n * (1 - sum_{k=1}^{m} rho_x(k) * rho_y(k))
                / (1 + sum_{k=1}^{m} rho_x(k) * rho_y(k))

    其中 m = max_lag，rho_x(k) 和 rho_y(k) 分别是 x 和 y 的
    自相关函数（ACF）在滞后 k 处的值。

    对于独立序列（白噪声），rho_x(k) = rho_y(k) = 0 (k>0)，
    因此 n_eff = n。对于高度自相关的序列，n_eff << n。

    该方法也被 R 包 ``modified.ttest`` 实现。

    References
    ----------
    Pyper, C.J. & Peterman, R.M. (1998). "Reducing bias in estimates of
    environmental change." Can. J. Fish. Aquat. Sci., 55: 2128-2143.
    """
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 移除 NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 4:
        return np.nan, np.nan, n, n

    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)

    # 计算自相关函数 ACF
    def _acf(arr: np.ndarray, lag: int) -> float:
        if lag == 0:
            return 1.0
        n_lag = len(arr) - lag
        if n_lag <= 0:
            return 0.0
        return np.sum(arr[:-lag] * arr[lag:]) / np.sum(arr * arr)

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    # 计算 sum(rho_x(k) * rho_y(k)) for k=1..max_lag
    sum_rho_product = 0.0
    for k in range(1, max_lag + 1):
        rho_x = _acf(x_centered, k)
        rho_y = _acf(y_centered, k)
        sum_rho_product += rho_x * rho_y

    # Pyper-Peterman effective sample size
    if 1 + sum_rho_product <= 0:
        n_eff = 2
    else:
        n_eff = n * (1 - sum_rho_product) / (1 + sum_rho_product)
        n_eff = max(2.0, min(n_eff, n))

    n_eff_int = int(round(n_eff))

    # Pearson r
    r, _ = stats.pearsonr(x, y)

    # 修正自由度
    df = n_eff_int - 2
    if df < 1:
        df = 1

    # t 统计量 -> p 值
    if abs(r) >= 1.0:
        p_corrected = 0.0
    else:
        t_stat = r * np.sqrt(df / (1 - r**2))
        p_corrected = 2.0 * stats.t.sf(abs(t_stat), df)

    return float(r), float(p_corrected), n_eff_int, int(n)


class SedimentationRateAnalyzer:
    """沉积速率分析器"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.SedimentationRateAnalyzer")

    def calculate(
        self, section: StratigraphicSection, smooth: bool = True, frac: float = 0.3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算沉积速率

        Parameters:
            section: 地层剖面 (需要包含 ages)
            smooth: 是否使用 LOWESS 平滑
            frac: 平滑参数

        Returns:
            (heights, rates, smoothed_rates)
        """
        heights = section.heights
        ages = section.ages

        if ages is None:
            raise ValueError("需要年龄数据来计算沉积速率")

        rates = np.zeros(len(heights))
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.gradient(heights, ages)
            rates = -rates  # 沉积为正

        smoothed_rates = rates.copy()
        if smooth and len(heights) > 3:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess as stats_lowess

                sorted_idx = np.argsort(heights)
                h_sorted = heights[sorted_idx]
                r_sorted = rates[sorted_idx]
                # statsmodels lowess returns 2D array [x_sorted, y_smoothed]
                result = stats_lowess(r_sorted, h_sorted, frac=frac, return_sorted=True)
                # Interpolate back to original height order
                smoothed_rates = np.interp(heights, result[:, 0], result[:, 1])
            except Exception as e:
                self._logger.warning(f"LOWESS smoothing failed: {e}")

        self._logger.info(f"Sedimentation rates computed: {len(heights)} points, mean={np.mean(rates):.2f} m/Myr")

        return heights, rates, smoothed_rates
