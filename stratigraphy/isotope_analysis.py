# stratigraphy/isotope_analysis.py
"""
Isotope Time Series Analysis for PaleoAST

Provides tools for analyzing isotope (δ13C, δ18O, 87Sr/86Sr, εNd) time series
including trend extraction, excursion detection, spectral analysis, and
correlation analysis.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IsotopeData:
    """同位素时间序列数据"""

    depth: np.ndarray  # 深度序列
    age: np.ndarray  # 年龄序列
    d13C: np.ndarray | None = None  # δ13C 值
    d18O: np.ndarray | None = None  # δ18O 值
    sr: np.ndarray | None = None  # 锶同位素 87Sr/86Sr
    nd: np.ndarray | None = None  # 钕同位素 εNd
    other: dict | None = None  # 其他同位素数据

    def __post_init__(self):
        """验证长度一致性"""
        lengths = [len(self.depth), len(self.age)]
        if self.d13C is not None:
            lengths.append(len(self.d13C))
        if self.d18O is not None:
            lengths.append(len(self.d18O))
        if self.sr is not None:
            lengths.append(len(self.sr))
        if self.nd is not None:
            lengths.append(len(self.nd))

        if len(set(lengths)) > 1:
            raise ValueError("All isotope arrays must have same length as depth/age")

    def get_isotope_names(self) -> list[str]:
        """获取所有可用的同位素名称"""
        names = []
        if self.d13C is not None:
            names.append("d13C")
        if self.d18O is not None:
            names.append("d18O")
        if self.sr is not None:
            names.append("sr")
        if self.nd is not None:
            names.append("nd")
        if self.other:
            names.extend(self.other.keys())
        return names

    def get_isotope_array(self, name: str) -> np.ndarray | None:
        """按名称获取同位素数组"""
        if name == "d13C":
            return self.d13C
        elif name == "d18O":
            return self.d18O
        elif name == "sr":
            return self.sr
        elif name == "nd":
            return self.nd
        elif self.other and name in self.other:
            return self.other[name]
        return None


@dataclass
class Excursion:
    """Excursion 事件"""

    start_idx: int
    end_idx: int
    start_age: float
    end_age: float
    peak_idx: int
    peak_value: float
    magnitude: float  # 偏离背景的程度
    isotope: str = ""  # 同位素名称


@dataclass
class IsotopeTrend:
    """趋势分析结果"""

    method: str  # 'moving_average', 'polynomial', 'lowess'
    fitted_values: np.ndarray
    params: dict


@dataclass
class IsotopeResult:
    """同位素分析结果"""

    data: IsotopeData
    trends: dict = field(default_factory=dict)  # {isotope_name: IsotopeTrend}
    excursions: list[Excursion] = field(default_factory=list)  # 检测到的 excursion
    spectral_peaks: dict = field(default_factory=dict)  # {isotope_name: [(period, power), ...]}
    correlations: dict = field(default_factory=dict)  # {(name1, name2): (r, p)}

    def summary(self) -> str:
        lines = [
            "Isotope Time Series Analysis Results",
            "=" * 50,
            f"Data points: {len(self.data.depth)}",
            f"Age range: {self.data.age.min():.2f} - {self.data.age.max():.2f}",
            "",
            f"Excursions detected: {len(self.excursions)}",
        ]
        if self.spectral_peaks:
            lines.append(f"Spectral peaks: {sum(len(p) for p in self.spectral_peaks.values())}")
        if self.correlations:
            lines.append(f"Correlations computed: {len(self.correlations)}")
        return "\n".join(lines)


def compute_moving_average(values: np.ndarray, window: int = 5, mode: str = "center") -> np.ndarray:
    """
    计算移动平均

    参数:
        values: 输入数据
        window: 窗口大小 (必须是奇数)
        mode: 'center' 或 'right'

    返回:
        平滑后的数组
    """
    if window < 1:
        return values

    if window % 2 == 0:
        window += 1  # 必须奇数

    if mode == "center":
        # 中心移动平均
        smoothed = np.convolve(values, np.ones(window) / window, mode="same")
    else:
        # 右对齐
        smoothed = np.zeros_like(values)
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed[i] = np.mean(values[start : i + 1])

    return smoothed


def detect_excursions_from_values(
    values: np.ndarray,
    threshold: float = 2.0,
    min_duration: int = 2,
    background: str = "mean",
    age: np.ndarray | None = None,
) -> list[Excursion]:
    """
    检测 isotope excursion (异常偏移)

    参数:
        values: 同位素值序列
        threshold: 阈值 (标准差倍数)
        min_duration: 最小持续点数
        background: 背景估计方法 ('mean', 'median')
        age: 可选的年龄序列。如果提供，Excursion 的 ``start_age`` 和
             ``end_age`` 字段会使用 ``age[start_idx]`` / ``age[end_idx]``；
             否则回退为索引值 (与旧实现保持兼容)。

    返回:
        Excursion 列表
    """
    values = np.asarray(values)
    if age is not None:
        age = np.asarray(age)
        if len(age) != len(values):
            raise ValueError(f"age length ({len(age)}) must match values length ({len(values)})")

    # 估计背景和标准差
    if background == "mean":
        bg = np.mean(values)
        std = np.std(values)
    else:
        bg = np.median(values)
        # MAD (Median Absolute Deviation) - scale by 1.4826 for normal distribution
        mad = np.median(np.abs(values - bg))
        std = mad * 1.4826

    if std == 0:
        std = 1.0

    # 计算 z-score
    z_scores = np.abs(values - bg) / std

    # 标记 excursion 点
    is_excursion = z_scores > threshold

    # 找连续的 excursion 区域
    excursions = []
    in_excursion = False
    start_idx = 0

    def _age_at(idx: int) -> float:
        """Map an array index to its age, falling back to the index itself."""
        return float(age[idx]) if age is not None else float(idx)

    for i, exc in enumerate(is_excursion):
        if exc and not in_excursion:
            in_excursion = True
            start_idx = i
        elif not exc and in_excursion:
            in_excursion = False
            end_idx = i - 1
            duration = end_idx - start_idx + 1

            if duration >= min_duration:
                peak_idx = start_idx + np.argmax(np.abs(values[start_idx : end_idx + 1] - bg))
                peak_value = values[peak_idx]

                excursions.append(
                    Excursion(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        start_age=_age_at(start_idx),
                        end_age=_age_at(end_idx),
                        peak_idx=peak_idx,
                        peak_value=peak_value,
                        magnitude=(peak_value - bg) / std,
                    )
                )

    # 处理结尾的 excursion
    if in_excursion:
        end_idx = len(values) - 1
        duration = end_idx - start_idx + 1

        if duration >= min_duration:
            peak_idx = start_idx + np.argmax(np.abs(values[start_idx : end_idx + 1] - bg))
            peak_value = values[peak_idx]

            excursions.append(
                Excursion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_age=_age_at(start_idx),
                    end_age=_age_at(end_idx),
                    peak_idx=peak_idx,
                    peak_value=peak_value,
                    magnitude=(peak_value - bg) / std,
                )
            )

    return excursions


def compute_correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> tuple[float, float]:
    """
    计算两组数据的相关系数

    参数:
        x, y: 数据数组
        method: 'pearson' 或 'spearman'

    返回:
        (r, p_value)
    """
    from scipy import stats

    x = np.asarray(x)
    y = np.asarray(y)

    # 移除 NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)

    return r, p


def fit_polynomial_trend(age: np.ndarray, values: np.ndarray, degree: int = 2) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    多项式趋势拟合

    参数:
        age: 年龄序列
        values: 同位素值
        degree: 多项式阶数

    返回:
        (fitted_values, coeffs, stats)
    """
    from scipy import stats

    # 去除 NaN
    mask = ~np.isnan(values)
    age_valid = age[mask]
    values_valid = values[mask]

    # 多项式拟合
    coeffs = np.polyfit(age_valid, values_valid, degree)

    # 计算拟合值
    fitted = np.polyval(coeffs, age)

    # R² 和 p-value
    ss_res = np.sum((values_valid - np.polyval(coeffs, age_valid)) ** 2)
    ss_tot = np.sum((values_valid - np.mean(values_valid)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # F-test for overall significance
    n = len(values_valid)
    k = degree + 1
    if n > k and ss_res > 0:
        f_stat = (ss_tot - ss_res) / (k - 1) / (ss_res / (n - k))
        p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
    else:
        f_stat = 0.0
        p_value = 1.0

    stats_dict = {"r2": r2, "f_stat": f_stat, "p_value": p_value, "coefficients": coeffs}

    return fitted, coeffs, stats_dict


def lowess_smooth(age: np.ndarray, values: np.ndarray, span: float = 0.3) -> np.ndarray:
    """
    LOWESS (Locally Weighted Scatterplot Smoothing)

    参数:
        age: 年龄序列
        values: 同位素值
        span: 平滑窗口比例 (0-1)

    返回:
        平滑后的值
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as stats_lowess

    # 去除 NaN
    mask = ~np.isnan(values) & ~np.isnan(age)
    age_valid = age[mask]
    values_valid = values[mask]

    if len(age_valid) < 3:
        return values_valid

    # LOWESS 拟合 - returns array with [age, smoothed_value] sorted by age
    result = stats_lowess(values_valid, age_valid, frac=span, return_sorted=True)

    # Interpolate back to original age points
    smoothed = np.interp(age, result[:, 0], result[:, 1])

    return smoothed


def remove_outliers(values: np.ndarray, method: str = "iqr", threshold: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """
    移除异常值

    参数:
        values: 输入数据
        method: 'iqr' (四分位距) 或 'zscore'
        threshold: 阈值

    返回:
        (cleaned_values, mask) mask=True 表示保留
    """
    values = np.asarray(values)
    mask = np.ones(len(values), dtype=bool)

    if method == "iqr":
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (values >= lower) & (values <= upper)

    elif method == "zscore":
        z = np.abs((values - np.mean(values)) / np.std(values))
        mask = z < threshold

    return values[mask], mask


class IsotopeAnalyzer:
    """同位素时间序列分析器"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.IsotopeAnalyzer")
        self._last_result: IsotopeResult | None = None

    def analyze(
        self,
        data: IsotopeData,
        detect_excursions: bool = True,
        excursion_threshold: float = 2.0,
        excursion_min_duration: int = 2,
        compute_correlations: bool = True,
    ) -> IsotopeResult:
        """
        执行同位素时间序列分析

        参数:
            data: IsotopeData 对象
            detect_excursions: 是否检测 excursion
            excursion_threshold: excursion 检测阈值
            excursion_min_duration: 最小持续时间
            compute_correlations: 是否计算相关性

        返回:
            IsotopeResult
        """
        self._logger.info(f"Starting isotope analysis: {len(data.depth)} points")

        excursions = []
        correlations = {}

        # 检测每个同位素的 excursion
        if detect_excursions:
            isotope_names = data.get_isotope_names()
            for name in isotope_names:
                values = data.get_isotope_array(name)
                if values is not None:
                    excs = detect_excursions_from_values(
                        values,
                        threshold=excursion_threshold,
                        min_duration=excursion_min_duration,
                        age=data.age,
                    )
                    for e in excs:
                        e.isotope = name
                    excursions.extend(excs)

        # 计算相关性
        if compute_correlations:
            isotope_names = data.get_isotope_names()
            for i, name1 in enumerate(isotope_names):
                for name2 in isotope_names[i + 1 :]:
                    arr1 = data.get_isotope_array(name1)
                    arr2 = data.get_isotope_array(name2)
                    if arr1 is not None and arr2 is not None:
                        r, p = compute_correlation(arr1, arr2)
                        correlations[(name1, name2)] = (r, p)

        result = IsotopeResult(data=data, excursions=excursions, correlations=correlations)

        self._last_result = result
        self._logger.info(f"Isotope analysis complete: {len(excursions)} excursions, {len(correlations)} correlations")

        return result

    def last_result(self) -> IsotopeResult | None:
        """获取上次分析结果"""
        return self._last_result
