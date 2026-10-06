# stratigraphy/isotope_analysis.py
"""
Isotope Time Series Analysis for PaleoAST

Provides tools for analyzing isotope (δ13C, δ18O, 87Sr/86Sr, εNd) time series
including trend extraction, excursion detection, spectral analysis, and
correlation analysis.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
from collections.abc import Callable
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
    metadata: dict = field(default_factory=dict)  # auxiliary results (e.g. paleotemperature)

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


# =============================================================================
# δ¹⁸O scale conversion (VPDB ↔ VSMOW)
# =============================================================================
#
# Paleotemperature equations in the literature are calibrated on different
# reference frames:
#
#   VSMOW (Vienna Standard Mean Ocean Water) — the modern oceanographic
#       reference; open-ocean surface water is by definition 0 ‰ VSMOW.
#   VPDB (Vienna Pee Dee Belemnite) — the reference most paleoclimate
#       laboratories actually report carbonate analyses on (NBS-19 /
#       IAEA-603 are tied to it). Typical planktonic foraminifera run
#       δc ≈ -2 ‰ VPDB.
#
# Two conversion conventions exist:
#
# 1. The linear Coplen (1988) formula, ratified by IUPAC:
#        δ_VSMOW = 1.03091 × δ_VPDB + 30.91
#        δ_VPDB  = (δ_VSMOW - 30.91) / 1.03091
#    This is the modern (post-1988) conversion and is what
#    Kim & O'Neil (1997) used.
#
# 2. The historical "Coplen-Tyler" simple offset (still in widespread
#    use in paleoceanography):
#        δ_VPDB  ≈ δ_VSMOW - 0.27 ‰
#    This pre-dates Coplen (1988) and is what Erez & Luz (1983) and
#    Bemis et al. (1998) used when fitting their equations. The 0.27 ‰
#    offset is also what these equations implicitly assume when the
#    user supplies δw on the modern oceanographic VSMOW scale.
#
# WHY TWO CONVERSIONS
# -------------------
# The two conversions differ by an order of magnitude in their numerical
# effect. With the linear Coplen 1988 formula, SMOW (δ_VSMOW = 0) maps
# to δ_VPDB ≈ -29.99 ‰ — which is the right way to convert a *measured*
# δ¹⁸O of seawater that has been equilibrated with CO2 and run on a
# mass spec calibrated to VPDB. But it is NOT what Erez & Luz and
# Bemis et al. did when they fitted their calibrations: their published
# polynomial coefficients are only consistent with the small-offset
# 0.27 ‰ convention. Using the linear formula with their coefficients
# gives delta_diff ≈ +30 ‰ (instead of ~0 ‰) and temperature answers
# 100+ °C off.
#
# Both conventions are exposed in the API below. The paleotemperature
# functions accept VPDB for δc (laboratory convention) and either VPDB
# or VSMOW for δw (controlled by a parameter), and the conversion they
# apply is documented in each function's docstring.
#
# References
# ----------
# Coplen, T. B. (1988). "Normalization of oxygen and hydrogen isotope
#     data." Chemical Geology (Isotope Geoscience Section), 72, 293-297.
# Coplen, T. B. et al. (1983). "Improvements in the gaseous hydrogen-water
#     equilibration technique for hydrogen isotope analysis." Abstracts
#     with Programs, Geological Society of America, 15, 549.
# Friedman, I. & O'Neil, J. R. (1977). "Compilation of stable isotope
#     fractionation factors of geochemical interest." USGS Prof. Paper
#     440-KK.
VP_VPDB_TO_VSMOW_SLOPE = 1.03091
VP_VPDB_TO_VSMOW_INTERCEPT = 30.91

# Historical offset used by Erez & Luz (1983) and Bemis et al. (1998)
# when fitting their paleotemperature calibrations. This is the
# "pre-Coplen-1988" conversion that was standard practice in early-80s
# paleoceanography and is still used by modern implementations of E&L-
# and Bemis-style equations (e.g. Pearson 2012).
EL_BEMIS_VSMOW_TO_VPDB_OFFSET = 0.27


def vpdb_to_vsmow(delta_vpdb: float | np.ndarray) -> float | np.ndarray:
    """Convert δ¹⁸O from VPDB to VSMOW using the linear Coplen (1988)
    formula:

        δ_VSMOW = 1.03091 × δ_VPDB + 30.91

    This is the appropriate conversion for *measured* carbonate (CO2
    equilibrated at 25 °C) when the working standard is tied to VPDB
    but the user wants the VSMOW value to feed e.g. Kim & O'Neil (1997).

    Reference: Coplen (1988), ratified by IUPAC.
    """
    return VP_VPDB_TO_VSMOW_SLOPE * np.asarray(delta_vpdb, dtype=float) + VP_VPDB_TO_VSMOW_INTERCEPT


#: Genus names for which the Bemis et al. (1998) δw correction is a plain
#: "no vital effect" calibration rather than a warning-worthy miss.
_GENERIC_GENUS_NAMES = frozenset({"generic", "default", "none", "unknown", ""})


def _genus_offset(genus: str, corrections: dict[str, float]) -> tuple[float, bool]:
    """Look up a genus-specific delta-w offset, tolerating how it is spelled.

    Accepts the abbreviated form the table is written in ("G. ruber"), the full
    binomial a palaeontologist would actually type ("Globigerinoides ruber"), a
    bare epithet ("ruber"), and any capitalisation or surrounding whitespace.

    Returns ``(offset, matched)``. The flag is False when nothing matched and
    the generic calibration is being used, so the caller can say so rather than
    quietly returning a different temperature.

    Matching is on the species epithet -- the last whitespace-separated token --
    because that is the part that identifies the species, and "G." and
    "Globigerinoides" are the same genus spelled two ways.
    """
    if not isinstance(genus, str):
        return 0.0, False

    cleaned = " ".join(genus.split()).strip()
    if cleaned.lower() in _GENERIC_GENUS_NAMES:
        return 0.0, True

    parts = cleaned.replace(",", " ").split()
    epithet = parts[-1].lower() if parts else ""

    # exact table key first, so an explicitly spelled abbreviation still wins
    if cleaned in corrections:
        return float(corrections[cleaned]), True
    for key, value in corrections.items():
        if " ".join(key.split()).lower() == cleaned.lower():
            return float(value), True
    if epithet in corrections:
        return float(corrections[epithet]), True
    return 0.0, False


def vsmow_to_vpdb(delta_vsmow: float | np.ndarray, *, formula: str = "coplen") -> float | np.ndarray:
    """Convert δ¹⁸O from VSMOW to VPDB.

    With ``formula='coplen'`` (default — modern, post-1988):
        δ_VPDB = (δ_VSMOW - 30.91) / 1.03091

    With ``formula='el_bemis'`` (pre-1988 historical offset used by
    Erez & Luz (1983) and Bemis et al. (1998) in their published
    calibrations):
        δ_VPDB ≈ δ_VSMOW - 0.27 ‰

    The two formulae differ by ~30 ‰ in magnitude: use the right one
    for the equation you are feeding. Kim & O'Neil (1997) uses
    ``formula='coplen'``; Erez & Luz and Bemis use ``formula='el_bemis'``.

    References
    ----------
    Coplen, T. B. (1988), as above.
    Erez, J. & Luz, B. (1983), as below.
    Bemis, B. E. et al. (1998), as below.
    """
    arr = np.asarray(delta_vsmow, dtype=float)
    if formula == "coplen":
        return (arr - VP_VPDB_TO_VSMOW_INTERCEPT) / VP_VPDB_TO_VSMOW_SLOPE
    if formula == "el_bemis":
        return arr - EL_BEMIS_VSMOW_TO_VPDB_OFFSET
    raise ValueError(f"formula must be 'coplen' or 'el_bemis', got '{formula}'")


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
        # 中心移动平均: 边缘按实际重叠窗口归一化。
        # 旧实现 np.convolve(mode="same") 在两端仍除以完整窗口,
        # 导致序列两端被系统性压低。
        smoothed = np.convolve(values, np.ones(window), mode="same")
        counts = np.convolve(np.ones_like(values), np.ones(window), mode="same")
        smoothed = smoothed / counts
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
        compute_paleotemperature: bool = False,
        delta18O_sw_vsmow: float = 0.0,
        genus: str = "generic",
        equation: str = "erez_luz",
    ) -> IsotopeResult:
        """
        执行同位素时间序列分析

        参数:
            data: IsotopeData 对象
            detect_excursions: 是否检测 excursion
            excursion_threshold: excursion 检测阈值
            excursion_min_duration: 最小持续时间
            compute_correlations: 是否计算相关性
            compute_paleotemperature: 是否计算 δ¹⁸O 古温度
            delta18O_sw_vsmow: 海水 δ¹⁸O (‰ VSMOW, 现代海洋学惯例),
                仅当 ``compute_paleotemperature=True`` 时使用
            genus: Bemis 等属种修正 (仅当 ``equation='bemis'`` 时生效)
            equation: 古温度方程, ``'erez_luz'`` 或 ``'bemis'``

        返回:
            IsotopeResult; 当 ``compute_paleotemperature=True`` 时,
            ``result.metadata['paleotemperature']`` 包含
            ``temperatures_c``、``equation``、``delta18O_sw_vsmow``、
            ``genus``、``valid_count``、``out_of_range_count`` 字段
        """
        self._logger.info(f"Starting isotope analysis: {len(data.depth)} points")

        excursions = []
        correlations = {}
        metadata: dict = {}

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

        # 计算古温度 (缺陷 1 修复: δsw 一律按 VSMOW 接收, 函数内部换算到 VPDB)
        if compute_paleotemperature:
            if data.d18O is None:
                self._logger.warning("compute_paleotemperature=True but data.d18O is None — skipped")
            else:
                temps = np.full(len(data.d18O), np.nan, dtype=float)
                valid = np.zeros(len(data.d18O), dtype=bool)
                out_of_range = 0
                for i, dc in enumerate(data.d18O):
                    if dc is None or np.isnan(dc):
                        continue
                    if equation == "erez_luz":
                        try:
                            with np.errstate(invalid="ignore"):
                                t_val = float(
                                    IsotopeAnalyzer.compute_paleotemperature_erez_luz(
                                        delta18O_sw=delta18O_sw_vsmow,
                                        delta18O_c=float(dc),
                                        delta18O_sw_scale="vsmow",
                                    )
                                )
                        except ValueError:
                            continue
                    elif equation == "bemis":
                        try:
                            with np.errstate(invalid="ignore"):
                                t_val = float(
                                    IsotopeAnalyzer.compute_paleotemperature_bemis(
                                        delta18O_c=float(dc),
                                        delta18O_sw=delta18O_sw_vsmow,
                                        genus=genus,
                                        delta18O_sw_scale="vsmow",
                                    )
                                )
                        except ValueError:
                            continue
                    else:
                        raise ValueError(f"Unknown paleotemperature equation: {equation}")
                    if np.isfinite(t_val) and 0.0 <= t_val <= 35.0:
                        valid[i] = True
                    else:
                        out_of_range += 1
                    temps[i] = t_val
                metadata["paleotemperature"] = {
                    "temperatures_c": temps,
                    "equation": equation,
                    "delta18O_sw_vsmow": float(delta18O_sw_vsmow),
                    "genus": genus,
                    "valid_count": int(valid.sum()),
                    "out_of_range_count": int(out_of_range),
                }

        result = IsotopeResult(
            data=data,
            excursions=excursions,
            correlations=correlations,
            metadata=metadata,
        )

        self._last_result = result
        self._logger.info(f"Isotope analysis complete: {len(excursions)} excursions, {len(correlations)} correlations")

        return result

    def last_result(self) -> IsotopeResult | None:
        """获取上次分析结果"""
        return self._last_result

    @staticmethod
    def compute_paleotemperature_erez_luz(
        delta18O_sw: float,
        delta18O_c: float,
        delta18O_sw_scale: str = "vsmow",
    ) -> float:
        """
        Erez & Luz (1983) paleotemperature equation.

        Equation (Eq. 5 of the paper, restated with the canonical a = 17.0,
        b = -4.52, c = +0.03 coefficients used in subsequent literature):

            T(°C) = 17.0 - 4.52 × (δc - δw) + 0.03 × (δc - δw)²

        Valid range: 16-25 °C (calibrated on cultured planktonic foraminifera,
        Globigerinoides sacculifer). See the References section below.

        SCALE CONVENTION — READ BEFORE CALLING
        -------------------------------------
        Erez & Luz's calibration tabulated BOTH δc and δw on a VPDB-like
        scale, with seawater values offset from VSMOW by 0.27 ‰ (the
        pre-Coplen-1988 conversion). Their published coefficients only
        make sense with that small-offset convention — feeding in
        δw_VPDB = -29.99 ‰ (Coplen 1988 linear conversion of SMOW)
        instead of -0.27 ‰ shifts the polynomial by ~30 ‰ and gives
        temperatures ~100 °C off.

        This function accepts δc on VPDB (laboratory convention) and δw on
        either scale, controlled by ``delta18O_sw_scale``:

          * ``"vsmow"`` (default, modern oceanographic convention): δw is
            converted to the small-offset VPDB scale (SMOW → -0.27 ‰)
            used by E&L via :func:`vsmow_to_vpdb` with ``formula='el_bemis'``.
          * ``"vpdb"`` (raw VPDB): δw is treated as-is — no conversion.

        Parameters
        ----------
        delta18O_sw : float
            Seawater δ¹⁸O in ‰ (on the scale declared by
            ``delta18O_sw_scale``).
        delta18O_c : float
            Carbonate δ¹⁸O in ‰ VPDB.
        delta18O_sw_scale : {"vpdb", "vsmow"}
            Reference frame of ``delta18O_sw``.

        Returns
        -------
        float
            Paleotemperature in °C.

        Notes
        -----
        Outside the calibrated 16-25 °C window the polynomial returns
        physically unreasonable values. The function emits a
        ``UserWarning`` when the output falls outside [0, 35] °C —
        callers should not treat the number as a calibrated estimate in
        that case.

        References
        ----------
        Erez, J. & Luz, B. (1983). "Experimental paleotemperature equation
            for planktonic foraminifera." Geochim. Cosmochim. Acta 47(11),
            2115-2128. (Original calibration on cultured G. sacculifer; the
            VPDB-like small-offset convention for δw described above.)
        Coplen, T. B. (1988). "Normalization of oxygen and hydrogen
            isotope data." Chem. Geol. 72, 293-297.
        """
        # Scale conversion: E&L used the pre-1988 small-offset convention
        # (Coplen-Tyler), NOT the linear Coplen (1988) formula.
        if delta18O_sw_scale == "vsmow":
            delta18O_sw_vpdb = float(vsmow_to_vpdb(delta18O_sw, formula="el_bemis"))
        elif delta18O_sw_scale == "vpdb":
            delta18O_sw_vpdb = float(delta18O_sw)
        else:
            raise ValueError(f"delta18O_sw_scale must be 'vpdb' or 'vsmow', got '{delta18O_sw_scale}'")

        delta_diff = delta18O_c - delta18O_sw_vpdb
        T = 17.0 - 4.52 * delta_diff + 0.03 * (delta_diff**2)

        if not (0.0 <= T <= 35.0):
            import warnings as _warnings

            _warnings.warn(
                f"Erez & Luz (1983) returns T = {T:.2f} °C, which is outside "
                f"the calibrated 16-25 °C window — the input (δc - δw) = "
                f"{delta_diff:.2f} ‰ is unphysical or unconvertible. Do "
                "not interpret the result as a calibrated paleotemperature.",
                UserWarning,
                stacklevel=2,
            )
        return float(T)

    @staticmethod
    def compute_paleotemperature_bemis(
        delta18O_c: float,
        delta18O_sw: float = 0.0,
        genus: str = "generic",
        delta18O_sw_scale: str = "vsmow",
    ) -> float:
        """
        Bemis et al. (1998) paleotemperature equation with genus-specific
        calibrations.

        Equation (Erez & Luz form, refit on a larger culture set, with
        genus-specific intercepts that incorporate both vital-effect
        offsets and a δw correction):

            T(°C) = 16.998 - 4.52 × (δc - δw_eff)

        where δw_eff = δw_vpdb + Δ_genus. The published Δ_genus values
        (Bemis et al. 1998, Table 2) were fitted on the small-offset
        VPDB scale used by Erez & Luz (δw_VPDB ≈ δw_VSMOW - 0.27 ‰, NOT
        the Coplen 1988 linear conversion).

        SCALE CONVENTION
        ----------------
        Same as :meth:`compute_paleotemperature_erez_luz`. Default scale
        is ``"vsmow"`` (modern oceanographic convention). The function
        converts internally using the historical 0.27 ‰ offset. Pass
        ``delta18O_sw_scale="vpdb"`` if you have already converted.

        Parameters
        ----------
        delta18O_c : float
            Carbonate δ¹⁸O in ‰ VPDB.
        delta18O_sw : float, default 0.0
            Seawater δ¹⁸O (default: SMOW, the historical baseline).
        genus : {"G. ruber", "G. sacculifer", "generic"}
            Genus-specific calibration. ``"generic"`` applies no extra
            offset (uses the Erez & Luz coefficients as published).
        delta18O_sw_scale : {"vpdb", "vsmow"}
            Reference frame of ``delta18O_sw``.

        Returns
        -------
        float
            Paleotemperature in °C.

        References
        ----------
        Bemis, B. E., Spero, H. J., Bijma, J. & Lea, D. W. (1998).
            "Reevaluation of the oxygen isotopic composition of planktonic
            foraminifera: Experimental results and revised paleotemperature
            equations." Paleoceanography 13(2), 150-160.
        """
        # Genus-specific δw correction (Bemis et al. 1998, Table 2) — these
        # were fitted on the small-offset VPDB scale, so the conversion
        # MUST use the el_bemis formula (Coplen-Tyler offset 0.27 ‰).
        #
        # Keyed on the species epithet, and looked up through `_genus_offset`,
        # which accepts the abbreviated ("G. ruber"), the full binomial
        # ("Globigerinoides ruber") and the bare epithet, in any case. Looking
        # these up literally meant that every name a user would realistically
        # type fell through to `generic` -- silently, with no warning -- and
        # silently dropping the offset shifts the temperature by 1.22 °C for
        # G. ruber.
        genus_corrections = {
            "ruber": 0.27,
            "sacculifer": 0.22,
        }
        delta_genus, matched = _genus_offset(genus, genus_corrections)
        if not matched:
            import warnings as _warnings

            _warnings.warn(
                f"No Bemis et al. (1998) offset is tabulated for genus "
                f"{genus!r}; using the generic calibration "
                f"({sorted(genus_corrections)} are known). Pass "
                f"genus='generic' to select it deliberately and silence this.",
                UserWarning,
                stacklevel=2,
            )

        if delta18O_sw_scale == "vsmow":
            delta18O_sw_vpdb = float(vsmow_to_vpdb(delta18O_sw, formula="el_bemis"))
        elif delta18O_sw_scale == "vpdb":
            delta18O_sw_vpdb = float(delta18O_sw)
        else:
            raise ValueError(f"delta18O_sw_scale must be 'vpdb' or 'vsmow', got '{delta18O_sw_scale}'")

        delta_w_effective = delta18O_sw_vpdb + delta_genus
        delta_diff = delta18O_c - delta_w_effective
        T = 16.998 - 4.52 * delta_diff

        if not (0.0 <= T <= 35.0):
            import warnings as _warnings

            _warnings.warn(
                f"Bemis et al. (1998) returns T = {T:.2f} °C, which is "
                f"outside the calibrated 16-25 °C window — the input "
                f"(δc - δw_eff) = {delta_diff:.2f} ‰ is unphysical or "
                "unconvertible. Do not interpret the result as a calibrated "
                "paleotemperature.",
                UserWarning,
                stacklevel=2,
            )
        return float(T)

    @staticmethod
    def compute_paleotemperature_kim_oneil(delta18O_sw: float, delta18O_c: float) -> float:
        """
        Kim & O'Neil (1997) 古温度方程 - 碳酸盐 δ¹⁸O

        公式: 1000 ln α = 18.03 × (10³/T) - 32.42
        其中 α = (1 + δc_VSMOW/1000) / (1 + δw/1000)
        T 单位: Kelvin

        注意: 分馏方程要求碳酸盐与海水处于同一标尺 (VSMOW)。
            δc 以 VPDB 报告时须先转换:
            δc_VSMOW = 1.03091 × δc_VPDB + 30.91
            (缺此换算会得到 ~300 °C 的荒谬结果——2026-09 复审
            发现的标尺混用缺陷)

        参数:
            delta18O_sw: 海水 δ¹⁸O (‰ VSMOW)
            delta18O_c: 碳酸盐 δ¹⁸O (‰ VPDB)

        返回:
            古温度 (°C)
        """
        # VPDB -> VSMOW 标尺转换
        delta18O_c_vsmow = 1.03091 * delta18O_c + 30.91

        # 转换为 alpha
        alpha = (1 + delta18O_c_vsmow / 1000) / (1 + delta18O_sw / 1000)

        # 求解温度 (T in Kelvin)
        # 1000 ln α = 18.03 * (1000/T) - 32.42
        # => 1000 ln α + 32.42 = 18.03 * (1000/T)
        # => T = 18030 / (1000 ln α + 32.42)
        ln_alpha = np.log(alpha)
        T_kelvin = 18030.0 / (1000.0 * ln_alpha + 32.42)

        # 转换为 Celsius
        T_celsius = T_kelvin - 273.15
        return float(T_celsius)


def block_bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Block Bootstrap 置信区间（适用于自相关时间序列）。

    传统 bootstrap 假设样本独立，但地层同位素记录存在时间自相关。
    Block bootstrap（Politis & Romano 1994）通过保留 block 内部的时间结构
    来构建有效的置信区间。

    Parameters
    ----------
    data : array-like, 1D
        时间序列数据。
    statistic_func : callable
        计算统计量的函数，接受 1D 数组返回标量。
    block_size : int, optional
        Block 长度。如果为 None，则使用 Politis & White (2004) 自动选择法。
    n_bootstrap : int, default 1000
        Bootstrap 迭代次数。
    alpha : float, default 0.05
        置信区间的显著性水平（返回 1-alpha CI）。

    Returns
    -------
    ci_lower, ci_upper : float
        置信区间下界和上界。

    Notes
    -----
    Block bootstrap 算法（Politis & Romano 1994, JASA）：

    1. 估计最优 block size b（Politis & White 2004 自动选择法）
    2. 从序列中随机抽取长度为 b 的 blocks（可重叠）
    3. 将选中的 blocks 拼接为长度为 n 的伪序列
    4. 对伪序列计算统计量
    5. 取 bootstrap 分布的 alpha/2 和 1-alpha/2 分位数

    覆盖率和 CI 宽度取决于自相关结构与 block size 的匹配程度。

    References
    ----------
    Politis, D.N. & Romano, J.P. (1994). "The stationary bootstrap."
    J. Am. Stat. Assoc., 89: 1303-1313.

    Politis, D.N. & White, H. (2004). "Automatic block-length selection
    for the dependent bootstrap." Econometric Reviews, 23: 53-70.
    """
    data = np.asarray(data, dtype=float)

    # 移除 NaN
    mask = ~np.isnan(data)
    data = data[mask]

    n = len(data)
    if n < 4:
        return np.nan, np.nan

    # 1. 估计最优 block size（Politis-White 2004）
    if block_size is None:
        block_size = _optimal_block_size(data)
        block_size = max(1, min(block_size, n // 2))

    # 2. Block bootstrap 重采样
    bootstrap_stats = np.empty(n_bootstrap)
    # 本地 RNG：旧实现用 np.random.randint 的全局流，CI 无法复现，且会改写
    # 同进程其它随机分析的结果。
    rng = np.random.default_rng(seed)

    for i in range(n_bootstrap):
        # 随机选择 blocks 并拼接为长度为 n 的伪序列
        resampled = np.empty(n)
        pos = 0
        while pos < n:
            start = rng.integers(0, n - block_size + 1)
            block = data[start : start + block_size]
            copy_len = min(len(block), n - pos)
            resampled[pos : pos + copy_len] = block[:copy_len]
            pos += copy_len

        bootstrap_stats[i] = statistic_func(resampled)

    # 3. 计算分位数 CI
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(ci_lower), float(ci_upper)


def _optimal_block_size(data: np.ndarray) -> int:
    """
    使用 Politis-White 2004 方法自动选择最优 block size。

    基于数据的高阶自相关结构选择 block 长度 b，使得
    b -> 0 as n -> infinity 同时 b * n^(-1/3) -> infinity。
    """
    n = len(data)
    data = data - data.mean()

    # 计算 ACF
    max_lag = min(n // 2, int(np.sqrt(n)) + 1)
    acf_values = np.zeros(max_lag + 1)
    var_sum = np.sum(data**2)

    if var_sum == 0:
        return max(1, n // 10)

    acf_values[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf_values[lag] = np.sum(data[:-lag] * data[lag:]) / var_sum

    # 找到第一个通过混洗检验的滞后（block size 估计）
    # 简化的选择：使用直到 ACF 首次不显著的滞后
    threshold = 1.96 / np.sqrt(n)
    m = 1
    for lag in range(1, max_lag + 1):
        if abs(acf_values[lag]) < threshold:
            break
        m = lag

    # 防止过度平滑
    b = max(1, min(m, int(np.sqrt(n))))

    # Politis-White 建议 b ~ n^(1/3) 作为默认下界
    b = max(b, int(n ** (1.0 / 3.0)))

    return int(b)
