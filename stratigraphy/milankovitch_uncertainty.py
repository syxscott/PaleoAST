# =============================================================================
# FILE: stratigraphy/milankovitch_uncertainty.py
# =============================================================================
"""
米兰科维奇频率的不确定度与区间（Waltham 2015 口径）。

用途
====
把地层记录里**量到的周期**匹配到具体的米兰科维奇轨道（岁差 / 倾角 / 偏心率），
并给出每个轨道在给定年龄上的**理论频率**及其**不确定度区间**（frequency
window）。本模块提供四个入口:

* :func:`milankovitch_interval_frequencies` —— 某时刻各轨道谐波的理论频率与
  频率区间;
* :func:`milankovitch_intervals` —— 一段年龄范围（一个 "interval"）内各轨道的
  频率窗口，用于把一段地层记录里的周期整体匹配过去;
* :func:`match_period_to_orbit` —— 把一个实测周期匹配到最可能的轨道;
* :func:`waltham_orbital_solution` —— Waltham (2015) 的正演不确定度传播
  （拖曳因子 → 地月距离 → 黄赤交角 → 日长 → 岁差率）;
* :func:`bayesian_frequency_solution` —— Waltham (2015, EPSL) 口径的**贝叶斯
  线性回归**：把一组实测周期整体平移到理论频率集合上，给出频率修正量与
  不确定度。

另外提供**界面门面** :class:`MilankovitchUncertaintyAnalyzer`：生成的参数表单
填不了"轨道解"这种对象（它只能填数字、字符串与表格的某一列），因此门面
按名字自己加载内置的 La04 解，只暴露年龄轴与两个标量参数。

方法
====
理论频率来自仓库内置的天文解 (:mod:`stratigraphy.astronomy`)，分三种情形:

``precession_21``（气候岁差）
    取仓库周期表里的**经典带** 19–23 kyr（中心 21 kyr）。理由: 气候岁差
    周期是"岁差载波被偏心率调幅"后的**观测间距**，而
    :mod:`stratigraphy.astronomy` 暴露的升交点黄经是**常数速率**的线性
    近似（13.969 °/kyr），其日照量实现也未包含"黄经对固定参考系的岁差相位"
    项，因此从这条解上直接量出来的岁差间距带有系统偏差。经典带 19–23 kyr
    正是文献里"岁差周期在 19–23 kyr 之间游走"的表述。

``obliquity_41``（倾角）与 ``eccentricity_100``（偏心率）
    在目标年龄附近开一个分析窗口（默认 ±300 kyr），**数周期**: 找出序列的
    极大值、剔掉拍频造成的"小抖动"（极值间距小于参考周期 60% 时保留较高者），
    取相邻极大值间距的**截尾均值**作为中心周期，取这些间距的**标准差**作为
    观测不确定度。倾角序列是一条近乎单色的谱线（41 kyr），偏心率序列则是
    g4/g5 双线拍频形成的**宽带**——这正是文献里 "~100 kyr 周期" 的峰在
    80–130 kyr 之间迁移的原因，用"数周期"而不是"找谱峰"才能给出有意义的
    局部周期。

不确定度由两部分**平方和**合成:

* **观测项** —— 上述周期间距自身的散布（倾角 ~5%、偏心率 ~10%）；
* **理论项（Waltham 正演）** —— :func:`waltham_orbital_solution` 给出的岁差率
  相对不确定度经"岁差驱动"传递到频率上的相对不确定度（年轻人的年龄上
  几乎为零，越往古越大）。

:func:`waltham_orbital_solution` 本身复现 Waltham (2015, JSR) 的正演链:
潮汐拖曳因子的取值范围 → 地月距离范围 → 平均黄赤交角范围 → 日长范围 →
岁差率范围。其中地月距离用 Waltham 的拖曳因子模型
``a(t)^6.5 = a0^6.5 - 6.5 C ∫ f dt``，日长用角动量守恒
``I_E ω cos ε + m √(G M_E a) = const``，岁差率用四极矩力矩
``p_A ∝ ω (m/a³ + m_s/a_s³) cos ε``（常数按现在的岁差率 50.287961″/yr
标定）。

与天文解模块的关系
==================
本模块**只读**使用 :mod:`stratigraphy.astronomy` 的公开 API:

* :func:`~stratigraphy.astronomy.load_astronomical_solution` —— 取 La04 轨道解;
* :class:`~stratigraphy.astronomy.AstronomicalSolution` —— 取偏心率、黄赤交角、
  近日点黄经序列;
* :func:`~stratigraphy.astronomy.milankovitch_periods` —— 取经典周期表
  （``precession_19/21/23``、``obliquity_41``、``eccentricity_100``），作为
  周期族的命名与经典带。

本模块不复制、不修改 :mod:`stratigraphy.astronomy` 的任何实现。

简化的说明（诚实性）
====================
Waltham (2015) 的完整模型包含带噪声的贝叶斯推断与不确定度传播。本模块
**没有**复现论文的全部统计细节，具体简化如下:

1. **贝叶斯回归只做了共模频率平移**。:func:`bayesian_frequency_solution` 用
   共轭高斯（平坦先验）解出"所有观测频率相对理论频率的整体偏移量"及其
   后验标准差，没有层次噪声模型、没有边缘化掉每条谐波各自的不确定度。
2. **正演链的拖曳因子历史没有积分**。地月距离用**目标年龄处**的拖曳因子
   区间乘以整段时间（而不是沿时间积分 ``∫ f dt``），这与参考计算器的做法
   一致，在第一个百万年内差别可以忽略，越往古差别越大。
3. **岁差率对频率的敏感度按比例近似**。链里给出的是岁差率范围，转成
   频率区间时按 ``δf/f ≈ δp_A/p_A`` 近似（忽略了"频率 = 岁差率 − 固定
   偏移"里的那几项）。
4. **偏心率族的理论不确定度为零**。潮汐耗散历史不影响偏心率轨道，其
   不确定度只来自解自身（这里取周期间距的散布）。
5. **岁差族用经典带而非解导出值**（理由见上）。

参考文献
========
* Waltham, T., 2015. Milankovitch period uncertainties and their impact on
  cyclostratigraphy. *Journal of Sedimentary Research*, 85(4): 688-704.
  doi:10.2110/jsr.2015.66.（拖曳因子模型与周期不确定度传播）
* Waltham, T., 2015. Solving the orbital equations of the Milankovitch cycles.
  *Earth and Planetary Science Letters*, 405: 499-509.（贝叶斯线性回归口径：
  把观测周期匹配到理论轨道频率上，给出频率窗口）
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. Solution for
  the eccentricity and obliquity of the Earth's orbits over 5 Myr.
  *Astronomy and Astrophysics*, 413: 755-767.（La04 轨道解与岁差率常数）
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. A 250-Myr
  solution to the Pliocene-Pleistocene ice ages. *Earth and Planetary Science
  Letters*, 224: 299-313.
* Lisiecki, L. E. & Raymo, M. E., 2005. Four million years of climate
  oscillations: phase relationships of Milankovitch cycles and eccentricity
  cycles. *Earth and Planetary Science Letters*, 237: 369-398.（~100 kyr 谱带
  随时间的迁移）
* Berger, A. & Loutre, M. F., 1992. Astronomical solutions for paleoclimate
  studies over the last 3 million years. *Earth and Planetary Science Letters*,
  111: 369-382.（气候岁差的分支结构）
* Jeffreys, H., 1962. The Earth: Its Origin and History. Pergamon.
  （岁差率的四极矩力矩表达）
* Walker, J. C. W., 1977. Tidal friction and the evolution of the Moon's
  orbit. *Journal of Geophysics Research*, 82: 1407-1425.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from stratigraphy.astronomy import (
    AstronomicalSolution,
    load_astronomical_solution,
    milankovitch_periods,
)
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "FrequencyRegression",
    "MilankovitchFrequency",
    "MilankovitchInterval",
    "MilankovitchUncertaintyAnalyzer",
    "MilankovitchUncertaintyResult",
    "PeriodMatch",
    "WalthamSolution",
    "bayesian_frequency_solution",
    "match_period_to_orbit",
    "milankovitch_interval_frequencies",
    "milankovitch_intervals",
    "waltham_orbital_solution",
]


# =============================================================================
# 常量
# =============================================================================

#: 默认的分析窗口半宽 (kyr)。窗口覆盖目标年龄前后各 300 kyr, 里面含约
#: 14 个倾角周期、6 个偏心率周期。
DEFAULT_HALF_WINDOW_KYR = 300.0

#: 拖曳因子的现代表达式区间 (Waltham 2015)。
_DRAG_FACTOR_PRESENT = (20.0, 22.0)

#: 拖曳因子在 4.5 Ga 的取值区间 (Waltham 2015)。
_DRAG_FACTOR_4500_MA = (6.77, 6.93)

#: 拖曳因子弛豫时间常数 (Myr) 的取值区间。
_DRAG_TIME_CONSTANTS_MYR = (100.0, 1000.0)

#: 随机拖曳分量的对数标准差与"包络拟合"的时间常数 (Myr)。
_DRAG_STOCHASTIC_SIGMA = 1.48
_DRAG_ENVELOPE_TAU_MYR = 2103.0
_DRAG_ENVELOPE_OFFSET = 5.69
_DRAG_ENVELOPE_GAIN = 1.225

#: 潮汐常数 C: ``a(t)^6.5 = a0^6.5 - 6.5 C ∫ f dt`` (m^6.5 / Myr)。
_TIDAL_CONSTANT = 3.15576e50

#: 现在的地月距离 (m)。
_EARTH_MOON_DISTANCE_M = 3.84399e8

#: 地球的转动惯量 (kg m^2), 即 0.3307 M R^2。
_EARTH_MOMENT_OF_INERTIA = 8.016e37

#: 地球质量 (kg) 与半径 (m)。
_EARTH_MASS = 5.9722e24
_EARTH_RADIUS = 6.371e6

#: 月球质量 (kg)。
_MOON_MASS = 7.3477e22

#: 太阳质量 (kg) 与平均距离 (m)。
_SUN_MASS = 1.9885e30
_SUN_DISTANCE = 1.495978707e11

#: 万有引力常数 (m^3 kg^-1 s^-2)。
_GRAVITY = 6.6743e-11

#: 现在的恒星自转角速度 (rad/s)。
_SIDEREAL_ROTATION_RATE = 7.292115e-5

#: 现在的平岁差 (黄经) 速率, 单位 arcsec/yr (Laskar et al. 2004)。
_PRECESSION_ARCSEC_PER_YEAR = 50.287961

#: 现在的平岁差 (黄经) 速率, 单位 度/kyr。
PRECESSION_DEG_PER_KYR = _PRECESSION_ARCSEC_PER_YEAR / 3600.0 * 1000.0

#: 黄赤交角 ~ 地月距离的经验拟合 (Waltham 2015); 自变量单位是 1000 km。
_OBLIQUITY_FIT_COEFFICIENTS = (
    84.23171,
    -0.8430214,
    0.003838065,
    -7.656218e-6,
    5.991817e-9,
)

#: 极值剔除时允许的最小间距占参考周期的比例。
_MIN_SEPARATION_FACTOR = 0.6

#: 计数周期前对序列做的移动平均长度 (kyr)。
_SMOOTHING_KYR = 11.0

#: 截尾均值的截尾比例 (两端各截)。
_TRIM_FRACTION = 0.1


# =============================================================================
# 谐波目录
# =============================================================================


@dataclass(frozen=True)
class _Harmonic:
    """一个米兰科维奇谐波的描述。

    Attributes:
        name: 谐波名 (与周期表同名)。
        family: 轨道族: ``precession`` / ``obliquity`` / ``eccentricity``。
        order: 族内序号 (从 1 开始)。
        reference_period_kyr: 经典周期 (kyr), 取自周期表。
        band_kyr: 经典周期带 (kyr); 为 None 表示没有经典带。
        from_solution: 中心周期是否由天文解数周期得到。
    """

    name: str
    family: str
    order: int
    reference_period_kyr: float
    band_kyr: tuple[float, float] | None
    from_solution: bool


def _build_catalogue() -> dict[str, _Harmonic]:
    """用仓库的周期表构造谐波目录。

    经典周期与经典带都取自 :func:`stratigraphy.astronomy.milankovitch_periods`,
    本模块不另立一套数值。
    """
    table = milankovitch_periods()
    return {
        "precession_21": _Harmonic(
            name="precession_21",
            family="precession",
            order=1,
            reference_period_kyr=table["precession_21"],
            band_kyr=(table["precession_19"], table["precession_23"]),
            from_solution=False,
        ),
        "obliquity_41": _Harmonic(
            name="obliquity_41",
            family="obliquity",
            order=1,
            reference_period_kyr=table["obliquity_41"],
            band_kyr=None,
            from_solution=True,
        ),
        "eccentricity_100": _Harmonic(
            name="eccentricity_100",
            family="eccentricity",
            order=1,
            reference_period_kyr=table["eccentricity_100"],
            band_kyr=None,
            from_solution=True,
        ),
    }


#: 谐波目录 (名字与 :func:`stratigraphy.astronomy.milankovitch_periods` 对齐)。
_HARMONICS: dict[str, _Harmonic] = _build_catalogue()


# =============================================================================
# 结果容器
# =============================================================================


@dataclass(frozen=True)
class MilankovitchFrequency:
    """某个年龄上某个米兰科维奇谐波的理论频率与频率区间。

    Attributes:
        name: 谐波名 (如 ``"precession_21"``)。
        family: 轨道族。
        order: 族内序号 (从 1 开始)。
        age_ma: 该结果的年龄 (Ma)。
        frequency: 理论频率 (1/kyr)。
        frequency_min: 频率区间的下界 (1/kyr)。
        frequency_max: 频率区间的上界 (1/kyr)。
        sigma: 区间半宽 (1/kyr), 即 1σ。
        reference_period_kyr: 经典周期 (kyr), 取自周期表。
        source: 理论频率的来源: ``"solution"`` (天文解数周期) 或
            ``"reference_table"`` (经典周期表)。
    """

    name: str
    family: str
    order: int
    age_ma: float
    frequency: float
    frequency_min: float
    frequency_max: float
    sigma: float
    reference_period_kyr: float
    source: str

    @property
    def period_kyr(self) -> float:
        """理论周期 (kyr)。"""
        return 1.0 / self.frequency

    @property
    def period_min_kyr(self) -> float:
        """区间对应的最短周期 (kyr)。"""
        return 1.0 / self.frequency_max

    @property
    def period_max_kyr(self) -> float:
        """区间对应的最长周期 (kyr)。"""
        return 1.0 / self.frequency_min

    def summary(self) -> str:
        """一行可读摘要。"""
        return _("{0} at {1:.3f} Ma: {2:.6f} 1/kyr ({3:.3f} kyr), window [{4:.6f}, {5:.6f}] 1/kyr").format(
            self.name,
            self.age_ma,
            self.frequency,
            self.period_kyr,
            self.frequency_min,
            self.frequency_max,
        )


@dataclass(frozen=True)
class MilankovitchInterval:
    """一段年龄范围内各轨道的频率窗口。

    Attributes:
        age_start_ma: 区间起始年龄 (Ma)。
        age_end_ma: 区间结束年龄 (Ma)。
        entries: 各谐波的频率窗口 (每个谐波一条), 频率上下界是该年龄范围内
            所有采样点的并集。
    """

    age_start_ma: float
    age_end_ma: float
    entries: tuple[MilankovitchFrequency, ...]

    def by_name(self, name: str) -> MilankovitchFrequency:
        """按谐波名取频率窗口。

        Raises:
            ValidationError: 若名字未知。
        """
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise ValidationError(
            f"Unknown Milankovitch harmonic: {name!r}",
            details={"known": sorted(_HARMONICS)},
        )

    def summary(self) -> str:
        """一行可读摘要。"""
        return _("Milankovitch interval {0:.3f}-{1:.3f} Ma with {2} harmonic windows").format(
            self.age_start_ma, self.age_end_ma, len(self.entries)
        )


@dataclass(frozen=True)
class WalthamSolution:
    """Waltham (2015) 正演链在某一年龄上的结果。

    Attributes:
        age_ma: 年龄 (Ma)。
        drag_factor_min: 潮汐拖曳因子的下界。
        drag_factor_max: 潮汐拖曳因子的上界。
        earth_moon_distance_min_km: 地月距离下界 (km)。
        earth_moon_distance_max_km: 地月距离上界 (km)。
        obliquity_min_deg: 平均黄赤交角下界 (度)。
        obliquity_max_deg: 平均黄赤交角上界 (度)。
        day_length_min_hours: 日长下界 (小时, 太阳日)。
        day_length_max_hours: 日长上界 (小时, 太阳日)。
        precession_rate_min_deg_per_kyr: 平岁差率下界 (度/kyr)。
        precession_rate_max_deg_per_kyr: 平岁差率上界 (度/kyr)。
    """

    age_ma: float
    drag_factor_min: float
    drag_factor_max: float
    earth_moon_distance_min_km: float
    earth_moon_distance_max_km: float
    obliquity_min_deg: float
    obliquity_max_deg: float
    day_length_min_hours: float
    day_length_max_hours: float
    precession_rate_min_deg_per_kyr: float
    precession_rate_max_deg_per_kyr: float

    @property
    def precession_rate_deg_per_kyr(self) -> float:
        """平岁差率的中值 (度/kyr)。"""
        return 0.5 * (self.precession_rate_min_deg_per_kyr + self.precession_rate_max_deg_per_kyr)

    @property
    def precession_rate_relative_sigma(self) -> float:
        """岁差率的相对 1σ 不确定度。"""
        mean = self.precession_rate_deg_per_kyr
        if mean <= 0.0:
            return 0.0
        return 0.5 * (self.precession_rate_max_deg_per_kyr - self.precession_rate_min_deg_per_kyr) / mean

    @property
    def earth_moon_distance_km(self) -> float:
        """地月距离的中值 (km)。"""
        return 0.5 * (self.earth_moon_distance_min_km + self.earth_moon_distance_max_km)

    @property
    def day_length_hours(self) -> float:
        """日长的中值 (小时, 太阳日)。"""
        return 0.5 * (self.day_length_min_hours + self.day_length_max_hours)

    def summary(self) -> str:
        """一行可读摘要。"""
        return _(
            "Waltham solution at {0:.3f} Ma: precession {1:.4f} deg/kyr, "
            "day length {2:.3f} h, Earth-Moon distance {3:.0f} km"
        ).format(
            self.age_ma,
            self.precession_rate_deg_per_kyr,
            self.day_length_hours,
            self.earth_moon_distance_km,
        )


@dataclass(frozen=True)
class PeriodMatch:
    """一个实测周期与某条轨道的匹配结果。

    Attributes:
        orbit_name: 匹配到的谐波名。
        family: 轨道族。
        order: 族内序号。
        age_ma: 匹配所用的年龄 (Ma)。
        observed_period_kyr: 实测周期 (kyr)。
        theoretical_period_kyr: 该轨道的理论周期 (kyr)。
        period_min_kyr: 该轨道周期窗口的下界 (kyr)。
        period_max_kyr: 该轨道周期窗口的上界 (kyr)。
        deviation: 相对偏差 ((实测 − 理论) / 理论)。
        deviation_kyr: 绝对偏差 (kyr)。
        in_window: 实测周期是否落在该轨道的频率窗口内。
        alternatives: 同样能容纳该实测周期的其它谐波名。
    """

    orbit_name: str
    family: str
    order: int
    age_ma: float
    observed_period_kyr: float
    theoretical_period_kyr: float
    period_min_kyr: float
    period_max_kyr: float
    deviation: float
    deviation_kyr: float
    in_window: bool
    alternatives: tuple[str, ...]

    @property
    def matched(self) -> bool:
        """是否落在窗口内 (即匹配成立)。"""
        return self.in_window

    def summary(self) -> str:
        """一行可读摘要。"""
        return _("{0}: observed {1:.3f} kyr vs theoretical {2:.3f} kyr ({3:+.2f}%, in window: {4})").format(
            self.orbit_name,
            self.observed_period_kyr,
            self.theoretical_period_kyr,
            100.0 * self.deviation,
            _("yes") if self.in_window else _("no"),
        )


@dataclass(frozen=True)
class FrequencyRegression:
    """贝叶斯线性回归 (共模频率平移) 的结果。

    Attributes:
        frequency_offset: 频率修正量 (1/kyr): 观测频率整体相对理论频率的平移。
        frequency_offset_sigma: 修正量的后验 1σ (1/kyr)。
        residuals: 每个观测的残差 (1/kyr, 观测频率 − 理论频率 − 修正量)。
        weights: 每个观测的权重 (1/(1/kyr)^2)。
        samples: 后验抽样样本 (1/kyr); 未抽样时为 None。
    """

    frequency_offset: float
    frequency_offset_sigma: float
    residuals: npt.NDArray
    weights: npt.NDArray
    samples: npt.NDArray | None

    def summary(self) -> str:
        """一行可读摘要。"""
        return _("Frequency offset {0:+.6f} 1/kyr +/- {1:.6f} 1/kyr ({2} cycles)").format(
            self.frequency_offset,
            self.frequency_offset_sigma,
            int(self.residuals.size),
        )


# =============================================================================
# 内部工具
# =============================================================================


def _require_positive(value: float, name: str) -> float:
    """校验一个正数参数。"""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValidationError(_("{0} must be a finite positive number").format(name), details={"value": value})
    return number


def _validate_ages(age_ma: float | npt.NDArray, solution: AstronomicalSolution) -> npt.NDArray:
    """校验年龄并返回一维数组。

    Raises:
        ValidationError: 若年龄为空、含非有限值、为负、或超出解的时间范围。
    """
    ages = np.atleast_1d(np.asarray(age_ma, dtype=float))
    if ages.size == 0:
        raise ValidationError(_("age_ma must contain at least one age"))
    if not np.all(np.isfinite(ages)):
        raise ValidationError(_("age_ma must contain finite ages only"))
    if np.any(ages < 0.0):
        raise ValidationError(
            _("age_ma must not contain negative ages"),
            details={"minimum": float(np.min(ages))},
        )

    lower = float(np.min(solution.times_ma))
    upper = float(np.max(solution.times_ma))
    if float(np.max(ages)) > upper or float(np.min(ages)) < lower:
        raise ValidationError(
            _("Requested ages fall outside the astronomical solution's time range"),
            details={"range_ma": [lower, upper]},
        )
    return ages


def _trimmed_mean(values: npt.NDArray) -> float:
    """截尾均值 (两端各截 10%)。

    对只有三个间距的小样本, 截尾后即为中位数 —— 这正是偏心率序列需要的
    稳健性: 拍频低谷里会出现一个偏短的间距, 它不该把中心值拖走。
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        raise ValidationError(_("Cannot average an empty set of cycle spacings"))
    cut = int(np.ceil(_TRIM_FRACTION * ordered.size))
    if ordered.size - 2 * cut <= 0:
        return float(np.mean(ordered))
    return float(np.mean(ordered[cut : ordered.size - cut]))


def _sorted_series(solution: AstronomicalSolution, attribute: str) -> tuple[npt.NDArray, npt.NDArray]:
    """取出按年龄升序排列的时间轴 (kyr) 与指定序列。

    Parameters:
        solution: 天文解。
        attribute: :class:`~stratigraphy.astronomy.AstronomicalSolution` 的
            序列属性名 (``eccentricity`` 或 ``obliquity``)。
    """
    order = np.argsort(np.asarray(solution.times_ma, dtype=float))
    times_kyr = np.asarray(solution.times_ma, dtype=float)[order] * 1000.0
    values = np.asarray(getattr(solution, attribute), dtype=float)[order]
    return times_kyr, values


def _thinned_maxima(positions: npt.NDArray, heights: npt.NDArray, min_separation: float) -> npt.NDArray:
    """剔掉靠得太近的极大值: 一对间距不足的极大值里只保留较高的那个。"""
    keep = list(range(positions.size))
    changed = True
    while changed and len(keep) > 1:
        changed = False
        for position in range(len(keep) - 1):
            first = keep[position]
            second = keep[position + 1]
            if positions[second] - positions[first] < min_separation:
                weaker = first if heights[first] <= heights[second] else second
                keep.remove(weaker)
                changed = True
                break
    return positions[keep]


def _cycle_spacings(
    times_kyr: npt.NDArray,
    values: npt.NDArray,
    center_kyr: float,
    half_window_kyr: float,
    reference_period_kyr: float,
) -> npt.NDArray:
    """数目标年龄附近的周期, 返回相邻极大值的间距 (kyr)。

    只有**两个端点都落在窗口内**的间距才被保留, 避免把窗口边缘截断的
    间距当成真实的周期。
    """
    lower = center_kyr - half_window_kyr
    upper = center_kyr + half_window_kyr
    inside = (times_kyr >= lower) & (times_kyr <= upper)
    window_times = times_kyr[inside]
    window_values = values[inside]
    if window_times.size < 3:
        return np.empty(0, dtype=float)

    step = window_times[1] - window_times[0]
    smooth_bins = max(3, round(_SMOOTHING_KYR / step))
    smoothed = np.convolve(window_values, np.ones(smooth_bins) / smooth_bins, mode="same")
    interior = np.where((smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] >= smoothed[2:]))[0] + 1
    if interior.size == 0:
        return np.empty(0, dtype=float)

    positions: list[float] = []
    heights: list[float] = []
    for index in interior:
        left, middle, right = smoothed[index - 1], smoothed[index], smoothed[index + 1]
        denominator = left - 2.0 * middle + right
        shift = 0.5 * (left - right) / denominator if denominator != 0.0 else 0.0
        positions.append(float(window_times[index] + shift * step))
        heights.append(float(middle))

    position_array = np.asarray(positions, dtype=float)
    height_array = np.asarray(heights, dtype=float)
    kept = _thinned_maxima(position_array, height_array, _MIN_SEPARATION_FACTOR * reference_period_kyr)
    if kept.size < 2:
        return np.empty(0, dtype=float)

    spacings = np.diff(kept)
    complete = (kept[:-1] >= lower) & (kept[1:] <= upper)
    return spacings[complete]


def _estimate_solution_period(
    solution: AstronomicalSolution,
    age_ma: float,
    harmonic: _Harmonic,
    half_window_kyr: float,
) -> tuple[float, float]:
    """从天文解上数周期, 返回 (中心周期 kyr, 1σ 周期不确定度 kyr)。

    周期数不足两个时退回经典周期, 且不确定度取经典带宽度。
    """
    attribute = "obliquity" if harmonic.family == "obliquity" else "eccentricity"
    times_kyr, values = _sorted_series(solution, attribute)
    spacings = _cycle_spacings(times_kyr, values, age_ma * 1000.0, half_window_kyr, harmonic.reference_period_kyr)

    if spacings.size < 2:
        band = harmonic.band_kyr
        half_width = 0.5 * (band[1] - band[0]) if band is not None else 0.05 * harmonic.reference_period_kyr
        return harmonic.reference_period_kyr, half_width

    central = _trimmed_mean(spacings)
    if spacings.size > 1:
        spread = float(np.std(spacings, ddof=1))
    else:
        spread = 0.0
    spread = max(spread, 0.5 * (np.max(spacings) - np.min(spacings)))
    return central, spread


# =============================================================================
# Waltham (2015) 正演链
# =============================================================================


def _drag_factor_range(
    age_ma: float,
    present: tuple[float, float] = _DRAG_FACTOR_PRESENT,
    past: tuple[float, float] = _DRAG_FACTOR_4500_MA,
) -> tuple[float, float]:
    """潮汐拖曳因子的取值范围。

    两条分支 (Waltham 2015): 长期弛豫分支 (随年龄从现值弛豫到 4.5 Ga 的
    值) 与随机分支 (对数正态的独立增量, 以及一条对观测包络的经验拟合)。

    Parameters:
        age_ma: 年龄 (Ma)。
        present: 现今拖曳因子的取值区间。
        past: 4.5 Ga 处拖曳因子的取值区间。

    Returns:
        (下界, 上界)。
    """
    age = float(age_ma)
    f0_min, f0_max = float(present[0]), float(present[1])
    f45_min, f45_max = float(past[0]), float(past[1])
    tau_min, tau_max = _DRAG_TIME_CONSTANTS_MYR

    def secular(f0: float, f45: float, tau: float) -> float:
        alpha = (tau / 4500.0) * (1.0 - np.exp(-4500.0 / tau))
        mean = (f45 - alpha * f0) / (1.0 - alpha)
        if age <= 0.0:
            return f0
        return float(mean + (tau / age) * (f0 - mean) * (1.0 - np.exp(-age / tau)))

    secular_min = secular(f0_min, f45_min, tau_min)
    secular_max = secular(f0_max, f45_max, tau_max)

    tau = tau_min
    sigma = _DRAG_STOCHASTIC_SIGMA
    increments = 4500.0 / tau
    scale = ((1.0 + increments) * f45_max - f0_max) / (increments * np.exp(sigma / np.sqrt(increments)))
    count = 1.0 + age / tau
    stochastic_max = float((f0_max + count * scale * np.exp(sigma / np.sqrt(count))) / (1.0 + count))
    stochastic_min = float(_DRAG_ENVELOPE_OFFSET + _DRAG_ENVELOPE_GAIN * (1.0 - np.exp(-age / _DRAG_ENVELOPE_TAU_MYR)))

    return min(secular_min, stochastic_min), max(secular_max, stochastic_max)


def _earth_moon_distance_range(age_ma: float, drag_min: float, drag_max: float) -> tuple[float, float]:
    """地月距离范围 (km)。

    用 Waltham 的拖曳因子关系 ``a(t)^6.5 = a0^6.5 - 6.5 C f t``; 简化: 取目标
    年龄处的拖曳因子区间乘以整段时间 (参考计算器的做法)。
    """
    age = float(age_ma)
    start = _EARTH_MOON_DISTANCE_M**6.5
    inner_min = start - 6.5 * _TIDAL_CONSTANT * age * drag_max
    inner_max = start - 6.5 * _TIDAL_CONSTANT * age * drag_min
    inner_min = max(inner_min, 0.0)
    inner_max = max(inner_max, 0.0)
    distance_min = inner_min ** (1.0 / 6.5) / 1000.0
    distance_max = inner_max ** (1.0 / 6.5) / 1000.0
    return distance_min, distance_max


def _mean_obliquity_deg(distance_km: float) -> float:
    """平均黄赤交角 (度) ~ 地月距离的经验拟合 (Waltham 2015)。"""
    scaled = distance_km / 1000.0
    coefficients = _OBLIQUITY_FIT_COEFFICIENTS
    return float(
        coefficients[0]
        + coefficients[1] * scaled
        + coefficients[2] * scaled**2
        + coefficients[3] * scaled**3
        + coefficients[4] * scaled**4
    )


def _rotation_rate(distance_km: float, obliquity_deg: float) -> float:
    """自转角速度 (rad/s)。

    角动量守恒: ``I_E ω cos ε + m √(G M_E a) = const``; 常数由现在的地月
    距离与现在的自转速率定出, 因此过去的距离更小 → 月球轨道角动量更小 →
    自转更快 (日更短)。
    """
    reference_orbital = _MOON_MASS * np.sqrt(_GRAVITY * _EARTH_MASS * _EARTH_MOON_DISTANCE_M)
    constant = (
        _EARTH_MOMENT_OF_INERTIA * _SIDEREAL_ROTATION_RATE * np.cos(np.radians(_mean_obliquity_deg(384399.0)))
        + reference_orbital
    )
    orbital = _MOON_MASS * np.sqrt(_GRAVITY * _EARTH_MASS * distance_km * 1000.0)
    return float((constant - orbital) / (_EARTH_MOMENT_OF_INERTIA * np.cos(np.radians(obliquity_deg))))


def _day_length_hours(rotation_rate: float) -> float:
    """太阳日长度 (小时)。"""
    return float(2.0 * np.pi * (1.0 + 1.0 / 365.24) / (rotation_rate * 3600.0))


def _precession_rate_deg_per_kyr(distance_km: float, rotation_rate: float, obliquity_deg: float) -> float:
    """平岁差率 (度/kyr)。

    四极矩力矩给出的岁差率正比于 ``ω (m/a³ + m_s/a_s³) cos ε``; 常数按现在
    的岁差率 (Laskar et al. 2004: 50.287961″/yr) 标定。
    """
    reference_terms = _MOON_MASS / _EARTH_MOON_DISTANCE_M**3 + _SUN_MASS / _SUN_DISTANCE**3
    reference_obliquity = _mean_obliquity_deg(384399.0)
    reference = _SIDEREAL_ROTATION_RATE * reference_terms * np.cos(np.radians(reference_obliquity))
    calibration = PRECESSION_DEG_PER_KYR / reference

    terms = _MOON_MASS / (distance_km * 1000.0) ** 3 + _SUN_MASS / _SUN_DISTANCE**3
    return float(calibration * rotation_rate * terms * np.cos(np.radians(obliquity_deg)))


def waltham_orbital_solution(
    age_ma: float,
    *,
    drag_factor_present: tuple[float, float] = _DRAG_FACTOR_PRESENT,
    drag_factor_4500_ma: tuple[float, float] = _DRAG_FACTOR_4500_MA,
) -> WalthamSolution:
    """Waltham (2015) 的正演解: 拖曳因子 → 地月距离 → 黄赤交角 → 日长 → 岁差率。

    这条正演链**不是**贝叶斯线性回归。Acycle 的 ``MilankovitchCal.m``
    (读自 ``misc/MilankovitchCal.m``) 做的正是这条链: 它按 Waltham (2015,
    JSR) 的拖曳因子模型算出地月距离范围, 再经角动量守恒与四极矩力矩得到
    日长与岁差率范围, 最后把岁差率换算成倾角与气候岁差的周期窗口。
    贝叶斯线性回归是**另一篇**论文 (Waltham 2015, EPSL) 的口径, 本模块在
    :func:`bayesian_frequency_solution` 里单独实现。

    Parameters:
        age_ma: 年龄 (Ma), 必须非负且有限。
        drag_factor_present: 现今拖曳因子的取值区间。
        drag_factor_4500_ma: 4.5 Ga 处拖曳因子的取值区间。

    Returns:
        :class:`WalthamSolution`。

    Raises:
        ValidationError: 若年龄非法, 或拖曳因子区间非法。
    """
    age = float(age_ma)
    if not np.isfinite(age):
        raise ValidationError(_("age_ma must be finite"), details={"value": age_ma})
    if age < 0.0:
        raise ValidationError(_("age_ma must not be negative"), details={"value": age_ma})

    for label, bounds in (("drag_factor_present", drag_factor_present), ("drag_factor_4500_ma", drag_factor_4500_ma)):
        low, high = (float(bounds[0]), float(bounds[1]))
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high < low:
            raise ValidationError(
                _("{0} must be a pair of positive increasing numbers").format(label),
                details={"value": bounds},
            )

    drag_min, drag_max = _drag_factor_range(age, drag_factor_present, drag_factor_4500_ma)
    distance_min, distance_max = _earth_moon_distance_range(age, drag_min, drag_max)

    obliquity_low = _mean_obliquity_deg(distance_max)
    obliquity_high = _mean_obliquity_deg(distance_min)

    rate_low = _rotation_rate(distance_min, obliquity_high)
    rate_high = _rotation_rate(distance_max, obliquity_low)
    day_min = _day_length_hours(rate_high)
    day_max = _day_length_hours(rate_low)

    precession_low = _precession_rate_deg_per_kyr(distance_max, rate_high, obliquity_low)
    precession_high = _precession_rate_deg_per_kyr(distance_min, rate_low, obliquity_high)

    return WalthamSolution(
        age_ma=age,
        drag_factor_min=drag_min,
        drag_factor_max=drag_max,
        earth_moon_distance_min_km=distance_min,
        earth_moon_distance_max_km=distance_max,
        obliquity_min_deg=min(obliquity_low, obliquity_high),
        obliquity_max_deg=max(obliquity_low, obliquity_high),
        day_length_min_hours=min(day_min, day_max),
        day_length_max_hours=max(day_min, day_max),
        precession_rate_min_deg_per_kyr=min(precession_low, precession_high),
        precession_rate_max_deg_per_kyr=max(precession_low, precession_high),
    )


# =============================================================================
# 频率与区间
# =============================================================================


def _harmonic_window(
    harmonic: _Harmonic,
    solution: AstronomicalSolution,
    age_ma: float,
    uncertainty: float,
    half_window_kyr: float,
    chain_sigma: float,
) -> MilankovitchFrequency:
    """构造单个谐波在某个年龄上的频率与频率区间。"""
    if harmonic.from_solution:
        central_period, spread = _estimate_solution_period(solution, age_ma, harmonic, half_window_kyr)
        source = "solution"
    else:
        central_period = harmonic.reference_period_kyr
        band = harmonic.band_kyr
        spread = 0.5 * (band[1] - band[0]) if band is not None else 0.05 * central_period
        source = "reference_table"

    relative = np.hypot(spread / central_period, chain_sigma)
    relative *= uncertainty
    relative = min(relative, 0.9)

    # 窗口在**周期**空间里定义 (观测不确定度本来就来自周期间距), 再换算成
    # 频率窗口; 换算是单调的, 因此理论频率一定落在窗口内。
    frequency = 1.0 / central_period
    period_min = central_period * (1.0 - relative)
    period_max = central_period * (1.0 + relative)
    return MilankovitchFrequency(
        name=harmonic.name,
        family=harmonic.family,
        order=harmonic.order,
        age_ma=age_ma,
        frequency=frequency,
        frequency_min=1.0 / period_max,
        frequency_max=1.0 / period_min,
        sigma=frequency * relative,
        reference_period_kyr=harmonic.reference_period_kyr,
        source=source,
    )


def _interval_frequencies(
    age_ma: float | npt.NDArray,
    solution: AstronomicalSolution,
    uncertainty: float,
    half_window_kyr: float,
) -> tuple[MilankovitchFrequency, ...]:
    """:func:`milankovitch_interval_frequencies` 的内部实现。

    天文解由调用方传入 —— 公开函数不暴露它, 因为生成的参数表单填不了
    "轨道解"这种对象 (它只能填数字、字符串与表格的某一列)。
    """
    scale = _require_positive(uncertainty, "uncertainty")
    window = _require_positive(half_window_kyr, "half_window_kyr")
    ages = _validate_ages(age_ma, solution)

    entries: list[MilankovitchFrequency] = []
    for age in ages:
        chain = waltham_orbital_solution(float(age))
        chain_sigma = chain.precession_rate_relative_sigma
        for harmonic in _HARMONICS.values():
            chain_part = 0.0 if harmonic.family == "eccentricity" else chain_sigma
            entries.append(
                _harmonic_window(
                    harmonic,
                    solution,
                    float(age),
                    scale,
                    window,
                    chain_part,
                )
            )
    return tuple(entries)


def milankovitch_interval_frequencies(
    age_ma: float | npt.NDArray,
    *,
    uncertainty: float = 1.0,
    half_window_kyr: float = DEFAULT_HALF_WINDOW_KYR,
) -> tuple[MilankovitchFrequency, ...]:
    """各米兰科维奇轨道在给定年龄上的理论频率与不确定度区间。

    轨道解固定用内置的 La04 解 (:func:`stratigraphy.astronomy.load_astronomical_solution`),
    因此本函数的参数全部是表单能表达的类型 (数组与标量)。

    Parameters:
        age_ma: 年龄 (Ma), 可为标量或数组。
        uncertainty: 频率区间半宽的**相对缩放因子** (>0): 1.0 表示按模型
            给出的 1σ 区间, 2.0 表示宽一倍。
        half_window_kyr: 数周期的分析窗口半宽 (kyr)。

    Returns:
        :class:`MilankovitchFrequency` 的元组。标量年龄返回每个谐波一条;
        数组年龄按"年龄优先"的顺序排列 (第 i 个年龄的三条谐波在前) —— 即
        每个年龄各得到**一组**结果。

    Raises:
        ValidationError: 若年龄非法、区间超出解的范围, 或参数非法。
    """
    return _interval_frequencies(
        age_ma,
        load_astronomical_solution("la04"),
        uncertainty,
        half_window_kyr,
    )


def milankovitch_intervals(
    age_ma: float,
    *,
    age_end_ma: float | None = None,
    interval_kyr: float | None = None,
    uncertainty: float = 1.0,
    half_window_kyr: float = DEFAULT_HALF_WINDOW_KYR,
    samples: int = 9,
) -> MilankovitchInterval:
    """一段年龄范围内各轨道的频率窗口 (Milankovitch 区间)。

    区间内每个谐波的频率上下界是该范围内所有采样点的**并集**, 因此区间
    越长窗口越宽 —— 这正是"把一段地层记录里的周期整体匹配过去"时需要的
    东西。

    Parameters:
        age_ma: 区间起始年龄 (Ma)。
        age_end_ma: 区间结束年龄 (Ma); 给出时优先于 ``interval_kyr``。
        interval_kyr: 区间长度 (kyr), 以 ``age_ma`` 为中心。
        uncertainty: 频率区间半宽的相对缩放因子 (>0)。
        half_window_kyr: 数周期的分析窗口半宽 (kyr)。
        samples: 区间内的采样点数 (>=2)。

    Returns:
        :class:`MilankovitchInterval`。

    Raises:
        ValidationError: 若参数非法 (年龄倒置、采样点数不足等)。
    """
    count = int(samples)
    if count < 2:
        raise ValidationError(_("samples must be at least 2"), details={"value": samples})

    active = load_astronomical_solution("la04")
    earliest = float(np.min(active.times_ma))
    latest = float(np.max(active.times_ma))

    start = float(age_ma)
    if age_end_ma is not None:
        end = float(age_end_ma)
    elif interval_kyr is not None:
        span = _require_positive(interval_kyr, "interval_kyr")
        start = start - 0.5 * span / 1000.0
        end = start + span / 1000.0
    else:
        end = start
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValidationError(_("age_ma must be finite"))
    if end < start:
        raise ValidationError(
            _("The interval must not end before it starts"),
            details={"start_ma": start, "end_ma": end},
        )

    # 区间被裁到天文解覆盖的年龄范围内 (求解需要非负年龄)。
    start = max(start, earliest)
    end = min(end, latest)
    if end < start:
        raise ValidationError(
            _("The interval falls outside the astronomical solution's time range"),
            details={"range_ma": [earliest, latest]},
        )

    grid = np.linspace(start, end, count)
    entries = _interval_frequencies(
        grid,
        active,
        uncertainty,
        half_window_kyr,
    )

    merged: list[MilankovitchFrequency] = []
    for harmonic in _HARMONICS.values():
        block = [entry for entry in entries if entry.name == harmonic.name]
        middle = block[len(block) // 2]
        merged.append(
            MilankovitchFrequency(
                name=middle.name,
                family=middle.family,
                order=middle.order,
                age_ma=middle.age_ma,
                frequency=middle.frequency,
                frequency_min=min(entry.frequency_min for entry in block),
                frequency_max=max(entry.frequency_max for entry in block),
                sigma=0.5 * (max(entry.frequency_max for entry in block) - min(entry.frequency_min for entry in block)),
                reference_period_kyr=middle.reference_period_kyr,
                source=middle.source,
            )
        )

    return MilankovitchInterval(age_start_ma=start, age_end_ma=end, entries=tuple(merged))


def match_period_to_orbit(
    observed_period_kyr: float,
    age_ma: float,
    *,
    tolerance: float = 0.0,
    uncertainty: float = 1.0,
    half_window_kyr: float = DEFAULT_HALF_WINDOW_KYR,
) -> PeriodMatch:
    """把一个实测周期匹配到最可能的米兰科维奇轨道。

    匹配规则: 先看实测周期落在哪条轨道的频率窗口内 (窗口按 ``tolerance``
    放宽); 若有命中, 取相对偏差最小的那条; 若一条都没命中, 仍返回相对偏差
    最小的那条, 但 ``in_window`` 为 False。

    Parameters:
        observed_period_kyr: 实测周期 (kyr), 必须为正。
        age_ma: 记录所属的年龄 (Ma)。
        tolerance: 额外的相对容差 (>=0), 加在窗口的两侧。
        uncertainty: 频率区间半宽的相对缩放因子 (>0)。
        half_window_kyr: 数周期的分析窗口半宽 (kyr)。

    Returns:
        :class:`PeriodMatch`。

    Raises:
        ValidationError: 若实测周期非法 (非正、非有限) 或年龄非法。
    """
    observed = float(observed_period_kyr)
    if not np.isfinite(observed) or observed <= 0.0:
        raise ValidationError(
            _("observed_period_kyr must be a finite positive number"),
            details={"value": observed_period_kyr},
        )
    slack = float(tolerance)
    if not np.isfinite(slack) or slack < 0.0:
        raise ValidationError(_("tolerance must be a finite non-negative number"), details={"value": tolerance})

    entries = _interval_frequencies(
        age_ma,
        load_astronomical_solution("la04"),
        uncertainty,
        half_window_kyr,
    )

    scored = []
    for entry in entries:
        deviation = (observed - entry.period_kyr) / entry.period_kyr
        low = entry.period_min_kyr * (1.0 - slack)
        high = entry.period_max_kyr * (1.0 + slack)
        scored.append((entry, deviation, low <= observed <= high))

    hits = [item for item in scored if item[2]]
    pool = hits if hits else scored
    best = min(pool, key=lambda item: abs(item[1]))
    entry, deviation, in_window = best

    alternatives = tuple(sorted(item[0].name for item in scored if item[0].name != entry.name and item[2]))

    return PeriodMatch(
        orbit_name=entry.name,
        family=entry.family,
        order=entry.order,
        age_ma=entry.age_ma,
        observed_period_kyr=observed,
        theoretical_period_kyr=entry.period_kyr,
        period_min_kyr=entry.period_min_kyr,
        period_max_kyr=entry.period_max_kyr,
        deviation=deviation,
        deviation_kyr=observed - entry.period_kyr,
        in_window=in_window,
        alternatives=alternatives,
    )


# =============================================================================
# 贝叶斯线性回归口径
# =============================================================================


def bayesian_frequency_solution(
    observed_periods_kyr: npt.NDArray,
    theoretical_frequencies: npt.NDArray,
    *,
    period_sigmas_kyr: npt.NDArray | None = None,
    seed: int | None = None,
    n_samples: int = 0,
) -> FrequencyRegression:
    """贝叶斯线性回归: 把一组实测周期整体平移到理论频率集合上。

    模型: 每条实测周期的频率 ``f_obs = 1 / P_obs`` 服从
    ``N(f_theory + Δ, σ_f²)``, 其中 ``Δ`` 是所有谐波共用的频率修正量,
    ``σ_f = σ_P / P_theory²``。先验取平坦 (共轭), 因此后验仍是高斯:

    * 后验均值 ``Δ̂ = Σ w_i (f_obs,i - f_theory,i) / Σ w_i``;
    * 后验标准差 ``1 / sqrt(Σ w_i)``。

    这是 Waltham (2015, EPSL) 口径的一个**简化版**: 只解共模平移, 没有层次
    噪声模型, 也没有把每条谐波各自的不确定度边缘化掉。

    Parameters:
        observed_periods_kyr: 实测周期 (kyr)。
        theoretical_frequencies: 与之对应的理论频率 (1/kyr)。
        period_sigmas_kyr: 每个实测周期的 1σ 不确定度 (kyr); 默认按经典值的
            2% 取。
        seed: 后验抽样使用的随机种子; 给了才抽样。
        n_samples: 后验抽样点数 (0 表示不抽样)。

    Returns:
        :class:`FrequencyRegression`。

    Raises:
        ValidationError: 若输入为空、长度不一致或含非法数值。
    """
    observed = np.atleast_1d(np.asarray(observed_periods_kyr, dtype=float))
    theoretical = np.atleast_1d(np.asarray(theoretical_frequencies, dtype=float))
    if observed.size == 0 or theoretical.size == 0:
        raise ValidationError(_("observed_periods_kyr and theoretical_frequencies must not be empty"))
    if observed.size != theoretical.size:
        raise ValidationError(
            _("observed_periods_kyr and theoretical_frequencies must have the same length"),
            details={"observed": int(observed.size), "theoretical": int(theoretical.size)},
        )
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValidationError(_("observed_periods_kyr must contain finite positive numbers"))
    if not np.all(np.isfinite(theoretical)) or np.any(theoretical <= 0.0):
        raise ValidationError(_("theoretical_frequencies must contain finite positive numbers"))

    if period_sigmas_kyr is None:
        sigmas = 0.02 * observed
    else:
        sigmas = np.atleast_1d(np.asarray(period_sigmas_kyr, dtype=float))
        if sigmas.size != observed.size:
            raise ValidationError(
                _("period_sigmas_kyr must have the same length as observed_periods_kyr"),
                details={"observed": int(observed.size), "sigmas": int(sigmas.size)},
            )
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValidationError(_("period_sigmas_kyr must contain finite positive numbers"))

    observed_frequencies = 1.0 / observed
    # df/dP = -1/P^2, 而 f = 1/P, 所以 σ_f = σ_P / P^2 = σ_P * f^2。
    frequency_sigmas = sigmas * theoretical**2
    weights = 1.0 / frequency_sigmas**2

    offset = float(np.sum(weights * (observed_frequencies - theoretical)) / np.sum(weights))
    offset_sigma = float(1.0 / np.sqrt(np.sum(weights)))
    residuals = observed_frequencies - theoretical - offset

    samples: npt.NDArray | None = None
    if n_samples > 0:
        count = int(n_samples)
        if count <= 0:
            raise ValidationError(_("n_samples must be positive"), details={"value": n_samples})
        generator = np.random.default_rng(seed)
        samples = generator.normal(offset, offset_sigma, size=count)

    return FrequencyRegression(
        frequency_offset=offset,
        frequency_offset_sigma=offset_sigma,
        residuals=residuals,
        weights=weights,
        samples=samples,
    )


# =============================================================================
# 供界面调用的门面
# =============================================================================


@dataclass(frozen=True)
class MilankovitchUncertaintyResult:
    """米兰科维奇频率分析的结果。

    Attributes:
        ages_ma: 请求的年龄轴 (Ma)。
        entries: 各年龄各谐波的理论频率与频率区间, 按"年龄优先"排列 (每个
            年龄一组, 组内按谐波目录的顺序)。
        uncertainty: 频率区间半宽的相对缩放因子。
        half_window_kyr: 数周期的分析窗口半宽 (kyr)。
        solution_name: 使用的轨道解名称。
    """

    ages_ma: npt.NDArray
    entries: tuple[MilankovitchFrequency, ...]
    uncertainty: float
    half_window_kyr: float
    solution_name: str

    def for_age(self, age_ma: float) -> tuple[MilankovitchFrequency, ...]:
        """取出某个年龄上的全部谐波条目。

        Raises:
            ValidationError: 若结果里没有这个年龄。
        """
        matches = tuple(entry for entry in self.entries if entry.age_ma == float(age_ma))
        if not matches:
            raise ValidationError(
                _("No Milankovitch frequencies for the requested age"),
                details={"age_ma": age_ma},
            )
        return matches

    def summary(self) -> str:
        """一行可读摘要。"""
        return _("Milankovitch frequencies for {0} ages ({1:.3f}-{2:.3f} Ma), {3} harmonic windows").format(
            int(self.ages_ma.size),
            float(np.min(self.ages_ma)),
            float(np.max(self.ages_ma)),
            len(self.entries),
        )


class MilankovitchUncertaintyAnalyzer:
    """米兰科维奇频率分析, 供生成的参数表单驱动。

    轨道解本身是一个对象, 表单给不了 (生成的表单只能填数字、字符串与表格
    的某一列); 门面按名字自己加载内置的 La04 解, 只让用户填年龄轴与两个
    标量。
    """

    def analyze(
        self,
        ages: npt.NDArray,
        uncertainty: float = 1.0,
        half_window_kyr: float = DEFAULT_HALF_WINDOW_KYR,
    ) -> MilankovitchUncertaintyResult:
        """算一组年龄上各米兰科维奇轨道的理论频率与频率区间。

        Parameters:
            ages: 年龄轴 (Ma), 一列; 每个年龄都会得到一组谐波结果。
            uncertainty: 频率区间半宽的相对缩放因子 (>0)。
            half_window_kyr: 数周期的分析窗口半宽 (kyr)。

        Returns:
            :class:`MilankovitchUncertaintyResult`。

        Raises:
            ValidationError: 若年龄非法 (空、含 NaN、为负或超出解的范围) 或
                参数非法。
        """
        scale = 1.0 if uncertainty is None else float(uncertainty)
        window = DEFAULT_HALF_WINDOW_KYR if half_window_kyr is None else float(half_window_kyr)

        solution = load_astronomical_solution("la04")
        entries = _interval_frequencies(
            ages,
            solution,
            scale,
            window,
        )
        return MilankovitchUncertaintyResult(
            ages_ma=np.atleast_1d(np.asarray(ages, dtype=float)),
            entries=entries,
            uncertainty=scale,
            half_window_kyr=window,
            solution_name=solution.name,
        )
