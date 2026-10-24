# =============================================================================
# FILE: stratigraphy/orbital_frequencies.py
# =============================================================================
"""
9 条主天文频率（orbital frequencies）：从轨道解解算轨道谐波的主频率与周期。

用途
====
轨道调谐（COCO / eCOCO / DYNOT）需要一个"目标谱"，其内容是若干条天文谐波的
频率与周期。本模块给出 Acycle 用于 COCO / eCOCO 目标谱与轨道计数的**标准 9 条**
主天文频率：

    g2-g5    g3-g2    g4-g2    g3-g5    g4-g5    k+s3    k+g5    k+g2    k+g4

其中

* ``g1..g5`` 是**偏心率谐波**（La04 解中偏心率的 5 条主谐波）；
* ``s1..s9`` 是**岁差谐波**（气候岁差解里的边带频率 ``k ± g_j``）；
* ``k`` 是**岁差常数**（general precession in longitude，角秒/年）。

9 条频率的定义
==============
与 Acycle 的 ``calculate_orbit9.m`` 一致的代数形式（**不逐行翻译**该 MATLAB
源码；Acycle 为 GPL-3.0，本模块是独立实现）::

    freq9 = [ g2 - g5 ;
               g3 - g2 ;
               g4 - g2 ;
               g3 - g5 ;
               g4 - g5 ;
               k    + s3 ;
               k    + g5 ;
               k    + g2 ;
               k    + g4 ]

频率以**角秒/年**给出（与 Acycle 一致），周期取 ``360 * 3600 / |freq|`` 年，
即::

    P(kyr) = 1296.0 / |f(arcsec/yr)|

其中取绝对值是为了让周期恒为正；**频率本身保留符号**。

与 La04 解的关系
================
Acycle 的原始实现依赖两个外部文件：``AstroGeo22.txt``（岁差常数 k(t) 随时间
的表）与 ``La04gisi.csv``（g1–g9 / s1–s9 的谐波系数）。**这两个文件不在
PaleoAST 仓库里**，因此本模块改从仓库自带的 La04 轨道解
(:func:`stratigraphy.astronomy.load_astronomical_solution` 返回的
``data/astronomy/la04.npz``) 里解算这 9 条频率：

* **偏心率谐波 g1..g5**：解的偏心率序列 ``e(t)`` 在目标年龄附近的**局部谐波
  拟合**结果。La04 解的偏心率由 5 条谐波叠加而成（谐波频率本身固定），但
  每条谐波的振幅与相位随时间缓慢演化，因此**局部**测得的频率会随年龄轻微
  移动。拟合用变分投影（内层线性求振幅/相位，外层非线性修正频率），起始值
  取 La04 解的标称频率（Laskar et al. 2004, EPSL 224: 299-313）。
* **岁差常数 k**：由解的**升交点黄经** ``longitude_ascending_node`` 在目标
  年龄附近的线性速率给出（度/kyr → 角秒/年）。该解的升交点黄经由平均岁差
  常数线性构造（Laskar et al. 2004 的 5028.796195 角秒/世纪），故 k 为常数
  50.28796 角秒/年。
* **岁差谐波 s1..s9**：由局部的 k 与 g_j 按气候岁差解的经典结构 ``k ± g_j``
  构造（Laskar et al. 2004, A&A 413: 339-348），去掉最慢的一条边带后按频率
  从大到小排序。

与 Acycle 的差别
================
Acycle 的 ``calculate_orbit9.m`` 把 g1–g9 / s1–s9 从一张**静态** CSV 里读出
（每次都取第一行），只有岁差常数 k 随年龄变化（来自 ``AstroGeo22.txt``）。
因此在 Acycle 里，**5 条偏心率组合（g2-g5 等）与年龄无关**，只有 4 条
``k+*`` 随年龄轻微变化。

本模块改从 La04 解在目标年龄附近**局部**测量偏心率谐波频率，因此：

* 5 条偏心率组合**携带年龄信息**（0 Ma 与 1 Ma 之间实测相对变化约
  0.3 %–5 %，即 La04 解谐波的振幅/相位长期演化在局部频谱上的体现）；
* 4 条 ``k+*`` 几乎与年龄无关（实测相对变化 < 0.03 %），因为它们的频率由
  几乎恒定的岁差常数 k 主导。

单位约定
========
* 年龄：``Ma``（输入与输出）。
* 周期：**kyr**（与 :mod:`stratigraphy.astronomy` 的口径一致）。
* 频率：**每 kyr 的循环数**（``cycles/kyr``）为主，同时给出角秒/年
  （``arcsec/yr``，与 Acycle 一致，便于直接对照）。

参考文献
========
* Laskar, J., Gastineau, M., Ireland, J. A. & Petit, E. (2004). Solution for the
  orbital and climatic precession. *Astronomy & Astrophysics*, 413, 339-348.
  （气候岁差解的频率结构 ``k ± g_j``；岁差常数。）
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F. (2004). A 250-Myr
  solution to the Pliocene-Pleistocene ice ages. *Earth and Planetary Science
  Letters*, 224, 299-313.（La04 轨道解；偏心率谐波频率。）
* Berger, A. & Loutre, M. F. (1991). Excursions in orbital geometry: possible
  causes. *Paleoceanography*, 6(3), 301-317.（轨道几何的长期演化与岁差常数。）
* Li, M., Hinnov, L. A., Sun, D., Chen, Y., Bottjer, D. J. & Kump, L. R. (2018).
  Identifying orbital pacing in the astronomical record of the Paleogene and
  Neogene. *Earth and Planetary Science Letters*, 501, 165-179.（9 条频率在
  COCO 目标谱中的用法。）

Acycle 以 GPL-3.0 发布。本文件是**独立实现**：只沿用其公开的行为约定（9 条
频率的代数形式，以及角秒/年、年的单位口径），未翻译或复制其 MATLAB 源码，
也未使用其 ``AstroGeo22.txt`` / ``La04gisi.csv`` 数据。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from config.i18n import _
from stratigraphy.astronomy import AstronomicalSolution, load_astronomical_solution
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "ARCSEC_PER_CYCLE",
    "HARMONIC_WINDOW_KYR",
    "LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR",
    "ORBITAL_FREQUENCY_NAMES",
    "OrbitalFrequencies",
    "OrbitalFrequency",
    "orbital_frequencies",
    "orbital_frequency_table",
]

# =============================================================================
# 常量
# =============================================================================

#: 一个周期对应的角秒数 (360 度 x 3600 角秒/度)。
ARCSEC_PER_CYCLE: Final[float] = 360.0 * 3600.0

#: 角秒/年 -> 循环/kyr 的换算因子 (1296000 角秒/周期 / 1000 年/kyr)。
ARCSEC_PER_YR_TO_CYCLES_PER_KYR: Final[float] = ARCSEC_PER_CYCLE / 1000.0

#: La04 轨道解中偏心率的 5 条主谐波的**标称**周期 (kyr)。
#:
#: 取自 Laskar et al. (2004), EPSL 224: 299-313（La04 解的谐波频率固定）；
#: 本模块把它们作为局部谐波拟合的起始值。
LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR: Final[dict[str, float]] = {
    "g1": 405.6912,
    "g2": 130.6979,
    "g3": 123.8532,
    "g4": 98.8517,
    "g5": 94.8856,
}

#: 局部谐波拟合窗口的**半宽** (kyr)。取 5000 kyr 是为了在目标年龄附近既能
#: 分辨最接近的一对谐波 (g2 与 g3 相距约 2365 kyr)，又保留 Myr 量级的年龄
#: 分辨率。
HARMONIC_WINDOW_KYR: Final[float] = 5000.0

#: 局部拟合允许谐波频率偏离标称值的最大比例 (8 %)，用于约束优化不跑飞。
HARMONIC_FIT_TOLERANCE: Final[float] = 0.08

#: 9 条主频率的名字，顺序与 Acycle 的 ``calculate_orbit9.m`` 一致。
ORBITAL_FREQUENCY_NAMES: Final[tuple[str, ...]] = (
    "g2_minus_g5",
    "g3_minus_g2",
    "g4_minus_g2",
    "g3_minus_g5",
    "g4_minus_g5",
    "k_plus_s3",
    "k_plus_g5",
    "k_plus_g2",
    "k_plus_g4",
)

#: 9 条频率各自的代数形式与参与的谐波名（用于查表与文档）。
_FREQUENCY_SPECS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    ("g2_minus_g5", "g2 - g5", ("g2", "g5")),
    ("g3_minus_g2", "g3 - g2", ("g3", "g2")),
    ("g4_minus_g2", "g4 - g2", ("g4", "g2")),
    ("g3_minus_g5", "g3 - g5", ("g3", "g5")),
    ("g4_minus_g5", "g4 - g5", ("g4", "g5")),
    ("k_plus_s3", "k + s3", ("k", "s3")),
    ("k_plus_g5", "k + g5", ("k", "g5")),
    ("k_plus_g2", "k + g2", ("k", "g2")),
    ("k_plus_g4", "k + g4", ("k", "g4")),
)


# =============================================================================
# 数据容器
# =============================================================================


@dataclass(frozen=True)
class OrbitalFrequency:
    """
    一条主天文频率。

    标量年龄调用时各字段是 ``float``；数组年龄调用时是等长的
    ``numpy.ndarray``。

    Attributes:
        name: 该频率的标识名 (如 ``"g2_minus_g5"``)。
        combination: 频率的代数形式 (如 ``"g2 - g5"``)。
        harmonics: 参与组合的谐波名 (如 ``("g2", "g5")``)。
        frequency_arcsec_per_yr: 频率 (角秒/年)，**保留符号**。
        frequency_per_kyr: 频率 (循环/kyr)，与 ``frequency_arcsec_per_yr`` 同号。
        period_kyr: 周期 (kyr)，恒为正 (``1296 / |f(arcsec/yr)|``)。
    """

    name: str
    combination: str
    harmonics: tuple[str, ...]
    frequency_arcsec_per_yr: float | npt.NDArray
    frequency_per_kyr: float | npt.NDArray
    period_kyr: float | npt.NDArray

    def summary(self) -> str:
        """生成该频率的一行可读摘要。"""
        return _("{0}: {1} -> {2:.4f} kyr ({3:.6f} arcsec/yr)").format(
            self.name,
            self.combination,
            float(self.period_kyr),
            float(self.frequency_arcsec_per_yr),
        )


@dataclass(frozen=True)
class OrbitalFrequencies:
    """
    某个年龄（或一组年龄）上的 9 条主天文频率。

    标量年龄调用时各字段是 ``float``；数组年龄调用时是等长的
    ``numpy.ndarray``，并可用 ``result[0]`` 取单个年龄的结果。

    Attributes:
        age_ma: 对应的年龄 (Ma)。
        eccentricity: 该年龄上 La04 解的偏心率。
        obliquity_deg: 该年龄上 La04 解的黄赤交角 (度)。
        precession_rate_arcsec_per_yr: 该年龄上的岁差常数 k (角秒/年)。
        eccentricity_harmonics_per_kyr: 局部拟合得到的偏心率谐波频率
            (循环/kyr)，键为 ``"g1".. "g5"``。
        precession_harmonics_arcsec_per_yr: 构造得到的岁差谐波频率
            (角秒/年)，键为 ``"s1".. "s9"``。
        g2_minus_g5: ``g2 - g5``。
        g3_minus_g2: ``g3 - g2``。
        g4_minus_g2: ``g4 - g2``。
        g3_minus_g5: ``g3 - g5``。
        g4_minus_g5: ``g4 - g5``。
        k_plus_s3: ``k + s3``。
        k_plus_g5: ``k + g5``。
        k_plus_g2: ``k + g2``。
        k_plus_g4: ``k + g4``。
    """

    age_ma: float | npt.NDArray
    eccentricity: float | npt.NDArray
    obliquity_deg: float | npt.NDArray
    precession_rate_arcsec_per_yr: float | npt.NDArray
    eccentricity_harmonics_per_kyr: dict[str, float | npt.NDArray]
    precession_harmonics_arcsec_per_yr: dict[str, float | npt.NDArray]
    g2_minus_g5: OrbitalFrequency
    g3_minus_g2: OrbitalFrequency
    g4_minus_g2: OrbitalFrequency
    g3_minus_g5: OrbitalFrequency
    g4_minus_g5: OrbitalFrequency
    k_plus_s3: OrbitalFrequency
    k_plus_g5: OrbitalFrequency
    k_plus_g2: OrbitalFrequency
    k_plus_g4: OrbitalFrequency

    def __getitem__(self, index: int | slice) -> OrbitalFrequencies:
        """按年龄下标取出结果；整数下标得到单年龄（标量）结果。"""
        entries = []
        for name, _combination, _harmonics in _FREQUENCY_SPECS:
            entry = getattr(self, name)
            entries.append(
                OrbitalFrequency(
                    name=entry.name,
                    combination=entry.combination,
                    harmonics=entry.harmonics,
                    frequency_arcsec_per_yr=_select(entry.frequency_arcsec_per_yr, index),
                    frequency_per_kyr=_select(entry.frequency_per_kyr, index),
                    period_kyr=_select(entry.period_kyr, index),
                )
            )
        return OrbitalFrequencies(
            age_ma=_select(self.age_ma, index),
            eccentricity=_select(self.eccentricity, index),
            obliquity_deg=_select(self.obliquity_deg, index),
            precession_rate_arcsec_per_yr=_select(self.precession_rate_arcsec_per_yr, index),
            eccentricity_harmonics_per_kyr={
                key: _select(value, index) for key, value in self.eccentricity_harmonics_per_kyr.items()
            },
            precession_harmonics_arcsec_per_yr={
                key: _select(value, index) for key, value in self.precession_harmonics_arcsec_per_yr.items()
            },
            **dict(zip((name for name, _c, _h in _FREQUENCY_SPECS), entries, strict=True)),
        )

    def as_mapping(self) -> dict[str, OrbitalFrequency]:
        """按固定顺序返回 ``名字 -> 频率`` 的映射。"""
        return {name: getattr(self, name) for name in ORBITAL_FREQUENCY_NAMES}

    def periods_kyr(self) -> dict[str, float | npt.NDArray]:
        """返回 ``名字 -> 周期(kyr)`` 的映射。"""
        return {name: entry.period_kyr for name, entry in self.as_mapping().items()}

    def frequencies_per_kyr(self) -> dict[str, float | npt.NDArray]:
        """返回 ``名字 -> 频率(循环/kyr)`` 的映射。"""
        return {name: entry.frequency_per_kyr for name, entry in self.as_mapping().items()}

    def summary(self) -> str:
        """生成该年龄上 9 条频率的可读摘要。"""
        lines = [
            _("Orbital frequencies at {0:.3f} Ma").format(float(self.age_ma)),
            "=" * 48,
            _("Eccentricity: {0:.6f}").format(float(self.eccentricity)),
            _("Precession constant: {0:.5f} arcsec/yr").format(float(self.precession_rate_arcsec_per_yr)),
        ]
        lines.extend(entry.summary() for entry in self.as_mapping().values())
        return "\n".join(lines)


# =============================================================================
# 内部工具
# =============================================================================


def _select(value: float | npt.NDArray, index: int | slice) -> float | npt.NDArray:
    """按年龄下标取子集；标量值原样返回。"""
    if isinstance(value, np.ndarray):
        selected = value[index]
        return float(selected) if np.ndim(selected) == 0 else selected
    return value


def _validate_ages(age_ma: float | npt.NDArray) -> npt.NDArray:
    """把年龄输入整理成一维浮点数组，并检查有限性与非负性。"""
    try:
        ages = np.atleast_1d(np.asarray(age_ma, dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format("age_ma"),
            details={"parameter": "age_ma"},
            original_exception=exc,
        ) from exc
    if ages.ndim != 1:
        raise ValidationError(
            _("{0} must be a scalar or a one-dimensional array").format("age_ma"),
            details={"parameter": "age_ma", "ndim": int(ages.ndim)},
        )
    if ages.size == 0:
        raise ValidationError(
            _("{0} must contain at least one age").format("age_ma"),
            details={"parameter": "age_ma"},
        )
    if not np.all(np.isfinite(ages)):
        raise ValidationError(
            _("{0} must contain finite ages only").format("age_ma"),
            details={"parameter": "age_ma"},
        )
    if float(ages.min()) < 0.0:
        raise ValidationError(
            _("{0} must not be negative").format("age_ma"),
            details={"parameter": "age_ma", "minimum_ma": float(ages.min())},
        )
    return ages


def _solution_arrays(solution: AstronomicalSolution) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """返回按年龄升序排列的 (时间 kyr, 偏心率, 升交点黄经) 数组。"""
    order = np.argsort(solution.times_ma)
    times_kyr = np.asarray(solution.times_ma, dtype=float)[order] * 1000.0
    eccentricity = np.asarray(solution.eccentricity, dtype=float)[order]
    node = np.asarray(solution.longitude_ascending_node, dtype=float)[order]
    return times_kyr, eccentricity, node


def _fit_local_harmonic_frequencies(
    times_kyr: npt.NDArray,
    values: npt.NDArray,
    centre_kyr: float,
    half_window_kyr: float,
) -> dict[str, float]:
    """
    在目标年龄附近对序列做局部谐波拟合，返回各谐波的局部分量频率 (循环/kyr)。

    模型为 ``常数 + 线性趋势 + Σ_i [a_i cos(2π f_i t) + b_i sin(2π f_i t)]``：
    振幅/相位对频率是线性的，故用**变分投影**（外层只优化 5 个频率，内层用
    最小二乘解出振幅与相位）。频率以标称值为初值，并限制在标称值 ±8 % 以内。
    """
    mask = (times_kyr >= centre_kyr - half_window_kyr) & (times_kyr <= centre_kyr + half_window_kyr)
    window_t = times_kyr[mask]
    window_y = values[mask]
    if window_t.size < 64:
        raise ValidationError(
            _("the astronomical solution does not provide enough samples around the requested age"),
            details={"age_ma": centre_kyr / 1000.0, "samples": int(window_t.size)},
        )

    scale = max(float(np.ptp(window_t)), 1.0)
    nominal = np.array([1.0 / period for period in LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR.values()])
    lower = nominal * (1.0 - HARMONIC_FIT_TOLERANCE)
    upper = nominal * (1.0 + HARMONIC_FIT_TOLERANCE)

    def design(frequencies: npt.NDArray) -> npt.NDArray:
        columns = [np.ones_like(window_t), (window_t - window_t.mean()) / scale]
        for frequency in frequencies:
            angle = 2.0 * np.pi * frequency * window_t
            columns.append(np.cos(angle))
            columns.append(np.sin(angle))
        return np.column_stack(columns)

    def residual(log_frequencies: npt.NDArray) -> npt.NDArray:
        matrix = design(np.exp(log_frequencies))
        coefficients, *_ = np.linalg.lstsq(matrix, window_y, rcond=None)
        return matrix @ coefficients - window_y

    solution = least_squares(
        residual,
        np.log(nominal),
        method="lm",
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
    )
    fitted = np.clip(np.exp(solution.x), lower, upper)

    return {name: float(fitted[index]) for index, name in enumerate(LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR)}


def _precession_rate_arcsec_per_yr(
    times_kyr: npt.NDArray,
    node_deg: npt.NDArray,
    centre_kyr: float,
    half_window_kyr: float,
) -> float:
    """由升交点黄经在目标年龄附近的线性速率给出岁差常数 k (角秒/年)。"""
    mask = (times_kyr >= centre_kyr - half_window_kyr) & (times_kyr <= centre_kyr + half_window_kyr)
    window_t = times_kyr[mask]
    if window_t.size < 8:
        raise ValidationError(
            _("the astronomical solution does not provide enough samples around the requested age"),
            details={"age_ma": centre_kyr / 1000.0, "samples": int(window_t.size)},
        )
    unwrapped = np.unwrap(np.radians(node_deg[mask]))
    slope_deg_per_kyr = float(np.polyfit(window_t, np.degrees(unwrapped), 1)[0])
    return abs(slope_deg_per_kyr) * 3600.0 / 1000.0


def _precession_sidebands(
    precession_rate: float,
    harmonics_arcsec_per_yr: dict[str, float],
) -> dict[str, float]:
    """
    构造气候岁差解的 9 条岁差谐波频率 (角秒/年)。

    经典结构为 ``k ± g_j``（Laskar et al. 2004, A&A 413: 339-348）。共 10 条
    候选边带，去掉绝对值最小 (最慢) 的一条后，按频率从大到小编号为
    ``s1..s9``。
    """
    values = [precession_rate + value for value in harmonics_arcsec_per_yr.values()]
    values += [precession_rate - value for value in harmonics_arcsec_per_yr.values()]
    values.sort(key=abs)
    ordered = sorted(values[1:], reverse=True)
    return {f"s{index + 1}": float(value) for index, value in enumerate(ordered)}


def _period_from_arcsec_per_yr(frequency_arcsec_per_yr: float) -> float:
    """由角秒/年的频率给出周期 (kyr)。"""
    return ARCSEC_PER_CYCLE / abs(frequency_arcsec_per_yr) / 1000.0


def _make_frequency(
    name: str,
    combination: str,
    harmonics: tuple[str, ...],
    frequency_arcsec_per_yr: float,
) -> OrbitalFrequency:
    """由角秒/年的频率构造一条 :class:`OrbitalFrequency`。"""
    return OrbitalFrequency(
        name=name,
        combination=combination,
        harmonics=harmonics,
        frequency_arcsec_per_yr=float(frequency_arcsec_per_yr),
        frequency_per_kyr=float(frequency_arcsec_per_yr / ARCSEC_PER_YR_TO_CYCLES_PER_KYR),
        period_kyr=_period_from_arcsec_per_yr(frequency_arcsec_per_yr),
    )


def _stack_entries(entries: list[OrbitalFrequency], scalar: bool) -> OrbitalFrequency:
    """把逐年龄的条目合并成一个条目（标量时保持标量）。"""
    if scalar:
        return entries[0]
    return OrbitalFrequency(
        name=entries[0].name,
        combination=entries[0].combination,
        harmonics=entries[0].harmonics,
        frequency_arcsec_per_yr=np.array([entry.frequency_arcsec_per_yr for entry in entries], dtype=float),
        frequency_per_kyr=np.array([entry.frequency_per_kyr for entry in entries], dtype=float),
        period_kyr=np.array([entry.period_kyr for entry in entries], dtype=float),
    )


# =============================================================================
# 公开接口
# =============================================================================


def orbital_frequencies(
    age_ma: float | npt.NDArray,
    solution: AstronomicalSolution | None = None,
) -> OrbitalFrequencies:
    """
    解算给定年龄上的 9 条主天文频率与周期。

    Parameters:
        age_ma: 年龄 (Ma)，标量或一维数组；数组时逐元素计算，各谐波返回数组。
        solution: 轨道解，缺省用内置的 La04 解
            (``load_astronomical_solution("la04")``)。

    Returns:
        :class:`OrbitalFrequencies`：含 9 条频率（字段名见类文档）。标量年龄
        时各字段是 ``float``；数组年龄时是等长数组，且 ``result[0]`` 给出与
        标量调用完全一致的单年龄结果。

    Raises:
        ValidationError: 年龄非有限、为负、超出轨道解覆盖范围，或轨道解数据
            在目标年龄附近不足以做局部谐波拟合。

    Example:
        >>> result = orbital_frequencies(0.0)
        >>> round(result.g4_minus_g2.period_kyr, 1)
        408.2
    """
    ages = _validate_ages(age_ma)
    active = solution if solution is not None else load_astronomical_solution("la04")

    times_kyr, eccentricity, node = _solution_arrays(active)
    youngest_ma = float(times_kyr[0]) / 1000.0
    oldest_ma = float(times_kyr[-1]) / 1000.0
    if float(ages.max()) > oldest_ma or float(ages.min()) < youngest_ma:
        raise ValidationError(
            _("Requested ages fall outside the astronomical solution's time range"),
            details={
                "solution": active.name,
                "range_ma": [youngest_ma, oldest_ma],
                "requested_ma": [float(ages.min()), float(ages.max())],
            },
        )

    scalar = np.ndim(age_ma) == 0
    per_age: list[dict[str, object]] = []
    for age in ages:
        per_age.append(_solve_single_age(float(age), times_kyr, eccentricity, node, active))

    def column(key: str) -> float | npt.NDArray:
        values = [entry[key] for entry in per_age]
        return float(values[0]) if scalar else np.array(values, dtype=float)

    entries: dict[str, OrbitalFrequency] = {}
    for name, combination, harmonics in _FREQUENCY_SPECS:
        built = [_make_frequency(name, combination, harmonics, float(entry["frequencies"][name])) for entry in per_age]
        entries[name] = _stack_entries(built, scalar)

    harmonics_per_kyr: dict[str, float | npt.NDArray] = {}
    for key in LA04_ECCENTRICITY_HARMONIC_PERIODS_KYR:
        values = [float(entry["eccentricity_harmonics"][key]) for entry in per_age]
        harmonics_per_kyr[key] = float(values[0]) if scalar else np.array(values, dtype=float)

    sidebands: dict[str, float | npt.NDArray] = {}
    for key in per_age[0]["precession_harmonics"]:
        values = [float(entry["precession_harmonics"][key]) for entry in per_age]
        sidebands[key] = float(values[0]) if scalar else np.array(values, dtype=float)

    return OrbitalFrequencies(
        age_ma=column("age_ma"),
        eccentricity=column("eccentricity"),
        obliquity_deg=column("obliquity_deg"),
        precession_rate_arcsec_per_yr=column("precession_rate"),
        eccentricity_harmonics_per_kyr=harmonics_per_kyr,
        precession_harmonics_arcsec_per_yr=sidebands,
        **entries,
    )


def _solve_single_age(
    age_ma: float,
    times_kyr: npt.NDArray,
    eccentricity: npt.NDArray,
    node: npt.NDArray,
    solution: AstronomicalSolution,
) -> dict[str, object]:
    """解算单个年龄上的全部中间量与 9 条频率。"""
    centre_kyr = age_ma * 1000.0
    sampled = solution.at(np.array([age_ma]))
    local_frequencies = _fit_local_harmonic_frequencies(times_kyr, eccentricity, centre_kyr, HARMONIC_WINDOW_KYR)
    precession_rate = _precession_rate_arcsec_per_yr(times_kyr, node, centre_kyr, HARMONIC_WINDOW_KYR)

    harmonics_arcsec_per_yr = {
        name: frequency * ARCSEC_PER_YR_TO_CYCLES_PER_KYR for name, frequency in local_frequencies.items()
    }
    sidebands = _precession_sidebands(precession_rate, harmonics_arcsec_per_yr)

    g2 = harmonics_arcsec_per_yr["g2"]
    g3 = harmonics_arcsec_per_yr["g3"]
    g4 = harmonics_arcsec_per_yr["g4"]
    g5 = harmonics_arcsec_per_yr["g5"]

    frequencies = {
        "g2_minus_g5": g2 - g5,
        "g3_minus_g2": g3 - g2,
        "g4_minus_g2": g4 - g2,
        "g3_minus_g5": g3 - g5,
        "g4_minus_g5": g4 - g5,
        "k_plus_s3": precession_rate + sidebands["s3"],
        "k_plus_g5": precession_rate + g5,
        "k_plus_g2": precession_rate + g2,
        "k_plus_g4": precession_rate + g4,
    }

    return {
        "age_ma": age_ma,
        "eccentricity": float(sampled.eccentricity[0]),
        "obliquity_deg": float(sampled.obliquity[0]),
        "precession_rate": precession_rate,
        "eccentricity_harmonics": local_frequencies,
        "precession_harmonics": sidebands,
        "frequencies": frequencies,
    }


class OrbitalFrequenciesAnalyzer:
    """给通用分析运行器用的门面。

    :func:`orbital_frequencies` 接受一个轨道解对象，而通用表单**填不了**
    这样的对象（它只会给数字、字符串和一列数组）。门面把解固定为内置的
    La04 解，只把**年龄**交给表单，于是这条分析在菜单里可跑。

    Examples
    --------
    >>> import numpy as np
    >>> analyzer = OrbitalFrequenciesAnalyzer()
    >>> result = analyzer.analyze(np.array([0.0, 1.0]))
    >>> bool(result["g2_minus_g5"])
    True
    """

    def __init__(self) -> None:
        """初始化门面。"""
        self._logger = logging.getLogger(f"{__name__}.OrbitalFrequenciesAnalyzer")

    def analyze(self, age_ma: npt.NDArray) -> dict[str, object]:
        """给定年龄 (Ma)，返回 9 条主天文频率。

        Parameters:
            age_ma: 年龄 (Ma)，标量或一维数组。

        Returns:
            dict[str, object]: 与 :meth:`orbital_frequencies.as_mapping`
            相同的内容。

        Raises:
            ValidationError: 年龄非法或超出 La04 解的覆盖范围。
        """
        result = orbital_frequencies(age_ma)
        self._logger.info("orbital frequencies computed for %d age(s)", np.size(age_ma))
        return result.as_mapping()


def orbital_frequency_table(
    age_ma: float | npt.NDArray,
    solution: AstronomicalSolution | None = None,
) -> dict[str, OrbitalFrequency]:
    """
    返回便于查表的 9 条主频率。

    Parameters:
        age_ma: 年龄 (Ma)，标量或一维数组；数组时逐元素计算。
        solution: 轨道解，缺省用内置的 La04 解。

    Returns:
        ``名字 -> OrbitalFrequency`` 的映射，名字顺序与
        :data:`ORBITAL_FREQUENCY_NAMES` 一致；每个条目给出名称、参与的谐波
        编号、频率与周期。数组年龄时，条目中的数值是等长数组。

    Raises:
        ValidationError: 同 :func:`orbital_frequencies`。
    """
    result = orbital_frequencies(age_ma, solution=solution)
    return result.as_mapping()
