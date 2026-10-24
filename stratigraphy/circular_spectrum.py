# =============================================================================
# FILE: stratigraphy/circular_spectrum.py
# =============================================================================
"""
圆周谱（circular spectrum）：角度型数据的圆谐分析。

用途
====
方向、方位角、古地磁极性、构造走向这类**角度型**变量不是线性变量：0° 与
360° 相邻，350° 与 10° 只差 20° 而不是 340°。把角度直接当线性量做傅里叶
变换（或直接取算术平均）会得出完全错误的结论——例如一组均匀分布在 350° 与
10° 附近的方位角，其线性均值接近 180°，而真实的主方向是 0°。

本模块提供两种互为补充的分析：

* :func:`circular_spectrum` —— **圆谐分析**。把角度展开成圆上的 Fourier 级数
  ``C_k = mean(exp(i k theta))``，取一阶、二阶圆谐的合成长度与方向。这是
  Fisher (1993) 与 Mardia & Jupp (2000) 的标准做法，也是"这批方位角的主
  方向是什么、集中程度有多高"的答案。
* :func:`circular_strength_spectrum` —— **周期栅格上的合成向量长度谱**
  （Acycle 的 "circular spectrum" 子功能的原义）：对一组事件坐标，逐个周期
  把坐标折回 ``[0, p)``，计算该折回坐标的合成长度 ``R``，看哪一个周期能把
  这批事件"收拢"在一起。

圆谐的定义
==========
角度 ``theta_j``（弧度）写成复指数形式::

    C_k = (1/N) * sum_j exp(i k theta_j)          k = 1, 2, ...

``C_k`` 是第 ``k`` 阶圆谐的系数，写成 ``R_k exp(i phi_k)``：

* **合成长度** ``R_k = |C_k|``，取值 ``0..1``。``R_k`` 越大，第 ``k`` 阶谐波
  在这批角度里占的比重越大。``R_1`` 是一阶（单峰）集中度，``R_2`` 是二阶
  （双峰/轴向）集中度。
* **主方向** ``phi_k = atan2(Im C_k, Re C_k)``，用 ``atan2`` 而不是
  ``acos``：``acos`` 只能给出 ``0..180°``，会把方向的正负号丢掉。
* **谐波功率** ``power_k = R_k^2``，即该阶谐波解释的圆方差比例
  （Mardia & Jupp 2000, §4.4）。

本模块的对外口径：

* ``resultant_length`` = **R_1**（一阶合成长度，圆周集中度）；
* ``mean_direction`` = **一阶谐波的主方向** ``phi_1``（度，``0..360``）；
* ``power`` = 各阶谐波的 ``R_k^2``；``frequencies`` = 谐波数 ``k = 1..n``。

为什么必须用 ``cos/sin``：角度是循环变量，圆谐的定义本身就建立在
``exp(i k theta)`` 上；用线性傅里叶（对角度直接做 FFT）会把 350° 当成远离
10° 的点，第二阶谐波会彻底失真。

参考文献
========
* Fisher, N. I. (1993). Statistical Analysis of Circular Data. Cambridge
  University Press.（圆谐、合成长度与主方向的经典定义）
* Mardia, K. V. & Jupp, P. E. (2000). Circular Statistics. Wiley.（圆谐功率、
  二阶谐波与双峰/轴向判别）
* Zar, J. H. (2010). Biostatistical Analysis. 5th ed. Prentice Hall.
  （一阶/二阶圆谐与方向性统计）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "circular spectrum" 子功能）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考它公开的行为约定
（周期栅格上折回坐标、求平均合成向量长度、折回相位坐标），未翻译或复制其
MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 圆谐分析至少需要的角度个数。
MIN_ANGLES = 2

#: 允许的最多谐波数。
MAX_HARMONICS = 64

#: 合成向量长度谱至少需要的周期个数。
MIN_PERIODS = 2


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class CircularSpectrumResult:
    """
    圆谐分析的结果。

    Attributes:
        frequencies: 谐波数 ``k = 1, 2, ..., n_harmonics``（整数周期）。
        power: 各阶谐波的功率 ``R_k^2``，即该阶谐波解释的圆方差比例。
        resultant_length: 一阶合成长度 ``R_1``（``0..1``），圆周集中度。
        mean_direction: 一阶谐波的主方向（度，``0..360``）。
    """

    frequencies: npt.NDArray
    power: npt.NDArray
    resultant_length: float
    mean_direction: float

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Circular spectrum"),
            "=" * 48,
            _("Harmonics: {0}").format(self.frequencies.size),
            _("Resultant length R1: {0}").format(f"{self.resultant_length:.4f}"),
            _("Mean direction: {0} degrees").format(f"{self.mean_direction:.2f}"),
        ]
        if self.frequencies.size >= 2:
            second = float(np.sqrt(self.power[1]))
            lines.append(_("Second harmonic length R2: {0}").format(f"{second:.4f}"))
        return "\n".join(lines)


@dataclass(frozen=True)
class CircularStrengthResult:
    """
    周期栅格上的合成向量长度谱（Acycle "circular spectrum" 的原义）。

    Attributes:
        periods: 周期栅格（坐标单位）。
        strength: 每个周期上的合成长度 ``R``（``0..1``）：把事件坐标按该周期
            折回 ``[0, p)`` 之后的集中程度。
        phase_coordinates: 每个周期上的折回相位坐标，取值 ``[0, p)``。
    """

    periods: npt.NDArray
    strength: npt.NDArray
    phase_coordinates: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            _("Circular strength spectrum"),
            "=" * 48,
            _("Periods: {0}").format(self.periods.size),
        ]
        if self.periods.size:
            peak = int(np.argmax(self.strength))
            lines.append(
                _("Peak strength: {0} at period {1}").format(f"{self.strength[peak]:.4f}", f"{self.periods[peak]:.6g}")
            )
        return "\n".join(lines)


# =============================================================================
# Internal helpers
# =============================================================================


def _as_vector(values: object, name: str, *, minimum_size: int) -> npt.NDArray:
    """把输入转换为一维有限数组，并检查长度下限。"""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if array.ndim != 1:
        raise ValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"parameter": name, "ndim": int(array.ndim)},
        )
    if array.size < minimum_size:
        raise ValidationError(
            _("{0} needs at least {1} points; got {2}").format(name, minimum_size, int(array.size)),
            details={"parameter": name, "length": int(array.size), "minimum": minimum_size},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return array


def _circular_harmonics(radians: npt.NDArray, harmonics: int) -> tuple[npt.NDArray, npt.NDArray]:
    """计算前 ``harmonics`` 阶圆谐系数。

    Returns:
        tuple: ``(合成长度 R_k, 主方向 phi_k（弧度）)`` 两个数组。
    """
    order = np.arange(1, harmonics + 1, dtype=float)
    coefficients = np.mean(np.exp(1j * np.outer(order, radians)), axis=1)
    resultant = np.abs(coefficients)
    directions = np.arctan2(coefficients.imag, coefficients.real)
    return resultant, directions


# =============================================================================
# Public interface
# =============================================================================


def circular_spectrum(
    angles_deg: npt.NDArray,
    n_harmonics: int = 6,
) -> CircularSpectrumResult:
    """角度型数据的圆谐分析（一阶与二阶圆谐）。

    把角度写成圆上的 Fourier 级数 ``C_k = mean(exp(i k theta))``，得到各阶
    谐波的合成长度 ``R_k`` 与主方向 ``phi_k``。**角度是循环变量**，所以
    必须走 ``cos/sin``（等价于复指数）这条路；直接对角度做线性傅里叶或取
    算术平均，在 350°/10° 这种跨零度分布上会给出完全错误的主方向。

    Parameters:
        angles_deg: 角度值（度）。任何有限值都可以，内部按 360° 取模。
        n_harmonics: 计算的谐波阶数，至少 1，缺省 6。

    Returns:
        CircularSpectrumResult: ``frequencies`` 是谐波数 ``k = 1..n_harmonics``，
        ``power`` 是各阶的 ``R_k^2``，``resultant_length`` 是 ``R_1``，
        ``mean_direction`` 是一阶谐波的主方向（度，``0..360``）。

    Raises:
        ValidationError: 角度个数太少、含非有限值，或谐波阶数非法。

    Notes:
        若角度在圆上完全均匀分布，则 ``R_1 -> 0``，此时主方向在数学上没有
        意义，函数返回 ``0.0`` 度。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.circular_spectrum import circular_spectrum
        >>> angles = np.array([88.0, 91.0, 93.0, 89.0, 90.0])
        >>> result = circular_spectrum(angles)
        >>> bool(abs(result.mean_direction - 90.0) < 5.0)
        True
    """
    angles = _as_vector(angles_deg, "angles_deg", minimum_size=MIN_ANGLES)
    harmonics = int(n_harmonics)
    if harmonics != n_harmonics or not (1 <= harmonics <= MAX_HARMONICS):
        raise ValidationError(
            _("n_harmonics must be an integer between 1 and {0}").format(MAX_HARMONICS),
            details={"n_harmonics": n_harmonics},
        )

    radians = np.deg2rad(np.mod(angles, 360.0))
    resultant, directions = _circular_harmonics(radians, harmonics)

    order = np.arange(1, harmonics + 1, dtype=float)
    power = resultant**2
    mean_direction = float(np.mod(np.degrees(directions[0]), 360.0))
    resultant_length = float(resultant[0])

    logger.info(
        "Circular spectrum complete: %d angles, %d harmonics, R1 = %.4f, direction = %.2f degrees",
        int(angles.size),
        harmonics,
        resultant_length,
        mean_direction,
    )
    return CircularSpectrumResult(
        frequencies=order,
        power=power,
        resultant_length=resultant_length,
        mean_direction=mean_direction,
    )


def circular_strength_spectrum(
    event_coordinates: npt.NDArray,
    period_range: tuple[float, float] | None = None,
    n_periods: int = 100,
    grid: str = "linear",
) -> CircularStrengthResult:
    """周期栅格上的合成向量长度谱（Acycle "circular spectrum" 的原义）。

    对每个候选周期 ``p``，把事件坐标折回 ``[0, p)``：

        C = mean(cos(2 * pi * mod(x, p) / p))
        S = mean(sin(2 * pi * mod(x, p) / p))
        strength = hypot(C, S)

    ``strength`` 接近 1 表示这批事件在该周期上"收拢"到相位坐标的同一处，
    即它们可能由同一个周期性的过程产生。折回相位坐标为
    ``p * mod(atan2(S, C), 2 * pi) / (2 * pi)``，取值 ``[0, p)``。

    与 :func:`circular_spectrum` 的分工：那个函数分析的是**角度值本身**的
    集中方向；这个函数分析的是**事件坐标**在某个周期上的相位集中。

    Parameters:
        event_coordinates: 事件坐标（任意单位），会被排序；重复坐标不允许。
        period_range: ``(最短周期, 最长周期)``。缺省时取
            ``(最小间隔 / 2, 坐标跨度 / 2)``。
        n_periods: 周期栅格的点数，至少 2。
        grid: ``"linear"``（等间距）或 ``"log"``（等比）。

    Returns:
        CircularStrengthResult: 周期栅格、每个周期的合成长度与折回相位坐标。

    Raises:
        ValidationError: 事件坐标非法（太少、含非有限值、重复）、周期范围
            非法、周期数太少或栅格模式未知。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.circular_spectrum import circular_strength_spectrum
        >>> events = np.arange(0.0, 100.0, 5.0)
        >>> result = circular_strength_spectrum(events, period_range=(2.0, 20.0))
        >>> float(result.strength.max()) > 0.5
        True
    """
    events = _as_vector(event_coordinates, "event_coordinates", minimum_size=MIN_ANGLES)
    ordered = np.sort(events)
    if np.any(np.diff(ordered) <= 0.0):
        raise ValidationError(
            _("event coordinates must be unique"),
            details={"n_points": int(ordered.size)},
        )

    periods_count = int(n_periods)
    if periods_count != n_periods or periods_count < MIN_PERIODS:
        raise ValidationError(
            _("n_periods must be an integer of at least {0}").format(MIN_PERIODS),
            details={"n_periods": n_periods},
        )
    mode = str(grid).lower()
    if mode not in ("linear", "log"):
        raise ValidationError(
            _("grid must be 'linear' or 'log'"),
            details={"grid": grid},
        )

    spacing = np.diff(ordered)
    span = float(ordered[-1] - ordered[0])
    if period_range is None:
        low = float(np.min(spacing)) / 2.0
        high = span / 2.0
    else:
        try:
            low, high = (float(period_range[0]), float(period_range[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValidationError(
                _("period_range must be a pair of numbers"),
                details={"period_range": repr(period_range)},
                original_exception=exc,
            ) from exc
    if not (np.isfinite(low) and np.isfinite(high)) or not (0.0 < low < high):
        raise ValidationError(
            _("period_range must satisfy 0 < minimum < maximum"),
            details={"period_range": [low, high]},
        )

    if mode == "linear":
        periods = np.linspace(low, high, periods_count)
    else:
        periods = np.logspace(np.log10(low), np.log10(high), periods_count)

    strength = np.zeros(periods.size)
    phase_coordinates = np.zeros(periods.size)
    for index, period in enumerate(periods):
        reduced = np.mod(ordered, period)
        angle = 2.0 * np.pi * reduced / period
        cosine_mean = float(np.mean(np.cos(angle)))
        sine_mean = float(np.mean(np.sin(angle)))
        magnitude = float(np.hypot(cosine_mean, sine_mean))
        strength[index] = min(1.0, magnitude)
        phase = period * float(np.mod(np.arctan2(sine_mean, cosine_mean), 2.0 * np.pi)) / (2.0 * np.pi)
        phase_coordinates[index] = 0.0 if phase >= period else phase

    logger.info(
        "Circular strength spectrum complete: %d events, %d periods, peak strength %.4f",
        int(ordered.size),
        int(periods.size),
        float(strength.max()) if strength.size else float("nan"),
    )
    return CircularStrengthResult(
        periods=periods,
        strength=strength,
        phase_coordinates=phase_coordinates,
    )
