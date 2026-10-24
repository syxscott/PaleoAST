# =============================================================================
# FILE: stratigraphy/inverse_percentile.py
# =============================================================================
"""
逆百分位（inverse percentile）与定义域变换（domain transform）。

本模块实现 Acycle ``code/misc/`` 工具箱里的两件小事，并说明它们之间的关系。

逆百分位
--------
Acycle 的 ``code/misc/invprctile.m``（8.6 KB，作者 Durga Lal Shrestha，
CSIRO）计算的是**样本的"非超越概率"**：给定样本 ``x`` 与一个分位数值
``xq``，返回 ``xq`` 在样本中的非超越概率（百分数）。它内部先把样本排序，
给第 ``i`` 个次序统计量一个**绘图位置**（plotting position）

    pᵢ = (i − p₁) / (n + p₂)

（默认 Hazen：``p₁ = 0.5``、``p₂ = 0``，即 ``pᵢ = (i − 0.5)/n``），再在相邻
次序统计量之间线性插值；样本范围之外分别取 0 % 与 100 %。支持的绘图位置
有 Weibull、Median、APL、Blom、Cunnane、Gringorten、Hazen，外加自定义的
``(p₁, p₂)``。

因此 ``invprctile`` 是"分位数值 → 概率"的映射，其**逆映射**（"概率 → 分位
数值"）正是本模块的主函数 :func:`inverse_percentile`：给定样本与一个非超越
概率，返回对应的分位数值。:func:`non_exceedance_probability` 则按 Acycle 的
方向实现，二者互为精确反函数（绘图位置相同时）。

⚠️ 口径提醒：绘图位置族与 ``numpy.quantile`` 的默认算法（``method="linear"``，
即 ``h = (n−1)q``）**不是同一个约定**，二者的分位数值相差约 ``O(1/n)`` 个
次序统计量。本模块坚持 Acycle 的绘图位置口径，因此
``inverse_percentile`` 与 ``numpy.quantile`` 不会逐位相同，但在大样本上非常
接近。

定义域变换
----------
Acycle 的 ``code/misc/domaintrans.m``（719 B）与 ``invprctile`` **并没有配对
关系**：二者只是同在 ``code/misc/`` 工具箱里的两个独立小工具。``domaintrans``
做的是"定义域变换"——把一列速率（或步长）乘以采样率后累积分，得到新的
定义域坐标：

    x = cumsum(x₁ · t)

其中 ``t`` 可以是常数（采样率）或与 ``x₁`` 等长的向量（逐点采样率）。典型
用途是把"每步的速率"积分成"到第 n 步为止的累计量"（例如由沉积速率积分出
厚度、由速率积分出年龄域），从而在**非均匀采样**下仍能得到一个可比较的
坐标轴。**该变换现由 :func:`stratigraphy.small_tools.domain_transform` 实现**（它同时提供累积分与解析信号两种方式），本模块只保留分位数与超越概率。

Acycle 是 GPL 软件，本模块按上述规格**独立实现**，未翻译其 MATLAB 源码。

参考资料
--------
1. Stedinger, J.R., Vogel, R.M. & Foufoula-Georgiu, E., 1995. Frequency
   analysis of extreme events. In Maidment, D. (ed.) *Handbook of Hydrology*.
   McGraw-Hill.（``invprctile.m`` 头注释引用的绘图位置族出处。）
2. Hazen, J.A., 1914. *Hydraulic Tables*. U.S. Department of Agriculture.
   （Hazen 绘图位置。）
3. Blom, G., 1956. On the distribution of the median and the mean of a
   sample from any continuous population. *Biometrika* 43(3/4): 572-575.
   （Blom 绘图位置。）

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

__all__ = [
    "PLOTTING_POSITIONS",
    "inverse_percentile",
    "non_exceedance_probability",
]

#: 绘图位置族 → ``(p₁, p₂)``，使第 ``i`` 个次序统计量的位置为 ``(i − p₁)/(n + p₂)``。
PLOTTING_POSITIONS: dict[str, tuple[float, float]] = {
    "weibull": (0.0, 1.0),
    "median": (0.3175, 0.365),
    "apl": (0.35, 0.0),
    "blom": (0.375, 0.25),
    "cunnane": (0.4, 0.2),
    "gringorten": (0.44, 0.12),
    "hazen": (0.5, 0.0),
}


def _as_sample(values: npt.NDArray) -> npt.NDArray:
    """把输入转成一维、有限的样本数组。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValidationError(
            _("the sample must be a one-dimensional array; got {0} dimensions").format(array.ndim),
            details={"ndim": int(array.ndim)},
        )
    if array.size == 0:
        raise ValidationError(_("the sample must not be empty"), details={"size": 0})
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("the sample contains non-finite values"),
            details={"n_non_finite": int(np.sum(~np.isfinite(array)))},
        )
    return array


def _resolve_plotting_position(
    plotting_position: str | tuple[float, float],
) -> tuple[float, float]:
    """把绘图位置名（或自定义 ``(p₁, p₂)``）解析成 ``(p₁, p₂)``。"""
    if isinstance(plotting_position, str):
        key = plotting_position.strip().lower()
        if key not in PLOTTING_POSITIONS:
            raise ValidationError(
                _("plotting position must be one of {0}; got '{1}'").format(
                    ", ".join(sorted(PLOTTING_POSITIONS)), plotting_position
                ),
                details={"plotting_position": plotting_position},
            )
        return PLOTTING_POSITIONS[key]
    if isinstance(plotting_position, (tuple, list)) and len(plotting_position) == 2:
        p1 = float(plotting_position[0])
        p2 = float(plotting_position[1])
        if not (np.isfinite(p1) and np.isfinite(p2)):
            raise ValidationError(
                _("custom plotting position parameters must be finite"),
                details={"p1": p1, "p2": p2},
            )
        if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
            raise ValidationError(
                _("custom plotting position parameters must lie between 0 and 1"),
                details={"p1": p1, "p2": p2},
            )
        return p1, p2
    raise ValidationError(
        _("plotting position must be a name or a (p1, p2) pair"),
        details={"plotting_position": repr(plotting_position)},
    )


def _plotting_positions(
    values: npt.NDArray,
    plotting_position: str | tuple[float, float],
) -> tuple[npt.NDArray, npt.NDArray]:
    """返回"唯一取值"与其绘图位置。

    重复取值按 Acycle ``invprctile.m`` 的做法（``unique(...,'last')``）只保留
    **最后一次出现**的次序统计量。
    """
    p1, p2 = _resolve_plotting_position(plotting_position)
    ordered = np.sort(values)
    n = ordered.size
    # 每个唯一取值的最后一次出现处的下标（0 基）。
    last_index = np.flatnonzero(np.append(ordered[1:] != ordered[:-1], True))
    unique_values = ordered[last_index]
    positions = (last_index.astype(float) + 1.0 - p1) / (float(n) + p2)
    return unique_values, positions


def non_exceedance_probability(
    values: npt.NDArray,
    quantile: float,
    plotting_position: str | tuple[float, float] = "hazen",
) -> float:
    """给定样本与一个分位数值，返回它的非超越概率。

    这是 Acycle ``code/misc/invprctile.m`` 的方向（本函数返回 ``[0, 1]`` 的
    概率，Acycle 返回百分数）。

    Parameters:
        values: 样本（一维、有限）。
        quantile: 待求概率的分位数值。
        plotting_position: 绘图位置名（见 :data:`PLOTTING_POSITIONS`）或自定义
            ``(p₁, p₂)``。

    Returns:
        float: ``quantile`` 的非超越概率，落在 ``[0, 1]``。

    Raises:
        ValidationError: 输入维度、有限性或绘图位置不合法时。
    """
    array = _as_sample(values)
    if not np.isfinite(quantile):
        raise ValidationError(
            _("the quantile must be a finite number; got {0}").format(quantile),
            details={"quantile": float(quantile)},
        )
    unique_values, positions = _plotting_positions(array, plotting_position)
    if unique_values.size == 1:
        return 1.0 if quantile >= float(unique_values[0]) else 0.0

    index = int(np.searchsorted(unique_values, quantile, side="right")) - 1
    if index < 0:
        return 0.0
    if index >= unique_values.size - 1:
        return 1.0
    lower_value = float(unique_values[index])
    upper_value = float(unique_values[index + 1])
    lower_position = float(positions[index])
    upper_position = float(positions[index + 1])
    fraction = (float(quantile) - lower_value) / (upper_value - lower_value)
    return lower_position + fraction * (upper_position - lower_position)


def inverse_percentile(
    values: npt.NDArray,
    probability: float,
    plotting_position: str | tuple[float, float] = "hazen",
) -> float:
    """给定样本与一个非超越概率，返回对应的分位数值。

    这是 :func:`non_exceedance_probability`（Acycle ``invprctile.m`` 的方向）
    的**逆映射**：在样本的次序统计量之间按绘图位置线性插值，把概率还原成
    分位数值。

    Parameters:
        values: 样本（一维、有限）。
        probability: 非超越概率，取值 ``[0, 1]``。
        plotting_position: 绘图位置名（见 :data:`PLOTTING_POSITIONS`）或自定义
            ``(p₁, p₂)``。

    Returns:
        float: 分位数值。

    Raises:
        ValidationError: 输入维度、有限性、概率取值或绘图位置不合法时。
    """
    array = _as_sample(values)
    if not np.isfinite(probability):
        raise ValidationError(
            _("the probability must be a finite number; got {0}").format(probability),
            details={"probability": float(probability)},
        )
    if not 0.0 <= float(probability) <= 1.0:
        raise ValidationError(
            _("the probability must lie between 0 and 1; got {0}").format(probability),
            details={"probability": float(probability)},
        )

    unique_values, positions = _plotting_positions(array, plotting_position)
    if unique_values.size == 1:
        return float(unique_values[0])

    index = int(np.searchsorted(positions, float(probability), side="right")) - 1
    if index < 0:
        return float(unique_values[0])
    if index >= positions.size - 1:
        return float(unique_values[-1])
    lower_value = float(unique_values[index])
    upper_value = float(unique_values[index + 1])
    lower_position = float(positions[index])
    upper_position = float(positions[index + 1])
    fraction = (float(probability) - lower_position) / (upper_position - lower_position)
    return lower_value + fraction * (upper_value - lower_value)
