# =============================================================================
# FILE: stratigraphy/moving_summary.py
# =============================================================================
"""
滑动窗口统计：固定带宽与固定点数两种窗口定义。

两种窗口的区别是**窗口怎么选**，统计量本身是一样的：

* :func:`fixed_bandwidth_summary` —— 固定**时间带宽**：窗口的宽度按时间固定
  （例如 ±50 kyr），因此在采样不均匀的剖面上，每个窗口包含的样本数可以
  不同。窗口为闭区间 ``[t - bandwidth / 2, t + bandwidth / 2]``，
  ``bandwidth`` 是窗口的**总时间宽度**（``bandwidth = 100`` 即 ±50）。
* :func:`fixed_count_summary` —— 固定**样本点数**：窗口按样本序号展开，
  与坐标无关，因此每个窗口包含的点数相同。``window`` 是窗口包含的
  **总点数**（``window = 21`` 即 ±10 个样本）。点数为偶数时按 Acycle 的
  约定多取一个前向点。

两种实现都以**输入的每个样本点为窗口中心**，输出与输入等长；靠近序列两端
时窗口自动收缩（不向另一侧借样本），统计量因此只由窗口内实际存在的样本
决定。

统计量
~~~~~~
每个窗口给出均值、样本标准差（``ddof=1``）、最小值、最大值与均值的
标准误 ``std / sqrt(n)``。窗口内只有一个样本时，标准差与标准误取 0。

参考文献
~~~~~~~~
* Acycle 的固定带宽与固定点数滑动统计（``acycleFixedBandwidthSummary.m``、
  ``acycleFixedCountMovingSummary.m``）给出了窗口选取与端点收缩的约定。
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. Computers & Geosciences 127, 12-22.
* Tukey, J. W. (1954). The choice of summarizing means and their
  justifications.（滑动平均与稳健统计的经典讨论）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考其公开的行为约定
（窗口选取、端点收缩、样本方差用 ``n - 1``），未翻译或复制其 MATLAB 源码。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 滑动统计至少需要的样本点数。
MIN_SAMPLES = 2

#: 固定点数窗口允许的最少点数（标准差需要至少 2 个样本，留一点余量）。
MIN_WINDOW_POINTS = 3


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class MovingSummaryResult:
    """
    滑动窗口统计的输出。

    Attributes:
        centres: 窗口中心的坐标，与输入等长。
        mean: 每个窗口内的均值。
        std: 每个窗口内的样本标准差（``ddof=1``）。
        minimum: 每个窗口内的最小值。
        maximum: 每个窗口内的最大值。
        standard_error: 每个窗口内均值的标准误 ``std / sqrt(n)``。
    """

    centres: npt.NDArray
    mean: npt.NDArray
    std: npt.NDArray
    minimum: npt.NDArray
    maximum: npt.NDArray
    standard_error: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        return (
            "Moving summary\n"
            f"{'=' * 48}\n"
            f"Windows: {int(self.centres.size)}\n"
            f"Mean range: [{float(self.mean.min()):.6g}, {float(self.mean.max()):.6g}]\n"
            f"Median standard error: {float(np.median(self.standard_error)):.6g}"
        )


# =============================================================================
# Validation helpers
# =============================================================================


def _as_series(values: object, name: str, *, minimum_size: int = MIN_SAMPLES) -> npt.NDArray:
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


def _validate_pair(
    time: Sequence[float] | npt.NDArray, values: Sequence[float] | npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """检查时间轴与数值轴：等长、时间严格递增、数值有限。"""
    axis = _as_series(time, "time")
    series = _as_series(values, "values")
    if axis.size != series.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time_length": int(axis.size), "values_length": int(series.size)},
        )
    if np.any(np.diff(axis) <= 0.0):
        raise ValidationError(
            _("time must be strictly increasing"),
            details={"parameter": "time"},
        )
    return axis, series


def _summarise(
    centres: npt.NDArray, values: npt.NDArray, starts: npt.NDArray, stops: npt.NDArray
) -> MovingSummaryResult:
    """按给定的窗口边界逐窗计算均值、标准差、极值与标准误。"""
    count = int(centres.size)
    means = np.empty(count, dtype=float)
    deviations = np.empty(count, dtype=float)
    minimums = np.empty(count, dtype=float)
    maximums = np.empty(count, dtype=float)
    errors = np.empty(count, dtype=float)
    for index in range(count):
        window = values[starts[index] : stops[index]]
        size = int(window.size)
        if size == 0:
            means[index] = np.nan
            deviations[index] = np.nan
            minimums[index] = np.nan
            maximums[index] = np.nan
            errors[index] = np.nan
            continue
        means[index] = float(window.mean())
        deviations[index] = float(window.std(ddof=1)) if size > 1 else 0.0
        minimums[index] = float(window.min())
        maximums[index] = float(window.max())
        errors[index] = deviations[index] / math.sqrt(size)
    return MovingSummaryResult(
        centres=centres,
        mean=means,
        std=deviations,
        minimum=minimums,
        maximum=maximums,
        standard_error=errors,
    )


# =============================================================================
# Public API
# =============================================================================


def fixed_bandwidth_summary(
    time: Sequence[float] | npt.NDArray,
    values: Sequence[float] | npt.NDArray,
    bandwidth: float,
) -> MovingSummaryResult:
    """
    固定时间带宽的滑动统计。

    窗口是以每个样本点为中心、宽度为 ``bandwidth`` 的**闭**时间区间
    ``[t - bandwidth / 2, t + bandwidth / 2]``。带宽固定意味着窗口内的
    样本数随采样密度变化。

    Parameters:
        time: 严格递增的时间轴（单位由调用方决定，例如 kyr 或 Ma）。
        values: 与 ``time`` 等长的数值序列。
        bandwidth: 窗口的总时间宽度，必须为正有限数。

    Returns:
        MovingSummaryResult: 每个样本点处的窗口统计量。

    Raises:
        ValidationError: 输入非法（长度不一致、时间非递增、数值非有限）
            或 ``bandwidth`` 不是正数。

    Examples:
        >>> import numpy as np
        >>> axis = np.arange(100.0)
        >>> series = np.full(100, 3.0)
        >>> result = fixed_bandwidth_summary(axis, series, bandwidth=10.0)
        >>> float(result.mean.max())
        3.0
    """
    axis, series = _validate_pair(time, values)
    width = float(bandwidth)
    if not np.isfinite(width) or width <= 0.0:
        raise ValidationError(
            _("bandwidth must be a positive finite number"),
            details={"bandwidth": bandwidth},
        )

    half = 0.5 * width
    starts = np.searchsorted(axis, axis - half, side="left")
    stops = np.searchsorted(axis, axis + half, side="right")
    return _summarise(axis, series, starts, stops)


def fixed_count_summary(
    time: Sequence[float] | npt.NDArray,
    values: Sequence[float] | npt.NDArray,
    window: int,
) -> MovingSummaryResult:
    """
    固定样本点数的滑动统计。

    窗口按**样本序号**展开，与坐标间距无关：``window`` 是窗口包含的总
    点数。点数为奇数时窗口关于中心点对称（``-h .. +h``）；点数为偶数时
    按 Acycle 的约定多取一个前向点（``-(h-1) .. +h``）。靠近序列两端时
    窗口自动收缩，不向另一侧借样本。

    Parameters:
        time: 严格递增的时间轴（单位由调用方决定）。
        values: 与 ``time`` 等长的数值序列。
        window: 窗口包含的样本点数，必须是至少 3 的整数。

    Returns:
        MovingSummaryResult: 每个样本点处的窗口统计量。

    Raises:
        ValidationError: 输入非法（长度不一致、时间非递增、数值非有限）
            或 ``window`` 不是至少 3 的整数。

    Examples:
        >>> import numpy as np
        >>> axis = np.arange(10.0)
        >>> series = np.arange(10.0)
        >>> result = fixed_count_summary(axis, series, window=3)
        >>> float(result.mean[5])
        5.0
    """
    axis, series = _validate_pair(time, values)
    points = int(window)
    if points != window or points < MIN_WINDOW_POINTS:
        raise ValidationError(
            _("window must be an integer of at least {0}").format(MIN_WINDOW_POINTS),
            details={"window": window},
        )

    half = points // 2
    backward = half if points % 2 == 1 else max(0, half - 1)
    forward = half
    indices = np.arange(axis.size)
    starts = np.maximum(0, indices - backward)
    stops = np.minimum(axis.size, indices + forward + 1)
    return _summarise(axis, series, starts, stops)
