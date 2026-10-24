# =============================================================================
# FILE: stratigraphy/series_edit.py
# =============================================================================
"""
序列编辑：按时间区间删除或保留记录的一部分。

频谱分析前的数据准备里，经常需要先把某一段可疑的采样（采样过密的地层段、
仪器漂移段、明显的重做段）从记录里剔掉，或者反过来只保留某一段有意义的
区间。Acycle（https://github.com/mingsongli/acycle）在 Math 菜单下提供
"序列数据工程"一类工具，本模块实现其中最基础的两步：

* :func:`remove_part` —— 删掉给定时间区间内的点；
* :func:`remove_range` —— 只保留给定时间区间内的点。

设计约定
~~~~~~~~
* 区间用**时间轴上的数值**给出（不是下标），闭区间 ``[起, 止]``，两端都
  包含，与 :func:`stratigraphy.series_tools.select_intervals` 的 ``"time"``
  语义一致。
* 区间必须**落在记录的时间跨度之内**，且起点不晚于终点，否则抛出
  ``ValidationError``：一个完全在记录之外（或横跨记录边界）的区间几乎总是
  调用方的坐标写错了，静默地"删掉 0 个点"会把这种错误藏起来。
* 保留原有采样顺序，不插值、不补点、不改数值。
* 所有异常都是 ``utils.exceptions.ValidationError``
  （``DataValidationError`` 的别名），不使用裸 ``ValueError``。

行为依据（Acycle 公开行为，非源码翻译）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``code/seriesanalysis/select_interval.m``：按时间轴上的闭区间截取子
  序列；本模块的 ``remove_range`` 与之同义。
* ``code/guicode/serieseditor.m``：在界面上一段一段地剔除记录。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# 内部工具
# =============================================================================


def _as_pair(time: npt.NDArray, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """把输入整理成两条等长的一维序列。

    Args:
        time: 时间轴。
        values: 数值序列。

    Returns:
        ``(time, values)``，都是 ``float64`` 一维数组。

    Raises:
        ValidationError: 输入不是一维、长度不一致、为空，或时间轴含有非
            有限值。
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or v.ndim != 1:
        raise ValidationError(
            _("time and values must be one-dimensional arrays"),
            details={"time_ndim": int(t.ndim), "values_ndim": int(v.ndim)},
        )
    if t.size != v.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time_length": int(t.size), "values_length": int(v.size)},
        )
    if t.size == 0:
        raise ValidationError(
            _("time and values must not be empty"),
            details={"length": int(t.size)},
        )
    if not bool(np.all(np.isfinite(t))):
        raise ValidationError(
            _("time must contain only finite values"),
            details={"n_nonfinite": int(np.count_nonzero(~np.isfinite(t)))},
        )
    return t, v


def _interval_bounds(interval: tuple[float, float], t: npt.NDArray) -> tuple[float, float]:
    """标准化并校验一个时间区间。

    Args:
        interval: ``(起, 止)``，单位与时间轴一致。
        t: 时间轴，用于检查区间是否落在记录的时间跨度内。

    Returns:
        ``(lo, hi)``，两个有限数且 ``lo <= hi``。

    Raises:
        ValidationError: 区间不是一对数、含有非有限值、起点晚于终点，或
            区间落在记录的时间跨度之外。
    """
    try:
        pair = tuple(interval)
    except TypeError as exc:
        raise ValidationError(
            _("interval must be a pair of numbers"),
            details={"interval": str(interval)},
            original_exception=exc,
        ) from exc
    if len(pair) != 2:
        raise ValidationError(
            _("interval must have exactly two bounds"),
            details={"interval": str(interval)},
        )
    try:
        lo = float(pair[0])
        hi = float(pair[1])
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("interval must be a pair of numbers"),
            details={"interval": str(interval)},
            original_exception=exc,
        ) from exc
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValidationError(
            _("interval must have finite bounds"),
            details={"start": lo, "end": hi},
        )
    if lo > hi:
        raise ValidationError(
            _("interval starts after it ends"),
            details={"start": lo, "end": hi},
        )

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if lo < t_min or hi > t_max:
        raise ValidationError(
            _("interval lies outside the time range of the record"),
            details={
                "start": lo,
                "end": hi,
                "time_min": t_min,
                "time_max": t_max,
            },
        )
    return lo, hi


# =============================================================================
# 公开接口
# =============================================================================


def remove_part(
    time: npt.NDArray,
    values: npt.NDArray,
    interval: tuple[float, float],
) -> tuple[npt.NDArray, npt.NDArray]:
    """删掉时间区间内的点。

    区间是**时间轴上的数值**给出的闭区间 ``[起, 止]``，两端的点都会被
    删除。其余点按原来的顺序返回，时间轴与数值轴同时被删减。

    Args:
        time: 时间轴，一维、有限、非空。
        values: 数值序列，与 ``time`` 等长。
        interval: ``(起, 止)``，闭区间，必须落在记录的时间跨度之内。

    Returns:
        ``(剩余时间, 剩余数值)`` 二元组。

    Raises:
        ValidationError: 输入不合法、区间反了，或区间落在记录的时间跨度
            之外。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.series_edit import remove_part
        >>> t = np.arange(0.0, 10.0, 1.0)
        >>> y = t * 2.0
        >>> kept_t, _ = remove_part(t, y, (2.0, 4.0))
        >>> kept_t.size
        7
    """
    t, v = _as_pair(time, values)
    lo, hi = _interval_bounds(interval, t)
    keep = ~((t >= lo) & (t <= hi))
    return t[keep], v[keep]


def remove_range(
    time: npt.NDArray,
    values: npt.NDArray,
    keep: tuple[float, float],
) -> tuple[npt.NDArray, npt.NDArray]:
    """只保留时间区间内的点（删掉区间之外的一切）。

    区间是**时间轴上的数值**给出的闭区间 ``[起, 止]``，两端的点都会被
    保留。其余点按原来的顺序返回，时间轴与数值轴同时被删减。

    Args:
        time: 时间轴，一维、有限、非空。
        values: 数值序列，与 ``time`` 等长。
        keep: ``(起, 止)``，闭区间，必须落在记录的时间跨度之内。

    Returns:
        ``(保留时间, 保留数值)`` 二元组。

    Raises:
        ValidationError: 输入不合法、区间反了，或区间落在记录的时间跨度
            之外。

    Examples:
        >>> import numpy as np
        >>> from stratigraphy.series_edit import remove_range
        >>> t = np.arange(0.0, 10.0, 1.0)
        >>> y = t * 2.0
        >>> kept_t, _ = remove_range(t, y, (2.0, 4.0))
        >>> kept_t.size
        3
    """
    t, v = _as_pair(time, values)
    lo, hi = _interval_bounds(keep, t)
    inside = (t >= lo) & (t <= hi)
    return t[inside], v[inside]
