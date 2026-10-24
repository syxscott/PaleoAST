# =============================================================================
# FILE: stratigraphy/series_tools.py
# =============================================================================
"""
序列数据工程工具：频谱分析之前的序列准备步骤。

本模块实现 Acycle（https://github.com/mingsongli/acycle）Math 菜单中
"序列数据工程"一类工具的 Python 版本：插值、选段、合并序列、插入缺口、
去除尖峰、数值截尾、排序去重与数值求导。Acycle 以 GPL-3.0 发布，因此
本文件是**独立实现**，只参考其公开的行为约定，未翻译或复制其 MATLAB 源码。

设计约定
~~~~~~~~
* 时间轴一律是**一维浮点数组**。除 :func:`merge_series`（负责排序）与
  :func:`sort_unique`（只处理数值）之外，其余函数都要求输入时间轴严格
  递增，否则抛出 ``ValidationError``。Acycle 的 ``datapreproc`` 同样是
  先排序、再合并重复时间点、最后才插值。
* 缺失数据用 ``NaN`` 表示。本模块只做"标记"，**不做零填充**、不去趋势、
  不标准化。
* 所有异常都是 ``utils.exceptions.ValidationError``（``DataValidationError``
  的别名），不使用裸 ``ValueError``，便于调用方统一捕获。
* 插值默认**不外推**：落在原序列时间范围之外的采样点返回 ``NaN``，
  与 Acycle ``acycleInterpolateSeries`` 的 "no extrapolation" 行为一致。

行为依据（Acycle 公开行为，非源码翻译）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``code/misc/datapreproc.m``：去 NaN、去 inf、按时间排序、重复时间点取
  均值、采样不均匀时按中位采样率重采样。
* ``code/seriesanalysis/acycleInterpolateSeries.m``：固定线性插值、不外推、
  不补零、不去趋势、保留所有数值列。
* ``code/seriesanalysis/select_interval.m``：按时间轴上的闭区间
  ``[t1, t2]`` 截取子序列。
* ``code/seriesanalysis/find_extrema.m``、``findduplicate.m``：局部极值判定
  与重复时间点的合并策略。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from config.i18n import _
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

#: 支持的插值方法。
INTERPOLATION_METHODS = ("linear", "cubic", "nearest")

#: :func:`select_intervals` 支持的区间解释方式。
INTERVAL_MODES = ("auto", "time", "index")

#: :func:`merge_series` 支持的重复时间点处理策略。
MERGE_KEEP_POLICIES = ("first", "last", "mean", "error")


# =============================================================================
# 内部工具
# =============================================================================


def _as_float_1d(array: object, name: str, *, require_finite: bool = False) -> npt.NDArray:
    """把输入转换为一维 float 数组，并检查形状与（可选的）有限性。

    Parameters:
        array: 待转换的对象。
        name: 参数名，用于错误信息。
        require_finite: 是否要求所有元素有限（插值等操作必须有限）。

    Returns:
        一维 float 数组（可能是原数组的视图，不保证是新对象）。

    Raises:
        ValidationError: 输入不可转为数值、不是一维、为空，或含有非有限值。
    """
    try:
        arr = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if arr.ndim != 1:
        raise ValidationError(
            _("{0} must be a one-dimensional array").format(name),
            details={"parameter": name, "ndim": int(arr.ndim)},
        )
    if arr.size == 0:
        raise ValidationError(
            _("{0} must not be empty").format(name),
            details={"parameter": name},
        )
    if require_finite and not bool(np.all(np.isfinite(arr))):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return arr


def _check_pair(time: npt.NDArray, values: npt.NDArray) -> None:
    """检查时间轴与数值轴长度一致。"""
    if time.size != values.size:
        raise ValidationError(
            _("time and values must have the same length"),
            details={"time_length": int(time.size), "values_length": int(values.size)},
        )


def _check_increasing(time: npt.NDArray, name: str = "time") -> None:
    """检查时间轴严格递增。"""
    if time.size < 2:
        raise ValidationError(
            _("{0} must contain at least two points").format(name),
            details={"parameter": name, "length": int(time.size)},
        )
    if not bool(np.all(np.diff(time) > 0)):
        raise ValidationError(
            _("{0} must be strictly increasing").format(name),
            details={"parameter": name},
        )


def _interval_bounds(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """把区间列表标准化为 ``[(lo, hi), ...]`` 并检查其合法性。

    Raises:
        ValidationError: 区间不是两个数、含有非有限值，或起点晚于终点。
    """
    bounds: list[tuple[float, float]] = []
    for position, interval in enumerate(intervals):
        try:
            pair = tuple(interval)
        except TypeError as exc:
            raise ValidationError(
                _("Interval {0} must be a pair of numbers").format(position),
                details={"position": position},
                original_exception=exc,
            ) from exc
        if len(pair) != 2:
            raise ValidationError(
                _("Interval {0} must have exactly two bounds").format(position),
                details={"position": position},
            )
        try:
            lo = float(pair[0])
            hi = float(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("Interval {0} must be a pair of numbers").format(position),
                details={"position": position},
                original_exception=exc,
            ) from exc
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValidationError(
                _("Interval {0} must have finite bounds").format(position),
                details={"position": position},
            )
        if lo > hi:
            raise ValidationError(
                _("Interval {0} starts after it ends").format(position),
                details={"position": position, "start": lo, "end": hi},
            )
        bounds.append((lo, hi))
    return bounds


def _is_index_pair(lo: float, hi: float, size: int) -> bool:
    """判断一对边界是否可以解释为整数下标。"""
    if float(lo).is_integer() and float(hi).is_integer():
        return 0 <= lo <= hi <= size - 1
    return False


def _is_series_pair(obj: object) -> bool:
    """判断对象是否形如 ``(time, values)``。"""
    try:
        pair = tuple(obj)  # type: ignore[call-overload]
    except TypeError:
        return False
    if len(pair) != 2:
        return False
    return all(isinstance(part, (np.ndarray, list, tuple)) for part in pair)


# =============================================================================
# 插值
# =============================================================================


def interpolate_series(
    time: npt.NDArray,
    values: npt.NDArray,
    new_time: npt.NDArray,
    method: str = "linear",
    *,
    extrapolate: bool = False,
) -> npt.NDArray:
    """
    把序列重采样到新的时间轴上。

    插值只依赖时间坐标，不假设等间距：不等间距的时间轴同样按真实坐标
    插值（``linear`` 用分段线性，``cubic`` 用自然三次样条，``nearest``
    取最近邻样本）。

    Parameters:
        time: 原始时间轴，必须严格递增且全部有限。
        values: 与 ``time`` 等长的数值序列，必须全部有限。
        new_time: 目标时间轴，顺序任意。
        method: ``"linear"``、``"cubic"`` 或 ``"nearest"``。
        extrapolate: 是否允许外推。默认 ``False``：落在
            ``[time[0], time[-1]]`` 之外的点返回 ``NaN``（不外推、不补零）。

    Returns:
        与 ``new_time`` 等长的插值结果数组。

    Raises:
        ValidationError: 数组为空、长度不匹配、时间轴非严格递增、方法未知，
            或数值含有 NaN/inf。
    """
    t = _as_float_1d(time, "time", require_finite=True)
    v = _as_float_1d(values, "values", require_finite=True)
    target = _as_float_1d(new_time, "new_time", require_finite=True)
    _check_pair(t, v)
    _check_increasing(t)
    if method not in INTERPOLATION_METHODS:
        raise ValidationError(
            _("method must be one of {0}; got '{1}'").format(", ".join(INTERPOLATION_METHODS), method),
            details={"method": method, "valid_methods": list(INTERPOLATION_METHODS)},
        )

    if method == "linear":
        result = np.interp(target, t, v)
        if extrapolate:
            # np.interp 只做"夹取"（超出范围取端点值），这里改成真正的
            # 线性外推：用首/末两个节点连线的斜率向外延伸。
            below = target < t[0]
            above = target > t[-1]
            if bool(np.any(below)):
                slope = (v[1] - v[0]) / (t[1] - t[0])
                result = np.where(below, v[0] + (target - t[0]) * slope, result)
            if bool(np.any(above)):
                slope = (v[-1] - v[-2]) / (t[-1] - t[-2])
                result = np.where(above, v[-1] + (target - t[-1]) * slope, result)
        else:
            result = np.where((target < t[0]) | (target > t[-1]), np.nan, result)
        return result

    if method == "cubic":
        # CubicSpline 处理不等间距；点数少于 4 时自动退化为二次/线性，
        # 这与 scipy 的 not-a-knot 边界条件一致。
        spline = CubicSpline(t, v, extrapolate=bool(extrapolate))
        return np.asarray(spline(target), dtype=float)

    # nearest：取距离最近的样本；距离相同时取左侧样本（与 scipy interp1d 一致）。
    position = np.searchsorted(t, target, side="left")
    left = np.clip(position - 1, 0, t.size - 1)
    right = np.clip(position, 0, t.size - 1)
    take_left = np.abs(target - t[left]) <= np.abs(t[right] - target)
    selected = np.where(take_left, left, right)
    result = v[selected]
    if not extrapolate:
        result = np.where((target < t[0]) | (target > t[-1]), np.nan, result)
    return result


# =============================================================================
# 选段
# =============================================================================


def select_intervals(
    time: npt.NDArray,
    values: npt.NDArray,
    intervals: Sequence[tuple[float, float]],
    *,
    mode: str = "auto",
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    按区间截取序列的子序列。

    每个区间是一对边界 ``(起, 止)``，闭区间（两端都包含），语义取决于
    ``mode``：

    * ``"time"``：边界是**时间值**，取出 ``time`` 落在 ``[起, 止]`` 内的
      样本（Acycle ``select_interval`` 的行为）。
    * ``"index"``：边界是**整数下标**，取出 ``time[起:止 + 1]``。
    * ``"auto"``（默认）：先按时间值解释；只有当该区间按时间取值选不中
      任何样本、而两个边界又恰好是合法的整数下标时，才改按下标解释。

    多个区间的结果按时间排序后拼接返回；相互重叠的区间会保留重复样本。

    Parameters:
        time: 时间轴。
        values: 与 ``time`` 等长的数值序列。
        intervals: 区间列表，每项为 ``(起, 止)``。
        mode: ``"auto"``、``"time"`` 或 ``"index"``。

    Returns:
        ``(选中时间, 选中数值)`` 二元组。

    Raises:
        ValidationError: 数组为空、长度不匹配、区间非法，或 ``mode`` 未知。
    """
    t = _as_float_1d(time, "time", require_finite=True)
    v = _as_float_1d(values, "values")
    _check_pair(t, v)
    if mode not in INTERVAL_MODES:
        raise ValidationError(
            _("mode must be one of {0}; got '{1}'").format(", ".join(INTERVAL_MODES), mode),
            details={"mode": mode, "valid_modes": list(INTERVAL_MODES)},
        )
    bounds = _interval_bounds(intervals)

    picked_time: list[npt.NDArray] = []
    picked_values: list[npt.NDArray] = []
    for lo, hi in bounds:
        mask = (t >= lo) & (t <= hi)
        use_index = mode == "index"
        if mode == "auto" and not bool(mask.any()):
            use_index = _is_index_pair(lo, hi, t.size)
        if use_index:
            if not _is_index_pair(lo, hi, t.size):
                raise ValidationError(
                    _("Interval ({0}, {1}) is not a valid index range").format(lo, hi),
                    details={"start": lo, "end": hi, "length": int(t.size)},
                )
            start, stop = int(lo), int(hi)
            picked_time.append(t[start : stop + 1])
            picked_values.append(v[start : stop + 1])
        else:
            picked_time.append(t[mask])
            picked_values.append(v[mask])

    if not picked_time:
        return t[:0], v[:0]
    selected_time = np.concatenate(picked_time)
    selected_values = np.concatenate(picked_values)
    order = np.argsort(selected_time, kind="stable")
    return selected_time[order], selected_values[order]


# =============================================================================
# 合并
# =============================================================================


def merge_series(
    series: Sequence[tuple[npt.NDArray, npt.NDArray]],
    keep: str = "first",
    *extra: tuple[npt.NDArray, npt.NDArray],
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    把多条序列按时间合并成一条，并按时间升序排列。

    重复时间点的处理由 ``keep`` 明确指定，**不做静默取舍**：

    * ``"first"``：保留重复时间点中**最先出现**的那一条。
    * ``"last"``：保留重复时间点中**最后出现**的那一条。
    * ``"mean"``：把重复时间点的数值取**均值**（Acycle ``findduplicate``
      的做法）。
    * ``"error"``：发现重复时间点即抛出 ``ValidationError``。

    Parameters:
        series: ``(time, values)`` 序列的列表。也接受
            ``merge_series((t1, v1), (t2, v2))`` 这样的可变参数写法，以及
            ``merge_series((t, v))`` 这样只传一条序列的写法。
        keep: 重复时间点的处理策略，见上。
        *extra: 仅用于支持可变参数写法。

    Returns:
        ``(合并后的时间, 合并后的数值)`` 二元组，时间严格递增。

    Raises:
        ValidationError: 输入为空、元素不是 ``(time, values)`` 二元组、长度
            不匹配、``keep`` 未知，或 ``keep="error"`` 且存在重复时间点。
    """
    items: list[object]
    if not isinstance(keep, str):
        # merge_series(s1, s2, ...) 写法：keep 位置放的是另一条序列。
        items = [series, keep, *extra]
        keep = "first"
    else:
        if extra:
            raise ValidationError(
                _("merge_series accepts either a sequence of series or a keep policy, not both"),
                details={"keep": keep},
            )
        items = list(series)

    if keep not in MERGE_KEEP_POLICIES:
        raise ValidationError(
            _("keep must be one of {0}; got '{1}'").format(", ".join(MERGE_KEEP_POLICIES), keep),
            details={"keep": keep, "valid_policies": list(MERGE_KEEP_POLICIES)},
        )

    if len(items) == 2 and not (_is_series_pair(items[0]) and _is_series_pair(items[1])):
        # merge_series((t, v))：只传了一条序列。
        items = [items]

    if not items:
        raise ValidationError(_("at least one series is required"))

    times: list[npt.NDArray] = []
    values: list[npt.NDArray] = []
    for position, item in enumerate(items):
        if not _is_series_pair(item):
            raise ValidationError(
                _("Series {0} must be a (time, values) pair").format(position),
                details={"position": position},
            )
        pair = tuple(item)  # type: ignore[call-overload]
        t = _as_float_1d(pair[0], "time")
        v = _as_float_1d(pair[1], "values")
        _check_pair(t, v)
        times.append(t)
        values.append(v)

    all_time = np.concatenate(times)
    all_values = np.concatenate(values)
    order = np.argsort(all_time, kind="stable")
    all_time = all_time[order]
    all_values = all_values[order]

    unique_time, inverse, counts = np.unique(all_time, return_inverse=True, return_counts=True)
    duplicated = bool(np.any(counts > 1))
    if not duplicated:
        return all_time, all_values

    if keep == "error":
        raise ValidationError(
            _("duplicate time values were found while merging"),
            details={"n_duplicate_times": int(np.sum(counts > 1)), "n_points": int(all_time.size)},
        )

    if keep == "first":
        first_index = np.unique(all_time, return_index=True)[1]
        return unique_time, all_values[first_index]

    if keep == "last":
        reversed_time = all_time[::-1]
        reversed_values = all_values[::-1]
        unique_reversed, last_index = np.unique(reversed_time, return_index=True)
        # unique_reversed 已按时间升序，last_index 指向倒序数组中的最后一次出现。
        return unique_reversed, reversed_values[last_index]

    # keep == "mean"
    sums = np.bincount(inverse, weights=all_values, minlength=unique_time.size)
    return unique_time, sums / counts


# =============================================================================
# 缺口
# =============================================================================


def add_gaps(
    time: npt.NDArray,
    values: npt.NDArray,
    gaps: Sequence[tuple[float, float]],
) -> npt.NDArray:
    """
    把指定时间区间内的样本标记为缺失（``NaN``）。

    时间轴本身**不改动**：区间内的样本只是被标记为 ``NaN``，因此返回值与
    输入 ``values`` 等长，可以直接与原来的 ``time`` 配对使用。缺口边界是
    闭区间。

    Parameters:
        time: 时间轴。
        values: 与 ``time`` 等长的数值序列。
        gaps: 缺口区间列表，每项为 ``(起, 止)``（时间值）。

    Returns:
        新的数值数组，缺口内的样本为 ``NaN``。

    Raises:
        ValidationError: 数组为空、长度不匹配或区间非法。
    """
    t = _as_float_1d(time, "time", require_finite=True)
    v = _as_float_1d(values, "values")
    _check_pair(t, v)
    bounds = _interval_bounds(gaps)

    result = v.copy()
    for lo, hi in bounds:
        result[(t >= lo) & (t <= hi)] = np.nan
    return result


# =============================================================================
# 去峰
# =============================================================================


def remove_peaks(
    time: npt.NDArray,
    values: npt.NDArray,
    threshold: float,
    min_distance: int = 1,
    *,
    fill: str = "nan",
) -> npt.NDArray:
    """
    去掉**超过阈值**的局部峰（尖峰）。

    峰的判定沿用 ``scipy.signal.find_peaks``：一个样本是局部峰，当且仅当它
    比左右邻居都大（端点不算峰）。``threshold`` 是峰高阈值，只有峰高
    **严格大于**阈值的峰才会被去掉。

    Parameters:
        time: 时间轴（仅用于校验长度，尖峰位置由数值序列决定）。
        values: 与 ``time`` 等长的数值序列。
        threshold: 峰高阈值。
        min_distance: 两个峰之间的最小样本间隔（``find_peaks`` 的
            ``distance`` 参数）。
        fill: 被去掉的峰如何填补：``"nan"``（默认，标记为缺失）或
            ``"interpolate"``（用左右邻居线性插值）。

    Returns:
        与输入等长的新数值数组。

    Raises:
        ValidationError: 数组为空、长度不匹配、``min_distance`` 小于 1、
            或 ``fill`` 未知。
    """
    t = _as_float_1d(time, "time")
    v = _as_float_1d(values, "values")
    _check_pair(t, v)
    if not np.isfinite(threshold):
        raise ValidationError(
            _("threshold must be finite"),
            details={"threshold": float(threshold)},
        )
    if int(min_distance) < 1:
        raise ValidationError(
            _("min_distance must be >= 1"),
            details={"min_distance": int(min_distance)},
        )
    if fill not in ("nan", "interpolate"):
        raise ValidationError(
            _("fill must be one of nan, interpolate; got '{0}'").format(fill),
            details={"fill": fill},
        )

    result = v.copy()
    if v.size < 3:
        return result

    peaks = find_peaks(v, distance=int(min_distance))[0]
    if peaks.size == 0:
        return result
    selected = peaks[v[peaks] > float(threshold)]
    if selected.size == 0:
        return result

    if fill == "nan":
        result[selected] = np.nan
        return result

    # interpolate：用左右邻居的线性插值替代尖峰。
    good = np.flatnonzero(np.isfinite(v))
    for index in selected:
        left = good[good < index]
        right = good[good > index]
        if left.size == 0 or right.size == 0:
            result[index] = np.nan
            continue
        x0, x1 = t[left[-1]], t[right[0]]
        y0, y1 = v[left[-1]], v[right[0]]
        if x1 == x0:
            result[index] = y0
        else:
            result[index] = y0 + (y1 - y0) * (t[index] - x0) / (x1 - x0)
    return result


# =============================================================================
# 截尾 / 去重 / 求导
# =============================================================================


def clip_values(
    values: npt.NDArray,
    lower: float | None = None,
    upper: float | None = None,
) -> npt.NDArray:
    """
    把数值截断到 ``[lower, upper]`` 区间内。

    Parameters:
        values: 数值序列。
        lower: 下界，``None`` 表示不设下界。
        upper: 上界，``None`` 表示不设上界。

    Returns:
        截断后的新数组（不修改入参）。

    Raises:
        ValidationError: 输入为空、边界非有限，或 ``lower > upper``。
    """
    v = _as_float_1d(values, "values")
    if lower is not None and not np.isfinite(lower):
        raise ValidationError(_("lower must be finite"), details={"lower": float(lower)})
    if upper is not None and not np.isfinite(upper):
        raise ValidationError(_("upper must be finite"), details={"upper": float(upper)})
    if lower is not None and upper is not None and float(lower) > float(upper):
        raise ValidationError(
            _("lower must not be greater than upper"),
            details={"lower": float(lower), "upper": float(upper)},
        )

    result = v.copy()
    if lower is not None:
        result = np.maximum(result, float(lower))
    if upper is not None:
        result = np.minimum(result, float(upper))
    return result


def sort_unique(values: npt.NDArray) -> tuple[npt.NDArray, int]:
    """
    排序去重，并报告被去掉的重复项个数。

    Parameters:
        values: 数值序列。

    Returns:
        ``(升序去重后的数组, 被去掉的重复项个数)``。被去掉的个数为
        ``len(values) - len(unique(values))``。

    Raises:
        ValidationError: 输入为空。
    """
    v = _as_float_1d(values, "values")
    unique = np.unique(v)
    removed = int(v.size - unique.size)
    return unique, removed


def derivative(time: npt.NDArray, values: npt.NDArray) -> npt.NDArray:
    """
    数值求导，尊重不等间距的时间轴。

    采用中心差分（端点用单边差分），等价于 ``numpy.gradient(values, time)``：
    间隔不等时，斜率按各自的实际间隔计算，而不是按单位下标。

    Parameters:
        time: 时间轴，必须严格递增。
        values: 与 ``time`` 等长的数值序列。

    Returns:
        与输入等长的导数数组。

    Raises:
        ValidationError: 数组为空、长度不匹配、点数少于 2，或时间轴非严格
            递增。
    """
    t = _as_float_1d(time, "time", require_finite=True)
    v = _as_float_1d(values, "values")
    _check_pair(t, v)
    _check_increasing(t)
    return np.gradient(v, t)
