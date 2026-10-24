# =============================================================================
# FILE: stratigraphy/frequency_stabilization.py
# =============================================================================
"""
频率稳定（Frequency Stabilization）：坐标可变的高斯带通滤波。

用途
====
Acycle 界面上的 "Dynamic Filter / 频率稳定" 实际执行的是**坐标可变的高斯带通
滤波**：用户在图上用鼠标点出**下边界**与**上边界**两条控制点曲线，滤波器的
通过频带随坐标一起移动，于是可以把一个**周期在漂移**的信号拉稳（例如古地磁
倒转频率从 10 Ma 的 20 Myr 漂移到 10 Ma 的 10 Myr）。

这与 :mod:`stratigraphy.dynamic_filter` 是**两套不同的算法**：后者是分段信噪比
动态滤波（按局部噪声水平调节滤波强度，频带固定）；本模块的频带本身随坐标移动。
两者的名字都带 "dynamic filter"，但语义完全不同，因此单列一个模块。

算法
====
设输入是 N×2 的 ``[坐标, 数值]``，坐标**严格递增**且**等间距**（本模块不对
数据行做排序、合并、删除或插值）。滤波按以下步骤进行：

1. **控制点路径。** 下边界与上边界各给出一组 ``[坐标, 频率]`` 控制点。坐标
   严格递增且落在数据域内；单行控制点表示**恒定**边界，多行控制点则在相邻
   控制点之间**线性插值**，并在第一个/最后一个控制点之外**保持常值**。
   两条完整路径在**每一点**上必须满足 ``0 < lower < upper < Nyquist``
   （``Nyquist = 1 / (2 * dt)``，``dt`` 为采样间隔）。
2. **窗口排布。** ``window_length`` 是窗口**首尾跨度**（坐标单位），除以
   ``median(diff(坐标))`` 后**四舍五入**到最近的采样间隔，再**加一个端点
   样本**得到窗口样本数，且窗口至少含 8 个样本。``step_length`` 是相邻窗口
   起点的位移，同样量化到整数样本，至少 1 个样本且**不得超过窗口样本数**
   ——因此窗口之间不会留下覆盖空隙。若按等距起点栅格排布会留下未覆盖的尾巴，
   则**追加**一个终止于最后一个样本的窗口。
3. **逐窗滤波。** 每个窗口先**局部去均值**（去的是该窗的算术均值），再乘
   以**半样本错位**的严格正 Hann 权重的**平方根**（分析权重），在**显式补零**
   的 FFT 上做高斯滤波，乘**同一个**合成权重（同样是平方根），最后
   **overlap-add** 累加。
4. **归一化。** 输出除以**合成权重之和**（逐点累计的 Hann 权重），而不是除以
   非零滤波值的个数。

高斯带通的频率响应
==================
本模块使用**对称高斯带通**，其上下频率是**精确的半功率（−3.0103 dB）点**。
设 ``f_c = (lower + upper) / 2`` 为通带中心、``b = (upper - lower) / 2`` 为半
带宽，则频率 ``f`` 处的**幅值**增益为

.. code-block:: text

    G(f) = exp( -ln(sqrt(2)) * ((f - f_c) / b)**2 )

于是

* ``f = f_c`` 时 ``G = 1``（通带中心增益为 1）；
* ``f = lower`` 与 ``f = upper`` 时 ``|f - f_c| = b``，故 ``G = 1 / sqrt(2)``，
  **功率**增益 ``G**2 = 0.5``，即 ``10 * log10(0.5) = -3.0103 dB``；
* ``f`` 离开通带越远增益越小，且在带外**单调**下降。

:func:`gaussian_bandpass_response` 公开的就是这条曲线，滤波实现调用的是
**同一个函数**，因此"公开的响应曲线"与"滤波实际使用的响应"不可能漂移。

参数约束
========
以下约束与 Acycle 的行为约定逐条对应，违反任意一条都抛出
``utils.exceptions.ValidationError``：

* 坐标必须**严格递增**，且**等间距**：相对间距偏差（``max |diff - dt| / dt``，
  ``dt = median(diff)``）不得超过 ``1e-5``；
* 坐标与数值的**长度必须一致**，且点数不少于 8；
* 所有坐标、数值、控制点都必须是**有限**实数；
* 控制点必须形如 ``K×2`` 的 ``[坐标, 频率]``，``K >= 1``；坐标严格递增、
  落在数据域 ``[coordinate_start, coordinate_end]`` 内，频率严格为正；
* 通带在整条路径上满足 ``0 < lower < upper < Nyquist``；
* ``window_length`` 与 ``step_length`` 必须是有限正数；``window_length``
  不得超过坐标总跨度；窗口样本数至少 8、不得超过输入点数；
* ``step_length`` 至少 1 个样本，且不得超过窗口样本数。

返回值
======
:class:`FrequencyStabilizationResult` 给出滤波序列、逐点的覆盖计数与覆盖权重，
以及每个窗口的 ``[窗口中心, 下边界频率, 上边界频率, 中心频率, 窗口起始坐标,
窗口结束坐标, 窗口起始下标, 窗口结束下标]`` 信息矩阵（其中下标是输入序列的
**0 基**下标）。

参考文献
========
* Thibault, N. & Rizzi, G. (2019). Dynamic filtering of univariate series:
  A scale-space approach. *Geology*, 47(1), 72–76.（动态滤波的原始方法）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12–22.（Acycle 的 "Frequency Stabilization" 子功能）
* Oppenheim, A. V. & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.（窗函数法、overlap-add 与零填充 FFT）
* Harris, F. J. (1978). On the use of windows for harmonic analysis with the
  Fourier transform. *Proceedings of the IEEE*, 66(1), 51–83.（Hann 窗）

Acycle 以 GPL-3.0 发布，因此本文件是**独立实现**：只参考其公开的行为约定
（控制点路径、窗口排布、半功率高斯带通、overlap-add 归一化），未翻译或复制
其 MATLAB 源码。

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

#: 采样间距的相对容差：``max |diff - dt| / dt`` 不得超过该值。
SPACING_TOLERANCE = 1.0e-5

#: 输入记录至少需要的样本点数。
MIN_SAMPLES = 8

#: 滤波窗口至少需要的样本点数。
MIN_WINDOW_SAMPLES = 8

#: 高斯响应的指数常数：``exp(-HALF_POWER_EXPONENT * x**2)``。
#:
#: ``ln(sqrt(2))`` 使得 ``x = ±1``（即上下边界频率处）时增益恰为 ``1/sqrt(2)``，
#: 也就是精确的半功率点（-3.0103 dB）。
HALF_POWER_EXPONENT = math.log(math.sqrt(2.0))

#: 半功率点的**幅值**增益。
HALF_POWER_GAIN = 1.0 / math.sqrt(2.0)


# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class FrequencyStabilizationResult:
    """
    频率稳定（坐标可变高斯带通滤波）的输出。

    Attributes:
        coordinates: 输入坐标（与输入等长，原样返回）。
        filtered: 滤波后的序列，与输入等长。
        coverage_count: 逐点被多少个窗口覆盖（处处至少为 1）。
        coverage_weight: 逐点的合成权重之和（除以它即完成归一化）。
        windows: 窗口信息矩阵，每行是
            ``[窗口中心坐标, 下边界频率, 上边界频率, 中心频率,
            窗口起始坐标, 窗口结束坐标, 窗口起始下标, 窗口结束下标]``，
            其中下标是输入序列的 **0 基**下标。
        lower_frequency: 每个窗口的下边界频率。
        upper_frequency: 每个窗口的上边界频率。
        center_frequency: 每个窗口的通带中心频率。
    """

    coordinates: npt.NDArray
    filtered: npt.NDArray
    coverage_count: npt.NDArray
    coverage_weight: npt.NDArray
    windows: npt.NDArray
    lower_frequency: npt.NDArray
    upper_frequency: npt.NDArray
    center_frequency: npt.NDArray

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = [
            "Frequency stabilization",
            "=" * 48,
            f"Samples: {int(self.filtered.size)}",
            f"Windows: {int(self.windows.shape[0])}",
            f"Lower frequency: {float(np.min(self.lower_frequency)):.6g} - {float(np.max(self.lower_frequency)):.6g}",
            f"Upper frequency: {float(np.min(self.upper_frequency)):.6g} - {float(np.max(self.upper_frequency)):.6g}",
            f"Minimum coverage: {int(np.min(self.coverage_count))}",
        ]
        return "\n".join(lines)


# =============================================================================
# Validation helpers
# =============================================================================


def _as_float_array(values: object, name: str) -> npt.NDArray:
    """把输入转成一维有限浮点数组。"""
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
    if array.size == 0:
        raise ValidationError(
            _("{0} must not be empty").format(name),
            details={"parameter": name},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} must contain only finite values").format(name),
            details={"parameter": name},
        )
    return array


def _validate_series(
    coordinate: object,
    values: object,
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """
    校验输入序列，返回 ``(坐标, 数值, 采样间隔)``。

    坐标必须**严格递增**且**等间距**：采样间隔取 ``median(diff(坐标))``，
    相对偏差不得超过 :data:`SPACING_TOLERANCE`。
    """
    coordinates = _as_float_array(coordinate, "coordinate")
    series = _as_float_array(values, "values")

    if coordinates.size != series.size:
        raise ValidationError(
            _("coordinate and values must have the same length"),
            details={"coordinate_length": int(coordinates.size), "values_length": int(series.size)},
        )
    if coordinates.size < MIN_SAMPLES:
        raise ValidationError(
            _("{0} needs at least {1} samples; got {2}").format("coordinate", MIN_SAMPLES, int(coordinates.size)),
            details={"parameter": "coordinate", "length": int(coordinates.size), "minimum": MIN_SAMPLES},
        )

    spacing = np.diff(coordinates)
    if np.any(spacing <= 0.0):
        raise ValidationError(
            _("coordinate must be strictly increasing"),
            details={"parameter": "coordinate"},
        )

    sample_interval = float(np.median(spacing))
    if not math.isfinite(sample_interval) or sample_interval <= 0.0:
        raise ValidationError(
            _("the sample interval must be finite and positive"),
            details={"parameter": "coordinate"},
        )
    relative_departure = float(np.max(np.abs(spacing - sample_interval)) / sample_interval)
    if relative_departure > SPACING_TOLERANCE:
        raise ValidationError(
            _("{0} must be regularly sampled; the relative spacing departure {1:.3g} exceeds {2:.3g}").format(
                "coordinate", relative_departure, SPACING_TOLERANCE
            ),
            details={
                "parameter": "coordinate",
                "relative_departure": relative_departure,
                "tolerance": SPACING_TOLERANCE,
            },
        )

    return coordinates, series, sample_interval


def _validate_control_points(points: object, name: str, start: float, end: float) -> npt.NDArray:
    """
    校验控制点矩阵，返回 ``K×2`` 的浮点数组。

    控制点必须是 ``[坐标, 频率]``：坐标严格递增、落在数据域内，频率为正。
    """
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} control points must be numeric").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValidationError(
            _("{0} control points must be a K-by-2 matrix of [coordinate, frequency]").format(name),
            details={"parameter": name, "shape": list(np.shape(array))},
        )
    if array.shape[0] < 1:
        raise ValidationError(
            _("{0} control points must contain at least one row").format(name),
            details={"parameter": name},
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            _("{0} control points must contain only finite values").format(name),
            details={"parameter": name},
        )
    if array.shape[0] > 1 and np.any(np.diff(array[:, 0]) <= 0.0):
        raise ValidationError(
            _("{0} control coordinates must be strictly increasing").format(name),
            details={"parameter": name},
        )
    if np.any(array[:, 0] < start) or np.any(array[:, 0] > end):
        raise ValidationError(
            _("{0} control coordinates must lie inside the coordinate domain [{1:.6g}, {2:.6g}]").format(
                name, start, end
            ),
            details={"parameter": name, "coordinate_start": start, "coordinate_end": end},
        )
    if np.any(array[:, 1] <= 0.0):
        raise ValidationError(
            _("{0} control frequencies must be strictly positive").format(name),
            details={"parameter": name},
        )
    return array


def _round_half_away_from_zero(value: float) -> int:
    """把非负数四舍五入到整数（0.5 向上，与 Acycle 的量化约定一致）。"""
    return math.floor(value + 0.5)


def _positive_scalar(value: object, name: str) -> float:
    """把输入转成有限正标量。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("{0} must be a number").format(name),
            details={"parameter": name},
            original_exception=exc,
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValidationError(
            _("{0} must be a finite positive number").format(name),
            details={"parameter": name},
        )
    return number


def _half_sample_shifted_hann(samples: int) -> npt.NDArray:
    """
    半样本错位的 Hann 权重（严格为正）。

    权重为 ``0.5 - 0.5 * cos(2 * pi * (i + 0.5) / N)``：端点处不是 0，因此
    不会出现"窗口边缘样本完全不被覆盖"的空隙。
    """
    phase = (np.arange(samples, dtype=float) + 0.5) / float(samples)
    weight = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValidationError(
            _("the deterministic overlap-add window is invalid"),
            details={"samples": int(samples)},
        )
    return weight


# =============================================================================
# Public API
# =============================================================================


def gaussian_bandpass_response(
    frequencies: Sequence[float] | npt.NDArray,
    lower_frequency: float,
    upper_frequency: float,
    sampling_interval: float,
) -> npt.NDArray:
    """
    对称高斯带通的**幅值**频率响应。

    响应公式（与滤波实现用的是**同一条**）：

    .. code-block:: text

        f_c = (lower + upper) / 2
        b   = (upper - lower) / 2
        G(f) = exp( -ln(sqrt(2)) * ((f - f_c) / b)**2 )

    由此可得：

    * ``G(f_c) = 1``（通带中心增益为 1）；
    * ``G(lower) = G(upper) = 1 / sqrt(2)``，**功率**增益恰为 ``0.5``，
      即 ``10 * log10(0.5) = -3.0103 dB`` 的**精确半功率点**；
    * 离开通带后增益**单调**下降。

    Parameters:
        frequencies: 求值的频率轴，**每单位坐标**的循环频率（物理频率）。
        lower_frequency: 通带下边界频率（每单位坐标）。
        upper_frequency: 通带上边界频率（每单位坐标）。
        sampling_interval: 采样间隔（坐标单位）；物理 Nyquist 频率为
            ``1 / (2 * sampling_interval)``。

    Returns:
        npt.NDArray: 与 ``frequencies`` 等长的实数增益曲线。

    Raises:
        ValidationError: 频率轴不是一维有限数组、通带边界非法
            （``0 < lower < upper < Nyquist``）或采样间隔不是有限正数。
    """
    axis = _as_float_array(frequencies, "frequencies")
    interval = _positive_scalar(sampling_interval, "sampling_interval")
    nyquist = 0.5 / interval

    lower = float(lower_frequency)
    upper = float(upper_frequency)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValidationError(
            _("the passband boundaries must be finite"),
            details={"lower_frequency": lower, "upper_frequency": upper},
        )
    if not 0.0 < lower < upper:
        raise ValidationError(
            _("the passband boundaries must satisfy 0 < lower < upper"),
            details={"lower_frequency": lower, "upper_frequency": upper},
        )
    if upper >= nyquist:
        raise ValidationError(
            _("the upper passband frequency must be strictly below the Nyquist frequency {0:.6g}").format(nyquist),
            details={"upper_frequency": upper, "nyquist": nyquist},
        )

    center = lower + (upper - lower) / 2.0
    half_bandwidth = (upper - lower) / 2.0
    offset = (axis - center) / half_bandwidth
    return np.exp(-HALF_POWER_EXPONENT * offset**2)


def interpolate_control_path(
    control_points: Sequence[Sequence[float]] | npt.NDArray,
    coordinates: Sequence[float] | npt.NDArray,
) -> npt.NDArray:
    """
    在给定坐标上求值控制点路径。

    路径是**分段线性**的：在相邻控制点之间线性插值，在第一个控制点之前与
    最后一个控制点之后**保持常值**（取端点值）。单行控制点表示一条**恒定**
    路径。

    Parameters:
        control_points: ``K×2`` 的 ``[坐标, 频率]`` 控制点；坐标必须严格递增。
        coordinates: 求值位置的坐标（任意顺序均可）。

    Returns:
        npt.NDArray: 与 ``coordinates`` 等长的路径取值。

    Raises:
        ValidationError: 控制点形状非法、坐标不严格递增、含非有限值，或求值
            坐标不是一维有限数组。
    """
    try:
        points = np.asarray(control_points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("control points must be numeric"),
            details={"parameter": "control_points"},
            original_exception=exc,
        ) from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValidationError(
            _("control points must be a K-by-2 matrix of [coordinate, frequency]"),
            details={"parameter": "control_points", "shape": list(np.shape(points))},
        )
    if not np.all(np.isfinite(points)):
        raise ValidationError(
            _("control points must contain only finite values"),
            details={"parameter": "control_points"},
        )
    if points.shape[0] > 1 and np.any(np.diff(points[:, 0]) <= 0.0):
        raise ValidationError(
            _("control coordinates must be strictly increasing"),
            details={"parameter": "control_points"},
        )

    query = _as_float_array(coordinates, "coordinates")
    if points.shape[0] == 1:
        return np.full(query.shape, float(points[0, 1]), dtype=float)
    # ``np.interp`` 在端点之外返回端点值，正是"端点外保持常值"的语义。
    return np.interp(query, points[:, 0], points[:, 1])


def frequency_stabilization_filter(
    coordinate: Sequence[float] | npt.NDArray,
    values: Sequence[float] | npt.NDArray,
    lower_control_points: Sequence[Sequence[float]] | npt.NDArray,
    upper_control_points: Sequence[Sequence[float]] | npt.NDArray,
    window_length: float,
    step_length: float,
) -> FrequencyStabilizationResult:
    """
    用坐标可变的高斯带通滤波"拉稳"一个周期在漂移的信号。

    下边界与上边界各由一组 ``[坐标, 频率]`` 控制点给出，两条路径线性插值、
    端点外保持常值，并在整条路径上满足 ``0 < lower < upper < Nyquist``。

    每个窗口先局部去均值，再乘半样本错位 Hann 权重的平方根，在显式补零的
    FFT 上乘以 :func:`gaussian_bandpass_response` 给出的高斯响应，逆变换后
    乘同一个平方根权重并 overlap-add；最后逐点除以合成权重之和。

    Parameters:
        coordinate: 一维坐标序列，严格递增且等间距（相对容差 ``1e-5``）。
        values: 与 ``coordinate`` 等长的一维数值序列。
        lower_control_points: 下边界控制点 ``K×2`` 的 ``[坐标, 频率]``。
        upper_control_points: 上边界控制点 ``K×2`` 的 ``[坐标, 频率]``。
        window_length: 窗口**首尾跨度**（坐标单位），有限正数；量化到整数
            采样间隔后再加一个端点样本，窗口样本数至少为 8。
        step_length: 相邻窗口起点的位移（坐标单位），有限正数；量化后至少
            1 个样本且不得超过窗口样本数。

    Returns:
        FrequencyStabilizationResult: 滤波序列、逐点覆盖量与窗口信息。

    Raises:
        ValidationError: 输入违反本模块文档中列出的任意一条约束。

    Examples:
        >>> import numpy as np
        >>> coordinate = np.arange(512, dtype=float)
        >>> values = np.sin(2.0 * np.pi * coordinate / 32.0)
        >>> result = frequency_stabilization_filter(
        ...     coordinate,
        ...     values,
        ...     [[0.0, 0.02]],
        ...     [[0.0, 0.05]],
        ...     window_length=128.0,
        ...     step_length=32.0,
        ... )
        >>> result.filtered.size == values.size
        True
    """
    coordinates, series, sample_interval = _validate_series(coordinate, values)
    coordinate_start = float(coordinates[0])
    coordinate_end = float(coordinates[-1])
    coordinate_span = coordinate_end - coordinate_start
    nyquist = 0.5 / sample_interval
    total = int(coordinates.size)

    lower_points = _validate_control_points(lower_control_points, "lower", coordinate_start, coordinate_end)
    upper_points = _validate_control_points(upper_control_points, "upper", coordinate_start, coordinate_end)

    # 两条折线的联合断点就是全部极值可能出现的位置：在每两个相邻断点之间，
    # 上下边界都是线性的，因此"lower < upper"与"upper < Nyquist"只需在断点
    # 处检查。数据域的两个端点也要纳入。
    check_coordinates = np.unique(
        np.concatenate(
            (
                coordinates[[0]],
                lower_points[:, 0],
                upper_points[:, 0],
                coordinates[[-1]],
            )
        )
    )
    lower_path = interpolate_control_path(lower_points, check_coordinates)
    upper_path = interpolate_control_path(upper_points, check_coordinates)
    if np.any(lower_path <= 0.0):
        raise ValidationError(
            _("the lower passband frequency must be strictly positive"),
            details={"parameter": "lower_control_points"},
        )
    if np.any(upper_path >= nyquist):
        raise ValidationError(
            _("the upper passband frequency must be strictly below the Nyquist frequency {0:.6g}").format(nyquist),
            details={"parameter": "upper_control_points", "nyquist": nyquist},
        )
    if np.any(lower_path >= upper_path):
        raise ValidationError(
            _("the passband paths must satisfy lower < upper"),
            details={"parameter": "lower_control_points"},
        )

    window_span = _positive_scalar(window_length, "window_length")
    if window_span > coordinate_span:
        raise ValidationError(
            _("{0} must not exceed the coordinate span {1:.6g}").format("window_length", coordinate_span),
            details={"parameter": "window_length", "coordinate_span": coordinate_span},
        )
    window_intervals = _round_half_away_from_zero(window_span / sample_interval)
    window_samples = window_intervals + 1
    if window_samples < MIN_WINDOW_SAMPLES:
        raise ValidationError(
            _("{0} resolves to {1} samples; at least {2} are required").format(
                "window_length", window_samples, MIN_WINDOW_SAMPLES
            ),
            details={
                "parameter": "window_length",
                "window_samples": window_samples,
                "minimum": MIN_WINDOW_SAMPLES,
            },
        )
    if window_samples > total:
        raise ValidationError(
            _("{0} resolves to {1} samples, more than the {2} available samples").format(
                "window_length", window_samples, total
            ),
            details={"parameter": "window_length", "window_samples": window_samples, "samples": total},
        )

    step_span = _positive_scalar(step_length, "step_length")
    step_samples = _round_half_away_from_zero(step_span / sample_interval)
    if step_samples < 1:
        raise ValidationError(
            _("{0} resolves to zero samples; increase it to at least half a sample interval").format("step_length"),
            details={"parameter": "step_length"},
        )
    if step_samples > window_samples:
        raise ValidationError(
            _("{0} resolves to {1} samples, more than the {2}-sample window; this would leave coverage gaps").format(
                "step_length", step_samples, window_samples
            ),
            details={
                "parameter": "step_length",
                "step_samples": step_samples,
                "window_samples": window_samples,
            },
        )

    # 窗口起点（0 基下标）：等距栅格；必要时追加一个终止于最后一个样本的窗口。
    last_start = total - window_samples
    start_indices = np.arange(0, last_start + 1, step_samples, dtype=int)
    if int(start_indices[-1]) != last_start:
        start_indices = np.append(start_indices, last_start)
    end_indices = start_indices + window_samples - 1
    window_centers = coordinates[start_indices] + (coordinates[end_indices] - coordinates[start_indices]) / 2.0

    lower_frequency = interpolate_control_path(lower_points, window_centers)
    upper_frequency = interpolate_control_path(upper_points, window_centers)
    center_frequency = lower_frequency + (upper_frequency - lower_frequency) / 2.0
    if not np.all(np.isfinite(center_frequency)):
        raise ValidationError(
            _("a passband center is not representable as a finite number"),
            details={"parameter": "lower_control_points"},
        )

    # 显式补零的 FFT：长度取不小于 ``2 * window_samples - 1`` 的 2 的幂。
    minimum_fft_length = 2 * window_samples - 1
    fft_length = 1 << (minimum_fft_length - 1).bit_length()
    frequency_axis = np.fft.rfftfreq(fft_length, d=sample_interval)

    weight = _half_sample_shifted_hann(window_samples)
    analysis_weight = np.sqrt(weight)

    value_scale = float(np.max(np.abs(series)))
    if value_scale == 0.0:
        scaled = np.zeros(total, dtype=float)
    else:
        scaled = series / value_scale

    accumulator = np.zeros(total, dtype=float)
    coverage_weight = np.zeros(total, dtype=float)
    coverage_count = np.zeros(total, dtype=int)

    for window_index in range(int(start_indices.size)):
        begin = int(start_indices[window_index])
        end = int(end_indices[window_index])
        segment = scaled[begin : end + 1]
        centered = segment - float(np.mean(segment))

        spectrum = np.fft.rfft(centered * analysis_weight, n=fft_length)
        response = gaussian_bandpass_response(
            frequency_axis,
            float(lower_frequency[window_index]),
            float(upper_frequency[window_index]),
            sample_interval,
        )
        filtered = np.fft.irfft(spectrum * response, n=fft_length)[:window_samples]

        accumulator[begin : end + 1] += filtered * analysis_weight
        coverage_weight[begin : end + 1] += weight
        coverage_count[begin : end + 1] += 1

    if np.any(coverage_count < 1) or np.any(coverage_weight <= 0.0):
        raise ValidationError(
            _("the window schedule failed to cover every sample with a positive synthesis weight"),
            details={"parameter": "step_length"},
        )

    normalized = accumulator / coverage_weight
    if value_scale == 0.0:
        filtered_values = np.zeros(total, dtype=float)
    else:
        filtered_values = normalized * value_scale

    windows = np.column_stack(
        (
            window_centers,
            lower_frequency,
            upper_frequency,
            center_frequency,
            coordinates[start_indices],
            coordinates[end_indices],
            start_indices.astype(float),
            end_indices.astype(float),
        )
    )

    return FrequencyStabilizationResult(
        coordinates=coordinates,
        filtered=filtered_values,
        coverage_count=coverage_count,
        coverage_weight=coverage_weight,
        windows=windows,
        lower_frequency=lower_frequency,
        upper_frequency=upper_frequency,
        center_frequency=center_frequency,
    )
