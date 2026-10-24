# =============================================================================
# FILE: stratigraphy/plot_digitizer.py
# =============================================================================
"""
图像数字化（Plot Digitizer）：把已发表的图变成可用的数值序列。

做什么
------
已发表的论文里常常只有一张图——地层柱状图、δ13C 曲线、伽马测井曲线——原始
数据却没有附上。本模块按 Acycle（https://github.com/mingsongli/acycle，GPL）
的 "Plot Digitizer"（官网 Examples #13）的做法，从**位图**里把那条线抠出来：

1. 用户在图上点四个参考点：x 轴起点、x 轴终点、y 轴起点、y 轴终点，并给出
   这四个位置对应的数据值；
2. 模块据此建立"像素坐标 → 数据坐标"的仿射标定；
3. 模块把曲线从背景里分离出来，逐列（曲线竖直占优时逐行）取一个点，再经标定
   换算成数据坐标，输出 (x, y) 序列。

依据与独立性
------------
流程参考 Acycle 的 ``code/guicode/plotdigitizer.m``：四点标定、"每个横截面取
一个点"的走向，以及"深色像素 = 墨迹"的二值化思路。**Acycle 是 GPL 软件，此处
没有翻译它的 MATLAB 源码**，而是按上面的规格用通用图像处理方法独立实现：

* 阈值用 **Otsu** 自动估计（Acycle 用的是分位数截断），并用中位数判定极性；
* 曲线提取用"掩膜 + 连续性选择 + 强度加权质心"，不依赖连通域标记；
* 贯穿绘图区的坐标轴框与网格线会被自动剔除，以免被当成曲线。

坐标约定
--------
* **像素坐标**：0 基，x 向右增大，y 向下增大（与 numpy 数组下标一致，也与
  ``imshow`` 的显示方向一致）。数组的行号就是 y 像素。
* **标定是仿射的**::

      x_data = x0值 + (x1值 - x0值) * (x像素 - x0像素) / (x1像素 - x0像素)
      y_data = y0值 + (y1值 - y0值) * (y像素 - y0像素) / (y1像素 - y0像素)

  因为 y 像素向下增大，而地层柱状图的 y 轴向上增大，用户只要把"y 轴起点"
  点在**下**端、"y 轴终点"点在上端，映射自动就是对的；反过来点也照样成立
  ——标定只认用户给的数值，不认方向。
* **绘图区**：四个参考点张成的矩形。x 像素取两个 x 参考点的横坐标范围，
  y 像素取两个 y 参考点的纵坐标范围。抠线只在这个矩形内进行。

阈值怎么定
----------
* ``threshold=None``（默认）：先用 Otsu 在归一化灰度直方图上求分割阈值，再看
  **中位数**落在阈值的哪一侧来判定极性——中位数在阈值之上说明背景是亮的、
  墨迹是暗的，取 ``灰度 <= 阈值``；反之取 ``灰度 >= 阈值``。``invert=True``
  把这个自动判定整个翻过来。
* ``threshold`` 给定：按**暗墨迹**处理，取 ``灰度 <= 阈值``；``invert=True``
  时取 ``灰度 >= 阈值``。阈值是归一化到 [0, 1] 的灰度值。
* ``colour`` 给定：改用颜色匹配（RGB 空间欧氏距离 <= 0.30），此时
  ``threshold`` 与 ``invert`` 不再参与掩膜的构造。

输入 / 输出
-----------
* 输入：``image`` 是 numpy 数组（RGB 或灰度，二维或三维），**由调用方负责读
  图**；本模块不引入任何新依赖。
* 输出：``DigitizedSeries``，含同长的 ``x`` / ``y`` 两个一维数组。

已知限制
--------
* 假定绘图区里只有**一条**曲线。曲线交叉或同时有多条线时，逐列取点会挑错；
  此时请用 ``colour`` 指定颜色，或者分开抠。
* 贯穿绘图区的横线/竖线（坐标轴框、网格线）会被自动剔除，以免它们被当成
  曲线；代价是**恰好横贯整个绘图区**的水平曲线也会被一并剔除。
* 阶梯状剖面的"竖直台阶"那一列上，取到的是台阶上下端之间的中点；采样时应
  避开台阶所在的列。
* 核心算法（标定与抠线）与 Qt 完全无关，可以在无头环境下测试；只有
  :func:`build_digitizer_dialog` 需要 PyQt6，且它的导入是延迟的。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from config.i18n import _
from utils.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - 只为类型检查器存在，运行时不导入 Qt
    from PyQt6.QtWidgets import QDialog

logger = logging.getLogger(__name__)

__all__ = [
    "AxisCalibration",
    "DigitizedSeries",
    "build_digitizer_dialog",
    "calibrate_axes",
    "digitize_curve",
    "digitize_profile",
]

# =============================================================================
# 常量
# =============================================================================

#: RGB 转灰度的权重（与 Acycle ``toGray`` 相同）。
_GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)

#: 判定"贯穿绘图区"的横线/竖线时要求的覆盖率。
_SPAN_COVERAGE = 0.90

#: 竖直占优（曲线比绘图区更高）的判据，与 Acycle 的 1.25 一致。
_VERTICAL_MAJOR_RATIO = 1.25

#: 颜色匹配的最大 RGB 欧氏距离（灰度与颜色都归一化到 [0, 1]）。
_COLOUR_RADIUS = 0.30

#: 抠出来的点少于这个数，就认为没有找到曲线。
_MIN_POINTS = 2


# =============================================================================
# 标定
# =============================================================================


@dataclass(frozen=True)
class AxisCalibration:
    """
    像素坐标到数据坐标的映射。

    标定是**仿射**的：x 与 y 各自由两个参考点确定，中间线性插值。像素坐标
    0 基、y 向下增大；数据坐标的方向完全由用户给出的数值决定，因此 y 轴
    向上还是向下增长都能表达。

    Attributes:
        x_pixels: 两个 x 参考点的像素横坐标。
        x_values: 两个 x 参考点的数据横坐标。
        y_pixels: 两个 y 参考点的像素纵坐标。
        y_values: 两个 y 参考点的数据纵坐标。
    """

    x_pixels: tuple[float, float]
    x_values: tuple[float, float]
    y_pixels: tuple[float, float]
    y_values: tuple[float, float]

    def __post_init__(self) -> None:
        """校验参考点：像素对不能重合，否则换算会除零。"""
        for name, pair in (
            ("x_pixels", self.x_pixels),
            ("x_values", self.x_values),
            ("y_pixels", self.y_pixels),
            ("y_values", self.y_values),
        ):
            if len(pair) != 2:
                raise ValidationError(
                    _("axis calibration field {0} must hold exactly two numbers").format(name),
                    details={"field": name, "length": len(pair)},
                )
            if not all(math.isfinite(float(value)) for value in pair):
                raise ValidationError(
                    _("axis calibration field {0} must be finite").format(name),
                    details={"field": name},
                )
        if float(self.x_pixels[0]) == float(self.x_pixels[1]):
            raise ValidationError(
                _("the two x reference points share a pixel column, so the x axis cannot be calibrated"),
                details={"x_pixels": [float(self.x_pixels[0]), float(self.x_pixels[1])]},
            )
        if float(self.y_pixels[0]) == float(self.y_pixels[1]):
            raise ValidationError(
                _("the two y reference points share a pixel row, so the y axis cannot be calibrated"),
                details={"y_pixels": [float(self.y_pixels[0]), float(self.y_pixels[1])]},
            )

    def to_data(self, x_pixel: float, y_pixel: float) -> tuple[float, float]:
        """
        把一个像素坐标换算成数据坐标。

        Parameters:
            x_pixel: 像素横坐标（0 基）。
            y_pixel: 像素纵坐标（0 基，向下增大）。

        Returns:
            tuple[float, float]: (数据横坐标, 数据纵坐标)。
        """
        x_span = float(self.x_pixels[1]) - float(self.x_pixels[0])
        y_span = float(self.y_pixels[1]) - float(self.y_pixels[0])
        x_value = (
            float(self.x_values[0])
            + (float(x_pixel) - float(self.x_pixels[0])) * (float(self.x_values[1]) - float(self.x_values[0])) / x_span
        )
        y_value = (
            float(self.y_values[0])
            + (float(y_pixel) - float(self.y_pixels[0])) * (float(self.y_values[1]) - float(self.y_values[0])) / y_span
        )
        return (x_value, y_value)

    @classmethod
    def from_reference_points(cls, points: Sequence[Sequence[float]]) -> AxisCalibration:
        """
        由四个参考点构造标定。

        Parameters:
            points: 四个点，每个点是 ``(x_pix, y_pix, x_val, y_val)`` 四元组，
                顺序为：x 轴起点、x 轴终点、y 轴起点、y 轴终点。前两个点提供
                像素横坐标与数据横坐标，后两个点提供像素纵坐标与数据纵坐标；
                每个四元组里用不到的那一半数值会被忽略。

        Returns:
            AxisCalibration: 构造好的标定。

        Raises:
            ValidationError: 点数不是 4、某个点不是四元组、数值非有限，或两个
                同轴参考点重合。
        """
        rows = _validated_reference_points(points)
        return cls(
            x_pixels=(float(rows[0][0]), float(rows[1][0])),
            x_values=(float(rows[0][2]), float(rows[1][2])),
            y_pixels=(float(rows[2][1]), float(rows[3][1])),
            y_values=(float(rows[2][3]), float(rows[3][3])),
        )


def calibrate_axes(points: Sequence[Sequence[float]]) -> AxisCalibration:
    """
    用四个参考点建立像素到数据的标定。

    这是 :meth:`AxisCalibration.from_reference_points` 的函数形式，方便直接
    从界面或脚本调用。

    Parameters:
        points: 四个 ``(x_pix, y_pix, x_val, y_val)`` 四元组，顺序为 x 轴起点、
            x 轴终点、y 轴起点、y 轴终点。

    Returns:
        AxisCalibration: 标定对象。

    Raises:
        ValidationError: 点数不是 4、某个点不是四元组、数值非有限，或两个
            同轴参考点重合（会除零）。
    """
    return AxisCalibration.from_reference_points(points)


def _validated_reference_points(points: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    """检查四个参考点的个数、形状与数值，返回 float 元组列表。"""
    if points is None:
        raise ValidationError(_("four reference points are required to calibrate the axes"))
    try:
        rows = list(points)
    except TypeError as exc:
        raise ValidationError(_("four reference points are required to calibrate the axes")) from exc
    if len(rows) != 4:
        raise ValidationError(
            _("four reference points are required to calibrate the axes; got {0}").format(len(rows)),
            details={"n_points": len(rows)},
        )
    cleaned: list[tuple[float, ...]] = []
    for index, row in enumerate(rows):
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                _("reference point {0} must be a sequence of numbers").format(index + 1),
                details={"index": int(index)},
            ) from exc
        if len(values) != 4:
            raise ValidationError(
                _("reference point {0} must hold four numbers (x_pix, y_pix, x_val, y_val)").format(index + 1),
                details={"index": int(index), "length": len(values)},
            )
        if not all(math.isfinite(value) for value in values):
            raise ValidationError(
                _("reference point {0} must be finite").format(index + 1),
                details={"index": int(index)},
            )
        cleaned.append(tuple(values))
    return cleaned


# =============================================================================
# 数字化结果
# =============================================================================


@dataclass(frozen=True)
class DigitizedSeries:
    """
    抠出来的数值序列。

    Attributes:
        x: 数据横坐标，一维数组。
        y: 数据纵坐标，一维数组，与 ``x`` 等长。
    """

    x: npt.NDArray
    y: npt.NDArray

    def __post_init__(self) -> None:
        """检查 x 与 y 等长。"""
        if np.shape(self.x) != np.shape(self.y):
            raise ValidationError(
                _("the digitized x and y arrays must have the same shape"),
                details={"x_shape": list(np.shape(self.x)), "y_shape": list(np.shape(self.y))},
            )

    def as_pairs(self) -> list[tuple[float, float]]:
        """返回 [(x, y), ...] 形式的 Python 浮点列表。"""
        return [(float(x_value), float(y_value)) for x_value, y_value in zip(self.x, self.y, strict=True)]

    def summary(self) -> str:
        """返回一行人类可读的摘要，用于状态栏与日志。"""
        if self.x.size == 0:
            return _("no digitized points")
        return _("digitized points: {0}; x in [{1:.4g}, {2:.4g}]; y in [{3:.4g}, {4:.4g}]").format(
            int(self.x.size),
            float(np.min(self.x)),
            float(np.max(self.x)),
            float(np.min(self.y)),
            float(np.max(self.y)),
        )


# =============================================================================
# 抠线
# =============================================================================


def digitize_curve(
    image: npt.NDArray,
    calibration: AxisCalibration,
    threshold: float | None = None,
    invert: bool = False,
    colour: tuple[int, int, int] | None = None,
) -> DigitizedSeries:
    """
    把图里的一条曲线抠成数值序列。

    做法：先把图像转成归一化灰度，按阈值（或颜色）分离出"墨迹"，裁到标定
    确定的绘图区里，剔除贯穿绘图区的坐标轴框与网格线，然后沿绘图区的长轴
    方向逐个横截面取一个点——每个横截面里可能有几段墨迹，用与上一个点的
    连续性挑出最可能的那一段，并以该段像素的**强度加权质心**作为亚像素位置。
    最后用标定把像素坐标换算成数据坐标。

    Parameters:
        image: RGB 或灰度的二维/三维 numpy 数组（由调用方负责读图）。
        calibration: 像素到数据的标定。
        threshold: 墨迹的灰度阈值（归一化到 [0, 1]）。``None`` 表示用 Otsu
            自动估计，并按中位数判定极性。
        invert: ``True`` 时把墨迹判成"比背景亮"的像素。
        colour: 只抠这个 RGB 颜色（0..255）的像素；给出后 ``threshold`` 与
            ``invert`` 不再参与掩膜的构造。

    Returns:
        DigitizedSeries: 按 x 排序的 (x, y) 序列。

    Raises:
        ValidationError: 图像不合法、标定不合法、绘图区内找不到墨迹，或找到的
            点太少。
    """
    if not isinstance(calibration, AxisCalibration):
        raise ValidationError(
            _("calibration must be an AxisCalibration instance"),
            details={"type": type(calibration).__name__},
        )
    gray, rgb = _as_gray(image)
    mask, weights = _build_ink(gray, rgb, threshold, invert, colour)
    region, x_offset, y_offset = _plot_region(mask, calibration)
    region = _drop_spanning_lines(region)
    region_weights = weights[
        y_offset : y_offset + region.shape[0],
        x_offset : x_offset + region.shape[1],
    ]
    if not region.any():
        raise ValidationError(
            _("no ink was found inside the calibrated plotting area"),
            details={"plot_width": int(region.shape[1]), "plot_height": int(region.shape[0])},
        )
    horizontal_major = region.shape[0] <= _VERTICAL_MAJOR_RATIO * region.shape[1]
    traced = _trace_curve(region, region_weights, horizontal_major)
    if len(traced) < _MIN_POINTS:
        raise ValidationError(
            _("fewer than {0} points were found inside the calibrated plotting area").format(_MIN_POINTS),
            details={"n_points": len(traced)},
        )
    pixels = np.asarray(traced, dtype=float)
    pixels[:, 0] += float(x_offset)
    pixels[:, 1] += float(y_offset)
    pixels = pixels[np.argsort(pixels[:, 0], kind="stable")]
    values = [calibration.to_data(float(x_pixel), float(y_pixel)) for x_pixel, y_pixel in pixels]
    x_values = np.asarray([value[0] for value in values], dtype=float)
    y_values = np.asarray([value[1] for value in values], dtype=float)
    logger.debug("digitize_curve: %s", DigitizedSeries(x=x_values, y=y_values).summary())
    return DigitizedSeries(x=x_values, y=y_values)


def digitize_profile(
    image: npt.NDArray,
    calibration: AxisCalibration,
    column_pixels: Sequence[int],
    threshold: float | None = None,
    invert: bool = False,
) -> DigitizedSeries:
    """
    按给定的像素列抠出剖面（柱状图、测井曲线）上的垂直位置。

    与 :func:`digitize_curve` 的区别是：这里不是"每个横截面取一个点"，而是
    只在调用方指定的若干列上取点——适合"每隔几毫米采一次"的剖面采样。每一列
    里若有几段墨迹，用与上一列的连续性挑出最可能的那一段。

    Parameters:
        image: RGB 或灰度的二维/三维 numpy 数组。
        calibration: 像素到数据的标定。
        column_pixels: 要采样的像素列号（0 基）。要给一段范围，先用
            ``range(start, stop)`` 展开成整数列表。
        threshold: 墨迹的灰度阈值；``None`` 表示自动估计。
        invert: ``True`` 时把墨迹判成"比背景亮"的像素。

    Returns:
        DigitizedSeries: 按 x 排序的 (x, y) 序列；某一列没有墨迹时该列被跳过。

    Raises:
        ValidationError: 列号列表为空、含非整数或越界，图像不合法，或所有列都
            没有墨迹。
    """
    if not isinstance(calibration, AxisCalibration):
        raise ValidationError(
            _("calibration must be an AxisCalibration instance"),
            details={"type": type(calibration).__name__},
        )
    gray, rgb = _as_gray(image)
    columns = _validated_columns(column_pixels, gray.shape[1])
    mask, weights = _build_ink(gray, rgb, threshold, invert, None)
    region, x_offset, y_offset = _plot_region(mask, calibration)
    region = _drop_spanning_lines(region)
    region_weights = weights[
        y_offset : y_offset + region.shape[0],
        x_offset : x_offset + region.shape[1],
    ]
    points: list[tuple[float, float]] = []
    previous: float | None = None
    for column in sorted(set(columns)):
        local = column - x_offset
        if local < 0 or local >= region.shape[1]:
            continue
        hits = np.flatnonzero(region[:, local])
        if hits.size == 0:
            continue
        runs = _runs(hits)
        if len(runs) == 1:
            run = runs[0]
        elif previous is None:
            run = max(runs, key=lambda block: (int(block.size), -float(block[0])))
        else:
            run = min(runs, key=lambda block: (abs(float(block.mean()) - previous), -int(block.size)))
        value = _weighted_centroid(run, region_weights[:, local])
        previous = value
        points.append((float(column), value + float(y_offset)))
    if not points:
        raise ValidationError(
            _("no ink was found in the requested pixel columns"),
            details={"n_columns": len(columns)},
        )
    pixels = np.asarray(points, dtype=float)
    pixels = pixels[np.argsort(pixels[:, 0], kind="stable")]
    values = [calibration.to_data(float(x_pixel), float(y_pixel)) for x_pixel, y_pixel in pixels]
    x_values = np.asarray([value[0] for value in values], dtype=float)
    y_values = np.asarray([value[1] for value in values], dtype=float)
    logger.debug("digitize_profile: %s", DigitizedSeries(x=x_values, y=y_values).summary())
    return DigitizedSeries(x=x_values, y=y_values)


# =============================================================================
# 图像处理内部函数
# =============================================================================


def _as_gray(image: npt.NDArray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    把输入图像转成归一化灰度，并（如果是彩色）保留归一化 RGB。

    Parameters:
        image: 二维灰度或三维 RGB/RGBA 的 numpy 数组。

    Returns:
        tuple: ``(灰度数组, RGB 数组或 None)``，两者都归一化到 [0, 1]。
    """
    if image is None:
        raise ValidationError(_("an image array is required"))
    array = np.asarray(image)
    if array.dtype == bool or not np.issubdtype(array.dtype, np.number):
        raise ValidationError(
            _("the image must be a numeric numpy array"),
            details={"dtype": str(array.dtype)},
        )
    if array.ndim not in (2, 3):
        raise ValidationError(
            _("the image must be a 2-D grayscale or a 3-D RGB array"),
            details={"shape": list(array.shape)},
        )
    if array.size == 0:
        raise ValidationError(_("the image is empty"), details={"shape": list(array.shape)})
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValidationError(
            _("the image must have 1, 3 or 4 channels"),
            details={"channels": int(array.shape[2])},
        )
    values = _normalise(array)
    if values.ndim == 2:
        return values, None
    if values.shape[2] == 1:
        return values[:, :, 0], None
    rgb = values[:, :, :3]
    gray = _GRAY_WEIGHTS[0] * rgb[:, :, 0] + _GRAY_WEIGHTS[1] * rgb[:, :, 1] + _GRAY_WEIGHTS[2] * rgb[:, :, 2]
    return gray, rgb


def _normalise(array: np.ndarray) -> np.ndarray:
    """把整数或浮点图像归一化到 [0, 1]。"""
    if np.issubdtype(array.dtype, np.integer):
        maximum = int(np.iinfo(array.dtype).max)
        scale = 255.0 if maximum > 255 else float(maximum)
    else:
        peak = float(np.max(array))
        scale = 255.0 if peak > 1.0 else 1.0
    values = np.asarray(array, dtype=float) / scale
    return np.clip(values, 0.0, 1.0)


def _otsu_threshold(gray: np.ndarray) -> float:
    """
    用 Otsu 法在归一化灰度直方图上求分割阈值。

    Parameters:
        gray: 归一化到 [0, 1] 的灰度数组。

    Returns:
        float: 阈值；常数图像返回接近 0 的值（因而选不出任何"暗"像素）。
    """
    histogram, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.5
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(histogram).astype(float)
    weight_high = total - weight_low
    moment_low = np.cumsum(histogram * centres)
    moment_total = float(moment_low[-1])
    safe_low = np.where(weight_low > 0.0, weight_low, 1.0)
    safe_high = np.where(weight_high > 0.0, weight_high, 1.0)
    mean_low = np.where(weight_low > 0.0, moment_low / safe_low, 0.0)
    mean_high = np.where(weight_high > 0.0, (moment_total - moment_low) / safe_high, 0.0)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between = np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
    return float(centres[int(np.argmax(between))])


def _validated_colour(colour: Sequence[int]) -> np.ndarray:
    """检查颜色参数，返回归一化到 [0, 1] 的 RGB 向量。"""
    try:
        values = [int(value) for value in colour]
    except (TypeError, ValueError) as exc:
        raise ValidationError(_("colour must be an (R, G, B) triple")) from exc
    if len(values) != 3:
        raise ValidationError(
            _("colour must be an (R, G, B) triple"),
            details={"length": len(values)},
        )
    if any(value < 0 or value > 255 for value in values):
        raise ValidationError(
            _("colour components must be between 0 and 255"),
            details={"colour": values},
        )
    return np.asarray(values, dtype=float) / 255.0


def _build_ink(
    gray: np.ndarray,
    rgb: np.ndarray | None,
    threshold: float | None,
    invert: bool,
    colour: tuple[int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    分离"墨迹"与背景。

    Returns:
        tuple: ``(墨迹掩膜, 墨迹强度权重)``；权重在墨迹处大、背景处小，用于
        计算亚像素的质心。
    """
    if colour is not None:
        if rgb is None:
            raise ValidationError(_("colour matching needs an RGB image"))
        target = _validated_colour(colour)
        distance = np.linalg.norm(rgb - target.reshape(1, 1, 3), axis=2)
        mask = distance <= _COLOUR_RADIUS
        weights = np.clip(1.0 - distance / _COLOUR_RADIUS, 0.0, 1.0)
        return mask, weights
    if threshold is None:
        cut = _otsu_threshold(gray)
        bright = float(np.median(gray)) <= cut
        if invert:
            bright = not bright
    else:
        cut = float(threshold)
        if not math.isfinite(cut) or not 0.0 <= cut <= 1.0:
            raise ValidationError(
                _("threshold must be a number between 0 and 1"),
                details={"threshold": cut},
            )
        bright = bool(invert)
    mask = gray >= cut if bright else gray <= cut
    weights = np.clip(gray, 0.0, 1.0) if bright else np.clip(1.0 - gray, 0.0, 1.0)
    return mask, weights


def _plot_region(mask: np.ndarray, calibration: AxisCalibration) -> tuple[np.ndarray, int, int]:
    """
    裁出四个参考点张成的矩形。

    Returns:
        tuple: ``(区域掩膜, x 偏移, y 偏移)``；区域内的坐标加上偏移即为整图
        的像素坐标。
    """
    height, width = mask.shape
    x_low = max(0, min(math.floor(min(calibration.x_pixels)), width - 1))
    x_high = max(0, min(math.ceil(max(calibration.x_pixels)), width - 1))
    y_low = max(0, min(math.floor(min(calibration.y_pixels)), height - 1))
    y_high = max(0, min(math.ceil(max(calibration.y_pixels)), height - 1))
    if x_high <= x_low or y_high <= y_low:
        raise ValidationError(
            _("the reference points do not enclose an area inside the image"),
            details={"x_range": [x_low, x_high], "y_range": [y_low, y_high]},
        )
    return mask[y_low : y_high + 1, x_low : x_high + 1], x_low, y_low


def _drop_spanning_lines(region: np.ndarray) -> np.ndarray:
    """
    剔除贯穿绘图区的横线与竖线。

    坐标轴框与网格线会横贯整个绘图区，若不剔除，逐列取点时它们会与真正的
    曲线争夺同一个位置。判据是：某一行（或某一列）的墨迹覆盖了绘图区该方向
    的 90% 以上。代价是**恰好横贯整个绘图区**的水平曲线也会被剔除——那种图
    请改用 :func:`digitize_profile` 按列取值。
    """
    if not region.any():
        return region
    cleaned = region.copy()
    n_columns = cleaned.shape[1]
    if n_columns > 0:
        row_coverage = cleaned.sum(axis=1) / float(n_columns)
        cleaned[row_coverage >= _SPAN_COVERAGE, :] = False
    n_rows = cleaned.shape[0]
    if n_rows > 0:
        column_coverage = cleaned.sum(axis=0) / float(n_rows)
        cleaned[:, column_coverage >= _SPAN_COVERAGE] = False
    return cleaned


def _runs(hits: np.ndarray) -> list[np.ndarray]:
    """把排好序的像素下标切成若干段连续的区间。"""
    if hits.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(hits) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [hits.size - 1]))
    return [hits[start : end + 1] for start, end in zip(starts, ends, strict=True)]


def _weighted_centroid(rows: np.ndarray, column: np.ndarray) -> float:
    """求一段连续像素的强度加权质心（亚像素位置）。"""
    values = np.asarray(column, dtype=float)[rows]
    total = float(values.sum())
    if total <= 0.0:
        return float(np.mean(rows))
    return float(np.sum(rows * values) / total)


def _trace_curve(
    region: np.ndarray,
    weights: np.ndarray,
    horizontal_major: bool,
) -> list[tuple[float, float]]:
    """
    沿绘图区的长轴方向逐个横截面取一个点。

    Parameters:
        region: 绘图区的墨迹掩膜。
        weights: 与 ``region`` 同形的墨迹强度权重。
        horizontal_major: ``True`` 表示逐列取点（曲线以 x 为主），
            ``False`` 表示逐行取点。

    Returns:
        list: ``[(x 像素, y 像素), ...]``，坐标是绘图区内的局部坐标。
    """
    n_major = region.shape[1] if horizontal_major else region.shape[0]
    points: list[tuple[float, float]] = []
    previous: float | None = None
    previous_index: int | None = None
    step = 0.0
    for index in range(n_major):
        line = region[:, index] if horizontal_major else region[index, :]
        hits = np.flatnonzero(line)
        if hits.size == 0:
            continue
        runs = _runs(hits)
        if len(runs) == 1:
            run = runs[0]
        elif previous is None:
            run = max(runs, key=lambda block: (int(block.size), -float(block[0])))
        else:
            predicted = previous + step if previous_index is not None and index == previous_index + 1 else previous
            run = min(runs, key=lambda block: (abs(float(np.mean(block)) - predicted), -int(block.size)))
        column = weights[:, index] if horizontal_major else weights[index, :]
        value = _weighted_centroid(run, column)
        if previous is not None and previous_index is not None and index == previous_index + 1:
            step = value - previous
        previous = value
        previous_index = index
        if horizontal_major:
            points.append((float(index), value))
        else:
            points.append((value, float(index)))
    return points


def _validated_columns(column_pixels: Sequence[int], width: int) -> list[int]:
    """检查要采样的像素列号。"""
    if column_pixels is None:
        raise ValidationError(_("at least one pixel column is required"))
    try:
        raw = list(column_pixels)
    except TypeError as exc:
        raise ValidationError(_("at least one pixel column is required")) from exc
    if not raw:
        raise ValidationError(_("at least one pixel column is required"))
    columns: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValidationError(
                _("pixel columns must be integers"),
                details={"value": repr(item)},
            )
        value = int(item)
        if value < 0 or value >= width:
            raise ValidationError(
                _("pixel column {0} is outside the image (0..{1})").format(value, width - 1),
                details={"column": value, "width": int(width)},
            )
        columns.append(value)
    return columns


# =============================================================================
# 交互外壳（薄薄一层，PyQt6 延迟导入）
# =============================================================================


def build_digitizer_dialog(image: npt.NDArray, parent=None) -> QDialog:
    """
    构造一个 PyQt6 对话框，让用户在图上点四个参考点。

    对话框的用法：

    1. 在图上依次点四个参考点——x 轴起点、x 轴终点、y 轴起点、y 轴终点；
    2. 在四个输入框里填这四个位置对应的**数据**值；
    3. 按"标定"生成 :class:`AxisCalibration`，按"抠线"运行
       :func:`digitize_curve`。

    对话框提供两个读取方法：``calibration()`` 与 ``series()``。

    Parameters:
        image: RGB 或灰度的 numpy 数组（由调用方负责读图）。
        parent: 父控件，可为 ``None``。

    Returns:
        QDialog: 对话框实例。

    Raises:
        ImportError: 没有安装 PyQt6。
        ValidationError: 图像不合法。

    Notes:
        PyQt6 的导入放在函数体内，因此本模块在没有 Qt 的环境里依然可以
        import，核心算法（标定与抠线）与 Qt 完全无关。
    """
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
        )
    except ImportError as exc:
        raise ImportError(
            "PyQt6 is required for the plot digitizer dialog; install it with: pip install PyQt6"
        ) from exc

    gray, rgb = _as_gray(image)
    height, width = gray.shape
    if rgb is not None:
        buffer = np.ascontiguousarray(np.round(rgb * 255.0).astype(np.uint8))
        picture = QImage(buffer.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    else:
        buffer = np.ascontiguousarray(np.round(gray * 255.0).astype(np.uint8))
        picture = QImage(buffer.data, width, height, width, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(picture)

    class ReferenceCanvas(QLabel):
        """可点击的图像视图，记录四个参考点的像素坐标。"""

        def __init__(self, pixmap: QPixmap, parent=None) -> None:
            super().__init__(parent)
            self.setPixmap(pixmap)
            self.setFixedSize(pixmap.size())
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.reference_points: list[tuple[int, int]] = []

        def mousePressEvent(self, event) -> None:
            if len(self.reference_points) >= 4:
                return
            position = event.position().toPoint()
            self.reference_points.append((int(position.x()), int(position.y())))
            self.update()

        def paintEvent(self, event) -> None:
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setPen(QPen(QColor(220, 30, 30), 2))
            for index, (x_pixel, y_pixel) in enumerate(self.reference_points):
                painter.drawEllipse(x_pixel - 5, y_pixel - 5, 10, 10)
                painter.drawText(x_pixel + 8, y_pixel - 8, str(index + 1))
            painter.end()

    class DigitizerDialog(QDialog):
        """图像数字化对话框。"""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setWindowTitle(_("Plot digitizer"))
            self._image = image
            self._buffer = buffer
            self._calibration: AxisCalibration | None = None
            self._series: DigitizedSeries | None = None

            layout = QVBoxLayout(self)
            scroll = QScrollArea(self)
            self._canvas = ReferenceCanvas(pixmap, scroll)
            scroll.setWidget(self._canvas)
            scroll.setWidgetResizable(False)
            layout.addWidget(scroll)

            form = QFormLayout()
            self._spins: list[QDoubleSpinBox] = []
            for caption, value in (
                (_("x value at the first reference point"), 0.0),
                (_("x value at the second reference point"), 1.0),
                (_("y value at the third reference point"), 0.0),
                (_("y value at the fourth reference point"), 1.0),
            ):
                spin = QDoubleSpinBox(self)
                spin.setRange(-1.0e12, 1.0e12)
                spin.setDecimals(6)
                spin.setValue(value)
                form.addRow(caption, spin)
                self._spins.append(spin)
            layout.addLayout(form)

            actions = QHBoxLayout()
            calibrate_button = QPushButton(_("Calibrate axes"), self)
            digitize_button = QPushButton(_("Digitize curve"), self)
            actions.addWidget(calibrate_button)
            actions.addWidget(digitize_button)
            layout.addLayout(actions)

            self._status = QLabel("", self)
            layout.addWidget(self._status)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                self,
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

            calibrate_button.clicked.connect(self._on_calibrate)
            digitize_button.clicked.connect(self._on_digitize)

        def calibration(self) -> AxisCalibration | None:
            """返回标定；还没按过"标定"时返回 ``None``。"""
            return self._calibration

        def series(self) -> DigitizedSeries | None:
            """返回抠出来的序列；还没按过"抠线"时返回 ``None``。"""
            return self._series

        def _on_calibrate(self) -> None:
            points = self._canvas.reference_points
            if len(points) < 4:
                self._status.setText(_("click four reference points on the image"))
                return
            x_first, x_second, y_third, y_fourth = (spin.value() for spin in self._spins)
            references = [
                (float(points[0][0]), float(points[0][1]), x_first, 0.0),
                (float(points[1][0]), float(points[1][1]), x_second, 0.0),
                (float(points[2][0]), float(points[2][1]), 0.0, y_third),
                (float(points[3][0]), float(points[3][1]), 0.0, y_fourth),
            ]
            try:
                self._calibration = AxisCalibration.from_reference_points(references)
            except ValidationError as exc:
                self._status.setText(str(exc))
                return
            self._status.setText(_("axes calibrated"))

        def _on_digitize(self) -> None:
            if self._calibration is None:
                self._status.setText(_("calibrate the axes first"))
                return
            try:
                self._series = digitize_curve(self._image, self._calibration)
            except ValidationError as exc:
                self._status.setText(str(exc))
                return
            self._status.setText(self._series.summary())

    return DigitizerDialog(parent)
