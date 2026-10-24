# =============================================================================
# FILE: tests/golden/test_plot_digitizer.py
# =============================================================================
"""
``stratigraphy.plot_digitizer``（图像数字化 / Plot Digitizer）的测试。

测试围绕四条主线组织：

* **合成真值。** 自己造一张白底图，用 numpy 画一条已知方程的直线
  （像素坐标 ``y = -0.5 x + 100``），把坐标轴标定好，然后断言
  ``digitize_curve`` 抠出来的点**落在那条直线上**——判据是"像素当量"的
  残差：把数据坐标换回像素坐标，与理论直线的距离必须小于 1 个像素。
  同时断言点数与曲线横跨的列数相符（一条逐列取点的曲线，每列一个点）。
  阶梯图同理：``digitize_profile`` 必须把已知的台阶高度还原出来。

* **标定的口径。** ``to_data`` 的换算在参考点上必须精确成立；两个同轴参考
  点重合会让换算除零，必须抛 ``utils.exceptions.ValidationError``；参考点
  个数不对（四元组不是四个点）同样必须抛 ``ValidationError``。

* **极性与阈值。** 默认阈值由 Otsu 自动估计，极性由中位数判定：暗底白线
  必须也能抠出来；显式阈值加 ``invert=True`` 必须把"比背景亮"的像素当作
  墨迹；给定 ``colour`` 时只抠那个颜色。

* **可测试性。** 核心算法与 Qt 无关，因此本文件在 ``QT_QPA_PLATFORM=offscreen``
  的无头环境下跑。模块本身必须在**没有 PyQt6** 的环境里也能 import——这一条
  用一个屏蔽了 PyQt6 的子进程来证明，而不是靠"这台机器恰好装了 Qt"。

参考来源（断言中引用的数值出处）
--------------------------------
* Li, M. S. et al., *Acycle: Plot Digitizer*, https://github.com/mingsongli/acycle
  （Examples #13；四点标定与"逐截面取一点"的走向。本仓库是按规格独立实现，
  未翻译其 GPL 源码。）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"plot_digitizer"}


import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stratigraphy.plot_digitizer import (
    AxisCalibration,
    DigitizedSeries,
    build_digitizer_dialog,
    calibrate_axes,
    digitize_curve,
    digitize_profile,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 仓库根目录（子进程测试要靠它把仓库放进 sys.path）。
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 合成直线在**像素坐标**里的方程：y = 斜率 * x + 截距。
_LINE_SLOPE = -0.5
_LINE_INTERCEPT = 100.0

#: 合成直线横跨的像素列范围。
_LINE_X0 = 20
_LINE_X1 = 180

#: 画布尺寸（高、宽）。
_CANVAS_HEIGHT = 150
_CANVAS_WIDTH = 200

#: 判定"抠得准"时允许的残差（像素当量）。
_PIXEL_TOLERANCE = 1.0

#: 合成直线的标定：x 像素 (20, 180) -> 数据 x (0, 100)；
#: y 像素 (130, 5) -> 数据 y (0, 50)（y 像素向下增大，所以 y 标定的斜率为负）。
_LINE_REFERENCES = [
    (float(_LINE_X0), 90.0, 0.0, 0.0),
    (float(_LINE_X1), 10.0, 100.0, 0.0),
    (float(_LINE_X0), 130.0, 0.0, 0.0),
    (float(_LINE_X0), 5.0, 0.0, 50.0),
]

#: 合成直线在**数据坐标**里的方程（由上面的标定换算而来）。
_DATA_Y_AT_ZERO = 16.0
_DATA_SLOPE = 0.32

#: 阶梯图的画布尺寸与台阶。
_STEP_HEIGHT = 160
_STEP_WIDTH = 240
_STEP_PLATEAUS = ((20, 80, 120), (80, 140, 80), (140, 220, 40))

#: 阶梯图的标定：x 像素 (20, 220) -> 数据 x (0, 200)；
#: y 像素 (140, 20) -> 数据 y (0, 120)。
_STEP_REFERENCES = [
    (20.0, 120.0, 0.0, 0.0),
    (220.0, 40.0, 200.0, 0.0),
    (20.0, 140.0, 0.0, 0.0),
    (20.0, 20.0, 0.0, 120.0),
]

# =============================================================================
# 造图辅助
# =============================================================================


def _blank_canvas(height: int = _CANVAS_HEIGHT, width: int = _CANVAS_WIDTH, value: int = 255) -> np.ndarray:
    """造一张单色画布。"""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _draw_line(
    image: np.ndarray, slope: float, intercept: float, x0: int, x1: int, colour: tuple[int, int, int]
) -> None:
    """在画布上画一条直线（逐列取整，像素中心的整数位置）。"""
    height = image.shape[0]
    for x in range(x0, x1 + 1):
        y = round(slope * x + intercept)
        if 0 <= y < height:
            image[y, x] = colour


def _line_image(background: int = 255, ink: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """造一张白底（或黑底）上面画着合成直线的图。"""
    image = _blank_canvas(value=background)
    _draw_line(image, _LINE_SLOPE, _LINE_INTERCEPT, _LINE_X0, _LINE_X1, ink)
    return image


def _step_image() -> np.ndarray:
    """造一张阶梯图：三段平台加两段竖直台阶。"""
    image = _blank_canvas(height=_STEP_HEIGHT, width=_STEP_WIDTH)
    for x_start, x_end, y_pixel in _STEP_PLATEAUS:
        image[y_pixel, x_start : x_end + 1] = 0
    image[_STEP_PLATEAUS[1][2] : _STEP_PLATEAUS[0][2] + 1, _STEP_PLATEAUS[0][1]] = 0
    image[_STEP_PLATEAUS[2][2] : _STEP_PLATEAUS[1][2] + 1, _STEP_PLATEAUS[1][1]] = 0
    return image


def _to_pixels(calibration: AxisCalibration, x_values: np.ndarray, y_values: np.ndarray):
    """把数据坐标换回像素坐标（标定的逆），用来算像素级残差。"""
    x_pixels = calibration.x_pixels[0] + (np.asarray(x_values, dtype=float) - calibration.x_values[0]) * (
        calibration.x_pixels[1] - calibration.x_pixels[0]
    ) / (calibration.x_values[1] - calibration.x_values[0])
    y_pixels = calibration.y_pixels[0] + (np.asarray(y_values, dtype=float) - calibration.y_values[0]) * (
        calibration.y_pixels[1] - calibration.y_pixels[0]
    ) / (calibration.y_values[1] - calibration.y_values[0])
    return x_pixels, y_pixels


def _line_residual(calibration: AxisCalibration, series: DigitizedSeries) -> np.ndarray:
    """抠出来的点到理论直线的距离（像素当量）。"""
    x_pixels, y_pixels = _to_pixels(calibration, series.x, series.y)
    return np.abs(y_pixels - (_LINE_SLOPE * x_pixels + _LINE_INTERCEPT))


# =============================================================================
# 标定
# =============================================================================


def test_to_data_maps_pixels_onto_values():
    """标定是仿射的：参考点上精确成立，中间线性插值。"""
    calibration = AxisCalibration(
        x_pixels=(10.0, 110.0),
        x_values=(0.0, 100.0),
        y_pixels=(90.0, 10.0),
        y_values=(0.0, 80.0),
    )
    assert calibration.to_data(10.0, 90.0) == (0.0, 0.0)
    assert calibration.to_data(110.0, 10.0) == (100.0, 80.0)
    assert calibration.to_data(60.0, 50.0) == (50.0, 40.0)


def test_calibrate_axes_matches_from_reference_points():
    """函数形式与类方法给出同一个标定。"""
    from_points = AxisCalibration.from_reference_points(_LINE_REFERENCES)
    from_function = calibrate_axes(_LINE_REFERENCES)
    assert from_function == from_points
    assert from_function.x_pixels == (float(_LINE_X0), float(_LINE_X1))
    assert from_function.x_values == (0.0, 100.0)
    assert from_function.y_pixels == (130.0, 5.0)
    assert from_function.y_values == (0.0, 50.0)


def test_coincident_x_reference_points_raise():
    """两个 x 参考点落在同一列会让换算除零，必须拒绝。"""
    with pytest.raises(ValidationError):
        calibrate_axes(
            [
                (10.0, 5.0, 0.0, 0.0),
                (10.0, 6.0, 1.0, 0.0),
                (20.0, 90.0, 0.0, 0.0),
                (30.0, 10.0, 0.0, 1.0),
            ]
        )


def test_coincident_y_reference_points_raise():
    """两个 y 参考点落在同一行同样会除零。"""
    with pytest.raises(ValidationError):
        calibrate_axes(
            [
                (10.0, 5.0, 0.0, 0.0),
                (20.0, 6.0, 1.0, 0.0),
                (30.0, 90.0, 0.0, 0.0),
                (40.0, 90.0, 0.0, 1.0),
            ]
        )


def test_wrong_number_of_reference_points_raises():
    """参考点不是四个，或某个点不是四元组，都必须拒绝。"""
    with pytest.raises(ValidationError):
        calibrate_axes([(10.0, 5.0, 0.0, 0.0), (20.0, 6.0, 1.0, 0.0), (30.0, 90.0, 0.0, 0.0)])
    with pytest.raises(ValidationError):
        calibrate_axes(
            [
                (10.0, 5.0, 0.0),
                (20.0, 6.0, 1.0, 0.0),
                (30.0, 90.0, 0.0, 0.0),
                (40.0, 90.0, 0.0, 1.0),
            ]
        )


def test_non_finite_reference_point_raises():
    """参考点里出现 NaN/inf 必须拒绝。"""
    with pytest.raises(ValidationError):
        calibrate_axes(
            [
                (10.0, 5.0, 0.0, 0.0),
                (float("nan"), 6.0, 1.0, 0.0),
                (30.0, 90.0, 0.0, 0.0),
                (40.0, 90.0, 0.0, 1.0),
            ]
        )


# =============================================================================
# 合成真值：曲线
# =============================================================================


def test_digitize_curve_recovers_a_known_line():
    """合成真值：抠出来的点必须落在已知直线上，残差小于 1 个像素当量。"""
    image = _line_image()
    calibration = calibrate_axes(_LINE_REFERENCES)
    series = digitize_curve(image, calibration)

    assert isinstance(series, DigitizedSeries)
    assert series.x.size == series.y.size
    # 一条逐列取点的直线，每一列一个点。
    assert series.x.size == _LINE_X1 - _LINE_X0 + 1
    # 像素当量的残差。
    residual = _line_residual(calibration, series)
    assert residual.max() < _PIXEL_TOLERANCE
    # 数据坐标里也应当落在同一条直线上（1 像素 = 0.4 个数据单位）。
    expected = _DATA_Y_AT_ZERO + _DATA_SLOPE * series.x
    assert np.abs(series.y - expected).max() < _PIXEL_TOLERANCE * 0.4
    # x 单调不减，且端点落在标定的两端。
    assert np.all(np.diff(series.x) > 0.0)
    assert series.x[0] == pytest.approx(0.0)
    assert series.x[-1] == pytest.approx(100.0)


def test_digitized_series_pairs_and_summary():
    """``as_pairs`` 给出 (x, y) 列表，``summary`` 给出可读摘要。"""
    image = _line_image()
    calibration = calibrate_axes(_LINE_REFERENCES)
    series = digitize_curve(image, calibration)
    pairs = series.as_pairs()
    assert len(pairs) == series.x.size
    assert isinstance(pairs[0][0], float)
    summary = series.summary()
    assert str(series.x.size) in summary


def test_digitize_curve_ignores_the_axis_frame():
    """坐标轴框横贯绘图区，不能被当成曲线。"""
    image = _line_image()
    image[130, _LINE_X0 : _LINE_X1 + 1] = 0
    image[5:130, _LINE_X0] = 0
    calibration = calibrate_axes(_LINE_REFERENCES)
    series = digitize_curve(image, calibration)
    assert series.x.size >= _LINE_X1 - _LINE_X0
    assert _line_residual(calibration, series).max() < _PIXEL_TOLERANCE


def test_digitize_curve_finds_a_bright_line_on_a_dark_background():
    """极性由中位数自动判定：暗底白线也要抠得出来。"""
    image = _line_image(background=0, ink=(255, 255, 255))
    calibration = calibrate_axes(_LINE_REFERENCES)
    series = digitize_curve(image, calibration)
    assert series.x.size == _LINE_X1 - _LINE_X0 + 1
    assert _line_residual(calibration, series).max() < _PIXEL_TOLERANCE


def test_explicit_threshold_with_invert_selects_bright_ink():
    """显式阈值 + ``invert=True``：取"比背景亮"的像素。"""
    image = _line_image(background=0, ink=(255, 255, 255))
    calibration = calibrate_axes(_LINE_REFERENCES)
    series = digitize_curve(image, calibration, threshold=0.5, invert=True)
    assert series.x.size == _LINE_X1 - _LINE_X0 + 1
    assert _line_residual(calibration, series).max() < _PIXEL_TOLERANCE


def test_colour_matching_picks_the_requested_colour():
    """两条不同颜色的曲线，按颜色指定只抠其中一条。"""
    image = _blank_canvas()
    _draw_line(image, _LINE_SLOPE, _LINE_INTERCEPT, _LINE_X0, _LINE_X1, (255, 0, 0))
    _draw_line(image, _LINE_SLOPE, _LINE_INTERCEPT - 40.0, _LINE_X0, _LINE_X1, (0, 0, 255))
    calibration = calibrate_axes(_LINE_REFERENCES)
    series = digitize_curve(image, calibration, colour=(255, 0, 0))
    assert series.x.size > 0
    assert _line_residual(calibration, series).max() < _PIXEL_TOLERANCE


def test_colour_matching_needs_an_rgb_image():
    """灰度图没有颜色可选，必须报错而不是静默给出错误结果。"""
    gray_image = np.full((_CANVAS_HEIGHT, _CANVAS_WIDTH), 255, dtype=np.uint8)
    calibration = calibrate_axes(_LINE_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_curve(gray_image, calibration, colour=(0, 0, 0))


# =============================================================================
# 合成真值：剖面 / 阶梯图
# =============================================================================


def test_digitize_profile_recovers_step_heights():
    """阶梯图：每个平台上的采样列必须给出已知的台阶高度。"""
    image = _step_image()
    calibration = calibrate_axes(_STEP_REFERENCES)
    columns = [40, 60, 100, 120, 160, 180]
    series = digitize_profile(image, calibration, columns)

    assert series.x.size == len(columns)
    # 标定是 y_data = 140 - y_pixel，所以平台行 120/80/40 对应高度 20/60/100。
    expected = [20.0, 20.0, 60.0, 60.0, 100.0, 100.0]
    assert np.abs(series.y - np.asarray(expected)).max() < 0.5
    # x 也要落在正确的位置（x_data = x_pixel - 20）。
    assert np.abs(series.x - np.asarray([float(column - 20) for column in columns])).max() < 1e-9


def test_digitize_profile_rejects_bad_columns():
    """列号列表为空、含非整数或越界，都必须报错。"""
    image = _step_image()
    calibration = calibrate_axes(_STEP_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_profile(image, calibration, [])
    with pytest.raises(ValidationError):
        digitize_profile(image, calibration, [40.5])
    with pytest.raises(ValidationError):
        digitize_profile(image, calibration, [_STEP_WIDTH + 10])


# =============================================================================
# 边界
# =============================================================================


def test_blank_image_raises():
    """全白图里没有墨迹，必须报错。"""
    calibration = calibrate_axes(_LINE_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_curve(_blank_canvas(), calibration)


def test_black_image_raises():
    """全黑图里也没有墨迹。"""
    calibration = calibrate_axes(_LINE_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_curve(_blank_canvas(value=0), calibration)


def test_empty_image_raises():
    """空数组不是图像，必须报错。"""
    calibration = calibrate_axes(_LINE_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_curve(np.zeros((0, 0, 3), dtype=np.uint8), calibration)


def test_image_with_wrong_shape_raises():
    """一维数组不是图像。"""
    calibration = calibrate_axes(_LINE_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_curve(np.zeros(10, dtype=np.uint8), calibration)


def test_digitize_curve_requires_a_calibration():
    """标定对象类型不对时必须报错，而不是在换算时崩掉。"""
    with pytest.raises(ValidationError):
        digitize_curve(_line_image(), "not-a-calibration")


def test_out_of_range_threshold_raises():
    """阈值必须在 [0, 1] 之间。"""
    calibration = calibrate_axes(_LINE_REFERENCES)
    with pytest.raises(ValidationError):
        digitize_curve(_line_image(), calibration, threshold=1.5)


# =============================================================================
# 与 Qt 解耦
# =============================================================================

#: 一个屏蔽 PyQt6 的子进程脚本：模块必须在没有 Qt 的环境里也能 import。
_BLOCKED_QT_SCRIPT = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "PyQt6" or name.startswith("PyQt6."):
            raise ImportError("PyQt6 is blocked for this test")
        return None


sys.meta_path.insert(0, _Blocker())
for name in [key for key in sys.modules if key == "PyQt6" or key.startswith("PyQt6.")]:
    del sys.modules[name]

import numpy as np

import stratigraphy.plot_digitizer as digitizer

assert digitizer.AxisCalibration is not None
assert digitizer.digitize_curve is not None
assert "PyQt6" not in sys.modules
try:
    digitizer.build_digitizer_dialog(np.zeros((4, 4, 3), dtype=np.uint8))
except ImportError as exc:
    assert "PyQt6" in str(exc), str(exc)
else:
    raise AssertionError("build_digitizer_dialog must fail without PyQt6")
print("BLOCKED_IMPORT_OK")
"""


def test_module_imports_without_pyqt6():
    """核心模块必须在没有 PyQt6 的环境里也能 import。"""
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCKED_QT_SCRIPT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "BLOCKED_IMPORT_OK" in completed.stdout


def test_module_namespace_has_no_qt_objects():
    """模块顶层不得留下任何 PyQt6 对象——Qt 的导入必须是延迟的。"""
    module = importlib.import_module("stratigraphy.plot_digitizer")
    offenders = [name for name, value in vars(module).items() if type(value).__module__.startswith("PyQt6")]
    assert offenders == []


def test_dialog_factory_is_exported():
    """薄壳工厂函数必须在（延迟导入 Qt 的前提下）对外可见。"""
    module = importlib.import_module("stratigraphy.plot_digitizer")
    assert callable(module.build_digitizer_dialog)
    assert callable(build_digitizer_dialog)
