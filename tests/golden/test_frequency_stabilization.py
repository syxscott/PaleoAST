# =============================================================================
# FILE: tests/golden/test_frequency_stabilization.py
# =============================================================================
"""
``stratigraphy.frequency_stabilization``（频率稳定）的测试。

测试围绕三条主线组织：

* **半功率真值。** 本模块的通带上下频率是**精确的半功率点**：在该频率处
  幅值增益恰为 ``1 / sqrt(2)``，即**功率**增益恰为 ``0.5``（``-3.0103 dB``）；
  通带中心的增益为 ``1``。这些数字直接来自高斯响应的定义
  ``G(f) = exp(-ln(sqrt(2)) * ((f - f_c) / b)**2)``，不是"跑一遍把输出冻结
  下来"——断言直接读公开的响应函数 :func:`gaussian_bandpass_response`。
  另有一条断言检查**滤波实现实际使用的传递函数**与这条曲线一致：把纯正弦
  送进滤波器，用最小二乘投影测出输出幅度，与响应曲线的预测值对照。

* **通带跟着信号走（最有说服力的一条）。** 构造一个"前半段周期 10、后半段
  周期 20"的**时间变信号**并叠加白噪声，再用一条**随时间线性移动**的通带
  去滤。滤波输出的**局部周期**必须**跟得上**控制点给出的周期：前半段测出
  10、后半段测出 20（容差 15%）。周期是从**被测产物本身**测出来的——用零
  穿越间隔与 FFT 主峰两种互相独立的方法测，而不是照抄实现的输出。同时用
  原始（未滤波）记录做对照：它的局部周期**测不出** 20，所以这条断言不是
  平凡成立的。

* **控制点路径与覆盖。** 三点控制点的线性插值必须与手算一致，端点之外
  必须**保持常值**；单行控制点表示**恒定**边界，必须与"频率相同的多点
  路径"给出**完全一致**的结果；输出必须与输入等长、处处被覆盖
  （``coverage_count >= 1``、``coverage_weight > 0``），且重复调用结果完全
  相同。

边界测试断言抛 ``utils.exceptions.ValidationError``：坐标非递增、间距不均匀、
控制点越界或非严格递增、``lower >= upper``、``upper >= Nyquist``、窗口样本
数不足 8、``step_length`` 为 0 或超过窗口长度、坐标与数值长度不一致、点数
不足。

参考来源（断言中引用的数值出处）
--------------------------------
* Thibault, N. & Rizzi, G. (2019). Dynamic filtering of univariate series:
  A scale-space approach. *Geology*, 47(1), 72–76.（动态滤波的原始方法：
  逐窗滤波 + 频域带通）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12–22.（Acycle 的 "Frequency Stabilization" 子功能：用户点出上下频率边界
  曲线，滤波器随坐标移动）
* Oppenheim, A. V. & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.（窗函数法、overlap-add 与零填充 FFT；本模块的分析/
  合成权重是半样本错位 Hann 权重的平方根）
* Harris, F. J. (1978). On the use of windows for harmonic analysis with the
  Fourier transform. *Proceedings of the IEEE*, 66(1), 51–83.（Hann 窗的定义
  ``0.5 - 0.5 cos(2 pi n / N)``）
* Numerical Recipes (3rd ed.). Cambridge University Press, 2007.（零穿越法测
  周期、离散傅里叶变换求主峰）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"frequency_stabilization"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.frequency_stabilization import (
    frequency_stabilization_filter,
    gaussian_bandpass_response,
    interpolate_control_path,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的采样点数与采样间隔（坐标单位）。
N_POINTS = 2000
SAMPLE_INTERVAL = 1.0

#: 合成记录的总跨度（坐标单位）。
SPAN = float(N_POINTS - 1)

#: 物理 Nyquist 频率（每单位坐标）：``1 / (2 * dt)``。
NYQUIST = 0.5 / SAMPLE_INTERVAL

#: 时间变信号的两个时段：前半段周期 10、后半段周期 20。
FIRST_PERIOD = 10.0
SECOND_PERIOD = 20.0

#: 与两个时段对应的通带（每单位坐标的循环频率）。
FIRST_BAND = (0.08, 0.12)
SECOND_BAND = (0.04, 0.06)

#: 白噪声标准差（相对于振幅为 1 的信号）。
NOISE = 0.4

#: 移动带通滤波的窗口长度与步长（坐标单位）。
WINDOW_LENGTH = 100.0
STEP_LENGTH = 20.0

#: 频响断言用的固定带通与采样间隔。
FIXED_BAND = (0.02, 0.06)
FIXED_INTERVAL = 1.0


# =============================================================================
# 辅助函数
# =============================================================================


def _two_period_record(seed: int = 7) -> tuple[npt.NDArray, npt.NDArray]:
    """
    "前半段周期 10、后半段周期 20"的时间变信号 + 白噪声。

    相位用**积分**构造，因此周期切换处相位连续（不会人为造一个跳变）。

    Returns:
        (坐标, 数值)。
    """
    rng = np.random.default_rng(seed)
    coordinate = np.arange(N_POINTS, dtype=float) * SAMPLE_INTERVAL
    period = np.where(coordinate < SPAN / 2.0, FIRST_PERIOD, SECOND_PERIOD)
    phase = 2.0 * np.pi * np.cumsum(1.0 / (period * SAMPLE_INTERVAL))
    values = np.sin(phase) + NOISE * rng.standard_normal(N_POINTS)
    return coordinate, values


def _moving_band_control_points() -> tuple[list[list[float]], list[list[float]]]:
    """
    随时间移动的通带控制点。

    前 40% 的时间用 ``FIRST_BAND``，后 40% 的时间用 ``SECOND_BAND``，中间
    20% 在两者之间**线性过渡**。控制点坐标严格递增，且落在数据域内。

    Returns:
        (下边界控制点, 上边界控制点)。
    """
    lower = [
        [0.0, FIRST_BAND[0]],
        [0.4 * SPAN, FIRST_BAND[0]],
        [0.6 * SPAN, SECOND_BAND[0]],
        [SPAN, SECOND_BAND[0]],
    ]
    upper = [
        [0.0, FIRST_BAND[1]],
        [0.4 * SPAN, FIRST_BAND[1]],
        [0.6 * SPAN, SECOND_BAND[1]],
        [SPAN, SECOND_BAND[1]],
    ]
    return lower, upper


def _zero_crossing_period(segment: npt.NDArray) -> float:
    """
    用零穿越间隔测得的周期。

    对去均值后的片段求符号变化点，用线性插值把穿越时刻细化到样本之间，
    相邻两次穿越的间隔是**半周期**，因此周期是平均间隔的两倍。

    Returns:
        测得的周期；穿越点不足两个时返回 ``nan``。
    """
    centred = np.asarray(segment, dtype=float) - float(np.mean(segment))
    signs = np.sign(centred)
    changes = np.nonzero(np.diff(signs) != 0.0)[0]
    if changes.size < 2:
        return float("nan")
    times = []
    for index in changes:
        left = centred[index]
        right = centred[index + 1]
        if right == left:
            times.append(float(index))
            continue
        times.append(float(index) + float(-left / (right - left)))
    return float(2.0 * np.mean(np.diff(np.asarray(times, dtype=float))))


def _dominant_period(segment: npt.NDArray) -> float:
    """
    用离散傅里叶变换主峰测得的周期（与零穿越法互相独立）。

    Returns:
        主峰频率的倒数；找不到主峰时返回 ``nan``。
    """
    centred = np.asarray(segment, dtype=float) - float(np.mean(segment))
    spectrum = np.abs(np.fft.rfft(centred))
    frequencies = np.fft.rfftfreq(centred.size, d=SAMPLE_INTERVAL)
    peak = int(np.argmax(spectrum[1:])) + 1
    if frequencies[peak] <= 0.0:
        return float("nan")
    return float(1.0 / frequencies[peak])


def _projection_amplitude(segment: npt.NDArray, frequency: float) -> float:
    """把片段投影到已知频率的 ``cos``/``sin`` 基上，测得正弦分量的幅度。"""
    index = np.arange(segment.size, dtype=float)
    design = np.column_stack(
        [
            np.cos(2.0 * np.pi * frequency * index),
            np.sin(2.0 * np.pi * frequency * index),
        ]
    )
    coefficients = np.linalg.lstsq(design, segment, rcond=None)[0]
    return float(np.hypot(coefficients[0], coefficients[1]))


def _relative_error(measured: float, expected: float) -> float:
    """相对误差（期望值为 0 时返回绝对误差）。"""
    if expected == 0.0:
        return abs(measured)
    return abs(measured - expected) / abs(expected)


# =============================================================================
# 半功率真值
# =============================================================================


class TestHalfPowerResponse:
    """通带边界必须是精确的半功率点。"""

    def test_the_band_edges_are_exactly_half_power(self):
        """上下边界处的**功率**增益恰为 0.5（幅值 ``1 / sqrt(2)``）。"""
        lower, upper = FIXED_BAND
        edges = gaussian_bandpass_response(
            np.array([lower, upper]),
            lower,
            upper,
            FIXED_INTERVAL,
        )

        assert edges[0] == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6), (
            f"the gain at the lower edge is {edges[0]:.12g}, not 1/sqrt(2)"
        )
        assert edges[1] == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6), (
            f"the gain at the upper edge is {edges[1]:.12g}, not 1/sqrt(2)"
        )
        # 功率增益才是"半功率"的严格含义：-3.0103 dB。
        assert float(edges[0] ** 2) == pytest.approx(0.5, rel=1e-6)
        assert float(edges[1] ** 2) == pytest.approx(0.5, rel=1e-6)

    def test_the_passband_centre_has_unit_gain(self):
        """通带中心的增益必须是 1。"""
        lower, upper = FIXED_BAND
        centre = 0.5 * (lower + upper)
        gain = gaussian_bandpass_response(np.array([centre]), lower, upper, FIXED_INTERVAL)

        assert float(gain[0]) == pytest.approx(1.0, rel=1e-12)

    def test_the_response_decreases_monotonically_outside_the_band(self):
        """离开通带后增益必须单调下降（向下与向上两个方向）。"""
        lower, upper = FIXED_BAND
        axis = np.linspace(0.0, NYQUIST, 2001)
        gain = gaussian_bandpass_response(axis, lower, upper, FIXED_INTERVAL)

        above = gain[axis > upper]
        below = gain[axis < lower]

        assert np.all(np.diff(above) < 0.0), "the gain must decrease monotonically above the upper edge"
        assert np.all(np.diff(below) > 0.0), "the gain must increase monotonically towards the lower edge"
        assert float(np.max(gain)) == pytest.approx(1.0, rel=1e-12)

    def test_the_filter_uses_the_published_response(self):
        """滤波器**实际**的传递函数必须与公开的响应曲线一致。

        把纯正弦送进滤波器，用最小二乘投影测出输出幅度，与
        :func:`gaussian_bandpass_response` 的预测值对照：通带中心处应当
        原样通过，上下边界处应当只剩 ``1 / sqrt(2)``。
        """
        coordinate = np.arange(N_POINTS, dtype=float) * FIXED_INTERVAL
        lower, upper = FIXED_BAND
        interior = slice(400, N_POINTS - 400)

        for frequency in (lower, 0.5 * (lower + upper), upper):
            pure = np.sin(2.0 * np.pi * frequency * coordinate)
            result = frequency_stabilization_filter(
                coordinate,
                pure,
                [[0.0, lower]],
                [[0.0, upper]],
                window_length=400.0,
                step_length=200.0,
            )
            measured = _projection_amplitude(result.filtered[interior], frequency)
            predicted = float(gaussian_bandpass_response(np.array([frequency]), lower, upper, FIXED_INTERVAL)[0])

            assert _relative_error(measured, predicted) < 0.05, (
                f"the filter passes {frequency} with gain {measured:.6g}, "
                f"but the published response predicts {predicted:.6g}"
            )

    def test_the_filter_suppresses_frequencies_outside_the_band(self):
        """带外的成分必须被压掉。"""
        coordinate = np.arange(N_POINTS, dtype=float) * FIXED_INTERVAL
        lower, upper = FIXED_BAND
        out_of_band = 0.20
        pure = np.sin(2.0 * np.pi * out_of_band * coordinate)
        result = frequency_stabilization_filter(
            coordinate,
            pure,
            [[0.0, lower]],
            [[0.0, upper]],
            window_length=400.0,
            step_length=200.0,
        )
        measured = _projection_amplitude(result.filtered[400 : N_POINTS - 400], out_of_band)

        assert measured < 0.05, f"the out-of-band component survives with amplitude {measured:.6g}"


# =============================================================================
# 控制点路径
# =============================================================================


class TestControlPathInterpolation:
    """控制点路径必须是分段线性、端点外保持常值。"""

    def test_three_control_points_interpolate_linearly(self):
        """三点控制点的插值值必须与手算一致。"""
        control = [[0.0, 0.05], [50.0, 0.10], [100.0, 0.15]]
        query = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        values = interpolate_control_path(control, query)

        assert values[0] == pytest.approx(0.05)
        assert values[1] == pytest.approx(0.075)
        assert values[2] == pytest.approx(0.10)
        assert values[3] == pytest.approx(0.125)
        assert values[4] == pytest.approx(0.15)

    def test_the_path_is_held_constant_outside_the_control_points(self):
        """第一个控制点之前、最后一个控制点之后必须保持常值。"""
        control = [[20.0, 0.08], [60.0, 0.12]]
        query = np.array([0.0, 10.0, 19.999, 40.0, 60.0, 80.0, 1000.0])
        values = interpolate_control_path(control, query)

        assert values[0] == pytest.approx(0.08)
        assert values[1] == pytest.approx(0.08)
        assert values[2] == pytest.approx(0.08)
        assert values[3] == pytest.approx(0.10)
        assert values[4] == pytest.approx(0.12)
        assert values[5] == pytest.approx(0.12)
        assert values[6] == pytest.approx(0.12)

    def test_a_single_control_point_is_a_constant_path(self):
        """单行控制点表示一条恒定路径。"""
        query = np.array([0.0, 13.5, 100.0, 999.0])
        values = interpolate_control_path([[7.0, 0.037]], query)

        assert np.all(values == pytest.approx(0.037))

    def test_the_path_passes_through_every_control_point(self):
        """插值路径必须精确穿过每一个控制点。"""
        control = [[0.0, 0.05], [50.0, 0.10], [100.0, 0.15]]
        values = interpolate_control_path(control, np.array([0.0, 50.0, 100.0]))

        assert np.allclose(values, np.array([0.05, 0.10, 0.15]))


# =============================================================================
# 恒定带通
# =============================================================================


class TestConstantBand:
    """单行控制点 = 固定带通。"""

    def test_a_single_control_point_gives_a_constant_band(self):
        """单行控制点给出的每个窗口的上下边界都是同一个常数。"""
        coordinate, values = _two_period_record()
        lower, upper = FIXED_BAND
        result = frequency_stabilization_filter(
            coordinate,
            values,
            [[0.0, lower]],
            [[0.0, upper]],
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert np.all(result.lower_frequency == pytest.approx(lower))
        assert np.all(result.upper_frequency == pytest.approx(upper))
        assert np.all(result.center_frequency == pytest.approx(0.5 * (lower + upper)))

    def test_a_constant_multi_point_path_matches_a_single_point_path(self):
        """频率相同的多点路径必须与单行控制点给出**完全一致**的结果。"""
        coordinate, values = _two_period_record()
        lower, upper = FIXED_BAND
        single = frequency_stabilization_filter(
            coordinate,
            values,
            [[0.0, lower]],
            [[0.0, upper]],
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )
        multi = frequency_stabilization_filter(
            coordinate,
            values,
            [[0.0, lower], [SPAN / 2.0, lower], [SPAN, lower]],
            [[0.0, upper], [SPAN / 2.0, upper], [SPAN, upper]],
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert np.allclose(single.filtered, multi.filtered)
        assert np.allclose(single.coverage_weight, multi.coverage_weight)

    def test_the_control_coordinate_does_not_change_a_constant_band(self):
        """恒定边界下，控制点放在坐标域的哪一端都不影响结果。"""
        coordinate, values = _two_period_record()
        lower, upper = FIXED_BAND
        at_start = frequency_stabilization_filter(
            coordinate,
            values,
            [[0.0, lower]],
            [[0.0, upper]],
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )
        at_end = frequency_stabilization_filter(
            coordinate,
            values,
            [[SPAN, lower]],
            [[SPAN, upper]],
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert np.allclose(at_start.filtered, at_end.filtered)


# =============================================================================
# 频率跟踪（本模块最核心的能力）
# =============================================================================


class TestFrequencyTracking:
    """输出的局部周期必须跟上控制点给出的周期。"""

    def test_the_filtered_period_follows_the_moving_passband(self):
        """前半段测出周期 10、后半段测出周期 20（容差 15%）。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        first = result.filtered[int(0.05 * N_POINTS) : int(0.35 * N_POINTS)]
        last = result.filtered[int(0.65 * N_POINTS) : int(0.95 * N_POINTS)]

        first_period = _zero_crossing_period(first)
        last_period = _zero_crossing_period(last)

        assert _relative_error(first_period, FIRST_PERIOD) < 0.15, (
            f"the filtered first half has a period of {first_period:.4f}, not {FIRST_PERIOD}"
        )
        assert _relative_error(last_period, SECOND_PERIOD) < 0.15, (
            f"the filtered second half has a period of {last_period:.4f}, not {SECOND_PERIOD}"
        )

    def test_the_dominant_frequency_follows_the_moving_passband(self):
        """用 FFT 主峰（与零穿越法独立）复测同一个契约。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        first = result.filtered[int(0.05 * N_POINTS) : int(0.35 * N_POINTS)]
        last = result.filtered[int(0.65 * N_POINTS) : int(0.95 * N_POINTS)]

        assert _relative_error(_dominant_period(first), FIRST_PERIOD) < 0.15
        assert _relative_error(_dominant_period(last), SECOND_PERIOD) < 0.15

    def test_the_unfiltered_record_does_not_show_the_second_period(self):
        """对照：原始记录**测不出**后半段的周期，所以上面的断言不是平凡的。"""
        _coordinate, values = _two_period_record()
        last = values[int(0.65 * N_POINTS) : int(0.95 * N_POINTS)]

        assert _relative_error(_zero_crossing_period(last), SECOND_PERIOD) > 0.15

    def test_the_out_of_band_noise_is_suppressed(self):
        """带外的噪声必须被压下去。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        first = result.filtered[int(0.05 * N_POINTS) : int(0.35 * N_POINTS)]
        last = result.filtered[int(0.65 * N_POINTS) : int(0.95 * N_POINTS)]
        index = np.arange(first.size, dtype=float)

        for segment, frequency in ((first, 1.0 / FIRST_PERIOD), (last, 1.0 / SECOND_PERIOD)):
            fitted = np.column_stack(
                [
                    np.cos(2.0 * np.pi * frequency * index),
                    np.sin(2.0 * np.pi * frequency * index),
                ]
            )
            coefficients = np.linalg.lstsq(fitted, segment, rcond=None)[0]
            residual = segment - fitted @ coefficients

            assert float(np.std(residual)) < 0.5 * NOISE, (
                f"the residual noise is {np.std(residual):.4g}, not much below the input {NOISE}"
            )


# =============================================================================
# 覆盖与形状
# =============================================================================


class TestCoverage:
    """输出必须与输入等长，且处处被覆盖。"""

    def test_every_sample_is_covered(self):
        """``coverage_count >= 1`` 且 ``coverage_weight > 0``（不留空隙）。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert np.all(result.coverage_count >= 1)
        assert np.all(result.coverage_weight > 0.0)
        assert result.coverage_count.size == values.size

    def test_the_output_has_the_same_length_as_the_input(self):
        """输出与输入等长。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert result.filtered.size == values.size
        assert result.coordinates.size == values.size
        assert result.coverage_count.size == values.size
        assert result.coverage_weight.size == values.size

    def test_every_output_value_is_finite(self):
        """输出与诊断量全部有限。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert np.all(np.isfinite(result.filtered))
        assert np.all(np.isfinite(result.coverage_weight))
        assert np.all(np.isfinite(result.windows))

    def test_the_window_table_is_consistent(self):
        """窗口信息矩阵的每一行必须自洽。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        result = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        windows = result.windows
        assert windows.shape[1] == 8
        assert windows.shape[0] >= 2
        # 中心频率是上下边界的平均。
        assert np.allclose(windows[:, 3], windows[:, 1] + (windows[:, 2] - windows[:, 1]) / 2.0)
        # 每个窗口的下标必须落在输入范围内，且窗口确实覆盖了它。
        for row in windows:
            start = int(row[6])
            end = int(row[7])
            assert 0 <= start < end < values.size
            assert coordinate[start] == pytest.approx(row[4])
            assert coordinate[end] == pytest.approx(row[5])


# =============================================================================
# 确定性
# =============================================================================


class TestDeterminism:
    """同样的输入必须给出同样的输出。"""

    def test_repeated_calls_agree(self):
        """两次调用必须完全一致。"""
        coordinate, values = _two_period_record()
        lower_points, upper_points = _moving_band_control_points()
        one = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )
        two = frequency_stabilization_filter(
            coordinate,
            values,
            lower_points,
            upper_points,
            window_length=WINDOW_LENGTH,
            step_length=STEP_LENGTH,
        )

        assert np.allclose(one.filtered, two.filtered)
        assert np.allclose(one.windows, two.windows)
        assert np.array_equal(one.coverage_count, two.coverage_count)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def _record(self) -> tuple[npt.NDArray, npt.NDArray]:
        return _two_period_record()

    def _points(self) -> tuple[list[list[float]], list[list[float]]]:
        return _moving_band_control_points()

    def test_non_increasing_coordinates_are_rejected(self):
        coordinate, values = self._record()
        broken = coordinate.copy()
        broken[10] = broken[9]
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(broken, values, lower_points, upper_points, WINDOW_LENGTH, STEP_LENGTH)

    def test_unevenly_spaced_coordinates_are_rejected(self):
        coordinate, values = self._record()
        broken = coordinate.copy()
        broken[100] += 0.5
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(broken, values, lower_points, upper_points, WINDOW_LENGTH, STEP_LENGTH)

    def test_mismatched_lengths_are_rejected(self):
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate, values[:-1], lower_points, upper_points, WINDOW_LENGTH, STEP_LENGTH
            )

    def test_too_few_points_are_rejected(self):
        coordinate = np.arange(4, dtype=float)
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                np.sin(coordinate),
                [[0.0, 0.02]],
                [[0.0, 0.06]],
                WINDOW_LENGTH,
                STEP_LENGTH,
            )

    def test_non_finite_values_are_rejected(self):
        coordinate, values = self._record()
        broken = values.copy()
        broken[7] = np.nan
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(coordinate, broken, lower_points, upper_points, WINDOW_LENGTH, STEP_LENGTH)

    def test_control_points_outside_the_data_domain_are_rejected(self):
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                [[-1.0, 0.08]],
                upper_points,
                WINDOW_LENGTH,
                STEP_LENGTH,
            )
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                lower_points,
                [[SPAN + 1.0, 0.12]],
                WINDOW_LENGTH,
                STEP_LENGTH,
            )

    def test_control_points_that_are_not_strictly_increasing_are_rejected(self):
        coordinate, values = self._record()
        _lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                [[100.0, 0.08], [100.0, 0.09]],
                upper_points,
                WINDOW_LENGTH,
                STEP_LENGTH,
            )

    def test_non_positive_control_frequencies_are_rejected(self):
        coordinate, values = self._record()
        _lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                [[0.0, 0.0]],
                upper_points,
                WINDOW_LENGTH,
                STEP_LENGTH,
            )

    def test_a_lower_boundary_above_the_upper_boundary_is_rejected(self):
        coordinate, values = self._record()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                [[0.0, 0.20]],
                [[0.0, 0.10]],
                WINDOW_LENGTH,
                STEP_LENGTH,
            )

    def test_an_upper_boundary_at_or_above_nyquist_is_rejected(self):
        coordinate, values = self._record()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                [[0.0, 0.10]],
                [[0.0, NYQUIST]],
                WINDOW_LENGTH,
                STEP_LENGTH,
            )

    def test_a_window_with_too_few_samples_is_rejected(self):
        """窗口样本数必须至少为 8。"""
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                lower_points,
                upper_points,
                window_length=2.0,
                step_length=1.0,
            )

    def test_a_zero_step_length_is_rejected(self):
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                lower_points,
                upper_points,
                WINDOW_LENGTH,
                0.0,
            )

    def test_a_step_longer_than_the_window_is_rejected(self):
        """步长超过窗口样本数会留下覆盖空隙，必须报错。"""
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                lower_points,
                upper_points,
                WINDOW_LENGTH,
                2.0 * WINDOW_LENGTH,
            )

    def test_a_window_longer_than_the_record_is_rejected(self):
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                lower_points,
                upper_points,
                window_length=10.0 * SPAN,
                step_length=STEP_LENGTH,
            )

    def test_a_non_positive_window_length_is_rejected(self):
        coordinate, values = self._record()
        lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                lower_points,
                upper_points,
                0.0,
                STEP_LENGTH,
            )

    def test_an_invalid_control_path_is_rejected(self):
        """控制点形状非法必须报错（而不是抛裸 ``ValueError``）。"""
        coordinate, values = self._record()
        _lower_points, upper_points = self._points()
        with pytest.raises(ValidationError):
            frequency_stabilization_filter(
                coordinate,
                values,
                [[0.0, 0.08, 0.09]],
                upper_points,
                WINDOW_LENGTH,
                STEP_LENGTH,
            )
