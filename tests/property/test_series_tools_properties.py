# =============================================================================
# FILE: tests/property/test_series_tools_properties.py
# =============================================================================
"""
``stratigraphy.series_tools`` 的测试：插值、选段、合并、补缺口、去峰、
截尾、排序去重与数值求导。

断言里的期望值都来自**输入自身的性质**或**独立推导**，而不是"跑一遍把
输出冻结下来"（那只能证明实现是确定性的，证明不了它算得对）：

* 线性函数在任意时间轴（包括不等间距）上的线性插值必须回到该函数本身；
* 样条插值在节点处必须取回原样本，这是插值的定义；
* 分段线性插值在某个区间内值，可以用手算的线性权重独立算出；
* 合并后的时间轴必须严格递增，长度等于各序列长度之和；
* 线性函数的数值导数必须处处等于其斜率；
* 缺口、截尾、去重都是**逐点可查**的操作，期望值可以直接写出。

边界行为同样被断言：空序列、非单调时间轴、非法区间都必须抛
``ValidationError``，而不是悄悄返回一个空数组或乱序结果。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {
    "add_gaps",
    "clip_values",
    "derivative",
    "interpolate_series",
    "remove_peaks",
    "select_intervals",
    "sort_unique",
}


from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.series_tools import (
    add_gaps,
    clip_values,
    derivative,
    interpolate_series,
    merge_series,
    remove_peaks,
    select_intervals,
    sort_unique,
)
from utils.exceptions import ValidationError

# ---------------------------------------------------------------------------
# 固定夹具
# ---------------------------------------------------------------------------

#: 等间距时间轴上的线性函数 y = 3t + 1。
LINEAR_TIME = np.linspace(0.0, 10.0, 11)
LINEAR_VALUES = 3.0 * LINEAR_TIME + 1.0

#: 不等间距时间轴（故意不是等差数列），用于验证插值尊重真实坐标。
UNEVEN_TIME = np.array([0.0, 0.5, 1.0, 2.5, 4.0])
UNEVEN_VALUES = 2.0 * UNEVEN_TIME - 3.0


class TestInterpolateSeries:
    """插值必须回到原函数，且尊重不等间距的时间轴。"""

    def test_linear_interpolation_reproduces_a_linear_function(self):
        """线性函数在等间距时间轴上插值后仍然等于该函数。"""
        new_time = np.linspace(0.0, 10.0, 21)
        result = interpolate_series(LINEAR_TIME, LINEAR_VALUES, new_time)
        assert np.allclose(result, 3.0 * new_time + 1.0, rtol=0.0, atol=1e-8)

    def test_linear_interpolation_reproduces_a_linear_function_on_an_uneven_axis(self):
        """不等间距的时间轴上，线性插值同样回到该函数。

        如果实现误以为时间轴等距（例如按样本序号而不是按坐标插值），
        这里的误差会远大于 1e-8。
        """
        new_time = np.array([0.25, 0.75, 1.75, 3.25, 3.9])
        result = interpolate_series(UNEVEN_TIME, UNEVEN_VALUES, new_time)
        assert np.allclose(result, 2.0 * new_time - 3.0, rtol=0.0, atol=1e-8)

    def test_linear_interpolation_matches_the_hand_computed_segment_value(self):
        """一段区间内的插值值可以用线性权重独立算出。"""
        # 节点：t=1.0 -> v=-1，t=2.5 -> v=2；在 t=1.75 处
        # 权重 = (1.75 - 1.0) / (2.5 - 1.0) = 0.5，期望 -1 + 0.5 * 3 = 0.5
        result = interpolate_series(UNEVEN_TIME, UNEVEN_VALUES, np.array([1.75]))
        assert result[0] == pytest.approx(0.5, abs=1e-12)

    def test_cubic_interpolation_returns_the_original_samples_at_the_nodes(self):
        """插值的定义：节点处必须取回原样本。"""
        result = interpolate_series(UNEVEN_TIME, UNEVEN_VALUES, UNEVEN_TIME, method="cubic")
        assert np.allclose(result, UNEVEN_VALUES, rtol=0.0, atol=1e-10)

    def test_cubic_interpolation_reproduces_a_quadratic(self):
        """二次函数的三次样条插值必须回到该二次函数。"""
        time = np.array([0.0, 0.7, 1.5, 3.0, 4.4, 6.0])
        values = 2.0 * time**2 - 3.0 * time + 1.0
        new_time = np.array([0.35, 1.1, 2.25, 5.2])
        result = interpolate_series(time, values, new_time, method="cubic")
        assert np.allclose(result, 2.0 * new_time**2 - 3.0 * new_time + 1.0, rtol=0.0, atol=1e-8)

    def test_nearest_interpolation_picks_the_closest_sample(self):
        """最近邻插值取距离最近的样本。

        节点为 0、0.5、1.0、2.5、4.0，因此 0.4 -> 0.5、1.2 -> 1.0、
        2.4 -> 2.5，对应的样本下标是 1、2、3。
        """
        result = interpolate_series(UNEVEN_TIME, UNEVEN_VALUES, np.array([0.4, 1.2, 2.4]), method="nearest")
        expected = UNEVEN_VALUES[[1, 2, 3]]
        assert np.allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_values_outside_the_time_range_are_not_extrapolated(self):
        """默认不外推：时间范围之外返回 NaN。"""
        result = interpolate_series(LINEAR_TIME, LINEAR_VALUES, np.array([-1.0, 0.0, 10.0, 11.0]))
        assert np.isnan(result[0])
        assert np.isnan(result[3])
        assert np.all(np.isfinite(result[1:3]))

    def test_extrapolation_can_be_requested_explicitly(self):
        """显式要求外推时，范围之外得到有限值。"""
        result = interpolate_series(LINEAR_TIME, LINEAR_VALUES, np.array([-1.0]), extrapolate=True)
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(3.0 * -1.0 + 1.0, abs=1e-8)

    def test_unknown_method_raises(self):
        """未知插值方法必须报错，而不是悄悄退回线性。"""
        with pytest.raises(ValidationError):
            interpolate_series(LINEAR_TIME, LINEAR_VALUES, LINEAR_TIME, method="pchip")

    def test_non_monotonic_time_axis_raises(self):
        """时间轴非严格递增时必须报错。"""
        time = np.array([0.0, 2.0, 1.0, 3.0])
        with pytest.raises(ValidationError):
            interpolate_series(time, np.arange(4.0), np.array([0.5]))

    def test_empty_input_raises(self):
        """空序列必须报错。"""
        with pytest.raises(ValidationError):
            interpolate_series(np.array([]), np.array([]), np.array([0.0]))

    def test_mismatched_lengths_raise(self):
        """时间轴与数值长度不一致时必须报错。"""
        with pytest.raises(ValidationError):
            interpolate_series(LINEAR_TIME, LINEAR_VALUES[:-1], LINEAR_TIME)


class TestSelectIntervals:
    """选段必须取到预期数量的样本。"""

    def test_select_by_time_values_returns_the_expected_number_of_points(self):
        """按时间值取闭区间 [2, 5]。"""
        time = np.arange(10.0)
        values = 2.0 * time
        selected_time, selected_values = select_intervals(time, values, [(2.0, 5.0)])
        assert selected_time.size == 4
        assert selected_values.size == 4
        assert np.allclose(selected_time, [2.0, 3.0, 4.0, 5.0])
        assert np.allclose(selected_values, [4.0, 6.0, 8.0, 10.0])

    def test_select_by_index_returns_the_expected_number_of_points(self):
        """按下标取闭区间 [2, 5]。"""
        time = np.arange(10.0)
        values = 2.0 * time
        selected_time, selected_values = select_intervals(time, values, [(2, 5)], mode="index")
        assert selected_time.size == 4
        assert np.allclose(selected_time, [2.0, 3.0, 4.0, 5.0])
        assert np.allclose(selected_values, [4.0, 6.0, 8.0, 10.0])

    def test_auto_mode_falls_back_to_indices_when_no_time_matches(self):
        """auto 模式：时间值取不中、而边界是合法下标时，按下标解释。"""
        time = np.array([0.0, 10.0, 20.0, 30.0])
        values = time * 2.0
        selected_time, _ = select_intervals(time, values, [(1, 2)])
        assert selected_time.size == 2
        assert np.allclose(selected_time, [10.0, 20.0])

    def test_multiple_intervals_are_concatenated_in_time_order(self):
        """多个区间的结果按时间排序拼接。"""
        time = np.arange(10.0)
        values = time
        selected_time, selected_values = select_intervals(time, values, [(5.0, 6.0), (1.0, 2.0)])
        assert np.allclose(selected_time, [1.0, 2.0, 5.0, 6.0])
        assert np.allclose(selected_values, [1.0, 2.0, 5.0, 6.0])

    def test_reversed_interval_raises(self):
        """起点晚于终点的区间是非法区间。"""
        with pytest.raises(ValidationError):
            select_intervals(LINEAR_TIME, LINEAR_VALUES, [(5.0, 1.0)])

    def test_malformed_interval_raises(self):
        """不是两个数的区间必须报错。"""
        with pytest.raises(ValidationError):
            select_intervals(LINEAR_TIME, LINEAR_VALUES, [(1.0, 2.0, 3.0)])

    def test_unknown_mode_raises(self):
        """未知解释方式必须报错。"""
        with pytest.raises(ValidationError):
            select_intervals(LINEAR_TIME, LINEAR_VALUES, [(1.0, 2.0)], mode="guess")


class TestMergeSeries:
    """合并必须按时间排序，并且不静默处理重复时间点。"""

    def test_merged_series_is_sorted_and_has_the_combined_length(self):
        """合并后时间严格递增，长度等于两条序列长度之和。"""
        first_time = np.array([3.0, 5.0, 7.0])
        second_time = np.array([1.0, 4.0, 8.0])
        merged_time, merged_values = merge_series([(first_time, first_time * 10.0), (second_time, second_time * 100.0)])
        assert merged_time.size == first_time.size + second_time.size
        assert np.all(np.diff(merged_time) > 0)
        assert np.allclose(merged_time, [1.0, 3.0, 4.0, 5.0, 7.0, 8.0])
        assert np.allclose(merged_values, [100.0, 30.0, 400.0, 50.0, 70.0, 800.0])

    def test_duplicate_times_keep_the_first_occurrence_by_default(self):
        """默认策略保留最先出现的那一条。"""
        merged_time, merged_values = merge_series(
            [
                (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
                (np.array([2.0, 3.0]), np.array([200.0, 30.0])),
            ]
        )
        assert np.allclose(merged_time, [1.0, 2.0, 3.0])
        assert np.allclose(merged_values, [10.0, 20.0, 30.0])

    def test_duplicate_times_can_keep_the_last_occurrence(self):
        """keep="last" 保留最后出现的那一条。"""
        _, merged_values = merge_series(
            [
                (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
                (np.array([2.0, 3.0]), np.array([200.0, 30.0])),
            ],
            keep="last",
        )
        assert np.allclose(merged_values, [10.0, 200.0, 30.0])

    def test_duplicate_times_can_be_averaged(self):
        """keep="mean" 把重复时间点的数值取均值。"""
        _, merged_values = merge_series(
            [
                (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
                (np.array([2.0, 3.0]), np.array([200.0, 30.0])),
            ],
            keep="mean",
        )
        assert np.allclose(merged_values, [10.0, 110.0, 30.0])

    def test_duplicate_times_can_raise(self):
        """keep="error" 时重复时间点必须报错，而不是静默取舍。"""
        with pytest.raises(ValidationError):
            merge_series(
                [
                    (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
                    (np.array([2.0, 3.0]), np.array([200.0, 30.0])),
                ],
                keep="error",
            )

    def test_unknown_keep_policy_raises(self):
        """未知策略必须报错。"""
        with pytest.raises(ValidationError):
            merge_series([(np.array([1.0]), np.array([1.0]))], keep="whatever")

    def test_varargs_call_style_is_accepted(self):
        """merge_series(s1, s2) 这种写法也要能用。"""
        merged_time, merged_values = merge_series(
            (np.array([2.0]), np.array([20.0])),
            (np.array([1.0]), np.array([10.0])),
        )
        assert np.allclose(merged_time, [1.0, 2.0])
        assert np.allclose(merged_values, [10.0, 20.0])


class TestAddGaps:
    """缺口必须变成 NaN，其余样本保持不变。"""

    def test_samples_inside_the_gap_become_nan(self):
        """缺口区间内的样本变成 NaN，区间外不受影响。"""
        result = add_gaps(LINEAR_TIME, LINEAR_VALUES, [(2.0, 4.0)])
        assert result.size == LINEAR_VALUES.size
        assert np.all(np.isnan(result[(LINEAR_TIME >= 2.0) & (LINEAR_TIME <= 4.0)]))
        outside = (LINEAR_TIME < 2.0) | (LINEAR_TIME > 4.0)
        assert np.allclose(result[outside], LINEAR_VALUES[outside])

    def test_gap_outside_the_record_marks_nothing(self):
        """缺口落在记录之外时什么也不改。"""
        result = add_gaps(LINEAR_TIME, LINEAR_VALUES, [(100.0, 200.0)])
        assert np.allclose(result, LINEAR_VALUES)

    def test_reversed_gap_raises(self):
        """起点晚于终点的缺口是非法的。"""
        with pytest.raises(ValidationError):
            add_gaps(LINEAR_TIME, LINEAR_VALUES, [(4.0, 2.0)])


class TestRemovePeaks:
    """超过阈值的峰必须被去掉。"""

    def test_peak_above_the_threshold_is_removed(self):
        """高于阈值的尖峰被标记为缺失，其余样本保持不变。"""
        time = np.arange(5.0)
        values = np.array([0.0, 1.0, 9.0, 1.0, 0.0])
        result = remove_peaks(time, values, threshold=5.0)
        assert np.isnan(result[2])
        assert np.allclose(result[[0, 1, 3, 4]], values[[0, 1, 3, 4]])

    def test_peaks_below_the_threshold_are_kept(self):
        """未超过阈值的峰不动。"""
        time = np.arange(5.0)
        values = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
        result = remove_peaks(time, values, threshold=5.0)
        assert np.allclose(result, values)

    def test_interpolated_fill_replaces_the_peak_with_its_neighbours(self):
        """fill="interpolate" 用邻居的线性插值替代尖峰。"""
        time = np.arange(5.0)
        values = np.array([0.0, 1.0, 9.0, 1.0, 0.0])
        result = remove_peaks(time, values, threshold=5.0, fill="interpolate")
        assert np.all(np.isfinite(result))
        assert result[2] == pytest.approx(1.0, abs=1e-12)

    def test_invalid_min_distance_raises(self):
        """min_distance 小于 1 必须报错。"""
        with pytest.raises(ValidationError):
            remove_peaks(np.arange(5.0), np.arange(5.0), threshold=1.0, min_distance=0)

    def test_unknown_fill_raises(self):
        """未知填补方式必须报错。"""
        with pytest.raises(ValidationError):
            remove_peaks(np.arange(5.0), np.arange(5.0), threshold=1.0, fill="zero")


class TestClipValues:
    """截尾必须把超界值压到界上。"""

    def test_values_are_clipped_to_the_bounds(self):
        """超界值被截断到界上，界内值不变。"""
        values = np.array([-2.0, 0.5, 3.0])
        result = clip_values(values, 0.0, 1.0)
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_only_one_bound_is_needed(self):
        """只给一个界时，另一侧不受限制。"""
        values = np.array([-2.0, 0.5, 3.0])
        assert np.allclose(clip_values(values, lower=0.0), [0.0, 0.5, 3.0])
        assert np.allclose(clip_values(values, upper=1.0), [-2.0, 0.5, 1.0])

    def test_inverted_bounds_raise(self):
        """下界大于上界必须报错。"""
        with pytest.raises(ValidationError):
            clip_values(np.array([0.0, 1.0]), lower=1.0, upper=0.0)

    def test_input_is_not_modified(self):
        """截尾不得修改调用方的数组。"""
        values = np.array([-2.0, 0.5, 3.0])
        clip_values(values, 0.0, 1.0)
        assert np.allclose(values, [-2.0, 0.5, 3.0])


class TestSortUnique:
    """排序去重必须返回升序结果并报告去掉的重复数。"""

    def test_sorted_unique_values_and_removed_count(self):
        """结果升序去重，重复数正确。"""
        values = np.array([3.0, 1.0, 3.0, 2.0, 1.0, 1.0])
        unique, removed = sort_unique(values)
        assert np.allclose(unique, [1.0, 2.0, 3.0])
        assert removed == 3

    def test_no_duplicates_reports_zero(self):
        """没有重复时报告 0。"""
        unique, removed = sort_unique(np.array([2.0, 1.0]))
        assert np.allclose(unique, [1.0, 2.0])
        assert removed == 0

    def test_empty_input_raises(self):
        """空序列必须报错。"""
        with pytest.raises(ValidationError):
            sort_unique(np.array([]))


class TestDerivative:
    """数值求导必须尊重不等间距的时间轴。"""

    def test_derivative_of_a_linear_function_is_the_slope(self):
        """等间距时间轴上，线性函数的导数恒为斜率。"""
        result = derivative(LINEAR_TIME, LINEAR_VALUES)
        assert np.allclose(result, 3.0, rtol=0.0, atol=1e-10)

    def test_derivative_on_an_uneven_axis_is_still_the_slope(self):
        """不等间距时间轴上，导数同样处处等于斜率。"""
        result = derivative(UNEVEN_TIME, UNEVEN_VALUES)
        assert np.allclose(result, 2.0, rtol=0.0, atol=1e-10)

    def test_derivative_of_a_quadratic_is_linear(self):
        """二次函数的导数是线性函数。

        中心差分对二次函数是精确的，但只在等间距且位于内部时成立（端点
        用的是单边差分，节点左右不对称时也不精确），因此这里用等间距
        时间轴并只断言内部节点。
        """
        time = np.linspace(0.0, 4.0, 5)
        values = 3.0 * time**2 + 1.0
        result = derivative(time, values)
        assert np.allclose(result[1:-1], 6.0 * time[1:-1], rtol=0.0, atol=1e-10)

    def test_non_monotonic_time_axis_raises(self):
        """时间轴非严格递增时必须报错。"""
        with pytest.raises(ValidationError):
            derivative(np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0]))

    def test_single_point_raises(self):
        """只有一个点无法求导。"""
        with pytest.raises(ValidationError):
            derivative(np.array([0.0]), np.array([1.0]))


class TestEdgeCases:
    """所有公开函数对空输入都必须报错。"""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda: interpolate_series(np.array([]), np.array([]), np.array([0.0])), id="interpolate"),
            pytest.param(lambda: select_intervals(np.array([]), np.array([]), [(0.0, 1.0)]), id="select"),
            pytest.param(lambda: add_gaps(np.array([]), np.array([]), [(0.0, 1.0)]), id="gaps"),
            pytest.param(lambda: remove_peaks(np.array([]), np.array([]), 1.0), id="peaks"),
            pytest.param(lambda: clip_values(np.array([])), id="clip"),
            pytest.param(lambda: derivative(np.array([]), np.array([])), id="derivative"),
        ],
    )
    def test_empty_input_raises(self, call: Callable[[], None]):
        """空序列必须报错，而不是返回空数组。"""
        with pytest.raises(ValidationError):
            call()

    def test_length_mismatch_raises(self):
        """时间轴与数值长度不一致时必须报错。"""
        with pytest.raises(ValidationError):
            add_gaps(LINEAR_TIME, LINEAR_VALUES[:-1], [(1.0, 2.0)])

    def test_non_finite_values_are_rejected_by_interpolation(self):
        """含 NaN 的序列不能直接插值（Acycle 同样要求先预处理）。"""
        values = LINEAR_VALUES.copy()
        values[3] = np.nan
        with pytest.raises(ValidationError):
            interpolate_series(LINEAR_TIME, values, LINEAR_TIME)


def test_series_tools_returns_plain_float_arrays() -> None:
    """返回的必须是普通 float 数组，便于后续直接喂给频谱分析。"""
    result: npt.NDArray = interpolate_series(LINEAR_TIME, LINEAR_VALUES, LINEAR_TIME)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
