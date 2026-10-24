# =============================================================================
# FILE: tests/golden/test_moving_summary.py
# =============================================================================
"""
``stratigraphy.moving_summary`` 的测试：固定带宽与固定点数窗口。

判据来自窗口的**定义**，而不是"跑一遍把输出冻结下来"：

* 固定带宽：窗口是 ``[t - bandwidth / 2, t + bandwidth / 2]`` 的闭区间，
  所以窗口内的常数序列在每个中心点上的均值都等于该常数。
* 固定点数：窗口按样本序号展开，奇数点数关于中心点对称，偶数点数按
  Acycle 的约定多取一个前向点；端点处窗口收缩，不向另一侧借样本。
* 均值的标准误等于标准差除以窗口内样本数的平方根。
* 参数越界（点数窗口不足 3 点、带宽非正）必须抛出 ``ValidationError``。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"fixed_bandwidth_summary", "fixed_count_summary"}


import numpy as np
import pytest

from stratigraphy.moving_summary import fixed_bandwidth_summary, fixed_count_summary
from utils.exceptions import ValidationError

#: 测试用的时间轴（间距 1.0）与数值轴。
TIME = np.arange(20.0)
RAMP = np.arange(20.0)


class TestFixedBandwidthSummary:
    """固定带宽窗口必须按时间宽度取样。"""

    def test_a_constant_series_keeps_its_mean(self):
        """窗口内是常数时，每个窗口的均值都等于该常数。"""
        series = np.full(20, 5.0)
        result = fixed_bandwidth_summary(TIME, series, bandwidth=6.0)
        assert np.allclose(result.mean, 5.0)
        assert np.allclose(result.std, 0.0)
        assert np.allclose(result.standard_error, 0.0)

    def test_the_window_is_centred_on_each_sample(self):
        """斜坡序列在内部中心点的均值等于中心值。"""
        result = fixed_bandwidth_summary(TIME, RAMP, bandwidth=6.0)
        assert result.mean[10] == pytest.approx(10.0)
        assert result.mean[15] == pytest.approx(15.0)

    def test_the_window_spans_the_requested_time_width(self):
        """带宽越大，窗口覆盖的时间越多（跳变被摊进均值）。"""
        series = np.concatenate((np.zeros(10), np.ones(10)))
        narrow = fixed_bandwidth_summary(TIME, series, bandwidth=4.0)
        wide = fixed_bandwidth_summary(TIME, series, bandwidth=20.0)
        # 跳变落在 index 9 的窄窗口里，均值介于两个水平之间
        assert 0.0 < narrow.mean[9] < 1.0
        # 宽带窗口把整段序列都算进去
        assert wide.mean[9] == pytest.approx(0.5)
        assert wide.maximum[9] == pytest.approx(1.0)
        assert wide.minimum[9] == pytest.approx(0.0)

    def test_the_extremes_come_from_the_window(self):
        """最小值与最大值是窗口内的极值。"""
        series = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = fixed_bandwidth_summary(TIME[:5], series, bandwidth=3.0)
        assert result.minimum[2] == pytest.approx(1.0)
        assert result.maximum[2] == pytest.approx(3.0)

    def test_the_standard_error_is_the_standard_deviation_over_the_root_of_n(self):
        """标准误 = 标准差 / sqrt(n)。"""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fixed_bandwidth_summary(TIME[:5], series, bandwidth=5.0)
        assert result.std[2] == pytest.approx(float(np.std(series, ddof=1)))
        assert result.standard_error[2] == pytest.approx(result.std[2] / np.sqrt(5.0))

    def test_the_centres_are_the_input_times(self):
        """窗口中心就是输入的时间轴。"""
        result = fixed_bandwidth_summary(TIME, RAMP, bandwidth=4.0)
        assert np.allclose(result.centres, TIME)

    def test_a_non_positive_bandwidth_raises(self):
        """带宽必须为正。"""
        with pytest.raises(ValidationError):
            fixed_bandwidth_summary(TIME, RAMP, bandwidth=0.0)
        with pytest.raises(ValidationError):
            fixed_bandwidth_summary(TIME, RAMP, bandwidth=-5.0)

    def test_a_mismatched_length_raises(self):
        """时间轴与数值轴必须等长。"""
        with pytest.raises(ValidationError):
            fixed_bandwidth_summary(TIME, RAMP[:-1], bandwidth=4.0)

    def test_a_non_increasing_time_axis_raises(self):
        """时间轴必须严格递增。"""
        scrambled = TIME.copy()
        scrambled[3], scrambled[4] = scrambled[4], scrambled[3]
        with pytest.raises(ValidationError):
            fixed_bandwidth_summary(scrambled, RAMP, bandwidth=4.0)

    def test_too_few_samples_raise(self):
        """点数太少无法给出滑动统计。"""
        with pytest.raises(ValidationError):
            fixed_bandwidth_summary(TIME[:1], RAMP[:1], bandwidth=4.0)


class TestFixedCountSummary:
    """固定点数窗口必须按样本序号展开。"""

    def test_an_odd_window_is_centred_on_the_sample(self):
        """奇数点窗口关于中心点对称，斜坡的均值等于中心值。"""
        result = fixed_count_summary(TIME, RAMP, window=5)
        assert result.mean[10] == pytest.approx(10.0)
        assert result.mean[3] == pytest.approx(3.0)

    def test_an_even_window_takes_one_more_forward_point(self):
        """偶数点窗口多取一个前向点，均值落在中心与下一个点之间。"""
        result = fixed_count_summary(TIME, RAMP, window=4)
        assert result.mean[10] == pytest.approx(10.5)

    def test_the_window_shrinks_at_the_edges(self):
        """端点处窗口收缩，不向另一侧借样本。"""
        result = fixed_count_summary(TIME, RAMP, window=5)
        assert result.mean[0] == pytest.approx(1.0)
        assert result.mean[-1] == pytest.approx(18.0)

    def test_the_window_content_is_defined_by_the_step_signal(self):
        """阶梯信号在窗口跨过跳变时，均值介于两个水平之间。"""
        series = np.concatenate((np.zeros(10), np.ones(10)))
        result = fixed_count_summary(TIME, series, window=5)
        assert result.mean[7] == pytest.approx(0.0)
        assert result.mean[10] == pytest.approx(0.6)
        assert result.mean[12] == pytest.approx(1.0)

    def test_the_standard_error_is_the_standard_deviation_over_the_root_of_n(self):
        """标准误 = 标准差 / sqrt(n)。"""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fixed_count_summary(TIME[:5], series, window=5)
        assert result.std[2] == pytest.approx(float(np.std(series, ddof=1)))
        assert result.standard_error[2] == pytest.approx(result.std[2] / np.sqrt(5.0))

    def test_the_centres_are_the_input_times(self):
        """窗口中心就是输入的时间轴。"""
        result = fixed_count_summary(TIME, RAMP, window=5)
        assert np.allclose(result.centres, TIME)

    def test_a_window_below_three_points_raises(self):
        """点数窗口至少要 3 个样本。"""
        with pytest.raises(ValidationError):
            fixed_count_summary(TIME, RAMP, window=2)
        with pytest.raises(ValidationError):
            fixed_count_summary(TIME, RAMP, window=0)

    def test_a_non_integer_window_raises(self):
        """点数窗口必须是整数。"""
        with pytest.raises(ValidationError):
            fixed_count_summary(TIME, RAMP, window=4.5)

    def test_a_mismatched_length_raises(self):
        """时间轴与数值轴必须等长。"""
        with pytest.raises(ValidationError):
            fixed_count_summary(TIME, RAMP[:-1], window=5)
