# =============================================================================
# FILE: tests/golden/test_moving_median.py
# =============================================================================
"""
``stratigraphy.moving_median``（移动中值滤波）的测试。

本文件是**真值对照**测试：中值滤波的结果可以按定义逐点算出来，因此
测试里写了一份**直接实现**（一个慢但显然正确的循环）作为参照，与
生产代码的快速实现逐点比对。

要证明的核心性质
----------------
**用中值而不是用平均**：序列里混进一个孤立尖峰时，移动中值把它**压掉**
（中位数取的是窗口里的选中值，尖峰在窗口里占少数），而移动平均会被
尖峰**带偏**（尖峰分摊给邻域，尖峰所在位置被拉高，尖峰两侧也被拖偏）。
判据写成两条同时成立的断言：

* 中值滤波在尖峰处的输出与"没有尖峰时的基线"相差不到 1 %；
* 同一窗口的移动平均在尖峰处的输出与基线相差**十倍以上**。

窗口约定
--------
* 奇数窗口中心对称；
* 偶数窗口向前多取一个点（Acycle 的约定），因此窗口是
  ``[i-(h-1), i+h]``；
* 两端截断。

这三条都写进测试：直接实现按同样的约定逐点计算，与快速实现逐点相等，
所以约定一旦被改动，测试立刻失败。

边界
----
* 窗口小于 3、输入为空、输入含非有限值 → 抛
  ``utils.exceptions.ValidationError``。

参考来源
--------
* Tukey, J.W. 1977. Exploratory Data Analysis. Addison-Wesley.
* Hampel, F.R. 1974. The influence curve and its robustness properties.
  Journal of the American Statistical Association 69: 383-397.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"moving_median"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.moving_median import moving_median
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 基线值。
BASELINE = 5.0

#: 尖峰的振幅（相对基线）。
SPIKE = 100.0

#: 尖峰所在的位置（序列的中点）。
SPIKE_INDEX = 10

#: 测试序列的长度。
N_VALUES = 21

#: 滤波窗口。
WINDOW = 5


# =============================================================================
# 辅助函数
# =============================================================================


def _reference_moving_median(values: npt.NDArray, window: int) -> npt.NDArray:
    """按文档约定逐点计算的参照实现（慢，但显然正确）。

    奇数窗口中心对称；偶数窗口向前多取一个点；两端截断。
    """
    size = int(window)
    half = size // 2
    back, forward = (half, half) if size % 2 == 1 else (max(0, half - 1), half)
    output = np.empty(values.size, dtype=float)
    for index in range(values.size):
        start = max(0, index - back)
        stop = min(values.size, index + forward + 1)
        output[index] = float(np.median(values[start:stop]))
    return output


def _spiked_series() -> npt.NDArray:
    """常数基线 + 一个孤立尖峰。"""
    values = np.full(N_VALUES, BASELINE, dtype=float)
    values[SPIKE_INDEX] = BASELINE + SPIKE
    return values


# =============================================================================
# 真值对照
# =============================================================================


def test_an_isolated_spike_is_removed_while_the_mean_is_dragged():
    """中值把孤立尖峰压掉；同一窗口的移动平均却被尖峰带偏。"""
    values = _spiked_series()
    smoothed = moving_median(values, WINDOW).values

    assert abs(float(smoothed[SPIKE_INDEX]) - BASELINE) < 0.01 * SPIKE, (
        "the moving median must reject an isolated spike"
    )

    weights = np.ones(WINDOW) / WINDOW
    averaged = np.convolve(values, weights, mode="valid")
    offset = (WINDOW - 1) // 2
    mean_at_spike = float(averaged[SPIKE_INDEX - offset])
    # 尖峰把它自己的那份（SPIKE / WINDOW）摊给窗口里的每一个样本。
    assert abs(mean_at_spike - BASELINE) > 0.5 * SPIKE / WINDOW, (
        "the moving mean must be dragged by the spike -- that is why the median is used instead"
    )
    assert abs(mean_at_spike - BASELINE) > 10.0 * abs(float(smoothed[SPIKE_INDEX]) - BASELINE)


def test_the_result_matches_a_direct_implementation():
    """快速实现必须与逐点计算的参照实现完全一致（含两端截断）。"""
    rng = np.random.default_rng(17)
    for size in (1, 2, 3, 7, 20, 64):
        for window in (3, 4, 5, 8, 9):
            values = rng.standard_normal(size) * 10.0 + 3.0
            expected = _reference_moving_median(values, window)
            actual = moving_median(values, window).values
            assert actual.shape == values.shape
            assert np.allclose(actual, expected), f"size={size}, window={window}"


def test_the_window_is_centred_for_odd_windows():
    """奇数窗口中心对称：中点上的输出是中心邻域的中值。"""
    values = np.array([0.0, 1.0, 2.0, 30.0, 4.0, 5.0, 6.0])
    smoothed = moving_median(values, 3).values
    assert float(smoothed[3]) == pytest.approx(4.0), "the centre sample must not be pulled by its neighbour"
    assert float(smoothed[2]) == pytest.approx(2.0)


def test_an_even_window_takes_one_extra_forward_point():
    """偶数窗口：向前多取一个点（``[i-(h-1), i+h]``）。"""
    values = np.array([0.0, 1.0, 2.0, 30.0, 5.0, 6.0, 7.0])
    smoothed = moving_median(values, 4).values
    # 窗口 4 -> back = 1, forward = 2：位置 2 的窗口是 [1, 2, 30, 5]。
    assert float(smoothed[2]) == pytest.approx(3.5)
    # 位置 3 的窗口是 [2, 30, 5, 6]。
    assert float(smoothed[3]) == pytest.approx(5.5)
    # 对照：中心对称的奇数窗口 3 在位置 2 只看 [1, 2, 30]，看不到前面的 5。
    assert float(moving_median(values, 3).values[2]) == pytest.approx(2.0)


def test_a_step_edge_is_preserved_better_than_by_a_mean():
    """阶跃边缘：中值保留台阶，平均把它抹平。"""
    values = np.concatenate([np.full(15, 0.0), np.full(15, 10.0)])
    smoothed = moving_median(values, 5).values
    averaged = np.convolve(values, np.ones(5) / 5.0, mode="valid")

    median_edge = float(smoothed[14])
    mean_edge = float(averaged[14 - 2])
    assert abs(median_edge - 0.0) < 1.0, "the median filter must keep the lower level at the step"
    assert abs(mean_edge - 0.0) > abs(median_edge - 0.0), "the mean filter must blur the step more"


def test_a_constant_series_is_unchanged():
    """常数序列的中值滤波仍然是常数。"""
    values = np.full(30, 2.5)
    assert np.allclose(moving_median(values, 5).values, values)


def test_the_result_summarises_itself():
    """结果必须有一份给人看的摘要。

    摘要不是装饰：脚本控制台把结果的 ``summary()`` 印给用户，没有它就只剩
    dataclass 的 ``repr``，整条数组被摊在屏幕上。
    """
    result = moving_median(_spiked_series(), WINDOW)

    summary = result.summary()

    assert summary.strip()
    assert str(WINDOW) in summary


# =============================================================================
# 边界
# =============================================================================


@pytest.mark.parametrize("window", [0, 1, 2, -3])
def test_windows_below_three_are_rejected(window):
    """窗口小于 3 -> ValidationError。"""
    with pytest.raises(ValidationError):
        moving_median(np.arange(10.0), window)


def test_non_integer_windows_are_rejected():
    """窗口必须是整数 -> ValidationError。"""
    with pytest.raises(ValidationError):
        moving_median(np.arange(10.0), 3.5)


def test_empty_and_non_finite_inputs_are_rejected():
    """空输入、含非有限值 -> ValidationError。"""
    with pytest.raises(ValidationError):
        moving_median(np.array([]), 3)
    with pytest.raises(ValidationError):
        moving_median(np.array([1.0, 2.0, np.nan, 4.0]), 3)


def test_a_window_larger_than_the_series_is_handled():
    """窗口比序列还长时，两端截断后仍能算出结果。"""
    values = np.array([1.0, 2.0, 3.0])
    smoothed = moving_median(values, 9).values
    assert smoothed.shape == values.shape
    assert np.all(np.isfinite(smoothed))
    assert np.allclose(smoothed, _reference_moving_median(values, 9))
