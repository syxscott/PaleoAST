# =============================================================================
# FILE: tests/golden/test_depth_time.py
# =============================================================================
"""
``stratigraphy.depth_time``（深度 → 时间转换）的测试。

断言围绕三条主线组织：

* **锚点处精确**。给定"深度–年龄"锚点，插值出来的年龄在锚点深度上必须
  **逐位等于**给定的年龄（不是"接近"）。这是年龄模型最基本的性质：任何
  插值方法都不能把一个已知的年代约束插歪。
* **单调与分段线性**。年龄必须随深度单调递增；锚点之间是分段线性插值，
  因此在区间中点上的年龄等于两端年龄的平均 —— 这一条与手算结果对照，
  与插值实现无关。
* **端段速率线性外推**。锚点范围之外按端段速率外推，与 Acycle
  ``code/seriesanalysis/depthtotime.m`` 的算法规格一致：外推段与锚点段在
  边界处连续（深度等于最末锚点时，外推值等于该锚点的年龄）。

真值来源
--------
* 线性年龄–深度关系 ``age = 0.25 × depth``：真值由算术直接给出。
* 分段线性模型（两段不同速率）：中点年龄由两段速率手算给出。
* 锚点处精确性：真值即输入的锚点年龄本身。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"depth_to_time"}


import numpy as np
import pytest

from stratigraphy.depth_time import depth_to_time
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 线性年龄–深度关系的真值速率（Ma / m）。
TRUE_RATE_MA_PER_M = 0.25

#: 第一段的沉积速率（m / Ma），用于分段线性模型。
FIRST_RATE_M_PER_MA = 20.0

#: 第二段的沉积速率（m / Ma），用于分段线性模型。
SECOND_RATE_M_PER_MA = 40.0


# =============================================================================
# 辅助函数
# =============================================================================


def _linear_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一条严格线性的年龄–深度关系。

    Returns:
        (采样深度, 锚点深度, 锚点年龄)。
    """
    depths = np.linspace(0.0, 40.0, 9)
    anchor_depths = np.array([0.0, 20.0, 40.0])
    anchor_ages = TRUE_RATE_MA_PER_M * anchor_depths
    return depths, anchor_depths, anchor_ages


def _two_segment_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """两段不同沉积速率的年龄–深度关系。

    0–10 m 以 ``FIRST_RATE_M_PER_MA`` 沉积，10–30 m 以
    ``SECOND_RATE_M_PER_MA`` 沉积。锚点年龄由手算给出：
    10 m 处为 ``10 / FIRST_RATE``，30 m 处为
    ``10 / FIRST_RATE + 20 / SECOND_RATE``。
    """
    depths = np.linspace(0.0, 30.0, 13)
    anchor_depths = np.array([0.0, 10.0, 30.0])
    anchor_ages = np.array(
        [
            0.0,
            10.0 / FIRST_RATE_M_PER_MA,
            10.0 / FIRST_RATE_M_PER_MA + 20.0 / SECOND_RATE_M_PER_MA,
        ]
    )
    return depths, anchor_depths, anchor_ages


# =============================================================================
# 锚点处的精确性
# =============================================================================


class TestAnchorsAreExact:
    """锚点深度处的年龄必须逐位等于给定年龄。"""

    @pytest.mark.parametrize("model", ["linear", "pchip"])
    def test_ages_at_anchors_are_exactly_the_anchor_ages(self, model: str):
        _depths, anchor_depths, anchor_ages = _linear_case()

        result = depth_to_time(anchor_depths, anchor_depths, anchor_ages, model=model)

        assert np.allclose(result.ages_ma, anchor_ages, rtol=0.0, atol=1e-12), (
            f"the ages at the anchor depths are not the anchor ages: {result.ages_ma}"
        )

    def test_ages_are_monotonic_in_depth(self):
        depths, anchor_depths, anchor_ages = _linear_case()

        result = depth_to_time(depths, anchor_depths, anchor_ages)

        assert np.all(np.diff(result.ages_ma) > 0.0), "the ages are not strictly increasing with depth"

    def test_result_keeps_the_input_depths(self):
        depths, anchor_depths, anchor_ages = _linear_case()

        result = depth_to_time(depths, anchor_depths, anchor_ages)

        assert result.depths.shape == depths.shape
        assert np.allclose(result.depths, depths)


# =============================================================================
# 与手算真值对照
# =============================================================================


class TestAgainstHandComputedTruth:
    """插值结果必须与独立算出的真值一致。"""

    @pytest.mark.parametrize("model", ["linear", "pchip"])
    def test_linear_age_depth_relationship(self, model: str):
        """线性年龄–深度关系：age = 0.25 × depth（真值由算术给出）。"""
        depths, anchor_depths, anchor_ages = _linear_case()

        result = depth_to_time(depths, anchor_depths, anchor_ages, model=model)

        expected = TRUE_RATE_MA_PER_M * depths
        assert np.allclose(result.ages_ma, expected, rtol=1e-12, atol=1e-12), (
            f"the modelled ages do not follow the true linear age-depth law: {result.ages_ma}"
        )

    def test_midpoint_of_a_segment_is_the_average_of_its_ends(self):
        """分段线性插值：区间中点的年龄等于两端年龄的平均。"""
        _depths, anchor_depths, anchor_ages = _two_segment_case()
        midpoint_depth = 0.5 * (anchor_depths[1] + anchor_depths[2])
        result = depth_to_time(np.array([midpoint_depth]), anchor_depths, anchor_ages)

        expected = 0.5 * (anchor_ages[1] + anchor_ages[2])
        assert float(result.ages_ma[0]) == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_two_segment_rates_are_recovered_between_anchors(self):
        """第二段内的年龄必须由第二段的沉积速率决定（手算真值）。"""
        _depths, anchor_depths, anchor_ages = _two_segment_case()
        probe = np.array([20.0])

        result = depth_to_time(probe, anchor_depths, anchor_ages)

        expected = anchor_ages[1] + (20.0 - anchor_depths[1]) / SECOND_RATE_M_PER_MA
        assert float(result.ages_ma[0]) == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_extrapolation_uses_the_end_segment_rate(self):
        """锚点范围之外按端段速率线性外推，且在边界处与锚点段连续。"""
        anchor_depths = np.array([0.0, 10.0, 20.0])
        anchor_ages = np.array([0.0, 0.5, 1.0])
        # 10 m 对应 0.5 Ma ⇒ 20 m/Ma；末端外推到 25 m ⇒ 1.0 + 5/20 = 1.25 Ma。
        probe = np.array([20.0, 25.0, 30.0])

        result = depth_to_time(probe, anchor_depths, anchor_ages)

        expected = np.array([1.0, 1.25, 1.5])
        assert np.allclose(result.ages_ma, expected, rtol=1e-12, atol=1e-12), (
            f"the extrapolation does not follow the end-segment rate: {result.ages_ma}"
        )

    def test_leading_extrapolation_uses_the_first_segment_rate(self):
        """前端外推按第一段速率，且在边界处与锚点段连续。"""
        anchor_depths = np.array([0.0, 10.0, 20.0])
        anchor_ages = np.array([0.0, 0.5, 1.0])
        probe = np.array([-10.0, 0.0])

        result = depth_to_time(probe, anchor_depths, anchor_ages)

        expected = np.array([-0.5, 0.0])
        assert np.allclose(result.ages_ma, expected, rtol=1e-12, atol=1e-12)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_non_increasing_depths_are_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_non_increasing_anchors_are_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, 1.0]), np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0]))

    def test_non_increasing_anchor_ages_are_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0]))

    def test_mismatched_anchor_lengths_are_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))

    def test_a_single_anchor_is_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]))

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), model="spline")

    def test_non_finite_values_are_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([0.0, np.nan]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_empty_depths_are_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.array([]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValidationError):
            depth_to_time(np.zeros((2, 2)), np.array([0.0, 1.0]), np.array([0.0, 1.0]))


# =============================================================================
# 摘要
# =============================================================================


def test_summary_is_a_non_empty_string():
    depths, anchor_depths, anchor_ages = _linear_case()

    result = depth_to_time(depths, anchor_depths, anchor_ages)

    summary = result.summary()
    assert isinstance(summary, str)
    assert summary.strip()
