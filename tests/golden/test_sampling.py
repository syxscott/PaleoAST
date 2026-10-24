# =============================================================================
# FILE: tests/golden/test_sampling.py
# =============================================================================
"""
``stratigraphy.sampling``（采样均匀性检测）的测试。

断言围绕三条主线组织：

* **等间距 ⇒ 均匀**。一条完全等间距的时间轴必须被判为 ``is_uniform``，
  且所有相对间距都等于 1（相对间距 = 间距 / 中位间距）。
* **出现大间隔 ⇒ 不均匀**。在一段等间距序列里挖掉一个间隔之后，最大相对
  间距必须明显大于 1，判定必须翻成 ``False``。
* **容差是有意义的**。Acycle 口径的判据是"最大相对间距偏离 ≤ 容差"
  （默认 10 ppm）。因此同一个轻微抖动的序列，在默认容差下判为不均匀，
  把容差放宽后判为均匀 —— 这条断言同时钉死了容差的语义和默认值。

真值来源
--------
* 等间距序列的相对间距恒为 1（定义）。
* 大间隔用例的相对间距：间隔被挖掉后，中位间距仍是原间距，被挖掉的位置
  变成两倍间距 ⇒ 相对间距为 2.0（算术直接给出）。
* 判据本身：Acycle ``code/seriesanalysis/acycleSamplingIsUneven.m`` 的算法
  说明（相对间距容差 1e-5）。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"test_uniform_sampling"}


import numpy as np
import pytest

from stratigraphy.sampling import DEFAULT_TOLERANCE, test_uniform_sampling
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 等间距序列的间距。
STEP = 0.1

#: 等间距序列的点数。
N_POINTS = 101

#: 被挖掉的间隔的个数（挖掉一个 ⇒ 该处的间距翻倍）。
N_REMOVED = 1


# =============================================================================
# 辅助函数
# =============================================================================


def _regular_grid(n_points: int = N_POINTS, step: float = STEP) -> np.ndarray:
    """完全等间距的时间轴。"""
    return np.arange(n_points, dtype=float) * step


def _gapped_grid(n_removed: int = N_REMOVED, n_points: int = N_POINTS, step: float = STEP) -> np.ndarray:
    """在等间距序列里挖掉 ``n_removed`` 个间隔。"""
    return np.concatenate(
        [
            np.arange(n_points // 2, dtype=float) * step,
            np.arange(n_points // 2 + n_removed, n_points, dtype=float) * step,
        ]
    )


def _jittered_grid(magnitude: float = 5.0e-5, n_points: int = N_POINTS, step: float = STEP) -> np.ndarray:
    """交替抖动的网格。

    奇数号采样点整体平移 ``magnitude``，于是相邻间距交替为
    ``step + magnitude`` 与 ``step − magnitude``：相对偏离
    ``magnitude / step``（默认 5e-4）。
    """
    grid = _regular_grid(n_points, step)
    offsets = np.where(np.arange(n_points) % 2 == 0, 0.0, magnitude)
    return grid + offsets


# =============================================================================
# 等间距
# =============================================================================


class TestRegularGrid:
    """等间距序列必须判为均匀。"""

    def test_regular_grid_is_uniform(self):
        result = test_uniform_sampling(_regular_grid())

        assert result.is_uniform is True, f"a regular grid was flagged as uneven (p = {result.p_value:.4g})"

    def test_relative_gaps_of_a_regular_grid_are_all_one(self):
        result = test_uniform_sampling(_regular_grid())

        assert result.relative_gaps.shape == (N_POINTS - 1,)
        assert np.allclose(result.relative_gaps, 1.0, rtol=0.0, atol=1e-12)

    def test_regular_grid_has_a_large_p_value(self):
        """卡方检验不应把规则栅格判为与均匀采样相悖。"""
        result = test_uniform_sampling(_regular_grid())

        assert result.p_value > 0.05, f"a regular grid was rejected: p = {result.p_value:.4g}"

    def test_statistic_is_non_negative(self):
        result = test_uniform_sampling(_regular_grid())

        assert result.statistic >= 0.0


# =============================================================================
# 有大间隔
# =============================================================================


class TestGappedGrid:
    """出现大间隔的序列必须判为不均匀。"""

    def test_gapped_grid_is_not_uniform(self):
        result = test_uniform_sampling(_gapped_grid())

        assert result.is_uniform is False

    def test_the_gap_doubles_the_relative_spacing(self):
        """挖掉一个间隔 ⇒ 该处的相对间距为 2.0（中位间距不变）。"""
        result = test_uniform_sampling(_gapped_grid())

        assert np.max(result.relative_gaps) == pytest.approx(2.0, rel=1e-12)

    def test_more_gaps_make_it_even_more_uneven(self):
        mild = test_uniform_sampling(_gapped_grid(n_removed=1))
        severe = test_uniform_sampling(_gapped_grid(n_removed=5))

        assert severe.is_uniform is False
        assert float(np.max(severe.relative_gaps)) > float(np.max(mild.relative_gaps))


# =============================================================================
# 容差
# =============================================================================


class TestTolerance:
    """容差决定"轻微抖动"算不算均匀（Acycle 口径）。"""

    def test_default_tolerance_is_the_acycle_value(self):
        assert DEFAULT_TOLERANCE == 1.0e-5

    def test_a_slightly_jittered_grid_is_uneven_by_default(self):
        """相对偏离 5e-4 的抖动超过默认容差 1e-5 ⇒ 不均匀。"""
        result = test_uniform_sampling(_jittered_grid())

        assert result.is_uniform is False

    def test_a_slightly_jittered_grid_is_uniform_with_a_loose_tolerance(self):
        result = test_uniform_sampling(_jittered_grid(), tolerance=1.0e-2)

        assert result.is_uniform is True


# =============================================================================
# 退化输入
# =============================================================================


class TestDegenerateInput:
    """非有限或非递增的坐标轴按 Acycle 口径判为"不均匀"。"""

    def test_non_increasing_axis_is_not_uniform(self):
        result = test_uniform_sampling(np.array([0.0, 1.0, 1.0, 2.0]))

        assert result.is_uniform is False

    def test_non_finite_axis_is_not_uniform(self):
        result = test_uniform_sampling(np.array([0.0, 1.0, np.nan, 2.0]))

        assert result.is_uniform is False


# =============================================================================
# 结果对象与边界
# =============================================================================


class TestResultAndBoundaries:
    """结果对象与非法输入。"""

    def test_summary_is_a_non_empty_string(self):
        result = test_uniform_sampling(_regular_grid())

        summary = result.summary()
        assert isinstance(summary, str)
        assert summary.strip()

    def test_too_few_samples_are_rejected(self):
        with pytest.raises(ValidationError):
            test_uniform_sampling(np.array([0.0]))

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValidationError):
            test_uniform_sampling(np.array([]))

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValidationError):
            test_uniform_sampling(np.zeros((2, 3)))

    def test_negative_tolerance_is_rejected(self):
        with pytest.raises(ValidationError):
            test_uniform_sampling(_regular_grid(), tolerance=-1.0)

    def test_alpha_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValidationError):
            test_uniform_sampling(_regular_grid(), alpha=0.0)
