# =============================================================================
# FILE: tests/golden/test_inverse_percentile.py
# =============================================================================
"""
``stratigraphy.inverse_percentile``（逆百分位与超越概率）的测试。

断言围绕三条主线组织：

* **反解出的分位数值要接近原值**。对一条标准正态样本（numpy 生成），
  给定非超越概率 ``p``，反解出的分位数值必须与 ``numpy.quantile`` 给出的
  同一个分位数值接近。⚠️ 二者的绘图位置约定不同（Acycle 用 Hazen，
  numpy 默认用 ``h = (n−1)q``），因此这里断言的是"接近"（差值 ≪ 1 个标准差），
  而不是逐位相同；差异量级是 ``O(1/n)`` 个次序统计量。
* **两个方向互为反函数**。``non_exceedance_probability``（Acycle
  ``invprctile.m`` 的方向）与 ``inverse_percentile`` 必须精确互逆 —— 这条
  断言是逐位精确的。
* **可解析的小样本真值**。样本 ``1…100``、Hazen 绘图位置时，``p = 0.5``
  必须给出 50.5、``p = 0.25`` 必须给出 25.5（真值由算术直接给出）。

真值来源
--------
* 分位数值：Acycle ``code/misc/invprctile.m`` 的绘图位置约定（Hazen：
  第 ``i`` 个次序统计量的非超越概率是 ``(i − 0.5)/n``）。
* 可解析真值：``1…100`` 的 Hazen 位置 ``(i − 0.5)/100``，``p = 0.5`` 落在
  ``i = 50``（0.495）与 ``i = 51``（0.505）之间 ⇒ 线性插值给出 50.5。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"inverse_percentile"}


from itertools import pairwise

import numpy as np
import pytest

from stratigraphy.inverse_percentile import inverse_percentile, non_exceedance_probability
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 标准正态样本的点数。
N_SAMPLES = 4000

#: 随机种子（固定 ⇒ 可复现）。
SEED = 7

#: 与 ``numpy.quantile`` 对照时的容差（绘图位置约定不同 ⇒ 只要求接近）。
NPY_QUANTILE_TOLERANCE = 0.01


# =============================================================================
# 辅助函数
# =============================================================================


def _normal_sample(n_samples: int = N_SAMPLES, seed: int = SEED) -> np.ndarray:
    """标准正态样本。"""
    generator = np.random.default_rng(seed)
    return generator.standard_normal(n_samples)


def _integer_sample() -> np.ndarray:
    """样本 ``1…100``（Hazen 位置下分位数值可以手算）。"""
    return np.arange(1.0, 101.0)


# =============================================================================
# 与 numpy 的分位数值对照
# =============================================================================


class TestAgainstNumpyQuantiles:
    """反解出的分位数值必须接近 numpy 给出的同一个分位数值。"""

    @pytest.mark.parametrize("probability", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_normal_sample_quantiles_match_numpy(self, probability: float):
        sample = _normal_sample()

        value = inverse_percentile(sample, probability).value
        expected = float(np.quantile(sample, probability))

        assert abs(value - expected) < NPY_QUANTILE_TOLERANCE, (
            f"the recovered quantile at p = {probability} is {value:.6f}, numpy says {expected:.6f}"
        )

    def test_the_recovered_quantile_is_monotonic_in_the_probability(self):
        sample = _normal_sample()

        values = [inverse_percentile(sample, probability).value for probability in (0.1, 0.25, 0.5, 0.75, 0.9)]

        assert all(earlier < later for earlier, later in pairwise(values))

    def test_the_median_of_a_symmetric_sample_is_close_to_zero(self):
        sample = _normal_sample()

        assert abs(inverse_percentile(sample, 0.5).value) < 0.05


# =============================================================================
# 与 Acycle 方向互为反函数
# =============================================================================


class TestRoundTrip:
    """``non_exceedance_probability`` 与 ``inverse_percentile`` 必须精确互逆。"""

    @pytest.mark.parametrize("probability", [0.05, 0.25, 0.5, 0.75, 0.95])
    def test_probability_of_the_recovered_quantile_is_the_probability(self, probability: float):
        sample = _normal_sample()

        value = inverse_percentile(sample, probability).value
        recovered = non_exceedance_probability(sample, value)

        assert recovered == pytest.approx(probability, abs=1e-9)

    def test_recovered_quantile_of_the_recovered_probability_is_stable(self):
        sample = _normal_sample()

        value = inverse_percentile(sample, 0.9).value
        recovered = non_exceedance_probability(sample, value)

        assert inverse_percentile(sample, recovered).value == pytest.approx(value, abs=1e-9)


# =============================================================================
# 可解析的小样本真值
# =============================================================================


class TestAnalyticTruth:
    """小样本上的分位数值可以手算，必须逐位对上。"""

    def test_median_of_one_to_one_hundred(self):
        """Hazen 位置：``p = 0.5`` 落在第 50 与第 51 个次序统计量之间 ⇒ 50.5。"""
        value = inverse_percentile(_integer_sample(), 0.5).value

        assert value == pytest.approx(50.5, abs=1e-12)

    def test_quarter_of_one_to_one_hundred(self):
        """``p = 0.25`` 落在第 25 与第 26 个次序统计量之间 ⇒ 25.5。"""
        value = inverse_percentile(_integer_sample(), 0.25).value

        assert value == pytest.approx(25.5, abs=1e-12)

    def test_probability_zero_and_one_give_the_extremes(self):
        sample = _integer_sample()

        assert inverse_percentile(sample, 0.0).value == pytest.approx(1.0, abs=1e-12)
        assert inverse_percentile(sample, 1.0).value == pytest.approx(100.0, abs=1e-12)

    def test_non_exceedance_probability_of_the_extremes(self):
        sample = _integer_sample()

        assert non_exceedance_probability(sample, 0.0) == pytest.approx(0.0, abs=1e-12)
        assert non_exceedance_probability(sample, 101.0) == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_probability_below_zero_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(_integer_sample(), -0.1)

    def test_probability_above_one_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(_integer_sample(), 1.1)

    def test_non_finite_probability_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(_integer_sample(), float("nan"))

    def test_empty_sample_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(np.array([]), 0.5)

    def test_two_dimensional_sample_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(np.zeros((2, 3)), 0.5)

    def test_non_finite_sample_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(np.array([1.0, np.nan, 3.0]), 0.5)

    def test_unknown_plotting_position_is_rejected(self):
        with pytest.raises(ValidationError):
            inverse_percentile(_integer_sample(), 0.5, plotting_position="unknown")

    def test_non_finite_quantile_is_rejected(self):
        with pytest.raises(ValidationError):
            non_exceedance_probability(_integer_sample(), float("inf"))


# =============================================================================
# 摘要
# =============================================================================


def test_the_result_summarises_itself():
    """结果必须有一份给人看的摘要。

    脚本控制台把结果的 ``summary()`` 印给用户；没有它就只剩 dataclass 的
    ``repr``，一次逆百分位只剩一串裸数字，看不出它是什么。
    """
    result = inverse_percentile(_integer_sample(), 0.5)

    summary = result.summary()

    assert summary.strip()
    assert "50.5" in summary
