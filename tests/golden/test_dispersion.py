# =============================================================================
# FILE: tests/golden/test_dispersion.py
# =============================================================================
"""
``stratigraphy.dispersion``（离散度统计 / 离散分布拟合优度）的测试。

断言围绕三条主线组织：

* **完全符合期望分布 ⇒ 不显著**。观测频数逐位等于期望频数时，卡方统计量
  必须是 0、p 值必须是 1（真值即定义）。另有一条**随机**样本的用例：
  频数由期望分布的多项抽样得到（固定种子 ⇒ 可复现），p 值必须不显著。
* **明显偏离 ⇒ 极显著**。把全部频数堆到第一个箱里，统计量必须很大、
  p 值必须小到可以当成 0。三种方法（``chi2`` / ``g_test`` /
  ``anderson_darling``）都必须给出这个方向的结论。
* **期望值的三种写法等价**。``expected=None``（均匀期望）、期望**频数**、
  期望**概率**（和为 1）三者描述的是同一个零假设，结果必须一致。

真值来源
--------
* 完全吻合时的 ``χ² = 0`` 与 ``p = 1``：由 Pearson 卡方检验的定义直接给出。
* 偏离用例的统计量：由 ``Σ (Oᵢ − Eᵢ)² / Eᵢ`` 手算（观测 ``[100,0,0,0,0]``
  对均匀期望 ``[20,20,20,20,20]``：统计量 = ``80²/20 + 4 × 20²/20 = 400``，
  自由度 4）。
* 随机样本的 p 值：固定种子的多项抽样，可复现。
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"dispersion_statistic"}


import numpy as np
import pytest

from stratigraphy.dispersion import dispersion_statistic
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 均匀零假设下的箱数。
N_BINS = 5

#: 每箱的观测频数（完全符合期望）。
UNIFORM_COUNTS = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

#: 全部频数堆到第一箱（明显偏离均匀期望）。
CLUMPED_COUNTS = np.array([100.0, 0.0, 0.0, 0.0, 0.0])

#: 均匀期望（各箱相同）。
UNIFORM_EXPECTED = np.array([20.0, 20.0, 20.0, 20.0, 20.0])

#: 一组非均匀的期望概率（和为 1）。
SKEWED_PROBABILITIES = np.array([0.1, 0.2, 0.3, 0.25, 0.15])


# =============================================================================
# 辅助函数
# =============================================================================


def _multinomial_sample(seed: int = 0, n: int = 500) -> np.ndarray:
    """从 ``SKEWED_PROBABILITIES`` 抽一个多项样本（固定种子 ⇒ 可复现）。"""
    generator = np.random.default_rng(seed)
    return generator.multinomial(n, SKEWED_PROBABILITIES).astype(float)


# =============================================================================
# 完全符合期望分布
# =============================================================================


class TestConformingSample:
    """完全符合期望分布的样本必须给出不显著的 p 值。"""

    def test_exact_match_gives_zero_statistic_and_unit_p_value(self):
        result = dispersion_statistic(UNIFORM_COUNTS, UNIFORM_COUNTS)

        assert result.statistic == pytest.approx(0.0, abs=1e-12)
        assert result.p_value == pytest.approx(1.0, abs=1e-12)

    def test_sample_drawn_from_the_expected_distribution_is_not_significant(self):
        """随机多项样本：p 值必须不显著（固定种子 ⇒ 可复现）。"""
        counts = _multinomial_sample()

        result = dispersion_statistic(counts, SKEWED_PROBABILITIES)

        assert result.p_value > 0.05, f"a conforming sample was flagged as deviating: p = {result.p_value:.4g}"

    def test_uniform_expectation_matches_explicit_uniform_counts(self):
        """``expected=None``（均匀期望）与显式的均匀期望必须给出同一结果。"""
        counts = _multinomial_sample()
        scaled_expected = UNIFORM_EXPECTED * (counts.sum() / UNIFORM_EXPECTED.sum())

        implicit = dispersion_statistic(counts)
        explicit = dispersion_statistic(counts, scaled_expected)

        assert implicit.statistic == pytest.approx(explicit.statistic, rel=1e-12)
        assert implicit.p_value == pytest.approx(explicit.p_value, rel=1e-12)

    def test_expected_given_as_probabilities_equals_expected_given_as_counts(self):
        """期望写成"概率"（和为 1）与写成"频数"必须给出同一结果。"""
        counts = _multinomial_sample()
        expected_counts = SKEWED_PROBABILITIES * counts.sum()

        from_probabilities = dispersion_statistic(counts, SKEWED_PROBABILITIES)
        from_counts = dispersion_statistic(counts, expected_counts)

        assert from_probabilities.statistic == pytest.approx(from_counts.statistic, rel=1e-12)
        assert from_probabilities.p_value == pytest.approx(from_counts.p_value, rel=1e-12)


# =============================================================================
# 明显偏离期望分布
# =============================================================================


class TestDeviatingSample:
    """明显偏离期望分布的样本必须给出极小的 p 值。"""

    def test_chi2_statistic_matches_the_hand_computed_value(self):
        """卡方统计量 = Σ (Oᵢ − Eᵢ)²/Eᵢ = 80²/20 + 4 × 20²/20 = 400（手算真值）。"""
        result = dispersion_statistic(CLUMPED_COUNTS, UNIFORM_EXPECTED)

        assert result.statistic == pytest.approx(400.0, rel=1e-12)

    def test_chi2_p_value_is_tiny(self):
        result = dispersion_statistic(CLUMPED_COUNTS, UNIFORM_EXPECTED)

        assert result.p_value < 1e-10, f"the deviation was not detected: p = {result.p_value:.4g}"

    def test_chi2_detects_a_clumped_sample_against_the_uniform_expectation(self):
        result = dispersion_statistic(CLUMPED_COUNTS)

        assert result.p_value < 1e-10, f"the deviation was not detected: p = {result.p_value:.4g}"
        assert result.statistic > 0.0

    def test_g_test_detects_the_deviation(self):
        result = dispersion_statistic(CLUMPED_COUNTS, UNIFORM_EXPECTED, method="g_test")

        assert result.statistic > 0.0
        assert result.p_value < 1e-10, f"the deviation was not detected: p = {result.p_value:.4g}"

    def test_g_test_gives_zero_for_an_exact_match(self):
        result = dispersion_statistic(UNIFORM_COUNTS, UNIFORM_COUNTS, method="g_test")

        assert result.statistic == pytest.approx(0.0, abs=1e-12)
        assert result.p_value == pytest.approx(1.0, abs=1e-12)

    def test_anderson_darling_detects_the_deviation(self):
        result = dispersion_statistic(CLUMPED_COUNTS, UNIFORM_EXPECTED, method="anderson_darling")

        assert result.statistic > 0.0
        assert result.p_value < 0.01, f"the deviation was not detected: p = {result.p_value:.4g}"

    def test_anderson_darling_is_not_significant_for_an_exact_match(self):
        result = dispersion_statistic(UNIFORM_COUNTS, UNIFORM_COUNTS, method="anderson_darling")

        assert result.p_value > 0.05, f"an exact match was flagged: p = {result.p_value:.4g}"

    def test_anderson_darling_p_value_is_reproducible(self):
        first = dispersion_statistic(CLUMPED_COUNTS, UNIFORM_EXPECTED, method="anderson_darling")
        second = dispersion_statistic(CLUMPED_COUNTS, UNIFORM_EXPECTED, method="anderson_darling")

        assert first.p_value == second.p_value


# =============================================================================
# 结果对象
# =============================================================================


class TestResult:
    """结果对象必须把参与计算的观测/期望频数原样带出来。"""

    def test_observed_and_expected_are_returned(self):
        result = dispersion_statistic(UNIFORM_COUNTS, UNIFORM_EXPECTED)

        assert np.allclose(result.observed, UNIFORM_COUNTS)
        assert np.allclose(result.expected, UNIFORM_EXPECTED)

    def test_summary_is_a_non_empty_string(self):
        result = dispersion_statistic(UNIFORM_COUNTS, UNIFORM_EXPECTED)

        summary = result.summary()
        assert isinstance(summary, str)
        assert summary.strip()


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(UNIFORM_COUNTS, np.array([10.0, 10.0]))

    def test_negative_counts_are_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(np.array([10.0, -1.0, 10.0]), np.array([10.0, 10.0, 10.0]))

    def test_non_finite_counts_are_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(np.array([10.0, np.inf, 10.0]), np.array([10.0, 10.0, 10.0]))

    def test_a_single_bin_is_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(np.array([10.0]), np.array([10.0]))

    def test_zero_total_is_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(np.zeros(3), np.array([1.0, 1.0, 1.0]))

    def test_zero_expected_count_is_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(np.array([10.0, 10.0]), np.array([10.0, 0.0]))

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(UNIFORM_COUNTS, UNIFORM_EXPECTED, method="ks")

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValidationError):
            dispersion_statistic(np.zeros((2, 2)), np.zeros((2, 2)))
