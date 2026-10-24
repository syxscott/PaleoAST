# =============================================================================
# FILE: tests/golden/test_anderson_darling.py
# =============================================================================
"""
``stats.anderson_darling``（k 样本 Anderson-Darling 检验）的测试。

核心断言是**已发表的算例**：Scholz & Stephens (1987, p. 922) 用四种纸的
"光滑度"数据演示了该检验，四家实验室各测 8 次，共 32 个观测、3 个并列值。
论文（以及随该方法发布的实现）给出的数字是本文件的真值：

================================  ==============
量                                论文值
================================  ==============
统计量的均值 (k - 1)               3
统计量的标准差                     1.2037664
统计量（不校正并列值）             8.3558723
标准化统计量（不校正并列值）        4.4492622
概率（不校正并列值）                0.0022547
统计量（校正并列值）               8.3926093
标准化统计量（校正并列值）          4.4797806
概率（校正并列值）                  0.0021700
================================  ==============

本模块复现了这八个数字（相对误差 <= 1.7e-5，而论文值本身只印到 7 位小数，
故概率一栏的误差正好落在打印精度内）。

其余断言钉住的是这个检验**作为秩检验**应有的性质：

* **尺度不变**：全体数值同乘一个常数，统计量与 p 值一字不变——因为秩没变。
* **平移不变**：全体数值同加一个常数，同样不变。
* **同分布的分组不显著**：把同一总体随机分成 k 组，统计量应当落在零假设
  均值附近，p 值不应当小。
* **分得越开越显著**：两组逐渐拉开，统计量单调增大、p 值单调减小。
* **结论随 k 变化**：k 越大，同样分开的分组应当越容易被判为不同总体。

边界测试断言抛 ``utils.exceptions.ValidationError``。

参考来源（断言中引用的数值出处）
--------------------------------
* Scholz, F. W. & Stephens, M. A. (1987). K-Sample Anderson-Darling Tests.
  *Journal of the American Statistical Association*, 82(399), 918-924.
  （该检验的方法论文，以及上面那个算例。）
* Anderson, T. W. & Darling, D. A. (1952). Asymptotic theory of certain
  goodness of fit criteria based on stochastic processes. *Annals of
  Mathematical Statistics*, 23(2), 193-212.
* Darling, D. A. (1957). The Kolmogorov-Smirnov, Cramer-von Mises tests.
  *Annals of Mathematical Statistics*, 28(3), 823-838.
* Pettitt, A. N. (1976). A two-sample Anderson-Darling rank statistic.
  *Biometrika*, 63(1), 161-168.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"anderson_darling_ksample"}


import numpy as np
import pytest

from stats.anderson_darling import anderson_darling_ksample
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: Scholz & Stephens (1987) 的已发表算例：四种纸的光滑度，四家实验室。
PUBLISHED_LABORATORIES = {
    1: [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0],
    2: [39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8],
    3: [34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0],
    4: [34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8],
}

#: 论文给出的八个数。
PUBLISHED_MEAN = 3.0
PUBLISHED_STD = 1.2037664
PUBLISHED_STATISTIC_NO_TIES = 8.3558723
PUBLISHED_STANDARDIZED_NO_TIES = 4.4492622
PUBLISHED_P_NO_TIES = 0.0022547
PUBLISHED_STATISTIC = 8.3926093
PUBLISHED_STANDARDIZED = 4.4797806
PUBLISHED_P = 0.0021700


def _published_case() -> tuple[np.ndarray, np.ndarray]:
    """The published example as two parallel arrays."""
    values = np.array([v for group in PUBLISHED_LABORATORIES.values() for v in group], dtype=float)
    groups = np.array([g for g, group in PUBLISHED_LABORATORIES.items() for _ in group], dtype=float)
    return values, groups


# =============================================================================
# 已发表的算例：真值对照
# =============================================================================
class TestPublishedExample:
    """Scholz & Stephens (1987), p. 922."""

    def test_the_shape_of_the_example(self):
        values, groups = _published_case()
        result = anderson_darling_ksample(values, groups)
        assert result.n_groups == 4
        assert result.group_sizes == (8, 8, 8, 8)
        assert result.n_total == 32
        assert result.n_ties == 3
        assert result.n_distinct == 29

    def test_the_null_mean_and_standard_deviation(self):
        values, groups = _published_case()
        result = anderson_darling_ksample(values, groups)
        assert result.mean == pytest.approx(PUBLISHED_MEAN, abs=1e-12)
        assert result.std == pytest.approx(PUBLISHED_STD, rel=1e-7)

    def test_the_statistic_ignoring_ties(self):
        values, groups = _published_case()
        result = anderson_darling_ksample(values, groups)
        assert result.statistic_no_ties == pytest.approx(PUBLISHED_STATISTIC_NO_TIES, rel=1e-7)
        assert result.standardized_no_ties == pytest.approx(PUBLISHED_STANDARDIZED_NO_TIES, rel=1e-7)

    def test_the_statistic_adjusted_for_ties(self):
        values, groups = _published_case()
        result = anderson_darling_ksample(values, groups)
        assert result.statistic == pytest.approx(PUBLISHED_STATISTIC, rel=1e-7)
        assert result.standardized == pytest.approx(PUBLISHED_STANDARDIZED, rel=1e-7)

    def test_the_probabilities(self):
        """The published probabilities are printed to seven decimals.

        The rounding of the printed value is 5e-8, so the agreement has to be
        checked against that resolution rather than against machine epsilon.
        """
        values, groups = _published_case()
        result = anderson_darling_ksample(values, groups)
        assert result.p_value == pytest.approx(PUBLISHED_P, abs=5e-8)
        assert result.p_value_no_ties == pytest.approx(PUBLISHED_P_NO_TIES, abs=5e-8)

    def test_the_example_is_significant(self):
        values, groups = _published_case()
        result = anderson_darling_ksample(values, groups, alpha=0.05)
        assert result.significant is True
        assert result.p_value < 0.05


# =============================================================================
# 作为秩检验的性质
# =============================================================================
class TestRankInvariance:
    """The test uses ranks, so it cannot see a shift or a rescaling."""

    def test_multiplying_every_value_leaves_the_result_unchanged(self):
        values, groups = _published_case()
        base = anderson_darling_ksample(values, groups)
        scaled = anderson_darling_ksample(values * 7.5, groups)
        assert scaled.statistic == pytest.approx(base.statistic, rel=1e-12)
        assert scaled.standardized == pytest.approx(base.standardized, rel=1e-12)
        assert scaled.p_value == pytest.approx(base.p_value, rel=1e-12)

    def test_adding_a_constant_leaves_the_result_unchanged(self):
        values, groups = _published_case()
        base = anderson_darling_ksample(values, groups)
        shifted = anderson_darling_ksample(values - 1234.5, groups)
        assert shifted.statistic == pytest.approx(base.statistic, rel=1e-12)
        assert shifted.p_value == pytest.approx(base.p_value, rel=1e-12)

    def test_a_negative_multiplier_leaves_the_result_unchanged(self):
        values, groups = _published_case()
        base = anderson_darling_ksample(values, groups)
        flipped = anderson_darling_ksample(-values, groups)
        assert flipped.statistic == pytest.approx(base.statistic, rel=1e-12)


# =============================================================================
# 零假设下的行为
# =============================================================================
class TestUnderTheNull:
    """Groups drawn from one population must not look different."""

    def test_the_raw_statistic_centres_on_its_published_null_mean(self):
        """The null mean of the statistic is ``k - 1``.

        Measured over 300 random splits of one normal sample, the observed
        mean of the raw statistic is within 5% of k - 1.
        """
        rng = np.random.default_rng(20261024)
        observed = []
        for _ in range(300):
            values = rng.normal(size=120)
            groups = np.repeat([1.0, 2.0], 60)
            observed.append(anderson_darling_ksample(values, groups).statistic)
        mean = float(np.mean(observed))
        assert mean == pytest.approx(1.0, rel=0.05), f"observed null mean {mean}"

    def test_the_test_rejects_about_five_percent_of_the_time(self):
        """A test that is right about the null rejects at its nominal rate.

        The statistic's null distribution is right-skewed, and the p-value
        comes from a spline through five tabulated critical values, so the
        realised rate is allowed to sit a little above the nominal one.  It
        must not sit far above it: 120 splits at the 5% level have a
        standard error of about 2 percentage points, so a ceiling of 15%
        catches a statistic that is badly mis-scaled without failing on
        sampling noise.
        """
        rng = np.random.default_rng(11)
        trials = 120
        rejected = 0
        for _ in range(trials):
            values = rng.normal(size=120)
            groups = np.repeat([1.0, 2.0], 60)
            result = anderson_darling_ksample(values, groups, alpha=0.05)
            rejected += int(result.significant)
        rate = rejected / trials
        assert rate <= 0.15, f"{rejected}/{trials} rejections at the 5% level"

    def test_the_standardised_statistic_is_centred_on_zero(self):
        rng = np.random.default_rng(404)
        standardised = []
        for _ in range(200):
            values = rng.normal(size=120)
            groups = np.repeat([1.0, 2.0], 60)
            standardised.append(anderson_darling_ksample(values, groups).standardized)
        mean = float(np.mean(standardised))
        assert abs(mean) < 0.35, f"observed mean of the standardised statistic {mean}"

    def test_ties_do_not_break_the_calibration(self):
        """Rounded data produce ties, and the tie correction must cope."""
        rng = np.random.default_rng(5)
        trials = 120
        rejected = 0
        for _ in range(trials):
            values = np.round(rng.normal(size=120), 1)
            groups = np.repeat([1.0, 2.0], 60)
            result = anderson_darling_ksample(values, groups, alpha=0.05)
            rejected += int(result.significant)
        rate = rejected / trials
        assert rate <= 0.15, f"{rejected}/{trials} rejections at the 5% level with ties"


# =============================================================================
# 备择假设下的行为
# =============================================================================
class TestUnderTheAlternative:
    """Groups that really do differ must be detected."""

    def test_separating_the_groups_raises_the_statistic_and_lowers_the_p_value(self):
        """Pulling the groups apart makes the evidence stronger.

        The p-value is compared through the standardised statistic rather
        than through the p-value itself: the probability is mapped through
        a logistic function, so it reaches the bottom of the machine's
        range and two different separations then carry the same p-value.
        """
        rng = np.random.default_rng(7)
        previous_standardised = -np.inf
        previous_p = np.inf
        for offset in (0.0, 0.5, 1.0, 2.0, 4.0):
            values = np.concatenate([rng.normal(size=150), rng.normal(loc=offset, size=150)])
            groups = np.repeat([1.0, 2.0], 150)
            result = anderson_darling_ksample(values, groups)
            assert result.standardized > previous_standardised
            assert result.p_value <= previous_p
            previous_standardised = result.standardized
            previous_p = result.p_value

    def test_well_separated_groups_are_significant(self):
        rng = np.random.default_rng(3)
        values = np.concatenate([rng.normal(size=150), rng.normal(loc=4.0, size=150)])
        groups = np.repeat([1.0, 2.0], 150)
        result = anderson_darling_ksample(values, groups, alpha=0.05)
        assert result.significant is True
        assert result.p_value < 1e-6

    def test_a_difference_in_spread_alone_is_detected(self):
        """The point of an omnibus test: a change of scale at matched location."""
        rng = np.random.default_rng(23)
        values = np.concatenate([rng.normal(scale=1.0, size=200), rng.normal(scale=2.5, size=200)])
        groups = np.repeat([1.0, 2.0], 200)
        result = anderson_darling_ksample(values, groups, alpha=0.05)
        assert result.significant is True


# =============================================================================
# 多个组
# =============================================================================
class TestManyGroups:
    def test_three_groups_are_handled(self):
        rng = np.random.default_rng(5)
        values = np.concatenate([rng.normal(loc=offset, size=60) for offset in (0.0, 2.0, 4.0)])
        groups = np.repeat([1.0, 2.0, 3.0], 60)
        result = anderson_darling_ksample(values, groups)
        assert result.n_groups == 3
        assert result.group_sizes == (60, 60, 60)
        assert result.mean == pytest.approx(2.0)
        assert result.significant is True

    def test_string_group_labels_work(self):
        rng = np.random.default_rng(5)
        values = np.concatenate([rng.normal(loc=offset, size=60) for offset in (0.0, 2.0, 4.0)])
        groups = np.repeat(["low", "mid", "high"], 60)
        result = anderson_darling_ksample(values, groups)
        assert result.n_groups == 3
        assert result.group_labels == ("high", "low", "mid")

    def test_the_group_order_does_not_depend_on_the_row_order(self):
        rng = np.random.default_rng(5)
        values = np.concatenate([rng.normal(loc=offset, size=60) for offset in (0.0, 2.0, 4.0)])
        groups = np.repeat([1.0, 2.0, 3.0], 60)
        base = anderson_darling_ksample(values, groups)
        order = rng.permutation(values.size)
        shuffled = anderson_darling_ksample(values[order], groups[order])
        assert shuffled.statistic == pytest.approx(base.statistic, rel=1e-12)
        assert shuffled.p_value == pytest.approx(base.p_value, rel=1e-12)


# =============================================================================
# 边界
# =============================================================================
class TestBoundaries:
    def test_the_two_inputs_must_be_the_same_length(self):
        with pytest.raises(ValidationError):
            anderson_darling_ksample(np.arange(10.0), np.repeat([1.0, 2.0], 4))

    def test_one_group_is_not_a_test(self):
        with pytest.raises(ValidationError):
            anderson_darling_ksample(np.arange(10.0), np.ones(10))

    def test_a_group_that_is_too_small_is_refused(self):
        with pytest.raises(ValidationError):
            anderson_darling_ksample(np.arange(10.0), np.array([1.0] * 9 + [2.0]))

    def test_too_few_observations_are_refused(self):
        with pytest.raises(ValidationError):
            anderson_darling_ksample(np.arange(3.0), np.array([1.0, 2.0, 3.0]))

    def test_a_missing_value_is_refused(self):
        values, groups = _published_case()
        values[3] = np.nan
        with pytest.raises(ValidationError):
            anderson_darling_ksample(values, groups)

    def test_an_alpha_outside_the_unit_interval_is_refused(self):
        values, groups = _published_case()
        with pytest.raises(ValidationError):
            anderson_darling_ksample(values, groups, alpha=0.0)
        with pytest.raises(ValidationError):
            anderson_darling_ksample(values, groups, alpha=1.0)

    def test_a_constant_series_is_refused(self):
        """Every value tied: the statistic is not defined."""
        with pytest.raises(ValidationError):
            anderson_darling_ksample(np.full(20, 3.0), np.repeat([1.0, 2.0], 10))
