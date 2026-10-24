# =============================================================================
# FILE: tests/golden/test_coco_variants.py
# =============================================================================
"""
``stratigraphy.coco_variants``（COCO 的 Spearman 变体、置信区间、eCOCO 逐行
汇总与峰值筛选）的测试。

本文件围绕四条主线组织，每条断言的依据写在对应测试的 docstring 里。

* **Spearman 秩相关的定义性质。** 秩相关是"两条序列的秩"的 Pearson 相关，
  因此：

  - 在**完全单调**（线性）的关系上，秩相关必须**精确**等于 ``1.0``；
  - 在"整体单调但不是直线"的关系上（``y = x**3``），秩相关仍然**精确**等于
    ``1.0``，而 Pearson 只有 0.917——这正是 Pearson 做不到、秩相关能做到的
    那一半（"秩相关的定义性质"）；
  - 在**非单调**（V 形）的关系上，秩相关必须回到 0 附近，否则"秩相关恒为 1"
    这条错误实现会被放过；
  - 把 ``y`` 换成 ``y`` 的**严格单调变换**（``exp``、三次方），秩相关与它的
    p 值必须**一字不变**——秩没变，所以统计量与零分布都不变。

* **相关系数的置信区间。** Fisher z 变换区间（Fiore & Bonferroni, 2009）在
  bivariate normal 下的覆盖率必须接近名义水平：重复抽样 600 次，95% 区间的
  覆盖率必须落在 90%–99% 之间；区间必须包含点估计；样本更多时区间必须更窄；
  同一组数据上，自助法区间（Efron & Tibshirani, 1993）与 Fisher z 区间的
  相对宽度差必须小于 20%。

* **零假设下的显著性。** 把两条序列的**配对标签打乱**后，秩置换零分布
  （Fisher, 1915）的 p 值不应当系统性地小：100 次重复里 ``p < 0.05`` 的比例
  必须小于 15%（实测约 4%–5%）。同样的判据在 COCO 层面用一条**没有轨道信号**
  的记录检验：固定候选速率处的 p 值不应系统性地小。

* **eCOCO 网格与峰值筛选。** 逐行汇总的行数必须等于网格的行数，每行的计数
  必须等于该行速率图里的非 NaN 格数（Acycle ``eccount.m`` 的逐行语义）；筛
  出的峰数不超过请求的 ``n_peaks``，且"CI 小于 5%"的判据必须真的起作用——
  把阈值调严，筛出的峰数不增。

边界测试断言抛 ``utils.exceptions.ValidationError``；可复现性断言固定种子给出
完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Fisher, R. A. (1915). Frequency in the occurrence of events.
  *Journal of the Royal Statistical Society*, 78(2), 322-329.（秩相关的显著性：
  条件于秩结构的排列零分布；"单调关系给出秩相关 1"的定义性质）
* Fiore, A. R. & Bonferroni, R. (2009). A comparison of some confidence
  intervals for the correlation coefficient in the bivariate normal case.
  *Metrika*, 59(2), 139-168.（相关系数的 Fisher z 变换区间及其覆盖率）
* Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.（百分位自助法区间）
* Li, M., Hinnov, L. A., Sun, D., Chen, Y., Bottjer, D. J. & Kump, L. R. (2018).
  Identifying orbital pacing in the astronomical record of the Paleogene and
  Neogene. *Earth and Planetary Science Letters*, 501, 165-179.（COCO / eCOCO
  的统计框架）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "COCO"、"eCOCO" 与 "corrcoef" 子族）
* Acycle 的 ``code/corrcoef/cyclecorr_rank.m``（Spearman 秩相关系数与 p 值）、
  ``cyclesig_rank.m``（随机谱蒙特卡洛零分布）、``eccount.m``（沉积速率图的
  **逐行**汇总）、``eccpeaks.m``（"CI 小于 5%"的前 n 个峰筛选）——本文件断言的
  语义与这四个文件一致，差别见 ``stratigraphy/coco_variants.py`` 的 docstring。
"""

from __future__ import annotations

#: 本文件覆盖的分析（矩阵以此声明为准）
ANALYSES_COVERED = {"coco_spearman", "correlation_confidence_interval", "ecoco_row_summary", "significant_peaks"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.coco_variants import (
    ECocoGrid,
    coco_spearman,
    correlation_confidence_interval,
    ecoco_row_summary,
    significant_peaks,
    spearman_rank_correlation,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成轨道记录覆盖的天文时间跨度（Ma）。
ASTRO_SPAN_MA = 12.0

#: 合成轨道记录的采样点数。
ASTRO_POINTS = 4000

#: 合成记录里写入的轨道周期（kyr）——Berger (1989) 的九条现代轨道周期。
ORBITAL_PERIODS_KYR = (405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0)

#: 合成记录真实的沉积速率（坐标单位/Ma）。
TRUE_RATE = 2.0

#: 无轨道信号的对照记录：采样点数。
FEATURELESS_POINTS = 800

#: 无轨道信号的对照记录：坐标跨度。
FEATURELESS_SPAN = 20.0

#: 对照记录上用于检验零假设的固定候选速率。
FEATURELESS_RATE = 8.0

#: 零假设检验的重复次数。
NULL_REPEATS = 100

#: 零假设检验允许的假阳性比例上限（名义水平是 5%）。
NULL_FALSE_POSITIVE_LIMIT = 0.15

#: 置信区间覆盖率检验的重复次数。
COVERAGE_REPEATS = 600

#: 覆盖率检验允许的下界。
COVERAGE_LOWER = 0.90

#: 覆盖率检验允许的上界。
COVERAGE_UPPER = 0.99

#: 两种区间宽度允许的相对差。
WIDTH_TOLERANCE = 0.20


# =============================================================================
# 辅助函数
# =============================================================================


def _orbital_record() -> tuple[npt.NDArray, npt.NDArray]:
    """含九条轨道周期信号的合成记录。

    信号写在**天文时间轴**上，再乘以真实沉积速率变成坐标轴——这样"正确的沉积
    速率"就是 :data:`TRUE_RATE`，而功率谱与目标谱在真实速率处对齐得最好。

    Returns:
        (坐标轴, 测量序列)。
    """
    astro = np.linspace(0.0, ASTRO_SPAN_MA, ASTRO_POINTS)
    signal = np.zeros_like(astro)
    for period in ORBITAL_PERIODS_KYR:
        signal += np.sin(2.0 * np.pi * astro / (period / 1000.0))
    noise = 0.3 * np.random.default_rng(4).standard_normal(ASTRO_POINTS)
    return TRUE_RATE * astro, signal + noise


def _featureless_record(seed: int) -> tuple[npt.NDArray, npt.NDArray]:
    """没有轨道信号的对照记录（纯高斯白噪声）。

    它的功率谱没有任何带状结构，因此与目标谱的秩相关应当只反映抽样涨落。
    """
    time = np.linspace(0.0, FEATURELESS_SPAN, FEATURELESS_POINTS)
    values = np.random.default_rng(seed).standard_normal(FEATURELESS_POINTS)
    return time, values


def _bivariate_normal(rho: float, n: int, seed: int) -> tuple[npt.NDArray, npt.NDArray]:
    """相关系数为 ``rho`` 的二元正态样本。"""
    rng = np.random.default_rng(seed)
    first = rng.standard_normal(n)
    second = rho * first + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    return first, second


def _manual_grid() -> ECocoGrid:
    """手工构造的 eCOCO 网格（行 = 窗口，列 = 候选速率）。

    手工构造而不是调用 :func:`ecoco_row_summary`，是为了让"CI 小于 5%"的筛选
    条件有确切的、可读出来的取值，从而能逐条核对筛选逻辑。
    """
    return ECocoGrid(
        window_centres=np.array([1.0, 2.0, 3.0]),
        rates=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        # 速率图：非 NaN 格是"该窗口在该速率上显著"的位置。
        rate_map=np.array(
            [
                [np.nan, np.nan, 3.0, np.nan, 5.0],
                [np.nan, 2.0, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 4.0, np.nan],
            ]
        ),
        correlations=np.array(
            [
                [0.10, 0.20, 0.50, 0.40, 0.60],
                [0.10, 0.55, 0.20, 0.30, 0.40],
                [0.10, 0.20, 0.30, 0.45, 0.40],
            ]
        ),
        p_values=np.array(
            [
                [0.20, 0.10, 0.01, 0.03, 0.02],
                [0.20, 0.04, 0.10, 0.30, 0.40],
                [0.20, 0.10, 0.30, 0.02, 0.40],
            ]
        ),
        ci_lower=np.full((3, 5), 0.0),
        ci_upper=np.full((3, 5), 0.02),
        n_orbits=np.full((3, 5), 4),
        band_sizes=np.full((3, 5), 60.0),
    )


# =============================================================================
# Spearman 秩相关的定义性质
# =============================================================================


class TestSpearmanRankProperties:
    """秩相关的定义性质：单调性、不变性、以及"非单调时必须回到 0"。"""

    def test_perfectly_monotone_relation_gives_exactly_one(self):
        """完全单调（线性）的关系：秩相关必须**精确**等于 1.0。

        依据：秩相关是两条序列的**秩**的 Pearson 相关；两条序列的秩完全一致
        时，秩向量彼此成比例，Pearson 相关就是 1。这是定义，不是近似。
        """
        x = np.linspace(-1.0, 1.0, 60)
        result = spearman_rank_correlation(x, 3.0 * x + 1.0, n_permutations=99, seed=1)

        assert result.coefficient == 1.0, (
            f"the rank correlation of a perfectly monotone relation is {result.coefficient!r}, not 1.0"
        )
        # Pearson 在同一关系上也给出 1，两者在"直线"这一情形上不矛盾。
        assert float(np.corrcoef(x, 3.0 * x + 1.0)[0, 1]) == pytest.approx(1.0, abs=1e-12)

    def test_monotone_but_nonlinear_relation_gives_one_where_pearson_cannot(self):
        """整体单调但不是直线的关系：秩相关仍是 1.0，Pearson 做不到。

        ``y = x**3`` 在 ``[-1, 1]`` 上严格单调，秩相关因此是 1.0；而 Pearson
        只看线性，给出 0.917。这个对比就是"秩相关"与"Pearson"的分界线。
        """
        x = np.linspace(-1.0, 1.0, 60)
        result = spearman_rank_correlation(x, x**3, n_permutations=99, seed=1)
        pearson = float(np.corrcoef(x, x**3)[0, 1])

        assert result.coefficient == 1.0, (
            f"the rank correlation of a monotone relation is {result.coefficient!r}, not 1.0"
        )
        assert pearson < 1.0, (
            "the Pearson correlation of a monotone relation should stay below 1; "
            "the test relation is no longer monotone"
        )

    def test_non_monotone_relation_does_not_give_one(self):
        """非单调（V 形）的关系：秩相关必须回到 0 附近。

        没有这一条，"秩相关恒为 1"的错误实现也能通过前两条断言。
        """
        x = np.linspace(-1.0, 1.0, 60)
        result = spearman_rank_correlation(x, x**2, n_permutations=99, seed=1)

        assert abs(result.coefficient) < 0.1, (
            f"a non-monotone relation must not give a rank correlation of {result.coefficient!r}"
        )

    def test_rank_correlation_is_invariant_under_a_monotone_transform(self):
        """把 ``y`` 换成它的严格单调变换，秩相关必须**一字不变**。

        依据：秩只依赖次序，严格单调变换不改变次序，所以秩向量——以及由它算
        出来的相关系数——逐位相同。
        """
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200) + 0.4 * x

        plain = spearman_rank_correlation(x, y, n_permutations=199, seed=3)
        transformed = spearman_rank_correlation(x, np.exp(y), n_permutations=199, seed=3)

        assert transformed.coefficient == plain.coefficient, (
            "the rank correlation changed under the monotone transform exp(): "
            f"{plain.coefficient!r} -> {transformed.coefficient!r}"
        )

    def test_p_value_is_invariant_under_a_monotone_transform(self):
        """单调变换后 p 值也必须不变。

        p 值由秩向量的排列零分布决定，秩没变，零分布与统计量都没变。
        """
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200) + 0.4 * x

        plain = spearman_rank_correlation(x, y, n_permutations=199, seed=3)
        transformed = spearman_rank_correlation(x, y**3, n_permutations=199, seed=3)

        assert transformed.p_value == plain.p_value, (
            f"the p-value changed under a monotone transform: {plain.p_value!r} -> {transformed.p_value!r}"
        )

    def test_shuffled_labels_do_not_give_significant_p_values(self):
        """把配对标签打乱后，p 值不应当系统性地小。

        依据：Fisher (1915) 的条件秩分布。零假设"两条序列的秩是随机配对的"
        下，p 值应当接近均匀分布，因此 ``p < 0.05`` 的比例应当接近 5%。
        """
        rng = np.random.default_rng(101)
        hits = 0
        for repeat in range(NULL_REPEATS):
            x = rng.standard_normal(40)
            y = rng.permutation(rng.standard_normal(40))
            result = spearman_rank_correlation(x, y, n_permutations=199, seed=5)
            hits += int(result.p_value < 0.05)

        fraction = hits / NULL_REPEATS
        assert fraction < NULL_FALSE_POSITIVE_LIMIT, (
            f"{fraction:.1%} of the shuffled pairs came out with p < 0.05, "
            f"which is above the {NULL_FALSE_POSITIVE_LIMIT:.0%} limit"
        )


# =============================================================================
# 相关系数的置信区间
# =============================================================================


class TestCorrelationConfidenceInterval:
    """置信区间必须覆盖真值、随样本量收窄，并且两种方法彼此接近。"""

    def test_the_interval_contains_the_point_estimate(self):
        """区间必须包含点估计。"""
        first, second = _bivariate_normal(0.5, 60, seed=11)
        estimate = float(np.corrcoef(first, second)[0, 1])
        interval = correlation_confidence_interval(estimate, 60)

        assert interval.lower <= estimate <= interval.upper, (
            f"the interval [{interval.lower}, {interval.upper}] does not contain {estimate}"
        )
        assert interval.method == "fisher-z"

    def test_the_coverage_is_near_the_nominal_level(self):
        """Fisher z 区间的覆盖率必须接近名义水平。

        依据：Fiore & Bonferroni (2009) 对 bivariate normal 下相关系数区间的
        比较——Fisher z 变换区间的覆盖率应当接近名义水平。这里用 600 次重复
        抽样，要求覆盖率落在 90%–99% 之间。
        """
        rho_true = 0.5
        n = 60
        rng = np.random.default_rng(11)
        covered = 0
        for draw in range(COVERAGE_REPEATS):
            first = rng.standard_normal(n)
            second = rho_true * first + np.sqrt(1.0 - rho_true**2) * rng.standard_normal(n)
            estimate = float(np.corrcoef(first, second)[0, 1])
            interval = correlation_confidence_interval(estimate, n)
            covered += int(interval.lower <= rho_true <= interval.upper)

        coverage = covered / COVERAGE_REPEATS
        assert COVERAGE_LOWER <= coverage <= COVERAGE_UPPER, (
            f"the 95% interval covered the truth in {coverage:.1%} of the draws, "
            f"outside [{COVERAGE_LOWER:.0%}, {COVERAGE_UPPER:.0%}]"
        )

    def test_the_interval_narrows_with_more_samples(self):
        """样本更多时区间必须更窄。"""
        small = correlation_confidence_interval(0.5, 30)
        large = correlation_confidence_interval(0.5, 300)

        assert large.width < small.width, (
            f"the interval did not narrow with more samples: {small.width} -> {large.width}"
        )

    def test_the_bootstrap_interval_agrees_with_the_fisher_z_interval(self):
        """同一组数据上，自助法区间与 Fisher z 区间的宽度必须接近。

        依据：两种方法针对同一个量（相关系数），样本量相同时标准误的量级相同
        （秩相关系数的抽样方差比同值的 Pearson 相关系数略小，所以自助法区间略
        宽），因此相对宽度差应当很小。
        """
        first, second = _bivariate_normal(0.5, 60, seed=23)
        estimate = float(spearman_rank_correlation(first, second, n_permutations=99, seed=1).coefficient)

        fisher = correlation_confidence_interval(estimate, 60, method="fisher-z")
        bootstrap = correlation_confidence_interval(
            estimate, 60, method="bootstrap", x=first, y=second, seed=5, n_bootstrap=800
        )

        relative = abs(fisher.width - bootstrap.width) / fisher.width
        assert relative < WIDTH_TOLERANCE, (
            f"the interval widths differ by {relative:.1%}: fisher-z {fisher.width} vs bootstrap {bootstrap.width}"
        )
        assert bootstrap.method == "bootstrap"

    def test_the_bootstrap_interval_is_reproducible(self):
        """同一 seed 的自助法必须给出同一个区间。"""
        first, second = _bivariate_normal(0.5, 60, seed=29)
        one = correlation_confidence_interval(0.5, 60, method="bootstrap", x=first, y=second, seed=17, n_bootstrap=200)
        two = correlation_confidence_interval(0.5, 60, method="bootstrap", x=first, y=second, seed=17, n_bootstrap=200)

        assert one.lower == two.lower
        assert one.upper == two.upper


# =============================================================================
# COCO 的 Spearman 变体
# =============================================================================


class TestCocoSpearman:
    """COCO 的秩相关变体必须找到真实的沉积速率，并给出诚实的显著性。"""

    def test_the_best_rate_matches_the_true_sedimentation_rate(self):
        """最优沉积速率必须落在真值附近。

        依据：Li et al. (2018) 的 COCO 框架——记录功率谱与目标谱在公共频带上
        的相关系数在**正确**的沉积速率处取极大值。换成 Spearman 之后结论不变。
        """
        coords, values = _orbital_record()
        result = coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=25, n_monte_carlo=99, random_seed=2)

        assert abs(result.best_rate - TRUE_RATE) < 0.25, (
            f"the best rate is {result.best_rate:.4g}, far from the true {TRUE_RATE}"
        )
        assert result.best_correlation > 0.3, (
            f"the rank correlation at the best rate is only {result.best_correlation:.4f}"
        )

    def test_the_rank_correlation_and_its_p_value_are_reported(self):
        """结果里必须给出秩相关系数与它的 p 值。"""
        coords, values = _orbital_record()
        result = coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=12, n_monte_carlo=99, random_seed=2)

        assert result.correlations.shape == result.rates.shape
        assert result.p_values.shape == result.rates.shape
        assert np.all(np.isfinite(result.correlations))
        assert np.all(result.correlations >= -1.0)
        assert np.all(result.correlations <= 1.0)
        finite = np.isfinite(result.p_values)
        assert np.all(result.p_values[finite] > 0.0)
        assert np.all(result.p_values[finite] <= 1.0)
        assert result.best_p_value == pytest.approx(float(result.p_values[int(np.nanargmax(result.correlations))]))

    def test_the_null_hypothesis_p_values_are_not_systematically_small(self):
        """没有轨道信号的记录上，固定速率处的 p 值不应当系统性地小。

        依据：零假设"记录与目标谱的秩配对是随机的"。读的是**固定**候选速率
        （不是最优速率）处的 p 值——最优速率是按相关系数挑出来的，直接读它的
        p 值会把"择优"也算进去，多重比较会把假阳性率抬到 5% 以上。
        """
        hits = 0
        for repeat in range(NULL_REPEATS):
            time, values = _featureless_record(1000 + repeat)
            result = coco_spearman(
                time,
                values,
                rate_range=(FEATURELESS_RATE, FEATURELESS_RATE + 1e-6),
                n_rates=2,
                n_monte_carlo=99,
                random_seed=7,
            )
            hits += int(result.p_values[0] < 0.05)

        fraction = hits / NULL_REPEATS
        assert fraction < NULL_FALSE_POSITIVE_LIMIT, (
            f"{fraction:.1%} of the featureless records came out with p < 0.05, "
            f"which is above the {NULL_FALSE_POSITIVE_LIMIT:.0%} limit"
        )

    def test_the_monte_carlo_null_is_also_available(self):
        """AR(1) 随机谱零分布（Acycle ``corrsig.m`` 的语义）必须同时给出。"""
        coords, values = _orbital_record()
        result = coco_spearman(
            coords,
            values,
            rate_range=(1.5, 5.0),
            n_rates=8,
            n_monte_carlo=49,
            random_seed=2,
            null_model="both",
        )

        assert result.p_values_permutation is not None
        assert result.p_values_monte_carlo is not None
        assert result.p_values_permutation.shape == result.p_values_monte_carlo.shape
        # 默认的 p_values 是秩置换零分布给出的。
        assert np.allclose(result.p_values, result.p_values_permutation, equal_nan=True)

    def test_the_results_are_reproducible(self):
        """同一 seed 必须给出完全相同的结果。"""
        coords, values = _orbital_record()
        one = coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=10, n_monte_carlo=49, random_seed=9)
        two = coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=10, n_monte_carlo=49, random_seed=9)

        assert np.allclose(one.correlations, two.correlations)
        assert np.allclose(one.p_values, two.p_values)
        assert one.best_rate == two.best_rate

    def test_the_target_period_subset_is_honoured(self):
        """``n_target`` 必须真的换掉目标谱（Acycle ``orbit7`` 的语义）。

        少用两条轨道周期，比较频带的上界随之下移，目标谱的形状也改变——
        于是相关系数必然与用全部周期时不同。
        """
        coords, values = _orbital_record()
        full = coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=8, n_monte_carlo=49, random_seed=2)
        subset = coco_spearman(
            coords,
            values,
            rate_range=(1.5, 5.0),
            n_rates=8,
            n_monte_carlo=49,
            random_seed=2,
            n_target=7,
        )

        assert subset.target_frequencies.size > 0
        assert not np.allclose(full.correlations, subset.correlations, equal_nan=True)


# =============================================================================
# eCOCO 网格的逐行汇总
# =============================================================================


class TestEcocoRowSummary:
    """逐行汇总必须与网格逐格对得上。"""

    @staticmethod
    def _summary():
        coords, values = _orbital_record()
        return ecoco_row_summary(
            coords,
            values,
            rate_range=(1.5, 5.0),
            n_windows=4,
            n_rates=12,
            n_monte_carlo=49,
            random_seed=3,
        )

    def test_the_summary_has_one_row_per_grid_row(self):
        """汇总的行数必须等于 eCOCO 网格的行数。"""
        summary = self._summary()
        grid = summary.grid

        assert summary.rate_counts.size == grid.shape[0]
        assert summary.best_rates.size == grid.shape[0]
        assert summary.window_centres.size == grid.shape[0]

    def test_each_row_count_equals_the_observations_in_that_row(self):
        """每行的计数必须等于该行速率图里的非 NaN 格数。

        依据：Acycle ``eccount.m`` 的 ``ecsrn_n``——"该行非 NaN 值的个数"。
        """
        summary = self._summary()
        grid = summary.grid

        for row in range(grid.shape[0]):
            observed = int(np.count_nonzero(np.isfinite(grid.rate_map[row, :])))
            assert summary.rate_counts[row] == observed, (
                f"row {row} reports {summary.rate_counts[row]} observations but the map holds {observed}"
            )
        assert int(np.sum(summary.rate_counts)) == int(np.count_nonzero(np.isfinite(grid.rate_map)))

    def test_the_row_mean_and_median_come_from_the_rate_map(self):
        """逐行的速率均值与中位数必须由该行的速率图算出。"""
        summary = self._summary()
        grid = summary.grid

        for row in range(grid.shape[0]):
            present = grid.rate_map[row, :]
            present = present[np.isfinite(present)]
            if present.size == 0:
                assert np.isnan(summary.rate_mean[row])
                continue
            assert summary.rate_mean[row] == pytest.approx(float(np.mean(present)))
            assert summary.rate_median[row] == pytest.approx(float(np.median(present)))

    def test_the_best_rate_is_the_maximum_of_the_row(self):
        """每行的最优速率必须是该行相关系数最大处的速率。"""
        summary = self._summary()
        grid = summary.grid

        for row in range(grid.shape[0]):
            correlations = grid.correlations[row, :]
            if not np.any(np.isfinite(correlations)):
                continue
            column = int(np.nanargmax(correlations))
            assert summary.best_rates[row] == pytest.approx(float(grid.rates[column]))
            assert summary.best_correlations[row] == pytest.approx(float(correlations[column]))

    def test_the_confidence_interval_is_stored_per_cell(self):
        """每个格子的置信区间必须有限且夹住该格的相关系数。"""
        summary = self._summary()
        grid = summary.grid

        finite = np.isfinite(grid.correlations) & np.isfinite(grid.ci_lower) & np.isfinite(grid.ci_upper)
        assert bool(np.any(finite))
        assert np.all(grid.ci_lower[finite] <= grid.ci_upper[finite])
        assert np.all(grid.ci_lower[finite] <= grid.correlations[finite])
        assert np.all(grid.correlations[finite] <= grid.ci_upper[finite])


# =============================================================================
# 峰值筛选
# =============================================================================


class TestSignificantPeaks:
    """筛选必须守约：峰数不超限，判据必须真的起作用。"""

    def test_no_more_peaks_are_returned_than_requested(self):
        """每行筛出的峰数不超过请求的 ``n_peaks``。"""
        grid = _manual_grid()
        peaks = significant_peaks(grid, n_peaks=2, ci_limit=0.05)

        assert peaks.indices.shape == (3, 2)
        assert np.all(peaks.counts <= 2)
        assert int(np.sum(peaks.counts)) <= 2 * 3

    def test_the_peaks_are_the_largest_rates_that_pass_the_criteria(self):
        """选出的必须是通过判据的最大速率，且按速率从大到小排。"""
        grid = _manual_grid()
        peaks = significant_peaks(grid, n_peaks=3, ci_limit=0.05)

        # 第 0 行的速率图里有两个显著格（速率 5.0 与 3.0），都通过判据。
        assert peaks.counts[0] == 2
        assert list(peaks.rates[0][:2]) == [5.0, 3.0]
        # 第 1、2 行各只有一个显著格。
        assert peaks.counts[1] == 1
        assert peaks.counts[2] == 1

    def test_a_tighter_limit_never_returns_more_peaks(self):
        """把"CI 小于 5%"的阈值调严，筛出的峰数不增。

        这是"判据真的生效"的判据：如果峰数不随阈值变化，说明阈值根本没被用到。
        """
        grid = _manual_grid()
        loose = significant_peaks(grid, n_peaks=3, ci_limit=0.05)
        tight = significant_peaks(grid, n_peaks=3, ci_limit=0.005)

        assert int(np.sum(tight.counts)) <= int(np.sum(loose.counts))
        assert int(np.sum(tight.counts)) == 0

    def test_the_correlation_threshold_is_honoured(self):
        """相关系数低于阈值的格必须被筛掉。

        网格里有四个显著格，相关系数分别是 0.60、0.55、0.50、0.45；阈值抬到
        0.55 之后只剩下前两个。
        """
        grid = _manual_grid()
        peaks = significant_peaks(grid, n_peaks=3, ci_limit=0.05, corr_threshold=0.55)

        assert int(np.sum(peaks.counts)) == 2
        assert np.all(peaks.correlations[peaks.indices >= 0] >= 0.55)

    def test_the_orbit_threshold_is_honoured(self):
        """轨道计数不足的格必须被筛掉。"""
        grid = _manual_grid()
        peaks = significant_peaks(grid, n_peaks=3, ci_limit=0.05, min_orbits=9)

        assert int(np.sum(peaks.counts)) == 0

    def test_the_ci_width_criterion_filters_too(self):
        """另一种读法（置信区间宽度小于 5%）也必须能筛。

        Acycle ``eccpeaks.m`` 的 "ci < 5%" 有两种可能读法；本实现两种都提供，
        因此两种都必须真的起作用。
        """
        grid = _manual_grid()
        loose = significant_peaks(grid, n_peaks=3, ci_limit=0.05, criterion="ci_width")
        tight = significant_peaks(grid, n_peaks=3, ci_limit=0.01, criterion="ci_width")

        # 四个显著格的置信区间宽度都是 0.02，因此 5% 的阈值全部放行。
        assert int(np.sum(loose.counts)) == 4
        assert int(np.sum(tight.counts)) == 0

    def test_the_column_direction_is_available(self):
        """按列筛选（Acycle ``eccpeaks.m`` 的 ``dim=1``）也必须可用。"""
        grid = _manual_grid()
        peaks = significant_peaks(grid, n_peaks=2, ci_limit=0.05, dim="column")

        assert peaks.indices.shape == (5, 2)
        assert int(np.sum(peaks.counts)) == int(np.sum(np.isfinite(grid.rate_map)))


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_too_few_points_are_rejected(self):
        time = np.linspace(0.0, 1.0, 8)
        with pytest.raises(ValidationError):
            coco_spearman(time, np.sin(time), rate_range=(1.0, 2.0), n_rates=4, random_seed=1)

    def test_non_finite_values_are_rejected(self):
        coords, values = _orbital_record()
        broken = values.copy()
        broken[10] = np.nan
        with pytest.raises(ValidationError):
            coco_spearman(coords, broken, rate_range=(1.5, 5.0), n_rates=4, random_seed=1)

    def test_a_non_monotonic_time_axis_is_rejected(self):
        coords, values = _orbital_record()
        broken = coords.copy()
        broken[10] = broken[11]
        with pytest.raises(ValidationError):
            coco_spearman(broken, values, rate_range=(1.5, 5.0), n_rates=4, random_seed=1)

    @pytest.mark.parametrize(
        "rate_range",
        [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.5), (1.0, 1.0)],
    )
    def test_an_invalid_rate_range_is_rejected(self, rate_range: tuple[float, float]):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=rate_range, n_rates=4, random_seed=1)

    def test_an_inadmissible_rate_range_is_rejected(self):
        """所有候选速率都不可用时必须报错，而不是返回空结果。"""
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1e-6, 1e-5), n_rates=4, random_seed=1)

    def test_a_negative_monte_carlo_count_is_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=4, n_monte_carlo=-1, random_seed=1)

    def test_an_unknown_null_model_is_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=4, null_model="bootstrap", random_seed=1)

    def test_conflicting_seed_aliases_are_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=4, random_seed=1, seed=2)

    def test_conflicting_target_count_aliases_are_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=4, n_target=7, n_period=9, random_seed=1)

    def test_an_out_of_range_target_count_is_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=4, n_target=99, random_seed=1)

    def test_an_invalid_significance_level_is_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            coco_spearman(coords, values, rate_range=(1.5, 5.0), n_rates=4, sig_level=1.5, random_seed=1)

    def test_too_few_windows_are_rejected(self):
        coords, values = _orbital_record()
        with pytest.raises(ValidationError):
            ecoco_row_summary(coords, values, rate_range=(1.5, 5.0), n_windows=1, n_rates=4, random_seed=1)

    def test_an_out_of_range_correlation_is_rejected(self):
        with pytest.raises(ValidationError):
            correlation_confidence_interval(1.5, 60)

    def test_too_few_samples_for_an_interval_are_rejected(self):
        with pytest.raises(ValidationError):
            correlation_confidence_interval(0.5, 3)

    def test_an_invalid_confidence_level_is_rejected(self):
        with pytest.raises(ValidationError):
            correlation_confidence_interval(0.5, 60, level=1.5)

    def test_an_unknown_interval_method_is_rejected(self):
        with pytest.raises(ValidationError):
            correlation_confidence_interval(0.5, 60, method="analytic")

    def test_the_bootstrap_interval_needs_the_paired_data(self):
        with pytest.raises(ValidationError):
            correlation_confidence_interval(0.5, 60, method="bootstrap")

    def test_the_bootstrap_interval_rejects_degenerate_data(self):
        with pytest.raises(ValidationError):
            correlation_confidence_interval(0.5, 60, method="bootstrap", x=np.ones(60), y=np.arange(60.0))

    def test_the_rank_test_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationError):
            spearman_rank_correlation(np.arange(10.0), np.arange(9.0))

    def test_the_rank_test_rejects_a_constant_series(self):
        with pytest.raises(ValidationError):
            spearman_rank_correlation(np.arange(10.0), np.ones(10))

    def test_the_peak_filter_rejects_a_non_grid(self):
        with pytest.raises(ValidationError):
            significant_peaks(np.zeros((3, 3)), n_peaks=1)

    def test_the_peak_filter_rejects_an_out_of_range_limit(self):
        grid = _manual_grid()
        with pytest.raises(ValidationError):
            significant_peaks(grid, n_peaks=1, ci_limit=1.5)

    def test_the_peak_filter_rejects_a_zero_peak_count(self):
        grid = _manual_grid()
        with pytest.raises(ValidationError):
            significant_peaks(grid, n_peaks=0)

    def test_the_peak_filter_rejects_an_unknown_criterion(self):
        grid = _manual_grid()
        with pytest.raises(ValidationError):
            significant_peaks(grid, n_peaks=1, criterion="width")
