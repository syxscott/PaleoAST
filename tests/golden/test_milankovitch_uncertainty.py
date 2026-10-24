# =============================================================================
# FILE: tests/golden/test_milankovitch_uncertainty.py
# =============================================================================
"""
``stratigraphy.milankovitch_uncertainty``（米兰科维奇频率的不确定度与区间）的测试。

测试围绕六条主线组织，每条断言的依据都写在对应测试的 docstring 里：

* **轨道周期的量级。** 在 0 Ma（现代）处，三个轨道族的周期必须落在各自的
  **量级带**上：气候岁差 19–23 kyr、倾角 38–44 kyr、偏心率 85–115 kyr。
  这些带是文献里的经典值（见下），不是把实现跑出来的数字抄成期望值——
  实现给出的中心值若跑偏 10%，断言就会失败。
* **周期的年代依赖。** 偏心率周期在 1 Ma 与 0 Ma 必须**不同**（相对差异
  > 1%）。这是可验证的物理事实：La04 解的偏心率由 g4/g5 两条谱线拍频而成，
  其 "~100 kyr 周期" 的间距随拍频相位迁移（Lisiecki & Raymo 2005 观测到的
  正是 80–130 kyr 之间的谱带迁移）。因此"偏心率周期是常数"这种实现会被
  抓住。
* **匹配功能。** 给一个**故意加了 2% 偏差**的岁差周期，必须被匹配到岁差
  轨道（而不是倾角或偏心率轨道，也不是匹配失败）。这是本模块的用途本身。
* **区间的自洽性。** 频率窗口的上下界必须**包含**该轨道的理论频率；窗口
  宽度必须随不确定度设定**单调**变化（不确定度调大 → 区间变宽）；区间的
  窗口（若干年龄的并集）不窄于单点的窗口。
* **可复现性。** 同参数两次调用必须给出逐点相同的结果（含带种子的后验
  抽样）。
* **边界。** 非法年龄、空输入、NaN 等必须抛
  ``utils.exceptions.ValidationError``。

另外单独验证 **Waltham 正演链**：0 Ma 处的日长、地月距离与平岁差率必须
复现**已发表的现在值**（24.0 h、384 400 km、50.287961″/yr），并且日长在
过去必须更短、岁差率的不确定度必须随年龄增长。

参考来源（断言中引用的数值出处）
--------------------------------
* Waltham, T., 2015. Milankovitch period uncertainties and their impact on
  cyclostratigraphy. *Journal of Sedimentary Research*, 85(4): 688-704.
  doi:10.2110/jsr.2015.66.（拖曳因子模型；正演链：拖曳因子 → 地月距离 →
  黄赤交角 → 日长 → 岁差率）
* Waltham, T., 2015. Solving the orbital equations of the Milankovitch cycles.
  *Earth and Planetary Science Letters*, 405: 499-509.（贝叶斯线性回归口径：
  把观测周期匹配到理论频率上，给出频率窗口）
* Laskar, J., Gastineau, M., Jounder, E. & Bourgeois, F., 2004. Solution for
  the eccentricity and obliquity of the Earth's orbits over 5 Myr.
  *Astronomy and Astrophysics*, 413: 755-767.（La04 轨道解；平岁差率常数
  50.287961″/yr；本仓库 ``data/astronomy/la04.npz`` 即由该解转换而来）
* Lisiecki, L. E. & Raymo, M. E., 2005. Four million years of climate
  oscillations: phase relationships of Milankovitch cycles and eccentricity
  cycles. *Earth and Planetary Science Letters*, 237: 369-398.（~100 kyr 谱带
  随时间的迁移）
* Berger, A. & Loutre, M. F., 1992. Astronomical solutions for paleoclimate
  studies over the last 3 million years. *Earth and Planetary Science Letters*,
  111: 369-382.（气候岁差周期在 19–23 kyr 之间游走；倾角 ~41 kyr；
  偏心率 ~100 kyr）
* Jeffreys, H., 1962. The Earth: Its Origin and History. Pergamon.
  （岁差率的四极矩力矩表达）
* Walker, J. C. W., 1977. Tidal friction and the evolution of the Moon's
  orbit. *Journal of Geophysics Research*, 82: 1407-1425.（潮汐拖曳因子）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"milankovitch_uncertainty"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.astronomy import milankovitch_periods
from stratigraphy.milankovitch_uncertainty import (
    MilankovitchUncertaintyAnalyzer,
    bayesian_frequency_solution,
    match_period_to_orbit,
    milankovitch_interval_frequencies,
    milankovitch_intervals,
    waltham_orbital_solution,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 现代 (0 Ma) 处气候岁差周期的量级带 (kyr)。
#:
#: 依据: 经典岁差族横跨 ~19 kyr (短分支)、~21 kyr (中分支)、~23 kyr (长
#: 分支) —— Berger & Loutre (1992) 的气候岁差分支结构, 也是层序学里
#: "岁差周期 19–23 kyr" 的标准说法。仓库自己的周期表
#: (:func:`stratigraphy.astronomy.milankovitch_periods`) 同样把这三个分支
#: 列为 ``precession_19/21/23``。
PRECESSION_BAND_KYR = (19.0, 23.0)

#: 现代 (0 Ma) 处倾角周期的量级带 (kyr)。
#:
#: 依据: 倾角主谐波 ~41 kyr (Berger & Loutre 1992; 仓库周期表的
#: ``obliquity_41``)。这里放宽到 38–44 kyr: 模块给出的不是常数, 而是
#: **由天文解数周期得到的局部周期**, 其散布有几个百分点。
OBLIQUITY_BAND_KYR = (38.0, 44.0)

#: 现代 (0 Ma) 处偏心率短周期 (~100 kyr) 的量级带 (kyr)。
#:
#: 依据: 偏心率主谐波 ~100 kyr。带宽取 85–115 kyr: 偏心率序列是 g4/g5
#: 双线拍频形成的宽带, 其局部周期本身就带着 ~10% 的不确定度
#: (Lisiecki & Raymo 2005 记录了 80–130 kyr 的谱带迁移)。
ECCENTRICITY_BAND_KYR = (85.0, 115.0)

#: 现代 (0 Ma) 处的平岁差率 (度/kyr)。
#:
#: 依据: Laskar et al. (2004) 的平岁差 (黄经) 速率 50.287961″/yr,
#: 换算成度/kyr 是 13.9694 °/kyr。
PRECESSION_RATE_DEG_PER_KYR = 13.9694

#: 现代 (0 Ma) 处的日长 (小时, 太阳日)。
DAY_LENGTH_HOURS = 24.0

#: 现代 (0 Ma) 处的地月距离 (km)。
EARTH_MOON_DISTANCE_KM = 384_400.0

#: 匹配用例: 真实 ~21 kyr 的岁差周期, 人为加上 2% 的偏差。
OBSERVED_PRECESSION_KYR = 21.0 * 1.02

#: 匹配用例: 人为偏差的比例。
PRECESSION_ERROR = 0.02


# =============================================================================
# 辅助函数
# =============================================================================


def _entry(entries, name: str):
    """从结果元组里取出某个谐波。"""
    matches = [item for item in entries if item.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} entry, got {len(matches)}"
    return matches[0]


def _period(entries, name: str) -> float:
    """取出某个谐波的理论周期 (kyr)。"""
    return _entry(entries, name).period_kyr


# =============================================================================
# 轨道周期的量级
# =============================================================================


class TestOrbitalPeriodMagnitudes:
    """0 Ma 处三个轨道族的周期必须落在各自的量级带上。"""

    def test_precession_period_is_in_the_twenties_of_kyr(self):
        """气候岁差周期必须在 19–23 kyr。

        依据: 经典岁差族的三个分支 (19 / 21 / 23 kyr)。模块给的是经典带
        的中心值, 因此落在带内。
        """
        entries = milankovitch_interval_frequencies(0.0)
        period = _period(entries, "precession_21")

        assert PRECESSION_BAND_KYR[0] <= period <= PRECESSION_BAND_KYR[1], (
            f"the precession period at 0 Ma is {period:.3f} kyr, outside the {PRECESSION_BAND_KYR} kyr band"
        )

    def test_obliquity_period_is_about_forty_one_kyr(self):
        """倾角周期必须在 38–44 kyr。

        依据: 倾角主谐波 ~41 kyr。模块的值由 La04 解的倾角序列**数周期**
        得到, 因此允许几个百分点的散布。
        """
        entries = milankovitch_interval_frequencies(0.0)
        period = _period(entries, "obliquity_41")

        assert OBLIQUITY_BAND_KYR[0] <= period <= OBLIQUITY_BAND_KYR[1], (
            f"the obliquity period at 0 Ma is {period:.3f} kyr, outside the {OBLIQUITY_BAND_KYR} kyr band"
        )

    def test_eccentricity_period_is_about_one_hundred_kyr(self):
        """偏心率短周期必须在 85–115 kyr。

        依据: 偏心率主谐波 ~100 kyr; 带宽反映 g4/g5 拍频带来的局部起伏。
        """
        entries = milankovitch_interval_frequencies(0.0)
        period = _period(entries, "eccentricity_100")

        assert ECCENTRICITY_BAND_KYR[0] <= period <= ECCENTRICITY_BAND_KYR[1], (
            f"the eccentricity period at 0 Ma is {period:.3f} kyr, outside the {ECCENTRICITY_BAND_KYR} kyr band"
        )

    def test_the_reference_periods_come_from_the_shared_period_table(self):
        """每个谐波的经典周期必须等于天文模块周期表里的那一个。

        依据: 模块的口径是"复用 ``stratigraphy.astronomy`` 的周期表", 而不是
        自立一套数值; 这一点必须能被断言读出来。
        """
        table = milankovitch_periods()
        entries = milankovitch_interval_frequencies(0.0)

        assert _entry(entries, "precession_21").reference_period_kyr == table["precession_21"]
        assert _entry(entries, "obliquity_41").reference_period_kyr == table["obliquity_41"]
        assert _entry(entries, "eccentricity_100").reference_period_kyr == table["eccentricity_100"]

    def test_frequency_and_period_agree(self):
        """频率与周期必须互为倒数 (口径: 频率单位 1/kyr)。"""
        entries = milankovitch_interval_frequencies(0.0)

        for entry in entries:
            assert entry.frequency == pytest.approx(1.0 / entry.period_kyr, rel=1e-12)
            assert entry.period_min_kyr == pytest.approx(1.0 / entry.frequency_max, rel=1e-12)
            assert entry.period_max_kyr == pytest.approx(1.0 / entry.frequency_min, rel=1e-12)


# =============================================================================
# 周期的年代依赖
# =============================================================================


class TestAgeDependence:
    """轨道解有长期变化: 同一轨道的周期在不同年龄必须不同。"""

    def test_the_eccentricity_period_differs_between_zero_and_one_ma(self):
        """偏心率周期在 1 Ma 与 0 Ma 必须相差 > 1%。

        依据: La04 的偏心率是 g4 (~95 kyr) 与 g5 (~100 kyr) 两条谱线拍频
        而成, 拍频相位随时间移动, 因此 "~100 kyr 周期" 的局部间距在百万年
        尺度上就会变化 (Lisiecki & Raymo 2005)。把偏心率周期实现成常数
        (例如直接抄周期表的 100 kyr) 会在这里被抓住。
        """
        present = _period(milankovitch_interval_frequencies(0.0), "eccentricity_100")
        past = _period(milankovitch_interval_frequencies(1.0), "eccentricity_100")

        relative = abs(past - present) / present
        assert relative > 0.01, (
            f"the eccentricity period barely changed between 0 and 1 Ma: "
            f"{present:.3f} kyr vs {past:.3f} kyr (relative {relative:.4f})"
        )

    def test_both_eccentricity_periods_stay_in_the_eccentricity_band(self):
        """0 Ma 与 1 Ma 的偏心率周期都必须仍在偏心率量级带内。

        依据: 同上 —— 变化是局部的, 不该把周期推到别的族里去。
        """
        for age in (0.0, 1.0):
            period = _period(milankovitch_interval_frequencies(age), "eccentricity_100")
            assert ECCENTRICITY_BAND_KYR[0] <= period <= ECCENTRICITY_BAND_KYR[1], (
                f"the eccentricity period at {age} Ma is {period:.3f} kyr, outside the {ECCENTRICITY_BAND_KYR} kyr band"
            )

    def test_the_ages_are_reported_back(self):
        """结果必须带着自己的年龄, 否则无法把多年龄的结果分开。"""
        entries = milankovitch_interval_frequencies(np.array([0.0, 1.0]))
        ages = sorted({entry.age_ma for entry in entries})

        assert ages == [0.0, 1.0]
        assert len(entries) == 6


# =============================================================================
# 匹配
# =============================================================================


class TestMatching:
    """实测周期必须被匹配到正确的轨道。"""

    def test_a_two_percent_error_still_matches_the_precession(self):
        """加了 2% 偏差的 ~21 kyr 周期必须匹配到岁差轨道。

        依据: 岁差族的经典带是 19–23 kyr, 21 kyr 的中心值加 2% 之后仍然
        落在带内; 而 21.42 kyr 离倾角 (~41 kyr) 与偏心率 (~100 kyr) 都
        足够远, 因此正确匹配只有一种。
        """
        match = match_period_to_orbit(OBSERVED_PRECESSION_KYR, 0.0)

        assert match.orbit_name == "precession_21", (
            f"the {OBSERVED_PRECESSION_KYR:.3f} kyr cycle matched to {match.orbit_name!r}"
        )
        assert match.in_window
        assert match.deviation == pytest.approx(PRECESSION_ERROR, abs=0.005)
        assert match.deviation_kyr == pytest.approx(OBSERVED_PRECESSION_KYR - match.theoretical_period_kyr, rel=1e-12)

    def test_the_precession_match_is_unambiguous(self):
        """匹配岁差周期时, 别的轨道不该同时容纳它。"""
        match = match_period_to_orbit(OBSERVED_PRECESSION_KYR, 0.0)

        assert match.alternatives == ()
        assert match.family == "precession"

    @pytest.mark.parametrize(
        ("observed", "expected"),
        [(41.0, "obliquity_41"), (100.0, "eccentricity_100"), (21.0, "precession_21")],
    )
    def test_the_classic_periods_match_their_own_orbits(self, observed: float, expected: str):
        """三个族的经典周期必须各自匹配到自己的轨道。"""
        match = match_period_to_orbit(observed, 0.0)

        assert match.orbit_name == expected
        assert match.in_window

    def test_a_period_between_the_orbits_is_reported_as_outside_the_window(self):
        """落在所有轨道之外的周期必须被标成"不在窗口内", 而不是硬凑一个匹配。"""
        match = match_period_to_orbit(55.0, 0.0)

        assert not match.in_window
        assert match.orbit_name in {"precession_21", "obliquity_41", "eccentricity_100"}


# =============================================================================
# 区间的自洽性
# =============================================================================


class TestIntervalSelfConsistency:
    """频率窗口必须自洽, 并且随不确定度设定单调变化。"""

    def test_the_window_contains_the_theoretical_frequency(self):
        """频率上下界必须包含理论频率。"""
        entries = milankovitch_interval_frequencies(0.0)

        for entry in entries:
            assert entry.frequency_min <= entry.frequency <= entry.frequency_max, (
                f"the frequency window of {entry.name!r} "
                f"[{entry.frequency_min:.8f}, {entry.frequency_max:.8f}] does not contain "
                f"the theoretical frequency {entry.frequency:.8f}"
            )

    def test_the_window_contains_the_theoretical_period(self):
        """周期上下界同样必须包含理论周期。"""
        entries = milankovitch_interval_frequencies(0.0)

        for entry in entries:
            assert entry.period_min_kyr <= entry.period_kyr <= entry.period_max_kyr, (
                f"the period window of {entry.name!r} "
                f"[{entry.period_min_kyr:.3f}, {entry.period_max_kyr:.3f}] kyr does not contain "
                f"the theoretical period {entry.period_kyr:.3f} kyr"
            )

    def test_a_larger_uncertainty_widens_the_window(self):
        """不确定度设定调大 → 窗口必须单调变宽。"""
        narrow = milankovitch_interval_frequencies(0.0, uncertainty=1.0)
        medium = milankovitch_interval_frequencies(0.0, uncertainty=2.0)
        wide = milankovitch_interval_frequencies(0.0, uncertainty=4.0)

        for name in ("precession_21", "obliquity_41", "eccentricity_100"):
            widths = [
                _entry(block, name).frequency_max - _entry(block, name).frequency_min
                for block in (narrow, medium, wide)
            ]
            assert widths[0] < widths[1] < widths[2], f"the window of {name!r} does not widen monotonically: {widths}"

    def test_an_interval_covers_at_least_the_point_window(self):
        """区间的窗口 (若干年龄的并集) 不得窄于单点的窗口。"""
        point = milankovitch_interval_frequencies(0.0)
        interval = milankovitch_intervals(0.0, interval_kyr=400.0)

        for entry in point:
            spanned = interval.by_name(entry.name)
            assert spanned.frequency_min <= entry.frequency_min + 1e-12
            assert spanned.frequency_max >= entry.frequency_max - 1e-12

    def test_the_interval_reports_its_own_span(self):
        """区间必须报出自己的起止年龄。"""
        interval = milankovitch_intervals(0.0, age_end_ma=0.5)

        assert interval.age_start_ma == pytest.approx(0.0)
        assert interval.age_end_ma == pytest.approx(0.5)
        assert len(interval.entries) == 3


# =============================================================================
# 可复现性
# =============================================================================


class TestReproducibility:
    """同参数两次调用必须给出完全相同的结果。"""

    def test_the_frequencies_are_reproducible(self):
        """两次调用必须逐点相同。"""
        first = milankovitch_interval_frequencies(0.0)
        second = milankovitch_interval_frequencies(0.0)

        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert np.isclose(left.frequency, right.frequency, rtol=0.0, atol=0.0)
            assert np.isclose(left.frequency_min, right.frequency_min, rtol=0.0, atol=0.0)
            assert np.isclose(left.frequency_max, right.frequency_max, rtol=0.0, atol=0.0)

    def test_the_waltham_solution_is_reproducible(self):
        """正演链两次调用必须相同。"""
        first = waltham_orbital_solution(1.0)
        second = waltham_orbital_solution(1.0)

        assert first.precession_rate_deg_per_kyr == second.precession_rate_deg_per_kyr
        assert first.day_length_hours == second.day_length_hours
        assert first.earth_moon_distance_km == second.earth_moon_distance_km

    def test_the_posterior_samples_are_reproducible_with_the_same_seed(self):
        """带种子的后验抽样两次必须相同。"""
        periods = np.array([21.42, 41.0, 100.0])
        theory = np.array([1.0 / 21.0, 1.0 / 41.0, 1.0 / 100.0])

        first = bayesian_frequency_solution(periods, theory, seed=11, n_samples=512)
        second = bayesian_frequency_solution(periods, theory, seed=11, n_samples=512)

        assert first.samples is not None
        assert np.allclose(first.samples, second.samples)
        assert first.frequency_offset == pytest.approx(second.frequency_offset, rel=1e-12)


# =============================================================================
# Waltham 正演链
# =============================================================================


class TestWalthamSolution:
    """正演链必须复现已发表的现在值, 并保持正确的物理方向。"""

    def test_the_present_day_values_are_reproduced(self):
        """0 Ma 处: 日长 24 h、地月距离 384 400 km、岁差率 50.29″/yr。

        依据: Laskar et al. (2004) 的平岁差率 50.287961″/yr 与现在的日地
        月几何。允许 1% 的容差: 正演链里的拖曳因子与经验拟合本身就是
        近似。
        """
        solution = waltham_orbital_solution(0.0)

        assert solution.day_length_hours == pytest.approx(DAY_LENGTH_HOURS, rel=0.01)
        assert solution.earth_moon_distance_km == pytest.approx(EARTH_MOON_DISTANCE_KM, rel=0.01)
        assert solution.precession_rate_deg_per_kyr == pytest.approx(PRECESSION_RATE_DEG_PER_KYR, rel=0.01)

    def test_the_days_were_shorter_in_the_past(self):
        """过去的日长必须比现在短 (月球轨道角动量更小 → 自转更快)。

        依据: 角动量守恒 ``I_E ω cos ε + m √(G M_E a) = const``; 100 Ma 时
        地月距离只有现在的大约 99%, 日长必须明显更短。
        """
        present = waltham_orbital_solution(0.0)
        past = waltham_orbital_solution(100.0)

        assert past.day_length_max_hours < present.day_length_hours
        assert past.earth_moon_distance_max_km < present.earth_moon_distance_km

    def test_the_uncertainty_grows_towards_the_past(self):
        """岁差率的相对不确定度必须随年龄增长。

        依据: 潮汐耗散历史的拖曳因子在越老的年代越不确定 (Waltham 2015),
        而地月距离按 ``a^6.5`` 传播, 于是岁差率的相对不确定度随年龄放大。
        """
        young = waltham_orbital_solution(1.0)
        old = waltham_orbital_solution(1000.0)

        assert old.precession_rate_relative_sigma > young.precession_rate_relative_sigma
        assert young.precession_rate_relative_sigma >= 0.0


# =============================================================================
# 贝叶斯线性回归
# =============================================================================


class TestBayesianFrequencySolution:
    """共模频率平移的贝叶斯线性回归。"""

    def test_the_offset_recovers_a_known_shift(self):
        """把三条周期整体平移一个已知量, 回归必须把这个平移量找回来。

        依据: 模型是 ``f_obs = f_theory + Δ`` (Δ 是**频率**上的共同平移量,
        单位 1/kyr)。造一组整体平移了 2e-4 1/kyr 的数据, 后验均值必须回到
        2e-4, 且残差为零。
        """
        theory = np.array([1.0 / 21.0, 1.0 / 41.0, 1.0 / 100.0])
        shift = 2.0e-4
        observed = 1.0 / (theory + shift)

        regression = bayesian_frequency_solution(observed, theory, seed=3)

        assert regression.frequency_offset == pytest.approx(shift, rel=0.05)
        assert regression.frequency_offset_sigma > 0.0
        assert np.allclose(regression.residuals, 0.0, atol=1e-9)

    def test_the_residuals_are_reported_per_observation(self):
        """残差必须与观测一一对应。"""
        periods = np.array([21.42, 41.0, 100.0])
        theory = np.array([1.0 / 21.0, 1.0 / 41.0, 1.0 / 100.0])

        regression = bayesian_frequency_solution(periods, theory)

        assert regression.residuals.shape == periods.shape
        assert regression.weights.shape == periods.shape
        assert np.all(regression.weights > 0.0)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    @pytest.mark.parametrize("age", [-1.0, -0.001, np.nan, np.inf, -np.inf])
    def test_invalid_ages_are_rejected(self, age: float):
        """负数、NaN、无穷大的年龄必须被拒绝。"""
        with pytest.raises(ValidationError):
            milankovitch_interval_frequencies(age)

    def test_an_age_outside_the_solution_range_is_rejected(self):
        """超出天文解覆盖范围的年龄必须被拒绝。"""
        with pytest.raises(ValidationError):
            milankovitch_interval_frequencies(10_000.0)

    def test_an_empty_age_array_is_rejected(self):
        """空输入必须被拒绝。"""
        with pytest.raises(ValidationError):
            milankovitch_interval_frequencies(np.array([]))

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_an_invalid_uncertainty_is_rejected(self, scale: float):
        """不确定度缩放因子必须为正。"""
        with pytest.raises(ValidationError):
            milankovitch_interval_frequencies(0.0, uncertainty=scale)

    @pytest.mark.parametrize("window", [0.0, -100.0])
    def test_an_invalid_window_is_rejected(self, window: float):
        """分析窗口半宽必须为正。"""
        with pytest.raises(ValidationError):
            milankovitch_interval_frequencies(0.0, half_window_kyr=window)

    @pytest.mark.parametrize("observed", [0.0, -21.0, np.nan, np.inf])
    def test_an_invalid_observed_period_is_rejected(self, observed: float):
        """实测周期必须是正的有限数。"""
        with pytest.raises(ValidationError):
            match_period_to_orbit(observed, 0.0)

    def test_an_invalid_tolerance_is_rejected(self):
        """容差必须非负。"""
        with pytest.raises(ValidationError):
            match_period_to_orbit(21.0, 0.0, tolerance=-0.1)

    def test_an_unknown_harmonic_name_is_rejected(self):
        """区间里查一个不存在的谐波名必须报错。"""
        interval = milankovitch_intervals(0.0, interval_kyr=100.0)
        with pytest.raises(ValidationError):
            interval.by_name("no_such_harmonic")

    def test_an_interval_that_ends_before_it_starts_is_rejected(self):
        """起止年龄倒置必须报错。"""
        with pytest.raises(ValidationError):
            milankovitch_intervals(1.0, age_end_ma=0.0)

    def test_mismatched_regression_inputs_are_rejected(self):
        """回归的观测与理论频率长度不一致必须报错。"""
        with pytest.raises(ValidationError):
            bayesian_frequency_solution(np.array([21.0, 41.0]), np.array([1.0 / 21.0]))

    def test_empty_regression_inputs_are_rejected(self):
        """回归的空输入必须报错。"""
        with pytest.raises(ValidationError):
            bayesian_frequency_solution(np.array([]), np.array([]))

    def test_a_negative_waltham_age_is_rejected(self):
        """正演链的负年龄必须报错。"""
        with pytest.raises(ValidationError):
            waltham_orbital_solution(-1.0)

    def test_invalid_drag_factor_bounds_are_rejected(self):
        """拖曳因子区间必须是一对正数且递增。"""
        with pytest.raises(ValidationError):
            waltham_orbital_solution(0.0, drag_factor_present=(22.0, 20.0))


# =============================================================================
# 界面门面
# =============================================================================


class TestFacade:
    """门面必须给出与函数一致的结果, 并把非法输入挡在外面。"""

    def test_the_facade_matches_the_function(self):
        """门面返回的频率必须与直接调用函数逐条相等。

        依据: 门面只是把表单能表达的参数 (年龄轴 + 两个标量) 转交给
        :func:`milankovitch_interval_frequencies`, 它不得引入任何自己的
        口径差异 —— 否则"从界面跑出来的"与"从脚本跑出来的"会给出不同的答案。
        """
        ages = np.array([0.0, 0.5, 1.0])

        result = MilankovitchUncertaintyAnalyzer().analyze(ages)
        expected = milankovitch_interval_frequencies(ages)

        assert len(result.entries) == len(expected)
        for actual, reference in zip(result.entries, expected, strict=True):
            assert actual.age_ma == pytest.approx(reference.age_ma)
            assert actual.name == reference.name
            assert actual.frequency == pytest.approx(reference.frequency, rel=1e-12)
            assert actual.frequency_min == pytest.approx(reference.frequency_min, rel=1e-12)
            assert actual.frequency_max == pytest.approx(reference.frequency_max, rel=1e-12)

    def test_the_facade_returns_one_group_per_age(self):
        """每个年龄都必须得到自己的一组谐波结果。"""
        ages = np.array([0.0, 0.5, 1.0])

        result = MilankovitchUncertaintyAnalyzer().analyze(ages)

        assert len(result.entries) == 3 * ages.size
        for age in ages:
            group = result.for_age(float(age))
            assert len(group) == 3
            assert {entry.age_ma for entry in group} == {float(age)}

    @pytest.mark.parametrize(
        "ages",
        [np.array([0.0, np.nan]), np.array([0.0, 10_000.0]), np.array([-1.0]), np.array([])],
    )
    def test_the_facade_rejects_invalid_ages(self, ages: npt.NDArray):
        """含 NaN、超出解的范围、为负或空的年龄都必须抛 ``ValidationError``。"""
        with pytest.raises(ValidationError):
            MilankovitchUncertaintyAnalyzer().analyze(ages)

    def test_the_facade_result_is_reportable(self):
        """结果必须能生成可读摘要 (走 i18n)。"""
        result = MilankovitchUncertaintyAnalyzer().analyze(np.array([0.0, 1.0]))

        assert isinstance(result.summary(), str)
        assert result.solution_name


# =============================================================================
# 数据类型契约
# =============================================================================


class TestResultContract:
    """结果容器必须按约定暴露字段。"""

    def test_the_entries_expose_snake_case_fields(self):
        """字段名必须是 snake_case, 且带齐年龄/频率/窗口。"""
        entries = milankovitch_interval_frequencies(0.0)
        entry = entries[0]

        for attribute in ("name", "family", "order", "age_ma", "frequency", "frequency_min", "frequency_max", "sigma"):
            assert hasattr(entry, attribute)

    def test_the_ages_may_be_an_array(self):
        """年龄可以是数组, 返回值按年龄优先排列。"""
        ages = np.array([0.0, 0.5, 1.0])
        entries = milankovitch_interval_frequencies(ages)

        assert len(entries) == 3 * ages.size
        assert [entry.age_ma for entry in entries[:3]] == [0.0, 0.0, 0.0]
        assert [entry.age_ma for entry in entries[-3:]] == [1.0, 1.0, 1.0]

    def test_the_match_reports_the_orbital_family(self):
        """匹配结果必须报出轨道族与序号。"""
        match = match_period_to_orbit(OBSERVED_PRECESSION_KYR, 0.0)

        assert match.family == "precession"
        assert match.order == 1
        assert match.age_ma == pytest.approx(0.0)

    def test_the_summary_strings_are_produced(self):
        """可读摘要必须能生成 (走 i18n)。"""
        entries = milankovitch_interval_frequencies(0.0)
        interval = milankovitch_intervals(0.0, interval_kyr=100.0)

        assert isinstance(entries[0].summary(), str)
        assert isinstance(interval.summary(), str)
        assert isinstance(waltham_orbital_solution(0.0).summary(), str)
        assert isinstance(match_period_to_orbit(21.0, 0.0).summary(), str)
        assert isinstance(
            bayesian_frequency_solution(np.array([21.0]), np.array([1.0 / 21.0])).summary(),
            str,
        )
