# =============================================================================
# FILE: tests/golden/test_coco_advanced.py
# =============================================================================
"""
``stratigraphy.coco_advanced``（COCO/eCOCO 高级模式族）的测试。

测试围绕六条主线组织：

* **已知周期必须被找到。** 合成一条含 **21 kyr 与 19 kyr** 双峰的干净记录，
  自适应、交叉拟合、交错三种模式都必须把这两条轨道报成显著，而**记录里根本
  没有**的 25 kyr 与 13 kyr 必须不显著。判据取自**被测产物本身**
  （``significant_periods_kyr`` 与 ``period_p_values``），不把实现的输出抄成
  期望值。

* **与基础 COCO 的一致性。** 把高级模式自己估出来的目标谱**手动喂给**
  :func:`stratigraphy.coco.coco`，基础实现必须复现同一个相关系数与同一个最优
  速率——两套代码共用同一套频率栅格、重采样与相关系数，这是很强的交叉验证。
  另外，速率扫描给出的最优速率必须落在真值（1.0 Ma/Ma）附近。

* **自适应的收敛。** 背景与目标谱的不动点迭代必须在给定轮数内收敛（相邻两轮
  目标谱的相对变化小于容差），而且自适应目标谱给出的相关系数不低于固定
  Milankovitch 目标谱的情形。

* **交叉拟合的诚实性。** 在**纯噪声**记录上，交叉拟合给出的"显著比例"不应
  明显高于名义水平——这正是这个模式存在的意义。抽样离散是真实存在的，容差取
  名义水平的 3 个二项标准误，并用固定种子让测试可复现。

* **轨道追踪与轨道几何。** 已知双峰（19/21 kyr）上必须追出 **2 条轨道**，中心
  周期与真值的相对误差 < 2%；轨道计数在已知记录长度上必须与真值一致；轨道
  分组与固定权重必须按 Acycle 的规则给出。

* **边界与可复现性。** 点数太少、含 NaN、时间轴非单调必须抛
  ``utils.exceptions.ValidationError``；同一 seed 两次调用必须给出完全相同的
  结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Li, M., Hinnov, L. A., Sun, D., Chen, Y., Bottjer, D. J. & Kump, L. R. (2018).
  Identifying orbital pacing in the astronomical record of the Paleogene and
  Neogene. *Earth and Planetary Science Letters*, 501, 165-179.（COCO / eCOCO
  的统计框架：功率谱与目标谱的相关系数、蒙特卡洛显著性）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的功能划分与高级模式：自适应、双向交叉拟合、奇偶交错、
  轨道链追踪、轨道分组与固定权重）
* Mann, M. E. & Lees, J. M. (1996). Red noise and the search for periodicity in
  Quaternary climate data. *Paleoceanography*, 11, 499-515.（AR(1) 理论谱作为
  红噪声背景；本模块的背景曲线用它）
* Berger, A. (1989). Pleistocene climate from the top of the ice core. In
  *Geochemistry and Geophysics of Climate Change* (eds. Ruddiman, W. & Prell,
  W.). AGU, 13-24.（现代轨道周期表；固定目标谱与默认候选轨道周期表的来源）
* Horne, J. H. & Baliunas, S. (1986). The search for periods in the light curve
  of the suspected solar variability. *The Astrophysical Journal*, 302, 717-735.
  （红噪声零假设下的假警概率）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {
    "coco_adaptive",
    "coco_crossfit",
    "coco_interleaved",
    "coco_orbit_tracking",
    "num_orbits",
    "orbit_groups",
    "fixed_target_weights",
}


import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from stratigraphy.coco import coco
from stratigraphy.coco_advanced import (
    adaptive_coco,
    coco_orbit_tracking,
    crossfit_coco,
    fixed_target_weights,
    interleaved_coco,
    num_orbits,
    orbit_groups,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录覆盖的时间跨度（Ma）——坐标轴本身当作天文时间轴。
SPAN = 3.0

#: 合成记录的采样点数。
#:
#: 1200 点是为了让基础 COCO 的 Berger (1989) 目标谱（含 17 kyr 的最短周期）
#: 在 Nyquist 判据下仍然可用：dt = 3/1199 = 2.5 kyr，Nyquist = 200 cycles/Ma，
#: 高于 2 x 1000/17 = 117.6 cycles/Ma。
N_POINTS = 1200

#: 合成信号里的真实轨道周期（kyr）。
TRUE_PERIODS: tuple[float, ...] = (21.0, 19.0)

#: 候选轨道周期表：两条真的在信号里，两条**故意不在**（用来检验"错误周期不
#: 显著"）。模块内部按周期从长到短排序，所以这里的书写顺序无关。
CANDIDATE_PERIODS = np.array([21.0, 19.0, 25.0, 13.0])

#: 候选表里**不该**显著的周期（kyr）。
ABSENT_PERIODS: tuple[float, ...] = (25.0, 13.0)

#: 噪声记录用的种子（固定，保证"诚实性"这条断言可复现）。
NOISE_SEED = 4242

#: 蒙特卡洛条数。40 条的 p 值分辨率是 1/41 = 0.024，够分辨名义 0.05 水平。
MONTE_CARLO = 40

#: 名义显著性水平。
SIG_LEVEL = 0.05

#: 固定种子。
SEED = 20260101

#: 速率扫描区间（坐标单位/Ma）——真值是 1.0。
RATE_RANGE = (0.98, 1.02)

#: 速率扫描点数。
N_RATES = 5


# =============================================================================
# 辅助函数
# =============================================================================


def _time_axis() -> npt.NDArray:
    """合成记录的时间轴（Ma）。"""
    return np.linspace(0.0, SPAN, N_POINTS)


def _signal_record(
    periods: tuple[float, ...] = TRUE_PERIODS,
    amplitude: float = 1.0,
    noise: float = 0.0,
    seed: int = 11,
) -> tuple[npt.NDArray, npt.NDArray]:
    """合成含已知轨道周期的记录。

    轨道周期 ``p``（kyr）在 Ma 轴上的频率是 ``1000 / p``（cycles/Ma）。

    Returns:
        (时间轴, 数值)。
    """
    axis = _time_axis()
    values = np.zeros(axis.size, dtype=float)
    for period in periods:
        values = values + amplitude * np.sin(2.0 * np.pi * axis / (period / 1000.0))
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        values = values + noise * rng.standard_normal(axis.size)
    return axis, values


def _noise_record(seed: int = NOISE_SEED) -> tuple[npt.NDArray, npt.NDArray]:
    """纯白噪声记录（交叉拟合诚实性检验的零假设数据）。"""
    axis = _time_axis()
    rng = np.random.default_rng(seed)
    return axis, rng.standard_normal(axis.size)


def _period_is(result, period: float) -> bool:
    """结果里是否把这条轨道周期报成显著。"""
    return bool(np.any(np.isclose(result.significant_periods_kyr, period)))


def _period_p_value(result, period: float) -> float:
    """结果里这条轨道周期的 p 值。"""
    index = int(np.argmin(np.abs(np.asarray(result.periods_kyr) - period)))
    return float(result.period_p_values[index])


# =============================================================================
# 自适应 COCO
# =============================================================================


class TestAdaptiveCoco:
    """自适应模式必须把真实的轨道找出来，并且迭代必须收敛。"""

    def test_known_periods_are_significant(self):
        """21 kyr 与 19 kyr 必须被报成显著。

        依据：合成记录里确实有这两条周期，自适应目标谱把这两条频带里的峰高
        当作权重，零假设（重新估目标谱的 AR(1) 替代序列）给不出这么高的峰。
        """
        axis, values = _signal_record()
        result = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        for period in TRUE_PERIODS:
            assert _period_is(result, period), (
                f"{period} kyr carries real signal but is not reported as significant; "
                f"p = {_period_p_value(result, period):.4g}"
            )

    def test_absent_periods_are_not_significant(self):
        """记录里没有的周期必须**不**显著。

        依据：25 kyr 与 13 kyr 在合成记录里没有任何成分，背景以上的峰高只是
        数值泄漏，达不到 AR(1) 零假设的临界值。
        """
        axis, values = _signal_record()
        result = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        for period in ABSENT_PERIODS:
            assert not _period_is(result, period), (
                f"{period} kyr has no signal in the record yet is reported as significant; "
                f"p = {_period_p_value(result, period):.4g}"
            )
            assert _period_p_value(result, period) > SIG_LEVEL, (
                f"the p-value for the absent {period} kyr period is below the nominal level"
            )

    def test_iteration_converges(self):
        """背景/目标谱的不动点迭代必须在给定轮数内收敛。

        依据：自适应模式的定义就是"迭代地重新估计背景与目标谱，直到目标谱不再
        变化"；``target_change`` 是相邻两轮目标谱的相对变化。
        """
        axis, values = _signal_record()
        result = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            max_iterations=5,
            tolerance=1e-3,
            seed=SEED,
        )

        assert result.converged, "the background/target iteration did not converge"
        assert 1 <= result.iterations <= 5, f"the iteration used {result.iterations} rounds"
        assert result.target_change <= 1e-3, (
            f"the target spectrum still moves by {result.target_change:.3g} between the last rounds"
        )

    def test_adaptive_is_at_least_as_good_as_a_fixed_manual_target(self):
        """自适应目标谱给出的相关系数不低于固定目标谱的情形。

        依据：固定目标谱（Berger 1989 现代目标谱）在 19/21 kyr 之外还有一串轨道
        峰，记录里没有能量，相关系数因此被稀释；自适应目标谱只保留记录真正有
        的成分。两种模式用的是同一个比较频带，所以相关系数可以直接比。
        """
        axis, values = _signal_record()
        adaptive = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )
        fixed = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            target="fixed",
            seed=SEED,
        )

        assert adaptive.best_correlation >= fixed.best_correlation, (
            f"the adaptive correlation {adaptive.best_correlation:.4f} is below the fixed-target "
            f"correlation {fixed.best_correlation:.4f}"
        )

    def test_a_manual_target_spectrum_is_accepted(self):
        """调用方自带的目标谱必须可用，且给出的相关系数与基础 COCO 一致。

        依据：``target_spectrum`` 就是"手动指定目标谱"的情形；它与基础 COCO 的
        ``target_spectrum`` 参数同义，比较频带也按同一约定取目标谱的频率范围。
        """
        axis, values = _signal_record()
        freqs = np.linspace(30.0, 80.0, 400)
        manual = (freqs, np.exp(-0.5 * ((freqs - 52.63) / 2.0) ** 2))
        manual_result = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            target_spectrum=manual,
            seed=SEED,
        )
        base = coco(
            axis,
            values,
            rate_range=RATE_RANGE,
            n_rates=N_RATES,
            target_spectrum=manual,
            n_monte_carlo=0,
            random_seed=SEED,
        )

        assert base.best_rate == pytest.approx(manual_result.best_rate, abs=1e-9)
        assert base.best_correlation == pytest.approx(manual_result.best_correlation, rel=1e-9)

    def test_matches_the_base_coco_with_the_same_target(self):
        """把自适应估出来的目标谱手动喂给基础 COCO，必须复现同一个结果。

        依据：两套实现共用同一套频率栅格、重采样与相关系数，所以同一条记录、
        同一个目标谱下，相关系数与最优速率必须一致——这是最强的交叉验证。
        """
        axis, values = _signal_record()
        adaptive = adaptive_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )
        base = coco(
            axis,
            values,
            rate_range=RATE_RANGE,
            n_rates=N_RATES,
            target_spectrum=(adaptive.target_frequencies, adaptive.target_spectrum),
            n_monte_carlo=0,
            random_seed=SEED,
        )

        assert base.best_rate == pytest.approx(adaptive.best_rate, abs=1e-9), (
            f"the base COCO picks {base.best_rate:.6g} where the adaptive mode picks {adaptive.best_rate:.6g}"
        )
        assert base.best_correlation == pytest.approx(adaptive.best_correlation, rel=1e-9), (
            f"the base COCO gives {base.best_correlation:.8f} where the adaptive mode gives "
            f"{adaptive.best_correlation:.8f}"
        )

    def test_best_rate_is_the_true_rate(self):
        """速率扫描必须落在真值（1.0）上。

        依据：合成记录的坐标轴就是天文时间轴（Ma），沉积速率的真值是 1.0
        Ma/Ma；目标谱锚在名义轨道频率上，所以只有真速率能让记录的能量对上目标谱。
        """
        axis, values = _signal_record()
        result = adaptive_coco(
            axis,
            values,
            rate_range=RATE_RANGE,
            n_rates=N_RATES,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )

        assert result.best_rate == pytest.approx(1.0, abs=0.005), (
            f"the most likely rate is {result.best_rate:.6g}, not the true 1.0"
        )

    def test_inadmissible_rates_do_not_shift_the_result(self):
        """被跳过的候选速率不得让结果错位。

        依据：过慢的沉积速率会让重采样后的时间轴欠采样最短的轨道周期，这些
        速率不可用；跳过它们之后，最优速率与显著周期仍然必须落在真值上。
        """
        axis, values = _signal_record()
        result = adaptive_coco(
            axis,
            values,
            rate_range=(0.2, 1.2),
            n_rates=11,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )

        assert np.any(~np.isfinite(result.correlations)), "no rate was skipped, so this proves nothing"
        assert result.best_rate == pytest.approx(1.0, abs=0.005), (
            f"the most likely rate is {result.best_rate:.6g}, not the true 1.0"
        )
        for period in TRUE_PERIODS:
            assert _period_is(result, period), f"{period} kyr was lost when rates were skipped"


# =============================================================================
# 交叉拟合
# =============================================================================


class TestCrossfitCoco:
    """交叉拟合必须既找到信号，又在噪声上保持诚实。"""

    def test_known_periods_are_significant(self):
        """留出折上必须仍然看得到 19/21 kyr。"""
        axis, values = _signal_record()
        result = crossfit_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        for period in TRUE_PERIODS:
            assert _period_is(result, period), (
                f"{period} kyr is missing from the held-out result; p = {_period_p_value(result, period):.4g}"
            )

    def test_best_rate_is_the_true_rate(self):
        """交叉拟合的最优速率同样必须落在真值上。"""
        axis, values = _signal_record()
        result = crossfit_coco(
            axis,
            values,
            rate_range=RATE_RANGE,
            n_rates=N_RATES,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )

        assert result.best_rate == pytest.approx(1.0, abs=0.005), (
            f"the most likely rate is {result.best_rate:.6g}, not the true 1.0"
        )

    def test_the_consensus_is_the_smaller_of_the_two_directions(self):
        """共识相关系数必须等于两个方向里较小的那个。

        依据：双向共识是交叉拟合模式的核心——只在一边成立的巧合不算证据。
        """
        axis, values = _signal_record()
        result = crossfit_coco(
            axis,
            values,
            rate_range=RATE_RANGE,
            n_rates=N_RATES,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )

        expected = np.minimum(result.fold_correlations[0], result.fold_correlations[1])
        assert np.allclose(result.correlations, expected, equal_nan=True), (
            "the consensus correlation is not the minimum of the two directions"
        )

    def test_honesty_on_pure_noise(self):
        """纯噪声上，"显著比例"不得明显高于名义水平。

        依据：交叉拟合存在的意义就是消除"在同一份数据上既估计又检验"的自洽
        偏倚；零假设下留出检验的显著比例应当与名义水平相当。抽样离散是真实
        存在的，容差取名义水平的 3 个二项标准误
        （``se = sqrt(p (1 - p) / n)``）；种子固定，所以这条断言可复现。
        """
        axis, values = _noise_record()
        result = crossfit_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        assert result.n_tests > 0, "the held-out test did not run at all"
        standard_error = float(np.sqrt(SIG_LEVEL * (1.0 - SIG_LEVEL) / result.n_tests))
        tolerance = SIG_LEVEL + 3.0 * standard_error
        assert result.significant_fraction <= tolerance, (
            f"{result.significant_fraction:.3f} of the held-out tests came out significant on pure "
            f"noise, above the nominal {SIG_LEVEL} plus 3 standard errors ({tolerance:.3f})"
        )
        hits = int(np.sum(np.asarray(result.period_p_values) <= SIG_LEVEL))
        binomial = stats.binomtest(hits, result.n_tests, SIG_LEVEL, alternative="greater")
        assert binomial.pvalue > 0.05, (
            f"{hits} of {result.n_tests} held-out tests came out significant on pure noise; "
            f"the one-sided binomial p-value is {binomial.pvalue:.4g}"
        )

    def test_a_pure_noise_record_is_not_claimed_as_orbital(self):
        """纯噪声上不得报出显著轨道周期。"""
        axis, values = _noise_record()
        result = crossfit_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        assert result.significant_periods_kyr.size == 0, (
            f"the noise record produced significant periods {result.significant_periods_kyr}"
        )


# =============================================================================
# 交错模式
# =============================================================================


class TestInterleavedCoco:
    """交错模式必须既找到信号，又在噪声上保持诚实。"""

    def test_known_periods_are_significant(self):
        """奇折训练、偶折检验（反过来也一样）必须仍然看得到 19/21 kyr。"""
        axis, values = _signal_record()
        result = interleaved_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        for period in TRUE_PERIODS:
            assert _period_is(result, period), (
                f"{period} kyr is missing from the interleaved result; p = {_period_p_value(result, period):.4g}"
            )

    def test_best_rate_is_the_true_rate(self):
        """交错模式的最优速率必须落在真值上。"""
        axis, values = _signal_record()
        result = interleaved_coco(
            axis,
            values,
            rate_range=RATE_RANGE,
            n_rates=N_RATES,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            seed=SEED,
        )

        assert result.best_rate == pytest.approx(1.0, abs=0.005), (
            f"the most likely rate is {result.best_rate:.6g}, not the true 1.0"
        )

    def test_honesty_on_pure_noise(self):
        """纯噪声上，交错模式的显著比例不得明显高于名义水平。

        依据与交叉拟合相同：留出检验的零假设是有效的，显著比例应当与名义水平
        相当；容差取 3 个二项标准误，种子固定。
        """
        axis, values = _noise_record()
        result = interleaved_coco(
            axis,
            values,
            periods_kyr=CANDIDATE_PERIODS,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        assert result.n_tests > 0, "the held-out test did not run at all"
        standard_error = float(np.sqrt(SIG_LEVEL * (1.0 - SIG_LEVEL) / result.n_tests))
        tolerance = SIG_LEVEL + 3.0 * standard_error
        assert result.significant_fraction <= tolerance, (
            f"{result.significant_fraction:.3f} of the held-out tests came out significant on pure "
            f"noise, above the nominal {SIG_LEVEL} plus 3 standard errors ({tolerance:.3f})"
        )


# =============================================================================
# 轨道追踪
# =============================================================================


class TestOrbitTracking:
    """轨道追踪必须在已知双峰上追出两条轨道，中心周期要准。"""

    def test_two_orbits_are_tracked(self):
        """19/21 kyr 双峰必须追出 **2 条**轨道链。

        依据：每条轨道周期在每个窗口的谱上都是一个独立的显著峰，追踪规则把
        窗口之间足够接近的峰连成一条链。
        """
        axis, values = _signal_record()
        result = coco_orbit_tracking(
            axis,
            values,
            n_windows=3,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        assert result.n_orbits == 2, f"{result.n_orbits} orbit chain(s) were tracked, expected 2 for the 19/21 kyr pair"

    def test_tracked_periods_match_the_truth(self):
        """两条轨道的中心周期与真值的相对误差必须 < 2%。"""
        axis, values = _signal_record()
        result = coco_orbit_tracking(
            axis,
            values,
            n_windows=3,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        assert result.track_periods_kyr.size == 2, f"expected 2 tracked periods, got {result.track_periods_kyr}"
        for truth in TRUE_PERIODS:
            closest = float(np.min(np.abs(result.track_periods_kyr - truth)))
            relative = closest / truth
            assert relative < 0.02, (
                f"the tracked orbit for {truth} kyr is off by {100.0 * relative:.2f} %, beyond the 2 % tolerance"
            )

    def test_each_window_reports_a_significance_criterion(self):
        """每个窗口必须给出显著性阈值与假警概率。"""
        axis, values = _signal_record()
        result = coco_orbit_tracking(
            axis,
            values,
            n_windows=3,
            n_monte_carlo=MONTE_CARLO,
            sig_level=SIG_LEVEL,
            seed=SEED,
        )

        assert result.window_thresholds.size == 3
        assert np.all(np.isfinite(result.window_thresholds))
        assert np.all(result.window_thresholds > 0.0)
        assert np.all(result.window_p_values >= 0.0)
        assert np.all(result.window_p_values <= 1.0)
        assert result.sig_level == SIG_LEVEL


# =============================================================================
# 轨道几何
# =============================================================================


class TestNumOrbits:
    """轨道计数必须与记录长度给出的真值一致。"""

    def test_count_matches_the_record_length(self):
        """记录只有 100 kyr 长时，九条 Berger 轨道里只有 7 条能被分辨。

        依据：轨道周期必须不超过记录的时间跨度，否则连一个完整周期都没有；
        100 kyr 之上的是 95、41、39、37、23、19、17 kyr 共 7 条。
        """
        periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
        result = num_orbits(periods, 0.1, rate=1.0)

        assert int(result.n_orbits[0]) == 7, (
            f"{int(result.n_orbits[0])} orbits were counted, expected 7 for a 100 kyr record"
        )
        assert np.all(result.resolved[0, periods <= 100.0])
        assert not np.any(result.resolved[0, periods > 100.0])

    def test_count_is_monotone_in_the_record_length(self):
        """记录越长，能分辨的轨道只会越多。"""
        periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
        counts = [int(num_orbits(periods, span, rate=1.0).n_orbits[0]) for span in (0.01, 0.1, 0.5, 5.0)]

        assert counts == sorted(counts), f"the orbit count is not monotone in the record length: {counts}"
        assert counts[0] == 0, "a 10 kyr record cannot resolve any of these orbits"
        assert counts[-1] == 9, "a 5 Ma record resolves all nine orbits"

    def test_nyquist_criterion_removes_undersampled_orbits(self):
        """采样太粗时，高频轨道必须被 Nyquist 判据剔除。

        依据：轨道频率必须低于 Nyquist 频率 ``rate / (2 dx)``。
        """
        periods = np.array([41.0, 19.0])
        coarse = num_orbits(periods, 1.0, rate=1.0, sample_spacing=0.01)
        fine = num_orbits(periods, 1.0, rate=1.0, sample_spacing=0.0005)

        assert int(fine.n_orbits[0]) == 2, "a finely sampled 1 Ma record resolves both orbits"
        assert int(coarse.n_orbits[0]) == 1, (
            f"{int(coarse.n_orbits[0])} orbits survive a 0.01 Ma sampling step, expected 1"
        )
        assert bool(coarse.resolved[0, 0]), "the 41 kyr orbit must survive the coarse sampling"
        assert not bool(coarse.resolved[0, 1]), "the 19 kyr orbit must fail the Nyquist test"


class TestOrbitGroups:
    """轨道分组必须按 Acycle 的规则给出。"""

    def test_nine_periods_are_grouped_by_descending_rank(self):
        """九条周期按排名分成 1 / 4 / 1 / 3。

        依据：Acycle ``cocoOrbitGroups`` 的规则：按周期从长到短排名，第 1 条是
        长偏心率，第 2--5 条是短偏心率，第 6 条是倾角，第 7--9 条是岁差。
        """
        periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
        result = orbit_groups(periods)

        assert result.counts.tolist() == [1, 4, 1, 3]
        assert result.groups[0] == "long eccentricity"
        assert result.groups[1] == "short eccentricity"
        assert result.groups[5] == "obliquity"
        assert result.groups[8] == "precession"

    def test_the_grouping_is_invariant_to_the_input_order(self):
        """输入顺序不得改变分组结果。"""
        periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
        shuffled = np.array([19.0, 405.0, 37.0, 95.0, 23.0, 123.0, 17.0, 39.0, 41.0])

        straight = orbit_groups(periods)
        mixed = orbit_groups(shuffled)

        for index, period in enumerate(shuffled):
            assert straight.groups[int(np.argmin(np.abs(straight.periods_kyr - period)))] == mixed.groups[index]


class TestFixedTargetWeights:
    """固定目标权重必须按物理族给出。"""

    def test_nine_period_weights_follow_the_physical_groups(self):
        """偏心率 1.0、倾角 0.8、岁差 0.6。

        依据：Acycle ``cocoFixedTargetWeights`` 的历史口径（``period2spectrum``
        的约定）。
        """
        periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])
        result = fixed_target_weights(periods)
        groups = orbit_groups(periods)

        for index, group in enumerate(groups.index):
            expected = 1.0 if group <= 2 else (0.8 if group == 3 else 0.6)
            assert result.weights[index] == pytest.approx(expected), (
                f"the weight of {periods[index]} kyr is {result.weights[index]}, expected {expected}"
            )

    def test_threshold_rule_for_other_period_counts(self):
        """不是九条时退回按周期阈值给权重。"""
        result = fixed_target_weights(np.array([405.0, 41.0, 19.0]))

        assert result.weights.tolist() == [1.0, 0.8, 0.6]


# =============================================================================
# 显著性与可复现性
# =============================================================================


class TestSignificanceContract:
    """每个分析都必须把显著性判据一起交出来。"""

    @pytest.mark.parametrize("mode", ["adaptive", "crossfit", "interleaved"])
    def test_the_significance_criterion_is_reported(self, mode: str):
        """结果里必须有名义水平、p 值和"是否显著"的判定。"""
        axis, values = _signal_record()
        if mode == "adaptive":
            result = adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)
        elif mode == "crossfit":
            result = crossfit_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)
        else:
            result = interleaved_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)

        assert result.sig_level == SIG_LEVEL
        assert 0.0 <= result.best_p_value <= 1.0
        assert result.significant == (result.best_p_value <= SIG_LEVEL)
        assert np.all(result.p_values >= 0.0)
        assert np.all(result.p_values <= 1.0)

    def test_the_geometry_helpers_declare_that_they_do_not_test(self):
        """几何类的三个函数不做假设检验，必须把这一点说清楚。"""
        periods = np.array([405.0, 123.0, 95.0, 41.0, 39.0, 37.0, 23.0, 19.0, 17.0])

        for result in (num_orbits(periods, 0.1), orbit_groups(periods), fixed_target_weights(periods)):
            assert result.sig_level == SIG_LEVEL
            assert result.critical_value is None
            assert result.significant is False


class TestReproducibility:
    """同一 seed 必须给出完全相同的结果。"""

    def test_adaptive_is_reproducible(self):
        axis, values = _signal_record()
        one = adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)
        two = adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)

        assert np.allclose(one.correlations, two.correlations)
        assert np.allclose(one.p_values, two.p_values)
        assert np.allclose(one.period_p_values, two.period_p_values)
        assert np.allclose(one.target_spectrum, two.target_spectrum)
        assert one.best_rate == two.best_rate

    def test_crossfit_is_reproducible(self):
        axis, values = _signal_record()
        one = crossfit_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)
        two = crossfit_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)

        assert np.allclose(one.correlations, two.correlations)
        assert np.allclose(one.period_p_values, two.period_p_values)
        assert np.allclose(one.fold_correlations, two.fold_correlations)

    def test_interleaved_is_reproducible(self):
        axis, values = _signal_record()
        one = interleaved_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)
        two = interleaved_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, n_monte_carlo=MONTE_CARLO, seed=SEED)

        assert np.allclose(one.correlations, two.correlations)
        assert np.allclose(one.period_p_values, two.period_p_values)

    def test_tracking_is_reproducible(self):
        axis, values = _signal_record()
        one = coco_orbit_tracking(axis, values, n_windows=3, n_monte_carlo=MONTE_CARLO, seed=SEED)
        two = coco_orbit_tracking(axis, values, n_windows=3, n_monte_carlo=MONTE_CARLO, seed=SEED)

        assert one.n_orbits == two.n_orbits
        assert np.allclose(one.track_periods_kyr, two.track_periods_kyr)
        assert np.allclose(one.window_thresholds, two.window_thresholds)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_too_few_points_are_rejected(self):
        axis = np.linspace(0.0, 0.1, 8)
        values = np.sin(2.0 * np.pi * axis / 0.021)
        with pytest.raises(ValidationError):
            adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS)

    def test_non_finite_values_are_rejected(self):
        axis, values = _signal_record()
        broken = values.copy()
        broken[10] = np.nan
        with pytest.raises(ValidationError):
            crossfit_coco(axis, broken, periods_kyr=CANDIDATE_PERIODS)

    def test_non_increasing_time_axis_is_rejected(self):
        axis, values = _signal_record()
        broken = axis.copy()
        broken[10] = broken[11]
        with pytest.raises(ValidationError):
            interleaved_coco(broken, values, periods_kyr=CANDIDATE_PERIODS)

    def test_a_constant_series_is_rejected(self):
        axis = _time_axis()
        with pytest.raises(ValidationError):
            adaptive_coco(axis, np.ones_like(axis), periods_kyr=CANDIDATE_PERIODS)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_an_invalid_significance_level_is_rejected(self, level: float):
        axis, values = _signal_record()
        with pytest.raises(ValidationError):
            adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, sig_level=level)

    @pytest.mark.parametrize("red", [1.0, -1.0, 2.0])
    def test_an_invalid_red_noise_parameter_is_rejected(self, red: float):
        axis, values = _signal_record()
        with pytest.raises(ValidationError):
            crossfit_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, red=red)

    def test_invalid_orbital_periods_are_rejected(self):
        axis, values = _signal_record()
        with pytest.raises(ValidationError):
            adaptive_coco(axis, values, periods_kyr=np.array([21.0, -1.0]))

    def test_an_invalid_iteration_control_is_rejected(self):
        axis, values = _signal_record()
        with pytest.raises(ValidationError):
            adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, max_iterations=0)
        with pytest.raises(ValidationError):
            adaptive_coco(axis, values, periods_kyr=CANDIDATE_PERIODS, tolerance=0.0)

    def test_too_few_windows_are_rejected(self):
        axis, values = _signal_record()
        with pytest.raises(ValidationError):
            coco_orbit_tracking(axis, values, n_windows=1)

    def test_invalid_orbit_count_inputs_are_rejected(self):
        periods = np.array([405.0, 19.0])
        with pytest.raises(ValidationError):
            num_orbits(periods, 0.0)
        with pytest.raises(ValidationError):
            num_orbits(periods, -1.0)
        with pytest.raises(ValidationError):
            num_orbits(np.array([0.0]), 1.0)

    def test_invalid_group_inputs_are_rejected(self):
        with pytest.raises(ValidationError):
            orbit_groups(np.array([np.nan, 19.0]))
        with pytest.raises(ValidationError):
            fixed_target_weights(np.array([-19.0]))
