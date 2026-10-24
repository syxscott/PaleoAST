# =============================================================================
# FILE: tests/golden/test_robust_lomb.py
# =============================================================================
"""
``stratigraphy.robust_lomb`` 的测试：稳健 AR(1) 背景的 Lomb-Scargle 周期图。

期望值来自**定义本身**，而不是"跑一遍把输出冻结下来"：

* AR(1) 背景的形状是 ``(1-rho²)/(1-2 rho cos(2 pi f dt)+rho²)``。对一条已知
  系数 ``rho`` 的 AR(1) 模拟序列，稳健拟合得到的系数必须接近真值（抽样误差由
  ``1/sqrt(N)`` 控制，判据取 ±0.15）。
* 背景是拟合到**中值平滑谱**上的，因此它的水平低于观测谱的平均水平：低频端的
  背景谱必须低于记录谱——这正是"稳健"二字的含义（周期信号不会把红噪声水平
  抬起来）。
* 频率轴是**每单位时间**，默认频带是 ``[1/T, n/(2T)]``（下界是 Rayleigh 分辨率
  ``1/T``，上界是有效 Nyquist），栅格由**记录**决定而不是由采样密度决定：栅格
  步长永远不粗于 ``1/T``，所以同一条记录采 500 点还是 5000 点，主周期之差必须
  小于 5%，而且加密采样只会让频带更宽。
* 稳健版覆盖的频带必须与普通 Lomb-Scargle（``spectral_analysis``）一致或更宽，
  轨道带上的周期（例如 3 Myr 记录里的 21 kyr）必须落在带内并被找回。
* 置信线由卡方分布给出（2 个自由度）：``chi_p = background * chi2_ppf(p, 2) / 2``。
* 同一个种子必须给出逐点相同的结果；不给种子时必须发出 ``RuntimeWarning``。
* 边界（点数太少、数值为常数、栅格点数与迭代轮数与过采样倍数越界、频率区间
  非法）必须抛 ``ValidationError``。

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"robust_lomb_scargle"}


import numpy as np
import pytest
from scipy import stats as sp_stats

from stratigraphy.robust_lomb import robust_lomb_scargle
from stratigraphy.spectral_analysis import SpectralAnalyzer
from utils.exceptions import ValidationError

#: AR(1) 模拟序列的真系数。
TRUE_RHO = 0.7

#: 模拟序列的长度（采样间隔为 1）。
RECORD_LENGTH = 1000

#: 测试用的信号周期。
CYCLE_PERIOD = 25.0

#: 轨道带周期：3 Myr 记录、5 kyr 采样时的 21 kyr 预cession 带。
ORBITAL_PERIOD = 21.0


def _ar1_series(rho: float, size: int, seed: int) -> np.ndarray:
    """生成平稳 AR(1) 序列 ``x[t] = rho * x[t-1] + eps[t]``。"""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=size)
    series = np.zeros(size, dtype=float)
    for index in range(1, size):
        series[index] = rho * series[index - 1] + innovations[index]
    return series


def _cycle_record(count: int) -> tuple[np.ndarray, np.ndarray]:
    """带一条已知周期信号的记录，采样密度由 ``count`` 决定。"""
    step = 500.0 / (count - 1)
    time = np.arange(count, dtype=float) * step
    values = np.sin(2.0 * np.pi * time / CYCLE_PERIOD) + 0.3 * _ar1_series(0.4, count, seed=3)
    return time, values


def _peak_period(result) -> float:
    """结果中功率最大处的周期。"""
    peak = int(np.argmax(result.power))
    return 1.0 / float(result.frequencies[peak])


class TestRobustLombAr1Background:
    """稳健 AR(1) 背景必须重现记录自身的持续性。"""

    def test_the_ar1_coefficient_is_close_to_the_true_coefficient(self):
        """已知系数的 AR(1) 序列，稳健拟合必须接近真值。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        result = robust_lomb_scargle(time, values, random_seed=11)
        assert result.ar1_rho == pytest.approx(TRUE_RHO, abs=0.15)

    def test_white_noise_has_a_near_zero_coefficient(self):
        """白噪声没有持续性，AR(1) 系数必须接近 0。"""
        time = np.arange(800, dtype=float)
        values = np.random.default_rng(2).normal(size=800)
        result = robust_lomb_scargle(time, values, random_seed=1)
        assert result.ar1_rho == pytest.approx(0.0, abs=0.2)

    def test_the_background_is_below_the_record_at_low_frequencies(self):
        """低频端的背景谱必须低于记录谱（背景拟合在中值平滑谱上）。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        result = robust_lomb_scargle(time, values, random_seed=11)
        low = slice(0, max(1, result.frequencies.size // 10))
        assert float(np.mean(result.background[low])) < float(np.mean(result.power[low]))

    def test_the_confidence_lines_bracket_the_background(self):
        """95% 置信线由卡方分布给出，必须把背景夹在中间。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        result = robust_lomb_scargle(time, values, random_seed=11)
        assert np.all(result.lower_95 < result.background)
        assert np.all(result.background < result.upper_95)
        upper_factor = float(sp_stats.chi2.ppf(0.95, 2)) / 2.0
        assert result.upper_95 == pytest.approx(result.background * upper_factor, rel=1e-12)


class TestRobustLombFrequencyGrid:
    """频率栅格由记录决定，加密采样不能让它变粗。"""

    def test_the_grid_is_in_cycles_per_time_unit(self):
        """频率是每单位时间的循环频率，1/f 就是以时间单位表示的周期。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        result = robust_lomb_scargle(time, values, random_seed=11)
        assert np.all(result.frequencies > 0.0)
        assert np.all(np.diff(result.frequencies) > 0.0)
        assert result.frequencies[0] == pytest.approx(1.0 / (time[-1] - time[0]))

    def test_the_grid_is_never_coarser_than_the_rayleigh_resolution(self):
        """栅格步长必须不粗于记录的 Rayleigh 分辨率 1/T。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        result = robust_lomb_scargle(time, values, random_seed=11)
        resolution = 1.0 / (time[-1] - time[0])
        step = float(result.frequencies[1] - result.frequencies[0])
        assert step <= resolution

    def test_denser_sampling_keeps_the_period_stable(self):
        """同一条记录采 500 点与 5000 点：主周期之差小于 5%，步长都不粗于 1/T。"""
        coarse_time, coarse_values = _cycle_record(500)
        dense_time, dense_values = _cycle_record(5000)
        coarse = robust_lomb_scargle(coarse_time, coarse_values, random_seed=5)
        dense = robust_lomb_scargle(dense_time, dense_values, random_seed=5)
        resolution = 1.0 / (coarse_time[-1] - coarse_time[0])
        assert float(coarse.frequencies[1] - coarse.frequencies[0]) <= resolution
        assert float(dense.frequencies[1] - dense.frequencies[0]) <= resolution
        # 加密采样不会让频带变窄：可分辨的周期只会更多。
        assert dense.frequencies[-1] >= coarse.frequencies[-1]
        coarse_period = _peak_period(coarse)
        dense_period = _peak_period(dense)
        assert abs(coarse_period - dense_period) / coarse_period < 0.05

    def test_the_peak_period_of_a_known_cycle_is_recovered(self):
        """已知周期的信号，峰值周期必须被复原。"""
        time, values = _cycle_record(2000)
        result = robust_lomb_scargle(time, values, random_seed=5)
        assert _peak_period(result) == pytest.approx(CYCLE_PERIOD, rel=0.05)

    def test_the_orbital_band_is_covered(self):
        """3 Myr 记录里的 21 kyr 轨道周期必须落在频带内并被找回。

        回归：旧实现的栅格步长取了 ``1/(5T)``，频带因此只覆盖到 Nyquist 的
        1/5，21 kyr 的频率落在带外，最强峰变成了噪声峰。
        """
        time = np.arange(0.0, 3000.5, 5.0)
        values = np.sin(2.0 * np.pi * time / ORBITAL_PERIOD) + np.random.default_rng(7).normal(0.0, 0.3, size=time.size)
        result = robust_lomb_scargle(time, values, random_seed=3)
        frequency = 1.0 / ORBITAL_PERIOD
        assert result.frequencies[0] <= frequency <= result.frequencies[-1]
        assert _peak_period(result) == pytest.approx(ORBITAL_PERIOD, rel=0.02)

    def test_the_band_is_never_narrower_than_the_plain_lomb_scargle(self):
        """稳健版覆盖的频带必须与普通 Lomb-Scargle 一致或更宽。"""
        time = np.arange(0.0, 3000.5, 5.0)
        values = np.sin(2.0 * np.pi * time / ORBITAL_PERIOD) + np.random.default_rng(7).normal(0.0, 0.3, size=time.size)
        plain = SpectralAnalyzer().analyze(time, values, ar1_significance=False)
        robust = robust_lomb_scargle(time, values, random_seed=3)
        assert robust.frequencies[0] <= plain.frequencies[0] + 1e-12
        assert robust.frequencies[-1] >= plain.frequencies[-1] - 1e-12

    def test_oversampling_narrows_the_band_and_refines_the_step(self):
        """``oversampling > 1`` 用频带宽度换更细的步长，默认是 1。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        default = robust_lomb_scargle(time, values, random_seed=11)
        refined = robust_lomb_scargle(time, values, random_seed=11, oversampling=4.0)
        default_step = float(default.frequencies[1] - default.frequencies[0])
        refined_step = float(refined.frequencies[1] - refined.frequencies[0])
        assert refined_step < default_step
        assert refined.frequencies[-1] < default.frequencies[-1]


class TestRobustLombReproducibility:
    """随机部分（REDFIT 的假警概率自助法）必须可复现。"""

    def test_a_fixed_seed_reproduces_the_result(self):
        """同一个种子必须给出逐点相同的结果。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        first = robust_lomb_scargle(time, values, random_seed=11)
        second = robust_lomb_scargle(time, values, random_seed=11)
        assert np.array_equal(first.power, second.power)
        assert np.array_equal(first.background, second.background)
        assert first.ar1_rho == second.ar1_rho

    def test_without_a_seed_the_result_warns(self):
        """不给种子时必须发出 ``RuntimeWarning``（仓库对随机分析的统一约定）。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        with pytest.warns(RuntimeWarning):
            robust_lomb_scargle(time, values)

    def test_the_summary_reports_the_coefficient(self):
        """摘要必须给出稳健 AR(1) 系数与峰值周期。"""
        time = np.arange(RECORD_LENGTH, dtype=float)
        values = _ar1_series(TRUE_RHO, RECORD_LENGTH, seed=5)
        result = robust_lomb_scargle(time, values, random_seed=11)
        text = result.summary()
        assert f"{result.ar1_rho:.4f}" in text


class TestRobustLombBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_too_few_points_raise(self):
        """点数太少无法给出周期图。"""
        with pytest.raises(ValidationError):
            robust_lomb_scargle(np.arange(4.0), np.arange(4.0), random_seed=1)

    def test_a_constant_series_raises(self):
        """常数序列没有周期可找。"""
        time = np.arange(200.0)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.ones(200), random_seed=1)

    def test_mismatched_lengths_raise(self):
        """时间与数值必须等长。"""
        time = np.arange(200.0)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.ones(199), random_seed=1)

    def test_a_non_increasing_time_axis_raises(self):
        """时间轴必须严格递增。"""
        time = np.arange(200.0)
        broken = time.copy()
        broken[7] = broken[6]
        with pytest.raises(ValidationError):
            robust_lomb_scargle(broken, np.sin(time), random_seed=1)

    def test_too_few_frequencies_raise(self):
        """频率点数必须够用。"""
        time = np.arange(200.0)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.sin(time), n_frequencies=4, random_seed=1)

    def test_no_iteration_raises(self):
        """至少要迭代一轮才能得到 AR(1) 背景。"""
        time = np.arange(200.0)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.sin(time), max_iterations=0, random_seed=1)

    def test_an_oversampling_below_one_raises(self):
        """过采样倍数小于 1 会让栅格比 Rayleigh 分辨率还粗。"""
        time = np.arange(200.0)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.sin(time), oversampling=0.5, random_seed=1)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.sin(time), oversampling=float("nan"), random_seed=1)

    def test_an_invalid_frequency_range_raises(self):
        """频率区间必须是正数且递增。"""
        time = np.arange(200.0)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.sin(time), frequency_range=(0.0, 1.0), random_seed=1)
        with pytest.raises(ValidationError):
            robust_lomb_scargle(time, np.sin(time), frequency_range=(2.0, 1.0), random_seed=1)
