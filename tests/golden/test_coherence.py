# =============================================================================
# FILE: tests/golden/test_coherence.py
# =============================================================================
"""
``stratigraphy.coherence``（相干性与相位）的测试。

测试围绕三条主线组织：

* **真值对照。** 构造"两条序列 = 同一个 20 周期信号 + 各自的独立噪声"，
  相干性在 20 周期处必须接近 1；换成"两条完全独立的序列"，相干性必须远
  低于 1。这是本文件最核心的断言：它同时钉死了相干性的定义
  （``|S_xy|^2 / (S_xx * S_yy)``）、频率轴的口径（**每单位时间**，所以
  20 周期信号的频率是 ``1/20`` 而不是某个"每样本"的数）和平滑互谱的构造。

* **相位。** 把第二条序列**延迟**若干个时间单位（等价于若干个样本），估计
  出来的相位差必须与理论滞后 ``-2 * pi * f * dt`` 一致——**符号与量级都要
  对**。本模块的相位约定是 ``conj(A_hat_first) * A_hat_second``，所以第二条
  序列滞后时相位为**负**；符号写反会让估计值与理论相差约 180°，被容差
  5° 稳稳抓住。

* **显著性。** 零假设下 Welch 平滑后的幅度平方相干性服从 Beta(1, K-1)，
  临界值是 ``1 - alpha^(1/(K-1))``。测试直接与 ``scipy.stats.beta`` 的分位数
  对照，并断言"高于临界值才算显著"这条判定规则确实生效。

边界测试断言抛 ``utils.exceptions.ValidationError``；可复现性断言固定种子
给出完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Welch, P. D. (1967). The use of fast Fourier transform for the estimation
  of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15,
  70-73.（分段平均的交叉谱估计量）
* Priestley, M. B. (1981). Spectral Analysis and Time Series. Academic Press.
  （第 12 章：相干性的抽样分布 Beta(1, K-1)）
* Maran, A. & Ripley, B. D. (1997). Time Series Analysis and Forecasting.
  Springer.（Welch 相干性的零假设分布与 F(2, 2K-2) 等价形式）
* Scargle, J. D. (1982). Studies in astronomical time series analysis.
  *IEEE Transactions on Information Theory*, 28, 513-523.（Lomb-Scargle 周期图）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "coherence & phase" 子功能）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"coherence"}


import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from stratigraphy.coherence import coherence, coherence_significance
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录覆盖的时间跨度（任意时间单位）。
SPAN = 200.0

#: 合成记录的采样点数。
N_POINTS = 1000

#: 合成信号的周期（时间单位）—— 对应频率 1/20 每单位时间。
SIGNAL_PERIOD = 20.0

#: 合成信号的频率（每单位时间）。
SIGNAL_FREQUENCY = 1.0 / SIGNAL_PERIOD

#: 白噪声的标准差（相对于振幅为 1 的信号）。
NOISE = 0.1

#: 独立序列对照所用的噪声标准差。
INDEPENDENT_NOISE = 1.0


# =============================================================================
# 辅助函数
# =============================================================================


def _time_axis() -> npt.NDArray:
    """合成记录的时间轴。"""
    return np.linspace(0.0, SPAN, N_POINTS)


def _shared_signal_record(
    seed: int = 11,
    noise: float = NOISE,
    delay: float = 0.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """两条序列共用同一个 20 周期信号，各自叠加独立噪声。

    ``delay`` 是第二条序列相对第一条序列的**延迟**（时间单位）：第二条序列
    在时刻 ``t`` 取的是第一条序列在 ``t - delay`` 处的值。因此在频率 ``f``
    上，两者的相位差应当是 ``-2 * pi * f * delay``。

    Returns:
        (时间轴, 第一条序列, 第二条序列)。
    """
    rng = np.random.default_rng(seed)
    time = _time_axis()
    first = np.sin(2.0 * np.pi * time / SIGNAL_PERIOD)
    second = np.sin(2.0 * np.pi * (time - delay) / SIGNAL_PERIOD)
    first = first + noise * rng.standard_normal(N_POINTS)
    second = second + noise * rng.standard_normal(N_POINTS)
    return time, first, second


def _independent_records(
    seed: int = 5,
    noise: float = INDEPENDENT_NOISE,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """两条完全独立的序列（不含共同信号）。

    这是相干性的**零假设**对照：它们之间没有任何共同成分，相干性应当只
    剩下抽样噪声。

    Returns:
        (时间轴, 第一条序列, 第二条序列)。
    """
    rng = np.random.default_rng(seed)
    time = _time_axis()
    first = noise * rng.standard_normal(N_POINTS)
    second = noise * rng.standard_normal(N_POINTS)
    return time, first, second


def _nearest_bin(result, frequency: float) -> int:
    """返回最接近给定频率的频率栅格下标。"""
    return int(np.argmin(np.abs(result.frequencies - frequency)))


def _uneven_record(
    seed: int = 3,
    delay: float = 5.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """采样间隔不均匀的记录（地层记录的真实情形）。

    采样点由随机间隔累加得到，再整体缩放到 ``SPAN``。这条用例是本模块与
    FFT 型 Welch 实现的关键区别：分段 Lomb-Scargle 基在不等间距采样下仍然
    无偏，而基于 FFT 的分段平均要求等间距采样。

    Returns:
        (时间轴, 第一条序列, 第二条序列)。
    """
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(0.05, 0.35, 1200)
    time = np.cumsum(gaps)
    time = time / time[-1] * SPAN
    first = np.sin(2.0 * np.pi * time / SIGNAL_PERIOD) + NOISE * rng.standard_normal(time.size)
    second = np.sin(2.0 * np.pi * (time - delay) / SIGNAL_PERIOD) + NOISE * rng.standard_normal(time.size)
    return time, first, second


# =============================================================================
# 相干性：真值对照
# =============================================================================


class TestCoherenceGroundTruth:
    """相干性必须把"共同的强迫"和"互不相关的序列"区分开。"""

    def test_shared_signal_is_coherent_at_the_signal_period(self):
        """同一个 20 周期信号：20 周期处的相干性必须接近 1。"""
        time, first, second = _shared_signal_record()
        result = coherence(time, first, second)

        index = _nearest_bin(result, SIGNAL_FREQUENCY)
        assert result.coherence[index] > 0.9, (
            f"the coherence at the signal frequency is only {result.coherence[index]:.4f}"
        )
        # 峰值必须落在这个信号上，而不是别的频率上。
        peak = int(np.argmax(result.coherence))
        assert abs(result.frequencies[peak] - SIGNAL_FREQUENCY) < 1.0 / SIGNAL_PERIOD, (
            f"the coherence peaks at {result.frequencies[peak]:.6g}, not at {SIGNAL_FREQUENCY}"
        )

    def test_independent_series_are_far_from_coherent(self):
        """两条完全独立的序列：相干性必须远低于 1。

        判据用**中位数**而不是最大值：零假设下每个频率点的相干性都是一次
        Beta 抽样，500 个频率点里的最大值天然会偏高，拿最大值当判据测的是
        极值统计而不是估计量本身。
        """
        time, first, second = _independent_records()
        independent = coherence(time, first, second)
        shared_time, shared_first, shared_second = _shared_signal_record()
        shared = coherence(shared_time, shared_first, shared_second)

        index = _nearest_bin(independent, SIGNAL_FREQUENCY)
        assert independent.coherence[index] < 0.5, (
            f"the independent pair reaches {independent.coherence[index]:.4f} at the signal frequency"
        )
        assert float(np.median(independent.coherence)) < 0.5 * float(shared.coherence[index]), (
            "the independent pair is not clearly less coherent than the shared-signal pair"
        )
        assert float(np.median(shared.coherence)) < 0.5 * float(shared.coherence[index]), (
            "the shared-signal pair does not stand out from its own background"
        )

    def test_coherence_is_bounded_and_finite(self):
        """相干性必须落在 ``[0, 1]`` 且处处有限。"""
        time, first, second = _shared_signal_record()
        result = coherence(time, first, second)

        assert result.coherence.shape == result.frequencies.shape
        assert np.all(np.isfinite(result.coherence))
        assert np.all(result.coherence >= 0.0)
        assert np.all(result.coherence <= 1.0)
        assert np.all(np.isfinite(result.phase))
        assert np.all(np.isfinite(result.cross_power))

    def test_frequency_axis_is_per_unit_time(self):
        """频率轴必须是**每单位时间**的循环频率。

        若是"每样本"的归一化频率，20 周期信号的频率不可能是 ``1/20``。
        """
        time, first, second = _shared_signal_record()
        result = coherence(time, first, second)

        index = _nearest_bin(result, SIGNAL_FREQUENCY)
        assert abs(result.frequencies[index] - SIGNAL_FREQUENCY) < 1e-3, (
            f"the signal frequency {SIGNAL_FREQUENCY} is not on the frequency axis"
        )
        assert float(np.min(result.frequencies)) > 0.0
        assert np.all(np.diff(result.frequencies) > 0.0)

    def test_results_are_reproducible(self):
        """同一输入必须给出完全相同的结果。"""
        time, first, second = _shared_signal_record()
        one = coherence(time, first, second)
        two = coherence(time, first, second)

        assert np.array_equal(one.coherence, two.coherence)
        assert np.array_equal(one.phase, two.phase)
        assert one.n_effective == two.n_effective


# =============================================================================
# 不等间距采样
# =============================================================================


class TestUnevenSampling:
    """不等间距采样下估计量必须仍然无偏。"""

    def test_unevenly_sampled_records_are_still_coherent(self):
        """采样间隔不均匀时，信号频率处的相干性必须仍然接近 1。"""
        time, first, second = _uneven_record()
        result = coherence(time, first, second)

        index = _nearest_bin(result, SIGNAL_FREQUENCY)
        assert result.coherence[index] > 0.9, (
            f"the coherence of an unevenly sampled record is only {result.coherence[index]:.4f}"
        )

    def test_the_phase_of_an_unevenly_sampled_record_is_accurate(self):
        """不等间距采样下，相位差仍必须与理论滞后一致。"""
        delay = 5.0
        time, first, second = _uneven_record(delay=delay)
        result = coherence(time, first, second)

        index = _nearest_bin(result, SIGNAL_FREQUENCY)
        estimated = float(np.degrees(result.phase[index]))
        theoretical = float(np.degrees(-2.0 * np.pi * result.frequencies[index] * delay))

        assert abs(estimated - theoretical) < 5.0, (
            f"the estimated phase {estimated:.2f} degrees does not match the theoretical lag {theoretical:.2f} degrees"
        )


# =============================================================================
# 相位
# =============================================================================


class TestPhase:
    """相位差必须与理论滞后一致——符号与量级都要对。"""

    @pytest.mark.parametrize("delay", [2.5, 5.0, 7.5])
    def test_phase_matches_the_theoretical_lag(self, delay: float):
        """第二条序列延迟 ``delay``：相位差必须是 ``-2 * pi * f * delay``。

        容差 5°：抽样噪声带来的相位误差远小于这个量级，而**符号写反**会
        造成约 180° 的偏差，被稳稳抓住。
        """
        time, first, second = _shared_signal_record(delay=delay)
        result = coherence(time, first, second)

        index = _nearest_bin(result, SIGNAL_FREQUENCY)
        estimated = float(np.degrees(result.phase[index]))
        theoretical = float(np.degrees(-2.0 * np.pi * result.frequencies[index] * delay))

        assert abs(estimated - theoretical) < 5.0, (
            f"the estimated phase {estimated:.2f} degrees does not match the theoretical lag {theoretical:.2f} degrees"
        )

    def test_a_delay_gives_a_negative_phase(self):
        """滞后必须是**负**相位（约定：相位是第二条相对第一条的相位差）。"""
        time, first, second = _shared_signal_record(delay=5.0)
        result = coherence(time, first, second)
        index = _nearest_bin(result, SIGNAL_FREQUENCY)

        assert result.phase[index] < 0.0, "a delayed second series must give a negative phase"

    def test_identical_series_have_zero_phase(self):
        """两条序列完全相同时相位差必须是 0。"""
        time, first, _second = _shared_signal_record()
        result = coherence(time, first, first)
        index = _nearest_bin(result, SIGNAL_FREQUENCY)

        assert abs(float(np.degrees(result.phase[index]))) < 1.0, (
            f"identical series give a phase of {np.degrees(result.phase[index]):.3f} degrees"
        )
        assert result.coherence[index] > 0.99

    def test_phase_is_reported_in_radians(self):
        """相位必须以弧度给出（``0..pi`` 与 ``-pi..0`` 之间循环）。"""
        time, first, second = _shared_signal_record(delay=5.0)
        result = coherence(time, first, second)

        assert np.all(result.phase <= np.pi)
        assert np.all(result.phase >= -np.pi)


# =============================================================================
# 显著性
# =============================================================================


class TestSignificance:
    """零假设临界值必须是 Beta(1, K-1) 的分位数。"""

    def test_critical_value_matches_the_beta_null(self):
        """临界值必须等于 Beta(1, K-1) 的 ``level`` 分位数。

        有效自由度 ``n_effective = 2K``，于是 ``K = n_effective / 2``。
        """
        n_effective = 8.0
        n_segments = n_effective / 2.0
        critical = coherence_significance(np.array([0.1, 0.5, 0.9]), n_effective, 0.95)
        expected = float(stats.beta.ppf(0.95, 1.0, n_segments - 1.0))

        assert critical == pytest.approx(expected, rel=1e-12), (
            f"the critical value {critical:.6f} does not match the Beta(1, {n_segments - 1:.0f}) quantile "
            f"{expected:.6f}"
        )

    def test_critical_value_is_stricter_for_fewer_averages(self):
        """独立平均越少，临界值必须越高（越保守）。"""
        weak = coherence_significance(np.array([0.5]), 4.0, 0.95)
        strong = coherence_significance(np.array([0.5]), 16.0, 0.95)

        assert weak > strong

    def test_observed_coherence_above_the_threshold_is_significant(self):
        """高于临界值的相干性才算显著。"""
        time, first, second = _shared_signal_record()
        result = coherence(time, first, second)
        critical = coherence_significance(result.coherence, result.n_effective, 0.95)

        index = _nearest_bin(result, SIGNAL_FREQUENCY)
        assert result.coherence[index] > critical, "the shared signal must clear the critical value"

    def test_effective_degrees_of_freedom_follow_the_segment_count(self):
        """有效自由度必须是分段平均数的两倍。"""
        time, first, second = _shared_signal_record()
        four = coherence(time, first, second, n_segments=4)
        eight = coherence(time, first, second, n_segments=8)

        assert four.n_effective == 8.0
        assert eight.n_effective == 16.0


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_mismatched_lengths_are_rejected(self):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            coherence(time, first[:-1], second)

    def test_too_few_points_are_rejected(self):
        time = np.linspace(0.0, 1.0, 8)
        with pytest.raises(ValidationError):
            coherence(time, np.sin(time), np.cos(time))

    def test_non_finite_values_are_rejected(self):
        time, first, second = _shared_signal_record()
        broken = first.copy()
        broken[10] = np.nan
        with pytest.raises(ValidationError):
            coherence(time, broken, second)

    def test_non_increasing_time_axis_is_rejected(self):
        time, first, second = _shared_signal_record()
        broken = time.copy()
        broken[10] = broken[11]
        with pytest.raises(ValidationError):
            coherence(broken, first, second)

    def test_constant_series_is_rejected(self):
        time, _first, _second = _shared_signal_record()
        with pytest.raises(ValidationError):
            coherence(time, np.ones_like(time), np.zeros_like(time))

    @pytest.mark.parametrize(
        "frequency_range",
        [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.5), (1.0, 1.0)],
    )
    def test_invalid_frequency_range_is_rejected(self, frequency_range: tuple[float, float]):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            coherence(time, first, second, frequency_range=frequency_range)

    def test_band_below_the_segment_resolution_is_rejected(self):
        """低于 Welch 分段分辨率的频带无法分辨，必须报错而不是给出垃圾结果。"""
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            coherence(time, first, second, frequency_range=(1.0e-4, 2.0e-4))

    def test_invalid_frequency_count_is_rejected(self):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            coherence(time, first, second, n_frequencies=3)

    @pytest.mark.parametrize("n_segments", [1, 0, -2, 999])
    def test_invalid_segment_count_is_rejected(self, n_segments: int):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            coherence(time, first, second, n_segments=n_segments)

    def test_significance_rejects_degenerate_degrees_of_freedom(self):
        with pytest.raises(ValidationError):
            coherence_significance(np.array([0.5]), 2.0)

    def test_significance_rejects_invalid_levels(self):
        with pytest.raises(ValidationError):
            coherence_significance(np.array([0.5]), 8.0, level=1.5)

    def test_significance_rejects_out_of_range_coherence(self):
        with pytest.raises(ValidationError):
            coherence_significance(np.array([1.5]), 8.0)
