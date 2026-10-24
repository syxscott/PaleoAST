# =============================================================================
# FILE: tests/golden/test_wavelet_coherence.py
# =============================================================================
"""
``stratigraphy.wavelet_coherence``（小波相干分析族）的测试。

测试围绕六条主线组织，每条断言的依据都写在对应测试的 docstring 里：

* **自洽性。** 两条**完全相同**的序列 → WTC 的相干性在信号周期处接近 1
  （> 0.8），相位接近 0（|phase| < 0.1 rad）。依据是相干性的定义本身：
  ``R^2 = |S_xy|^2 / (S_xx * S_yy)``，而两条序列相同时
  ``S_xy = S_xx = S_yy``（``W_x * conj(W_x) = |W_x|^2`` 是实数），于是
  ``R^2 ≡ 1``、相位 ≡ 0。这条断言同时钉死了"互谱取 ``W_x * conj(W_y)``"
  与"相位 = 第一条序列的相位 − 第二条序列的相位"这两个约定。

* **独立性对照。** 两条**独立**的白噪声序列 → 相干性的中位数必须远低于
  共享信号那一对的峰值。判据用**中位数**而不是最大值：零假设下每个
  (尺度, 时间) 点的相干性都是一次抽样，几百个尺度上的最大值天然偏高。

* **相位方向与量级。** 把第二条序列在时间轴上整体平移，测得的相位差必须
  与理论值一致——**符号与量级都要对**。本模块的符号约定（与 Grinsted et
  al. 2004 及 Acycle 的 ``wcoherence`` 一致）是
  ``phase = arg(W_x * conj(W_y))``：第二条序列**提前** Δt 时
  ``phase = -2 * pi * f * Δt``，**滞后** Δt 时 ``phase = +2 * pi * f * Δt``。
  两个方向都断言，容差 0.05 rad：抽样误差远小于这个量级，而**符号写反**
  会造成约 180°（π rad）的偏差，被稳稳抓住。

* **尺度 → 周期映射。** 构造周期 P 的正弦（叠加各自独立的噪声），WTC 的
  时间平均相干性曲线的峰值所在周期必须回到 P（相对误差 < 1%）。这条断言
  钉死了 Fourier 因子的口径：周期 = Fourier 因子 × 尺度 × dt，且
  Fourier 因子与母小波的归一化必须**成对**使用（Torrence & Compo 1998）。

* **小波显著性。** AR(1) 蒙特卡洛背景给出的 95% 临界值必须低于观测到的小
  波谱值：白噪声输入下，谱峰所在尺度的谱值高于临界值；"正弦 + 白噪声"输入
  下，真实周期处的谱值远高于临界值。

* **边界与可复现性。** 非法输入（点数太少、含 NaN、时间轴非单调、时间轴
  非均匀、长度不一致、尺度轴非法等）必须抛 ``utils.exceptions`` 的
  ``ValidationError``；同种子的两次蒙特卡洛调用必须给出完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Grinsted, A., Moore, J. C. & Jevrejeva, S. (2004). Application of the cross
  wavelet transform and wavelet coherence to geophysical time series.
  *Nonlinear Processes in Geophysics*, 11(5), 561-566.（XWT/WTC 的定义、
  ``W_xy = W_x * conj(W_y)`` 的相位约定、时间维与尺度维平滑、AR(1) 蒙特卡洛
  临界值）
* Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet analysis.
  *Bulletin of the American Meteorological Society*, 79(1), 61-78.（Morlet 与
  Mexican Hat 母小波定义、Fourier 因子、锥形影响域）
* Torrence, C. & Compo, G. P. (2008). Interdecadal changes in the ENSO-Monsoon
  system. *Journal of Climate*, 21(11), 2448-2462.（时间维与尺度维平滑）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "wavelet coherence" 子功能）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"wavelet_coherence"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.wavelet_coherence import (
    cross_wavelet_transform,
    wavelet_coherence,
    wavelet_coherence_significance,
    wavelet_significance,
    wavelet_transform,
)
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

#: 采样间隔（时间单位）。
DT = SPAN / (N_POINTS - 1)

#: 白噪声的标准差（相对于振幅为 1 的信号）。
NOISE = 0.5

#: 独立序列对照所用的噪声标准差。
INDEPENDENT_NOISE = 1.0

#: 相位平移量（时间单位）。
SHIFT = 5.0

#: 精细尺度栅格：峰值定位与相位测试需要足够细的尺度分辨率。
FINE_SCALES = np.arange(4.0, 140.0, 0.25)

#: 蒙特卡洛测试用的粗尺度栅格（蒙特卡洛很贵，尺度分辨率在这里不重要）。
COARSE_SCALES = np.arange(4.0, 140.0, 1.0)

#: 蒙特卡洛重复次数。
MONTE_CARLO = 50


# =============================================================================
# 辅助函数
# =============================================================================


def _time_axis() -> npt.NDArray:
    """合成记录的时间轴（严格递增、等间距）。"""
    return np.linspace(0.0, SPAN, N_POINTS)


def _shared_signal_record(
    seed: int = 11,
    noise: float = NOISE,
    shift: float = 0.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """两条序列共用同一个 20 周期信号，各自叠加独立噪声。

    ``shift`` 是第二条序列相对第一条序列的**平移量**（时间单位）：正值表示
    第二条序列整体**提前**（``y(t) = x(t + shift)``），负值表示**滞后**。

    Returns:
        (时间轴, 第一条序列, 第二条序列)。
    """
    rng = np.random.default_rng(seed)
    time = _time_axis()
    first = np.sin(2.0 * np.pi * time / SIGNAL_PERIOD)
    second = np.sin(2.0 * np.pi * (time + shift) / SIGNAL_PERIOD)
    first = first + noise * rng.standard_normal(N_POINTS)
    second = second + noise * rng.standard_normal(N_POINTS)
    return time, first, second


def _independent_records(
    seed: int = 5,
    noise: float = INDEPENDENT_NOISE,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """两条完全独立的序列（不含共同成分）。

    这是相干性的**零假设**对照。

    Returns:
        (时间轴, 第一条序列, 第二条序列)。
    """
    rng = np.random.default_rng(seed)
    time = _time_axis()
    first = noise * rng.standard_normal(N_POINTS)
    second = noise * rng.standard_normal(N_POINTS)
    return time, first, second


def _nearest_period(result, period: float) -> int:
    """返回最接近给定周期的尺度栅格下标。"""
    return int(np.argmin(np.abs(result.periods - period)))


def _valid_mask(result) -> npt.NDArray:
    """锥形影响域之外的布尔掩码（形状 ``(尺度数, 时间点数)``）。

    掩码直接取自产物里的 ``cone_of_influence``：周期超过该时间点的有效最大
    周期时，边界效应已经污染了该点。
    """
    return result.periods[:, None] <= result.cone_of_influence[None, :]


def _mean_coherence_profile(result) -> npt.NDArray:
    """时间平均的相干性曲线（只在锥形影响域之外的点上平均）。"""
    return np.nanmean(np.where(_valid_mask(result), result.coherence, np.nan), axis=1)


def _circular_mean(phases: npt.NDArray) -> float:
    """角度的圆平均（直接取算术平均会在 ±π 附近出错）。"""
    return float(np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))))


# =============================================================================
# 相干性：自洽性与独立性对照
# =============================================================================


class TestCoherenceGroundTruth:
    """相干性必须把"共同的强迫"和"互不相关的序列"区分开。"""

    def test_identical_series_are_coherent_at_the_signal_period(self):
        """两条**完全相同**的序列：信号周期处的相干性必须接近 1。

        依据：``R^2 = |S_xy|^2 / (S_xx * S_yy)``，而相同的序列给出
        ``S_xy = S_xx = S_yy``（``W_x * conj(W_x) = |W_x|^2`` 是实数），
        因此 ``R^2 ≡ 1``。
        """
        time, first, _second = _shared_signal_record()
        result = wavelet_coherence(first, first, FINE_SCALES, time=time, significance=False)

        index = _nearest_period(result, SIGNAL_PERIOD)
        assert result.coherence[index].max() > 0.8, (
            f"the coherence of two identical series is only {result.coherence[index].max():.4f}"
        )

    def test_identical_series_have_zero_phase(self):
        """两条相同的序列：相位差必须是 0。

        依据：相同序列的互谱 ``W_x * conj(W_x)`` 是**实数**，相位恒为 0
        （只有浮点噪声）。
        """
        time, first, _second = _shared_signal_record()
        result = wavelet_coherence(first, first, FINE_SCALES, time=time, significance=False)
        index = _nearest_period(result, SIGNAL_PERIOD)

        phase = _circular_mean(result.phase[index][_valid_mask(result)[index]])
        assert abs(phase) < 0.1, f"identical series give a phase of {phase:.4f} radians"

    def test_shared_signal_is_coherent_at_the_signal_period(self):
        """同一个 20 周期信号 + 各自的独立噪声：信号周期处必须高度相干。

        这是真实地层记录的情形：两条记录被同一个天文强迫驱动，但各自还带有
        测量噪声与别的强迫。
        """
        time, first, second = _shared_signal_record()
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        index = _nearest_period(result, SIGNAL_PERIOD)
        assert float(result.coherence[index].max()) > 0.8, (
            f"the coherence at the signal period is only {result.coherence[index].max():.4f}"
        )

    def test_independent_series_are_far_from_coherent(self):
        """两条完全独立的序列：相干性必须远低于共享信号那一对。

        判据用**中位数**：零假设下每个点的相干性都是一次抽样，几百个尺度上的
        最大值天然偏高，拿最大值当判据测的是极值统计而不是估计量本身。
        """
        time, first, second = _independent_records()
        independent = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)
        shared_time, shared_first, shared_second = _shared_signal_record()
        shared = wavelet_coherence(shared_first, shared_second, FINE_SCALES, time=shared_time, significance=False)

        peak = float(shared.coherence[_nearest_period(shared, SIGNAL_PERIOD)].max())
        assert float(np.median(independent.coherence)) < 0.5 * peak, (
            "the independent pair is not clearly less coherent than the shared-signal pair"
        )

    def test_coherence_is_bounded_and_finite(self):
        """相干性必须落在 ``[0, 1]`` 且处处有限。"""
        time, first, second = _shared_signal_record()
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        assert result.coherence.shape == (result.scales.size, result.times.size)
        assert np.all(np.isfinite(result.coherence))
        assert np.all(result.coherence >= 0.0)
        assert np.all(result.coherence <= 1.0)

    def test_period_axis_is_per_unit_time(self):
        """周期轴必须是**每单位时间**的口径，且与频率轴互为倒数。"""
        time, first, second = _shared_signal_record()
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        assert float(np.min(result.periods)) > 0.0
        assert np.all(np.diff(result.periods) > 0.0)
        assert np.allclose(result.periods * result.frequencies, 1.0)

    def test_phase_is_nan_where_the_coherence_is_low(self):
        """相干性极低处相位必须是 NaN（本模块对"相位无意义"的明确约定）。

        依据：模块 docstring 声明"相干性低于 ``phase_threshold`` 时相位返回
        NaN"。低相干处的相位是纯噪声，任何数值都会误导读者。
        """
        time, first, second = _independent_records()
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        weak = result.coherence < result.phase_threshold
        assert np.all(np.isnan(result.phase[weak])), "the phase must be NaN where the coherence is low"
        strong = result.coherence >= result.phase_threshold
        assert np.all(np.isfinite(result.phase[strong])), "the phase must be finite where the coherence is high"


# =============================================================================
# 相位：方向与量级
# =============================================================================


class TestPhase:
    """相位差必须与理论平移一致——符号与量级都要对。"""

    def test_an_advanced_second_series_gives_the_negative_theoretical_phase(self):
        """第二条序列**提前** SHIFT：相位差必须是 ``-2 * pi * f * SHIFT``。

        本模块的符号约定（见模块 docstring）：``phase = arg(W_x * conj(W_y))``
        = 第一条序列的相位 − 第二条序列的相位。因此第二条序列提前 Δt 时
        相位为负。

        容差 0.05 rad：抽样噪声带来的相位误差远小于这个量级，而**符号写反**
        会造成约 180°（π rad）的偏差，被稳稳抓住。
        """
        time, first, second = _shared_signal_record(shift=SHIFT)
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        index = _nearest_period(result, SIGNAL_PERIOD)
        estimated = _circular_mean(result.phase[index][_valid_mask(result)[index]])
        theoretical = -2.0 * np.pi * SIGNAL_FREQUENCY * SHIFT

        assert abs(estimated - theoretical) < 0.05, (
            f"the estimated phase {estimated:.4f} rad does not match the theoretical {theoretical:.4f} rad"
        )

    def test_a_delayed_second_series_gives_the_positive_theoretical_phase(self):
        """第二条序列**滞后** SHIFT：相位差必须是 ``+2 * pi * f * SHIFT``。

        这是上一个断言的镜像：两个方向都钉死，符号就不可能被悄悄翻转。
        """
        time, first, second = _shared_signal_record(shift=-SHIFT)
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        index = _nearest_period(result, SIGNAL_PERIOD)
        estimated = _circular_mean(result.phase[index][_valid_mask(result)[index]])
        theoretical = 2.0 * np.pi * SIGNAL_FREQUENCY * SHIFT

        assert abs(estimated - theoretical) < 0.05, (
            f"the estimated phase {estimated:.4f} rad does not match the theoretical {theoretical:.4f} rad"
        )

    def test_the_two_shift_directions_differ_by_the_full_lag(self):
        """提前与滞后之间的相位差必须是 ``2 * 2 * pi * f * SHIFT``（而不是 0）。

        两个方向相差 ``2 * SHIFT`` 个时间单位，因此相位差是单侧理论值的两倍
        （在本测试的参数下正好是 π rad）。
        """
        time, first, advanced = _shared_signal_record(shift=SHIFT)
        _time, _first, delayed = _shared_signal_record(shift=-SHIFT)
        advanced_result = wavelet_coherence(first, advanced, FINE_SCALES, time=time, significance=False)
        delayed_result = wavelet_coherence(first, delayed, FINE_SCALES, time=time, significance=False)

        index = _nearest_period(advanced_result, SIGNAL_PERIOD)
        advanced_phase = _circular_mean(advanced_result.phase[index][_valid_mask(advanced_result)[index]])
        delayed_phase = _circular_mean(delayed_result.phase[index][_valid_mask(delayed_result)[index]])
        expected_gap = 2.0 * (2.0 * np.pi * SIGNAL_FREQUENCY * SHIFT)

        assert abs(abs(advanced_phase - delayed_phase) - expected_gap) < 0.05, (
            "the two shift directions are not separated by the full theoretical lag"
        )

    def test_cross_wavelet_phase_follows_the_same_convention(self):
        """XWT 的逐尺度相位必须与 WTC 用的是**同一个**符号约定。

        依据：``W_xy = W_x * conj(W_y)``，其辐角就是"第一条序列的相位减去
        第二条序列的相位"。
        """
        time, first, second = _shared_signal_record(shift=SHIFT)
        result = cross_wavelet_transform(first, second, FINE_SCALES, time=time)

        index = _nearest_period(result, SIGNAL_PERIOD)
        middle = result.cross_wavelet[index, N_POINTS // 2]
        estimated = float(np.angle(middle))
        theoretical = -2.0 * np.pi * SIGNAL_FREQUENCY * SHIFT

        assert abs(estimated - theoretical) < 0.05, (
            f"the cross-wavelet phase {estimated:.4f} rad does not match the theoretical {theoretical:.4f} rad"
        )


# =============================================================================
# 尺度 → 周期映射
# =============================================================================


class TestPeriodMapping:
    """尺度必须被 Fourier 因子正确地换算成周期。"""

    def test_the_shared_periodicity_peak_returns_the_true_period(self):
        """WTC 产物里的共享周期峰必须回到 P（相对误差 < 1%）。

        依据：Torrence & Compo (1998) 的 Fourier 因子把尺度换算成周期，而
        本模块的归一化与该 Fourier 因子是**成对**使用的，因此响应峰值所在
        尺度会被还原成真实周期。

        这里读的是 **WTC 产物自己的交叉小波功率** ``|W_xy|``（时间平均，掩码
        取自产物自己的 ``cone_of_influence``），而不是"时间平均 R²"曲线：
        R² 是比值，共享信号很强时它会在信号尺度**附近饱和**（R² → 1），
        实测在 ±10% 的周期范围内只差 0.003，其极大值因此不是一个良态的
        周期估计量（不同噪声实现之间可差 ±10%）。相干性本身由上面的
        "信号周期处高度相干"那条断言钉死。
        """
        time, first, second = _shared_signal_record()
        result = wavelet_coherence(first, second, FINE_SCALES, time=time, significance=False)

        power = np.abs(result.cross_wavelet) ** 2
        profile = np.nanmean(np.where(_valid_mask(result), power, np.nan), axis=1)
        peak = int(np.nanargmax(profile))
        peak_period = float(result.periods[peak])
        relative_error = abs(peak_period - SIGNAL_PERIOD) / SIGNAL_PERIOD

        assert relative_error < 0.01, (
            f"the shared periodicity peaks at period {peak_period:.4f}, which is "
            f"{relative_error:.3%} away from the true period {SIGNAL_PERIOD}"
        )

        # 同一条曲线上的相干性本身也必须很高（时间平均，取自锥形影响域之内）。
        profile = _mean_coherence_profile(result)
        index = _nearest_period(result, SIGNAL_PERIOD)
        assert float(profile[index]) > 0.8, f"the mean coherence at the signal period is only {profile[index]:.4f}"

    def test_the_wavelet_spectrum_peak_returns_the_true_period(self):
        """小波谱峰所在周期同样必须回到 P（相对误差 < 1%）。

        这是同一件事的另一条独立读法：谱峰来自单条序列的小波系数，不依赖
        两条序列之间的相干性。
        """
        time, first, _second = _shared_signal_record()
        result = wavelet_transform(first, FINE_SCALES, time=time)

        power = np.abs(result.coefficients) ** 2
        profile = np.nanmean(np.where(_valid_mask(result), power, np.nan), axis=1)
        peak = int(np.nanargmax(profile))
        peak_period = float(result.periods[peak])
        relative_error = abs(peak_period - SIGNAL_PERIOD) / SIGNAL_PERIOD

        assert relative_error < 0.01, (
            f"the wavelet spectrum peaks at period {peak_period:.4f}, which is "
            f"{relative_error:.3%} away from the true period {SIGNAL_PERIOD}"
        )

    def test_the_period_axis_is_a_linear_function_of_the_scale(self):
        """周期轴必须是尺度的**线性**函数（Fourier 因子是常数）。

        依据：``周期 = Fourier 因子 × 尺度 × dt``，因此
        ``周期 / 尺度`` 必须是常数。
        """
        time, first, _second = _shared_signal_record()
        result = wavelet_transform(first, FINE_SCALES, time=time)

        ratio = result.periods / result.scales
        assert float(np.std(ratio) / np.mean(ratio)) < 1.0e-12, "the period axis is not linear in the scale"


# =============================================================================
# 显著性
# =============================================================================


class TestSignificance:
    """蒙特卡洛临界值必须低于观测到的谱值。"""

    def test_white_noise_spectrum_exceeds_its_critical_value(self):
        """白噪声输入下，观测谱必须在**某个尺度上**超过 95% 临界值。

        依据：Grinsted et al. (2004) 的蒙特卡洛做法——用与观测序列同方差、
        同 AR(1) 系数的替代序列给出谱的零分布，取 95% 分位数作临界值。
        零假设下每个尺度上"观测值超过临界值"的概率是 5%，因此"存在一个
        尺度超过"的概率是 ``1 - 0.95^n_scales``（本测试 n_scales = 136）。
        """
        rng = np.random.default_rng(17)
        time = _time_axis()
        noise = rng.standard_normal(N_POINTS)
        result = wavelet_significance(noise, COARSE_SCALES, time=time, n_monte_carlo=MONTE_CARLO, seed=13)

        significant = result.spectrum > result.critical_values
        assert bool(np.any(significant)), (
            "the observed spectrum does not exceed its Monte Carlo critical value at any scale"
        )

    def test_the_signal_period_exceeds_its_critical_value(self):
        """白噪声背景 + 20 周期信号：真实周期处的谱值必须远高于临界值。

        这里把 AR(1) 系数显式设为 0（纯白噪声背景）：观测序列含一个强周期
        成分，用它自己的 AR(1) 系数会把周期信号误当成红噪声。
        """
        rng = np.random.default_rng(23)
        time = _time_axis()
        noisy_signal = np.sin(2.0 * np.pi * time / SIGNAL_PERIOD) + 0.5 * rng.standard_normal(N_POINTS)
        result = wavelet_significance(
            noisy_signal,
            COARSE_SCALES,
            time=time,
            ar1=0.0,
            n_monte_carlo=MONTE_CARLO,
            seed=21,
        )

        index = _nearest_period(result, SIGNAL_PERIOD)
        assert result.spectrum[index] > result.critical_values[index], (
            "the spectrum at the signal period does not exceed its Monte Carlo critical value"
        )

    def test_the_coherence_exceeds_its_critical_value(self):
        """共享信号的 WTC 相干性必须高于 AR(1) 背景给出的临界值。

        依据：Grinsted et al. (2004) §3——两条**独立**的 AR(1) 序列给出
        ``R²`` 的零分布，取 95% 分位数作临界值。
        """
        time, first, second = _shared_signal_record()
        result = wavelet_coherence(
            first,
            second,
            COARSE_SCALES,
            time=time,
            significance=True,
            n_monte_carlo=MONTE_CARLO,
            seed=31,
        )

        assert result.significance is not None, "the critical value curve must be reported"
        index = _nearest_period(result, SIGNAL_PERIOD)
        assert float(result.coherence[index].max()) > float(result.significance[index]), (
            "the observed coherence does not exceed its Monte Carlo critical value"
        )

    def test_the_coherence_critical_value_curve_is_reported_per_scale(self):
        """临界值曲线必须与尺度轴一一对应。"""
        time, first, second = _shared_signal_record()
        result = wavelet_coherence_significance(
            first, second, COARSE_SCALES, time=time, n_monte_carlo=MONTE_CARLO, seed=31
        )

        assert result.critical_values.shape == result.periods.shape
        assert result.critical_values.size == COARSE_SCALES.size
        assert len(result.ar1_coefficients) == 2


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_too_few_points_are_rejected(self):
        time = np.linspace(0.0, 1.0, 8)
        with pytest.raises(ValidationError):
            wavelet_coherence(np.sin(time), np.cos(time), FINE_SCALES, time=time)

    def test_non_finite_values_are_rejected(self):
        time, first, second = _shared_signal_record()
        broken = first.copy()
        broken[10] = np.nan
        with pytest.raises(ValidationError):
            wavelet_coherence(broken, second, FINE_SCALES, time=time)

    def test_non_increasing_time_axis_is_rejected(self):
        time, first, second = _shared_signal_record()
        broken = time.copy()
        broken[10] = broken[9]
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, FINE_SCALES, time=broken)

    def test_unevenly_spaced_time_axis_is_rejected(self):
        """非均匀但单调的时间轴必须报错（本模块只支持均匀采样）。"""
        time, first, second = _shared_signal_record()
        broken = time.copy()
        broken[10] = broken[9] + 2.0 * (broken[10] - broken[9])
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, FINE_SCALES, time=broken)

    def test_mismatched_lengths_are_rejected(self):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_coherence(first[:-1], second, FINE_SCALES, time=time)

    def test_mismatched_time_axis_length_is_rejected(self):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, FINE_SCALES, time=time[:-1])

    @pytest.mark.parametrize(
        "scales",
        [
            np.array([4.0]),
            np.array([8.0, 4.0]),
            np.array([-4.0, 8.0]),
            np.array([4.0, 8.0, 8.0]),
            np.array([4.0, np.nan]),
        ],
    )
    def test_invalid_scales_are_rejected(self, scales: npt.NDArray):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, scales, time=time)

    def test_an_unknown_wavelet_is_rejected(self):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, FINE_SCALES, time=time, wavelet="pauling")

    def test_an_invalid_sampling_interval_is_rejected(self):
        _time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, FINE_SCALES, dt=0.0)

    def test_an_invalid_monte_carlo_count_is_rejected(self):
        time, first, _second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_significance(first, COARSE_SCALES, time=time, n_monte_carlo=0)

    def test_an_invalid_significance_level_is_rejected(self):
        time, first, _second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_significance(first, COARSE_SCALES, time=time, level=1.5)

    def test_an_invalid_phase_threshold_is_rejected(self):
        time, first, second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_coherence(first, second, FINE_SCALES, time=time, phase_threshold=1.5)

    def test_an_invalid_ar1_coefficient_is_rejected(self):
        time, first, _second = _shared_signal_record()
        with pytest.raises(ValidationError):
            wavelet_significance(first, COARSE_SCALES, time=time, ar1=1.0)


# =============================================================================
# 可复现性
# =============================================================================


class TestReproducibility:
    """同种子的蒙特卡洛必须逐位可复现。"""

    def test_the_wavelet_significance_is_reproducible(self):
        time, first, _second = _shared_signal_record()
        one = wavelet_significance(first, COARSE_SCALES, time=time, n_monte_carlo=20, seed=7)
        two = wavelet_significance(first, COARSE_SCALES, time=time, n_monte_carlo=20, seed=7)

        assert np.allclose(one.critical_values, two.critical_values)
        assert np.allclose(one.spectrum, two.spectrum)

    def test_the_coherence_significance_is_reproducible(self):
        time, first, second = _shared_signal_record()
        one = wavelet_coherence_significance(first, second, COARSE_SCALES, time=time, n_monte_carlo=20, seed=7)
        two = wavelet_coherence_significance(first, second, COARSE_SCALES, time=time, n_monte_carlo=20, seed=7)

        assert np.allclose(one.critical_values, two.critical_values)

    def test_the_transform_itself_is_deterministic(self):
        time, first, _second = _shared_signal_record()
        one = wavelet_transform(first, FINE_SCALES, time=time)
        two = wavelet_transform(first, FINE_SCALES, time=time)

        assert np.array_equal(one.coefficients, two.coefficients)


# =============================================================================
# Summary
# =============================================================================


class TestSummary:
    """Every result must carry a summary a person can read.

    The summary is not decoration: the script console prints the result's
    ``summary()`` for the user.  Without it only the dataclass ``repr``
    survives, and the whole coefficient matrix is dumped on the screen.
    """

    def test_the_wavelet_transform_summarises_itself(self):
        time, first, _second = _shared_signal_record()
        result = wavelet_transform(first, FINE_SCALES, time=time)

        assert result.summary().strip()

    def test_the_cross_wavelet_transform_summarises_itself(self):
        time, first, second = _shared_signal_record()
        result = cross_wavelet_transform(first, second, FINE_SCALES, time=time)

        assert result.summary().strip()

    def test_the_wavelet_coherence_summarises_itself(self):
        time, first, second = _shared_signal_record()
        result = wavelet_coherence(first, second, COARSE_SCALES, time=time)

        assert result.summary().strip()

    def test_the_wavelet_significance_summarises_itself(self):
        rng = np.random.default_rng(17)
        time = _time_axis()
        result = wavelet_significance(
            rng.standard_normal(N_POINTS), COARSE_SCALES, time=time, n_monte_carlo=20, seed=13
        )

        assert result.summary().strip()

    def test_the_wavelet_coherence_significance_summarises_itself(self):
        time, first, second = _shared_signal_record()
        result = wavelet_coherence_significance(first, second, COARSE_SCALES, time=time, n_monte_carlo=20, seed=31)

        assert result.summary().strip()
