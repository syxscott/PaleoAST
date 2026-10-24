# =============================================================================
# FILE: tests/golden/test_sliding_spectral_moments.py
# =============================================================================
"""
``stratigraphy.sliding_spectral_moments``（谱矩、滑窗谱矩、沉积速率反演、年龄模型）
的测试。

测试围绕五条主线组织，每条断言的依据都写在各自的 docstring 里：

* **谱矩的解析真值（最硬的一条）。** 功率谱在频带 ``[f1, f2]`` 上**均匀**时，
  谱矩有闭式解::

      M1/M0 = (f1 + f2) / 2
      M2/M0 = (f1^2 + f1 f2 + f2^2) / 3
      M4/M0 = (f1^4 + f1^3 f2 + f1^2 f2^2 + f1 f2^3 + f2^4) / 5

  这是把 ``M_j = ∫ f^j P(f) df`` 在常值功率上积分的结果，与实现无关。测试用
  数值积分构造这个解析谱，再断言模块算出的谱矩与之吻合（相对误差 < 1 %）。
  同一条真值还钉死了**频率轴的单位**（每单位坐标的循环频率）与**零频处理**
  （直流分量默认不计入谱矩）。

* **单频谱线。** 功率集中在一个频率 ``f0`` 时，平均波长必须约为 ``1/f0``、谱
  质心必须约为 ``f0``。这条把“平均周期”与“谱质心”两个字段的口径分开。

* **滑窗一致性。** 把记录拼成“前半段是 20 个周期（周期 20）、后半段是周期为
  5 的快速段”，滑窗谱矩在两段给出的**平均周期**必须分别接近 20 与 5
  （容差 10 %）。窗口必须足够长才能同时分辨这两种周期：前段的慢周期要求窗长
  ≥ 若干个慢周期，后段的快段因此必须含有足够多的快周期（只有 5 个快周期的
  短段无法被同一个窗口分辨，这一点由单独的“短记录”断言覆盖）。

* **沉积速率的闭合性。** Harker et al. (2013) 的关系是 ``λ = S · T_orb``，于是
  ``S = λ̂ / T_orb``。给定一个合成记录（波长 ``λ`` 已知）与目标周期 ``T``，
  反演必须还原出 ``S = λ / T``：这条断言把 docstring 里的关系式变成了可执行
  的契约。

* **年龄模型的闭式解。** 恒定速率下 ``age(z) = age_0 + (z - z_0) / S``；线性
  变速率下 ``age(z) = age_0 + ln(S(z)/S_0) / g``（ Kidder & Ledger 2018 的非线性
  映射）。测试与手算闭式解逐点比对，精确到 ``1e-9``。

边界测试断言抛 ``utils.exceptions.ValidationError``；可复现性断言固定种子给出
完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Harker, R. D., Turner, S. P., Ridgwell, A. & Lehman, B. (2013). Sedimentation
  rate from spectral analysis. *Geology*, 41(3), 307-310.（``λ = S · T_orb``：
  由观测波长与轨道周期反演沉积速率）
* Kidder, C. & Ledger, E. (2018). The importance of a nonlinear map for
  paleoclimatic sedimentary records. *Earth and Planetary Science Letters*, 481,
  111-119.（沉积速率随时间的非线性演化，深度—年龄映射因此是非线性的）
* Priestley, M. B. (1981). Spectral Analysis and Time Series. Academic Press.
  （谱矩 ``M_j = ∫ f^j P(f) df`` 与标准化谱形描述量的定义）
* Sinnesael, M., Zivanovic, M., De Vleeschouwer, D. & Claeys, P. (2018). Spectral
  Moments in Cyclostratigraphy: Advantages and Disadvantages compared to more
  classic Approaches. *Paleoceanography and Paleoclimatology*, 33, 493-510.
  （滑窗谱矩的用法与陷阱）
* Scargle, J. D. (1982). Studies in astronomical time series analysis.
  *IEEE Transactions on Information Theory*, 28, 513-523.（Lomb-Scargle 周期图；
  本模块的频谱走 :mod:`stratigraphy.spectral_analysis` 的同一条路线）
* Li, M., Hinnov, L. & Kump, L. (2019). Acycle: Time-series analysis software
  for paleoclimate research and education. *Computers & Geosciences*, 127,
  12-22.（Acycle 的 "spectral moments" 与 "sedimentation rate age model" 子功能）
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"sliding_spectral_moments", "sedimentation_rate_age_model"}


import numpy as np
import pytest

from stratigraphy.sliding_spectral_moments import (
    sedimentation_rate_age_model,
    sedimentation_rate_from_spectral,
    sliding_spectral_moments,
    spectral_moments,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 每单位坐标的采样点数。
SAMPLES_PER_UNIT = 10.0

#: 慢段的周期（坐标单位）。
SLOW_PERIOD = 20.0

#: 慢段的周期个数（“前半段 20 周期”）。
SLOW_CYCLES = 20.0

#: 快段的周期（坐标单位）。
FAST_PERIOD = 5.0

#: 快段的周期个数。快段必须含有足够多的快周期，滑窗才能在同一个窗口里同时
#: 分辨慢段的 20 周期与快段的 5 周期（窗长要盖住几个慢周期）。
FAST_CYCLES = 40.0

#: 滑窗的窗长（坐标单位）与步长（坐标单位）。
WINDOW_UNITS = 100.0
STEP_UNITS = 20.0

#: 滑窗断言的容差（相对误差）。
SLIDING_TOLERANCE = 0.10

#: 谱矩解析真值断言的容差（相对误差）。
MOMENT_TOLERANCE = 0.01

#: 闭合性断言的容差（相对误差）。
CLOSURE_TOLERANCE = 0.02

#: 频带解析真值用的频带边界（每单位坐标的循环频率）。
BAND_LOW = 0.01
BAND_HIGH = 0.50

#: 频带解析真值用的频点数（梯形积分的收敛）。
BAND_POINTS = 20001


# =============================================================================
# 辅助函数
# =============================================================================


def _uniform_spectrum(
    low: float = BAND_LOW,
    high: float = BAND_HIGH,
    n_points: int = BAND_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """频带 ``[low, high]`` 上功率恒定的解析功率谱。

    频率栅格取���用梯形积分，所以积分值与解析解的差是 ``O(h^2)``，在
    ``n_points`` 足够大时远小于 1 %。
    """
    axis = np.linspace(low, high, n_points)
    density = np.ones_like(axis)
    return axis, density


def _two_segment_record() -> tuple[np.ndarray, np.ndarray, float]:
    """前半段 20 个周期（周期 20）、后半段周期为 5 的快速段。

    Returns:
        (坐标轴, 记录值, 快段起点)。
    """
    slow_span = SLOW_CYCLES * SLOW_PERIOD
    fast_span = FAST_CYCLES * FAST_PERIOD
    slow_axis = np.arange(0.0, slow_span, 1.0 / SAMPLES_PER_UNIT)
    fast_axis = slow_span + np.arange(0.0, fast_span, 1.0 / SAMPLES_PER_UNIT)
    axis = np.concatenate([slow_axis, fast_axis])
    values = np.concatenate(
        [
            np.cos(2.0 * np.pi * slow_axis / SLOW_PERIOD),
            np.cos(2.0 * np.pi * (fast_axis - slow_span) / FAST_PERIOD),
        ]
    )
    return axis, values, slow_span


def _uniform_record(
    n_points: int = 4000,
    span: float = 200.0,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """一条带固定种子噪声的记录（可复现性断言用）。"""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, span, n_points)
    values = np.cos(2.0 * np.pi * axis / 20.0) + 0.05 * rng.standard_normal(n_points)
    return axis, values


# =============================================================================
# 谱矩：解析真值
# =============================================================================


class TestSpectralMomentsGroundTruth:
    """均匀谱的闭式解是谱矩最硬的真值。"""

    def test_uniform_band_matches_the_closed_form(self):
        """频带上的均匀谱必须给出闭式解的谱矩。

        闭式解（把 ``M_j = ∫ f^j P(f) df`` 在常值功率上积分）::

            M1/M0 = (f1 + f2) / 2
            M2/M0 = (f1^2 + f1 f2 + f2^2) / 3
            M4/M0 = (f1^4 + f1^3 f2 + f1^2 f2^2 + f1 f2^3 + f2^4) / 5
        """
        axis, density = _uniform_spectrum()
        result = spectral_moments(axis, density)

        expected_first = 0.5 * (BAND_LOW + BAND_HIGH)
        expected_second = (BAND_LOW**2 + BAND_LOW * BAND_HIGH + BAND_HIGH**2) / 3.0
        expected_fourth = (
            BAND_LOW**4 + BAND_LOW**3 * BAND_HIGH + BAND_LOW**2 * BAND_HIGH**2 + BAND_LOW * BAND_HIGH**3 + BAND_HIGH**4
        ) / 5.0

        assert result.m1 / result.m0 == pytest.approx(expected_first, rel=MOMENT_TOLERANCE), (
            f"M1/M0 is {result.m1 / result.m0:.6g}, expected {expected_first:.6g}"
        )
        assert result.m2 / result.m0 == pytest.approx(expected_second, rel=MOMENT_TOLERANCE), (
            f"M2/M0 is {result.m2 / result.m0:.6g}, expected {expected_second:.6g}"
        )
        assert result.m4 / result.m0 == pytest.approx(expected_fourth, rel=MOMENT_TOLERANCE), (
            f"M4/M0 is {result.m4 / result.m0:.6g}, expected {expected_fourth:.6g}"
        )

    def test_uniform_band_has_the_analytic_mean_period_and_bandwidth(self):
        """均匀谱的平均波长是质心频率的倒数，相对带宽有闭式解。

        均匀分布的方差是 ``(f2 - f1)^2 / 12``，因此
        ``bandwidth = (f2 - f1) / sqrt(12)``、``mean_period = 2 / (f1 + f2)``。
        """
        axis, density = _uniform_spectrum()
        result = spectral_moments(axis, density)

        expected_bandwidth = (BAND_HIGH - BAND_LOW) / np.sqrt(12.0)
        assert result.bandwidth == pytest.approx(expected_bandwidth, rel=MOMENT_TOLERANCE)
        assert result.relative_bandwidth == pytest.approx(
            expected_bandwidth / (0.5 * (BAND_LOW + BAND_HIGH)), rel=MOMENT_TOLERANCE
        )
        assert result.mean_period == pytest.approx(1.0 / result.central_frequency, rel=1e-12), (
            "the mean period must be the reciprocal of the spectral centroid"
        )

    def test_uniform_band_has_the_analytic_kurtosis(self):
        """均匀谱的无量纲峰度型量有闭式解。

            均匀分布的 ``E[f^4] / E[f^2]^2 = (1/5) / (1/3)^2 = 9/5``（在 ``[0, 1]``
        上；一般区间按同样的多项式比值计算）。
        """
        axis, density = _uniform_spectrum()
        result = spectral_moments(axis, density)

        expected = (
            (
                BAND_LOW**4
                + BAND_LOW**3 * BAND_HIGH
                + BAND_LOW**2 * BAND_HIGH**2
                + BAND_LOW * BAND_HIGH**3
                + BAND_HIGH**4
            )
            / 5.0
        ) / ((BAND_LOW**2 + BAND_LOW * BAND_HIGH + BAND_HIGH**2) / 3.0) ** 2
        assert result.spectral_kurtosis_normalized == pytest.approx(expected, rel=MOMENT_TOLERANCE), (
            f"the normalised kurtosis is {result.spectral_kurtosis_normalized:.6g}, expected {expected:.6g}"
        )

    def test_single_line_gives_the_line_period(self):
        """功率集中在单一频率 ``f0`` 时：平均波长 ≈ ``1/f0``、谱质心 ≈ ``f0``。"""
        centre = 0.25
        width = 0.002
        axis = np.linspace(1.0e-4, 1.0, 20001)
        density = np.exp(-0.5 * ((axis - centre) / width) ** 2)

        result = spectral_moments(axis, density)

        assert result.central_frequency == pytest.approx(centre, rel=1e-3), (
            f"the spectral centroid is {result.central_frequency:.6g}, expected {centre}"
        )
        assert result.mean_period == pytest.approx(1.0 / centre, rel=1e-3), (
            f"the mean period is {result.mean_period:.6g}, expected {1.0 / centre}"
        )
        assert result.bandwidth < 0.01 * centre, "a single line must have a narrow bandwidth"

    def test_zero_frequency_is_excluded_by_default(self):
        """直流分量默认不计入谱矩。

            FFT 型周期图的第一���频点装的是记录均值；把它算进 ``M0`` 会把谱质心
        拉向零频。默认 ``exclude_dc=True`` 时，直流频点的功率不得进入任何谱矩。
        """
        axis = np.concatenate([[0.0], np.linspace(0.1, 1.0, 5000)])
        density = np.concatenate([[1.0e6], np.ones(5000)])

        without_dc = spectral_moments(axis, density)
        with_dc = spectral_moments(axis, density, exclude_dc=False)

        # 去掉直流后，M0 只是非零频带上的积分：频带 [0.1, 1.0] 上功率恒为 1。
        assert without_dc.m0 == pytest.approx(0.9, rel=1e-9), (
            f"M0 is {without_dc.m0:.6g}, expected the integral over the non-DC band (0.9)"
        )
        assert with_dc.m0 > without_dc.m0, "keeping the DC bin must add its power to M0"
        # 直流的功率全落在零频上，会把谱质心拉向零频——这正是默认排除它的理由。
        assert with_dc.central_frequency < without_dc.central_frequency, (
            "the DC bin must drag the spectral centroid towards zero"
        )

    def test_moments_are_the_integrals_of_the_supplied_band(self):
        """``M0`` 必须是给定频带上的积分 ``∫ P df``。"""
        axis, density = _uniform_spectrum()
        result = spectral_moments(axis, density)

        assert result.m0 == pytest.approx(BAND_HIGH - BAND_LOW, rel=MOMENT_TOLERANCE)


# =============================================================================
# 滑窗谱矩
# =============================================================================


class TestSlidingSpectralMoments:
    """滑窗谱矩必须把两段的不同周期分辨出来。"""

    def test_mean_period_follows_the_two_segments(self):
        """两段的平均周期必须分别接近 20 与 5（容差 10 %）。"""
        axis, values, slow_span = _two_segment_record()
        result = sliding_spectral_moments(
            axis,
            values,
            window_duration=WINDOW_UNITS,
            step_duration=STEP_UNITS,
        )

        centres = result.window_centers
        slow = (centres > WINDOW_UNITS / 2.0) & (centres < slow_span - WINDOW_UNITS / 2.0)
        fast = (centres > slow_span + WINDOW_UNITS / 2.0) & (
            centres < slow_span + FAST_CYCLES * FAST_PERIOD - WINDOW_UNITS / 2.0
        )
        assert slow.sum() >= 3, "the slow segment must hold several windows"
        assert fast.sum() >= 3, "the fast segment must hold several windows"

        slow_period = float(np.median(result.mean_period[slow]))
        fast_period = float(np.median(result.mean_period[fast]))

        assert slow_period == pytest.approx(SLOW_PERIOD, rel=SLIDING_TOLERANCE), (
            f"the slow segment gives a mean period of {slow_period:.4g}, expected {SLOW_PERIOD}"
        )
        assert fast_period == pytest.approx(FAST_PERIOD, rel=SLIDING_TOLERANCE), (
            f"the fast segment gives a mean period of {fast_period:.4g}, expected {FAST_PERIOD}"
        )

    def test_peak_period_agrees_with_the_mean_period(self):
        """逐窗的谱峰周期必须与同一窗口的平均周期同量级。

        这是交叉检查：谱峰周期是“最强的那条谱线”，平均周期是“整条谱的质心”，
        两者在信号干净时必须一致到同一个量级。断言只针对**完全落在某一段内部**
        的窗口：跨段的窗口里同时有 20 周期与 5 周期，两者的量级本来就不同。
        """
        axis, values, slow_span = _two_segment_record()
        result = sliding_spectral_moments(
            axis,
            values,
            window_duration=WINDOW_UNITS,
            step_duration=STEP_UNITS,
        )

        centres = result.window_centers
        inside = (centres > WINDOW_UNITS / 2.0) & (centres < slow_span - WINDOW_UNITS / 2.0)
        inside |= (centres > slow_span + WINDOW_UNITS / 2.0) & (
            centres < slow_span + FAST_CYCLES * FAST_PERIOD - WINDOW_UNITS / 2.0
        )
        usable = inside & np.isfinite(result.peak_period)
        assert usable.sum() > 0
        ratio = result.mean_period[usable] / result.peak_period[usable]
        assert bool(np.all(ratio > 0.5)) and bool(np.all(ratio < 2.0)), (
            f"the mean period and the peak period disagree by more than a factor of two: {ratio}"
        )

    def test_window_centres_are_ordered_and_inside_the_record(self):
        """窗口中心必须严格递增并落在记录内部。"""
        axis, values, _slow_span = _two_segment_record()
        result = sliding_spectral_moments(
            axis,
            values,
            window_duration=WINDOW_UNITS,
            step_duration=STEP_UNITS,
        )

        assert result.window_centers.size > 1
        assert bool(np.all(np.diff(result.window_centers) > 0.0))
        assert float(result.window_centers[0]) > float(axis[0])
        assert float(result.window_centers[-1]) < float(axis[-1])
        assert bool(np.all(result.window_starts < result.window_centers))
        assert bool(np.all(result.window_ends > result.window_centers))

    def test_window_and_step_in_points_and_in_time_units_agree(self):
        """窗长用点数与用时长给出的是同一个窗口。

        以**时长**给出时，窗长按记录自身的平均采样间隔换算：间隔数四舍五入到
        最近整数，样本数再加一（Acycle 的 ``acycleSpectralMoments.m`` 是同一套
        换算口径），因此它与“点数”写法最多差一个样本。
        """
        axis, values, _slow_span = _two_segment_record()
        by_points = sliding_spectral_moments(
            axis,
            values,
            window=int(WINDOW_UNITS * SAMPLES_PER_UNIT),
            step=int(STEP_UNITS * SAMPLES_PER_UNIT),
        )
        by_time = sliding_spectral_moments(
            axis,
            values,
            window_duration=WINDOW_UNITS,
            step_duration=STEP_UNITS,
        )

        assert by_time.window_points == round(WINDOW_UNITS * SAMPLES_PER_UNIT) + 1
        assert abs(by_points.window_points - by_time.window_points) <= 1
        assert by_points.step_points == by_time.step_points
        common = min(by_points.mean_period.size, by_time.mean_period.size)
        assert np.allclose(
            by_points.mean_period[:common],
            by_time.mean_period[:common],
            rtol=0.05,
        )

    def test_window_count_follows_the_step(self):
        """窗口个数必须是 ``floor((n - W) / step) + 1``。"""
        axis, values, _slow_span = _two_segment_record()
        window_points = 500
        step_points = 50
        result = sliding_spectral_moments(axis, values, window=window_points, step=step_points)

        expected = (axis.size - window_points) // step_points + 1
        assert result.window_centers.size == expected
        assert result.window_points == window_points
        assert result.step_points == step_points

    def test_a_short_record_still_resolves_its_own_period(self):
        """只有 5 个快周期的短段：整条记录的谱质心仍必须给出周期 5。

        滑窗分辨不了这么短的段（窗长要同时盖住慢周期），但**整条记录**的谱矩
        可以：谱峰在 ``1/5``，把频带限制在该峰附近即可。
        """
        axis = np.arange(0.0, 5.0 * FAST_PERIOD, 1.0 / SAMPLES_PER_UNIT)
        values = np.cos(2.0 * np.pi * axis / FAST_PERIOD)

        result = spectral_moments(*_spectrum_in_band(axis, values, (0.1, 0.35)))
        assert result.mean_period == pytest.approx(FAST_PERIOD, rel=SLIDING_TOLERANCE), (
            f"the mean period of the five-cycle record is {result.mean_period:.4g}"
        )

    def test_n_windows_and_overlap_tile_the_record(self):
        """给出窗数与重叠比例时，窗口必须**恰好**是 ``n_windows`` 个。

        这是仓库其它滑窗模块（:mod:`stratigraphy.evolutionary`）的约定：窗长由
        窗数与重叠比例反解，窗口等距铺开。
        """
        axis, values, _slow_span = _two_segment_record()
        result = sliding_spectral_moments(axis, values, n_windows=8, overlap=0.5)

        assert result.window_centers.size == 8
        assert bool(np.all(np.diff(result.window_centers) > 0.0))
        assert float(result.window_centers[0]) > float(axis[0])
        assert float(result.window_centers[-1]) < float(axis[-1])

    def test_relative_sedimentation_rate_has_unit_mean(self):
        """带宽趋势反推的**相对**沉积速率必须均值为 1。

        Acycle 的 ``spectralmoments.m`` 给出的关系是
        ``sr = srmean · Bw(1) / Bwtrend`` 再平移使均值为 ``srmean``；本模块只有
        一条坐标轴、没有定年信息，``srmean`` 取参考值 1，因此输出是相对速率。
        """
        axis, values, _slow_span = _two_segment_record()
        result = sliding_spectral_moments(axis, values, n_windows=8, overlap=0.5)

        assert result.sedimentation_rate is not None, (
            "the bandwidth trend of this record is positive, so the rate must be defined"
        )
        assert float(np.mean(result.sedimentation_rate)) == pytest.approx(1.0, rel=1e-9)
        assert bool(np.all(result.sedimentation_rate > 0.0))

    def test_results_are_reproducible(self):
        """同一输入必须给出完全相同的结果。"""
        axis, values = _uniform_record()
        one = sliding_spectral_moments(axis, values, window=400, step=100)
        two = sliding_spectral_moments(axis, values, window=400, step=100)

        assert np.allclose(one.mean_period, two.mean_period)
        assert np.allclose(one.bandwidth, two.bandwidth)
        assert np.allclose(one.window_centers, two.window_centers)


def _spectrum_in_band(
    axis: np.ndarray,
    values: np.ndarray,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """记录在给定频带上的单边谱（走仓库既有的 Lomb-Scargle 路线）。"""
    from stratigraphy.spectral_analysis import SpectralAnalyzer

    spectrum = SpectralAnalyzer().analyze(
        axis,
        values,
        frequency_range=band,
        n_frequencies=500,
        ar1_significance=False,
    )
    return spectrum.frequencies, spectrum.power


# =============================================================================
# 沉积速率反演
# =============================================================================


class TestSedimentationRateFromSpectral:
    """Harker et al. (2013) 的关系必须闭合。"""

    def test_the_inverted_rate_recovers_the_sedimentation_rate(self):
        """给定波长 ``λ`` 与目标周期 ``T``，反演必须还原 ``S = λ / T``。

        合成记录是在深度坐标上以波长 ``λ = S · T`` 采样的余弦，因此它的谱质心
        波长就是 ``λ``；Harker 的关系 ``λ = S · T_orb`` 反解出 ``S = λ / T_orb``。
        """
        sedimentation_rate = 0.1
        forcing_period = 20.0
        wavelength = sedimentation_rate * forcing_period

        axis = np.linspace(0.0, 100.0 * wavelength, 4000)
        values = np.cos(2.0 * np.pi * axis / wavelength)

        result = sedimentation_rate_from_spectral(axis, values, [forcing_period])

        assert result.moments.mean_period == pytest.approx(wavelength, rel=CLOSURE_TOLERANCE), (
            f"the observed mean period is {result.moments.mean_period:.6g}, expected {wavelength:.6g}"
        )
        assert result.sedimentation_rates[0] == pytest.approx(sedimentation_rate, rel=CLOSURE_TOLERANCE), (
            f"the inverted rate is {result.sedimentation_rates[0]:.6g}, expected {sedimentation_rate}"
        )

    def test_rates_scale_inversely_with_the_target_period(self):
        """``S ∝ 1/T_orb``：目标周期加倍，速率减半。"""
        axis = np.linspace(0.0, 200.0, 4000)
        values = np.cos(2.0 * np.pi * axis / 2.0)

        result = sedimentation_rate_from_spectral(axis, values, [19.0, 21.0, 23.0, 41.0])

        assert result.target_periods.size == 4
        ratios = result.sedimentation_rates * result.target_periods
        assert bool(np.allclose(ratios, ratios[0])), (
            f"the product S * T must be constant across the target periods, got {ratios}"
        )
        assert ratios[0] == pytest.approx(result.moments.mean_period, rel=1e-12)

    def test_uncertainties_are_positive_and_finite(self):
        """不确定度必须是正的有限数。"""
        axis, values, _slow_span = _two_segment_record()
        result = sedimentation_rate_from_spectral(axis, values)

        assert result.uncertainties.size == result.sedimentation_rates.size
        assert bool(np.all(np.isfinite(result.uncertainties)))
        assert bool(np.all(result.uncertainties > 0.0))
        assert result.n_cycles > 1.0

    def test_the_amplitude_relation_gives_a_second_estimate(self):
        """振幅关系（箱式滤波衰减）给出第二条、独立的速率估计。

        合成记录是**时间域**余弦在深度上的箱式平均：每个样品代表的时间长度
        是 ``Δt = Δz / S``，于是观测振幅被 ``|sin(pi x)/(pi x)|``（``x = Δt / T``）
        衰减。由观测振幅反解 ``x`` 再反解平均波长，得到的速率必须与真值一致
        到百分之几十以内（这条关系的灵敏度本身就有限）。
        """
        forcing_amplitude = 1.0
        forcing_period = 20.0
        sedimentation_rate = 0.1
        depth_spacing = 0.25

        time_step = depth_spacing / sedimentation_rate
        sample_times = np.arange(0.0, 400.0 * time_step, time_step)
        depths = sample_times * sedimentation_rate
        attenuation = np.sin(np.pi * time_step / forcing_period) / (np.pi * time_step / forcing_period)
        values = forcing_amplitude * attenuation * np.cos(2.0 * np.pi * sample_times / forcing_period)

        result = sedimentation_rate_from_spectral(
            depths,
            values,
            [forcing_period],
            forcing_amplitude=forcing_amplitude,
        )

        # 波长关系：深度域的波长是 S · T = 2。
        assert result.moments.mean_period == pytest.approx(sedimentation_rate * forcing_period, rel=CLOSURE_TOLERANCE)
        assert bool(np.all(np.isfinite(result.amplitude_rates))), (
            "the amplitude relation must give a finite rate estimate"
        )
        assert result.amplitude_rates[0] == pytest.approx(sedimentation_rate, rel=0.05), (
            f"the amplitude-based rate is {result.amplitude_rates[0]:.6g}, expected {sedimentation_rate}"
        )


# =============================================================================
# 年龄-深度模型
# =============================================================================


class TestSedimentationRateAgeModel:
    """年龄模型必须与 ``age(z) = ∫ dz / S(z)`` 的闭式解一致。"""

    def test_constant_rate_matches_the_closed_form(self):
        """恒定速率：``age(z) = age_0 + (z - z_0) / S``（精确到 1e-9）。"""
        depths = np.linspace(0.0, 10.0, 11)
        rate = 0.5

        model = sedimentation_rate_age_model(depths, rate, age_start=3.0)

        expected = 3.0 + (depths - depths[0]) / rate
        assert np.allclose(model.age, expected, rtol=0.0, atol=1e-9)
        assert model.model == "constant"

    def test_linear_rate_matches_the_closed_form(self):
        """线性变速率：``age(z) = age_0 + ln(S(z) / S_0) / g``（精确到 1e-9）。

        这是 Kidder & Ledger (2018) 的非线性深度—年龄映射：速率随深度线性变化
        时，年龄是深度的**对数**函数，而不是线性函数。
        """
        depths = np.linspace(0.0, 10.0, 11)
        base_rate = 1.0
        gradient = 0.1

        model = sedimentation_rate_age_model(
            depths,
            base_rate,
            age_start=0.0,
            model="linear",
            rate_gradient=gradient,
        )

        rates = base_rate + gradient * (depths - depths[0])
        expected = np.log(rates / base_rate) / gradient
        assert np.allclose(model.age, expected, rtol=0.0, atol=1e-9)
        assert model.model == "linear"

    def test_piecewise_constant_rate_uses_the_left_endpoint(self):
        """分段常数速率：区间用左端点的速率积分。"""
        depths = np.array([0.0, 1.0, 2.0, 4.0])
        rates = np.array([1.0, 2.0, 4.0, 8.0])

        model = sedimentation_rate_age_model(depths, rates)

        expected = np.array([0.0, 1.0, 1.5, 2.0])
        assert np.allclose(model.age, expected, rtol=0.0, atol=1e-12)
        assert model.model == "piecewise_constant"

    def test_decreasing_depth_axis_is_accepted(self):
        """深度递减时年龄同样递减，模型按输入顺序返回。"""
        depths = np.linspace(10.0, 0.0, 11)
        model = sedimentation_rate_age_model(depths, 0.5)

        assert bool(np.all(np.diff(model.age) < 0.0))
        assert float(model.age[-1]) == pytest.approx(0.0, abs=1e-12)

    def test_the_piecewise_rate_must_stay_positive(self):
        """速率必须为正。"""
        depths = np.linspace(0.0, 10.0, 5)
        with pytest.raises(ValidationError):
            sedimentation_rate_age_model(depths, np.zeros_like(depths))


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    def test_window_points_that_are_too_few_are_rejected(self):
        """窗口点数太少（少于 4 个点）必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, values, window=3)

    def test_window_longer_than_the_record_is_rejected(self):
        """窗口比记录还长必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, values, window=axis.size + 10)

    def test_window_duration_longer_than_the_record_is_rejected(self):
        """以时长给出的窗口比记录还长必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, values, window_duration=float(axis[-1] - axis[0]) * 2.0)

    def test_window_and_window_duration_are_mutually_exclusive(self):
        """两个窗口参数不能同时给。"""
        axis, values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, values, window=100, window_duration=10.0)

    def test_a_missing_window_is_rejected(self):
        """既不给点数也不给时长必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, values)

    def test_non_finite_values_are_rejected(self):
        """含 NaN 的记录必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        broken = values.copy()
        broken[10] = np.nan
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, broken, window=500)

    def test_a_non_increasing_time_axis_is_rejected(self):
        """时间轴非严格递增必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        broken = axis.copy()
        broken[10] = broken[11]
        with pytest.raises(ValidationError):
            sliding_spectral_moments(broken, values, window=500)

    def test_mismatched_lengths_are_rejected(self):
        """长度不一致必须报错。"""
        axis, values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, values[:-1], window=500)

    def test_a_constant_record_is_rejected(self):
        """常数序列没有可分析的谱，必须报错而不是给出垃圾结果。"""
        axis, _values, _slow_span = _two_segment_record()
        with pytest.raises(ValidationError):
            sliding_spectral_moments(axis, np.ones_like(axis), window=500)

    def test_non_monotonic_depth_is_rejected(self):
        """深度轴非单调必须报错。"""
        depths = np.array([0.0, 1.0, 2.0, 1.5, 4.0])
        with pytest.raises(ValidationError):
            sedimentation_rate_age_model(depths, 0.5)

    def test_invalid_target_periods_are_rejected(self):
        """目标周期非正或非有限必须报错。"""
        axis, values = _uniform_record()
        with pytest.raises(ValidationError):
            sedimentation_rate_from_spectral(axis, values, [0.0])
        with pytest.raises(ValidationError):
            sedimentation_rate_from_spectral(axis, values, [-19.0])
        with pytest.raises(ValidationError):
            sedimentation_rate_from_spectral(axis, values, [np.nan])

    def test_invalid_spectra_are_rejected(self):
        """频率轴与功率非法时必须报错。"""
        axis = np.linspace(0.01, 0.5, 100)
        with pytest.raises(ValidationError):
            spectral_moments(axis, np.ones_like(axis)[:-1])
        with pytest.raises(ValidationError):
            spectral_moments(axis, -np.ones_like(axis))
        with pytest.raises(ValidationError):
            spectral_moments(axis[::-1], np.ones_like(axis))
        with pytest.raises(ValidationError):
            spectral_moments(axis, np.zeros_like(axis))
