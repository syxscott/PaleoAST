# =============================================================================
# FILE: tests/golden/test_coco.py
# =============================================================================
"""
``stratigraphy.coco``（COCO / eCOCO 地层年代学）的测试。

测试围绕三条主线组织：

* **合成真值。** 用一个**已知的沉积速率**记录一段信号, 再让 COCO 反演速率。
  判据刻意采用稳健形式 —— 真值处的相关系数不低于最高相关性的 80 %,
  且真值落在得分最高点附近 —— 而**不是**断言 ``best_rate`` 精确等于真值：
  速率栅格是离散的, 相关系数曲面本身也有平台, 精确相等不是任何插值/搜索
  方法能保证的性质。两条真值记录分别是：

  1. *纯 20 kyr 岁差信号* + 单带目标谱。Berger (1989) 的现代周期表里
     岁差族是 17/19/23 kyr, 并不含 20 kyr, 所以这条用例通过
     ``target_spectrum`` 参数自带一条以 20 kyr (50 cycles/Ma) 为中心的
     频带, 从而只检验"相关系数极大值是否落在真值速率上"这一件事；
  2. *Berger (1989) 谱形信号* + 模块默认目标谱, 检验默认路径。

* **频率轴口径。** 目标谱的频率必须是"每单位时间"(cycles/Ma), 而不是
  "每样本"。因此测试用已知的真值速率算出天文时间轴的 Rayleigh 分辨率与
  Nyquist 频率, 断言目标谱频率落在两者之间, 并且恰好落在 Berger (1989)
  的轨道周期频率 (1000 / 周期_kyr) 上。

* **性质而非冻结数字。** eCOCO 只断言"首个窗口≈前半段速率、末个窗口≈
  后半段速率"（跨越速率突变点的窗口必然给出过渡值, 不作断言）; 边界测试
  断言抛 ``utils.exceptions.ValidationError``; 可复现性断言固定种子给出
  完全相同的结果。

参考来源（断言中引用的数值出处）
--------------------------------
* Li, M., Kump, L.R., Hinnov, L.A., Mann, M.E., 2018. Cyclostratigraphic
  correlation of astronomical forcing and Earth system processes. *Earth and
  Planetary Science Letters*, 501: 165-179. (COCO / eCOCO 的方法本身。)
* Berger, A., 1978. Long-term climatic variation: astronomical theory.
  In Understanding the Climate. Academic Press. (现代轨道周期族：
  偏心率 ~100 kyr、倾角 ~41 kyr、岁差 ~21 kyr。)
* Berger, A., 1989. Milankovitch cycles in Earth and related cycles.
  In *Geology of Hungary*, 23-46. (现代（0 Ma）轨道周期表：
  405/123/95、41/39/37、23/19/17 kyr。)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"coco", "ecoco"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.coco import (
    berger1989_periods,
    coco,
    ecoco,
    milankovitch_target_spectrum,
)
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的真值沉积速率（坐标单位/Ma）。
TRUE_RATE = 4.0

#: 合成记录覆盖的天文时间跨度 (Ma)。
TRUE_SPAN_MA = 1.0

#: 合成记录的采样点数。
N_POINTS = 2000

#: 20 kyr 岁差信号的自相关周期 (Ma)。
PRECESSION_MA = 0.020

#: 20 kyr 对应的频率 (cycles/Ma)。
PRECESSION_FREQUENCY = 1.0 / PRECESSION_MA


# =============================================================================
# 辅助函数
# =============================================================================


def _single_band_target(
    centre: float = PRECESSION_FREQUENCY,
    half_width: float = 8.0,
    taper: float = 4.0,
    low: float = 5.0,
    high: float = 95.0,
) -> tuple[npt.NDArray, npt.NDArray]:
    """构造一条单频带目标谱 ``(频率, 功率)``。

    频率单位是 cycles/Ma。梯形频带在 ``centre`` 处取满值, 到
    ``centre ± half_width`` 处线性降到 0。
    """
    frequencies = np.linspace(low, high, 400)
    power = np.clip((half_width - np.abs(frequencies - centre)) / taper, 0.0, 1.0)
    return frequencies, power


def _precession_record(
    rate: float = TRUE_RATE,
    span_ma: float = TRUE_SPAN_MA,
    n_points: int = N_POINTS,
    noise: float = 0.05,
    seed: int = 3,
) -> tuple[npt.NDArray, npt.NDArray]:
    """按已知速率记录的纯 20 kyr 岁差信号。

    Returns:
        (地层坐标轴, 记录数值)。坐标轴单位是"坐标单位", 速率是
        "坐标单位/Ma", 所以坐标 = 速率 × 天文时间。
    """
    rng = np.random.default_rng(seed)
    astro_time = np.linspace(0.0, span_ma, n_points)
    values = np.sin(2.0 * np.pi * astro_time / PRECESSION_MA)
    values += noise * rng.standard_normal(n_points)
    return rate * astro_time, values


def _orbital_record(
    rate: float = TRUE_RATE,
    span_ma: float = TRUE_SPAN_MA,
    n_points: int = N_POINTS,
    noise: float = 0.02,
    seed: int = 7,
) -> tuple[npt.NDArray, npt.NDArray]:
    """按已知速率记录的、谱形与 Berger (1989) 目标谱一致的信号。

    信号包含偏心率（123/95 kyr）、倾角（41/39/37 kyr）与岁差
    （23/19/17 kyr）三个族, 权重与模块目标谱的 ``TARGET_BAND_WEIGHTS``
    一致。
    """
    rng = np.random.default_rng(seed)
    astro_time = np.linspace(0.0, span_ma, n_points)
    values = np.zeros(n_points, dtype=float)
    for period_kyr, amplitude in (
        (123.0, 1.0),
        (95.0, 1.0),
        (41.0, 0.6),
        (39.0, 0.6),
        (37.0, 0.6),
        (23.0, 0.4),
        (19.0, 0.4),
        (17.0, 0.4),
    ):
        values += amplitude * np.sin(2.0 * np.pi * astro_time / (period_kyr / 1000.0))
    values += noise * rng.standard_normal(n_points)
    return rate * astro_time, values


def _two_rate_record(
    rate_first: float,
    rate_second: float,
    span_ma: float = TRUE_SPAN_MA,
    n_points: int = 4000,
    seed: int = 11,
) -> tuple[npt.NDArray, npt.NDArray]:
    """前半段速率 ``rate_first``、后半段速率 ``rate_second`` 的记录。

    坐标轴在时间中点处速率改变, 因此坐标轴是**非等间距**的：前半段
    ``x = r1 * t``, 后半段 ``x = r1 * T/2 + r2 * (t - T/2)``。
    """
    rng = np.random.default_rng(seed)
    astro_time = np.linspace(0.0, span_ma, n_points)
    values = np.zeros(n_points, dtype=float)
    for period_kyr, amplitude in (
        (123.0, 1.0),
        (95.0, 1.0),
        (41.0, 0.6),
        (39.0, 0.6),
        (37.0, 0.6),
        (23.0, 0.4),
        (19.0, 0.4),
        (17.0, 0.4),
    ):
        values += amplitude * np.sin(2.0 * np.pi * astro_time / (period_kyr / 1000.0))
    values += 0.02 * rng.standard_normal(n_points)

    midpoint = 0.5 * span_ma
    coordinate = np.where(
        astro_time <= midpoint,
        rate_first * astro_time,
        rate_first * midpoint + rate_second * (astro_time - midpoint),
    )
    return coordinate, values


def _correlation_at(result, rate: float) -> float:
    """扫描结果中给定速率处的相关系数。"""
    index = int(np.argmin(np.abs(result.rates - rate)))
    return float(result.correlations[index])


# =============================================================================
# COCO：合成真值
# =============================================================================


class TestCocoGroundTruth:
    """COCO 必须把速率反演回真值附近。"""

    def test_correlation_peaks_at_the_true_rate(self):
        """纯 20 kyr 信号：真值处的相关系数必须显著高于其它速率。

        判据（稳健形式）：真值处的相关系数 ≥ 最高相关性的 80 %,
        且真值落在得分最高点的邻域内（±15 %）。
        """
        coordinate, values = _precession_record()
        frequencies, target = _single_band_target()

        result = coco(
            coordinate,
            values,
            rate_range=(0.8 * TRUE_RATE, 1.2 * TRUE_RATE),
            n_rates=100,
            target_spectrum=(frequencies, target),
            n_monte_carlo=0,
            random_seed=3,
        )

        best_correlation = float(np.nanmax(result.correlations))
        truth_correlation = _correlation_at(result, TRUE_RATE)
        assert truth_correlation >= 0.8 * best_correlation, (
            f"the correlation at the true rate ({truth_correlation:.4f}) is far below the "
            f"best correlation ({best_correlation:.4f})"
        )
        assert abs(result.best_rate - TRUE_RATE) <= 0.15 * TRUE_RATE, (
            f"the best rate {result.best_rate:.4f} is not in the neighbourhood of the true rate {TRUE_RATE}"
        )

    def test_default_target_recovers_the_true_rate(self):
        """默认（Berger 1989）目标谱：Berger 谱形信号的真值速率必须被找回。"""
        coordinate, values = _orbital_record()

        result = coco(
            coordinate,
            values,
            rate_range=(0.85 * TRUE_RATE, 1.15 * TRUE_RATE),
            n_rates=100,
            n_monte_carlo=0,
            random_seed=7,
        )

        best_correlation = float(np.nanmax(result.correlations))
        truth_correlation = _correlation_at(result, TRUE_RATE)
        assert truth_correlation >= 0.8 * best_correlation, (
            f"the correlation at the true rate ({truth_correlation:.4f}) is far below the "
            f"best correlation ({best_correlation:.4f})"
        )
        assert abs(result.best_rate - TRUE_RATE) <= 0.1 * TRUE_RATE, (
            f"the best rate {result.best_rate:.4f} is far from the true rate {TRUE_RATE}"
        )

    def test_correlation_curve_is_discriminating(self):
        """相关系数曲线必须随速率变化, 否则上面的断言没有意义。"""
        coordinate, values = _orbital_record()

        result = coco(
            coordinate,
            values,
            rate_range=(0.85 * TRUE_RATE, 1.15 * TRUE_RATE),
            n_rates=100,
            n_monte_carlo=0,
            random_seed=7,
        )

        correlations = result.correlations
        assert np.all(np.isfinite(correlations)), "some candidate rates were skipped"
        assert float(np.ptp(correlations)) > 0.2, "the correlation curve is too flat to be meaningful"
        assert float(np.max(correlations)) > 0.3, "the record does not correlate with the target spectrum"

    def test_best_rate_is_significant(self):
        """最优速率处的蒙特卡洛 p 值必须足够小。"""
        coordinate, values = _orbital_record()

        result = coco(
            coordinate,
            values,
            rate_range=(0.85 * TRUE_RATE, 1.15 * TRUE_RATE),
            n_rates=60,
            n_monte_carlo=100,
            random_seed=7,
        )

        assert 0.0 < result.best_p_value <= 0.05, f"the best rate is not significant: p = {result.best_p_value:.4g}"
        assert np.all(np.isfinite(result.p_values)), "some candidate rates have no p-value"

    def test_rate_confidence_interval_contains_the_truth(self):
        """最优速率的不确定度区间必须包含真值。"""
        coordinate, values = _orbital_record()

        result = coco(
            coordinate,
            values,
            rate_range=(0.85 * TRUE_RATE, 1.15 * TRUE_RATE),
            n_rates=100,
            n_monte_carlo=100,
            random_seed=7,
        )

        assert result.rate_confidence is not None, "the confidence interval was not determined"
        low, high = result.rate_confidence
        assert low <= TRUE_RATE <= high, (
            f"the confidence interval [{low:.4f}, {high:.4f}] does not contain the true rate {TRUE_RATE}"
        )


# =============================================================================
# 目标谱与频率轴口径
# =============================================================================


class TestTargetSpectrum:
    """目标谱的频率轴必须是"每单位时间", 且落在已公布的轨道周期上。"""

    def test_target_frequencies_are_bounded_by_the_astro_nyquist(self):
        """目标谱频率必须落在天文时间轴的 Rayleigh 分辨率与 Nyquist 之间。

        Rayleigh 分辨率 ``1/T`` 与 Nyquist 频率 ``1/(2 dt)`` 都由**已知
        的真值速率**算出：``T = X / r``, ``dt = dx / r``。因此这条断言
        同时钉死了频率轴的单位 —— 若是"每样本", 频率不可能超过 0.5。
        """
        coordinate, values = _orbital_record()

        result = coco(
            coordinate,
            values,
            rate_range=(0.85 * TRUE_RATE, 1.15 * TRUE_RATE),
            n_rates=40,
            n_monte_carlo=0,
            random_seed=7,
        )

        dx = float(np.median(np.diff(coordinate)))
        span_ma = float(coordinate[-1] - coordinate[0]) / TRUE_RATE
        rayleigh = 1.0 / span_ma
        nyquist = TRUE_RATE / (2.0 * dx)

        frequencies = result.target_frequencies
        assert frequencies.size > 0
        assert float(np.min(frequencies)) >= rayleigh - 1e-9, (
            "the target spectrum reaches below the Rayleigh resolution of the astro time axis"
        )
        assert float(np.max(frequencies)) <= nyquist + 1e-9, (
            "the target spectrum reaches above the Nyquist frequency of the astro time axis"
        )
        # "每单位时间" 的量级：轨道频率在 2–60 cycles/Ma 之间。
        assert float(np.max(frequencies)) > 10.0, "the frequency axis is not expressed per unit time"

    def test_target_spectrum_is_anchored_on_the_published_orbital_periods(self):
        """目标谱必须在 Berger (1989) 的轨道频率 (1000 / 周期_kyr) 上非零。"""
        coordinate, values = _orbital_record()

        result = coco(
            coordinate,
            values,
            rate_range=(0.85 * TRUE_RATE, 1.15 * TRUE_RATE),
            n_rates=40,
            n_monte_carlo=0,
            random_seed=7,
        )

        frequencies = result.target_frequencies
        spectrum = result.target_spectrum
        checked = 0
        for period_kyr in berger1989_periods().values():
            centre = 1000.0 / float(period_kyr)
            match = np.isclose(frequencies, centre, rtol=0.01)
            if not np.any(match):
                continue  # 该周期带落在记录分辨率之外, 不作要求
            checked += 1
            assert np.all(spectrum[match] > 0.0), (
                f"the target spectrum is zero at the {period_kyr} kyr band centre ({centre:.3f} cycles/Ma)"
            )
        assert checked >= 5, "too few orbital bands were inside the comparison band to check"

    def test_milankovitch_target_spectrum_is_a_function_of_the_given_frequencies(self):
        """目标谱必须逐点对应给定的频率, 而不是另起一套栅格。"""
        frequencies = np.array([10.0, 25.0, 50.0])
        spectrum = milankovitch_target_spectrum(frequencies)

        assert spectrum.shape == frequencies.shape
        assert np.all(spectrum >= 0.0)
        # 50 cycles/Ma 落在 19 kyr 岁差带内, 10 cycles/Ma 落在 95 kyr
        # 偏心率带内 —— 两处都必须非零。
        assert spectrum[2] > 0.0
        assert spectrum[0] > 0.0

    def test_berger1989_periods_contains_the_standard_periods(self):
        """周期表必须含有 100 / 41 / 21 kyr 附近的周期（容差 ±30 %）。"""
        periods = berger1989_periods()
        values = list(periods.values())
        for expected in (100.0, 41.0, 21.0):
            assert any(abs(value - expected) / expected <= 0.30 for value in values), (
                f"no period within 30 % of {expected} kyr in {periods}"
            )

    def test_berger1989_periods_returns_a_copy(self):
        periods = berger1989_periods()
        periods["obliquity_41kyr"] = -1.0
        assert berger1989_periods()["obliquity_41kyr"] == 41.0


# =============================================================================
# eCOCO：变化的沉积速率
# =============================================================================


class TestECoco:
    """eCOCO 必须逐窗给出局部沉积速率。"""

    def test_two_rate_record_recovers_both_rates(self):
        """前半段速率 r1、后半段速率 r2：首窗≈r1, 末窗≈r2。

        只有完全落在某一段速率内的窗口才作断言 —— 跨越速率突变点的窗口
        必然给出过渡值, 对它断言任何具体速率都是错的。
        """
        rate_first = 4.0
        rate_second = 8.0
        coordinate, values = _two_rate_record(rate_first, rate_second)

        result = ecoco(
            coordinate,
            values,
            rate_range=(0.6 * rate_first, 1.6 * rate_second),
            n_windows=4,
            n_rates=60,
            n_monte_carlo=0,
            random_seed=11,
        )

        assert result.best_rates.shape == (4,)
        assert result.correlations.shape == (4,)
        assert np.all(np.isfinite(result.best_rates)), "some windows were skipped"

        assert abs(result.best_rates[0] - rate_first) <= 0.2 * rate_first, (
            f"the first window gives {result.best_rates[0]:.4f}, expected ~{rate_first}"
        )
        assert abs(result.best_rates[-1] - rate_second) <= 0.2 * rate_second, (
            f"the last window gives {result.best_rates[-1]:.4f}, expected ~{rate_second}"
        )
        assert result.best_rates[-1] > 1.5 * result.best_rates[0], "the rate did not increase across the rate change"

    def test_window_centres_are_ordered(self):
        """窗口中心必须沿坐标轴递增。"""
        coordinate, values = _two_rate_record(4.0, 8.0)

        result = ecoco(
            coordinate,
            values,
            rate_range=(2.4, 12.8),
            n_windows=4,
            n_rates=40,
            n_monte_carlo=0,
            random_seed=11,
        )

        assert result.window_centres.shape == (4,)
        assert np.all(np.diff(result.window_centres) > 0)


# =============================================================================
# 边界
# =============================================================================


class TestBoundaries:
    """非法输入必须抛 ``ValidationError``。"""

    @pytest.mark.parametrize(
        "rate_range",
        [(0.0, 4.0), (-1.0, 4.0), (4.0, 1.0), (4.0, 4.0)],
    )
    def test_invalid_rate_range_is_rejected(self, rate_range: tuple[float, float]):
        coordinate, values = _precession_record()
        with pytest.raises(ValidationError):
            coco(coordinate, values, rate_range=rate_range, n_rates=10, n_monte_carlo=0, random_seed=1)

    def test_too_few_points_are_rejected(self):
        coordinate = np.linspace(0.0, 1.0, 10)
        values = np.sin(coordinate)
        with pytest.raises(ValidationError):
            coco(coordinate, values, rate_range=(1.0, 4.0), n_rates=10, n_monte_carlo=0, random_seed=1)

    def test_non_monotonic_time_axis_is_rejected(self):
        coordinate, values = _precession_record()
        broken = coordinate.copy()
        broken[10] = broken[11]
        with pytest.raises(ValidationError):
            coco(broken, values, rate_range=(1.0, 8.0), n_rates=10, n_monte_carlo=0, random_seed=1)

    def test_mismatched_lengths_are_rejected(self):
        coordinate, values = _precession_record()
        with pytest.raises(ValidationError):
            coco(coordinate, values[:-1], rate_range=(1.0, 8.0), n_rates=10, n_monte_carlo=0, random_seed=1)

    def test_constant_values_are_rejected(self):
        coordinate, _ = _precession_record()
        values = np.zeros_like(coordinate)
        with pytest.raises(ValidationError):
            coco(coordinate, values, rate_range=(1.0, 8.0), n_rates=10, n_monte_carlo=0, random_seed=1)

    def test_too_few_candidate_rates_are_rejected(self):
        coordinate, values = _precession_record()
        with pytest.raises(ValidationError):
            coco(coordinate, values, rate_range=(1.0, 8.0), n_rates=1, n_monte_carlo=0, random_seed=1)

    def test_no_admissible_rate_is_rejected(self):
        """速率区间全部不可用（记录太短, 分辨不出任何轨道带）时必须报错。"""
        coordinate, values = _precession_record()
        with pytest.raises(ValidationError):
            coco(
                coordinate,
                values,
                rate_range=(500.0, 900.0),
                n_rates=10,
                n_monte_carlo=0,
                random_seed=1,
            )

    def test_ecoco_rejects_too_few_windows(self):
        coordinate, values = _precession_record()
        with pytest.raises(ValidationError):
            ecoco(coordinate, values, rate_range=(1.0, 8.0), n_windows=1, n_rates=10, n_monte_carlo=0)

    def test_milankovitch_target_spectrum_rejects_invalid_frequencies(self):
        with pytest.raises(ValidationError):
            milankovitch_target_spectrum(np.array([]))
        with pytest.raises(ValidationError):
            milankovitch_target_spectrum(np.array([-1.0, 10.0]))


# =============================================================================
# 可复现性
# =============================================================================


class TestReproducibility:
    """固定种子必须给出完全相同的结果。"""

    def test_coco_is_reproducible(self):
        coordinate, values = _orbital_record()

        first = coco(
            coordinate,
            values,
            rate_range=(3.4, 4.6),
            n_rates=40,
            n_monte_carlo=50,
            random_seed=42,
        )
        second = coco(
            coordinate,
            values,
            rate_range=(3.4, 4.6),
            n_rates=40,
            n_monte_carlo=50,
            random_seed=42,
        )

        assert np.array_equal(first.correlations, second.correlations)
        assert np.array_equal(first.p_values, second.p_values)
        assert first.best_rate == second.best_rate
        assert first.best_correlation == second.best_correlation
        assert first.best_p_value == second.best_p_value

    def test_ecoco_is_reproducible(self):
        coordinate, values = _two_rate_record(4.0, 8.0)

        first = ecoco(
            coordinate,
            values,
            rate_range=(2.4, 12.8),
            n_windows=4,
            n_rates=30,
            n_monte_carlo=30,
            random_seed=42,
        )
        second = ecoco(
            coordinate,
            values,
            rate_range=(2.4, 12.8),
            n_windows=4,
            n_rates=30,
            n_monte_carlo=30,
            random_seed=42,
        )

        assert np.array_equal(first.best_rates, second.best_rates)
        assert np.array_equal(first.correlations, second.correlations)
        assert np.array_equal(first.window_centres, second.window_centres)
