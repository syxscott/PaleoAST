# =============================================================================
# FILE: tests/property/test_evolutionary_properties.py
# =============================================================================
"""
Tests for the evolutionary spectrum in ``stratigraphy.evolutionary``.

Every numeric expectation here is either

* a property of the *input* -- a sinusoid of a known period has that period in
  its spectrum, and a record whose first half oscillates with period 20 and
  whose second half oscillates with period 50 must report 20 and 50 in the two
  windows that cover those halves;
* a property of the *frequency grid*, which the module promises to derive from
  the **record length** (the Rayleigh resolution of the sliding window) and not
  from the number of samples; or
* a property of the *error contract*: a window that is longer than the record,
  a non-positive window, a non-positive frequency count, an unknown estimator
  and a series with no oscillation at all are all rejected with
  ``DataValidationError`` (``utils.exceptions.ValidationError`` is an alias of
  that class).

The test with teeth is ``test_frequency_grid_does_not_drift_with_sampling_density``.
The module used to build its frequency grid from the *sampling density*, so
sampling the same record more densely made the reported period worse: a period
of 7.0 was reported as 9.89 once the record was sampled at 20000 points.  The
tests below pin the fix: the grid step is the Rayleigh resolution of the window
divided by the zero-padding factor, and it does not move when the same record is
sampled ten times more densely.

What is deliberately NOT asserted: a number obtained by running this module and
freezing the output.  The tolerances are derived from the Rayleigh resolution
of the window instead.

References cited in the assertions:
    Press, W.H. et al. 1992. Numerical Recipes in C, 2nd ed. (Rayleigh
    resolution of a finite record.)
    Horne, J.H. & Baliunas, S.L. 1986. Astrophys. J. 302: 755-763.
    (Lomb-Scargle periodogram.)
    Thomson, D.J. 1982. Proc. IEEE 70: 1055-1096. (Multitaper method.)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"evolutionary_lomb_scargle", "evolutionary_multitaper", "evolutionary_spectrum"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.evolutionary import (
    evolutionary_lomb_scargle,
    evolutionary_multitaper,
    evolutionary_spectrum,
)
from utils.exceptions import DataValidationError

#: 记录时长（时间单位）。
SPAN = 100.0

#: 测试信号的周期（时间单位）。
PERIOD = 5.0

#: 滑动窗长度（时间单位）。窗长给出 Rayleigh 分辨率 ``1 / window``。
WINDOW = 50.0

#: 零填充倍数，与模块内的 ``INTERPOLATION`` 一致：栅格步长 = 分辨率 / 4。
ZERO_PADDING = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regular_record(n_points: int, period: float = PERIOD, span: float = SPAN) -> tuple[npt.NDArray, npt.NDArray]:
    """等间距采样的单频正弦记录。"""
    t = np.linspace(0.0, span, n_points)
    return t, np.sin(2.0 * np.pi * t / period)


def _dominant_period(result) -> float:
    """整个演化谱中最强主峰所对应的周期。"""
    power = result.power
    _row, col = np.unravel_index(int(np.argmax(power)), power.shape)
    return float(1.0 / result.frequencies[col])


# ---------------------------------------------------------------------------
# 1. The frequency grid belongs to the record, not to the sampling density
# ---------------------------------------------------------------------------


def test_frequency_grid_does_not_drift_with_sampling_density() -> None:
    """同一条记录采样十倍更密，报告的主周期不得漂移。

    这是本仓库修过的一个真实缺陷：栅格由采样密度决定时，采样越密周期反而
    估得越差（7.0 在 20000 个采样点时变成 9.89）。
    """
    t_coarse, v_coarse = _regular_record(500)
    t_fine, v_fine = _regular_record(5000)

    coarse = evolutionary_spectrum(t_coarse, v_coarse, window=WINDOW, method="fft")
    fine = evolutionary_spectrum(t_fine, v_fine, window=WINDOW, method="fft")

    period_coarse = _dominant_period(coarse)
    period_fine = _dominant_period(fine)

    assert abs(period_coarse - period_fine) / PERIOD < 0.05
    assert abs(period_coarse - PERIOD) / PERIOD < 0.05
    assert abs(period_fine - PERIOD) / PERIOD < 0.05


def test_frequency_grid_is_set_by_the_window_length_not_the_sample_count() -> None:
    """栅格首点 = 1 / 窗长（Rayleigh 分辨率），步长 = 分辨率 / 零填充倍数。"""
    coarse = evolutionary_spectrum(*_regular_record(500), window=WINDOW)
    fine = evolutionary_spectrum(*_regular_record(5000), window=WINDOW)

    resolution = 1.0 / WINDOW
    for result in (coarse, fine):
        assert result.frequencies[0] == pytest.approx(resolution, rel=0.01)
        step = result.frequencies[1] - result.frequencies[0]
        assert step == pytest.approx(resolution / ZERO_PADDING, rel=0.01)

    step_coarse = coarse.frequencies[1] - coarse.frequencies[0]
    step_fine = fine.frequencies[1] - fine.frequencies[0]
    assert abs(step_coarse - step_fine) / step_coarse < 0.01

    # 两个栅格在公共区间上逐点重合（加密采样只让栅格向上延伸）。
    n_common = min(coarse.frequencies.size, fine.frequencies.size)
    assert np.allclose(coarse.frequencies[:n_common], fine.frequencies[:n_common], rtol=5e-3)


def test_evolutionary_spectrum_shapes_are_consistent() -> None:
    """功率矩阵形状为 (窗口数, 频率数)，窗中心时间沿时间轴递增。"""
    t, v = _regular_record(2001)
    result = evolutionary_spectrum(t, v, window=WINDOW, method="fft")

    assert result.power.shape == (result.times.size, result.frequencies.size)
    assert result.times.size > 1
    assert np.all(np.diff(result.times) > 0.0)
    assert np.all(result.frequencies > 0.0)
    assert np.all(result.power >= 0.0)


# ---------------------------------------------------------------------------
# 2. Known periods are detected in the right window
# ---------------------------------------------------------------------------


def _two_period_record() -> tuple[npt.NDArray, npt.NDArray]:
    """前半段周期 20、后半段周期 50 的记录（总长 200，两段各 100）。"""
    t = np.linspace(0.0, 200.0, 4001)
    v = np.where(t < 100.0, np.sin(2.0 * np.pi * t / 20.0), np.sin(2.0 * np.pi * t / 50.0))
    return t, v


def test_known_periods_are_detected_in_each_window() -> None:
    """两个时间窗分别给出 20 和 50。"""
    t, v = _two_period_record()
    result = evolutionary_spectrum(t, v, window=100.0, method="fft", overlap=0.0)

    peaks = result.peak_periods()
    assert len(peaks) == 2
    assert abs(peaks[0] - 20.0) / 20.0 < 0.01
    assert abs(peaks[1] - 50.0) / 50.0 < 0.01


def test_known_periods_are_detected_with_the_default_overlap() -> None:
    """默认重叠（0.5）时，首窗与末窗分别落在两段纯信号上。"""
    t, v = _two_period_record()
    result = evolutionary_spectrum(t, v, window=100.0, method="fft")

    peaks = result.peak_periods()
    assert result.times.size >= 2
    assert abs(peaks[0] - 20.0) / 20.0 < 0.01
    assert abs(peaks[-1] - 50.0) / 50.0 < 0.01


# ---------------------------------------------------------------------------
# 3. The three estimators agree
# ---------------------------------------------------------------------------


def test_three_estimators_agree_on_the_dominant_period() -> None:
    """FFT / Lomb-Scargle / MTM 在同一信号上的主周期相对差小于 10%。"""
    t = np.linspace(0.0, SPAN, 2001)
    v = np.sin(2.0 * np.pi * t / PERIOD) + 0.1 * np.sin(2.0 * np.pi * t / 2.5)

    periods = []
    for method in ("fft", "lombscargle", "mtm"):
        result = evolutionary_spectrum(t, v, window=WINDOW, method=method)
        assert result.method == method
        periods.append(_dominant_period(result))

    reference = float(np.mean(periods))
    assert max(abs(p - reference) for p in periods) / reference < 0.10
    assert abs(reference - PERIOD) / PERIOD < 0.05


def test_convenience_wrappers_report_their_own_estimator() -> None:
    """两个便捷包装函数各自记录实际使用的估计器。"""
    t, v = _regular_record(1001)
    assert evolutionary_lomb_scargle(t, v, window=WINDOW).method == "lombscargle"
    assert evolutionary_multitaper(t, v, window=WINDOW).method == "mtm"


# ---------------------------------------------------------------------------
# 4. Unevenly sampled series
# ---------------------------------------------------------------------------


def _uneven_record(n_points: int = 1500) -> tuple[npt.NDArray, npt.NDArray]:
    """不等间距采样的单频正弦记录（随机抖动，固定种子）。"""
    rng = np.random.default_rng(20240917)
    t = np.sort(rng.uniform(0.0, SPAN, size=n_points))
    v = np.sin(2.0 * np.pi * t / PERIOD)
    return t, v


def test_unevenly_sampled_series_still_detects_the_period() -> None:
    """不等间距时间轴不崩溃，且仍能检出周期。"""
    t, v = _uneven_record()
    result = evolutionary_lomb_scargle(t, v, window=WINDOW)

    assert result.method == "lombscargle"
    assert result.times.size >= 1
    assert abs(_dominant_period(result) - PERIOD) / PERIOD < 0.05


def test_unevenly_sampled_series_falls_back_to_lomb_scargle() -> None:
    """等间距专属的估计器遇到不等间距数据时自动改用 Lomb-Scargle。"""
    t, v = _uneven_record()
    result = evolutionary_spectrum(t, v, window=WINDOW, method="fft")

    assert result.method == "lombscargle"
    assert abs(_dominant_period(result) - PERIOD) / PERIOD < 0.05


# ---------------------------------------------------------------------------
# 5. Error contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [0.0, -1.0, -50.0])
def test_non_positive_window_is_rejected(window: float) -> None:
    t, v = _regular_record(1001)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=window)


def test_window_longer_than_the_record_is_rejected() -> None:
    t, v = _regular_record(501)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=2.0 * SPAN)


def test_window_with_too_few_samples_is_rejected() -> None:
    """窗内样本数少于谱估计所需的下限时必须报错，而不是悄悄算一个错谱。"""
    t, v = _regular_record(500)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=1.0)


@pytest.mark.parametrize("n_frequencies", [0, -3])
def test_invalid_frequency_count_is_rejected(n_frequencies: int) -> None:
    t, v = _regular_record(1001)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=WINDOW, n_frequencies=n_frequencies)


def test_unknown_method_is_rejected() -> None:
    t, v = _regular_record(1001)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=WINDOW, method="wavelet")


@pytest.mark.parametrize("overlap", [1.0, 1.5, -0.1])
def test_invalid_overlap_is_rejected(overlap: float) -> None:
    t, v = _regular_record(1001)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=WINDOW, overlap=overlap)


def test_mismatched_lengths_are_rejected() -> None:
    t, v = _regular_record(1001)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v[:-1], window=WINDOW)


def test_constant_series_is_rejected() -> None:
    t = np.linspace(0.0, SPAN, 1001)
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, np.ones_like(t), window=WINDOW)


def test_non_increasing_time_is_rejected() -> None:
    t, v = _regular_record(1001)
    t[10] = t[5]
    with pytest.raises(DataValidationError):
        evolutionary_spectrum(t, v, window=WINDOW)
