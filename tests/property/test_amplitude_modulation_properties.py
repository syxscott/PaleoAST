# =============================================================================
# FILE: tests/property/test_amplitude_modulation_properties.py
# =============================================================================
"""
Tests for the amplitude-modulation analysis in
``stratigraphy.amplitude_modulation``.

Every numeric expectation here is either

* a property of the *input* -- the Hilbert envelope of a pure sinusoid is
  constant, and the envelope of an amplitude-modulated signal
  ``a(t) * cos(2 pi fc t)`` is the modulation ``a(t)`` itself;
* a property of the *analytic signal* -- the instantaneous phase of a cosine
  advances linearly with time at the rate ``2 pi fc``;
* a property of the *filter* -- a band-pass carrier must not keep the
  out-of-band trend; or
* a property of the *error contract*: an empty series, a constant series, a
  series with too few points, mismatched lengths, a non-increasing time axis
  and an impossible frequency band are all rejected with
  ``DataValidationError`` (``utils.exceptions.ValidationError`` is an alias of
  that class).

The records used here cover an **integer number of periods**.  The analytic
signal is computed with an FFT, which implicitly closes the record into a loop;
a record that ends in the middle of a cycle makes the Hilbert envelope ring at
its edges, so the "constant envelope" assertions would measure that artefact
rather than the method.

References cited in the assertions:
    Schreiber, T. & Schmitz, A. 2000. Vistas in Astronomy 44: 3139-3174.
    (Hilbert transform, instantaneous amplitude and phase.)
    Taner, T. 2000. Attributes revisited. Rock Solid Images, Inc. (Taner-
    Hilbert method: band-pass first, then the envelope.)
    Oppenheim, A.V. & Schafer, R.W. 2010. Discrete-Time Signal Processing.
    (Butterworth band-pass design; zero-phase forward-backward filtering.)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"amplitude_modulation", "tanner_hilbert"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.amplitude_modulation import (
    amplitude_modulation,
    tanner_hilbert,
)
from utils.exceptions import DataValidationError

#: 采样间隔（时间单位）。
DT = 0.005

#: 载波频率（每时间单位的周期数）。
CARRIER_FREQ = 1.0

#: 调制频率（每时间单位的周期数）。
MODULATION_FREQ = 0.25

#: 调制指数：包络在 ``1 - 0.5`` 与 ``1 + 0.5`` 之间摆动。
MODULATION_DEPTH = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_count(n_periods: float, period: float = 1.0 / CARRIER_FREQ) -> int:
    """覆盖给定个周期所需的样本数（整数，保证 FFT 的周期假设成立）。"""
    return int(np.round(n_periods * period / DT))


def _times(n_points: int) -> npt.NDArray:
    """等间距时间轴。"""
    return np.arange(n_points, dtype=float) * DT


def _amplitude_modulation(n_points: int) -> tuple[npt.NDArray, npt.NDArray]:
    """构造 ``a(t) * cos(2 pi fc t)``，返回 ``(time, modulation)``。"""
    t = _times(n_points)
    modulation = 1.0 + MODULATION_DEPTH * np.sin(2.0 * np.pi * MODULATION_FREQ * t)
    return t, modulation


# ---------------------------------------------------------------------------
# 1. A pure sinusoid has a constant envelope
# ---------------------------------------------------------------------------


def test_pure_sine_has_a_constant_envelope() -> None:
    """纯正弦的包络是常数，相对起伏小于 5%。"""
    n_points = _sample_count(8.0)
    t = _times(n_points)
    values = np.sin(2.0 * np.pi * CARRIER_FREQ * t)

    envelope, phase = tanner_hilbert(t, values)

    assert envelope.shape == values.shape
    assert phase.shape == values.shape
    mean = float(np.mean(envelope))
    assert mean == pytest.approx(1.0, rel=0.05)
    relative_variation = float(np.max(np.abs(envelope - mean)) / mean)
    assert relative_variation < 0.05


def test_constant_series_has_no_envelope() -> None:
    """常数序列没有可调制的振幅，直接报错。"""
    t = _times(200)
    with pytest.raises(DataValidationError):
        tanner_hilbert(t, np.ones_like(t))


# ---------------------------------------------------------------------------
# 2. The envelope reproduces the modulation
# ---------------------------------------------------------------------------


def test_envelope_reproduces_the_modulation() -> None:
    """调幅信号的包络能重现调制本身。"""
    n_points = _sample_count(20.0)
    t, modulation = _amplitude_modulation(n_points)
    values = modulation * np.cos(2.0 * np.pi * CARRIER_FREQ * t)

    result = amplitude_modulation(t, values)

    error = float(np.max(np.abs(result.envelope - modulation))) / float(np.mean(modulation))
    assert error < 0.05
    assert np.corrcoef(result.envelope, modulation)[0, 1] > 0.99
    assert result.modulation_depth() == pytest.approx(MODULATION_DEPTH * 2.0, rel=0.05)


def test_amplitude_modulation_returns_the_demeaned_series_as_carrier() -> None:
    """不给频带时，载波就是去均值后的原序列。"""
    n_points = _sample_count(4.0)
    t = _times(n_points)
    values = 3.0 + np.cos(2.0 * np.pi * CARRIER_FREQ * t)

    result = amplitude_modulation(t, values)

    assert np.allclose(result.carrier, values - np.mean(values))
    assert np.allclose(result.times, t)


# ---------------------------------------------------------------------------
# 3. The instantaneous phase advances linearly
# ---------------------------------------------------------------------------


def test_instantaneous_phase_grows_linearly() -> None:
    """余弦的瞬时相位沿时间线性增长，斜率等于载波角频率。"""
    n_points = _sample_count(20.0)
    t = _times(n_points)
    values = np.cos(2.0 * np.pi * CARRIER_FREQ * t)

    _, phase = tanner_hilbert(t, values)
    unwrapped = np.unwrap(phase)

    slope, intercept = np.polyfit(t, unwrapped, 1)
    assert slope == pytest.approx(2.0 * np.pi * CARRIER_FREQ, rel=0.01)
    residual = unwrapped - (slope * t + intercept)
    assert float(np.max(np.abs(residual))) < 0.05
    assert np.corrcoef(t, unwrapped)[0, 1] > 0.999


# ---------------------------------------------------------------------------
# 4. Band-limited carrier
# ---------------------------------------------------------------------------


def test_band_limits_the_carrier_to_the_requested_band() -> None:
    """带通滤波把带外的慢趋势剔除，带内载波基本原样保留。"""
    n_points = _sample_count(20.0)
    t = _times(n_points)
    trend = 5.0 * t / t[-1]
    values = trend + np.cos(2.0 * np.pi * CARRIER_FREQ * t)

    result = amplitude_modulation(t, values, band=(0.5 * CARRIER_FREQ, 1.5 * CARRIER_FREQ))

    assert abs(float(np.corrcoef(result.carrier, trend)[0, 1])) < 0.1
    # 纯余弦的包络幅值是 1，标准差是 1 / sqrt(2)。
    assert float(np.std(result.carrier)) == pytest.approx(1.0 / np.sqrt(2.0), rel=0.1)
    assert result.envelope.shape == values.shape


def test_band_must_lie_inside_the_nyquist_range() -> None:
    t, _ = _amplitude_modulation(_sample_count(4.0))
    values = np.cos(2.0 * np.pi * CARRIER_FREQ * t)
    nyquist = 1.0 / (2.0 * DT)
    with pytest.raises(DataValidationError):
        amplitude_modulation(t, values, band=(0.5, 2.0 * nyquist))


def test_band_must_be_ordered() -> None:
    t, _ = _amplitude_modulation(_sample_count(4.0))
    values = np.cos(2.0 * np.pi * CARRIER_FREQ * t)
    with pytest.raises(DataValidationError):
        amplitude_modulation(t, values, band=(2.0, 0.5))


# ---------------------------------------------------------------------------
# 5. Error contract
# ---------------------------------------------------------------------------


def test_empty_series_is_rejected() -> None:
    with pytest.raises(DataValidationError):
        amplitude_modulation(np.array([]), np.array([]))


def test_too_short_series_is_rejected() -> None:
    t = _times(4)
    values = np.array([1.0, 2.0, 1.0, 2.0])
    with pytest.raises(DataValidationError):
        amplitude_modulation(t, values)


def test_mismatched_lengths_are_rejected() -> None:
    t = _times(200)
    with pytest.raises(DataValidationError):
        amplitude_modulation(t, np.sin(t)[:-1])


def test_non_increasing_time_is_rejected() -> None:
    t = _times(200)
    values = np.sin(t)
    t[10] = t[5]
    with pytest.raises(DataValidationError):
        amplitude_modulation(t, values)


# ---------------------------------------------------------------------------
# 6. Unevenly sampled series
# ---------------------------------------------------------------------------


def test_unevenly_sampled_series_is_handled() -> None:
    """不等间距的时间轴不崩溃：先重采样到等间距栅格再取包络。"""
    rng = np.random.default_rng(20240917)
    t = np.sort(rng.uniform(0.0, 100.0, size=1500))
    values = np.sin(2.0 * np.pi * t / 5.0)

    envelope, phase = tanner_hilbert(t, values)

    assert envelope.shape == values.shape
    assert phase.shape == values.shape
    assert np.all(np.isfinite(envelope))
    assert float(np.mean(envelope)) == pytest.approx(1.0, rel=0.1)
