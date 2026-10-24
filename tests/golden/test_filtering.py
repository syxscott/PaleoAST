# =============================================================================
# FILE: tests/golden/test_filtering.py
# =============================================================================
"""
Tests for the filters in ``stratigraphy.filtering``.

Every numeric expectation here is either

* a property of the *input* -- a sinusoid of a known period has that period in
  its spectrum, and a least-squares fit of ``sin``/``cos`` plus a straight line
  recovers both the harmonic amplitude and the slope;
* a value derived from the published response function -- the Gaussian band-pass
  is documented as ``exp(-2 pi**2 sigma**2 (|f| - f0)**2)``, so its gain is 1 at
  the target period and ``exp(-1/2)`` at DC, and the Acycle band-pass is
  documented as ``exp(-ln(sqrt(2)) * ((|f| - fc) / half_width)**2)``, so its gain
  is ``1/sqrt(2)`` at each band edge; or
* a property of the *noise*, namely that a Gaussian low-pass removes broadband
  noise and a narrow band-pass removes the noise that lives outside the band.

What is deliberately NOT asserted: a number obtained by running this module and
freezing the output.  The tests that want determinism say so and say which
knob gives it.

References cited in the assertions:
    Berger, A. & Streck, M. 1994. (Gaussian filter.)
    Kodama, K.P. & Hinnov, L. 2015. Rock Magnetic Cyclostratigraphy. (Taner
    weighted band-pass.)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"bandpass_filter", "gaussian_filter", "taner_filter"}


import math

import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.filtering import bandpass_filter, gaussian_filter, taner_filter
from utils.exceptions import DataValidationError

#: 采样间隔 1 kyr（模块的时间轴单位是 Ma，所以 1 kyr = 0.001 Ma）。
DT_MA = 0.001

#: 目标周期，单位 kyr；在 DT_MA = 0.001 Ma 下也就是 20 个样本。
TARGET_PERIOD_KYR = 20.0


# ---------------------------------------------------------------------------
# Helpers. The decompositions are least-squares fits, so the harmonic amplitude
# and the linear trend are estimated jointly and do not contaminate each other.
# ---------------------------------------------------------------------------


def _decompose(values: npt.NDArray, period: float) -> tuple[float, float]:
    """Fit ``a sin + b cos + c + d t``; return (harmonic amplitude, slope)."""
    index = np.arange(values.size, dtype=float)
    design = np.column_stack(
        [
            np.sin(2.0 * np.pi * index / period),
            np.cos(2.0 * np.pi * index / period),
            np.ones_like(index),
            index,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(coefficients[0], coefficients[1])), float(coefficients[3])


def _power_at_period(values: npt.NDArray, period: float, dt: float) -> float:
    """Power of the DFT bin closest to the given period.

    ``period`` is counted in samples, like in :func:`_decompose`; the spectrum
    grid is in cycles per Ma, so the target frequency is ``1 / (period * dt)``.
    """
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(values.size, d=dt)
    index = int(np.argmin(np.abs(frequencies - 1.0 / (period * dt))))
    return float(abs(spectrum[index]) ** 2)


def _peak_to_floor(values: npt.NDArray, period: float, dt: float) -> float:
    """Power at the given period divided by the median power of the spectrum.

    ``period`` is counted in samples; the spectrum grid is in cycles per Ma, so
    the target frequency is ``1 / (period * dt)``.
    """
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(values.size, d=dt)
    power = np.abs(spectrum) ** 2
    index = int(np.argmin(np.abs(frequencies - 1.0 / (period * dt))))
    floor = float(np.median(np.delete(power, index)))
    return float(power[index]) / floor


# ---------------------------------------------------------------------------
# Gaussian filter
# ---------------------------------------------------------------------------


def test_gaussian_bandpass_keeps_the_target_period_and_damps_the_trend():
    """The 20-period component survives; the linear trend is pushed down."""
    size = 2000
    index = np.arange(size, dtype=float)
    signal = np.sin(2.0 * np.pi * index / 20.0) + 0.01 * index

    result = gaussian_filter(signal, cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA)

    assert result.method == "gaussian"
    assert result.filtered.size == size
    np.testing.assert_array_equal(result.original, signal)

    amplitude_filtered, slope_filtered = _decompose(result.filtered, 20.0)
    amplitude_original, slope_original = _decompose(signal, 20.0)
    assert amplitude_filtered >= 0.95 * amplitude_original
    # The documented response is exp(-2 pi^2 sigma^2 (|f| - f0)^2) with
    # sigma = T / (2 pi), so DC keeps exp(-1/2) of its amplitude.
    assert abs(slope_filtered) <= 0.75 * abs(slope_original)
    assert result.parameters["dc_gain"] == pytest.approx(math.exp(-0.5), rel=1e-9)


def test_gaussian_bandpass_passes_its_target_period_at_unit_gain():
    """Whatever period is asked for is the period that comes out, at gain 1."""
    size = 4096
    index = np.arange(size, dtype=float)
    for period in (10.0, 20.0, 40.0):
        signal = np.sin(2.0 * np.pi * index / period)
        result = gaussian_filter(signal, cutoff_period_kyr=period, dt=DT_MA)
        amplitude, _ = _decompose(result.filtered, period)
        assert amplitude == pytest.approx(1.0, abs=0.05)


def test_gaussian_lowpass_removes_high_frequency_noise():
    """A low-pass on signal plus white noise keeps the correlation with the signal."""
    size = 2000
    index = np.arange(size, dtype=float)
    signal = np.sin(2.0 * np.pi * index / 400.0)
    rng = np.random.default_rng(20261024)
    noisy = signal + rng.normal(scale=0.5, size=size)

    result = gaussian_filter(noisy, sigma=4.0 * DT_MA, dt=DT_MA)

    core = slice(50, size - 50)
    assert result.parameters["filter"] == "gaussian_lowpass"
    assert np.corrcoef(result.filtered[core], signal[core])[0, 1] > 0.95
    # The unfiltered record is much worse, so the improvement is the filter's.
    assert np.corrcoef(noisy[core], signal[core])[0, 1] < 0.95


# ---------------------------------------------------------------------------
# Taner filter
# ---------------------------------------------------------------------------


def test_taner_filter_raises_the_peak_to_noise_ratio_at_the_target_period():
    """The target period stands out far above the noise floor after filtering."""
    size = 2000
    index = np.arange(size, dtype=float)
    signal = np.sin(2.0 * np.pi * index / 100.0)
    rng = np.random.default_rng(4242)
    noisy = signal + rng.normal(scale=0.5, size=size)

    result = taner_filter(noisy, cutoff_period_kyr=100.0, dt=DT_MA)

    assert result.method == "taner"
    assert result.filtered.size == size
    before = _peak_to_floor(noisy, 100.0, DT_MA)
    after = _peak_to_floor(result.filtered, 100.0, DT_MA)
    assert after > 5.0 * before
    # The target period sits inside the pass band, so it is not weakened.
    assert _power_at_period(result.filtered, 100.0, DT_MA) > 0.8 * _power_at_period(noisy, 100.0, DT_MA)


# ---------------------------------------------------------------------------
# Band-pass filter
# ---------------------------------------------------------------------------


def test_bandpass_filter_separates_two_periods():
    """A band-pass around 20 samples leaves 100 samples strongly attenuated."""
    size = 4000
    index = np.arange(size, dtype=float)
    short = np.sin(2.0 * np.pi * index / 20.0)
    long = np.sin(2.0 * np.pi * index / 100.0)
    mixed = short + long

    result = bandpass_filter(mixed, low_period_kyr=15.0, high_period_kyr=30.0, dt=DT_MA)

    kept, _ = _decompose(result.filtered, 20.0)
    removed, _ = _decompose(result.filtered, 100.0)
    assert kept >= 0.8
    # The response is exp(-ln(sqrt(2)) * ((|f| - fc) / half_width)**2): the
    # 100-sample period is 2.4 half-widths away from the centre, i.e. a gain of
    # exp(-2.0) = 0.135.
    assert removed <= 0.25


# ---------------------------------------------------------------------------
# Unevenly spaced records
# ---------------------------------------------------------------------------


def _amplitude_at_period(values: npt.NDArray, times: npt.NDArray, period_ma: float) -> float:
    """Least-squares amplitude of a sinusoid of ``period_ma`` at the given times."""
    design = np.column_stack(
        [
            np.sin(2.0 * np.pi * times / period_ma),
            np.cos(2.0 * np.pi * times / period_ma),
            np.ones_like(times),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(coefficients[0], coefficients[1]))


def _residual_rms(values: npt.NDArray, times: npt.NDArray, period_ma: float) -> float:
    """RMS of what is left after removing the sinusoid, the constant and nothing else."""
    design = np.column_stack(
        [
            np.sin(2.0 * np.pi * times / period_ma),
            np.cos(2.0 * np.pi * times / period_ma),
            np.ones_like(times),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.sqrt(np.mean((values - design @ coefficients) ** 2)))


def test_filters_accept_an_unevenly_spaced_time_axis():
    """An irregularly sampled record is filtered on its actual time differences.

    The target period has to survive and the broadband noise around it has to go
    down, for every filter -- the time axis is in Ma, so the target period is
    100 kyr = 0.1 Ma regardless of how the samples are spaced.
    """
    size = 1200
    rng = np.random.default_rng(20261024)
    spacing = 0.001 * (1.0 + 0.6 * rng.random(size - 1))
    times = np.concatenate(([0.0], np.cumsum(spacing)))
    signal = np.sin(2.0 * np.pi * times / 0.1)
    noisy = signal + rng.normal(scale=0.5, size=size)

    results = [
        gaussian_filter(noisy, cutoff_period_kyr=100.0, dt=DT_MA, time=times),
        taner_filter(noisy, cutoff_period_kyr=100.0, dt=DT_MA, time=times),
        bandpass_filter(noisy, low_period_kyr=60.0, high_period_kyr=200.0, dt=DT_MA, time=times),
    ]

    for result in results:
        assert result.filtered.size == size
        assert result.parameters["time_axis"] == "resampled"
        # The target period is inside the pass band, so it is not weakened ...
        assert _amplitude_at_period(result.filtered, times, 0.1) > 0.8 * _amplitude_at_period(noisy, times, 0.1)
        # ... and the noise outside the band is pushed down.
        assert _residual_rms(result.filtered, times, 0.1) < 0.3 * _residual_rms(noisy, times, 0.1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_inputs_raise_validation_error():
    """Bad shapes and bad parameters raise DataValidationError (a.k.a. ValidationError)."""
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(3), cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA)
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(100), sigma=0.0, dt=DT_MA)
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(100), sigma=-2.0, dt=DT_MA)
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(100), cutoff_period_kyr=0.0, dt=DT_MA)
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(100), cutoff_period_kyr=-20.0, dt=DT_MA)
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(100), dt=DT_MA)
    with pytest.raises(DataValidationError):
        gaussian_filter(np.ones(100), cutoff_period_kyr=TARGET_PERIOD_KYR, dt=0.0)
    with pytest.raises(DataValidationError):
        taner_filter(np.ones(3), cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA)
    with pytest.raises(DataValidationError):
        taner_filter(np.ones(100), cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA, pad=-1)
    with pytest.raises(DataValidationError):
        bandpass_filter(np.ones(3), low_period_kyr=15.0, high_period_kyr=30.0, dt=DT_MA)
    with pytest.raises(DataValidationError):
        bandpass_filter(np.ones(100), low_period_kyr=30.0, high_period_kyr=15.0, dt=DT_MA)


def test_invalid_time_axes_raise_validation_error():
    """A time axis that does not fit the record raises DataValidationError."""
    values = np.ones(100)
    times = np.arange(100, dtype=float) * DT_MA
    with pytest.raises(DataValidationError):
        gaussian_filter(values, cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA, time=times[:50])
    with pytest.raises(DataValidationError):
        gaussian_filter(values, cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA, time=times[::-1].copy())
    with pytest.raises(DataValidationError):
        taner_filter(values, cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA, time=times[:50])
    with pytest.raises(DataValidationError):
        bandpass_filter(values, low_period_kyr=15.0, high_period_kyr=30.0, dt=DT_MA, time=times[::-1].copy())


def test_summary_reports_the_parameters_used():
    """The summary is part of the contract: it names the method and the parameters."""
    signal = np.sin(2.0 * np.pi * np.arange(400) / 20.0)
    result = gaussian_filter(signal, cutoff_period_kyr=TARGET_PERIOD_KYR, dt=DT_MA)
    text = result.summary()
    assert "gaussian" in text
    assert "cutoff_period_kyr=20.0" in text
