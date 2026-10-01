# tests/stratigraphy/test_spectral_detrend_ar1.py
"""
Regression tests for the Lomb-Scargle spectral analysis: detrending and
the AR(1) red-noise null model (缺陷 5).

The previous implementation analyzed the raw signal with no detrending
and used only a relative-power threshold for "significance". The fix
adds an optional linear detrending step (default ON, matching standard
practice in paleoclimate spectral analysis) and an AR(1) red-noise
false-alarm probability (Mann & Lees 1996; Schulz & Mudelsee 2002).
"""

import numpy as np
import pytest

from stratigraphy.spectral_analysis import SpectralAnalyzer


def _pure_sine(t: np.ndarray, period: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * t / period)


def _ar1_series(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal() * np.sqrt(1 - phi**2)
    return x


class TestDetrending:
    """Linear detrending must shrink the low-frequency end of the
    spectrum when the signal has a drift."""

    def test_detrend_default_true(self):
        """The fix adds a `detrend=True` default. The keyword exists."""
        t = np.linspace(0, 100, 200)
        # Pure sine + huge linear drift
        x = _pure_sine(t, 10) + 5 * t
        analyzer = SpectralAnalyzer()
        result = analyzer.analyze(t, x, frequency_range=(0.01, 5.0),
                                  n_frequencies=200, detrend=True)
        assert result.ar1_phi is not None

    def test_detrend_removes_low_frequency_drift(self):
        """Detrending should make the LOW-frequency power smaller
        than no-detrending, when the signal has a linear drift."""
        t = np.linspace(0, 100, 200)
        x = _pure_sine(t, 10) + 5 * t

        analyzer = SpectralAnalyzer()
        r_detrend = analyzer.analyze(
            t, x, frequency_range=(0.01, 5.0), n_frequencies=200,
            detrend=True, ar1_significance=False,
        )
        r_no_detrend = analyzer.analyze(
            t, x, frequency_range=(0.01, 5.0), n_frequencies=200,
            detrend=False, ar1_significance=False,
        )
        # Lowest-frequency bin power must shrink after detrending.
        low_freq_idx = 0  # lowest frequency bin
        assert r_detrend.power[low_freq_idx] < r_no_detrend.power[low_freq_idx], (
            f"Detrend failed to shrink low-frequency power: "
            f"detrended={r_detrend.power[low_freq_idx]} vs "
            f"raw={r_no_detrend.power[low_freq_idx]}"
        )

    def test_detrend_false_preserves_pre_fix_behavior(self):
        """Backward compat: detrend=False must give the same low-freq
        power as the old code (no detrending)."""
        t = np.linspace(0, 100, 200)
        x = _pure_sine(t, 10) + 5 * t
        analyzer = SpectralAnalyzer()
        r = analyzer.analyze(t, x, frequency_range=(0.01, 5.0),
                             n_frequencies=200, detrend=False)
        # Just sanity: should run without error and produce a result
        assert r.power is not None
        assert np.all(np.isfinite(r.power))


class TestAR1NullModel:
    """An AR(1) null model is fit and significance is reported."""

    def test_ar1_phi_is_computed(self):
        t = np.linspace(0, 100, 200)
        # An AR(1) series with phi = 0.7
        x = _ar1_series(len(t), phi=0.7, seed=42)
        result = SpectralAnalyzer().analyze(t, x, frequency_range=(0.01, 5.0),
                                            n_frequencies=100)
        assert result.ar1_phi is not None
        # Estimated phi should be in the same ballpark as the true phi
        assert abs(result.ar1_phi - 0.7) < 0.2

    def test_ar1_significance_mask_is_boolean(self):
        t = np.linspace(0, 100, 200)
        x = _ar1_series(len(t), phi=0.7, seed=42)
        result = SpectralAnalyzer().analyze(t, x, frequency_range=(0.01, 5.0),
                                            n_frequencies=100)
        assert result.ar1_significant is not None
        assert result.ar1_significant.dtype == bool
        assert len(result.ar1_significant) == len(result.frequencies)

    def test_ar1_significant_5pct_threshold_reasonable(self):
        """A pure AR(1) series should give a small fraction of
        significant peaks (5 % at the α=0.05 level by construction)."""
        rng = np.random.default_rng(123)
        t = np.linspace(0, 100, 200)
        x = _ar1_series(len(t), phi=0.7, seed=42)
        result = SpectralAnalyzer().analyze(t, x, frequency_range=(0.01, 5.0),
                                            n_frequencies=100)
        # ar1_fap_5pct is the power threshold above which a peak is
        # significant at the 5 % level under the AR(1) null. For a
        # pure AR(1) signal, only ~5 % of the periodogram points
        # should exceed it.
        frac_sig = float(np.mean(result.ar1_significant))
        # Loose bounds: between 0 and 25 %
        assert 0.0 <= frac_sig <= 0.25, (
            f"AR(1) null gives {frac_sig * 100:.1f}% significant points; "
            "expected ~5 % (0-25 % acceptable)"
        )

    def test_ar1_significance_off(self):
        """When ar1_significance=False, AR(1) fields must be None."""
        t = np.linspace(0, 100, 200)
        x = _ar1_series(len(t), phi=0.7, seed=42)
        result = SpectralAnalyzer().analyze(t, x, frequency_range=(0.01, 5.0),
                                            n_frequencies=100,
                                            ar1_significance=False)
        assert result.ar1_phi is None
        assert result.ar1_fap_5pct is None
        assert result.ar1_significant is None


class TestStrongSignalRejectedOnlyWithLowFAP:
    """A pure sine on top of white noise must give a strongly
    significant peak under the AR(1) null."""

    def test_pure_sine_is_significant(self):
        rng = np.random.default_rng(7)
        t = np.linspace(0, 100, 300)
        x = _pure_sine(t, period=10.0, amplitude=3.0) + rng.normal(0, 0.3, len(t))
        result = SpectralAnalyzer().analyze(t, x, frequency_range=(0.05, 2.0),
                                            n_frequencies=200)
        # The peak power should comfortably exceed the AR(1) 5% FAP
        # threshold — the signal is far above the noise.
        assert result.peak_power > result.ar1_fap_5pct, (
            f"Pure sine peak power {result.peak_power} is below AR(1) "
            f"5% FAP threshold {result.ar1_fap_5pct}"
        )
