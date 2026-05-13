# =============================================================================
# FILE: stratigraphy/spectral_analysis.py
# =============================================================================
"""
Spectral Analysis Module for PaleoAST

This module implements spectral analysis using the Lomb-Scargle periodogram
for unevenly sampled time series commonly found in geological data, and
Wavelet Continuous Transform (CWT) for time-frequency analysis.

Mathematical Foundation:

Lomb-Scargle Periodogram:

    P_n(ω) = (1/2) × [ (Σ y_j cos ω(t_j - τ))² / Σ cos² ω(t_j - τ)
                      + (Σ y_j sin ω(t_j - τ))² / Σ sin² ω(t_j - τ) ]

where τ is the time offset that orthogonalizes the sine and cosine terms:

    τ = (1/2ω) × arctan( Σ sin 2ωt_j / Σ cos 2ωt_j )

Wavelet CWT:

    W(a, b) = (1/√a) × ∫ x(t) × ψ*((t-b)/a) dt

where:
    ψ = mother wavelet function
    a = scale parameter
    b = translation parameter

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve

from config.i18n import _
from utils.exceptions import ComputationError
from utils.validators import validate_data_array

logger = logging.getLogger(__name__)


def _morlet_wavelet(n: int, w0: float = 5.0) -> np.ndarray:
    """
    Generate Morlet wavelet of length n.

    Parameters:
        n: Length of wavelet
        w0: Central frequency (default 5.0)

    Returns:
        Complex Morlet wavelet
    """
    n = int(n)
    half_n = n // 2
    t = np.arange(-half_n, half_n + 1)
    # Morlet wavelet: pi^{-1/4} * exp(i*w0*t) * exp(-t^2/2)
    wavelet = np.exp(1j * w0 * t) * np.exp(-t**2 / 2)
    wavelet *= np.pi**(-0.25)  # Normalization
    return wavelet


def _mexican_hat_wavelet(n: int) -> np.ndarray:
    """
    Generate Mexican Hat (Ricker) wavelet of length n.

    Parameters:
        n: Length of wavelet (should be odd)

    Returns:
        Mexican hat wavelet
    """
    n = int(n)
    if n % 2 == 0:
        n += 1
    half_n = n // 2
    t = np.arange(-half_n, half_n + 1)
    # Mexican hat: (1 - t^2) * exp(-t^2/2)
    wavelet = (1 - t**2) * np.exp(-t**2 / 2)
    wavelet *= np.pi**(-0.25)  # Normalization
    return wavelet


def _cwt_simple(signal: np.ndarray, wavelet_func, scales: np.ndarray) -> np.ndarray:
    """
    Simple Continuous Wavelet Transform using convolution.

    Parameters:
        signal: Input signal
        wavelet_func: Function to generate wavelet of given length
        scales: Wavelet scales

    Returns:
        CWT matrix of shape (len(scales), len(signal))
    """
    n_samples = len(signal)
    n_scales = len(scales)
    cwt_matrix = np.zeros((n_scales, n_samples), dtype=complex)

    for i, scale in enumerate(scales):
        # Wavelet length proportional to scale
        wavelet_len = max(2 * int(scale) + 1, 7)
        if wavelet_len % 2 == 0:
            wavelet_len += 1
        wavelet = wavelet_func(wavelet_len)
        # Normalize wavelet by scale
        wavelet = wavelet / np.sqrt(scale)
        # Convolve
        cwt_matrix[i] = fftconvolve(signal, wavelet, mode='same')

    return cwt_matrix


@dataclass
class WaveletResult:
    """
    Container for wavelet CWT analysis results.

    Attributes:
        time: Time axis
        scales: Wavelet scales used
        frequencies: Corresponding frequencies
        power: Wavelet power spectrum (time × scale)
        coi: Cone of Influence mask
        wavelet: Wavelet name used
        peak_scale: Scale with maximum power
        peak_frequency: Corresponding frequency
    """

    time: npt.NDArray
    scales: npt.NDArray
    frequencies: npt.NDArray
    power: npt.NDArray
    coi: npt.NDArray | None
    wavelet: str
    peak_scale: float | None
    peak_frequency: float | None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Wavelet CWT Analysis Results"),
            "=" * 50,
            _("Wavelet: {0}").format(self.wavelet),
            _("Time points: {0}").format(len(self.time)),
            _("Scales: {0}").format(len(self.scales)),
        ]

        if self.peak_frequency is not None:
            lines.append(_("Peak frequency: {0}").format(f"{self.peak_frequency:.4f}"))

        return "\n".join(lines)


@dataclass
class SpectralResult:
    """
    Container for spectral analysis results.
    """

    frequencies: npt.NDArray
    periods: npt.NDArray
    power: npt.NDArray
    peak_frequency: float | None
    peak_period: float | None
    peak_power: float | None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Spectral Analysis Results"),
            "=" * 50,
            _("Number of frequencies: {0}").format(len(self.frequencies)),
        ]

        if self.peak_frequency is not None:
            lines.append(_("Peak frequency: {0}").format(f"{self.peak_frequency:.6f}"))
            lines.append(_("Peak period: {0}").format(f"{self.peak_period:.4f}"))
            lines.append(_("Peak power: {0}").format(f"{self.peak_power:.4f}"))

        return "\n".join(lines)


class SpectralAnalyzer:
    """
    Lomb-Scargle periodogram analyzer for unevenly sampled time series.
    """

    def __init__(self) -> None:
        """Initialize the spectral analyzer."""
        self._lock = threading.RLock()
        self._last_result: SpectralResult | None = None

    def analyze(
        self,
        time: npt.NDArray,
        values: npt.NDArray,
        frequency_range: tuple[float, float] | None = None,
        n_frequencies: int = 1000,
    ) -> SpectralResult:
        """
        Perform Lomb-Scargle spectral analysis.

        Parameters:
            time: Time points (can be unevenly spaced)
            values: Signal values at each time point
            frequency_range: Tuple of (min_freq, max_freq). If None, auto-calculated.
            n_frequencies: Number of frequencies to evaluate

        Returns:
            SpectralResult: Spectral analysis results
        """
        with self._lock:
            logger.info(
                f"Starting Lomb-Scargle spectral analysis: {len(time)} data points, n_frequencies={n_frequencies}"
            )
            # Validate input
            time = validate_data_array(time, allow_nan=False, name="time").flatten()
            values = validate_data_array(values, allow_nan=False, name="values").flatten()

            if len(time) != len(values):
                raise ComputationError(
                    "Time and values arrays must have same length",
                    details={"time_length": len(time), "values_length": len(values)},
                )

            if len(time) < 4:
                raise ComputationError("Need at least 4 data points for spectral analysis")

            # Remove NaN values
            valid_mask = ~(np.isnan(time) | np.isnan(values))
            time = time[valid_mask]
            values = values[valid_mask]

            # Sort by time
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            values = values[sort_idx]

            # Determine frequency range
            if frequency_range is None:
                # Auto-calculate based on data
                time_span = time[-1] - time[0]
                if time_span <= 0:
                    raise ComputationError(
                        "Time span must be positive for spectral analysis",
                        details={"time_span": time_span, "n_points": len(time)},
                    )
                min_freq = 1.0 / (10 * time_span)  # Very low frequency
                max_freq = len(time) / (2 * time_span)  # Nyquist-like
            else:
                min_freq, max_freq = frequency_range

            logger.info(f"Frequency range: [{min_freq:.6f}, {max_freq:.6f}]")

            # Generate frequency array
            frequencies = np.linspace(min_freq, max_freq, n_frequencies)

            # Compute Lomb-Scargle periodogram
            power = self._lomb_scargle(time, values, frequencies)

            # Find peak
            peak_idx = np.argmax(power)
            peak_frequency = frequencies[peak_idx]
            peak_period = 1.0 / peak_frequency if peak_frequency > 0 else None
            peak_power = power[peak_idx]

            result = SpectralResult(
                frequencies=frequencies,
                periods=1.0 / frequencies,
                power=power,
                peak_frequency=peak_frequency,
                peak_period=peak_period,
                peak_power=peak_power,
            )

            self._last_result = result
            period_str = f"{peak_period:.4f}" if peak_period is not None else "N/A"
            logger.info(
                f"Spectral analysis complete: peak frequency={peak_frequency:.6f}, "
                f"peak period={period_str}, "
                f"peak power={peak_power:.4f}"
            )
            return result

    def _lomb_scargle(self, time: npt.NDArray, values: npt.NDArray, frequencies: npt.NDArray) -> npt.NDArray:
        """
        Compute Lomb-Scargle periodogram.

        Parameters:
            time: Time points
            values: Signal values
            frequencies: Frequencies to evaluate

        Returns:
            npt.NDArray: Power at each frequency
        """
        n = len(time)
        power = np.zeros(len(frequencies))

        # Precompute mean and variance
        mean_val = np.mean(values)
        values_centered = values - mean_val
        variance = np.var(values_centered)

        if variance == 0:
            return power

        for i, freq in enumerate(frequencies):
            # Compute tau (time offset)
            omega = 2 * np.pi * freq

            # Compute sin(2*omega*t) and cos(2*omega*t) sums
            sin_2wt_sum = np.sum(np.sin(2 * omega * time))
            cos_2wt_sum = np.sum(np.cos(2 * omega * time))

            if abs(sin_2wt_sum) < 1e-10 and abs(cos_2wt_sum) < 1e-10:
                tau = 0
            else:
                tau = np.arctan2(sin_2wt_sum, cos_2wt_sum) / (2 * omega)

            # Compute shifted times
            time_shifted = time - tau

            # Compute sine and cosine components
            sin_wt = np.sin(omega * time_shifted)
            cos_wt = np.cos(omega * time_shifted)

            # Compute numerator
            sum_sin = np.sum(values_centered * sin_wt)
            sum_cos = np.sum(values_centered * cos_wt)

            # Compute denominator
            sum_sin2 = np.sum(sin_wt**2)
            sum_cos2 = np.sum(cos_wt**2)

            # Lomb-Scargle power
            if sum_sin2 > 0 and sum_cos2 > 0:
                power[i] = 0.5 * ((sum_sin**2) / sum_sin2 + (sum_cos**2) / sum_cos2)
            else:
                power[i] = 0

        # Normalize by variance
        power = power / variance

        return power

    def find_significant_peaks(
        self, result: SpectralResult | None = None, threshold: float = 0.5
    ) -> list[dict[str, float]]:
        """
        Find significant peaks in the periodogram.

        Parameters:
            result: Spectral result. If None, uses last result.
            threshold: Power threshold as fraction of max power

        Returns:
            List of peak information dictionaries
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ComputationError("No spectral result available")

        logger.info(f"Finding significant peaks with threshold={threshold}")
        max_power = np.max(result.power)
        threshold_value = threshold * max_power

        peaks = []
        in_peak = False
        peak_start = 0

        for i in range(len(result.power)):
            if result.power[i] > threshold_value and not in_peak:
                in_peak = True
                peak_start = i
            elif result.power[i] <= threshold_value and in_peak:
                in_peak = False
                # Find peak within range
                peak_range = result.power[peak_start:i]
                if len(peak_range) > 0:
                    local_max_idx = np.argmax(peak_range) + peak_start
                    peaks.append(
                        {
                            "frequency": result.frequencies[local_max_idx],
                            "period": result.periods[local_max_idx],
                            "power": result.power[local_max_idx],
                            "relative_power": result.power[local_max_idx] / max_power,
                        }
                    )

        # Check if still in peak at end
        if in_peak:
            peak_range = result.power[peak_start:]
            if len(peak_range) > 0:
                local_max_idx = np.argmax(peak_range) + peak_start
                peaks.append(
                    {
                        "frequency": result.frequencies[local_max_idx],
                        "period": result.periods[local_max_idx],
                        "power": result.power[local_max_idx],
                        "relative_power": result.power[local_max_idx] / max_power,
                    }
                )

        # Sort by power
        peaks.sort(key=lambda x: x["power"], reverse=True)

        logger.info(f"Found {len(peaks)} significant peaks")
        return peaks

    def wavelet_transform(
        self,
        time: npt.NDArray,
        values: npt.NDArray,
        wavelet: str = "morlet",
        scales: npt.NDArray | None = None,
        dt: float = 1.0,
    ) -> WaveletResult:
        """
        Perform Continuous Wavelet Transform (CWT).

        Parameters:
            time: Time points
            values: Signal values
            wavelet: Wavelet type ('morlet', 'ricker', 'mexican_hat')
            scales: Wavelet scales. If None, auto-calculated.
            dt: Time step between samples

        Returns:
            WaveletResult: Wavelet analysis results
        """
        logger.info(f"Starting wavelet CWT: {len(values)} points, wavelet={wavelet}")

        # Validate input
        time = validate_data_array(time, allow_nan=False, name="time").flatten()
        values = validate_data_array(values, allow_nan=False, name="values").flatten()

        if len(time) != len(values):
            raise ComputationError("Time and values must have same length")

        # Auto-calculate scales if not provided
        if scales is None:
            # Use scales from 2 to len/2 (dyadic scales)
            n = len(values)
            scales = np.arange(2, min(n // 2, 100))

        # Compute CWT using scipy
        if wavelet == "morlet":
            # Morlet wavelet
            cwt_matrix = _cwt_simple(values, _morlet_wavelet, scales)
            wavelet_name = "Morlet"
        elif wavelet in ("ricker", "mexican_hat"):
            cwt_matrix = _cwt_simple(values, _mexican_hat_wavelet, scales)
            wavelet_name = "Mexican Hat"
        else:
            # Default to Morlet
            cwt_matrix = _cwt_simple(values, _morlet_wavelet, scales)
            wavelet_name = "Morlet"

        # Power spectrum
        power = np.abs(cwt_matrix) ** 2

        # Convert scales to frequencies
        # For Morlet: freq = (center_freq * sampling_rate) / scale
        # Center frequency for Morlet is ~0.8125
        sampling_rate = 1.0 / dt if dt > 0 else 1.0
        center_freq = 0.8125
        frequencies = (center_freq * sampling_rate) / scales

        # Find peak
        peak_idx = np.unravel_index(np.argmax(power), power.shape)
        peak_scale = scales[peak_idx[0]]
        peak_frequency = frequencies[peak_idx[0]]

        # Cone of Influence (COI) - regions where edge effects are significant
        # COI = scale * sqrt(2) in sample units
        coi = np.zeros_like(values, dtype=bool)
        for i in range(len(values)):
            # Time from edge
            t_from_edge = min(i, len(values) - 1 - i)
            # Maximum scale that is reliable at this position
            max_reliable_scale = t_from_edge / (center_freq * np.sqrt(2))
            coi[i] = np.any(scales > max_reliable_scale) if t_from_edge > 0 else False

        result = WaveletResult(
            time=time,
            scales=scales,
            frequencies=frequencies,
            power=power,
            coi=coi if np.any(coi) else None,
            wavelet=wavelet_name,
            peak_scale=peak_scale,
            peak_frequency=peak_frequency,
        )

        self._last_result = result
        logger.info(
            f"Wavelet CWT complete: peak_frequency={peak_frequency:.4f}, "
            f"peak_scale={peak_scale:.2f}"
        )
        return result

    @property
    def last_result(self) -> SpectralResult | WaveletResult | None:
        """Get the last spectral or wavelet result."""
        with self._lock:
            return self._last_result
