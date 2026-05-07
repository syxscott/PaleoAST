# =============================================================================
# FILE: stratigraphy/spectral_analysis.py
# =============================================================================
"""
Spectral Analysis Module for PaleoAST

This module implements spectral analysis using the Lomb-Scargle periodogram
for unevenly sampled time series commonly found in geological data.

Mathematical Foundation:

Lomb-Scargle Periodogram:

    P_n(ω) = (1/2) × [ (Σ y_j cos ω(t_j - τ))² / Σ cos² ω(t_j - τ)
                      + (Σ y_j sin ω(t_j - τ))² / Σ sin² ω(t_j - τ) ]

where τ is the time offset that orthogonalizes the sine and cosine terms:
    
    τ = (1/2ω) × arctan( Σ sin 2ωt_j / Σ cos 2ωt_j )

This formulation handles uneven sampling by finding optimal time shifts.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import threading

from utils.exceptions import ComputationError
from utils.validators import validate_data_array


@dataclass
class SpectralResult:
    """
    Container for spectral analysis results.
    """
    frequencies: npt.NDArray
    periods: npt.NDArray
    power: npt.NDArray
    peak_frequency: Optional[float]
    peak_period: Optional[float]
    peak_power: Optional[float]
    
    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            "Spectral Analysis Results",
            "=" * 50,
            f"Number of frequencies: {len(self.frequencies)}",
        ]
        
        if self.peak_frequency is not None:
            lines.append(f"Peak frequency: {self.peak_frequency:.6f}")
            lines.append(f"Peak period: {self.peak_period:.4f}")
            lines.append(f"Peak power: {self.peak_power:.4f}")
        
        return "\n".join(lines)


class SpectralAnalyzer:
    """
    Lomb-Scargle periodogram analyzer for unevenly sampled time series.
    """
    
    def __init__(self) -> None:
        """Initialize the spectral analyzer."""
        self._lock = threading.RLock()
        self._last_result: Optional[SpectralResult] = None
    
    def analyze(
        self,
        time: npt.NDArray,
        values: npt.NDArray,
        frequency_range: Optional[Tuple[float, float]] = None,
        n_frequencies: int = 1000
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
            # Validate input
            time = validate_data_array(time, allow_nan=False, name="time")
            values = validate_data_array(values, allow_nan=False, name="values")
            
            if len(time) != len(values):
                raise ComputationError(
                    "Time and values arrays must have same length",
                    details={
                        "time_length": len(time),
                        "values_length": len(values)
                    }
                )
            
            if len(time) < 4:
                raise ComputationError(
                    "Need at least 4 data points for spectral analysis"
                )
            
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
                min_freq = 1.0 / (10 * time_span)  # Very low frequency
                max_freq = len(time) / (2 * time_span)  # Nyquist-like
            else:
                min_freq, max_freq = frequency_range
            
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
                peak_power=peak_power
            )
            
            self._last_result = result
            return result
    
    def _lomb_scargle(
        self,
        time: npt.NDArray,
        values: npt.NDArray,
        frequencies: npt.NDArray
    ) -> npt.NDArray:
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
            sum_sin2 = np.sum(sin_wt ** 2)
            sum_cos2 = np.sum(cos_wt ** 2)
            
            # Lomb-Scargle power
            if sum_sin2 > 0 and sum_cos2 > 0:
                power[i] = 0.5 * ((sum_sin ** 2) / sum_sin2 + (sum_cos ** 2) / sum_cos2)
            else:
                power[i] = 0
        
        # Normalize by variance
        power = power / variance
        
        return power
    
    def find_significant_peaks(
        self,
        result: Optional[SpectralResult] = None,
        threshold: float = 0.5
    ) -> List[Dict[str, float]]:
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
                    peaks.append({
                        'frequency': result.frequencies[local_max_idx],
                        'period': result.periods[local_max_idx],
                        'power': result.power[local_max_idx],
                        'relative_power': result.power[local_max_idx] / max_power
                    })
        
        # Check if still in peak at end
        if in_peak:
            peak_range = result.power[peak_start:]
            if len(peak_range) > 0:
                local_max_idx = np.argmax(peak_range) + peak_start
                peaks.append({
                    'frequency': result.frequencies[local_max_idx],
                    'period': result.periods[local_max_idx],
                    'power': result.power[local_max_idx],
                    'relative_power': result.power[local_max_idx] / max_power
                })
        
        # Sort by power
        peaks.sort(key=lambda x: x['power'], reverse=True)
        
        return peaks
    
    @property
    def last_result(self) -> Optional[SpectralResult]:
        """Get the last spectral result."""
        with self._lock:
            return self._last_result
