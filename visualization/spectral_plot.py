# =============================================================================
# FILE: visualization/spectral_plot.py
# =============================================================================
"""
Spectral Analysis Visualization Module for PaleoAST

This module implements visualization for spectral analysis results including:
    - Periodogram plots
    - Power spectra
    - Peak identification

Author: PaleoAST Development Team
Version: 1.0.0
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from stratigraphy.spectral_analysis import SpectralResult
from config.colors import get_color_scheme


class SpectralPlotter:
    """
    Publication-quality spectral visualization engine.
    """
    
    def __init__(self) -> None:
        """Initialize the spectral plotter."""
        self._style = 'seaborn-v0_8-paper'
        self._figure_size = (10, 6)
        self._dpi = 300
        self._font_size = 10
    
    def plot_periodogram(
        self,
        result: SpectralResult,
        show_peaks: bool = True,
        peak_threshold: float = 0.5,
        title: Optional[str] = None
    ) -> Figure:
        """
        Create periodogram plot showing power spectrum.
        
        Parameters:
            result: Spectral analysis result
            show_peaks: Whether to highlight significant peaks
            peak_threshold: Threshold for peak detection
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        plt.style.use(self._style)
        
        fig, axes = plt.subplots(2, 1, figsize=self._figure_size, gridspec_kw={'height_ratios': [3, 1]})
        
        # Main periodogram
        ax1 = axes[0]
        
        ax1.plot(
            result.periods[::-1],  # Reverse for traditional periodogram view
            result.power[::-1],
            '-',
            color='#2C3E50',
            linewidth=1.5
        )
        
        # Fill under curve
        ax1.fill_between(
            result.periods[::-1],
            result.power[::-1],
            alpha=0.3,
            color='#3498DB'
        )
        
        # Mark peaks
        if show_peaks and result.peak_period:
            ax1.axvline(
                x=result.peak_period,
                color='#E74C3C',
                linestyle='--',
                linewidth=1.5,
                label=f"Peak period: {result.peak_period:.2f}"
            )
            ax1.legend(loc='upper right')
        
        ax1.set_xlabel('Period', fontsize=self._font_size)
        ax1.set_ylabel('Power', fontsize=self._font_size)
        
        if title is None:
            title = "Lomb-Scargle Periodogram"
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Bottom panel: significance
        ax2 = axes[1]
        
        # Show period on x-axis
        ax2.set_xlabel('Period', fontsize=self._font_size)
        ax2.set_ylabel('Significance', fontsize=self._font_size)
        ax2.set_yticks([])
        
        # Color code by power
        max_power = np.max(result.power)
        normalized_power = result.power / max_power
        
        colors = ['#3498DB' if p < peak_threshold else '#E74C3C' for p in normalized_power]
        
        for i in range(len(result.periods) - 1):
            ax2.axvspan(
                result.periods[::-1][i],
                result.periods[::-1][i+1],
                alpha=0.3,
                color=colors[i]
            )
        
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlim(ax1.get_xlim())
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_spectrum(
        self,
        result: SpectralResult,
        title: Optional[str] = None
    ) -> Figure:
        """
        Create frequency-domain spectrum plot.
        
        Parameters:
            result: Spectral analysis result
            title: Plot title
        
        Returns:
            matplotlib Figure object
        """
        plt.style.use(self._style)
        
        fig, ax = plt.subplots(figsize=self._figure_size)
        
        ax.plot(
            result.frequencies,
            result.power,
            '-',
            color='#2C3E50',
            linewidth=1.5
        )
        
        ax.fill_between(
            result.frequencies,
            result.power,
            alpha=0.3,
            color='#3498DB'
        )
        
        # Mark peak
        if result.peak_frequency:
            ax.axvline(
                x=result.peak_frequency,
                color='#E74C3C',
                linestyle='--',
                linewidth=1.5,
                label=f"Peak: {result.peak_frequency:.4f}"
            )
            ax.legend(loc='upper right')
        
        ax.set_xlabel('Frequency', fontsize=self._font_size)
        ax.set_ylabel('Power', fontsize=self._font_size)
        
        if title is None:
            title = "Power Spectrum"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_spectral_summary(
        self,
        result: SpectralResult,
        peaks: Optional[List[Dict[str, float]]] = None
    ) -> Figure:
        """
        Create comprehensive spectral analysis summary.
        
        Parameters:
            result: Spectral analysis result
            peaks: List of peak information dictionaries
        
        Returns:
            matplotlib Figure object
        """
        plt.style.use(self._style)
        
        fig = plt.figure(figsize=(12, 8))
        
        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Periodogram (large)
        ax1 = fig.add_subplot(gs[0, :])
        
        ax1.plot(result.periods[::-1], result.power[::-1], '-', color='#2C3E50', linewidth=1.5)
        ax1.fill_between(result.periods[::-1], result.power[::-1], alpha=0.3, color='#3498DB')
        
        if result.peak_period:
            ax1.axvline(x=result.peak_period, color='#E74C3C', linestyle='--', linewidth=1.5)
        
        ax1.set_xlabel('Period', fontsize=10)
        ax1.set_ylabel('Power', fontsize=10)
        ax1.set_title('Lomb-Scargle Periodogram', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 2. Peak table
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        
        if peaks and len(peaks) > 0:
            table_text = "Significant Peaks\n" + "="*40 + "\n"
            table_text += f"{'Period':>10} | {'Power':>10} | {'Rel.':>8}\n"
            table_text += "-"*40 + "\n"
            
            for peak in peaks[:5]:  # Show top 5 peaks
                table_text += f"{peak['period']:>10.2f} | {peak['power']:>10.2f} | {peak['relative_power']:>7.2f}\n"
            
            ax2.text(0.1, 0.9, table_text, transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace')
        else:
            ax2.text(0.5, 0.5, "No significant peaks detected",
                     transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        
        # 3. Summary stats
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        stats_text = "Spectral Analysis Summary\n" + "="*40 + "\n"
        stats_text += f"Peak Period: {result.peak_period:.4f}\n" if result.peak_period else "Peak Period: N/A\n"
        stats_text += f"Peak Frequency: {result.peak_frequency:.6f}\n" if result.peak_frequency else "Peak Frequency: N/A\n"
        stats_text += f"Peak Power: {result.peak_power:.4f}\n" if result.peak_power else "Peak Power: N/A\n"
        stats_text += f"Total Frequencies: {len(result.frequencies)}\n"
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Spectral Analysis Results', fontsize=14, fontweight='bold')
        
        return fig
