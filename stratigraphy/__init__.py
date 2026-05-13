# =============================================================================
# FILE: stratigraphy/__init__.py
# =============================================================================
"""
PaleoAST Biostratigraphy Package

This package implements biological stratigraphy and time series analysis.

Modules:
    - biostratigraphy: UA and RASC methods for stratigraphic correlation
    - spectral_analysis: Lomb-Scargle periodogram and wavelet CWT for time series

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .arma import ARMAResult, ARMAAnalyzer, ForecastResult
from .biostratigraphy import BioeventResult, RASCAnalyzer, UAAnalyzer, Zone
from .spectral_analysis import SpectralAnalyzer, SpectralResult, WaveletResult

__all__ = [
    "ARMAResult",
    "ARMAAnalyzer",
    "BioeventResult",
    "ForecastResult",
    "RASCAnalyzer",
    "SpectralAnalyzer",
    "SpectralResult",
    "UAAnalyzer",
    "WaveletResult",
    "Zone",
]
