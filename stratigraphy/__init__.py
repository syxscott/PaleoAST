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
version: 1.0.1
"""

from .arma import ARMAAnalyzer, ARMAResult, ForecastResult
from .biostratigraphy import BioeventResult, RASCAnalyzer, UAAnalyzer, Zone
from .correlation import (
    AgeModelAnalyzer,
    AgeModelResult,
    StratigraphicCorrelationAnalyzer,
    StratigraphicCorrelationResult,
    StratigraphicSection,
)
from .extinction import ExtinctionIntervalAnalyzer, ExtinctionIntervalResult
from .isotope_analysis import (
    Excursion,
    IsotopeAnalyzer,
    IsotopeData,
    IsotopeResult,
    IsotopeTrend,
)
from .spectral_analysis import SpectralAnalyzer, SpectralResult, WaveletResult

__all__ = [
    "ARMAAnalyzer",
    "ARMAResult",
    "AgeModelAnalyzer",
    "AgeModelResult",
    "BioeventResult",
    "Excursion",
    "ExtinctionIntervalAnalyzer",
    "ExtinctionIntervalResult",
    "ForecastResult",
    "IsotopeAnalyzer",
    "IsotopeData",
    "IsotopeResult",
    "IsotopeTrend",
    "RASCAnalyzer",
    "SpectralAnalyzer",
    "SpectralResult",
    "StratigraphicCorrelationAnalyzer",
    "StratigraphicCorrelationResult",
    "StratigraphicSection",
    "UAAnalyzer",
    "WaveletResult",
    "Zone",
]
