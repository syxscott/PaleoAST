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
from .correlation import (
    StratigraphicSection,
    StratigraphicCorrelationAnalyzer,
    StratigraphicCorrelationResult,
    AgeModelAnalyzer,
    AgeModelResult,
)
from .extinction import ExtinctionIntervalAnalyzer, ExtinctionIntervalResult
from .isotope_analysis import (
    IsotopeData,
    IsotopeResult,
    IsotopeTrend,
    Excursion,
    IsotopeAnalyzer,
)
from .spectral_analysis import SpectralAnalyzer, SpectralResult, WaveletResult

__all__ = [
    "AgeModelAnalyzer",
    "AgeModelResult",
    "ARMAResult",
    "ARMAAnalyzer",
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
