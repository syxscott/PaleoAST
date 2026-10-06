# =============================================================================
# FILE: stratigraphy/__init__.py
# =============================================================================
"""
PaleoAST Biostratigraphy Package

This package implements biological stratigraphy and time series analysis.

Modules:
    - biostratigraphy: UA and RASC methods for stratigraphic correlation
    - spectral_analysis: Lomb-Scargle periodogram and wavelet CWT for time series
    - time_bins: geologic time binning (palaeoverse-style) and occurrence binning

Author: PaleoAST Development Team
version: 1.1.0
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
from .cycles import (
    ACFResult,
    CycleAnalyzer,
    RedfitResult,
    ar1_prewhiten,
    autoassociation,
    autocorrelation,
    cross_correlation,
    insolation_series,
    mann_kendall,
    multitaper_spectrum,
    orbital_forcing,
    redfit,
    runs_test,
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
from .time_bins import (
    bin_time,
    get_scale,
    tax_expand_time,
    tax_range_time,
    time_bins,
)
from .timeaxis import TimeAxis

__all__ = [
    "ACFResult",
    "ARMAAnalyzer",
    "ARMAResult",
    "AgeModelAnalyzer",
    "AgeModelResult",
    "BioeventResult",
    "CycleAnalyzer",
    "Excursion",
    "ExtinctionIntervalAnalyzer",
    "ExtinctionIntervalResult",
    "ForecastResult",
    "IsotopeAnalyzer",
    "IsotopeData",
    "IsotopeResult",
    "IsotopeTrend",
    "RASCAnalyzer",
    "RedfitResult",
    "SpectralAnalyzer",
    "SpectralResult",
    "StratigraphicCorrelationAnalyzer",
    "StratigraphicCorrelationResult",
    "StratigraphicSection",
    "TimeAxis",
    "UAAnalyzer",
    "WaveletResult",
    "Zone",
    "ar1_prewhiten",
    "autoassociation",
    "autocorrelation",
    "bin_time",
    "cross_correlation",
    "get_scale",
    "insolation_series",
    "mann_kendall",
    "multitaper_spectrum",
    "orbital_forcing",
    "redfit",
    "runs_test",
    "tax_expand_time",
    "tax_range_time",
    "time_bins",
]
