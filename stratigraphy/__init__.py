# =============================================================================
# FILE: stratigraphy/__init__.py
# =============================================================================
"""
PaleoAST Biostratigraphy Package

This package implements biological stratigraphy and time series analysis.

Modules:
    - unitary_associations: Unitary Associations method for stratigraphic correlation
    - spectral_analysis: Lomb-Scargle periodogram for time series

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .spectral_analysis import SpectralAnalyzer, SpectralResult

__all__ = [
    'SpectralAnalyzer',
    'SpectralResult',
]
