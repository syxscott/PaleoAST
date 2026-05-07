# =============================================================================
# FILE: visualization/__init__.py
# =============================================================================
"""
PaleoAST Visualization Package

This package implements publication-quality plotting for all analysis modules.

Modules:
    - pca_plot: PCA visualization
    - diversity_plot: Biodiversity plots
    - spectral_plot: Spectral analysis plots

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .pca_plot import PCAPlotter
from .diversity_plot import DiversityPlotter
from .spectral_plot import SpectralPlotter

__all__ = [
    'PCAPlotter',
    'DiversityPlotter',
    'SpectralPlotter',
]
