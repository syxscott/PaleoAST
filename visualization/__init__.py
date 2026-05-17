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
    - allometry_plot: Allometry and morphological integration plots
    - evo_rate_plot: Evolutionary rate and phenogram plots
    - stratigraphy_plot: Stratigraphic range and extinction interval plots

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .allometry_plot import AllometryPlotter
from .diversity_plot import DiversityPlotter
from .evo_rate_plot import EvolutionRatePlotter
from .pca_plot import PCAPlotter
from .spectral_plot import SpectralPlotter
from .stratigraphy_plot import StratigraphyPlotter

__all__ = [
    "AllometryPlotter",
    "DiversityPlotter",
    "EvolutionRatePlotter",
    "PCAPlotter",
    "SpectralPlotter",
    "StratigraphyPlotter",
]
