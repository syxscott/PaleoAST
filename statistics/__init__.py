# =============================================================================
# FILE: statistics/__init__.py
# =============================================================================
"""
PaleoAST Statistics Package

This package contains statistical analysis engines including:
- PCA (Principal Component Analysis)
- PCoA (Principal Coordinate Analysis)
- NMDS (Non-metric Multidimensional Scaling)
- ANOSIM (Analysis of Similarities)
- PERMANOVA (Permutational MANOVA)
- Distance metrics computation

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .anosim import ANOSIMAnalyzer, ANOSIMResult
from .distance_metrics import DistanceMatrixResult, compute_distance_matrix
from .geometry import DisparityResult, GeometryAnalyzer, MSTResult
from .nmds import NMDSAnalyzer, NMDSResult
from .pca import PCAAnalyzer, PCAResult
from .pcoa import PCoAAnalyzer, PCoAResult
from .permanova import PERMANOVAAnalyzer, PERMANOVAResult

__all__ = [
    "ANOSIMAnalyzer",
    "ANOSIMResult",
    "DistanceMatrixResult",
    "DisparityResult",
    "GeometryAnalyzer",
    "MSTResult",
    "NMDSAnalyzer",
    "NMDSResult",
    "PCAAnalyzer",
    "PCAResult",
    "PCoAAnalyzer",
    "PCoAResult",
    "PERMANOVAAnalyzer",
    "PERMANOVAResult",
    "compute_distance_matrix",
]
