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

from .pca import PCAAnalyzer, PCAResult
from .pcoa import PCoAAnalyzer, PCoAResult
from .nmds import NMDSAnalyzer, NMDSResult
from .anosim import ANOSIMAnalyzer, ANOSIMResult
from .permanova import PERMANOVAAnalyzer, PERMANOVAResult
from .distance_metrics import DistanceMatrix, compute_distance_matrix

__all__ = [
    'PCAAnalyzer',
    'PCAResult',
    'PCoAAnalyzer',
    'PCoAResult',
    'NMDSAnalyzer',
    'NMDSResult',
    'ANOSIMAnalyzer',
    'ANOSIMResult',
    'PERMANOVAAnalyzer',
    'PERMANOVAResult',
    'DistanceMatrix',
    'compute_distance_matrix',
]
