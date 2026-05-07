# =============================================================================
# FILE: ecology/__init__.py
# =============================================================================
"""
PaleoAST Paleoecology Package

This package implements biodiversity and paleoecological diversity analysis.

Modules:
    - diversity: Alpha diversity indices
    - rarefaction: Rarefaction curve analysis

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .diversity import DiversityAnalyzer, compute_diversity_indices
from .rarefaction import RarefactionAnalyzer

__all__ = [
    'DiversityAnalyzer',
    'compute_diversity_indices',
    'RarefactionAnalyzer',
]
