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
version: 1.0.1
"""

from .beta_diversity import BetaDiversityAnalyzer, CoverageRarefactionAnalyzer, BetaDiversityResult, CoverageRarefactionResult
from .diversity import DiversityAnalyzer, compute_diversity_indices
from .dtw import DTWAnalyzer, DTWResult
from .null_models import NullModelAnalyzer, NullModelResult
from .rarefaction import RarefactionAnalyzer

__all__ = [
    "BetaDiversityAnalyzer",
    "BetaDiversityResult",
    "CoverageRarefactionAnalyzer",
    "CoverageRarefactionResult",
    "DiversityAnalyzer",
    "DTWAnalyzer",
    "DTWResult",
    "NullModelAnalyzer",
    "NullModelResult",
    "RarefactionAnalyzer",
    "compute_diversity_indices",
]
