# =============================================================================
# FILE: ecology/__init__.py
# =============================================================================
"""
PaleoAST Paleoecology Package

This package implements biodiversity and paleoecological diversity analysis.

Modules:
    - diversity: Alpha diversity indices
    - rarefaction: Rarefaction curve analysis
    - paleoenv: Correspondence Analysis (CA) paleo-environmental reconstruction

Author: PaleoAST Development Team
version: 1.0.1
"""

from .beta_diversity import (
    BetaDiversityAnalyzer,
    BetaDiversityResult,
    CoverageRarefactionAnalyzer,
    CoverageRarefactionResult,
)
from .diversity import DiversityAnalyzer, compute_diversity_indices
from .dtw import DTWAnalyzer, DTWResult
from .null_models import NullModelAnalyzer, NullModelResult
from .paleoenv import PaleoEnvironmentReconstructor, PaleoEnvironmentResult
from .rarefaction import RarefactionAnalyzer

__all__ = [
    "BetaDiversityAnalyzer",
    "BetaDiversityResult",
    "CoverageRarefactionAnalyzer",
    "CoverageRarefactionResult",
    "DTWAnalyzer",
    "DTWResult",
    "DiversityAnalyzer",
    "NullModelAnalyzer",
    "NullModelResult",
    "PaleoEnvironmentReconstructor",
    "PaleoEnvironmentResult",
    "RarefactionAnalyzer",
    "compute_diversity_indices",
]
