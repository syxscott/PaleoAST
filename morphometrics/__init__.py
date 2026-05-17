# =============================================================================
# FILE: morphometrics/__init__.py
# =============================================================================
"""
PaleoAST Geometric Morphometrics Package

This package implements geometric morphometrics analysis for landmark-based
shape analysis of paleontological specimens.

Modules:
    - gpa: Generalized Procrustes Analysis
    - tps: Thin-Plate Spline analysis
    - relative_warps: Relative Warps Analysis

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .allometry import AllometryAnalyzer, IntegrationAnalyzer, AllometryResult, PLSResult
from .evolution_rate import EvolutionRateAnalyzer, EvolutionRateResult
from .gpa import GPAAnalyzer, GPAResult
from .relative_warps import RelativeWarpsAnalyzer, RelativeWarpsResult
from .tps import TPSAnalyzer, TPSResult

__all__ = [
    "AllometryAnalyzer",
    "AllometryResult",
    "EvolutionRateAnalyzer",
    "EvolutionRateResult",
    "GPAAnalyzer",
    "GPAResult",
    "IntegrationAnalyzer",
    "PLSResult",
    "RelativeWarpsAnalyzer",
    "RelativeWarpsResult",
    "TPSAnalyzer",
    "TPSResult",
]
