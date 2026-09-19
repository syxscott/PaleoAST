# =============================================================================
# FILE: morphometrics/__init__.py
# =============================================================================
"""
PaleoAST Geometric Morphometrics Package

This package implements geometric morphometrics analysis for landmark-based
shape analysis of paleontological specimens.

Modules:
    - gpa: Generalized Procrustes Analysis
    - curves: Curve topology helpers (triples validation, even resampling)
    - tps: Thin-Plate Spline analysis
    - relative_warps: Relative Warps Analysis

Author: PaleoAST Development Team
version: 1.0.1
"""

from .allometry import AllometryAnalyzer, AllometryResult, IntegrationAnalyzer, PLSResult
from .curves import evenly_resample_curve, interior_sliders, validate_curves
from .evolution_rate import EvolutionRateAnalyzer, EvolutionRateResult
from .gpa import GPAAnalyzer, GPAResult, PartialGPAResult, partial_gpa
from .missing import MissingEstimateResult, estimate_missing
from .relative_warps import RelativeWarpsAnalyzer, RelativeWarpsResult
from .tangent import orp, tangent_vectors
from .tps import TPSAnalyzer, TPSResult

__all__ = [
    "AllometryAnalyzer",
    "AllometryResult",
    "EvolutionRateAnalyzer",
    "EvolutionRateResult",
    "GPAAnalyzer",
    "GPAResult",
    "IntegrationAnalyzer",
    "MissingEstimateResult",
    "PLSResult",
    "PartialGPAResult",
    "RelativeWarpsAnalyzer",
    "RelativeWarpsResult",
    "TPSAnalyzer",
    "TPSResult",
    "estimate_missing",
    "evenly_resample_curve",
    "interior_sliders",
    "orp",
    "partial_gpa",
    "tangent_vectors",
    "validate_curves",
]
