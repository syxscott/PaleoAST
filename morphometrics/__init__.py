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
    - shape_stats: Procrustes shape statistics (medians, Goodall F, Hotelling T²)

Author: PaleoAST Development Team
version: 1.0.1
"""

from .allometry import AllometryAnalyzer, AllometryResult, IntegrationAnalyzer, PLSResult
from .curves import evenly_resample_curve, interior_sliders, validate_curves
from .evolution_rate import EvolutionRateAnalyzer, EvolutionRateResult
from .gpa import GPAAnalyzer, GPAResult, PartialGPAResult, partial_gpa
from .missing import MissingEstimateResult, estimate_missing
from .relative_warps import RelativeWarpsAnalyzer, RelativeWarpsResult
from .shape_stats import (
    GoodallResult,
    HotellingT2Result,
    geometric_median,
    goodall_f,
    goodall_test,
    hotelling_t2,
    kendall_preshape,
    procrustes_distance,
    procrustes_median,
)
from .tangent import orp, tangent_vectors
from .tps import TPSAnalyzer, TPSResult

__all__ = [
    "AllometryAnalyzer",
    "AllometryResult",
    "EvolutionRateAnalyzer",
    "EvolutionRateResult",
    "GPAAnalyzer",
    "GPAResult",
    "GoodallResult",
    "HotellingT2Result",
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
    "geometric_median",
    "goodall_f",
    "goodall_test",
    "hotelling_t2",
    "interior_sliders",
    "kendall_preshape",
    "orp",
    "partial_gpa",
    "procrustes_distance",
    "procrustes_median",
    "tangent_vectors",
    "validate_curves",
]
