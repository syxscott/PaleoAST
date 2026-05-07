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

from .gpa import GPAAnalyzer, GPAResult
from .tps import TPSAnalyzer, TPSResult
from .relative_warps import RelativeWarpsAnalyzer, RelativeWarpsResult

__all__ = [
    'GPAAnalyzer',
    'GPAResult',
    'TPSAnalyzer',
    'TPSResult',
    'RelativeWarpsAnalyzer',
    'RelativeWarpsResult',
]
