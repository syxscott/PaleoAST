# =============================================================================
# FILE: controllers/__init__.py
# =============================================================================
"""
PaleoAST Controllers Package

This package implements the controller layer following MVC pattern,
bridging the UI (views) with the analysis engines (models).

Controllers:
    - statistics_controller: Coordinates statistical analyses
    - data_controller: Manages data operations

Author: PaleoAST Development Team
version: 1.0.1
"""

from .data_controller import DataController
from .statistics_controller import StatisticsController

__all__ = [
    "DataController",
    "StatisticsController",
]
