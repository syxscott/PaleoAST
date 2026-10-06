# =============================================================================
# FILE: plugins/__init__.py
# =============================================================================
"""
Analysis Plugin System for PaleoAST

This package provides a plugin architecture for extensible statistical analyses.
Plugins can register themselves to be discovered and executed by the StatisticsController.

Author: PaleoAST Development Team
version: 1.1.0
"""

from .base import AnalysisPlugin, AnalysisResult
from .decorators import auto_register, register_analysis
from .loader import discover_plugins_in_package
from .registry import AnalysisPluginRegistry, get_plugin_registry

__all__ = [
    "AnalysisPlugin",
    "AnalysisPluginRegistry",
    "AnalysisResult",
    "auto_register",
    "discover_plugins_in_package",
    "get_plugin_registry",
    "register_analysis",
]
