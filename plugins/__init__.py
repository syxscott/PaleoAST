# =============================================================================
# FILE: plugins/__init__.py
# =============================================================================
"""
Analysis Plugin System for PaleoAST

This package provides a plugin architecture for extensible statistical analyses.
Plugins can register themselves to be discovered and executed by the StatisticsController.

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .base import AnalysisPlugin, AnalysisResult
from .registry import AnalysisPluginRegistry, get_plugin_registry
from .decorators import register_analysis, auto_register
from .loader import load_builtin_plugins, discover_plugins_in_package

__all__ = [
    "AnalysisPlugin",
    "AnalysisResult",
    "AnalysisPluginRegistry",
    "get_plugin_registry",
    "register_analysis",
    "auto_register",
    "load_builtin_plugins",
    "discover_plugins_in_package",
]
