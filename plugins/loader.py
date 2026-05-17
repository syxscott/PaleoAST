# =============================================================================
# FILE: plugins/loader.py
# =============================================================================
"""
Plugin Loader for PaleoAST

Utilities for discovering and loading analysis plugins.

Author: PaleoAST Development Team
Version: 1.0.0
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any

from .registry import get_plugin_registry

logger = logging.getLogger(__name__)

# Built-in plugin modules (discovered at import time)
_BUILTIN_PLUGINS = [
    "statistics.pca",
    "statistics.pcoa",
    "statistics.nmds",
    "statistics.anosim",
    "statistics.permanova",
    "statistics.simper",
    "ecology.diversity",
]


def load_builtin_plugins() -> int:
    """
    Load all built-in analysis plugins.

    This imports the known analysis modules which register
    themselves via decorators or explicit registration.

    Returns:
        Number of plugins successfully loaded
    """
    loaded = 0
    for module_name in _BUILTIN_PLUGINS:
        try:
            importlib.import_module(module_name)
            loaded += 1
            logger.debug(f"Loaded plugin module: {module_name}")
        except ImportError as e:
            logger.warning(f"Failed to load plugin module '{module_name}': {e}")
    return loaded


def discover_plugins_in_package(package_path: Path) -> list[str]:
    """
    Discover plugin modules within a package directory.

    Looks for Python files that define AnalysisPlugin subclasses.

    Parameters:
        package_path: Path to the package directory

    Returns:
        List of module names found
    """
    if not package_path.is_dir():
        return []

    modules = []
    for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if not ispkg and not modname.startswith("_"):
            modules.append(modname)
    return modules


__all__ = ["load_builtin_plugins", "discover_plugins_in_package"]
