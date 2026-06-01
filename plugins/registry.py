# =============================================================================
# FILE: plugins/registry.py
# =============================================================================
"""
Plugin Registry for PaleoAST

Singleton registry that tracks all registered analysis plugins.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import threading

from .base import AnalysisPlugin

logger = logging.getLogger(__name__)


class AnalysisPluginRegistry:
    """
    Singleton registry for analysis plugins.

    Tracks all registered AnalysisPlugin instances and provides
    lookup and enumeration capabilities.
    """

    _instance: "AnalysisPluginRegistry | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "AnalysisPluginRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._logger = logging.getLogger(f"{__name__}.AnalysisPluginRegistry")
        self._plugins: dict[str, AnalysisPlugin] = {}
        self._logger.info("AnalysisPluginRegistry initialized")

    def register(self, plugin: AnalysisPlugin) -> None:
        """
        Register an analysis plugin.

        Parameters:
            plugin: The plugin instance to register

        Raises:
            ValueError: If a plugin with the same name is already registered
        """
        if not isinstance(plugin, AnalysisPlugin):
            raise TypeError(f"Plugin must be an AnalysisPlugin instance, got {type(plugin)}")
        name = plugin.name
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")
        self._plugins[name] = plugin
        self._logger.info(f"Registered plugin: {name} ({plugin.category})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.

        Parameters:
            name: Name of the plugin to unregister

        Returns:
            True if the plugin was found and removed, False otherwise
        """
        if name in self._plugins:
            del self._plugins[name]
            self._logger.info(f"Unregistered plugin: {name}")
            return True
        return False

    def get(self, name: str) -> AnalysisPlugin | None:
        """
        Get a registered plugin by name.

        Parameters:
            name: Plugin name

        Returns:
            The plugin instance, or None if not found
        """
        return self._plugins.get(name)

    def list_plugins(self, category: str | None = None) -> list[str]:
        """
        List all registered plugin names.

        Parameters:
            category: If provided, only return plugins in this category

        Returns:
            List of plugin names
        """
        if category is None:
            return list(self._plugins.keys())
        return [name for name, p in self._plugins.items() if p.category == category]

    def list_categories(self) -> list[str]:
        """List all unique plugin categories."""
        return list({p.category for p in self._plugins.values()})

    def get_all(self) -> dict[str, AnalysisPlugin]:
        """Get a copy of all registered plugins."""
        return self._plugins.copy()

    def clear(self) -> None:
        """Clear all registered plugins. For testing only."""
        self._plugins.clear()
        self._logger.info("Cleared all plugins")


def get_plugin_registry() -> AnalysisPluginRegistry:
    """Get the singleton AnalysisPluginRegistry instance."""
    return AnalysisPluginRegistry()
