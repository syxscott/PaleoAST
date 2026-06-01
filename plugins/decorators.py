# =============================================================================
# FILE: plugins/decorators.py
# =============================================================================
"""
Plugin Registration Decorators for PaleoAST

Provides decorators for convenient plugin registration.

Author: PaleoAST Development Team
version: 1.0.1
"""

from typing import Any, Callable

from .base import AnalysisPlugin
from .registry import get_plugin_registry


def register_analysis(
    name: str,
    description: str = "",
    category: str = "general",
    cache_key: str | None = None,
) -> Callable[[type], type]:
    """
    Class decorator to register an AnalysisPlugin subclass.

    The decorated class must inherit from AnalysisPlugin and define
    `name`, `description`, and `analyze`.

    Parameters:
        name: Unique identifier for the analysis
        description: Human-readable description
        category: Grouping category
        cache_key: Custom cache key (defaults to '{name}_result')

    Returns:
        The decorated class unchanged

    Example:
        @register_analysis(
            name="my_analysis",
            description="Does something useful",
            category="custom"
        )
        class MyAnalysisPlugin(AnalysisPlugin):
            @property
            def name(self) -> str:
                return "my_analysis"

            @property
            def description(self) -> str:
                return "Does something useful"

            def analyze(self, data, **kwargs) -> AnalysisResult:
                ...
    """
    def decorator(cls: type) -> type:
        if not issubclass(cls, AnalysisPlugin):
            raise TypeError(f"{cls.__name__} must inherit from AnalysisPlugin")

        instance = cls()
        get_plugin_registry().register(instance)
        return cls

    return decorator


def auto_register(cls: type) -> type:
    """
    Class decorator that auto-registers any subclass of AnalysisPlugin.

    The subclass must define `name`, `description`, and `analyze`.

    Example:
        @auto_register
        class MyPlugin(AnalysisPlugin):
            @property
            def name(self) -> str:
                return "my_plugin"

            @property
            def description(self) -> str:
                return "My plugin"

            def analyze(self, data, **kwargs):
                return AnalysisResult(...)
    """
    if not issubclass(cls, AnalysisPlugin):
        raise TypeError(f"{cls.__name__} must inherit from AnalysisPlugin")

    get_plugin_registry().register(cls())
    return cls
