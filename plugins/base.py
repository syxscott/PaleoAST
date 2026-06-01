# =============================================================================
# FILE: plugins/base.py
# =============================================================================
"""
Base Plugin Classes for PaleoAST Analysis Plugins

Author: PaleoAST Development Team
version: 1.0.1
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AnalysisResult:
    """Standard container for analysis results."""

    name: str
    data: Any
    metadata: dict[str, Any]
    success: bool = True
    error: str | None = None


class AnalysisPlugin(ABC):
    """
    Base class for analysis plugins.

    All analysis plugins must inherit from this class and implement
    the `analyze` method.

    Properties:
        name: Unique identifier for the analysis
        description: Human-readable description
        category: Grouping category (e.g., 'ordination', 'clustering', 'diversity')
        cache_key: Key used when caching results (None = no caching)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this analysis."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this analysis does."""
        raise NotImplementedError

    @property
    def category(self) -> str:
        """Category for grouping analyses."""
        return "general"

    @property
    def cache_key(self) -> str | None:
        """Cache key for this analysis. Return None to disable caching."""
        return f"{self.name}_result"

    @abstractmethod
    def analyze(self, data: Any, **kwargs: Any) -> AnalysisResult:
        """
        Execute the analysis.

        Parameters:
            data: Input data (typically numpy array or distance matrix)
            **kwargs: Additional parameters specific to the analysis

        Returns:
            AnalysisResult: The analysis outcome
        """
        raise NotImplementedError
