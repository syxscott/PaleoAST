# =============================================================================
# FILE: plugins/base.py
# =============================================================================
"""
Base Plugin Classes for PaleoAST Analysis Plugins

Author: PaleoAST Development Team
version: 1.1.0
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# The parameter an analyzer's data belongs in, found from its signature.
#
# The plugin contract is ``analyze(data, **kwargs)`` with ``data`` the whole
# spreadsheet, but the target's first positional parameter is not always
# the data. ``growth_model(model, times, values)`` takes a model name
# first, and ``orbital_forcing(period_kyr, ...)`` takes no data at all.
# Handing the sheet to the first positional parameter regardless gave the
# sheet to ``model`` and to ``period_kyr``.
#
# The rule: the first parameter whose annotation says it is a numpy array,
# or the one named "data" (the plot-menu facade writes ``data: object``).
# Everything else is left to the caller.
_DATA_SLOT_NAMES = frozenset({"data"})


def data_slot_name(signature: inspect.Signature) -> str | None:
    """The parameter the data belongs in, or None when there is none.

    Parameters
    ----------
    signature:
        The signature of the target method.

    Returns
    -------
    str | None
        The parameter name, or None for a target that takes no data.
    """
    for name, param in signature.parameters.items():
        if name in ("self", "kwargs", "args"):
            continue
        if name in _DATA_SLOT_NAMES:
            return name
        annotation = param.annotation
        text = annotation if isinstance(annotation, str) else str(annotation)
        if "ndarray" in text.lower():
            return name
    return None


@dataclass
class AnalysisResult:
    """Standard container for analysis results.

    Attributes:
        name: The analysis that produced it.
        data: Whatever the analysis returned.
        metadata: Where it came from, and its summary.
        success: False when the analysis reported a failure instead of
            raising one.
        error: The failure message, when there was one.

    An analyzer that *raises* is not turned into a failed result: the
    exception propagates, because the project's exception types carry
    information a message does not and callers match on them. A plugin
    that wants to report a failure without raising builds one with
    :meth:`failure`.
    """

    name: str
    data: Any
    metadata: dict[str, Any]
    success: bool = True
    error: str | None = None

    @classmethod
    def failure(
        cls,
        name: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> "AnalysisResult":
        """A result that reports a failure rather than raising one.

        Parameters
        ----------
        name:
            The analysis that failed.
        error:
            The message describing the failure.
        metadata:
            Where the analysis came from.

        Returns
        -------
        AnalysisResult
            A result with ``success`` False and ``error`` set.
        """
        return cls(
            name=name,
            data=None,
            metadata=dict(metadata or {}),
            success=False,
            error=error,
        )


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
