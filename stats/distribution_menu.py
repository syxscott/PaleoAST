# =============================================================================
# FILE: stats/distribution_menu.py
# =============================================================================
"""
The two-sample distribution tests, as the analysis runner can reach them.

WHY THIS EXISTS
---------------
``stats/distribution_tests.py`` holds the Kolmogorov-Smirnov two-sample
test and the Epps-Singleton test, and the application could run neither: the
analysis runner's contract is::

    plugin.analyze(data, **kwargs)

where ``data`` is the whole spreadsheet and ``**kwargs`` comes from a form
generated from the method's signature. ``ks_2samp(sample1: NDArray,
sample2: NDArray)`` handed that contract would receive the entire sheet as
``sample1`` and the form's ``sample2`` as the second positional argument --
a TypeError on the first click.

So this module is a thin facade, in the same shape as
``visualization/plot_menu.py``: every method takes the sheet first and
**column numbers** second, pulls the columns out, and delegates. No statistic
is recomputed here; a facade that recomputes anything is a second
implementation, and two implementations drift.

COLUMN NUMBERS, NOT COLUMN NAMES
--------------------------------
The generated form has no notion of "pick a column", so an integer is the
honest parameter type: ``x`` is the 0-based number of the column holding the
first sample.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from utils.exceptions import DataValidationError

from .distribution_tests import DistributionTestAnalyzer, EppsSingletonResult, KSTestResult

logger = logging.getLogger(__name__)

__all__ = ["DistributionTestMenu"]

#: How the Kolmogorov-Smirnov null is computed. The generated form finds
#: this constant by scanning this module for a name carrying "METHOD", and
#: offers it as a drop-down.
KS_METHODS = ("exact", "asymp")


def _matrix(data: object) -> npt.NDArray:
    """The sheet as a 2-D float matrix.

    A 1-D input is promoted to a single column rather than rejected, because
    "one column" is a legitimate thing to test and the user did not do
    anything wrong.
    """
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "these tests need numeric columns; the sheet has a column that is not numeric"
        ) from exc
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DataValidationError(f"these tests need a 1-D or 2-D data set; got shape {arr.shape}")
    if arr.size == 0:
        raise DataValidationError("the data set is empty")
    return arr


def _column(matrix: npt.NDArray, index: int, name: str) -> npt.NDArray:
    """One column by number, with the out-of-range case named.

    Falling back to the last column would silently test the wrong variable,
    which is the failure this whole module exists to make impossible.
    """
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise DataValidationError(f"{name} must be a column number, got {index!r}")
    if index < 0 or index >= matrix.shape[1]:
        raise DataValidationError(f"{name} is column {index + 1}, but the data has {matrix.shape[1]} columns")
    return matrix[:, index]


class DistributionTestMenu:
    """The two-sample distribution tests, reachable from the generated form.

    Every method takes the sheet as its first argument and column numbers for
    the rest, which is what the analysis runner's generated form can express.
    """

    #: The analyzers this facade wraps. The catalog's staleness test uses
    #: this to know that the analyzers are reachable: a facade that is
    #: catalogued makes its analyzers reachable too.
    wrapped_analyzers = (DistributionTestAnalyzer,)

    def __init__(self) -> None:
        """Initialise the facade and the analyzer behind it."""
        self._logger = logging.getLogger(f"{__name__}.DistributionTestMenu")
        self._inner = DistributionTestAnalyzer()

    def ks_2samp(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        method: str = "exact",
    ) -> KSTestResult:
        """Test two samples for equal distributions (Kolmogorov-Smirnov test)."""
        m = _matrix(data)
        return self._inner.ks_2samp(
            _column(m, x, "x"),
            _column(m, y, "y"),
            method=method,
        )

    def epps_singleton(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        n_permutations: int = 999,
        random_seed: int | None = None,
    ) -> EppsSingletonResult:
        """Test two samples for equal distributions (Epps-Singleton test)."""
        m = _matrix(data)
        return self._inner.epps_singleton(
            _column(m, x, "x"),
            _column(m, y, "y"),
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
