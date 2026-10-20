# =============================================================================
# FILE: stats/univariate_menu.py
# =============================================================================
"""
The univariate extra tests, as the analysis runner can reach them.

WHY THIS EXISTS
---------------
``stats/univariate_extra.py`` holds six test families the application could
not run at all: Brunner-Munzel, Mood's median test, Fliegler-Kileen,
outlier screening, Cochran-Mantel-Haenszel and binomial proportion
inference. The analysis runner's contract is::

    plugin.analyze(data, **kwargs)

where ``data`` is the whole spreadsheet and ``**kwargs`` comes from a form
generated from the method's signature. ``analyze(x: NDArray, y: NDArray)``
handed that contract would receive the entire sheet as ``x`` and the form's
``y`` as the second positional argument -- a TypeError on the first click.

So this module is a thin facade, in the same shape as
``visualization/plot_menu.py``: every method takes the sheet first and
**column numbers** second, pulls the columns out, and delegates. No statistic
is recomputed here; a facade that recomputes anything is a second
implementation, and two implementations drift.

COLUMN NUMBERS, NOT COLUMN NAMES
--------------------------------
The generated form has no notion of "pick a column", so an integer is the
honest parameter type: ``x`` is the 0-based number of the column holding the
first sample. ``-1`` means "not set", the same sentinel the plot menu uses.

THE TWO ANALYSES THAT NEED NO DATA
----------------------------------
The Cochran-Mantel-Haenszel test works on four counts of a 2x2 table, and
the binomial test on two counts. Neither has a sheet to read, so their facade
methods take no data parameter at all: ``plugins.base.data_slot_name`` then
returns ``None`` and the plugin calls them with the form's arguments alone.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from utils.exceptions import DataValidationError, ValidationError

from .univariate_extra import (
    BrunnerMunzelAnalyzer,
    BrunnerMunzelResult,
    CochranMantelHaenszelAnalyzer,
    CochranMantelHaenszelResult,
    FlignerKileenAnalyzer,
    FlignerKileenResult,
    MoodMedianAnalyzer,
    MoodMedianResult,
    OutlierAnalyzer,
    OutlierResult,
    ProportionAnalyzer,
    ProportionTestResult,
)

logger = logging.getLogger(__name__)

__all__ = ["UnivariateExtraMenu"]

#: The alternatives the tests in this family accept. The generated form finds
#: this constant by scanning this module for a name carrying "ALTERNATIVE",
#: and offers it as a drop-down.
ALTERNATIVES = ("two-sided", "greater", "less")

#: How an observation is screened for being an outlier.
OUTLIER_METHODS = ("zscore", "grubbs")

#: The centre the Fliegler-Kileen test removes before ranking.
FLIGNER_CENTERS = ("median", "mean")


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


def _split_by_label(
    labels: npt.NDArray,
    observations: npt.NDArray,
) -> tuple[list[npt.NDArray], list[str]]:
    """Split one column of observations by a column of group labels.

    Parameters
    ----------
    labels:
        The group label of each observation.
    observations:
        The observation itself.

    Returns
    -------
    tuple[list[npt.NDArray], list[str]]
        One sample per group, in order of first appearance, and the group
        names in the same order.
    """
    buckets: dict[str, list[float]] = {}
    order: list[str] = []
    for label, value in zip(labels, observations, strict=True):
        key = str(label)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(float(value))
    if len(order) < 2:
        raise DataValidationError(f"this test compares at least 2 groups, but the group column names {len(order)}")
    return [np.asarray(buckets[key], dtype=float) for key in order], order


class UnivariateExtraMenu:
    """The univariate extra tests, reachable from the generated form.

    Every method takes the sheet as its first argument and column numbers for
    the rest, which is what the analysis runner's generated form can express.
    The two methods that need no data at all take no data parameter.
    """

    #: The analyzers this facade wraps. The catalog's staleness test uses
    #: this to know that the analyzers are reachable: a facade that is
    #: catalogued makes its analyzers reachable too.
    wrapped_analyzers = (
        BrunnerMunzelAnalyzer,
        CochranMantelHaenszelAnalyzer,
        FlignerKileenAnalyzer,
        MoodMedianAnalyzer,
        OutlierAnalyzer,
        ProportionAnalyzer,
    )

    def __init__(self) -> None:
        """Initialise the facade and the analyzers behind it."""
        self._logger = logging.getLogger(f"{__name__}.UnivariateExtraMenu")
        self._brunner_munzel = BrunnerMunzelAnalyzer()
        self._mood_median = MoodMedianAnalyzer()
        self._fligner_kileen = FlignerKileenAnalyzer()
        self._outlier = OutlierAnalyzer()
        self._cochran_mantel_haenszel = CochranMantelHaenszelAnalyzer()
        self._proportion = ProportionAnalyzer()

    # -- two-sample tests --------------------------------------------------

    def brunner_munzel(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        alternative: str = "two-sided",
    ) -> BrunnerMunzelResult:
        """Test two samples for stochastic equality (Brunner-Munzel test)."""
        m = _matrix(data)
        return self._brunner_munzel.analyze(
            _column(m, x, "x"),
            _column(m, y, "y"),
            alternative=alternative,
        )

    def mood_median(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        alternative: str = "two-sided",
    ) -> MoodMedianResult:
        """Test two samples for a common median (Mood's median test)."""
        m = _matrix(data)
        return self._mood_median.analyze(
            _column(m, x, "x"),
            _column(m, y, "y"),
            alternative=alternative,
        )

    def fligner_kileen(
        self,
        data: object,
        groups: int = 0,
        values: int = 1,
        center: str = "median",
    ) -> FlignerKileenResult:
        """Test whether several groups share a common dispersion (Fliegler-Kileen test)."""
        if center not in FLIGNER_CENTERS:
            raise ValidationError(f"center must be one of {FLIGNER_CENTERS}, got {center!r}")
        m = _matrix(data)
        samples, names = _split_by_label(
            _column(m, groups, "groups"),
            _column(m, values, "values"),
        )
        return self._fligner_kileen.analyze(samples, labels=names, center=center)

    # -- one-sample tests --------------------------------------------------

    def outlier_screen(
        self,
        data: object,
        values: int = 0,
        method: str = "zscore",
        alpha: float = 0.05,
    ) -> OutlierResult:
        """Screen one column for outliers, by z-score or by Grubbs' test."""
        if method not in OUTLIER_METHODS:
            raise ValidationError(f"method must be one of {OUTLIER_METHODS}, got {method!r}")
        m = _matrix(data)
        return self._outlier.analyze(
            _column(m, values, "values"),
            criterion=method,
            alpha=alpha,
        )

    # -- tests that need no sheet -------------------------------------------

    def cochran_mantel_haenszel(
        self,
        a: int = 10,
        b: int = 20,
        c: int = 30,
        d: int = 40,
        correction: bool = False,
    ) -> CochranMantelHaenszelResult:
        """Test a common association across a stratified 2x2 table (Cochran-Mantel-Haenszel test)."""
        table = np.asarray([[[a, b], [c, d]]], dtype=float)
        return self._cochran_mantel_haenszel.analyze(table, correction=correction)

    def proportion_test(
        self,
        successes: int = 5,
        trials: int = 20,
        p0: float = 0.5,
        alternative: str = "two-sided",
        confidence: float = 0.95,
    ) -> ProportionTestResult:
        """Test one proportion against a null value (exact binomial test)."""
        return self._proportion.test_proportion(
            successes=successes,
            trials=trials,
            p0=p0,
            alternative=alternative,
            confidence=confidence,
        )
