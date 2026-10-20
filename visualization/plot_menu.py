# =============================================================================
# FILE: visualization/plot_menu.py
# =============================================================================
"""
The plot menu, as the analysis runner can reach it.

WHY THIS EXISTS
---------------
``visualization/generic_plots.py`` has the twenty-one plots, each taking
plain arrays, which is the right shape for testing and for the script console.
It is the wrong shape for the analysis runner, whose contract is::

    plugin.analyze(data, **kwargs)

where ``data`` is the whole spreadsheet and ``**kwargs`` comes from a form
generated from the method's signature. ``plot_xy(x, y, ...)`` handed that
contract would receive the entire matrix as ``x`` and the form's ``y`` as the
second positional argument -- a TypeError on the first click.

So this module is a thin facade: every method takes the sheet first and
**column numbers** second, pulls the columns out, and delegates. The plotting
itself is not reimplemented here; a facade that recomputes anything is a second
implementation, and two implementations drift.

COLUMN NUMBERS, NOT COLUMN NAMES
--------------------------------
The runner's form is generated from annotations, and it has no notion of
"pick a column". An integer is therefore the honest parameter type: a spin box
sized to the sheet. ``-1`` means "not set", which is how an ungrouped scatter
and an uncategorised bar chart say so.

WHAT IS DELIBERATELY ABSENT
---------------------------
``plot_parametric_surface`` takes a Python callable and cannot be driven by a
form at all, so it is not catalogued. It stays reachable from the script
console, which can pass a lambda, and out of the menu, which cannot.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from utils.exceptions import DataValidationError

from .generic_plots import GenericPlotter

logger = logging.getLogger(__name__)

__all__ = ["PlotMenuAnalyzer"]

NOT_SET = -1


def _matrix(data: object) -> npt.NDArray:
    """The sheet as a 2-D array.

    The runner passes whatever the data provider holds, which is usually a
    matrix but is a bare column when one series is loaded. A 1-D input is
    promoted to a single column rather than rejected, because "one column"
    is a legitimate thing to plot and the user did not do anything wrong.

    A text column is kept as it is. Casting the whole sheet to float
    destroyed the column the plot menu is most often asked to draw -- a
    taxon name -- and raised a raw ValueError when it could not, so a
    sheet with any text in it could not be plotted at all. The plotter
    that draws a column knows whether it needs numbers, so the check
    belongs there.
    """
    arr = np.asarray(data, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DataValidationError(f"the plot menu needs a 1-D or 2-D data set; got shape {arr.shape}")
    if arr.size == 0:
        raise DataValidationError("the data set is empty")
    if all(_is_number(v) for v in arr.ravel()):
        # A sheet of numbers stays a float sheet, so the plotter gets
        # the dtype it expects.
        arr = arr.astype(float)
    return arr


def _is_number(value: object) -> bool:
    """True for a number, and for a missing value."""
    if value is None:
        return True
    return isinstance(value, (int, float, np.integer, np.floating))


def _column(matrix: npt.NDArray[np.float64], index: int, name: str) -> npt.NDArray[np.float64]:
    """One column by number, with the out-of-range case named.

    Falling back to the last column would silently plot the wrong variable,
    which is the failure this whole module exists to make impossible.
    """
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise DataValidationError(f"{name} must be a column number, got {index!r}")
    if index < 0 or index >= matrix.shape[1]:
        raise DataValidationError(f"{name} is column {index + 1}, but the data has {matrix.shape[1]} columns")
    return matrix[:, index]


def _labels(text: object, name: str) -> list[str] | None:
    """Group names, comma separated, for a grouped plot.

    The plotter factorises a grouping column by first appearance, so the
    names go in that order: the first name is the first new value the
    column shows, the second the next new one, and so on. They are
    optional -- without them the plotter falls back to "Group 1",
    "Group 2" -- but a legend of "Group 1" is not much use next to a
    column of taxon names.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise DataValidationError(f"{name} needs some names, e.g. T. rex, S. allops")
    return parts


def _ints(text: object, name: str) -> list[int]:
    """Parse a comma-separated integer list from a form field."""
    if isinstance(text, (list, tuple)):
        return [int(v) for v in text]
    raw = str(text).strip()
    if not raw:
        raise DataValidationError(f"{name} needs some values, e.g. 10, 20, 5")
    try:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise DataValidationError(f"{name} must be whole numbers separated by commas: {text!r}") from exc


class PlotMenuAnalyzer:
    """The plot menu, reachable by name from the ribbon and the console.

    Every method takes the sheet as its first argument and column numbers for
    the rest, which is what the analysis runner's generated form can express.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.PlotMenuAnalyzer")
        self._plotter = GenericPlotter()

    # -- XY ---------------------------------------------------------------

    def plot_xy(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        groups: int = NOT_SET,
        labels: str = "",
        title: str = "",
        log_x: bool = False,
        log_y: bool = False,
        fit_line: bool = False,
    ) -> Figure:
        m = _matrix(data)
        return self._plotter.plot_xy(
            _column(m, x, "x"),
            _column(m, y, "y"),
            groups=_column(m, groups, "groups") if groups != NOT_SET else None,
            labels=_labels(labels, "labels"),
            title=title,
            log_x=log_x,
            log_y=log_y,
            fit_line=fit_line,
        )

    def plot_xy_with_errorbars(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        y_error: int = 2,
        x_error: int = NOT_SET,
        title: str = "",
    ) -> Figure:
        m = _matrix(data)
        return self._plotter.plot_xy_with_errorbars(
            _column(m, x, "x"),
            _column(m, y, "y"),
            _column(m, y_error, "y_error"),
            x_error=_column(m, x_error, "x_error") if x_error != NOT_SET else None,
            title=title,
        )

    # -- univariate --------------------------------------------------------

    def plot_histogram(self, data: object, values: int = 0, bins: int = 20, title: str = "") -> Figure:
        return self._plotter.plot_histogram(_column(_matrix(data), values, "values"), bins=bins, title=title)

    def plot_normal_probability(self, data: object, values: int = 0, title: str = "") -> Figure:
        return self._plotter.plot_normal_probability(_column(_matrix(data), values, "values"), title=title)

    def plot_polar(self, data: object, values: int = 0, title: str = "") -> Figure:
        return self._plotter.plot_polar(_column(_matrix(data), values, "values"), title=title)

    def plot_percentiles(self, data: object, groups: int = 0, values: int = 1, title: str = "") -> Figure:
        m = _matrix(data)
        return self._plotter.plot_percentiles(_column(m, groups, "groups"), _column(m, values, "values"), title=title)

    def plot_box_plot(
        self, data: object, groups: int = 0, values: int = 1, show_points: bool = False, title: str = ""
    ) -> Figure:
        m = _matrix(data)
        return self._plotter.plot_box_plot(
            _column(m, groups, "groups"), _column(m, values, "values"), show_points=show_points, title=title
        )

    # -- categorical -------------------------------------------------------

    def plot_bar_chart(
        self, data: object, values: int = 0, categories: int = NOT_SET, horizontal: bool = False, title: str = ""
    ) -> Figure:
        m = _matrix(data)
        return self._plotter.plot_bar_chart(
            _column(m, values, "values"),
            categories=_column(m, categories, "categories") if categories != NOT_SET else None,
            horizontal=horizontal,
            title=title,
        )

    def plot_pie_chart(self, data: object, values: int = 0, categories: int = NOT_SET, title: str = "") -> Figure:
        m = _matrix(data)
        return self._plotter.plot_pie_chart(
            _column(m, values, "values"),
            categories=_column(m, categories, "categories") if categories != NOT_SET else None,
            title=title,
        )

    # -- composition -------------------------------------------------------

    def plot_stacked_chart(self, data: object, normalize: bool = False, title: str = "") -> Figure:
        return self._plotter.plot_stacked_chart(_matrix(data), normalize=normalize, title=title)

    def plot_matrix(self, data: object, diverging: bool = False, title: str = "") -> Figure:
        return self._plotter.plot_matrix(_matrix(data), diverging=diverging, title=title)

    def plot_mosaic(self, data: object, title: str = "") -> Figure:
        return self._plotter.plot_mosaic(_matrix(data), title=title)

    def plot_radar(self, data: object, title: str = "") -> Figure:
        return self._plotter.plot_radar(_matrix(data), title=title)

    def plot_surface(self, data: object, as_points: bool = False, title: str = "") -> Figure:
        return self._plotter.plot_surface(_matrix(data), as_points=as_points, title=title)

    def plot_ternary(self, data: object, a: int = 0, b: int = 1, c: int = 2, title: str = "") -> Figure:
        m = _matrix(data)
        return self._plotter.plot_ternary(_column(m, a, "a"), _column(m, b, "b"), _column(m, c, "c"), title=title)

    def plot_bubble(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        size: int = 2,
        groups: int = NOT_SET,
        labels: str = "",
        title: str = "",
    ) -> Figure:
        m = _matrix(data)
        return self._plotter.plot_bubble(
            _column(m, x, "x"),
            _column(m, y, "y"),
            _column(m, size, "size"),
            groups=_column(m, groups, "groups") if groups != NOT_SET else None,
            labels=_labels(labels, "labels"),
            title=title,
        )

    def plot_vectors(self, data: object, x: int = 0, y: int = 1, u: int = 2, v: int = 3, title: str = "") -> Figure:
        m = _matrix(data)
        return self._plotter.plot_vectors(
            _column(m, x, "x"), _column(m, y, "y"), _column(m, u, "u"), _column(m, v, "v"), title=title
        )

    def plot_3d(
        self,
        data: object,
        x: int = 0,
        y: int = 1,
        z: int = 2,
        groups: int = NOT_SET,
        labels: str = "",
        mode: str = "scatter",
        title: str = "",
    ) -> Figure:
        m = _matrix(data)
        return self._plotter.plot_3d(
            _column(m, x, "x"),
            _column(m, y, "y"),
            _column(m, z, "z"),
            groups=_column(m, groups, "groups") if groups != NOT_SET else None,
            labels=_labels(labels, "labels"),
            mode=mode,
            title=title,
        )

    # -- diagram kinds whose input is typed rather than picked -------------

    def plot_venn(self, data: object, counts: str = "10,20,5", title: str = "") -> Figure:
        """A Venn diagram needs region counts, which is a list not a column."""
        return self._plotter.plot_venn(_ints(counts, "counts"), title=title)

    def plot_network(self, data: object, edges: str = "0,1;1,2;2,0", title: str = "") -> Figure:
        """Edges are typed as ``0,1;1,2`` rather than picked from columns.

        A network's nodes are its indices, not its columns, so a column picker
        would be the wrong control.
        """
        raw = str(edges).strip()
        if not raw:
            raise DataValidationError("edges needs some pairs, e.g. 0,1;1,2")
        pairs = []
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(",")
            if len(parts) != 2:
                raise DataValidationError(f"each edge must be two node numbers; got {chunk!r}")
            try:
                pairs.append([int(parts[0].strip()), int(parts[1].strip())])
            except ValueError as exc:
                raise DataValidationError(f"node numbers must be whole numbers; got {chunk!r}") from exc
        if not pairs:
            raise DataValidationError("edges needs at least one pair")
        return self._plotter.plot_network(np.asarray(pairs, dtype=int), title=title)
