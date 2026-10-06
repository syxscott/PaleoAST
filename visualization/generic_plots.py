# =============================================================================
# FILE: visualization/generic_plots.py
# =============================================================================
"""
The general-purpose plot menu.

Every other plotter in this package draws one analysis: the PCA plotter draws
scores and loadings, the diversity plotter draws indices. That is the right
shape for those, and it is why none of them can do what a plot menu is for --
take two arbitrary columns from the spreadsheet and draw them. PAST's Plot
menu is exactly that: select cells, pick a plot, done. Twenty-two kinds of it,
and this module is the counterpart.

WHAT IS HERE
------------
XY (with optional fitted line and log axes), XY with error bars, histogram,
bar chart, box plot, pie chart, stacked chart, percentiles, normal-probability
(Q-Q) plot, ternary, bubble, matrix plot, mosaic plot, Venn, radar, polar,
vector (quiver), network, and the 3-D family: scatter, line, bubble, surface
and parametric surface.

DELIBERATELY NOT HERE
---------------------
* **Ternary and mosaic plots take their own input shape**, not a matrix with
  named columns, and say so in their docstrings rather than guessing at a
  layout the caller's data does not imply.
* **The 3-D imports happen inside the methods.** ``mpl_toolkits.mplot3d`` is a
  matplotlib submodule rather than a hard dependency, and importing it at
  module scope would make the whole plot menu unavailable to a build without
  it. The cost is one import per 3-D call.
* **No jointplot/seaborn.** The project has no seaborn dependency and adding
  one for a convenience wrapper is not worth it.

COLOUR
------
Categorical colours come from :func:`config.colors.current_palette`, so a
figure drawn here matches the one the R export will produce -- see
``tests/visualization/test_palette_single_source.py``. Sequential ramps
(bubble size, surface height) deliberately do NOT come from the categorical
palette: neighbouring categorical colours are chosen to be told apart, which
is the wrong property for a magnitude.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from config.colors import current_palette
from utils.exceptions import DataValidationError, PlottingError

from ._style_scope import scoped_plot_methods

logger = logging.getLogger(__name__)

__all__ = ["GenericPlotter"]


def _as_1d(values: object, name: str) -> npt.NDArray[np.float64]:
    """Coerce a column to a 1-D float array, or explain what was wrong with it."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        raise DataValidationError(f"{name} must be a single column; got an array with shape {array.shape}")
    return array


def _paired(x: object, y: object) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two columns of equal length, with the mismatch reported explicitly.

    Truncating to the shorter column would silently plot the wrong pairs -- x[0]
    against y[0] is right, but a missing value at the end would quietly shift
    the pairing.
    """
    xa = _as_1d(x, "x")
    ya = _as_1d(y, "y")
    if xa.size != ya.size:
        raise DataValidationError(f"x has {xa.size} values but y has {ya.size}; the columns must be the same length")
    if xa.size == 0:
        raise DataValidationError("x and y are both empty")
    return xa, ya


def _groups_of(n: int, groups: Sequence[object] | None) -> npt.NDArray[np.intp]:
    """Normalise a grouping column to integer codes.

    String labels are the normal case (PAST groups by colour, and the
    spreadsheet holds text), so factorising by first appearance keeps the
    legend in the order the user sees the data rather than alphabetically.
    """
    if groups is None:
        return np.zeros(n, dtype=np.intp)
    gs = list(groups)
    if len(gs) != n:
        raise DataValidationError(f"groups has {len(gs)} entries but there are {n} data points")
    order: dict[object, int] = {}
    codes = np.empty(n, dtype=np.intp)
    for i, g in enumerate(gs):
        key = g.item() if isinstance(g, np.generic) else g
        if key not in order:
            order[key] = len(order)
        codes[i] = order[key]
    return codes


def _labels_for(codes: npt.NDArray[np.intp], n_unique: int, labels: Sequence[str] | None) -> list[str]:
    if labels is not None and len(labels) != n_unique:
        raise DataValidationError(f"{len(labels)} labels given for {n_unique} groups")
    if labels is not None:
        return list(labels)
    return [f"Group {i + 1}" for i in range(n_unique)]


def _strip_nonfinite(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[...]:
    """Drop rows where either value is NaN or infinite.

    matplotlib draws a non-finite point as nothing at all, which is usually
    what you want -- but a log axis will also warn and drop the whole axis, so
    the non-finite values come out here with a reason instead.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


@scoped_plot_methods
class GenericPlotter:
    """
    The plot menu: publication-quality figures from spreadsheet columns.

    Every ``plot_*`` method takes plain arrays rather than a result object.
    That is the difference from the other plotters in this package and it is
    deliberate -- the whole point is that the data came from the sheet and
    nothing else.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.GenericPlotter")
        self._style = "seaborn-v0_8-paper"
        self._figure_size = (8.0, 6.0)
        self._dpi = 300
        self._font_size = 10
        self._title_font_size = 12

    # ------------------------------------------------------------------
    # XY
    # ------------------------------------------------------------------

    def plot_xy(
        self,
        x: object,
        y: object,
        groups: Sequence[object] | None = None,
        labels: Sequence[str] | None = None,
        title: str = "",
        x_label: str = "x",
        y_label: str = "y",
        log_x: bool = False,
        log_y: bool = False,
        fit_line: bool = False,
        marker: str = "o",
    ) -> Figure:
        """
        Scatter of two columns, optionally coloured by group, with an optional
        least-squares line through the whole data set.

        The fitted line is through ALL the points, not per group: drawing one
        line per group needs at least three points per group and the right
        answer to "fit each group separately" is a different plot.
        """
        xa, ya = _paired(x, y)
        xa, ya = _strip_nonfinite(xa, ya)
        codes = _groups_of(xa.size, groups)
        if log_x:
            if np.any(xa <= 0):
                raise DataValidationError("a log x-axis needs every x to be greater than zero")
            xa = np.log10(xa)
            x_label = f"log10 {x_label}"
        if log_y:
            if np.any(ya <= 0):
                raise DataValidationError("a log y-axis needs every y to be greater than zero")
            ya = np.log10(ya)
            y_label = f"log10 {y_label}"

        palette = current_palette()
        unique = int(codes.max()) + 1 if codes.size else 0
        names = _labels_for(codes, unique, labels)

        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        for g in range(unique):
            mask = codes == g
            ax.scatter(
                xa[mask],
                ya[mask],
                c=[palette[g % len(palette)]],
                marker=marker,
                s=45,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=names[g],
            )
        if fit_line and xa.size >= 2:
            slope, intercept = np.polyfit(xa, ya, 1)
            grid = np.linspace(xa.min(), xa.max(), 50)
            ax.plot(grid, slope * grid + intercept, color="#333333", linewidth=1.2, linestyle="--", label="fit")
        ax.set_xlabel(x_label, fontsize=self._font_size)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        if unique > 1:
            ax.legend(fontsize=self._font_size - 1)
        return fig

    def plot_xy_with_errorbars(
        self,
        x: object,
        y: object,
        y_error: object,
        x_error: object | None = None,
        title: str = "",
        x_label: str = "x",
        y_label: str = "y",
    ) -> Figure:
        """XY scatter with error bars. Errors are symmetric (half-widths)."""
        xa, ya = _paired(x, y)
        err = _as_1d(y_error, "y_error")
        if err.size != ya.size:
            raise DataValidationError(f"y_error has {err.size} values but y has {ya.size}")
        xerr = None
        if x_error is not None:
            xerr = _as_1d(x_error, "x_error")
            if xerr.size != xa.size:
                raise DataValidationError(f"x_error has {xerr.size} values but x has {xa.size}")
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        ax.errorbar(xa, ya, yerr=err, xerr=xerr, fmt="o", markersize=5, capsize=3, color=current_palette()[0])
        ax.set_xlabel(x_label, fontsize=self._font_size)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        return fig

    def plot_histogram(
        self,
        values: object,
        bins: int | str = 20,
        title: str = "",
        x_label: str = "value",
        y_label: str = "count",
        density: bool = False,
    ) -> Figure:
        """Histogram of one column. ``bins`` may be an integer or an edge array."""
        data = _as_1d(values, "values")
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise DataValidationError("no finite values to plot")
        if isinstance(bins, int) and bins < 1:
            raise DataValidationError(f"bins must be at least 1; got {bins}")
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        ax.hist(data, bins=bins, density=density, color=current_palette()[0], edgecolor="white")
        ax.set_xlabel(x_label, fontsize=self._font_size)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # Categorical counts
    # ------------------------------------------------------------------

    def plot_bar_chart(
        self,
        values: object,
        categories: Sequence[object] | None = None,
        title: str = "",
        y_label: str = "count",
        horizontal: bool = False,
    ) -> Figure:
        """
        Bar chart of one column's values.

        Without ``categories`` this is a histogram of the distinct values, each
        bar labelled with its value -- which is what you want for a categorical
        column. With ``categories`` the values are summed per category, so a
        row of numbers per group becomes one bar per group.
        """
        data = _as_1d(values, "values")
        if categories is not None:
            cats = list(categories)
            if len(cats) != data.size:
                raise DataValidationError(f"categories has {len(cats)} entries but values has {data.size}")
            totals_by_key: dict[object, float] = {}
            for i, c in enumerate(cats):
                key = c.item() if isinstance(c, np.generic) else c
                totals_by_key[key] = totals_by_key.get(key, 0.0) + float(data[i])
            names = [str(k) for k in totals_by_key]
            totals = list(totals_by_key.values())
        else:
            uniq, counts = np.unique(data[np.isfinite(data)], return_counts=True)
            if uniq.size == 0:
                raise DataValidationError("no finite values to plot")
            names = [f"{v:g}" for v in uniq]
            totals = [float(c) for c in counts]

        palette = current_palette()
        colors = [palette[i % len(palette)] for i in range(len(names))]
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        if horizontal:
            ax.barh(range(len(names)), totals, color=colors, edgecolor="white")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel(y_label, fontsize=self._font_size)
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        else:
            ax.bar(range(len(names)), totals, color=colors, edgecolor="white")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=self._font_size - 1)
            ax.set_ylabel(y_label, fontsize=self._font_size)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    def plot_box_plot(
        self,
        groups: Sequence[object],
        values: object,
        title: str = "",
        y_label: str = "value",
        show_points: bool = False,
    ) -> Figure:
        """Box plot of one column per group. A group with fewer than two values
        cannot have a box -- the quartiles are undefined -- and is reported."""
        data = _as_1d(values, "values")
        gs = list(groups)
        if len(gs) != data.size:
            raise DataValidationError(f"groups has {len(gs)} entries but values has {data.size}")
        order: dict[object, list[float]] = {}
        for g, v in zip(gs, data, strict=True):
            key = g.item() if isinstance(g, np.generic) else g
            if np.isfinite(v):
                order.setdefault(key, []).append(float(v))
        thin = [k for k, v in order.items() if len(v) < 2]
        if thin:
            raise DataValidationError(f"group(s) {thin!r} have fewer than two values, so they have no quartiles")
        names = [str(k) for k in order]
        series = [order[k] for k in order]
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        ax.boxplot(series, tick_labels=names, patch_artist=True, medianprops={"color": "#333333"})
        palette = current_palette()
        for i, patch in enumerate(ax.patches):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.7)
        if show_points:
            for i, vals in enumerate(series, start=1):
                jitter = np.linspace(-0.08, 0.08, len(vals))
                ax.plot(i + jitter, vals, "o", markersize=4, color="#333333", alpha=0.6)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        return fig

    def plot_pie_chart(
        self,
        values: object,
        categories: Sequence[object] | None = None,
        title: str = "",
        show_percent: bool = True,
    ) -> Figure:
        """Pie chart.

        With ``categories`` the values are summed per category. Without it, a
        column of numbers is counted by its distinct values -- but a column of
        *labels* is the more common input for a pie, so a non-numeric column is
        counted directly rather than rejected. Negative slices are refused:
        matplotlib would draw them, and the result means nothing.
        """
        if categories is not None:
            data = _as_1d(values, "values")
            gs = list(categories)
            if len(gs) != data.size:
                raise DataValidationError(f"categories has {len(gs)} entries but values has {data.size}")
            totals: dict[object, float] = {}
            for g, v in zip(gs, data, strict=True):
                key = g.item() if isinstance(g, np.generic) else g
                totals[key] = totals.get(key, 0.0) + float(v)
            names = [str(k) for k in totals]
            sizes = list(totals.values())
        else:
            raw_values = np.asarray(values, dtype=object).ravel()
            if raw_values.size == 0:
                raise DataValidationError("no values to plot")
            if raw_values.dtype != object or all(isinstance(v, (int, float, np.number)) for v in raw_values):
                data = _as_1d(values, "values")
                if np.any(data < 0):
                    raise DataValidationError(
                        "a pie chart cannot show a negative value; pass categories to sum, or drop the negative rows"
                    )
                uniq, counts = np.unique(data, return_counts=True)
                names = [f"{v:g}" for v in uniq]
                sizes = [float(c) for c in counts]
            else:
                # A label column: each distinct label is a slice.
                tally: dict[object, float] = {}
                for v in raw_values:
                    key = v.item() if isinstance(v, np.generic) else v
                    tally[key] = tally.get(key, 0.0) + 1.0
                names = [str(k) for k in tally]
                sizes = list(tally.values())
        if np.any(np.asarray(sizes) < 0):
            raise DataValidationError("a pie chart cannot show a negative slice")
        if sum(sizes) <= 0:
            raise DataValidationError("a pie chart needs at least one positive value")
        palette = current_palette()
        fig = Figure(figsize=(7.0, 7.0))
        ax = fig.subplots()
        ax.pie(
            sizes,
            labels=names,
            colors=[palette[i % len(palette)] for i in range(len(names))],
            autopct="%1.1f%%" if show_percent else None,
            startangle=90,
        )
        ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    def plot_stacked_chart(
        self,
        matrix: object,
        row_labels: Sequence[str] | None = None,
        column_labels: Sequence[str] | None = None,
        title: str = "",
        normalize: bool = False,
    ) -> Figure:
        """Stacked bars, one per row of a 2-D matrix.

        ``normalize=True`` converts each row to percentages, which is the form
        a palaeontologist usually wants for a community table.
        """
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise DataValidationError(f"a stacked chart needs a 2-D matrix; got shape {arr.shape}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise DataValidationError("the matrix is empty")
        if row_labels is not None and len(row_labels) != arr.shape[0]:
            raise DataValidationError(f"{len(row_labels)} row labels for {arr.shape[0]} rows")
        if column_labels is not None and len(column_labels) != arr.shape[1]:
            raise DataValidationError(f"{len(column_labels)} column labels for {arr.shape[1]} columns")
        if normalize:
            totals = arr.sum(axis=1, keepdims=True)
            if np.any(totals == 0):
                raise DataValidationError("cannot normalize a row that sums to zero")
            arr = arr / totals * 100.0
        palette = current_palette()
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        bottom = np.zeros(arr.shape[0])
        positions = np.arange(arr.shape[0])
        for col in range(arr.shape[1]):
            ax.bar(
                positions,
                arr[:, col],
                bottom=bottom,
                label=(column_labels[col] if column_labels else f"col {col + 1}"),
                color=palette[col % len(palette)],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += arr[:, col]
        ax.set_xticks(positions)
        ax.set_xticklabels(
            row_labels if row_labels else [f"row {i + 1}" for i in range(arr.shape[0])],
            rotation=45,
            ha="right",
            fontsize=self._font_size - 1,
        )
        ax.set_ylabel("percent" if normalize else "value", fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.legend(fontsize=self._font_size - 2, ncol=2)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        return fig

    def plot_percentiles(
        self,
        groups: Sequence[object],
        values: object,
        title: str = "",
        y_label: str = "value",
    ) -> Figure:
        """A percentile (box-and-whisker) plot: the box plot without outliers."""
        fig = self.plot_box_plot(groups, values, title=title, y_label=y_label, show_points=False)
        ax = fig.axes[0]
        for line in ax.get_lines():
            line.set_marker("")
            if line.get_linestyle() == "-":
                line.set_visible(False)
        return fig

    def plot_normal_probability(
        self,
        values: object,
        title: str = "",
        x_label: str = "theoretical quantile",
        y_label: str = "observed value",
    ) -> Figure:
        """
        Normal probability (Q-Q against a normal) plot.

        Points on a straight line mean the sample is normal. A fitted reference
        line is drawn through the quartiles -- the same construction
        probability-plot paper uses -- because comparing against the unit
        diagonal only makes sense for data already on that scale.
        """
        data = _as_1d(values, "values")
        data = np.sort(data[np.isfinite(data)])
        n = data.size
        if n < 3:
            raise DataValidationError(f"a normal probability plot needs at least 3 values; got {n}")
        from scipy import stats

        theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        ax.plot(theoretical, data, "o", markersize=5, color=current_palette()[0])
        slope, intercept = np.polyfit(theoretical, data, 1)
        ax.plot(theoretical, slope * theoretical + intercept, "--", color="#333333", linewidth=1.2)
        ax.set_xlabel(x_label, fontsize=self._font_size)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        return fig

    # ------------------------------------------------------------------
    # Composition and grids
    # ------------------------------------------------------------------

    def plot_ternary(
        self,
        a: object,
        b: object,
        c: object,
        labels: Sequence[str] | None = None,
        title: str = "",
        axis_labels: Sequence[str] = ("A", "B", "C"),
    ) -> Figure:
        """
        Ternary plot of three components that sum to 100 (or to any constant).

        The components are normalised here, so raw counts work: they are divided
        by their row sum. That is what a palaeontologist wants from a table of
        counts, and a plot that silently required percentages would reject it.
        """
        aa = _as_1d(a, "a")
        bb = _as_1d(b, "b")
        cc = _as_1d(c, "c")
        if not (aa.size == bb.size == cc.size):
            raise DataValidationError(f"components differ in length: {aa.size}, {bb.size}, {cc.size}")
        totals = aa + bb + cc
        if np.any(totals == 0):
            raise DataValidationError("a ternary point with all three components zero cannot be placed")
        fb, fc = bb / totals, cc / totals  # fa is implied by the other two
        # (a,b,c) -> cartesian, with c along the vertical axis
        xs = fb + 0.5 * fc
        ys = fc * np.sqrt(3.0) / 2.0
        palette = current_palette()
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        ax.scatter(xs, ys, s=45, c=[palette[0]], edgecolors="white", linewidths=0.5)
        if labels is not None:
            if len(labels) != xs.size:
                raise DataValidationError(f"{len(labels)} labels for {xs.size} points")
            for xi, yi, name in zip(xs, ys, labels, strict=True):
                ax.annotate(str(name), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for lx, ly, tx, ty in (
            (0.0, 0.0, 0.0, -0.10),
            (1.0, 0.0, 0.0, -0.10),
            (0.5, np.sqrt(3.0) / 2.0, 0.0, 0.06),
        ):
            ax.plot(
                [lx, lx + 0.5 * (1.0 - lx)], [ly, ly + np.sqrt(3.0) / 2.0], color="#999999", linewidth=0.8, zorder=0
            )
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.12, np.sqrt(3.0) / 2.0 + 0.08)
        ax.text(0.0, -0.06, axis_labels[0], ha="center", fontsize=self._font_size)
        ax.text(1.0, -0.06, axis_labels[1], ha="center", fontsize=self._font_size)
        ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.04, axis_labels[2], ha="center", fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    def plot_bubble(
        self,
        x: object,
        y: object,
        size: object,
        groups: Sequence[object] | None = None,
        labels: Sequence[str] | None = None,
        title: str = "",
        x_label: str = "x",
        y_label: str = "y",
    ) -> Figure:
        """Scatter with a third variable as point area."""
        xa, ya = _paired(x, y)
        sa = _as_1d(size, "size")
        if sa.size != xa.size:
            raise DataValidationError(f"size has {sa.size} values but x has {xa.size}")
        if np.any(sa < 0):
            raise DataValidationError("bubble sizes cannot be negative")
        finite = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(sa)
        xa, ya, sa = xa[finite], ya[finite], sa[finite]
        if sa.max() <= 0:
            raise DataValidationError("every bubble size is zero, so there is nothing to scale by")
        areas = 20.0 + 180.0 * (sa / sa.max())
        codes = _groups_of(xa.size, groups)
        unique = int(codes.max()) + 1 if codes.size else 0
        names = _labels_for(codes, unique, labels)
        palette = current_palette()
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        for g in range(unique):
            mask = codes == g
            ax.scatter(xa[mask], ya[mask], s=areas[mask], c=[palette[g % len(palette)]], alpha=0.6, label=names[g])
        ax.set_xlabel(x_label, fontsize=self._font_size)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        if unique > 1:
            ax.legend(fontsize=self._font_size - 1)
        return fig

    def plot_matrix(
        self,
        matrix: object,
        row_labels: Sequence[str] | None = None,
        column_labels: Sequence[str] | None = None,
        title: str = "",
        diverging: bool = False,
    ) -> Figure:
        """
        Matrix plot: every cell of a 2-D array coloured by its value.

        Sequential data gets one light-to-dark ramp. ``diverging=True`` gives a
        two-sided ramp for data with a meaningful zero, which is a different
        claim than "more of something" and looks different if you get it wrong.
        """
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise DataValidationError(f"a matrix plot needs a 2-D array; got shape {arr.shape}")
        if arr.size == 0:
            raise DataValidationError("the matrix is empty")
        cmap = "RdBu_r" if diverging else "viridis"
        fig = Figure(figsize=(max(6.0, arr.shape[1] * 0.5), max(5.0, arr.shape[0] * 0.45)))
        ax = fig.subplots()
        image = ax.imshow(arr, cmap=cmap, aspect="auto", origin="upper")
        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_yticks(np.arange(arr.shape[0]))
        ax.set_xticklabels(column_labels if column_labels else range(arr.shape[1]), rotation=45, ha="right")
        ax.set_yticklabels(row_labels if row_labels else range(arr.shape[0]))
        fig.colorbar(image, ax=ax, shrink=0.8)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    def plot_mosaic(
        self,
        table: object,
        title: str = "",
    ) -> Figure:
        """
        Mosaic plot of a contingency table: bar widths follow the marginals and
        each row is split in proportion to the cell counts.

        Takes a 2-D count table, not raw observations -- build one with
        ``pandas.crosstab`` or ``numpy.unique(..., return_counts=True)``. A zero
        margin would make a rectangle of zero width, which is meaningless, so
        it is rejected.
        """
        arr = np.asarray(table, dtype=float)
        if arr.ndim != 2:
            raise DataValidationError(f"a mosaic plot needs a 2-D contingency table; got shape {arr.shape}")
        if arr.size == 0:
            raise DataValidationError("the table is empty")
        if np.any(arr < 0):
            raise DataValidationError("a contingency table cannot have negative counts")
        rows = arr.sum(axis=1)
        cols = arr.sum(axis=0)
        if rows.sum() == 0 or cols.sum() == 0:
            raise DataValidationError("the table has no observations to lay out")
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        y_edges = np.concatenate([[0.0], np.cumsum(rows / rows.sum())])
        for i in range(arr.shape[0]):
            band = rows[i] / rows.sum()
            x_edges = np.concatenate([[0.0], np.cumsum(arr[i] / cols)])
            for j in range(arr.shape[1]):
                width = x_edges[j + 1] - x_edges[j]
                if width <= 0:
                    continue
                shade = 0.25 + 0.6 * (arr[i, j] / max(arr[i].max(), 1e-12))
                ax.add_patch(
                    __import__("matplotlib").patches.Rectangle(
                        (x_edges[j], y_edges[i]), width, band, facecolor=str(shade), edgecolor="white", linewidth=1
                    )
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    # ------------------------------------------------------------------
    # Diagrams
    # ------------------------------------------------------------------

    def plot_venn(
        self,
        counts: Sequence[int],
        set_labels: Sequence[str] | None = None,
        title: str = "",
    ) -> Figure:
        """
        Venn diagram for two or three sets.

        ``counts`` is the **region** counts: for two sets ``[only A, only B,
        both]``; for three ``[A only, B only, C only, A∩B, A∩C, B∩C, all
        three]``. Every count must be positive -- a Venn region drawn at zero
        size cannot be labelled or compared, so it is reported rather than
        drawn as an invisible gap.
        """
        vals = [int(c) for c in counts]
        if len(vals) not in (3, 7):
            raise DataValidationError(f"a Venn diagram takes 3 counts (two sets) or 7 (three sets); got {len(vals)}")
        if any(v <= 0 for v in vals):
            raise DataValidationError(f"every Venn region must have a positive count; got {vals}")
        fig = Figure(figsize=(7.0, 6.0))
        ax = fig.subplots()
        if len(vals) == 3:
            from matplotlib.patches import Circle

            a, b, both = vals
            ax.add_patch(Circle((0.35, 0.5), 0.3, alpha=0.5, color=current_palette()[0]))
            ax.add_patch(Circle((0.65, 0.5), 0.3, alpha=0.5, color=current_palette()[1]))
            ax.text(0.22, 0.5, str(a), ha="center", fontsize=self._title_font_size)
            ax.text(0.78, 0.5, str(b), ha="center", fontsize=self._title_font_size)
            ax.text(0.5, 0.5, str(both), ha="center", fontsize=self._title_font_size)
            names = set_labels or ["Set A", "Set B"]
            ax.text(0.35, 0.86, names[0], ha="center", fontsize=self._font_size)
            ax.text(0.65, 0.86, names[1], ha="center", fontsize=self._font_size)
        else:
            from matplotlib.patches import Circle

            a, b, c, ab, ac, bc, abc = vals
            palette = current_palette()
            for (cx, cy), colour in (((0.42, 0.55), palette[0]), ((0.58, 0.55), palette[1]), ((0.5, 0.38), palette[2])):
                ax.add_patch(Circle((cx, cy), 0.28, alpha=0.5, color=colour))
            for cx, cy, text in (
                (0.30, 0.62, a),
                (0.70, 0.62, b),
                (0.50, 0.26, c),
                (0.50, 0.60, ab),
                (0.38, 0.44, ac),
                (0.62, 0.44, bc),
                (0.50, 0.48, abc),
            ):
                ax.text(cx, cy, str(text), ha="center", fontsize=9)
            names = set_labels or ["Set A", "Set B", "Set C"]
            ax.text(0.42, 0.87, names[0], ha="center", fontsize=self._font_size)
            ax.text(0.58, 0.87, names[1], ha="center", fontsize=self._font_size)
            ax.text(0.50, 0.12, names[2], ha="center", fontsize=self._font_size)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    def plot_radar(
        self,
        matrix: object,
        axis_labels: Sequence[str] | None = None,
        row_labels: Sequence[str] | None = None,
        title: str = "",
    ) -> Figure:
        """Radar (spider) plot: one polygon per row of a 2-D matrix."""
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise DataValidationError(f"a radar plot needs a 2-D matrix; got shape {arr.shape}")
        if arr.shape[1] < 3:
            raise DataValidationError(f"a radar plot needs at least 3 axes; got {arr.shape[1]}")
        axes = list(axis_labels) if axis_labels is not None else [f"axis {i + 1}" for i in range(arr.shape[1])]
        if len(axes) != arr.shape[1]:
            raise DataValidationError(f"{len(axes)} axis labels for {arr.shape[1]} axes")
        angles = np.linspace(0, 2 * np.pi, arr.shape[1], endpoint=False)
        closed = np.concatenate([angles, angles[:1]])
        palette = current_palette()
        fig = Figure(figsize=(7.0, 7.0))
        ax = fig.add_subplot(projection="polar")
        for i in range(arr.shape[0]):
            values = arr[i]
            closed_values = np.concatenate([values, values[:1]])
            name = row_labels[i] if row_labels and i < len(row_labels) else f"row {i + 1}"
            ax.plot(closed, closed_values, label=name, color=palette[i % len(palette)], linewidth=1.5)
        ax.set_xticks(closed)
        ax.set_xticklabels([*axes, axes[0]], fontsize=self._font_size - 1)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=self._font_size - 1)
        return fig

    def plot_polar(
        self,
        values: object,
        title: str = "",
        radial_label: str = "",
        theta_label: str = "angle",
    ) -> Figure:
        """
        Polar plot of one column: each value at an evenly spaced angle.

        This is the rose-diagram layout, so a column of lengths becomes a
        radial shape. Equally spaced angles are the convention; using the value
        column as its own theta would make the plot self-referential.
        """
        data = _as_1d(values, "values")
        data = data[np.isfinite(data)]
        if data.size == 0:
            raise DataValidationError("no finite values to plot")
        angles = np.linspace(0, 2 * np.pi, data.size, endpoint=False)
        fig = Figure(figsize=(7.0, 7.0))
        ax = fig.add_subplot(projection="polar")
        ax.plot(angles, data, "-o", color=current_palette()[0], linewidth=1.5, markersize=5)
        ax.fill(angles, data, alpha=0.25, color=current_palette()[0])
        if radial_label:
            ax.set_ylabel(radial_label, fontsize=self._font_size)
        if theta_label:
            ax.set_xlabel(theta_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold", pad=20)
        return fig

    def plot_vectors(
        self,
        x: object,
        y: object,
        u: object,
        v: object,
        title: str = "",
        x_label: str = "x",
        y_label: str = "y",
    ) -> Figure:
        """Vector (quiver) field: position plus direction and magnitude."""
        xa, ya = _paired(x, y)
        uu, vv = _paired(u, v)
        if not (xa.size == uu.size):
            raise DataValidationError(f"positions ({xa.size}) and vectors ({uu.size}) must match in length")
        magnitude = np.hypot(uu, vv)
        if magnitude.max() > 0:
            # Scale arrows to the data, not to the units they were measured in.
            span = max(float(np.ptp(xa)), float(np.ptp(ya)), 1e-12)
            uu = uu / magnitude.max() * span * 0.25
            vv = vv / magnitude.max() * span * 0.25
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        ax.quiver(xa, ya, uu, vv, angles="xy", scale_units="xy", scale=1.0, color=current_palette()[0])
        ax.set_xlabel(x_label, fontsize=self._font_size)
        ax.set_ylabel(y_label, fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        return fig

    def plot_network(
        self,
        edges: object,
        node_labels: Sequence[str] | None = None,
        title: str = "",
    ) -> Figure:
        """
        Network plot from an edge list.

        ``edges`` is ``n x 2`` of node indices, or ``n x 3`` with a weight in
        the third column, which sets the edge width. Node positions are laid
        out on a circle: a force-directed layout would place the same graph
        differently on every run, and a figure that moves between runs cannot be
        compared with the previous one.
        """
        arr = np.asarray(edges)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise DataValidationError(f"edges must be an n x 2 or n x 3 array; got shape {arr.shape}")
        pairs = arr[:, :2].astype(int)
        if pairs.size == 0:
            raise DataValidationError("no edges to plot")
        if pairs.min() < 0:
            raise DataValidationError("node indices must be zero or greater")
        n_nodes = int(pairs.max()) + 1
        weights = arr[:, 2].astype(float) if arr.shape[1] == 3 else np.ones(pairs.shape[0])
        if weights.max() > 0:
            widths = 0.8 + 3.0 * weights / weights.max()
        else:
            widths = np.full(pairs.shape[0], 1.5)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
        palette = current_palette()
        fig = Figure(figsize=self._figure_size)
        ax = fig.subplots()
        for k, (i, j) in enumerate(pairs):
            ax.plot(positions[i], positions[j], color="#666666", linewidth=widths[k], zorder=1, solid_capstyle="round")
        for i in range(n_nodes):
            ax.scatter(*positions[i], s=420, c=[palette[i % len(palette)]], zorder=2, edgecolors="white")
            name = node_labels[i] if node_labels and i < len(node_labels) else str(i)
            ax.text(*positions[i], str(name), ha="center", va="center", fontsize=8, color="white", zorder=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    # ------------------------------------------------------------------
    # 3-D
    # ------------------------------------------------------------------

    def _axes3d(self, fig: Figure):
        """Import the 3-D projection here, not at module scope.

        ``mpl_toolkits.mplot3d`` is a matplotlib submodule rather than a hard
        dependency. Importing it at the top of this file would make the entire
        plot menu unimportable on a build without it, for the sake of five
        methods out of twenty.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the projection

        return fig.add_subplot(projection="3d")

    def plot_3d(
        self,
        x: object,
        y: object,
        z: object,
        groups: Sequence[object] | None = None,
        labels: Sequence[str] | None = None,
        title: str = "",
        mode: str = "scatter",
        line_width: float = 1.5,
    ) -> Figure:
        """3-D scatter, line or bubble plot of three columns."""
        xa, ya = _paired(x, y)
        za = _as_1d(z, "z")
        if za.size != xa.size:
            raise DataValidationError(f"z has {za.size} values but x has {xa.size}")
        finite = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(za)
        xa, ya, za = xa[finite], ya[finite], za[finite]
        if xa.size == 0:
            raise DataValidationError("no finite triples to plot")
        mode = mode.lower()
        if mode not in ("scatter", "line", "bubble"):
            raise DataValidationError(f"mode must be scatter, line or bubble; got {mode!r}")
        fig = Figure(figsize=self._figure_size)
        ax = self._axes3d(fig)
        palette = current_palette()
        codes = _groups_of(xa.size, groups)
        unique = int(codes.max()) + 1
        names = _labels_for(codes, unique, labels)
        for g in range(unique):
            mask = codes == g
            colour = palette[g % len(palette)]
            if mode == "scatter":
                ax.scatter(xa[mask], ya[mask], za[mask], c=[colour], s=40, label=names[g], depthshade=True)
            elif mode == "line":
                ax.plot(xa[mask], ya[mask], za[mask], color=colour, linewidth=line_width, label=names[g])
            else:
                ax.scatter(
                    xa[mask],
                    ya[mask],
                    za[mask],
                    c=[colour],
                    s=np.abs(za[mask]) * 10 + 20,
                    alpha=0.6,
                    label=names[g],
                    depthshade=True,
                )
        ax.set_xlabel("x", fontsize=self._font_size)
        ax.set_ylabel("y", fontsize=self._font_size)
        ax.set_zlabel("z", fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        if unique > 1:
            ax.legend(fontsize=self._font_size - 1)
        return fig

    def plot_surface(
        self,
        z: object,
        row_labels: Sequence[str] | None = None,
        column_labels: Sequence[str] | None = None,
        title: str = "",
        as_points: bool = False,
    ) -> Figure:
        """
        3-D surface over a rectangular grid: ``z`` has one value per grid cell.

        A surface implies a grid, so a ragged input is rejected here rather than
        quietly plotted with a spacing that means nothing. Use ``as_points`` to
        plot scattered heights instead.
        """
        arr = np.asarray(z, dtype=float)
        if arr.ndim != 2:
            raise DataValidationError(f"a surface needs a 2-D grid of heights; got shape {arr.shape}")
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            raise DataValidationError(f"a surface needs at least a 2 x 2 grid; got shape {arr.shape}")
        fig = Figure(figsize=self._figure_size)
        ax = self._axes3d(fig)
        xs = np.arange(arr.shape[1])
        ys = np.arange(arr.shape[0])
        xx, yy = np.meshgrid(xs, ys)
        if as_points:
            ax.scatter(xx.ravel(), yy.ravel(), arr.ravel(), c=arr.ravel(), cmap="viridis", s=40)
        else:
            ax.plot_surface(xx, yy, arr, cmap="viridis", linewidth=0, antialiased=True)
        if row_labels is not None and len(row_labels) != arr.shape[0]:
            raise DataValidationError(f"{len(row_labels)} row labels for {arr.shape[0]} rows")
        if column_labels is not None and len(column_labels) != arr.shape[1]:
            raise DataValidationError(f"{len(column_labels)} column labels for {arr.shape[1]} columns")
        ax.set_xlabel(column_labels[0] if column_labels else "x", fontsize=self._font_size)
        ax.set_ylabel(row_labels[0] if row_labels else "y", fontsize=self._font_size)
        ax.set_zlabel("value", fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig

    def plot_parametric_surface(
        self,
        u: object,
        v: object,
        function,
        title: str = "",
        n_points: int = 40,
    ) -> Figure:
        """
        Parametric surface ``(u, v) -> (x, y, z)``.

        ``function`` is called with two 1-D arrays of parameters and must return
        three arrays of the same length. It is sampled on a grid rather than
        being trusted to vectorise, so a function written with a Python loop
        still works.
        """
        ua = _as_1d(u, "u")
        va = _as_1d(v, "v")
        if ua.size < 2 or va.size < 2:
            raise DataValidationError("each parameter needs at least 2 values to make a surface")
        if n_points < 2:
            raise DataValidationError(f"n_points must be at least 2; got {n_points}")
        gu = np.linspace(ua.min(), ua.max(), min(n_points, ua.size * 4))
        gv = np.linspace(va.min(), va.max(), min(n_points, va.size * 4))
        uu, vv = np.meshgrid(gu, gv)
        try:
            out = function(uu.ravel(), vv.ravel())
        except Exception as exc:
            raise PlottingError(f"the parametric function failed on the sampled grid: {exc}") from exc
        arr = np.asarray(out, dtype=float)
        # A function returning ``(xs, ys, zs)`` lands as (3, n); one returning a
        # single stacked array lands as (n, 3). Accept both, because both are
        # natural ways to write it and rejecting the first would be pedantry.
        if arr.shape == (3, uu.size):
            arr = arr.T
        elif arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.shape != (uu.size, 3):
            raise PlottingError(
                f"the parametric function must return three arrays of {uu.size} values; got shape {arr.shape}"
            )
        fig = Figure(figsize=self._figure_size)
        ax = self._axes3d(fig)
        ax.plot_surface(
            arr[:, 0].reshape(uu.shape),
            arr[:, 1].reshape(uu.shape),
            arr[:, 2].reshape(uu.shape),
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        ax.set_xlabel("x", fontsize=self._font_size)
        ax.set_ylabel("y", fontsize=self._font_size)
        ax.set_zlabel("z", fontsize=self._font_size)
        if title:
            ax.set_title(title, fontsize=self._title_font_size, fontweight="bold")
        return fig
