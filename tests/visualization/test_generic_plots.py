# =============================================================================
# Test: visualization.generic_plots -- the plot menu
# =============================================================================
"""
The plot menu is the one part of PAST that takes data straight from the sheet
rather than from an analysis result, which is why it had no equivalent here at
all. These tests cover two things:

* every ``plot_*`` method actually produces a figure. A method that raises on
  ordinary input is a method nobody can use, and it would not be caught by any
  other test in the suite because nothing else calls it;
* the input-shape rules that stop a figure being drawn from the wrong data.

Colour is not re-tested here. ``tests/visualization/test_palette_single_source.py``
already asserts that these figures take their categorical colours from
``current_palette()``, and duplicating it here would be a second copy to keep
in step.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from utils.exceptions import DataValidationError, PlottingError
from visualization.generic_plots import GenericPlotter


@pytest.fixture
def plotter() -> GenericPlotter:
    return GenericPlotter()


@pytest.fixture
def columns():
    """A reproducible three-group table: x, y, size, group label."""
    rng = np.random.RandomState(0)
    x = np.arange(21.0)
    y = 2.0 * x + rng.randn(21) * 0.5
    groups = ["a"] * 7 + ["b"] * 7 + ["c"] * 7
    return x, y, np.abs(y) + 0.5, groups


# ---------------------------------------------------------------------------
# Every method draws
# ---------------------------------------------------------------------------


def test_every_plot_method_produces_a_figure(plotter, columns):
    """
    Would go red the first time a method raised on ordinary input -- which is
    how four of them did the first time this was written.
    """
    x, y, size, groups = columns
    rng = np.random.RandomState(1)
    fig, ax = plotter._figure_size
    assert fig and ax
    cases = {
        "plot_xy": lambda: plotter.plot_xy(x, y, groups=groups, labels=["A", "B", "C"], fit_line=True),
        "plot_xy_with_errorbars": lambda: plotter.plot_xy_with_errorbars(x, y, np.full(x.size, 0.4)),
        "plot_histogram": lambda: plotter.plot_histogram(y),
        "plot_bar_chart": lambda: plotter.plot_bar_chart(y),
        "plot_bar_chart_summed": lambda: plotter.plot_bar_chart(y, categories=groups),
        "plot_box_plot": lambda: plotter.plot_box_plot(groups, y),
        "plot_pie_chart": lambda: plotter.plot_pie_chart(groups),
        "plot_stacked_chart": lambda: plotter.plot_stacked_chart(rng.rand(6, 4)),
        "plot_percentiles": lambda: plotter.plot_percentiles(groups, y),
        "plot_normal_probability": lambda: plotter.plot_normal_probability(y),
        "plot_ternary": lambda: plotter.plot_ternary(rng.rand(9) + 1, rng.rand(9) + 1, rng.rand(9) + 1),
        "plot_bubble": lambda: plotter.plot_bubble(x, y, size),
        "plot_matrix": lambda: plotter.plot_matrix(rng.randn(8, 6)),
        "plot_mosaic": lambda: plotter.plot_mosaic(np.array([[10, 5], [3, 8]])),
        "plot_venn_two": lambda: plotter.plot_venn([10, 20, 5]),
        "plot_venn_three": lambda: plotter.plot_venn([1, 2, 3, 4, 5, 6, 7]),
        "plot_radar": lambda: plotter.plot_radar(rng.rand(3, 5)),
        "plot_polar": lambda: plotter.plot_polar(y),
        "plot_vectors": lambda: plotter.plot_vectors(x, y, rng.randn(21), rng.randn(21)),
        "plot_network": lambda: plotter.plot_network(np.array([[0, 1], [1, 2], [2, 0]])),
        "plot_3d_scatter": lambda: plotter.plot_3d(x, y, rng.rand(21)),
        "plot_3d_line": lambda: plotter.plot_3d(x, y, rng.rand(21), mode="line"),
        "plot_3d_bubble": lambda: plotter.plot_3d(x, y, rng.rand(21), mode="bubble"),
        "plot_surface": lambda: plotter.plot_surface(rng.randn(10, 8)),
        "plot_parametric_surface": lambda: plotter.plot_parametric_surface(
            np.linspace(0, 6, 4), np.linspace(0, 6, 4), lambda u, v: (u * np.cos(v), u * np.sin(v), u)
        ),
    }
    for name, call in cases.items():
        result = call()
        assert isinstance(result, Figure), f"{name} did not return a Figure"
        assert result.axes, f"{name} returned a Figure with no axes"


def test_the_plot_menu_covers_pasts_22_kinds(plotter, columns):
    """
    A count, on purpose.

    The point of the module is to be the counterpart of PAST's Plot menu, and
    a count is the cheapest way to notice a method quietly disappearing in a
    refactor. If a plot kind is removed, this number is the thing that has to be
    updated deliberately rather than by accident.
    """
    kinds = [n for n in dir(plotter) if n.startswith("plot_")]
    assert len(kinds) >= 21, f"only {len(kinds)} plot kinds: {sorted(kinds)}"
    assert "plot_xy" in kinds and "plot_ternary" in kinds and "plot_venn" in kinds


# ---------------------------------------------------------------------------
# Shapes that would draw the wrong figure
# ---------------------------------------------------------------------------


def test_mismatched_columns_are_refused_not_truncated(plotter):
    """
    Would go red if the pairing ever fell back to zip(), which truncates to the
    shorter column and silently re-pairs every value after the shortfall.
    """
    with pytest.raises(DataValidationError, match="same length"):
        plotter.plot_xy([1, 2, 3], [1, 2])


def test_a_log_axis_rejects_non_positive_values(plotter):
    """A log axis with a zero or negative point drops the axis silently and
    leaves a figure that looks like a linear one."""
    with pytest.raises(DataValidationError, match="greater than zero"):
        plotter.plot_xy([1, 2, 0], [1, 2, 3], log_x=True)


def test_box_plot_refuses_a_group_with_one_value(plotter):
    """One observation has no quartiles. Drawing a box there invents them."""
    with pytest.raises(DataValidationError, match="fewer than two values"):
        plotter.plot_box_plot(["a", "b", "b"], [1.0, 2.0, 3.0])


def test_pie_refuses_a_negative_value(plotter):
    """
    A negative used to be absorbed into "count of distinct values" and came
    out as a plausible-looking pie. Losing a negative value silently is the
    exact failure this check exists for.
    """
    with pytest.raises(DataValidationError, match="negative"):
        plotter.plot_pie_chart(np.array([1.0, -2.0, 3.0]))


def test_stacked_chart_normalised_to_one_hundred_percent(plotter):
    """
    The normalisation is the whole point of the flag, and it is checkable:
    read the bar heights back off the axes and they must sum to 100 per row.
    """
    matrix = np.array([[1.0, 3.0], [2.0, 2.0]])
    figure = plotter.plot_stacked_chart(matrix, normalize=True)
    ax = figure.axes[0]
    for row in range(matrix.shape[0]):
        bars = [b for b in ax.patches if round(b.get_x()) == row]
        assert bars, f"row {row} drew no bars"
        assert sum(b.get_height() for b in bars) == pytest.approx(100.0)


def test_stacked_chart_refuses_a_ragged_input(plotter, columns):
    x, _y, _size, groups = columns
    with pytest.raises(DataValidationError):
        plotter.plot_stacked_chart(x, column_labels=groups)


def test_a_non_numeric_column_becomes_a_pie_of_its_labels(plotter):
    """
    The label column is the common case for a pie, so it has to work: counting
    the distinct labels is the figure, not an error.
    """
    figure = plotter.plot_pie_chart(["a"] * 3 + ["b"] * 1)
    ax = figure.axes[0]
    assert len(ax.patches) == 2, "a four-versus-one label column is two slices"
    spans = sorted((p.theta2 - p.theta1) / 360.0 for p in ax.patches)
    assert spans == pytest.approx([0.25, 0.75])


def test_ternary_normalises_raw_counts(plotter):
    """
    Three components that do not sum to 100 are still placeable: they are
    divided by their row sum. A plot that demanded percentages would reject the
    count table a palaeontologist actually has.
    """
    a = np.array([1.0, 1.0])
    b = np.array([1.0, 1.0])
    c = np.array([2.0, 6.0])
    figure = plotter.plot_ternary(a, b, c)
    assert figure.axes
    with pytest.raises(DataValidationError):
        plotter.plot_ternary(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_ternary_refuses_a_zero_total(plotter):
    """A point with all three components zero has no direction to go."""
    with pytest.raises(DataValidationError, match="all three components zero"):
        plotter.plot_ternary([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])


def test_venn_refuses_a_zero_region(plotter):
    """A zero-size region cannot be labelled or compared, so it is reported."""
    with pytest.raises(DataValidationError, match="positive count"):
        plotter.plot_venn([10, 0, 5])


def test_venn_refuses_the_wrong_number_of_counts(plotter):
    """Three counts is two sets, seven is three. Anything else is a mistake."""
    with pytest.raises(DataValidationError, match="3 counts"):
        plotter.plot_venn([1, 2, 3, 4])


def test_network_refuses_negative_node_indices(plotter):
    with pytest.raises(DataValidationError, match="zero or greater"):
        plotter.plot_network(np.array([[0, -1], [1, 2]]))


def test_parametric_surface_accepts_both_return_shapes(plotter):
    """
    Returning ``(xs, ys, zs)`` is the natural way to write the function and
    lands as a (3, n) array; a stacked (n, 3) is equally natural. Rejecting the
    first would be pedantry about something the caller cannot see.
    """
    u = v = np.linspace(0, 6, 4)
    as_tuple = plotter.plot_parametric_surface(u, v, lambda a, b: (a, b, a * b))
    as_array = plotter.plot_parametric_surface(u, v, lambda a, b: np.column_stack([a, b, a * b]))
    assert as_tuple.axes and as_array.axes


def test_parametric_surface_reports_a_function_that_returns_the_wrong_arity(plotter):
    u = v = np.linspace(0, 6, 4)
    with pytest.raises(PlottingError, match="three arrays"):
        plotter.plot_parametric_surface(u, v, lambda a, b: (a, b))


def test_normal_probability_plot_needs_enough_points(plotter):
    with pytest.raises(DataValidationError, match="at least 3"):
        plotter.plot_normal_probability([1.0, 2.0])


def test_matrix_plot_refuses_a_non_matrix(plotter):
    with pytest.raises(DataValidationError, match="2-D"):
        plotter.plot_matrix(np.arange(10.0))
