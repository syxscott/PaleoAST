# =============================================================================
# FILE: tests/cross_validation/test_vs_r_base_plots.py
# =============================================================================
"""
Cross-validation of the plot menu against base R's ``stats``.

What is compared
----------------
A figure cannot be compared with another figure -- R's graphics and
matplotlib's are not the same picture, and no amount of care would make
them so.  What CAN be compared is the **data behind the figure**: the
counts in the bars, the quartiles in the box, the angles in the pie, the
quantiles in the normal probability plot.  That is where a wrong number
would enter a publication, and it is what this file checks.

Every figure is built by the application, and the numbers are then read
back out of the matplotlib artists -- the bar heights, the box vertices,
the wedge angles -- and compared with what R computes from the same
input.  Reading the numbers back out of the picture is deliberate: it
checks the picture, not a helper function the picture happens to call.

What R is asked
---------------
``stats::lm`` for the least-squares line, ``stats::hist`` for the bin
counts, ``stats::tapply`` for the category totals, ``stats::quantile``
for the quartiles, ``stats::qqnorm`` for the theoretical quantiles of the
normal probability plot, and the arithmetic R itself would do for the pie
angles and the stacked bars.

Two differences in the QUARTILES are on the record
--------------------------------------------------
The box plot's quartiles are compared against R's ``quantile`` with
``type = 7`` -- the linear interpolation that numpy and R both default
to.  R's own ``boxplot`` does NOT use that: ``boxplot.stats`` computes
Tukey's hinges with ``fivenum``, which for an even-sized sample is a
different number.  Comparing against ``boxplot.stats`` would therefore
fail for a reason that has nothing to do with the application, and
comparing against ``quantile`` without saying which type would leave the
reader guessing.  The hinges are mentioned in the test so the difference
is visible rather than buried.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import r_labels, r_numbers, r_program, r_program_values, require

pytestmark = pytest.mark.cross_validation

#: The analyses this file compares against R.  ``scripts/validation_matrix.py``
#: reads this declaration rather than guessing from imported class names.
ANALYSES_COVERED = {
    "xy_plot",
    "xy_error_bars",
    "histogram",
    "bar_chart",
    "box_plot",
    "pie_chart",
    "stacked_chart",
    "normal_probability_plot",
    "percentiles",
}

R_STATS = require("stats")

#: Tolerances.  These are the same quantities, computed by two independent
#: implementations, so they agree to the last bits of a double.
RTOL = 1e-9
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Reading the numbers back out of the figure
# ---------------------------------------------------------------------------
def _axes(fig):
    return fig.axes[0]


def _bar_heights(fig) -> np.ndarray:
    """The heights of the bars of a bar-like figure."""
    return np.array([patch.get_height() for patch in _axes(fig).patches], dtype=float)


def _bar_left_edges(fig) -> np.ndarray:
    """The left edge of each bar."""
    return np.array([patch.get_x() for patch in _axes(fig).patches], dtype=float)


def _bar_widths(fig) -> np.ndarray:
    """The width of each bar."""
    return np.array([patch.get_width() for patch in _axes(fig).patches], dtype=float)


def _box_stats(fig) -> list[dict[str, float]]:
    """The statistics of every box in a box plot.

    The box is a path whose vertices give the two quartiles; the whiskers
    are the vertical lines; the median is the horizontal line that spans
    the whole box -- the caps are narrower, which is what tells them
    apart.  Every line is matched to its own box first: the lines of one
    box belong to that box, and picking them up from the whole axes would
    give every box the same whiskers.
    """
    ax = _axes(fig)
    out: list[dict[str, float]] = []
    for patch in ax.patches:
        vertices = np.asarray(patch.get_path().vertices, dtype=float)
        q1 = float(vertices[:, 1].min())
        q3 = float(vertices[:, 1].max())
        left = float(vertices[:, 0].min())
        right = float(vertices[:, 0].max())
        median = None
        whisker_ends: list[float] = []
        for line in ax.lines:
            xs = np.asarray(line.get_xdata(), dtype=float).ravel()
            ys = np.asarray(line.get_ydata(), dtype=float).ravel()
            if xs.size != 2:
                continue
            if xs.min() < left - 1e-9 or xs.max() > right + 1e-9:
                continue  # a line belonging to another box
            if xs[0] == xs[1]:
                whisker_ends.extend(ys.tolist())
            elif np.isclose(xs[0], left) and np.isclose(xs[1], right):
                median = float(ys[0])
        assert median is not None, "the median line of a box was not found"
        out.append(
            {
                "q1": q1,
                "median": median,
                "q3": q3,
                "whislo": float(min(whisker_ends)),
                "whishi": float(max(whisker_ends)),
            }
        )
    return out


def _fit_line(fig) -> tuple[float, float]:
    """The slope and the intercept of the drawn fit line.

    Taken from two points on the line rather than from the coefficients
    the plotter computed, so that the comparison is with R's
    coefficients and not with the plotter's own arithmetic.
    """
    line = _axes(fig).lines[0]
    xs = np.asarray(line.get_xdata(), dtype=float)
    ys = np.asarray(line.get_ydata(), dtype=float)
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    intercept = ys[0] - slope * xs[0]
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _xy() -> tuple[np.ndarray, np.ndarray]:
    """A scatter with a clear linear trend and some scatter around it."""
    rng = np.random.default_rng(1001)
    x = rng.uniform(1.0, 20.0, 40)
    y = 3.0 * x + 7.0 + rng.normal(0.0, 2.0, 40)
    return x, y


def _groups() -> tuple[list[str], np.ndarray]:
    """Three groups, in alphabetical order, with different spreads."""
    rng = np.random.default_rng(1002)
    groups = ["alpha"] * 18 + ["beta"] * 18 + ["gamma"] * 18
    values = np.concatenate(
        [
            rng.normal(10.0, 1.0, 18),
            rng.normal(12.0, 2.5, 18),
            rng.normal(9.0, 1.5, 18),
        ]
    )
    return groups, values


# ---------------------------------------------------------------------------
# XY scatter
# ---------------------------------------------------------------------------
_LM_PROGRAM = """
x <- @@x@@
y <- @@y@@
fit <- lm(y ~ x)
out <- c(intercept = unname(coef(fit)[1]), slope = unname(coef(fit)[2]))
out
"""


class TestXyPlot:
    """``stats::lm`` -- the least-squares line through the scatter."""

    def test_the_fit_line_matches_r(self):
        from visualization.generic_plots import GenericPlotter

        x, y = _xy()
        fig = GenericPlotter().plot_xy(x, y, fit_line=True)
        theirs = r_program(_LM_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        slope, intercept = _fit_line(fig)

        assert_allclose(slope, theirs["slope"], rtol=RTOL, atol=ATOL)
        assert_allclose(intercept, theirs["intercept"], rtol=RTOL, atol=ATOL)

    def test_the_points_are_the_data_that_went_in(self):
        from visualization.generic_plots import GenericPlotter

        x, y = _xy()
        fig = GenericPlotter().plot_xy(x, y)
        offsets = np.asarray(_axes(fig).collections[0].get_offsets(), dtype=float)
        assert_allclose(offsets[:, 0], x, rtol=RTOL, atol=ATOL)
        assert_allclose(offsets[:, 1], y, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# XY with error bars
# ---------------------------------------------------------------------------
_ERRORBAR_PROGRAM = """
y <- @@y@@
err <- @@err@@
out <- c(y - err, y + err)
out
"""


class TestXyErrorBars:
    """The error bars are symmetric half-widths around the point."""

    def test_the_bar_ends_match_r(self):
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1003)
        x = rng.uniform(0.0, 10.0, 25)
        y = rng.normal(5.0, 1.0, 25)
        errors = rng.uniform(0.1, 0.9, 25)

        fig = GenericPlotter().plot_xy_with_errorbars(x, y, errors)
        segments = _axes(fig).collections[0].get_segments()
        mine = np.array([[seg[0][1], seg[1][1]] for seg in segments], dtype=float)

        theirs = r_program_values(_ERRORBAR_PROGRAM, y=r_numbers(y), err=r_numbers(errors))
        assert_allclose(mine[:, 0], theirs[:25], rtol=RTOL, atol=ATOL)
        assert_allclose(mine[:, 1], theirs[25:], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
_HISTOGRAM_PROGRAM = """
x <- @@x@@
breaks <- @@breaks@@
h <- hist(x, breaks = breaks, plot = FALSE)
out <- h$counts
out
"""


class TestHistogram:
    """``stats::hist`` -- the counts in the bins."""

    def test_the_bin_counts_match_r(self):
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1004)
        values = rng.normal(10.0, 2.0, 120)

        fig = GenericPlotter().plot_histogram(values, bins=8)
        heights = _bar_heights(fig)
        lefts = _bar_left_edges(fig)
        widths = _bar_widths(fig)
        # The edges the application actually used, handed to R so that the
        # two are counting the same bins.
        edges = np.concatenate([lefts, [lefts[-1] + widths[-1]]]) if heights.size else lefts

        theirs = r_program_values(_HISTOGRAM_PROGRAM, x=r_numbers(values), breaks=r_numbers(edges))
        assert_allclose(heights, theirs, rtol=RTOL, atol=ATOL)
        assert heights.sum() == values.size

    def test_the_bins_span_the_data(self):
        """The bars cover the whole sample: no observation is dropped.

        A histogram that silently loses the values above the last break
        is a histogram of a subset, and the reader cannot tell.
        """
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1005)
        values = rng.normal(0.0, 1.0, 200)
        fig = GenericPlotter().plot_histogram(values, bins=10)
        lefts = _bar_left_edges(fig)
        widths = _bar_widths(fig)
        assert_allclose(float(lefts.min()), float(values.min()), rtol=RTOL, atol=ATOL)
        assert_allclose(float(lefts.max() + widths.max()), float(values.max()), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
_TAPPLY_PROGRAM = """
v <- @@v@@
g <- factor(@@g@@)
out <- as.numeric(tapply(v, g, sum))
out
"""


class TestBarChart:
    """``stats::tapply`` -- the total per category."""

    def test_the_bar_heights_match_r(self):
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_bar_chart(values, categories=groups)
        mine = _bar_heights(fig)
        theirs = r_program_values(_TAPPLY_PROGRAM, v=r_numbers(values), g=r_labels(groups))
        assert_allclose(mine, theirs, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------
_BOX_PROGRAM = """
x <- @@x@@
q <- quantile(x, c(0.25, 0.5, 0.75), type = 7)
iqr <- q[3] - q[1]
out <- c(
  q[1], q[2], q[3],
  min(x[x >= q[1] - 1.5 * iqr]),
  max(x[x <= q[3] + 1.5 * iqr])
)
out
"""


def _box_reference(values: np.ndarray) -> np.ndarray:
    """R's answer for one group: q1, median, q3, lower whisker, upper whisker."""
    return r_program_values(_BOX_PROGRAM, x=r_numbers(values))


class TestBoxPlot:
    """``stats::quantile`` -- the five numbers a box is made of."""

    def test_every_box_matches_r(self):
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_box_plot(groups, values)
        boxes = _box_stats(fig)
        assert len(boxes) == 3

        for group, box in zip(("alpha", "beta", "gamma"), boxes, strict=True):
            theirs = _box_reference(values[np.array(groups) == group])
            assert_allclose(box["q1"], theirs[0], rtol=RTOL, atol=ATOL)
            assert_allclose(box["median"], theirs[1], rtol=RTOL, atol=ATOL)
            assert_allclose(box["q3"], theirs[2], rtol=RTOL, atol=ATOL)
            assert_allclose(box["whislo"], theirs[3], rtol=RTOL, atol=ATOL)
            assert_allclose(box["whishi"], theirs[4], rtol=RTOL, atol=ATOL)

    def test_r_s_own_boxplot_uses_the_hinges_not_the_type_7_quartiles(self):
        """R's ``boxplot`` and R's ``quantile`` disagree, and that is on record.

        ``boxplot.stats`` computes Tukey's hinges with ``fivenum``, which
        for an even-sized sample is not the same number as the
        interpolated quantile.  The application uses the interpolated
        quantile, so comparing against ``boxplot.stats`` would fail for a
        reason that has nothing to do with it.  This test states the
        difference instead of leaving it for a reader to trip over.
        """
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_box_plot(groups, values)
        box = _box_stats(fig)[0]
        group_values = values[np.array(groups) == "alpha"]

        hinges = r_program_values(
            """
out <- fivenum(x)
out
""",
            x=r_numbers(group_values),
        )
        interpolated = _box_reference(group_values)
        # The hinges and the interpolated quartiles are different numbers
        # for an even-sized sample.
        assert not np.isclose(hinges[1], interpolated[0], rtol=1e-6)
        # ...and the application reports the interpolated one.
        assert_allclose(box["q1"], interpolated[0], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Percentile plot
# ---------------------------------------------------------------------------
class TestPercentiles:
    """The percentile plot is the box plot without the outlier markers."""

    def test_the_boxes_match_r(self):
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_percentiles(groups, values)
        boxes = _box_stats(fig)
        assert len(boxes) == 3
        for group, box in zip(("alpha", "beta", "gamma"), boxes, strict=True):
            theirs = _box_reference(values[np.array(groups) == group])
            assert_allclose(box["q1"], theirs[0], rtol=RTOL, atol=ATOL)
            assert_allclose(box["median"], theirs[1], rtol=RTOL, atol=ATOL)
            assert_allclose(box["q3"], theirs[2], rtol=RTOL, atol=ATOL)

    def test_the_median_and_the_whiskers_are_still_there(self):
        """Only the outlier markers go.

        The percentile plot is a box-and-whisker plot: a version that hid
        the median and the whiskers as well would show the interquartile
        range and nothing else.
        """
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_percentiles(groups, values)
        ax = _axes(fig)

        visible = [line for line in ax.lines if line.get_visible() and np.asarray(line.get_xdata()).size == 2]
        assert visible, "the percentile plot draws no lines at all"
        for line in visible:
            assert not line.get_marker(), "an outlier marker is still drawn"

    def test_the_outlier_markers_are_gone(self):
        """A box with an outlier in it must not show the outlier."""
        from visualization.generic_plots import GenericPlotter

        values = np.concatenate([np.linspace(0.0, 1.0, 20), [50.0]])
        groups = ["g"] * len(values)
        fig = GenericPlotter().plot_percentiles(groups, values)
        for line in _axes(fig).lines:
            assert not line.get_marker()


# ---------------------------------------------------------------------------
# Pie chart
# ---------------------------------------------------------------------------
_PIE_PROGRAM = """
sizes <- @@sizes@@
out <- cumsum(sizes) / sum(sizes) * 360
out
"""


class TestPieChart:
    """The slice angles are the cumulative shares."""

    def test_the_slice_angles_match_r(self):
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_pie_chart(values, categories=groups)
        wedges = _axes(fig).patches
        assert len(wedges) == 3

        widths = np.array([w.theta2 - w.theta1 for w in wedges], dtype=float)
        theirs = r_program_values(_PIE_PROGRAM, sizes=r_numbers(_totals(groups, values)))
        # The cumulative angle at the end of each slice.  The tolerance is
        # looser than elsewhere because matplotlib's ``pie`` casts the
        # slice sizes to float32, so the angles it stores carry only
        # single-precision accuracy.
        assert_allclose(np.cumsum(widths), theirs, rtol=1e-6, atol=1e-4)
        assert_allclose(float(widths.sum()), 360.0, rtol=1e-6, atol=1e-4)

    def test_the_first_slice_starts_at_the_top(self):
        from visualization.generic_plots import GenericPlotter

        groups, values = _groups()
        fig = GenericPlotter().plot_pie_chart(values, categories=groups)
        first = _axes(fig).patches[0]
        assert_allclose(first.theta1, 90.0, rtol=RTOL, atol=ATOL)


def _totals(groups: list[str], values: np.ndarray) -> list[float]:
    """The sum per category, in order of first appearance."""
    order: list[str] = []
    totals: dict[str, float] = {}
    for group, value in zip(groups, values, strict=True):
        if group not in totals:
            order.append(group)
            totals[group] = 0.0
        totals[group] += float(value)
    return [totals[group] for group in order]


# ---------------------------------------------------------------------------
# Stacked chart
# ---------------------------------------------------------------------------
class TestStackedChart:
    """The segments of a stacked bar are the columns of the matrix.

    The application draws one bar per column, so the bars come out in
    column-major order: all the rows of the first column, then all the
    rows of the second.  The tests put them back into row order before
    comparing, and the cumulative test works in the same order.
    """

    def test_the_segment_heights_match_r(self):
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1006)
        matrix = rng.uniform(1.0, 9.0, (4, 3))
        n_rows, n_cols = matrix.shape

        fig = GenericPlotter().plot_stacked_chart(matrix)
        mine = _bar_heights(fig).reshape(n_cols, n_rows).T.ravel()
        theirs = np.asarray(matrix, dtype=float).ravel()
        assert_allclose(mine, theirs, rtol=RTOL, atol=ATOL)

    def test_the_normalised_segments_are_row_percentages(self):
        """Normalising gives PERCENTAGES, not fractions.

        The y axis of a normalised stack is labelled "percent", so the
        segments have to add up to a hundred, not to one.
        """
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1007)
        matrix = rng.uniform(1.0, 9.0, (4, 3))
        n_rows, n_cols = matrix.shape

        fig = GenericPlotter().plot_stacked_chart(matrix, normalize=True)
        mine = _bar_heights(fig).reshape(n_cols, n_rows).T.ravel()
        expected = (matrix / matrix.sum(axis=1, keepdims=True) * 100.0).ravel()
        assert_allclose(mine, expected, rtol=RTOL, atol=ATOL)
        # Every row of a normalised stack adds up to a hundred.
        assert_allclose(
            mine.reshape(n_rows, n_cols).sum(axis=1),
            np.full(n_rows, 100.0),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_the_stacks_are_cumulative(self):
        """The second segment of a bar starts where the first one ended."""
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1008)
        matrix = rng.uniform(1.0, 9.0, (3, 4))
        n_rows, n_cols = matrix.shape

        fig = GenericPlotter().plot_stacked_chart(matrix)
        ax = _axes(fig)
        bottoms = np.array([patch.get_y() for patch in ax.patches], dtype=float)
        heights = _bar_heights(fig).reshape(n_cols, n_rows).T.ravel()

        # The bars come out column by column: the first n_rows patches
        # are the first column, and each bar of that column starts on the
        # baseline.
        assert_allclose(bottoms[:n_rows], np.zeros(n_rows), rtol=RTOL, atol=ATOL)
        # The second column of each bar starts on top of the first.
        for row in range(n_rows):
            assert_allclose(
                bottoms[n_rows + row],
                heights[row * n_cols],
                rtol=RTOL,
                atol=ATOL,
            )
            assert_allclose(
                bottoms[2 * n_rows + row],
                heights[row * n_cols] + heights[row * n_cols + 1],
                rtol=RTOL,
                atol=ATOL,
            )


# ---------------------------------------------------------------------------
# Normal probability plot
# ---------------------------------------------------------------------------
_QQNORM_PROGRAM = """
x <- @@x@@
n <- length(x)
p <- (1:n - 0.5) / n
y <- qnorm(p)
q.y <- quantile(y, c(0.25, 0.75), type = 7)
q.x <- quantile(x, c(0.25, 0.75), type = 7)
b <- (q.y[2] - q.y[1]) / (q.x[2] - q.x[1])
a <- q.y[1] - b * q.x[1]
out <- c(y, slope = 1 / b, intercept = -a / b)
out
"""


class TestNormalProbabilityPlot:
    """``stats::qqnorm`` -- the theoretical quantiles and the reference line."""

    def test_the_theoretical_quantiles_match_r(self):
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1009)
        values = rng.normal(10.0, 2.0, 40)

        fig = GenericPlotter().plot_normal_probability(values)
        line = _axes(fig).lines[0]
        mine = np.asarray(line.get_xdata(), dtype=float)

        theirs = r_program_values(_QQNORM_PROGRAM, x=r_numbers(values))
        assert_allclose(mine, theirs[:40], rtol=RTOL, atol=ATOL)

    def test_the_observed_values_are_the_sorted_data(self):
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1010)
        values = rng.normal(10.0, 2.0, 40)
        fig = GenericPlotter().plot_normal_probability(values)
        line = _axes(fig).lines[0]
        mine = np.asarray(line.get_ydata(), dtype=float)
        assert_allclose(mine, np.sort(values), rtol=RTOL, atol=ATOL)

    def test_the_reference_line_goes_through_the_quartiles(self):
        """The line is the quartile line, as the documentation says.

        It was a least-squares fit through every point, which is a
        different line: it leans on the tails, and on a Q-Q plot the tails
        are what the plot is there to show.  R's ``qqnorm`` draws the
        quartile line, and this checks the two agree.
        """
        from visualization.generic_plots import GenericPlotter

        rng = np.random.default_rng(1011)
        values = rng.normal(10.0, 2.0, 40)

        fig = GenericPlotter().plot_normal_probability(values)
        reference = _axes(fig).lines[1]
        xs = np.asarray(reference.get_xdata(), dtype=float)
        ys = np.asarray(reference.get_ydata(), dtype=float)
        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        intercept = ys[0] - slope * xs[0]

        theirs = r_program_values(_QQNORM_PROGRAM, x=r_numbers(values))
        assert_allclose(slope, theirs[40], rtol=RTOL, atol=ATOL)
        assert_allclose(intercept, theirs[41], rtol=RTOL, atol=ATOL)
