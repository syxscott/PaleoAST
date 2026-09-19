# =============================================================================
# Test: strat_column - dual-channel columns, ICS strips, log axis
# =============================================================================
"""
Tests for visualization.strat_column (W8 borrow).

Covers:
- lithology style resolution (exact, longest-substring, unknown, overrides)
- dual-channel rendering: pcolormesh fills + one PathPatch hatch per bed
- depth vs age axes, orientation (young/shallow on top), validation errors
- cascading ICS strips: clipped boundary ticks, colours, BT.601 font colour,
  fit-or-drop blanking of labels in thin cells
- log age axis: geometric label centres, rejection of 0 Ma
- multi-column correlation figure: shared limits, headers
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.patches import PathPatch, Rectangle

from visualization.strat_column import (
    UNKNOWN_LITHOLOGY_STYLE,
    StratigraphicColumnPlotter,
    _resolve_style,
)


@pytest.fixture(scope="module")
def plotter() -> StratigraphicColumnPlotter:
    return StratigraphicColumnPlotter()


@pytest.fixture(scope="module")
def age_fig(plotter: StratigraphicColumnPlotter):
    return plotter.plot_column(
        [34, 40, 5],
        ["fine sandstone", "marine limestone", "gray shale"],
        layer_bound_ages=[66.0, 100.0, 140.0, 145.0],
        geo_ranks=("eon", "era", "period", "epoch"),
    )


def _hatches(ax) -> list[str]:
    return [p.get_hatch() for p in ax.patches if isinstance(p, PathPatch) and p.get_hatch()]


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts if t.get_text()]


def _strip_rects(ax, x_max: float = 0.0) -> list[Rectangle]:
    return [
        p
        for p in ax.patches
        if isinstance(p, Rectangle) and p.get_x() < x_max and p.get_facecolor()[3] > 0
    ]


class TestStyleResolution:
    def test_exact_key(self):
        assert _resolve_style("sandstone", {"sandstone": ("#fff", "/")}) == ("#fff", "/")

    def test_case_and_whitespace_normalised(self):
        assert _resolve_style("  Shale ", {"shale": ("#708090", "|")}) == ("#708090", "|")

    def test_longest_substring_wins(self):
        table = {"stone": ("a", ""), "sandstone": ("b", "/")}
        assert _resolve_style("fine-grained sandstone", table) == ("b", "/")

    def test_unknown_falls_back(self):
        assert _resolve_style("unobtainium", {}) == UNKNOWN_LITHOLOGY_STYLE

    def test_plotter_table_is_overridable(self, plotter: StratigraphicColumnPlotter):
        original = dict(plotter.lithology_styles)
        try:
            plotter.lithology_styles["sandstone"] = ("#123456", "///")
            fig = plotter.plot_column([1], ["sandstone"])
            assert _hatches(fig.axes[0]) == ["///"]
        finally:
            plotter.lithology_styles = original


class TestDepthColumn:
    def test_one_hatch_per_hatched_bed(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column([2, 3, 1.5], ["sandstone", "shale", "coal"])
        ax = fig.axes[0]
        assert _hatches(ax) == ["/", "|"]  # coal has no hatch overlay

    def test_zero_at_top(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column([2, 3], ["sandstone", "shale"])
        bottom, top = fig.axes[0].get_ylim()
        assert bottom == pytest.approx(5.0) and top == pytest.approx(0.0)

    def test_lithology_labels_rendered(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column([2, 3], ["sandstone", "weird stuff"], title="t")
        assert "weird stuff" in _texts(fig.axes[0])

    def test_no_strips_in_depth_mode(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column([2], ["sandstone"], layer_bound_ages=None, geo_ranks=("period",))
        assert not _strip_rects(fig.axes[0])

    def test_tops_mismatch_raises(self, plotter: StratigraphicColumnPlotter):
        with pytest.raises(ValueError, match="tops must have"):
            plotter.plot_column([1, 2], ["a", "b"], tops=[0.0])

    def test_log_rejected_on_depth(self, plotter: StratigraphicColumnPlotter):
        with pytest.raises(ValueError, match="age axis"):
            plotter.plot_column([1], ["a"], log_scale=True)


class TestAgeColumn:
    def test_youngest_on_top(self, age_fig):
        bottom, top = age_fig.axes[0].get_ylim()
        assert bottom == pytest.approx(145.0) and top == pytest.approx(66.0)

    def test_ticks_are_clipped_boundaries(self, age_fig):
        labels = [t.get_text() for t in age_fig.axes[0].get_yticklabels()]
        assert labels == ["66", "145"]

    def test_ics_colours_used(self, age_fig):
        cols = {mcolors.to_hex(r.get_facecolor()) for r in _strip_rects(age_fig.axes[0])}
        assert "#7fc64e" in cols  # Cretaceous period fill
        assert "#67c5ca" in cols  # Mesozoic era fill

    def test_strip_names_present(self, age_fig):
        texts = _texts(age_fig.axes[0])
        assert {"Phanerozoic", "Mesozoic", "Cretaceous"} <= set(texts)

    def test_hatch_channel_survives_age_mode(self, age_fig):
        assert _hatches(age_fig.axes[0]) == ["/", "xx", "|"]

    def test_bad_lengths_raise(self, plotter: StratigraphicColumnPlotter):
        with pytest.raises(ValueError, match="layer_bound_ages must have"):
            plotter.plot_column([1, 2], ["a", "b"], layer_bound_ages=[0.0, 5.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            plotter.plot_column([1, 2], ["a", "b"], layer_bound_ages=[0.0, 5.0, 3.0])

    def test_log_rejects_zero(self, plotter: StratigraphicColumnPlotter):
        with pytest.raises(ValueError, match="strictly positive"):
            plotter.plot_column([1], ["a"], layer_bound_ages=[0.0, 5.0], log_scale=True)


class TestStripLabels:
    def test_dark_fill_gets_white_text(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column(
            [10, 20], ["shale", "limestone"], layer_bound_ages=[443.8, 464.0, 485.4], geo_ranks=("period",)
        )
        txt = next(t for t in fig.axes[0].texts if t.get_text() == "Ordovician")
        assert mcolors.to_hex(txt.get_color()) == "#ffffff"

    def test_light_fill_gets_black_text(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column(
            [5], ["chalk"], layer_bound_ages=[66.0, 100.0], geo_ranks=("period",)
        )
        txt = next(t for t in fig.axes[0].texts if t.get_text() == "Cretaceous")
        assert mcolors.to_hex(txt.get_color()) == "#000000"

    def test_thin_cell_label_is_blank(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_column(
            [30, 3.9], ["shale", "sandstone"], layer_bound_ages=[0.0, 0.0117, 34.0], geo_ranks=("epoch",)
        )
        texts = _texts(fig.axes[0])
        assert "Pleistocene" in texts
        assert "Holocene" not in texts  # 0.0117 Myr cell: blanked by autofit

    def test_log_centre_is_geometric_mean(self, plotter: StratigraphicColumnPlotter):
        import math

        fig = plotter.plot_column(
            [34, 40], ["shale", "sandstone"], layer_bound_ages=[66.0, 100.0, 145.0],
            geo_ranks=("period",), log_scale=True,
        )
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        txt = next(t for t in ax.texts if t.get_text() == "Cretaceous")
        assert txt.get_position()[1] == pytest.approx(math.sqrt(66.0 * 145.0), rel=1e-6)


class TestCorrelatedColumns:
    def test_shared_limits_and_headers(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_correlated_columns(
            [
                {"name": "Alpha", "bound_ages": [0.0117, 34.0, 66.0], "lithologies": ["sandstone", "shale"]},
                {"name": "Beta", "bound_ages": [23.03, 66.0, 100.0], "lithologies": ["chalk", "tuff"]},
            ],
            geo_ranks=("period",),
            log_scale=True,
        )
        ax = fig.axes[0]
        bottom, top = ax.get_ylim()
        assert bottom == pytest.approx(100.0) and top == pytest.approx(0.0117)
        assert {"Alpha", "Beta"} <= set(_texts(ax))
        # 2 sections x 2 beds, all hatched styles
        assert len(_hatches(ax)) == 4

    def test_empty_sections_raise(self, plotter: StratigraphicColumnPlotter):
        with pytest.raises(ValueError, match="at least one"):
            plotter.plot_correlated_columns([])

    def test_shape_mismatch_raises(self, plotter: StratigraphicColumnPlotter):
        with pytest.raises(ValueError, match="bound_ages must be"):
            plotter.plot_correlated_columns(
                [{"name": "A", "bound_ages": [0.0, 1.0], "lithologies": ["a", "b"]}]
            )

    def test_no_log_no_strips_works(self, plotter: StratigraphicColumnPlotter):
        fig = plotter.plot_correlated_columns(
            [{"name": "", "bound_ages": [0.0, 10.0], "lithologies": ["shale"]}], geo_ranks=()
        )
        assert _hatches(fig.axes[0]) == ["|"]


class TestHatchLinewidthIsLocal:
    def test_rcparam_restored(self, plotter: StratigraphicColumnPlotter):
        before = matplotlib.rcParams["hatch.linewidth"]
        plotter.plot_column([1, 2], ["sandstone", "shale"])
        assert matplotlib.rcParams["hatch.linewidth"] == before
        assert np.isfinite(before)
