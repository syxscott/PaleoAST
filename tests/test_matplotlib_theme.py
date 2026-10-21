# =============================================================================
# FILE: tests/test_matplotlib_theme.py
# =============================================================================
"""
The matplotlib theme mapping lives in exactly one place.

Four places used to map the application's design system onto matplotlib:
the module-level ``plt.rcParams.update(...)`` in ``views/ui_plot_canvas.py``,
``PlotCanvas.setDarkTheme``'s own update, ``PlotCanvas.theme_colors()`` and
``PlotCanvas._apply_axes_theme``.  The first two wrote matplotlib's own
names; the third invented a set of role names of its own; the fourth read
the third.

Two answers to the same question is how a dark-theme canvas ends up with a
white plot area: the canvas re-tints the axes it already has, but a figure
created afterwards takes its colours from rcParams, and those two numbers
were written separately.

These tests pin the consolidation:

  * the light and the dark mapping cover the SAME set of rcParams, so
    switching the theme cannot leave a key behind;
  * the colours the canvas re-tints an existing axes with and the colours
    a new figure inherits are the same numbers;
  * the theme-independent part of the mapping really is the same in both
    themes -- otherwise "the same key set" would hide a difference.
"""

from __future__ import annotations

import matplotlib
import pytest

from config.design_system import FigureTypography, get_palette
from config.matplotlib_theme import rc_params, theme_colors


class TestMapping:
    """The pure mapping, with no Qt and no canvas."""

    def test_both_themes_cover_the_same_keys(self):
        """Switching the theme must not leave a key behind.

        A key present in one theme and absent in the other keeps whatever
        value it had, which is how a dark theme ends up with a light
        surface.
        """
        light = rc_params(False)
        dark = rc_params(True)
        assert set(light) == set(dark)

    def test_the_theme_independent_part_is_identical(self):
        """The keys that are not about the theme must not change with it.

        A key that is in the mapping but differs between the themes is a
        key that is really theme-dependent and is hiding in the wrong
        group, or a mapping that was written twice.
        """
        light = rc_params(False)
        dark = rc_params(True)
        for key in ("figure.dpi", "savefig.dpi", "axes.spines.top", "grid.linewidth"):
            assert light[key] == dark[key], f"{key} differs between the themes"

    def test_the_surfaces_come_from_the_palette(self):
        """Every colour in the mapping is a palette colour, not a literal."""
        for dark in (False, True):
            palette = get_palette(dark)
            colors = theme_colors(dark)
            params = rc_params(dark)
            assert colors["axes_bg"] == palette.bg_secondary
            assert colors["figure_bg"] == palette.bg_primary
            assert colors["text"] == palette.text_primary
            assert params["axes.facecolor"] == palette.bg_secondary
            assert params["figure.facecolor"] == palette.bg_primary
            assert params["text.color"] == palette.text_primary
            assert params["grid.color"] == palette.border_light

    def test_the_type_scale_is_in_the_mapping(self):
        """The figure type scale reaches matplotlib through the mapping."""
        for dark in (False, True):
            params = rc_params(dark)
            for key, value in FigureTypography.rc_overrides().items():
                assert params[key] == value, f"{key} is missing from the {dark=} mapping"

    def test_the_themes_really_do_differ(self):
        """A mapping that ignores the theme would pass every test above.

        This is the one that would notice: the light and the dark
        surfaces must be different colours, and the dark one must be the
        darker of the two.
        """
        light = rc_params(False)["axes.facecolor"]
        dark = rc_params(True)["axes.facecolor"]
        assert light != dark

        def luminance(colour: str) -> float:
            red = int(colour[1:3], 16) / 255
            green = int(colour[3:5], 16) / 255
            blue = int(colour[5:7], 16) / 255
            return 0.2126 * red + 0.7152 * green + 0.0722 * blue

        assert luminance(dark) < luminance(light)


class TestCanvasUsesIt:
    """The canvas really goes through the one mapping."""

    @pytest.fixture
    def qt_app(self):
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(["paleoast-matplotlib-theme"])
        return app

    def test_the_canvas_theme_colors_are_the_modules(self, qt_app):
        from views.ui_plot_canvas import InteractivePlotCanvas

        canvas = InteractivePlotCanvas()
        try:
            for dark in (False, True):
                canvas._is_dark_theme = dark
                assert canvas.theme_colors() == theme_colors(dark)
        finally:
            canvas._figure.clear()
            canvas.close()

    def test_switching_the_theme_puts_the_dark_colours_into_rcparams(self, qt_app):
        """A figure created AFTER the switch inherits the dark theme.

        This is the path the consolidation is for: the rcParams a new
        figure takes its colours from must be the same mapping the canvas
        re-tints an existing axes with.
        """
        from views.ui_plot_canvas import InteractivePlotCanvas

        before = matplotlib.rcParams.copy()
        canvas = InteractivePlotCanvas()
        try:
            canvas.setDarkTheme(True)
            params = rc_params(True)
            for key in ("axes.facecolor", "figure.facecolor", "text.color", "grid.color"):
                assert matplotlib.rcParams[key] == params[key], f"{key} did not reach rcParams"
            assert canvas.theme_colors()["axes_bg"] == params["axes.facecolor"]
        finally:
            matplotlib.rcParams.update(before)
            canvas._figure.clear()
            canvas.close()

    def test_switching_back_restores_the_light_colours(self, qt_app):
        from views.ui_plot_canvas import InteractivePlotCanvas

        before = matplotlib.rcParams.copy()
        canvas = InteractivePlotCanvas()
        try:
            canvas.setDarkTheme(True)
            canvas.setDarkTheme(False)
            params = rc_params(False)
            for key in ("axes.facecolor", "figure.facecolor", "text.color", "grid.color"):
                assert matplotlib.rcParams[key] == params[key], f"{key} did not come back"
        finally:
            matplotlib.rcParams.update(before)
            canvas._figure.clear()
            canvas.close()
