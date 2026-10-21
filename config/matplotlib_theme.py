"""The one place that maps the application's design system onto matplotlib.

Four places used to do this, and they did not agree:

  * the module-level ``plt.rcParams.update(...)`` in
    ``views/ui_plot_canvas.py``, which wrote matplotlib's own names;
  * ``PlotCanvas.setDarkTheme``'s own ``plt.rcParams.update(...)``;
  * ``PlotCanvas.theme_colors()``, which invented a set of role names of
    its own ("axes_bg", "figure_bg", "border", "grid", ...);
  * ``PlotCanvas._apply_axes_theme``, which read the third.

The first two wrote one vocabulary and the third another, so a colour
that reached one of them but not the others was right in one place and
wrong in another.  That is how a dark-theme canvas ends up with a white
plot area: the canvas re-tints the axes it already has, but a figure
created afterwards takes its colours from rcParams, and the two answers
were written separately.

This module owns the mapping.  :func:`theme_colors` gives the role names
the canvas reads, :func:`rc_params` gives matplotlib's own names, and
both come from the same design-system palette, so they cannot drift
apart.  The canvas now asks this module for both instead of writing
either.
"""

from __future__ import annotations

from config.design_system import FigureTypography, get_palette

__all__ = ["rc_params", "theme_colors"]


def theme_colors(dark: bool = False) -> dict[str, str]:
    """The role colours of one theme, under the names the canvas reads.

    The names are the canvas' own, not matplotlib's: a legend frame, a
    reference line and an axes face are different things from an rcParam,
    and the canvas asks for them by role.
    """
    palette = get_palette(dark)
    return {
        "text": palette.text_primary,
        "text_muted": palette.text_secondary,
        "axes_bg": palette.bg_secondary,
        "figure_bg": palette.bg_primary,
        "border": palette.border_light,
        "grid": palette.border_light,
        "legend_bg": palette.bg_primary,
        "legend_edge": palette.border_light,
        "reference": palette.text_secondary,
    }


#: The rcParams that are the same in both themes.  They are grouped here
#: so that a reader can see at a glance which parts of the mapping are a
#: choice about the *chart* and which are a choice about the *theme*.
_THEME_INDEPENDENT: dict[str, object] = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
}


def rc_params(dark: bool = False) -> dict[str, object]:
    """The full matplotlib rcParams mapping for one theme.

    Applying this dict to ``plt.rcParams`` makes every figure created
    afterwards match the application's theme, whether the theme is the
    light one or the dark one.
    """
    colors = theme_colors(dark)
    return {
        **_THEME_INDEPENDENT,
        "axes.facecolor": colors["axes_bg"],
        "figure.facecolor": colors["figure_bg"],
        "axes.edgecolor": colors["border"],
        "axes.labelcolor": colors["text"],
        "xtick.color": colors["text"],
        "ytick.color": colors["text"],
        "text.color": colors["text"],
        "grid.color": colors["grid"],
        **FigureTypography.rc_overrides(),
    }
