# =============================================================================
# FILE: config/colors.py
# =============================================================================
"""
Color Schemes and Visual Configuration for PaleoAST

This module defines all color palettes used throughout the application,
including UI colors, chart colors, and colorblind-friendly options.

Color Theory Notes:
- Primary/Secondary colors follow professional scientific visualization standards
- Chart colors are selected for maximum distinguishability
- Colorblind-friendly palette follows the Okabe-Ito color scheme

Author: PaleoAST Development Team
version: 1.1.2
"""

from typing import Final

# =============================================================================
# UI COLOR SCHEME
# =============================================================================

# Primary application colors
PRIMARY_COLOR: Final[str] = "#2C3E50"
"""
Primary UI color - deep blue-gray for headers and primary actions.
Provides professional, scientific appearance.
"""

SECONDARY_COLOR: Final[str] = "#3498DB"
"""
Secondary UI color - bright blue for interactive elements.
Offers clear visual distinction from primary elements.
"""

ACCENT_COLOR: Final[str] = "#E74C3C"
"""
Accent color - coral red for warnings and highlights.
Ensures critical information draws attention.
"""

SUCCESS_COLOR: Final[str] = "#27AE60"
"""
Success color - green for successful operations.
"""

WARNING_COLOR: Final[str] = "#F39C12"
"""
Warning color - amber for cautionary alerts.
"""

INFO_COLOR: Final[str] = "#16A085"
"""
Info color - teal for informational messages.
"""

DANGER_COLOR: Final[str] = "#C0392B"
"""
Danger color - dark red for destructive actions.
"""


# =============================================================================
# SPREADSHEET CELL COLORS
# =============================================================================

CELL_HEADER_BG: Final[str] = "#ECF0F1"
"""
Background color for spreadsheet headers.
Light gray for clear distinction from data cells.
"""

CELL_HEADER_TEXT: Final[str] = "#2C3E50"
"""
Text color for spreadsheet headers.
Dark to ensure readability.
"""

CELL_SELECTED_BG: Final[str] = "#3498DB"
"""
Background color for selected cells.
Bright blue for clear selection visibility.
"""

CELL_SELECTED_TEXT: Final[str] = "#FFFFFF"
"""
Text color for selected cells.
White text provides contrast on blue background.
"""

CELL_EDITING_BG: Final[str] = "#FFFFFF"
"""
Background color for cells being edited.
White for clean text input appearance.
"""

CELL_GROUP_COLUMN_BG: Final[str] = "#E8F6F3"
"""
Background color for group-designated columns.
Light green tint indicates special column status.
"""

CELL_MISSING_VALUE_BG: Final[str] = "#FDF2E9"
"""
Background color for cells with missing values.
Orange tint indicates data quality issue.
"""


# =============================================================================
# CHART COLOR PALETTES
# =============================================================================

# Standard category colors for charts
CHART_COLORS: Final[list] = [
    "#0077BB",  # Blue
    "#EE7733",  # Orange
    "#009988",  # Teal
    "#CC3311",  # Red
    "#33BBEE",  # Light Blue
    "#EE3377",  # Pink
    "#BBBBBB",  # Gray
    "#000000",  # Black
]
"""
Standard color palette for multi-category charts.
8 distinct colors suitable for most visualization needs.
"""

# Extended palette for datasets with many groups
CHART_COLORS_EXTENDED: Final[list] = [
    "#332288",  # Indigo
    "#88CCEE",  # Cyan
    "#44AA99",  # Teal
    "#117733",  # Green
    "#999933",  # Olive
    "#CC6677",  # Rose
    "#882255",  # Purple
    "#AA4499",  # Violet
    "#DDDDDD",  # Light Gray
]
"""
Extended color palette for datasets with 10+ categories.
"""


# =============================================================================
# COLORBLIND-FRIENDLY PALETTE
# =============================================================================

# Okabe-Ito color palette - designed for color vision deficiency.
#
# The first eight are Okabe & Ito's own set, in their order -- which means the
# eighth is BLACK. This list used to end ... "#CC79A7", "#999999": grey had
# been substituted for black, so the eighth group in an in-app figure was grey
# while the eighth group in the exported R figure was black. The preview was
# not previewing. The ninth entry is a spare grey, carried over from the old
# list so a ninth group still gets a colour that is not black-on-white.
COLORBLIND_FRIENDLY_PALETTE: Final[list] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
    "#999999",  # Gray (spare, not part of Okabe-Ito)
]
"""
Okabe-Ito colorblind-friendly palette.
Optimized for deuteranopia, protanopia, and tritanopia.
Reference: Okabe & Ito (2002) Color Universal Design
"""

# IBM Color Blind Safe palette
IBM_COLORBLIND_SAFE: Final[list] = [
    "#648FFF",  # Blue 70
    "#785EF0",  # Purple 70
    "#DC267F",  # Magenta 70
    "#FE6100",  # Orange 70
    "#FFB000",  # Gold 70
]
"""
IBM Design colorblind-safe palette.
High contrast and distinguishable across color vision types.
"""


# =============================================================================
# MATPLOTLIB STYLESHEET CONFIGURATION
# =============================================================================

MATPLOTLIB_STYLE_PARAMS: Final[dict] = {
    # Font settings following Nature/Science guidelines
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 12,
    # Figure background
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "figure.autolayout": True,
    # Axes settings
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "axes.grid": True,
    "axes.grid.alpha": 0.3,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.labelcolor": "black",
    "axes.axisbelow": True,
    # Grid settings
    "grid.color": "#CCCCCC",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.4,
    # Line settings
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    # Legend settings
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    # Tick settings
    "xtick.color": "black",
    "xtick.direction": "out",
    "xtick.labelsize": 10,
    "xtick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.color": "black",
    "ytick.direction": "out",
    "ytick.labelsize": 10,
    "ytick.major.size": 6,
    "ytick.major.width": 1.2,
    # Savefig settings
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
"""
Matplotlib style parameters for publication-ready figures.
Follows Nature and Science journal formatting guidelines.
"""


# =============================================================================
# MARKER DEFINITIONS
# =============================================================================

CHART_MARKERS: Final[list] = [
    "o",  # Circle
    "s",  # Square
    "^",  # Triangle up
    "D",  # Diamond
    "v",  # Triangle down
    "p",  # Pentagon
    "h",  # Hexagon
    "*",  # Star
    "+",  # Plus
    "x",  # X
    "d",  # Thin diamond
    "2",  # Triangle up (Unicode alternative)
]
"""
Standard marker styles for scatter plots and charts.
Ensures clear distinction between groups in monochrome printing.
"""

# Marker cycle for use in matplotlibrc style cycling
MARKER_CYCLE: Final[list] = [
    {"marker": "o", "markersize": 8, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "s", "markersize": 8, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "^", "markersize": 8, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "D", "markersize": 7, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "v", "markersize": 8, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "p", "markersize": 8, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "h", "markersize": 8, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.5},
    {"marker": "*", "markersize": 10, "markerfacecolor": "auto", "markeredgecolor": "auto", "markeredgewidth": 1.0},
]
"""
Complete marker style specifications for cycled use.
Includes size and width adjustments for visual consistency.
"""


# =============================================================================
# GRADIENT COLORMAPS
# =============================================================================

# Sequential colormaps for continuous data
SEQUENTIAL_COLORMAPS: Final[dict] = {
    "viridis": "Perceptually uniform sequential colormap",
    "plasma": "Plasma colormap with warm tones",
    "inferno": "High-contrast dark sequential",
    "magma": "Dark sequential with warm colors",
    "cividis": "Colorblind-friendly sequential",
    "Blues": "Light to dark blue sequential",
    "Reds": "Light to dark red sequential",
    "Greens": "Light to dark green sequential",
}
"""
Sequential colormaps for gradient data visualization.
All are perceptually uniform for accurate data representation.
"""

# Diverging colormaps for data with meaningful center
DIVERGING_COLORMAPS: Final[dict] = {
    "RdBu": "Red-Blue diverging (classic)",
    "RdYlBu": "Red-Yellow-Blue diverging",
    "PiYG": "Pink-Green diverging",
    "PRGn": "Purple-Green diverging",
    "BrBG": "Brown-Blue-Green diverging",
    "seismic": "Blue-White-Red diverging",
    "coolwarm": "Blue-White-Red (balanced)",
}
"""
Diverging colormaps for data with meaningful zero or center point.
Useful for showing deviations from a reference value.
"""

# Categorical colormaps for discrete data
CATEGORICAL_COLORMAPS: Final[dict] = {
    "Set1": "8-color categorical (good for ≤8 categories)",
    "Set2": "8-color pastel categorical",
    "Set3": "12-color categorical",
    "tab10": "10-color categorical (matplotlib default)",
    "tab20": "20-color categorical",
    "Paired": "12-color paired categorical",
    "Accent": "8-color accent categorical",
}
"""
Categorical colormaps for discrete data categories.
Should NOT be used for continuous data.
"""


# =============================================================================
# EXPORT FORMAT COLORS (for PDF/SVG transparency)
# =============================================================================

TRANSPARENT_FILL: Final[str] = "none"
"""Transparent fill for vector graphics."""

DEFAULT_EDGE_COLOR: Final[str] = "#000000"
"""Default edge color for vector graphic outlines."""

DEFAULT_LINE_COLOR: Final[str] = "#333333"
"""Default line color for vector graphic strokes."""


# =============================================================================
# COMPATIBILITY ALIASES (for backward compatibility)
# =============================================================================

# Alias for CATEGORY_COLORS
CATEGORY_COLORS: Final[list] = CHART_COLORS
"""
Alias for CHART_COLORS for backward compatibility.
"""


# =============================================================================
# CATEGORICAL PALETTES -- THE SINGLE SOURCE OF TRUTH
# =============================================================================
#
# The interactive figures (matplotlib) and the exported figures (real ggplot2)
# must be the same picture. They can only be the same picture if both sides
# read the SAME hex values, so they are defined here once and the generated R
# script is handed this list verbatim rather than keeping its own copy.
#
# Previously each side had its own table with its own NAMES and its own values:
# R offered okabeito/greyscale/dark2/viridis while Python offered
# default/colorblind/extended/ibm, so nothing lined up, and the two Okabe-Ito
# tables disagreed about the eighth colour. Two tables means the preview cannot
# promise to match the export.
#
# The four values below are what R actually produced when asked, captured with
# R 4.5.2 / RColorBrewer / viridisLite, so the R script now needs neither
# package: it receives the colours outright instead of computing them.
# viridis is listed without its trailing "FF" alpha because alpha=1 is opaque;
# that keeps one spelling that both matplotlib and R read identically.

PALETTES: Final[dict[str, list[str]]] = {
    "okabeito": list(COLORBLIND_FRIENDLY_PALETTE),
    "greyscale": [
        "#0D0D0D",
        "#5A5A5A",
        "#7B7B7B",
        "#949494",
        "#A8A8A8",
        "#BABABA",
        "#CACACA",
        "#D9D9D9",
    ],
    "dark2": [
        "#1B9E77",
        "#D95F02",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
        "#A6761D",
        "#666666",
    ],
    "viridis": [
        "#440154",
        "#46337E",
        "#365C8D",
        "#277F8E",
        "#1FA187",
        "#4AC16D",
        "#9FDA3A",
        "#FDE725",
    ],
}
"""
The categorical palettes, keyed by the name the Preferences dialog stores.

``okabeito`` is the default: Okabe & Ito's set is separable under
deuteranopia, protanopia and tritanopia, and it survives greyscale, which is
what a journal photocopy demands.
"""

DEFAULT_PALETTE_NAME: Final[str] = "okabeito"
"""The palette used until the user picks another one."""

# Older spellings kept so existing callers keep working, mapped onto the
# canonical names above. "default" is deliberately absent: it used to mean
# CHART_COLORS while the application actually defaulted to something else
# entirely, and a name that resolves to a different palette than the app's
# real default is exactly the kind of quiet surprise this registry exists to
# remove.
PALETTE_ALIASES: Final[dict[str, str]] = {
    "colorblind": "okabeito",
    "okabe-ito": "okabeito",
}

_current_palette_name: str = DEFAULT_PALETTE_NAME


def palette_names() -> list[str]:
    """The canonical palette names, in the order a figure is judged by them."""
    return list(PALETTES)


def set_current_palette(name: str) -> list[str]:
    """Select the palette the interactive figures draw with.

    Called when preferences are applied. Returns the palette in use, so a
    caller can assert on it.
    """
    global _current_palette_name
    _current_palette_name = _canonical_palette_name(name)
    return PALETTES[_current_palette_name]


def current_palette() -> list[str]:
    """The palette the interactive figures should draw with.

    This is what the plotters call. It tracks the same preference the R
    export reads, so the in-app figure and the exported figure are the same
    figure.

    Returns a copy, like :func:`get_color_scheme`: handing out the registry's
    own list would let one caller appending a colour change every figure drawn
    after it.
    """
    return list(PALETTES[_current_palette_name])


def current_palette_name() -> str:
    """The name behind :func:`current_palette`."""
    return _current_palette_name


def _canonical_palette_name(name: str) -> str:
    resolved = PALETTE_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if resolved not in PALETTES:
        raise ValueError(
            f"unknown palette {name!r}; expected one of {sorted(PALETTES)} or an alias in {sorted(PALETTE_ALIASES)}"
        )
    return resolved


def resolve_palette_name(name: str = DEFAULT_PALETTE_NAME) -> str:
    """Canonical name for ``name``, or :class:`ValueError` if unknown.

    Callers that need the name as well as the colours -- the R export writes
    the chosen palette's name into the generated script -- go through here
    rather than reaching into the registry themselves.
    """
    return _canonical_palette_name(name)


def get_color_scheme(name: str = DEFAULT_PALETTE_NAME) -> list:
    """
    Get a palette by name.

    Parameters:
        name: A key of :data:`PALETTES`, or an alias in :data:`PALETTE_ALIASES`.

    Returns:
        A copy of the palette's hex colour codes.

    Raises:
        ValueError: if ``name`` is not a palette.

    An unknown name used to return CHART_COLORS without a word, so a typo --
    or a name copied from the R side, where the palette is called
    ``okabeito`` rather than ``colorblind`` -- produced a figure in a
    different palette and no indication of it. Two callers were relying on
    that behaviour to mean "the application's categorical palette"; they now
    call :func:`current_palette`, which is what they meant.
    """
    return list(PALETTES[_canonical_palette_name(name)])
