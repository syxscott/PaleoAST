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
version: 1.0.1
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

# Okabe-Ito color palette - designed for color vision deficiency
COLORBLIND_FRIENDLY_PALETTE: Final[list] = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray
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


# Color scheme getter
def get_color_scheme(name: str = "default") -> list:
    """
    Get color scheme by name.

    Parameters:
        name: Scheme name ('default', 'colorblind', 'extended')

    Returns:
        List of hex color codes
    """
    schemes = {
        "default": CHART_COLORS,
        "colorblind": COLORBLIND_FRIENDLY_PALETTE,
        "extended": CHART_COLORS_EXTENDED,
        "ibm": IBM_COLORBLIND_SAFE,
    }
    return schemes.get(name, CHART_COLORS)
