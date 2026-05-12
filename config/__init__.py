# =============================================================================
# FILE: config/__init__.py
# =============================================================================
"""
PaleoAST Configuration Package

This package contains all configuration-related modules for the PaleoAST
application, including constants, color schemes, and validators.

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .colors import (
    ACCENT_COLOR,
    CHART_COLORS,
    CHART_MARKERS,
    COLORBLIND_FRIENDLY_PALETTE,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
)
from .constants import (
    APP_AUTHOR,
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    MAX_WORKERS,
    NMDS_MAX_ITERATIONS,
    NMDS_RANDOM_RESTARTS,
    PERMUTATION_TESTS,
    SPREADSHEET_MAX_COLS,
    SPREADSHEET_MAX_ROWS,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from .i18n import (
    _,
    get_language,
    get_translator,
    register_translations,
    set_language,
)

__all__ = [
    "ACCENT_COLOR",
    "APP_AUTHOR",
    "APP_DESCRIPTION",
    "APP_NAME",
    "APP_VERSION",
    "CHART_COLORS",
    "CHART_MARKERS",
    "COLORBLIND_FRIENDLY_PALETTE",
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_FONT_SIZE",
    "MAX_WORKERS",
    "NMDS_MAX_ITERATIONS",
    "NMDS_RANDOM_RESTARTS",
    "PERMUTATION_TESTS",
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "SPREADSHEET_MAX_COLS",
    "SPREADSHEET_MAX_ROWS",
    "WINDOW_MIN_HEIGHT",
    "WINDOW_MIN_WIDTH",
    "_",
    "get_language",
    "get_translator",
    "register_translations",
    "set_language",
]
