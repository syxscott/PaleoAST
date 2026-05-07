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

from .constants import (
    APP_NAME,
    APP_VERSION,
    APP_AUTHOR,
    APP_DESCRIPTION,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    SPREADSHEET_MAX_ROWS,
    SPREADSHEET_MAX_COLS,
    MAX_WORKERS,
    PERMUTATION_TESTS,
    NMDS_MAX_ITERATIONS,
    NMDS_RANDOM_RESTARTS,
)

from .colors import (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    CHART_COLORS,
    CHART_MARKERS,
    COLORBLIND_FRIENDLY_PALETTE,
)

__all__ = [
    'APP_NAME',
    'APP_VERSION',
    'APP_AUTHOR',
    'APP_DESCRIPTION',
    'DEFAULT_FONT_FAMILY',
    'DEFAULT_FONT_SIZE',
    'WINDOW_MIN_WIDTH',
    'WINDOW_MIN_HEIGHT',
    'SPREADSHEET_MAX_ROWS',
    'SPREADSHEET_MAX_COLS',
    'MAX_WORKERS',
    'PERMUTATION_TESTS',
    'NMDS_MAX_ITERATIONS',
    'NMDS_RANDOM_RESTARTS',
    'PRIMARY_COLOR',
    'SECONDARY_COLOR',
    'ACCENT_COLOR',
    'CHART_COLORS',
    'CHART_MARKERS',
    'COLORBLIND_FRIENDLY_PALETTE',
]
