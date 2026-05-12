# =============================================================================
# FILE: views/__init__.py
# =============================================================================
"""
PaleoAST Views Package

This package implements the view layer using PyQt6 for the graphical
user interface, including main window, spreadsheet, dialogs, and
interactive visualization canvas.

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .ui_dialogs import (
    DiversityDialog,
    ImportDialog,
    NMDSOptionsDialog,
    PCADialog,
    PCoADialog,
    RarefactionDialog,
)
from .ui_main_window import MainWindow
from .ui_navigation import NavigationTree
from .ui_plot_canvas import InteractivePlotCanvas
from .ui_spreadsheet import ScientificSpreadsheet, SpreadsheetDelegate

__all__ = [
    "DiversityDialog",
    "ImportDialog",
    "InteractivePlotCanvas",
    "MainWindow",
    "NMDSOptionsDialog",
    "NavigationTree",
    "PCADialog",
    "PCoADialog",
    "RarefactionDialog",
    "ScientificSpreadsheet",
    "SpreadsheetDelegate",
]
