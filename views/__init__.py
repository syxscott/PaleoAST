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
from .ui_pcm_dialogs import AncestralStateDialog, PhyloANOVADialog, PhyloSignalDialog, PICDialog
from .ui_plot_canvas import InteractivePlotCanvas
from .ui_spreadsheet import ScientificSpreadsheet, SpreadsheetDelegate

from .ui_allometry_dialogs import AllometryDialog, PLSDialog
from .ui_beta_diversity_dialogs import BetaDiversityDialog, CoverageRarefactionDialog
from .ui_extinction_dialogs import ExtinctionIntervalDialog
from .ui_null_model_dialogs import NullModelDialog
from .ui_evolution_rate_dialogs import EvolutionRateDialog

__all__ = [
    "AllometryDialog",
    "AncestralStateDialog",
    "BetaDiversityDialog",
    "CoverageRarefactionDialog",
    "DiversityDialog",
    "EvolutionRateDialog",
    "ExtinctionIntervalDialog",
    "ImportDialog",
    "InteractivePlotCanvas",
    "MainWindow",
    "NMDSOptionsDialog",
    "NavigationTree",
    "NullModelDialog",
    "PCADialog",
    "PCoADialog",
    "PhyloANOVADialog",
    "PhyloSignalDialog",
    "PICDialog",
    "PLSDialog",
    "RarefactionDialog",
    "ScientificSpreadsheet",
    "SpreadsheetDelegate",
]
