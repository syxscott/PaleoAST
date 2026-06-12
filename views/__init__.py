# =============================================================================
# FILE: views/__init__.py
# =============================================================================
"""
PaleoAST Views Package

This package implements the view layer using PyQt6 for the graphical
user interface, including main window, spreadsheet, dialogs, and
interactive visualization canvas.

Author: PaleoAST Development Team
version: 1.0.1
"""

from .diagnostic_console import ConsoleLogHandler, DiagnosticConsole
from .floating_toolbar import FloatingToolBar
from .ui_allometry_dialogs import AllometryDialog, PLSDialog
from .ui_beta_diversity_dialogs import BetaDiversityDialog, CoverageRarefactionDialog
from .ui_dialogs import (
    BiostratigraphyDialog,
    DiversityDialog,
    ImportDialog,
    IsotopeAnalysisDialog,
    NMDSOptionsDialog,
    PCADialog,
    PCoADialog,
    PaleoEnvironmentDialog,
    RarefactionDialog,
    StratigraphicCorrelationDialog,
)
from .ui_evolution_rate_dialogs import EvolutionRateDialog
from .ui_extinction_dialogs import ExtinctionIntervalDialog
from .ui_imputation_dialog import ImputationDialog
from .ui_main_window import MainWindow
from .ui_navigation import NavigationTree
from .ui_null_model_dialogs import NullModelDialog
from .ui_pcm_dialogs import AncestralStateDialog, PhyloANOVADialog, PhyloSignalDialog, PICDialog
from .ui_plot_canvas import InteractivePlotCanvas
from .ui_spreadsheet import ScientificSpreadsheet, SpreadsheetDelegate

__all__ = [
    "AllometryDialog",
    "AncestralStateDialog",
    "BetaDiversityDialog",
    "BiostratigraphyDialog",
    "ConsoleLogHandler",
    "CoverageRarefactionDialog",
    "DiagnosticConsole",
    "DiversityDialog",
    "EvolutionRateDialog",
    "ExtinctionIntervalDialog",
    "FloatingToolBar",
    "ImportDialog",
    "ImputationDialog",
    "InteractivePlotCanvas",
    "IsotopeAnalysisDialog",
    "MainWindow",
    "NMDSOptionsDialog",
    "NavigationTree",
    "NullModelDialog",
    "PCADialog",
    "PCoADialog",
    "PICDialog",
    "PLSDialog",
    "PaleoEnvironmentDialog",
    "PhyloANOVADialog",
    "PhyloSignalDialog",
    "RarefactionDialog",
    "ScientificSpreadsheet",
    "SpreadsheetDelegate",
    "StratigraphicCorrelationDialog",
]
