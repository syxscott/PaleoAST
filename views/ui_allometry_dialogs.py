# =============================================================================
# FILE: views/ui_allometry_dialogs.py
# =============================================================================
"""
Allometry and Morphological Integration Dialogs for PaleoAST

Provides dialogs for:
    - Allometry analysis (size-shape regression)
    - Two-Block PLS analysis (morphological integration)

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.design_system import Typography, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


class BaseAllometryDialog(QDialog):
    """
    Base dialog for allometry and integration analyses.

    Provides common UI structure:
        - GPA data selection
        - Results display
    """

    resultsReady = pyqtSignal(dict)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.{title}")
        self._is_dark_theme = False

        self.setWindowTitle(title)
        self.setMinimumSize(700, 600)
        self.setModal(True)

        self._setup_ui()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        t = Typography()
        self.setStyleSheet(
            f"QDialog {{ background-color: {c.bg_primary}; color: {c.text_primary}; }}"
            f"QLabel {{ color: {c.text_primary}; font-size: {t.body_size}px; }}"
            f"QGroupBox {{ color: {c.text_primary}; font-weight: {t.medium}; "
            f"border: 1px solid {c.border_light}; border-radius: 4px; }}"
            f"QTextEdit {{ background-color: {c.bg_primary}; color: {c.text_primary}; "
            f"border: 1px solid {c.border_light}; border-radius: 4px; "
            f"font-family: 'Consolas', monospace; font-size: {t.body_sm_size}px; }}"
        )

    def _setup_ui(self) -> None:
        """Setup common UI structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        t = Typography()
        title_label = QLabel(self.windowTitle())
        title_font = QFont(t.family_primary, t.h4_size, QFont.Weight.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # GPA Data input
        data_group = QGroupBox(_("GPA-Aligned Data"))
        data_layout = QVBoxLayout(data_group)

        data_info = QLabel(
            _(
                "Select a GPA result from the workspace to analyze.\n"
                "The aligned configurations will be used for allometry/integration analysis."
            )
        )
        data_info.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        data_layout.addWidget(data_info)

        self._gpa_result_label = QLabel(_("No GPA result selected"))
        self._gpa_result_label.setStyleSheet("font-weight: bold; padding: 4px;")
        data_layout.addWidget(self._gpa_result_label)

        layout.addWidget(data_group)

        # Method-specific content (subclasses override)
        self._method_widget = QWidget()
        QVBoxLayout(self._method_widget)
        layout.addWidget(self._method_widget, 1)

        # Results
        results_group = QGroupBox(_("Results"))
        results_layout = QVBoxLayout(results_group)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(150)
        results_layout.addWidget(self._results_text)

        layout.addWidget(results_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._run_button = QPushButton(_("Run Analysis"))
        self._run_button.clicked.connect(self._on_run)
        button_layout.addWidget(self._run_button)

        self._close_button = QPushButton(_("Close"))
        self._close_button.clicked.connect(self.accept)
        button_layout.addWidget(self._close_button)

        layout.addLayout(button_layout)

    def set_gpa_data_info(self, info: str) -> None:
        """Set GPA result info label."""
        self._gpa_result_label.setText(info)

    def _on_run(self) -> None:
        """Run the analysis. Subclasses implement specific logic."""
        raise NotImplementedError


class AllometryDialog(BaseAllometryDialog):
    """
    Allometry Analysis Dialog.

    Analyzes relationship between centroid size and shape using
    multivariate regression of Procrustes coordinates on log centroid size.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Allometry Analysis (Size-Shape Relationship)"), parent)

        method_layout = self._method_widget.layout()

        # Options
        opts_group = QGroupBox(_("Options"))
        opts_layout = QVBoxLayout(opts_group)

        self._use_pca_check = QCheckBox(_("Reduce dimensionality with PCA"))
        self._use_pca_check.setChecked(False)
        opts_layout.addWidget(self._use_pca_check)

        pca_layout = QHBoxLayout()
        pca_layout.addWidget(QLabel(_("Number of components:")))
        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(2, 100)
        self._n_components_spin.setValue(10)
        self._n_components_spin.setEnabled(False)
        pca_layout.addWidget(self._n_components_spin)
        opts_layout.addLayout(pca_layout)

        self._use_pca_check.toggled.connect(self._n_components_spin.setEnabled)

        opts_layout.addStretch()
        method_layout.addWidget(opts_group)

    def _on_run(self) -> None:
        """Run allometry analysis."""
        try:
            # Get GPA data from state or use sample data
            # In real usage, this would come from the current workspace
            # For now, create sample data for testing

            QMessageBox.information(
                self,
                _("Information"),
                _(
                    "Allometry analysis requires GPA-aligned configurations.\n"
                    "Please run GPA analysis first and ensure data is loaded."
                ),
            )
        except Exception as e:
            self._logger.error(f"Allometry failed: {e}")
            from views.ui_main_window import format_user_error

            QMessageBox.critical(self, _("Error"), format_user_error(e, "异速生长分析"))


class PLSDialog(BaseAllometryDialog):
    """
    Two-Block Partial Least Squares (Integration) Dialog.

    Measures morphological integration between two blocks of shape variables
    using 2B-PLS analysis.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Morphological Integration (2B-PLS)"), parent)

        method_layout = self._method_widget.layout()

        # Block division options
        division_group = QGroupBox(_("Block Division Method"))
        division_layout = QVBoxLayout(division_group)

        self._division_combo = QComboBox()
        self._division_combo.addItems(
            [
                _("Anterior-Posterior Split"),
                _("Size-Matched Split"),
                _("Random Split"),
            ]
        )
        division_layout.addWidget(QLabel(_("How to divide landmarks into two blocks:")))
        division_layout.addWidget(self._division_combo)

        division_layout.addStretch()
        method_layout.addWidget(division_group)

        # PLS options
        pls_group = QGroupBox(_("PLS Options"))
        pls_layout = QVBoxLayout(pls_group)

        n_comp_layout = QHBoxLayout()
        n_comp_layout.addWidget(QLabel(_("Number of components:")))
        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(1, 50)
        self._n_components_spin.setValue(5)
        n_comp_layout.addWidget(self._n_components_spin)
        pls_layout.addLayout(n_comp_layout)

        pls_layout.addStretch()
        method_layout.addWidget(pls_group)

    def _on_run(self) -> None:
        """Run PLS analysis."""
        try:
            QMessageBox.information(
                self,
                _("Information"),
                _(
                    "PLS analysis requires two blocks of shape variables.\n"
                    "Please ensure GPA analysis has been run with the data loaded."
                ),
            )
        except Exception as e:
            self._logger.error(f"PLS failed: {e}")
            QMessageBox.critical(self, _("Error"), str(e))
