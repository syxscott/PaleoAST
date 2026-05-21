# =============================================================================
# FILE: views/ui_null_model_dialogs.py
# =============================================================================
"""
Null Model Co-occurrence Analysis Dialogs for PaleoAST

Provides dialogs for:
    - C-score analysis (Stone & Roberts 1990)
    - Swap randomization (Gotelli 2000)

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.design_system import Typography, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


class NullModelDialog(QDialog):
    """
    Null Model Co-occurrence Analysis Dialog.

    Tests for non-random patterns in species co-occurrence using
    standardized effect sizes vs. null models.
    """

    resultsReady = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.NullModelDialog")
        self._is_dark_theme = False

        self.setWindowTitle(_("Null Model Co-occurrence Analysis"))
        self.setMinimumSize(650, 600)
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
        self.setStyleSheet(f"QDialog {{ background-color: {c.bg_primary}; color: {c.text_primary}; }}"
                           f"QLabel {{ color: {c.text_primary}; font-size: {t.body_size}px; }}"
                           f"QGroupBox {{ color: {c.text_primary}; font-weight: {t.medium}; "
                           f"border: 1px solid {c.border_light}; border-radius: 4px; }}"
                           f"QTextEdit {{ background-color: {c.bg_primary}; color: {c.text_primary}; "
                           f"border: 1px solid {c.border_light}; border-radius: 4px; "
                           f"font-family: 'Consolas', monospace; font-size: {t.body_sm_size}px; }}"
                           f"QProgressBar {{ border: 1px solid {c.border_light}; "
                           f"border-radius: 4px; text-align: center; }}")

    def _setup_ui(self) -> None:
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        t = Typography()
        title_label = QLabel(self.windowTitle())
        title_font = QFont(t.family_primary, t.h4_size, QFont.Weight.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Data input
        data_group = QGroupBox(_("Species Occurrence Matrix"))
        data_layout = QVBoxLayout(data_group)

        data_info = QLabel(
            _("Enter species occurrence data (presence-absence matrix).\n"
              "Format: rows = species, columns = sites, values = 0/1")
        )
        data_info.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        data_layout.addWidget(data_info)

        self._data_input = QTextEdit()
        self._data_input.setMaximumHeight(120)
        self._data_input.setPlaceholderText(_("species1\t1\t0\t1\t0\nspecies2\t0\t1\t1\t1\n..."))
        data_layout.addWidget(self._data_input)

        layout.addWidget(data_group)

        # Method settings
        method_group = QGroupBox(_("Method Settings"))
        method_layout = QFormLayout(method_group)

        self._metric_combo = QComboBox()
        self._metric_combo.addItems([
            _("C-score (Stone & Roberts)"),
            _("C-score (standardized)"),
        ])
        method_layout.addRow(_("Statistic:"), self._metric_combo)

        self._algorithm_combo = QComboBox()
        self._algorithm_combo.addItems([
            _("Swap (Gotelli)"),
            _("RRS (row-randomized)"),
            _("RCS (column-randomized)"),
        ])
        method_layout.addRow(_("Algorithm:"), self._algorithm_combo)

        self._n_sim_spin = QSpinBox()
        self._n_sim_spin.setRange(100, 10000)
        self._n_sim_spin.setValue(1000)
        self._n_sim_spin.setSingleStep(100)
        method_layout.addRow(_("Simulations:"), self._n_sim_spin)

        self._n_workers_spin = QSpinBox()
        self._n_workers_spin.setRange(1, 8)
        self._n_workers_spin.setValue(4)
        method_layout.addRow(_("Parallel workers:"), self._n_workers_spin)

        layout.addWidget(method_group)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

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

    def _parse_occurrence_data(self) -> tuple:
        """Parse occurrence data from text input."""
        text = self._data_input.toPlainText().strip()
        if not text:
            return None

        try:
            import numpy as np
            lines = text.strip().split('\n')
            matrix = []

            for line in lines:
                parts = line.strip().split('\t')
                row = []
                for p in parts[1:]:  # Skip species name
                    row.append(int(float(p.strip())))
                if row:
                    matrix.append(row)

            return np.array(matrix)
        except Exception as e:
            self._logger.error(f"Data parsing failed: {e}")
            return None

    def _on_run(self) -> None:
        """Run null model analysis."""
        try:
            matrix = self._parse_occurrence_data()
            if matrix is None or matrix.size < 4:
                QMessageBox.warning(
                    self, _("Input Error"),
                    _("Please enter a valid occurrence matrix (at least 2 species x 2 sites).")
                )
                return

            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                QMessageBox.warning(
                    self, _("Input Error"),
                    _("Matrix must have at least 2 species and 2 sites.")
                )
                return

            self._run_button.setEnabled(False)
            self._progress_bar.setValue(0)

            from ecology.null_models import NullModelAnalyzer

            algorithm_map = {
                0: "swap",
                1: "rrs",
                2: "rcs",
            }
            algorithm = algorithm_map.get(self._algorithm_combo.currentIndex(), "swap")

            analyzer = NullModelAnalyzer()

            # Update progress
            def progress_callback(p):
                self._progress_bar.setValue(int(p * 100))

            metric_map = {
                0: "c_score",
                1: "checkerboard",
            }
            metric = metric_map.get(self._metric_combo.currentIndex(), "c_score")

            result = analyzer.analyze(
                presence_matrix=matrix,
                metric=metric,
                n_permutations=self._n_sim_spin.value(),
                algorithm=algorithm,
                n_workers=self._n_workers_spin.value(),
            )

            self._progress_bar.setValue(100)
            self._results_text.setPlainText(result.summary())
            self.resultsReady.emit(result.to_dict())

            QMessageBox.information(
                self, _("Results"),
                _("Analysis complete. Observed vs. expected pattern displayed above.")
            )

        except ImportError as e:
            self._logger.error(f"Missing dependency: {e}")
            QMessageBox.critical(self, _("Error"), _("Required analysis module not available."))
        except Exception as e:
            self._logger.error(f"Null model failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "零模型分析"))
        finally:
            self._run_button.setEnabled(True)