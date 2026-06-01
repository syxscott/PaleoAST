# =============================================================================
# FILE: views/ui_extinction_dialogs.py
# =============================================================================
"""
Extinction Confidence Interval Dialogs for PaleoAST

Provides dialogs for:
    - Extinction interval analysis (Marshall & Strauss-Sadler methods)

Author: PaleoAST Development Team
version: 1.0.1
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
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.design_system import Typography, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


class ExtinctionIntervalDialog(QDialog):
    """
    Dialog for extinction confidence interval analysis.

    Computes 95% confidence intervals for true extinction time
    based on observed LADs (Last Appearance Dates).
    """

    resultsReady = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.ExtinctionIntervalDialog")
        self._is_dark_theme = False

        self.setWindowTitle(_("Extinction Confidence Intervals"))
        self.setMinimumSize(500, 400)
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
                           f"QListWidget {{ background-color: {c.bg_primary}; color: {c.text_primary}; "
                           f"border: 1px solid {c.border_light}; border-radius: 4px; }}")

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

        # Method selection
        method_group = QGroupBox(_("Method"))
        method_layout = QVBoxLayout(method_group)

        method_layout.addWidget(QLabel(_("Select confidence interval method:")))

        self._method_combo = QComboBox()
        self._method_combo.addItems([
            _("Marshall (1990) - Poisson model"),
            _("Strauss-Sadler (1989) - Order statistics"),
        ])
        method_layout.addWidget(self._method_combo)

        layout.addWidget(method_group)

        # Parameters
        param_group = QGroupBox(_("Parameters"))
        param_layout = QFormLayout(param_group)

        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.80, 0.99)
        self._confidence_spin.setValue(0.95)
        self._confidence_spin.setSingleStep(0.01)
        self._confidence_spin.setPrefix(_("Confidence level: "))
        param_layout.addRow(_("Confidence Level:"), self._confidence_spin)

        self._detection_spin = QDoubleSpinBox()
        self._detection_spin.setRange(0.01, 1.00)
        self._detection_spin.setValue(0.70)
        self._detection_spin.setSingleStep(0.05)
        self._detection_spin.setPrefix(_("Detection probability: "))
        param_layout.addRow(_("Detection Probability:"), self._detection_spin)

        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.01, 100.0)
        self._interval_spin.setValue(1.0)
        self._interval_spin.setSingleStep(0.1)
        self._interval_spin.setPrefix(_("Sampling interval (m): "))
        param_layout.addRow(_("Sampling Interval:"), self._interval_spin)

        layout.addWidget(param_group)

        # Info
        info_label = QLabel(
            _("Enter LAD positions in the list below (one per line).\n"
              "LAD positions are layer/horizon numbers from top (higher = older).")
        )
        info_label.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        layout.addWidget(info_label)

        # LAD input
        self._lad_list = QListWidget()
        self._lad_list.setMaximumHeight(100)
        self._lad_list.setToolTip(_("Enter one LAD position per line"))
        layout.addWidget(self._lad_list)

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

    def _on_run(self) -> None:
        """Run extinction interval analysis."""
        try:
            # Parse LAD positions
            lad_text = self._lad_list.toPlainText().strip()
            if not lad_text:
                QMessageBox.warning(self, _("Input Error"), _("Please enter LAD positions."))
                return

            lad_positions = []
            for line in lad_text.split('\n'):
                line = line.strip()
                if line:
                    try:
                        lad_positions.append(float(line))
                    except ValueError:
                        pass

            if len(lad_positions) < 2:
                QMessageBox.warning(self, _("Input Error"), _("Need at least 2 LAD positions."))
                return

            import numpy as np
            from stratigraphy.extinction import ExtinctionIntervalAnalyzer

            analyzer = ExtinctionIntervalAnalyzer()
            method = "marshall" if self._method_combo.currentIndex() == 0 else "strauss_sadler"

            result = analyzer.analyze(
                lad_positions=np.array(lad_positions),
                sampling_interval=self._interval_spin.value(),
                detection_probability=self._detection_spin.value(),
                confidence_level=self._confidence_spin.value(),
                method=method,
            )

            # Show results
            QMessageBox.information(self, _("Results"), result.summary())

            # Emit signal for potential plotting
            self.resultsReady.emit(result.to_dict())

        except Exception as e:
            self._logger.error(f"Extinction analysis failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "灭绝置信区间"))
