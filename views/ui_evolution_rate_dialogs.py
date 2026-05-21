# =============================================================================
# FILE: views/ui_evolution_rate_dialogs.py
# =============================================================================
"""
Evolution Rate Analysis Dialogs for PaleoAST

Provides dialogs for:
    - Trait evolution model comparison (Random Walk, Directional, OU)
    - Rate estimation and model selection via AIC

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
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.design_system import Typography, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


class EvolutionRateDialog(QDialog):
    """
    Evolution Rate Analysis Dialog.

    Fits and compares trait evolution models:
        - Random Walk (Brownian Motion)
        - Directional (biased random walk)
        - Ornstein-Uhlenbeck (stasis with attraction)
    """

    resultsReady = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.EvolutionRateDialog")
        self._is_dark_theme = False

        self.setWindowTitle(_("Evolutionary Rate Analysis"))
        self.setMinimumSize(600, 550)
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
                           f"font-family: 'Consolas', monospace; font-size: {t.body_sm_size}px; }}")

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

        # Phylogenetic tree input
        tree_group = QGroupBox(_("Phylogenetic Tree (Newick)"))
        tree_layout = QVBoxLayout(tree_group)

        tree_info = QLabel(
            _("Enter a phylogenetic tree in Newick format.\n"
              "Tip: Connect to PCM analysis to import a tree.")
        )
        tree_info.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        tree_layout.addWidget(tree_info)

        self._tree_input = QTextEdit()
        self._tree_input.setMaximumHeight(80)
        self._tree_input.setPlaceholderText(_("(species1:0.5,species2:0.3):0.2;"))
        tree_layout.addWidget(self._tree_input)

        layout.addWidget(tree_group)

        # Trait data input
        trait_group = QGroupBox(_("Trait Values"))
        trait_layout = QVBoxLayout(trait_group)

        trait_info = QLabel(
            _("Enter trait values for each species (one per line, tab-separated).\n"
              "Format: species_name\\tvalue")
        )
        trait_info.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        trait_layout.addWidget(trait_info)

        self._trait_input = QTextEdit()
        self._trait_input.setMaximumHeight(100)
        self._trait_input.setPlaceholderText(_("species1\t2.5\nspecies2\t3.8\n..."))
        trait_layout.addWidget(self._trait_input)

        layout.addWidget(trait_group)

        # Model settings
        model_group = QGroupBox(_("Evolution Models"))
        model_layout = QFormLayout(model_group)

        self._models_combo = QComboBox()
        self._models_combo.addItems([
            _("All models (BM, Directional, OU)"),
            _("Brownian Motion only"),
            _("Directional only"),
            _("Ornstein-Uhlenbeck only"),
        ])
        model_layout.addRow(_("Models to fit:"), self._models_combo)

        self._aic_weight_check = QComboBox()
        self._aic_weight_check.addItems([
            _("No (compare AIC directly)"),
            _("Yes (compute AICc weights)"),
        ])
        model_layout.addRow(_("Compute AICc weights:"), self._aic_weight_check)

        layout.addWidget(model_group)

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

    def _parse_tree(self) -> str:
        """Parse tree from text input."""
        text = self._tree_input.toPlainText().strip()
        return text if text else None

    def _parse_traits(self) -> tuple:
        """Parse trait data from text input."""
        text = self._trait_input.toPlainText().strip()
        if not text:
            return None, None

        try:
            import numpy as np
            lines = text.strip().split('\n')
            names = []
            values = []

            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    names.append(parts[0])
                    values.append(float(parts[1]))

            return names, np.array(values)
        except Exception as e:
            self._logger.error(f"Trait parsing failed: {e}")
            return None, None

    def _on_run(self) -> None:
        """Run evolution rate analysis."""
        try:
            names, traits = self._parse_traits()
            if names is None or len(names) < 3:
                QMessageBox.warning(
                    self, _("Input Error"),
                    _("Please enter trait values for at least 3 species.")
                )
                return

            from morphometrics.evolution_rate import EvolutionRateAnalyzer

            # Model selection
            model_map = {
                0: ["random_walk", "directional", "stasis"],
                1: ["random_walk"],
                2: ["directional"],
                3: ["stasis"],
            }
            models = model_map.get(self._models_combo.currentIndex(), ["random_walk", "directional", "stasis"])

            compute_weights = self._aic_weight_check.currentIndex() == 1

            analyzer = EvolutionRateAnalyzer()

            # Note: Phylogenetic tree analysis requires separate implementation
            # For now, use time-series analysis
            tree_newick = self._parse_tree()
            if tree_newick:
                QMessageBox.information(
                    self, _("Information"),
                    _("Phylogenetic tree analysis will be available in a future update.\n"
                      "Running trait time-series analysis instead.")
                )

            # Trait-only analysis (time series)
            result = analyzer.analyze(
                trait_series=traits,
                time_intervals=None,  # Will use unit intervals
                models=models,
            )

            # Display results
            self._results_text.setPlainText(result.summary())
            self.resultsReady.emit(result.to_dict())

            QMessageBox.information(self, _("Results"), _("Analysis complete. Results displayed above."))

        except ImportError as e:
            self._logger.error(f"Missing dependency: {e}")
            QMessageBox.critical(self, _("Error"), _("Required analysis module not available."))
        except Exception as e:
            self._logger.error(f"Evolution rate failed: {e}")
            from views.ui_main_window import format_user_error
            QMessageBox.critical(self, _("Error"), format_user_error(e, "演化速率分析"))