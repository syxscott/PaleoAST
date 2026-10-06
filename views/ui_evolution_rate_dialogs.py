# =============================================================================
# FILE: views/ui_evolution_rate_dialogs.py
# =============================================================================
"""
Evolution Rate Analysis Dialogs for PaleoAST

Provides dialogs for:
    - Trait evolution model comparison (Random Walk, Directional, OU)
    - Rate estimation and model selection via AIC

Note on phylogenetic trees: the underlying
``morphometrics.evolution_rate.EvolutionRateAnalyzer`` only operates on
a 1-D trait series ordered in time (Foote's stratigraphic framework).
It does NOT consume a phylogenetic tree. The dialog therefore does
not collect a Newick string — adding one would silently drop it on the
floor and trick the user into thinking the analysis is phylogenetic.

Author: PaleoAST Development Team
version: 1.2.0
"""

import logging

from PyQt6.QtCore import pyqtSignal
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
)

from config.design_system import Typography, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


class EvolutionRateDialog(QDialog):
    """
    Evolution Rate Analysis Dialog.

    Fits and compares trait evolution models on a stratigraphically (or
    otherwise temporally) ordered trait series. The three models map
    directly to :class:`morphometrics.evolution_rate.EvolutionModel`:

        - ``random_walk`` (Brownian Motion): trait diffuses away from
          its initial value with variance growing linearly in time
          ``Var[x(t)] = sigma^2 * t``.
        - ``directional`` (biased random walk): the series drifts with
          a constant trend ``beta`` plus random walk.
        - ``stasis`` (Ornstein-Uhlenbeck): mean-reverting Ornstein-
          Uhlenbeck process with equilibrium ``theta`` and attraction
          strength ``alpha``.
    """

    resultsReady = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.EvolutionRateDialog")
        self._is_dark_theme = False

        self.setWindowTitle(_("Evolutionary Rate Analysis"))
        self.setMinimumSize(620, 600)
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

        # Notice: this analyzer does NOT take a phylogenetic tree.
        notice = QLabel(
            _(
                "Note: this analyzer fits Foote-style models to a temporally "
                "ordered trait series. Phylogenetic trees are not used here; "
                "use the PCM (PIC / ASR / Blomberg's K) dialogs for tree-based "
                "comparative methods."
            )
        )
        notice.setWordWrap(True)
        notice.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        layout.addWidget(notice)

        # Trait data input
        trait_group = QGroupBox(_("Trait Values"))
        trait_layout = QVBoxLayout(trait_group)

        trait_info = QLabel(
            _(
                "Enter trait values in stratigraphic / temporal order (one per line). "
                "Format: value  (optionally prefixed by  name<tab>value)"
            )
        )
        trait_info.setWordWrap(True)
        trait_info.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        trait_layout.addWidget(trait_info)

        self._trait_input = QTextEdit()
        self._trait_input.setMaximumHeight(120)
        self._trait_input.setPlaceholderText(_("2.5\n3.8\n3.2\n4.1\n..."))
        trait_layout.addWidget(self._trait_input)

        layout.addWidget(trait_group)

        # Model settings
        model_group = QGroupBox(_("Evolution Models"))
        model_layout = QFormLayout(model_group)

        self._models_combo = QComboBox()
        self._models_combo.addItems(
            [
                _("All models (Random walk, Directional, Stasis)"),
                _("Random walk only"),
                _("Directional only"),
                _("Stasis only"),
            ]
        )
        model_layout.addRow(_("Models to fit:"), self._models_combo)

        # AIC weights are always computed by the underlying engine
        # (``EvolutionRateAnalyzer.analyze`` returns AIC weights for
        # every fitted model unconditionally). The old "Yes/No (AICc
        # weights)" combo was misleading — it implied a switch that
        # the engine does not expose. The AIC weights are always shown
        # in the model-comparison table; nothing to toggle here.
        # Keeping the field would invite the user to think their choice
        # matters, so we remove it and rely on the summary text.

        # Numerical settings
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.5, 0.999)
        self._confidence_spin.setSingleStep(0.01)
        self._confidence_spin.setValue(0.95)
        self._confidence_spin.setDecimals(3)
        model_layout.addRow(_("Confidence level for rate CI:"), self._confidence_spin)

        self._bootstrap_spin = QSpinBox()
        self._bootstrap_spin.setRange(0, 9999)
        self._bootstrap_spin.setValue(199)
        self._bootstrap_spin.setSingleStep(50)
        model_layout.addRow(_("Bootstrap replicates (rate CI):"), self._bootstrap_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 10**6)
        self._seed_spin.setValue(42)
        model_layout.addRow(_("Random seed:"), self._seed_spin)

        layout.addWidget(model_group)

        # Results
        results_group = QGroupBox(_("Results"))
        results_layout = QVBoxLayout(results_group)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(180)
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

    def _parse_traits(self):
        """Parse trait data from text input.

        Accepts either ``value`` per line, ``name\\tvalue`` per line, or
        ``name=value`` per line. Returns a 1-D ``numpy.ndarray`` of
        values in the order given, or ``None`` when parsing fails or
        fewer than 3 values are supplied.
        """
        text = self._trait_input.toPlainText().strip()
        if not text:
            return None

        try:
            import numpy as np

            lines = text.split("\n")
            values = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Accept tab-separated or equals-separated "name=value"
                if "\t" in line:
                    value_str = line.split("\t", 1)[1].strip()
                elif "=" in line:
                    value_str = line.split("=", 1)[1].strip()
                else:
                    value_str = line
                values.append(float(value_str))

            arr = np.asarray(values, dtype=float)
            if arr.size < 3:
                return None
            return arr
        except Exception as e:
            self._logger.error(f"Trait parsing failed: {e}")
            return None

    def _on_run(self) -> None:
        """Run evolution rate analysis."""
        try:
            traits = self._parse_traits()
            if traits is None:
                QMessageBox.warning(
                    self,
                    _("Input Error"),
                    _("Please enter trait values for at least 3 specimens in temporal order."),
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

            analyzer = EvolutionRateAnalyzer()
            n_bootstrap = int(self._bootstrap_spin.value())
            result = analyzer.analyze(
                trait_series=traits,
                time_intervals=None,  # Will use unit intervals
                models=models,
                confidence_level=float(self._confidence_spin.value()),
                seed=int(self._seed_spin.value()),
                n_bootstrap=n_bootstrap,
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

            QMessageBox.critical(self, _("Error"), format_user_error(e, _("Evolution Rate Analysis")))
