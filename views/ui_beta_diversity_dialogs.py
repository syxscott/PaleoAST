# =============================================================================
# FILE: views/ui_beta_diversity_dialogs.py
# =============================================================================
"""
Beta Diversity and Coverage-Based Rarefaction Dialogs for PaleoAST

Provides dialogs for:
    - Coverage-based rarefaction (iNEXT-style)
    - Beta diversity decomposition (Baselga 2010)

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
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


class BaseBetaDialog(QDialog):
    """Base dialog for beta diversity analyses."""

    resultsReady = pyqtSignal(dict)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.{title}")
        self._is_dark_theme = False

        self.setWindowTitle(title)
        self.setMinimumSize(600, 500)
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

        # Data input
        data_group = QGroupBox(_("Species Abundance Data"))
        data_layout = QVBoxLayout(data_group)

        data_info = QLabel(
            _(
                "Enter species abundances per site (one row per site, tab or comma separated).\n"
                "Rows: sites/samples, Columns: species, Values: abundance counts."
            )
        )
        data_info.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        data_layout.addWidget(data_info)

        self._data_input = QTextEdit()
        self._data_input.setMaximumHeight(120)
        self._data_input.setPlaceholderText(_("Site1_sp1\tabundance1\nSite1_sp2\tabundance2\n..."))
        data_layout.addWidget(self._data_input)

        layout.addWidget(data_group)

        # Method-specific content
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

    def _parse_abundance_data(self) -> tuple:
        """Parse abundance data from text input."""
        text = self._data_input.toPlainText().strip()
        if not text:
            return None, None

        try:
            import numpy as np

            lines = text.strip().split("\n")
            # Format: each line is "site_name\tspecies1_count\tspecies2_count\t..."
            data_rows = []
            site_names = []

            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    site_names.append(parts[0])
                    row = [float(p) for p in parts[1:]]
                    data_rows.append(row)

            if not data_rows:
                return None, None

            # Pad to same length if needed
            max_len = max(len(row) for row in data_rows)
            for row in data_rows:
                row.extend([0.0] * (max_len - len(row)))

            abundance_matrix = np.array(data_rows)
            return abundance_matrix, site_names
        except Exception as e:
            self._logger.error(f"Data parsing failed: {e}")
            return None, None

    def _on_run(self) -> None:
        """Run the analysis. Subclasses implement specific logic."""
        raise NotImplementedError


class CoverageRarefactionDialog(BaseBetaDialog):
    """
    Coverage-Based Rarefaction Dialog.

    Computes sample-size- and coverage-based rarefaction curves
    using the iNEXT approach (Chao et al. 2014).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Coverage-Based Rarefaction (iNEXT)"), parent)

        method_layout = self._method_widget.layout()

        # Rarefaction options. ``CoverageRarefactionAnalyzer.coverage_rarefaction_hill``
        # accepts ``n_points`` (number of coverage levels to evaluate),
        # ``n_bootstrap`` and ``seed`` for the multinomial bootstrap CI.
        # The bare ``CoverageRarefactionAnalyzer.analyze`` wrapper ignores
        # the bootstrap knob, so this dialog routes through
        # ``coverage_rarefaction_hill`` to honour the user's choices.
        opts_group = QGroupBox(_("Rarefaction Settings"))
        opts_layout = QFormLayout(opts_group)

        self._endpoint_spin = QSpinBox()
        self._endpoint_spin.setRange(10, 1000)
        self._endpoint_spin.setValue(100)
        self._endpoint_spin.setToolTip(
            _("Number of coverage levels to evaluate between 0.10 and 0.99 (n_points).")
        )
        opts_layout.addRow(_("Coverage levels (n_points):"), self._endpoint_spin)

        self._n_boot_spin = QSpinBox()
        self._n_boot_spin.setRange(0, 5000)
        self._n_boot_spin.setValue(200)
        self._n_boot_spin.setToolTip(
            _("Number of multinomial bootstrap replicates for the CI. 0 skips the bootstrap.")
        )
        opts_layout.addRow(_("Bootstrap replicates (0 = skip):"), self._n_boot_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 10**6)
        self._seed_spin.setValue(42)
        opts_layout.addRow(_("Random seed:"), self._seed_spin)

        method_layout.addWidget(opts_group)

    def _on_run(self) -> None:
        """Run coverage-based rarefaction."""
        try:
            abundance_matrix, site_names = self._parse_abundance_data()
            if abundance_matrix is None or abundance_matrix.shape[0] < 2:
                QMessageBox.warning(
                    self, _("Input Error"), _("Please enter valid abundance data for at least 2 sites.")
                )
                return

            from ecology.beta_diversity import CoverageRarefactionAnalyzer

            analyzer = CoverageRarefactionAnalyzer()
            # Route through ``coverage_rarefaction_hill`` so the bootstrap
            # count and seed actually flow into the engine (the bare
            # ``analyze`` method silently ignores them).
            result = analyzer.coverage_rarefaction_hill(
                abundance_matrix=abundance_matrix,
                sample_names=site_names if site_names else None,
                n_points=int(self._endpoint_spin.value()),
                n_bootstrap=int(self._n_boot_spin.value()),
                seed=int(self._seed_spin.value()),
            )

            # Display results
            self._results_text.setPlainText(result.summary())
            self.resultsReady.emit(result.to_dict())

            QMessageBox.information(self, _("Results"), _("Analysis complete. Results displayed above."))

        except ImportError as e:
            self._logger.error(f"Missing dependency: {e}")
            QMessageBox.critical(self, _("Error"), _("Required analysis module not available."))
        except Exception as e:
            self._logger.error(f"Rarefaction failed: {e}")
            from views.ui_main_window import format_user_error

            QMessageBox.critical(self, _("Error"), format_user_error(e, "稀疏化分析"))


class BetaDiversityDialog(BaseBetaDialog):
    """
    Beta Diversity Decomposition Dialog.

    Decomposes beta diversity into turnover and nestedness components
    using Baselga (2010) methodology.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("Beta Diversity Decomposition"), parent)

        method_layout = self._method_widget.layout()

        # Beta diversity options. ``BetaDiversityAnalyzer.decompose_beta_diversity``
        # (ecology/beta_diversity.py) accepts only ``abundance_matrix``,
        # ``sample_names`` and ``metric``; there is no ``transform`` flag
        # or "full pairwise display mode" switch. Both the old
        # ``_transform_combo`` and ``_show_pairwise_check`` were dead
        # controls — surfaced in the UI but ignored by the engine. To
        # avoid silently dropping user input they have been replaced
        # with an info note that the decomposition always runs on the
        # binary (presence/absence) projection of the abundance matrix
        # and always returns both the three full matrices and the
        # per-pair table.

        opts_group = QGroupBox(_("Beta Diversity Settings"))
        opts_layout = QFormLayout(opts_group)

        self._metric_combo = QComboBox()
        self._metric_combo.addItems(
            [
                _("Sørensen index"),
                _("Jaccard index"),
            ]
        )
        opts_layout.addRow(_("Dissimilarity metric:"), self._metric_combo)

        method_layout.addWidget(opts_group)

        info_label = QLabel(
            _(
                "Decomposition always uses presence/absence (the abundance "
                "matrix is thresholded at zero). The full per-pair turnover / "
                "nestedness / total table is always returned alongside the "
                "three matrices — there is no toggle for either."
            )
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {get_palette(self._is_dark_theme).text_secondary}; font-size: 11px;")
        method_layout.addWidget(info_label)

    def _on_run(self) -> None:
        """Run beta diversity decomposition."""
        try:
            abundance_matrix, site_names = self._parse_abundance_data()
            if abundance_matrix is None or abundance_matrix.shape[0] < 2:
                QMessageBox.warning(
                    self, _("Input Error"), _("Please enter valid abundance data for at least 2 sites.")
                )
                return

            from ecology.beta_diversity import BetaDiversityAnalyzer

            metric = "sorensen" if self._metric_combo.currentIndex() == 0 else "jaccard"

            analyzer = BetaDiversityAnalyzer()
            result = analyzer.decompose_beta_diversity(
                abundance_matrix=abundance_matrix,
                sample_names=site_names if site_names else None,
                metric=metric,
            )

            # Display results
            self._results_text.setPlainText(result.summary())
            self.resultsReady.emit(result.to_dict())

            QMessageBox.information(self, _("Results"), _("Analysis complete. Results displayed above."))

        except ImportError as e:
            self._logger.error(f"Missing dependency: {e}")
            QMessageBox.critical(self, _("Error"), _("Required analysis module not available."))
        except Exception as e:
            self._logger.error(f"Beta diversity failed: {e}")
            from views.ui_main_window import format_user_error

            QMessageBox.critical(self, _("Error"), format_user_error(e, "Beta多样性"))
