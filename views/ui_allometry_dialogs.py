# =============================================================================
# FILE: views/ui_allometry_dialogs.py
# =============================================================================
"""
Allometry and Morphological Integration Dialogs for PaleoAST

Provides dialogs for:
    - Allometry analysis (size-shape regression)
    - Two-Block PLS analysis (morphological integration)

Author: PaleoAST Development Team
version: 1.2.0
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
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


class BaseAllometryDialog(QDialog):
    """
    Base dialog for allometry and integration analyses.

    Provides common UI structure:
        - GPA data selection
        - Results display
    """

    resultsReady = pyqtSignal(dict)

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        controller: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.{title}")
        self._is_dark_theme = False
        # Inherit the controller from the main window so that the
        # dialog shares the same analyzers / state cache. ``None`` is
        # supported for headless tests; ``_on_run`` will fall back to
        # building a private ``StatisticsController`` in that case.
        self._controller = controller

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
        self._results_text.setMaximumHeight(220)
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

    def _refresh_gpa_label(self) -> None:
        """Show whether a GPA result is available in the cached state."""
        controller = self._controller
        if controller is None:
            self._gpa_result_label.setText(_("No GPA result selected"))
            return
        try:
            cached = controller.get_cached_result("gpa_result")
        except Exception as e:  # defensive: state lookup must never raise into the UI
            self._logger.debug("GPA cache lookup failed: %s", e)
            cached = None
        if cached is None:
            self._gpa_result_label.setText(
                _(
                    "No cached GPA result. Run GPA first, or pass aligned "
                    "configurations explicitly when calling the analyzer."
                )
            )
        else:
            n = getattr(cached, "aligned_configurations", None)
            shape = getattr(n, "shape", None)
            self._gpa_result_label.setText(_("Cached GPA result available: shape = {0}").format(shape))

    def showEvent(self, event) -> None:
        """Refresh the GPA label every time the dialog is shown."""
        self._refresh_gpa_label()
        super().showEvent(event)

    def _get_controller(self):
        """Return the controller (inherited or a fresh one for headless tests)."""
        if self._controller is not None:
            return self._controller
        from controllers.statistics_controller import StatisticsController

        return StatisticsController()

    def _on_run(self) -> None:
        """Run the analysis. Subclasses implement specific logic."""
        raise NotImplementedError

    def _show_error(self, exc: Exception, label: str) -> None:
        self._logger.error(f"{label} failed: {exc}")
        try:
            from views.ui_main_window import format_user_error
        except Exception:
            format_user_error = None
        msg = format_user_error(exc, label) if format_user_error is not None else str(exc)
        QMessageBox.critical(self, _("Error"), msg)


class AllometryDialog(BaseAllometryDialog):
    """
    Allometry Analysis Dialog.

    Analyzes relationship between centroid size and shape using
    multivariate regression of Procrustes coordinates on log centroid size.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        controller: object | None = None,
    ) -> None:
        super().__init__(
            _("Allometry Analysis (Size-Shape Relationship)"),
            parent,
            controller=controller,
        )

        method_layout = self._method_widget.layout()

        # Options
        opts_group = QGroupBox(_("Options"))
        opts_layout = QFormLayout(opts_group)

        self._use_pca_check = QCheckBox(_("Reduce dimensionality with PCA"))
        self._use_pca_check.setChecked(False)
        opts_layout.addRow(self._use_pca_check)

        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(2, 100)
        self._n_components_spin.setValue(10)
        self._n_components_spin.setEnabled(False)
        opts_layout.addRow(_("Number of components:"), self._n_components_spin)

        # The "Regression method" combo offered OLS and RMA and was wired to
        # NOTHING: it was constructed, filled and added to the layout, and no
        # code path ever read ``_method_combo.currentIndex()``. Picking RMA ran
        # the OLS regression and labelled the result RMA. That is worse than
        # offering nothing, because the numbers come out looking right.
        #
        # It is removed rather than made to work, for now: RMA needs a real
        # implementation in morphometrics/allometry.py, which is a separate
        # piece of work with its own assumptions to state (it fits both axes'
        # error, so it needs measurement error estimates that the dialog does
        # not collect). The confidence-level spinner below was removed earlier
        # for exactly the same reason -- a control that cannot affect the
        # result should not be on screen. When RMA lands it comes back, wired.

        # NOTE: the previous "Confidence level" spinner was a dead
        # control — ``AllometryAnalyzer.analyze_allometry`` and
        # ``StatisticsController.analyze_allometry`` both ignore a
        # ``confidence_level`` argument, so any value typed here was
        # silently dropped. The engine does not currently expose a
        # bootstrap CI for the regression coefficients. The control is
        # not re-added because that would invite the same silent
        # failure; see the engine for a future CI implementation.

        self._use_pca_check.toggled.connect(self._n_components_spin.setEnabled)

        method_layout.addWidget(opts_group)

    def _on_run(self) -> None:
        """Run allometry analysis."""
        try:
            controller = self._get_controller()
            n_components = self._n_components_spin.value() if self._use_pca_check.isChecked() else None
            result = controller.analyze_allometry(n_components=n_components)
            self._results_text.setPlainText(result.summary())
            self.resultsReady.emit(result.to_dict())
        except Exception as e:
            self._show_error(e, _("Allometry analysis"))


class PLSDialog(BaseAllometryDialog):
    """
    Two-Block Partial Least Squares (Integration) Dialog.

    Measures morphological integration between two blocks of shape variables
    using 2B-PLS analysis.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        controller: object | None = None,
    ) -> None:
        super().__init__(
            _("Morphological Integration (2B-PLS)"),
            parent,
            controller=controller,
        )

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

        method_layout.addWidget(division_group)

        # PLS options
        pls_group = QGroupBox(_("PLS Options"))
        pls_layout = QFormLayout(pls_group)

        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(1, 50)
        self._n_components_spin.setValue(5)
        pls_layout.addRow(_("Number of components:"), self._n_components_spin)

        self._permutations_spin = QSpinBox()
        self._permutations_spin.setRange(0, 9999)
        self._permutations_spin.setValue(999)
        self._permutations_spin.setSingleStep(100)
        pls_layout.addRow(_("Permutations (r₁ test):"), self._permutations_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 10**6)
        self._seed_spin.setValue(42)
        pls_layout.addRow(_("Random seed:"), self._seed_spin)

        method_layout.addWidget(pls_group)

    def _on_run(self) -> None:
        """Run PLS analysis."""
        try:
            controller = self._get_controller()
            division_map = {
                0: "anterior_posterior",
                1: "size_matched",
                2: "random",
            }
            division = division_map.get(self._division_combo.currentIndex(), "anterior_posterior")
            seed = self._seed_spin.value() if self._permutations_spin.value() > 0 else None
            result = controller.analyze_pls(
                division=division,
                n_components=self._n_components_spin.value(),
                permutations=self._permutations_spin.value(),
                seed=seed,
            )
            self._results_text.setPlainText(result.summary())
            self.resultsReady.emit(result.to_dict())
        except Exception as e:
            self._show_error(e, _("PLS analysis"))
