# =============================================================================
# FILE: views/ui_macroevolution_dialogs.py
# =============================================================================
"""
Macroevolution analysis dialogs for PaleoAST.

Exposes the macroevolution/ engines, which were implemented and covered by
tests but had no entry point in the running application:

    - Foote cohort survivorship (cohort.py)
    - Diversity dynamics        (diversity.py)
    - Kaplan-Meier survival + log-rank (survival.py)
    - Fossilised birth-death simulation (fbd.py)

The engines remain the single source of truth; these dialogs only collect
parameters and dispatch through StatisticsController.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.design_system import Typography, get_palette
from config.i18n import _

logger = logging.getLogger(__name__)


def _themed_label(text: str, bold: bool = False, size: int | None = None) -> QLabel:
    label = QLabel(text)
    if bold or size:
        t = Typography()
        label.setFont(QFont(t.family_primary, size or t.h4_size, QFont.Weight.Bold if bold else QFont.Weight.Normal))
    return label


class MacroevolutionDialog(QDialog):
    """Tabbed dialog covering the macroevolution engines.

    Every tab reads the taxon ranges (or durations) from the columns of the
    matrix currently loaded in the spreadsheet, so no separate data import is
    needed; the column indices are configurable per tab.
    """

    resultsReady = pyqtSignal(str, object)

    def __init__(self, controller: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.MacroevolutionDialog")
        self._controller = controller
        self._is_dark_theme = False

        self.setWindowTitle(_("Macroevolution Analyses"))
        self.setMinimumSize(620, 520)
        self.setModal(True)

        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        c = get_palette(self._is_dark_theme)
        t = Typography()
        self.setStyleSheet(
            f"QDialog {{ background-color: {c.bg_primary}; color: {c.text_primary}; }}"
            f"QLabel {{ color: {c.text_primary}; font-size: {t.body_size}px; }}"
            f"QTabWidget::pane {{ border: 1px solid {c.border_light}; }}"
            f"QTabBar::tab {{ background: {c.bg_secondary}; color: {c.text_primary};"
            f"padding: 6px 12px; border: 1px solid {c.border_light}; }}"
            f"QTabBar::tab:selected {{ background: {c.bg_primary}; color: {c.accent_primary}; }}"
            f"QGroupBox {{ color: {c.text_primary}; font-weight: {t.medium};"
            f"border: 1px solid {c.border_light}; border-radius: 4px; }}"
            f"QPushButton {{ background: {c.bg_secondary}; color: {c.text_primary};"
            f"border: 1px solid {c.border_light}; border-radius: 4px; padding: 6px 14px; }}"
            f"QPushButton:hover {{ background: {c.bg_hover}; }}"
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        t = Typography()
        title = _themed_label(self.windowTitle(), bold=True, size=t.h4_size)
        layout.addWidget(title)
        layout.addWidget(
            QLabel(
                _(
                    "Taxon ranges and durations are read from the columns of the loaded "
                    "matrix. Ages are in Ma (older = larger)."
                )
            )
        )

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_cohort_tab(), _("Cohort Survivorship"))
        self._tabs.addTab(self._build_diversity_tab(), _("Diversity Dynamics"))
        self._tabs.addTab(self._build_survival_tab(), _("Survival Analysis"))
        self._tabs.addTab(self._build_fbd_tab(), _("FBD Simulation"))
        layout.addWidget(self._tabs, 1)

        # The call site uses ``setProperty("tab", N)`` (0=cohort,
        # 1=diversity, 2=survival, 3=FBD) to direct the user to the
        # correct tab. Previously the property was set but never read,
        # so every entry point opened on the cohort tab. ``property()``
        # returns ``None`` when the key is absent, so default to 0.
        requested = self.property("tab")
        if requested is not None:
            try:
                idx = int(requested)
                if 0 <= idx < self._tabs.count():
                    self._tabs.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ tabs
    def _build_cohort_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._cohort_fad = QSpinBox()
        self._cohort_fad.setRange(0, 9999)
        self._cohort_fad.setValue(0)
        self._cohort_fad.setToolTip(_("Column holding each taxon's first appearance date (FAD, Ma)"))
        form.addRow(_("FAD column:"), self._cohort_fad)

        self._cohort_lad = QSpinBox()
        self._cohort_lad.setRange(0, 9999)
        self._cohort_lad.setValue(1)
        self._cohort_lad.setToolTip(_("Column holding each taxon's last appearance date (LAD, Ma)"))
        form.addRow(_("LAD column:"), self._cohort_lad)

        self._cohort_bins = QSpinBox()
        self._cohort_bins.setRange(2, 40)
        self._cohort_bins.setValue(4)
        form.addRow(_("Time intervals:"), self._cohort_bins)

        self._cohort_conf = QDoubleSpinBox()
        self._cohort_conf.setRange(0.5, 0.999)
        self._cohort_conf.setSingleStep(0.01)
        self._cohort_conf.setValue(0.95)
        form.addRow(_("Confidence level:"), self._cohort_conf)

        run = QPushButton(_("Run"))
        run.clicked.connect(self._on_run_cohort)
        form.addRow(run)
        return page

    def _build_diversity_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._div_fad = QSpinBox()
        self._div_fad.setRange(0, 9999)
        form.addRow(_("FAD column:"), self._div_fad)

        self._div_lad = QSpinBox()
        self._div_lad.setRange(0, 9999)
        self._div_lad.setValue(1)
        form.addRow(_("LAD column:"), self._div_lad)

        self._div_bins = QSpinBox()
        self._div_bins.setRange(2, 60)
        self._div_bins.setValue(8)
        form.addRow(_("Time bins:"), self._div_bins)

        run = QPushButton(_("Run"))
        run.clicked.connect(self._on_run_diversity)
        form.addRow(run)
        return page

    def _build_survival_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._surv_time = QSpinBox()
        self._surv_time.setRange(0, 9999)
        form.addRow(_("Duration column:"), self._surv_time)

        self._surv_event = QSpinBox()
        self._surv_event.setRange(0, 9999)
        self._surv_event.setValue(1)
        self._surv_event.setToolTip(_("Column holding 1 for an observed event, 0 for censored"))
        form.addRow(_("Event column:"), self._surv_event)

        self._surv_compare = QCheckBox(_("Compare two groups (log-rank)"))
        self._surv_compare.setChecked(False)
        form.addRow(self._surv_compare)

        self._surv_time_b = QSpinBox()
        self._surv_time_b.setRange(0, 9999)
        self._surv_time_b.setValue(2)
        form.addRow(_("Group B duration column:"), self._surv_time_b)

        self._surv_event_b = QSpinBox()
        self._surv_event_b.setRange(0, 9999)
        self._surv_event_b.setValue(3)
        form.addRow(_("Group B event column:"), self._surv_event_b)

        run = QPushButton(_("Run"))
        run.clicked.connect(self._on_run_survival)
        form.addRow(run)
        return page

    def _build_fbd_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self._fbd_lambda = QDoubleSpinBox()
        self._fbd_lambda.setRange(0.0, 10.0)
        self._fbd_lambda.setSingleStep(0.05)
        self._fbd_lambda.setValue(0.5)
        form.addRow(_("Speciation rate (lambda):"), self._fbd_lambda)

        self._fbd_mu = QDoubleSpinBox()
        self._fbd_mu.setRange(0.0, 10.0)
        self._fbd_mu.setSingleStep(0.05)
        self._fbd_mu.setValue(0.2)
        form.addRow(_("Extinction rate (mu):"), self._fbd_mu)

        self._fbd_psi = QDoubleSpinBox()
        self._fbd_psi.setRange(0.0, 10.0)
        self._fbd_psi.setSingleStep(0.05)
        self._fbd_psi.setValue(0.1)
        form.addRow(_("Fossilisation rate (psi):"), self._fbd_psi)

        self._fbd_duration = QDoubleSpinBox()
        self._fbd_duration.setRange(0.1, 1000.0)
        self._fbd_duration.setSingleStep(1.0)
        self._fbd_duration.setValue(10.0)
        form.addRow(_("Duration:"), self._fbd_duration)

        self._fbd_reps = QSpinBox()
        self._fbd_reps.setRange(1, 200)
        self._fbd_reps.setValue(5)
        form.addRow(_("Replicates:"), self._fbd_reps)

        self._fbd_seed = QSpinBox()
        self._fbd_seed.setRange(0, 10**6)
        self._fbd_seed.setValue(42)
        self._fbd_seed.setToolTip(_("Same seed gives the same simulation"))
        form.addRow(_("Random seed:"), self._fbd_seed)

        run = QPushButton(_("Run"))
        run.clicked.connect(self._on_run_fbd)
        form.addRow(run)
        return page

    # ------------------------------------------------------------------ runs
    def _fail(self, exc: Exception, what: str) -> None:
        self._logger.error("%s failed: %s", what, exc)
        QMessageBox.critical(self, _("Analysis Error"), f"{what}: {exc}")

    def _on_run_cohort(self) -> None:
        try:
            result = self._controller.analyze_cohort_survivorship(
                fad_column=self._cohort_fad.value(),
                lad_column=self._cohort_lad.value(),
                n_intervals=self._cohort_bins.value(),
                confidence_level=self._cohort_conf.value(),
            )
        except Exception as e:
            self._fail(e, _("Cohort Survivorship"))
            return
        self.resultsReady.emit("cohort_survivorship", result)

    def _on_run_diversity(self) -> None:
        try:
            result = self._controller.analyze_diversity_dynamics(
                fad_column=self._div_fad.value(),
                lad_column=self._div_lad.value(),
                n_intervals=self._div_bins.value(),
            )
        except Exception as e:
            self._fail(e, _("Diversity Dynamics"))
            return
        self.resultsReady.emit("diversity_dynamics", result)

    def _on_run_survival(self) -> None:
        try:
            if self._surv_compare.isChecked():
                result = self._controller.compare_survival_groups(
                    time_a=self._surv_time.value(),
                    event_a=self._surv_event.value(),
                    time_b=self._surv_time_b.value(),
                    event_b=self._surv_event_b.value(),
                )
                self.resultsReady.emit("survival_logrank", result)
                return
            result = self._controller.analyze_survival(
                time_column=self._surv_time.value(),
                event_column=self._surv_event.value(),
            )
        except Exception as e:
            self._fail(e, _("Survival Analysis"))
            return
        self.resultsReady.emit("survival", result)

    def _on_run_fbd(self) -> None:
        try:
            result = self._controller.simulate_fbd(
                speciation_rate=self._fbd_lambda.value(),
                extinction_rate=self._fbd_mu.value(),
                fossilization_rate=self._fbd_psi.value(),
                duration=self._fbd_duration.value(),
                n_replicates=self._fbd_reps.value(),
                random_seed=self._fbd_seed.value(),
            )
        except Exception as e:
            self._fail(e, _("FBD Simulation"))
            return
        self.resultsReady.emit("fbd", result)


class Morpho3DDialog(QDialog):
    """Generalized Procrustes analysis of 3-D landmark configurations.

    The configurations come from the loaded matrix read as
    (n_specimens, n_landmarks, 3), which is the same reshape the 2-D GPA path
    uses, or from a TPS file via the drop handler.
    """

    resultsReady = pyqtSignal(object)

    def __init__(self, controller: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.Morpho3DDialog")
        self._controller = controller
        self._is_dark_theme = False

        self.setWindowTitle(_("3-D Morphometrics (GPA)"))
        self.setMinimumSize(520, 400)
        self.setModal(True)
        self._setup_ui()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        c = get_palette(self._is_dark_theme)
        t = Typography()
        self.setStyleSheet(
            f"QDialog {{ background-color: {c.bg_primary}; color: {c.text_primary}; }}"
            f"QLabel {{ color: {c.text_primary}; font-size: {t.body_size}px; }}"
            f"QSpinBox, QDoubleSpinBox, QComboBox {{ background: {c.bg_primary};"
            f"color: {c.text_primary}; border: 1px solid {c.border_light}; border-radius: 4px; }}"
            f"QPushButton {{ background: {c.bg_secondary}; color: {c.text_primary};"
            f"border: 1px solid {c.border_light}; border-radius: 4px; padding: 6px 14px; }}"
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        t = Typography()
        title = QLabel(self.windowTitle())
        title.setFont(QFont(t.family_primary, t.h4_size, QFont.Weight.Bold))
        layout.addWidget(title)

        note = QLabel(
            _(
                "The loaded matrix is reshaped to (specimens x landmarks x 3), so the "
                "number of landmarks per specimen is the total cell count divided by "
                "three times the number of specimens."
            )
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self._n_landmarks = QSpinBox()
        self._n_landmarks.setRange(3, 5000)
        self._n_landmarks.setValue(12)
        self._n_landmarks.setToolTip(_("Landmarks per specimen"))
        form.addRow(_("Landmarks per specimen:"), self._n_landmarks)

        self._use_concensus = QComboBox()
        self._use_concensus.addItems([_("Consensus (recommended)"), _("First specimen")])
        form.addRow(_("Starting consensus from:"), self._use_concensus)

        run = QPushButton(_("Run"))
        run.clicked.connect(self._on_run)
        form.addRow(run)
        layout.addLayout(form)
        layout.addStretch(1)

    def _on_run(self) -> None:
        from models.state_manager import get_state_manager

        matrix = get_state_manager().data_matrix
        if matrix is None:
            QMessageBox.warning(self, _("No Data"), _("Load a landmark matrix first."))
            return
        data = matrix.data
        n_spec = data.shape[0]
        n_land = self._n_landmarks.value()
        if n_spec * n_land * 3 != data.size:
            QMessageBox.warning(
                self,
                _("Shape Mismatch"),
                _(
                    "{0} specimens x {1} landmarks x 3 = {2} cells, but the matrix has {3}. Adjust the landmark count."
                ).format(n_spec, n_land, n_spec * n_land * 3, data.size),
            )
            return

        configs = [data[i].reshape(n_land, 3) for i in range(n_spec)]
        try:
            result = self._controller.analyze_gpa3d(configs)
        except Exception as e:
            self._logger.error("3-D GPA failed: %s", e)
            QMessageBox.critical(self, _("Analysis Error"), str(e))
            return
        self.resultsReady.emit(result)
