# =============================================================================
# FILE: views/ui_permutation_dialogs.py
# =============================================================================
"""
Permutation-test configuration dialogs.

Provides a single, reusable :class:`PermutationTestDialog` for both
ANOSIM and PERMANOVA.  Before this module existed the two handlers in
``views.ui_main_window`` were called with an empty ``params`` dict, so
neither distance metric nor permutation count was configurable and the
only available value was the ``9999`` default.

The dialog deliberately keeps the surface small so we don't fragment
the contract between the four permutations tests the project supports
(ANOSIM, PERMANOVA, SIMPER uses the same options, and the panel may be
re-used for future additions):

    * distance metric (ecology default: Bray-Curtis)
    * permutation count (with a "may be slow" notice when large)
    * random seed (blank = use ``None`` and emit a ``RuntimeWarning``
      to match the convention in ``stats/``)

A short statistical note is included so the user knows what the test
does, not just how to set it up.
"""
from __future__ import annotations

import warnings

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.design_system import BorderRadius, get_palette
from config.i18n import _

# Metric choices exposed in the dropdown.  The display strings map to
# scipy / ``stats.distance_metrics`` keyword arguments via
# ``str.lower().replace('-', '_')`` so ``Bray-Curtis`` -> ``bray_curtis``.
_METRICS: tuple[tuple[str, str], ...] = (
    ("Bray-Curtis", "bray_curtis"),
    ("Euclidean", "euclidean"),
    ("Jaccard", "jaccard"),
    ("Manhattan", "manhattan"),
)


class PermutationTestDialog(QDialog):
    """Configuration dialog for ANOSIM / PERMANOVA permutation tests.

    Emits a ``RuntimeWarning`` when the user leaves the seed blank, so
    downstream ``stats/`` code can keep its "no seed ⇒ non-reproducible"
    contract without needing to inspect the dialog.

    Signals:
        parametersChanged(dict): emitted whenever the user changes any
            parameter.  The main window does not rely on this, but it
            keeps the surface identical to ::class:BaseAnalysisDialog.
    """

    parametersChanged = pyqtSignal(dict)

    def __init__(self, parent=None, *, title: str, default_method_label: str = "ANOSIM") -> None:
        super().__init__(parent)
        self._default_method_label = default_method_label
        self._setup_ui(title)
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self, title: str) -> None:
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # --- Explanatory label --------------------------------------------------
        info = QLabel(
            _(
                "{0} tests whether groups differ in multivariate space by "
                "comparing within-group vs between-group (dis)similarity "
                "across many random label permutations.\n\n"
                "Bray-Curtis is the default for community / abundance data. "
                "More permutations give more precise p-values at the price of "
                "wall-time."
            ).format(self._default_method_label)
        )
        info.setWordWrap(True)
        root.addWidget(info)

        # --- Form layout --------------------------------------------------------
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # Distance metric
        self._metric_combo = QComboBox()
        for label, _value in _METRICS:
            self._metric_combo.addItem(label)
        self._metric_combo.setCurrentText("Bray-Curtis")
        self._metric_combo.setToolTip(
            _("Distance metric used to build the sample-sample dissimilarity matrix.")
        )
        form.addRow(_("Distance metric:"), self._metric_combo)

        # Permutation count
        n_perm_layout = QHBoxLayout()
        n_perm_layout.setContentsMargins(0, 0, 0, 0)
        self._n_perm_spin = QSpinBox()
        self._n_perm_spin.setRange(99, 99999)
        self._n_perm_spin.setSingleStep(100)
        self._n_perm_spin.setValue(9999)
        self._n_perm_spin.setToolTip(
            _("Number of random permutations used to estimate the p-value.")
        )
        n_perm_layout.addWidget(self._n_perm_spin)
        self._slow_warning_label = QLabel()
        self._slow_warning_label.setStyleSheet("color: #c08020;")
        self._n_perm_spin.valueChanged.connect(self._update_slow_warning)
        n_perm_layout.addWidget(self._slow_warning_label, 1)
        form.addRow(_("Permutations:"), self._make_form_row(n_perm_layout))

        # Random seed
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(0)
        # 0 in the UI stands in for "blank / unset" because QSpinBox has no
        # dedicated "blank" sentinel.  get_parameters() translates it back
        # to ``None`` so downstream code receives a real "no seed" value.
        self._seed_spin.setSpecialValueText(_("(empty — random)"))
        self._seed_spin.setToolTip(
            _("Reproducibility seed. Leave empty for a fresh permutation draw.")
        )
        form.addRow(_("Random seed:"), self._seed_spin)

        root.addLayout(form)

        # --- Bottom row ---------------------------------------------------------
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 8, 0, 0)
        self._seed_warning_check = QCheckBox(_("Warn me when no seed is set"))
        self._seed_warning_check.setChecked(True)
        self._seed_warning_check.setToolTip(
            _(
                "If checked, a RuntimeWarning is raised when the dialog is "
                "accepted with the seed blank — matches the convention in "
                "the stats layer (results are not reproducible)."
            )
        )
        bottom.addWidget(self._seed_warning_check)
        bottom.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        root.addLayout(bottom)

        # Initialise the "slow" notice.
        self._update_slow_warning(self._n_perm_spin.value())

    def _make_form_row(self, inner: QHBoxLayout):
        from PyQt6.QtWidgets import QWidget

        wrapper = QWidget()
        wrapper.setLayout(inner)
        return wrapper

    def _update_slow_warning(self, value: int) -> None:
        if value >= 9999:
            self._slow_warning_label.setText(_("(may take a while)"))
        else:
            self._slow_warning_label.setText("")

    def _apply_stylesheet(self) -> None:
        palette = get_palette(getattr(self, "_is_dark_theme", False))
        radii = BorderRadius()
        # ``border_default`` is not in the palette (which only carries a small
        # set of brand tokens); use the muted text colour as a subtle outline
        # that matches the disabled-label contrast.
        outline = getattr(palette, "text_disabled", "#94A3B8")
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {palette.bg_primary};
                color: {palette.text_primary};
            }}
            QLabel {{
                color: {palette.text_primary};
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {palette.bg_secondary};
                color: {palette.text_primary};
                border: 1px solid {outline};
                border-radius: {radii.sm}px;
                padding: 4px 6px;
            }}
            QPushButton {{
                background-color: {palette.bg_secondary};
                color: {palette.text_primary};
                border: 1px solid {outline};
                border-radius: {radii.sm}px;
                padding: 4px 12px;
            }}
            """
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setDarkTheme(self, is_dark: bool) -> None:
        self._is_dark_theme = bool(is_dark)
        self._apply_stylesheet()

    def get_parameters(self) -> dict:
        """Return the user-selected parameters.

        Contract (consumed by ``_on_run_anosim`` / ``_on_run_permanova``):

            ``metric``: scipy metric name (``bray_curtis``, …).
            ``n_permutations``: integer >= 99.
            ``random_seed``: integer, or ``None`` if the user left the
                spinbox at the "empty" sentinel.
        """
        metric_label = self._metric_combo.currentText()
        metric = metric_label.lower().replace("-", "_")
        n_permutations = int(self._n_perm_spin.value())
        seed_value = int(self._seed_spin.value())
        random_seed: int | None = seed_value if seed_value != 0 else None

        params = {
            "metric": metric,
            "n_permutations": n_permutations,
            "random_seed": random_seed,
        }
        self._parameters = params
        return params

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        params = self.get_parameters()
        self.parametersChanged.emit(params)
        if self._seed_warning_check.isChecked() and params["random_seed"] is None:
            # Match the stats-layer convention: no seed ⇒ non-reproducible.
            warnings.warn(
                "PermutationTestDialog: random_seed is None — permutation "
                "results are not reproducible.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.accept()


__all__ = ["PermutationTestDialog", "PreferencesDialog"]


class PreferencesDialog(QDialog):
    """User preferences dialog.

    Lets the user change:

        * interface language code (takes effect on restart -- the
          translator is owned by another agent)
        * CSV loading defaults (whether the CSV has a header row and a
          row-label column)
        * matplotlib figure DPI / figsize used by the interactive plot
          canvas
        * Rscript location, ggplot2 theme, base font size, output format and
          render timeout, for the editable-R export

    The dialog is intentionally simple: only options that have a
    documentable effect, persisted via ``QSettings`` under
    ``PaleoAST/PaleoAST``.
    """

    # These map 1:1 onto ggplot2's theme_*() / ggsave() arguments, so the
    # choice is written verbatim into the generated script.
    R_THEMES = ("classic", "bw", "minimal", "grey")
    R_OUTPUT_FORMATS = ("pdf", "svg", "png")
    # Named in the order a figure is judged: colour-vision safe first, then
    # print-safe, then a magnitude ramp. The label carries the citation so the
    # manuscript can name the source.
    R_PALETTES = (
        ("Okabe-Ito (colour-blind safe)", "okabeito"),
        ("Greyscale (photocopy safe)", "greyscale"),
        ("Dark2 (ColorBrewer)", "dark2"),
        ("Viridis (for a magnitude)", "viridis"),
    )

    def __init__(self, parent=None, *, current: dict | None = None) -> None:
        super().__init__(parent)
        current = current or {}
        self._current = {
            "language": current.get("language", "en"),
            "csv_has_header": bool(current.get("csv_has_header", True)),
            "csv_has_row_labels": bool(current.get("csv_has_row_labels", True)),
            "plot_dpi": int(current.get("plot_dpi", 100)),
            "plot_figsize": str(current.get("plot_figsize", "8,6")),
            "r_rscript": str(current.get("r_rscript", "")),
            "r_theme": str(current.get("r_theme", "classic")),
            "r_base_size": float(current.get("r_base_size", 9)),
            "r_output_format": str(current.get("r_output_format", "pdf")),
            "r_palette": str(current.get("r_palette", "okabeito")),
            "r_timeout": int(current.get("r_timeout", 300)),
        }
        if self._current["r_theme"] not in self.R_THEMES:
            self._current["r_theme"] = "classic"
        if self._current["r_output_format"] not in self.R_OUTPUT_FORMATS:
            self._current["r_output_format"] = "pdf"
        if self._current["r_palette"] not in [v for _, v in self.R_PALETTES]:
            self._current["r_palette"] = "okabeito"
        self._setup_ui()
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        from PyQt6.QtWidgets import (
            QDialogButtonBox,
            QFormLayout,
            QGroupBox,
            QLineEdit,
        )

        self.setWindowTitle(_("Preferences"))
        self.setModal(True)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        # The groups below size to their contents. Without a stretch before the
        # button row, a dialog taller than its content would hand the extra
        # height to the group boxes equally, so each one showed a band of
        # empty space under its last row and the sections read as unrelated
        # boxes. The stretch is added at the bottom instead.

        # --- Language -----------------------------------------------------
        lang_group = QGroupBox(_("Interface"))
        lang_form = QFormLayout(lang_group)
        self._language_edit = QLineEdit(self._current["language"])
        self._language_edit.setPlaceholderText("en")
        self._language_edit.setToolTip(
            _("Language code (e.g. ``en``, ``zh``).  Takes effect on restart.")
        )
        lang_form.addRow(_("Language:"), self._language_edit)
        root.addWidget(lang_group)

        # --- CSV defaults -------------------------------------------------
        csv_group = QGroupBox(_("CSV Loading Defaults"))
        from PyQt6.QtWidgets import QCheckBox

        csv_layout = QVBoxLayout(csv_group)
        self._csv_header_check = QCheckBox(_("CSV files have a header row"))
        self._csv_header_check.setChecked(self._current["csv_has_header"])
        csv_layout.addWidget(self._csv_header_check)
        self._csv_rowlabels_check = QCheckBox(_("First CSV column contains row labels"))
        self._csv_rowlabels_check.setChecked(self._current["csv_has_row_labels"])
        csv_layout.addWidget(self._csv_rowlabels_check)
        root.addWidget(csv_group)

        # --- Plot defaults ------------------------------------------------
        plot_group = QGroupBox(_("Plot Defaults"))
        plot_form = QFormLayout(plot_group)
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(50, 600)
        self._dpi_spin.setValue(self._current["plot_dpi"])
        plot_form.addRow(_("Figure DPI:"), self._dpi_spin)
        self._figsize_edit = QLineEdit(self._current["plot_figsize"])
        self._figsize_edit.setPlaceholderText("width,height (inches)")
        self._figsize_edit.setToolTip(
            _("Two comma-separated numbers, e.g. ``8,6`` for 8 inches wide.")
        )
        plot_form.addRow(_("Figure size:"), self._figsize_edit)
        root.addWidget(plot_group)

        # --- R plotting ---------------------------------------------------
        # These feed RPlotSpec in visualization/r_export.py. The theme and the
        # base size end up on the single THEME line of the generated script,
        # which is the one line a user edits to match a journal.
        r_group = QGroupBox(_("R Plotting (exportable ggplot2 script)"))
        r_form = QFormLayout(r_group)

        path_row = QWidget(r_group)
        path_layout = QVBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self._r_rscript_edit = QLineEdit(self._current["r_rscript"], path_row)
        self._r_rscript_edit.setPlaceholderText(_("auto-detect"))
        self._r_rscript_edit.setToolTip(
            _("Leave empty to find Rscript automatically. Otherwise give the full\n"
              "path, e.g. D:\\Program Files\\R\\R-4.5.2\\bin\\x64\\Rscript.exe")
        )
        path_layout.addWidget(self._r_rscript_edit)
        detect_btn = QPushButton(_("Detect R"), path_row)
        detect_btn.clicked.connect(self._on_detect_r)
        path_layout.addWidget(detect_btn)
        r_form.addRow(_("Rscript:"), path_row)

        self._r_detected = QLabel("", r_group)
        self._r_detected.setWordWrap(True)
        self._r_detected.setStyleSheet("color: #64748B; font-size: 11px;")
        r_form.addRow("", self._r_detected)
        self._refresh_r_status()

        self._r_theme_combo = QComboBox(r_group)
        self._r_theme_combo.addItems(list(self.R_THEMES))
        self._r_theme_combo.setCurrentText(self._current["r_theme"])
        self._r_theme_combo.setToolTip(
            _("ggplot2 theme written into the script as one THEME line.")
        )
        r_form.addRow(_("Theme:"), self._r_theme_combo)

        self._r_size_spin = QDoubleSpinBox(r_group)
        self._r_size_spin.setRange(4.0, 24.0)
        # Whole points. ggplot2's base_size is an integer in practice, and the
        # generated script writes it with ``:g`` anyway -- so showing "11.00"
        # here while the script says ``base_size = 11`` told the user two
        # different numbers for the same setting. The half-point step was a
        # precision the format spec does not carry.
        self._r_size_spin.setDecimals(0)
        self._r_size_spin.setSingleStep(1)
        self._r_size_spin.setValue(self._current["r_base_size"])
        r_form.addRow(_("Base font size:"), self._r_size_spin)

        self._r_format_combo = QComboBox(r_group)
        self._r_format_combo.addItems(list(self.R_OUTPUT_FORMATS))
        self._r_format_combo.setCurrentText(self._current["r_output_format"])
        r_form.addRow(_("Output format:"), self._r_format_combo)

        # Placed next to the theme rather than buried with the file paths:
        # this is the one setting that decides whether the figure reads for a
        # reader with colour vision deficiency, or in a photocopied journal.
        self._r_palette_combo = QComboBox(r_group)
        for label, value in self.R_PALETTES:
            self._r_palette_combo.addItem(_(label), value)
        idx = self._r_palette_combo.findData(self._current["r_palette"])
        self._r_palette_combo.setCurrentIndex(max(0, idx))
        self._r_palette_combo.setToolTip(
            _("Colour set for the groups in the figure. Okabe-Ito stays "
              "separable for colour-blind readers and in greyscale; "
              "viridis suits a continuous magnitude rather than a class.")
        )
        r_form.addRow(_("Group colours:"), self._r_palette_combo)

        self._r_timeout_spin = QSpinBox(r_group)
        self._r_timeout_spin.setRange(10, 3600)
        self._r_timeout_spin.setSingleStep(30)
        self._r_timeout_spin.setSuffix(_(" s"))
        self._r_timeout_spin.setValue(self._current["r_timeout"])
        self._r_timeout_spin.setToolTip(
            _("How long to wait for R before giving up. A huge grid can be slow.")
        )
        r_form.addRow(_("Timeout:"), self._r_timeout_spin)
        root.addWidget(r_group)
        # Push the action buttons to the bottom while the groups above stay
        # their natural height. A stretch is the right tool; setting
        # AlignTop on the layout would top-align the button row too and leave
        # the empty space *below* the buttons, which is worse than the
        # original problem of stretched group boxes.
        root.addStretch(1)
        # --- Bottom buttons -----------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_detect_r(self) -> None:
        """Fill the Rscript field with whatever auto-detection finds."""
        from visualization.r_render import find_rscript

        exe = find_rscript()
        if exe is None:
            self._r_rscript_edit.setText("")
        else:
            self._r_rscript_edit.setText(str(exe))
        self._refresh_r_status()

    def _refresh_r_status(self) -> None:
        """Tell the user immediately whether R is usable, instead of at run time."""
        from visualization.r_render import find_rscript

        exe = find_rscript(self._r_rscript_edit.text().strip())
        if exe is None:
            self._r_detected.setText(
                _("Rscript not found. Install R, or set its path above.")
            )
        else:
            self._r_detected.setText(_("Found: {0}").format(exe))

    def _apply_stylesheet(self) -> None:
        palette = get_palette(getattr(self, "_is_dark_theme", False))
        radii = BorderRadius()
        outline = getattr(palette, "text_disabled", "#94A3B8")
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {palette.bg_primary};
                color: {palette.text_primary};
            }}
            QLabel {{
                color: {palette.text_primary};
            }}
            QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox {{
                background-color: {palette.bg_secondary};
                color: {palette.text_primary};
                border: 1px solid {outline};
                border-radius: {radii.sm}px;
                padding: 4px 6px;
            }}
            QPushButton {{
                background-color: {palette.bg_secondary};
                color: {palette.text_primary};
                border: 1px solid {outline};
                border-radius: {radii.sm}px;
                padding: 4px 8px;
            }}
            QGroupBox {{
                border: 1px solid {outline};
                border-radius: {radii.sm}px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            """
        )

    def setDarkTheme(self, is_dark: bool) -> None:
        self._is_dark_theme = bool(is_dark)
        self._apply_stylesheet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_preferences(self) -> dict:
        """Return the user-selected preferences."""
        lang = (self._language_edit.text() or "en").strip()
        if not lang:
            lang = "en"
        return {
            "language": lang,
            "csv_has_header": self._csv_header_check.isChecked(),
            "csv_has_row_labels": self._csv_rowlabels_check.isChecked(),
            "plot_dpi": int(self._dpi_spin.value()),
            "plot_figsize": (self._figsize_edit.text() or "8,6").strip(),
            "r_rscript": self._r_rscript_edit.text().strip(),
            "r_theme": self._r_theme_combo.currentText(),
            "r_base_size": float(self._r_size_spin.value()),
            "r_output_format": self._r_format_combo.currentText(),
            "r_palette": str(self._r_palette_combo.currentData()),
            "r_timeout": int(self._r_timeout_spin.value()),
        }
