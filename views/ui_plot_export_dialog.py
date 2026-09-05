"""PyQt dialog for exporting a plot with customisable parameters.

Provides a small two-section form:

* Top: a file path selector + format combo (SVG / PDF / PNG / JPG).
* Middle: format-aware options (DPI for raster, quality for JPG,
  transparent / background / tight bbox / colour mode / size).

Presets in the bottom row let the user jump to common
configurations (Publication / Vector / Screen) without manually
tweaking every field.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.i18n import _
from plot_export import (
    ColorMode,
    ExportFormat,
    PlotExportOptions,
    apply_preset,
    parse_options_from_path,
)

logger = logging.getLogger(__name__)


_VECTOR_FORMATS = {"svg", "pdf"}
_RASTER_FORMATS = {"png", "jpg"}


class PlotExportDialog(QDialog):
    """Dialog that lets the user customise a plot export."""

    FORMAT_LABELS: dict[str, str] = {
        "svg": "SVG (vector)",
        "pdf": "PDF (vector)",
        "png": "PNG (raster)",
        "jpg": "JPEG (raster)",
    }

    PRESET_LABELS: dict[str, str] = {
        "publication": _("Publication (PDF/SVG, 600 dpi)"),
        "vector": _("Vector (SVG/PDF)"),
        "screen": _("Screen (PNG, 150 dpi)"),
    }

    # Labels that we re-read on language change. Filled in
    # ``_build_ui`` / ``retranslate``. We keep references to every
    # widget whose visible text depends on the active language so we
    # can update them in place.
    _LABELED_WIDGETS: list[tuple[str, str]] = []  # populated below

    def __init__(self, default_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.PlotExportDialog")
        self.setMinimumWidth(440)

        # The translation key for the window title is captured now so
        # it can be looked up again on ``retranslate``.
        self._window_title_key = "Export Plot"

        self._build_ui(Path(default_path))
        self._wire_signals()

        # Subscribe to runtime language change so the dialog refreshes
        # its visible text without requiring a rebuild. The translator
        # singleton may have been created before any QApplication was
        # instantiated (the common case in unit tests), in which case
        # its QObject base is not initialised and accessing the signal
        # raises ``RuntimeError``. We treat that as a soft failure:
        # the caller can still drive :meth:`retranslate` manually.
        try:
            from config.i18n import get_translator

            get_translator().language_changed.connect(self.retranslate)
        except (RuntimeError, AttributeError):
            pass
        self.retranslate()

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def retranslate(self, *_args) -> None:
        """Refresh all visible text to match the current language.

        Called automatically on language change; can also be invoked
        manually after the dialog is first constructed.
        """
        from config.i18n import get_translator

        # ``windowTitle`` and group-box titles accept ``str`` directly.
        self.setWindowTitle(_(self._window_title_key))
        if hasattr(self, "_file_group"):
            self._file_group.setTitle(_("File"))
        if hasattr(self, "_path_label"):
            self._path_label.setText(_("Path:"))
        if hasattr(self, "_format_label_widget"):
            self._format_label_widget.setText(_("Format:"))
        if hasattr(self, "_browse_btn"):
            self._browse_btn.setText(_("Browse..."))
        if hasattr(self, "_options_group"):
            self._options_group.setTitle(_("Options"))
        for attr, key in (
            ("_dpi_label_widget", "Resolution (DPI):"),
            ("_width_label_widget", "Width (in):"),
            ("_height_label_widget", "Height (in):"),
            ("_quality_label_widget", "JPEG quality:"),
            ("_bg_label_widget", "Background:"),
            ("_color_mode_label_widget", "Color mode:"),
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setText(_(key))
        if hasattr(self, "_width_spin"):
            self._width_spin.setSpecialValueText(_("auto"))
        if hasattr(self, "_height_spin"):
            self._height_spin.setSpecialValueText(_("auto"))
        if hasattr(self, "_transparent_checkbox"):
            self._transparent_checkbox.setText(_("Transparent background"))
        if hasattr(self, "_tight_checkbox"):
            self._tight_checkbox.setText(_("Crop margins (tight)"))
        if hasattr(self, "_embed_fonts_checkbox"):
            self._embed_fonts_checkbox.setText(_("Embed fonts (vector)"))
        if hasattr(self, "_presets_group"):
            self._presets_group.setTitle(_("Presets"))
        for preset_id, btn in getattr(self, "_preset_buttons", {}).items():
            btn.setText(self._preset_button_label(preset_id))
        # The format combo's *items* hold the display text — refresh
        # them so the dropdown reflects the new language.
        if hasattr(self, "_format_combo"):
            current_fmt = self._format_combo.currentData() or "png"
            for i, fmt in enumerate(("svg", "pdf", "png", "jpg")):
                self._format_combo.setItemText(i, self._format_label_text(fmt))
            # Restore the user's selection since setItemText does not
            # change currentIndex but may change currentText.
            idx = next(
                (
                    j
                    for j in range(self._format_combo.count())
                    if self._format_combo.itemData(j) == current_fmt
                ),
                0,
            )
            self._format_combo.setCurrentIndex(idx)
        # Background + color mode combos similarly.
        if hasattr(self, "_bg_combo"):
            current_bg = self._bg_value(self._bg_combo.currentText())
            self._bg_combo.blockSignals(True)
            self._bg_combo.clear()
            self._bg_combo.addItems(
                [self._bg_label(b) for b in ("white", "transparent", "theme")]
            )
            self._bg_combo.setCurrentText(self._bg_label(current_bg))
            self._bg_combo.blockSignals(False)
        if hasattr(self, "_color_mode_combo"):
            current_cm = self._color_value(self._color_mode_combo.currentText())
            self._color_mode_combo.blockSignals(True)
            self._color_mode_combo.clear()
            self._color_mode_combo.addItems(
                [self._color_label(c) for c in ("color", "grayscale")]
            )
            self._color_mode_combo.setCurrentText(self._color_label(current_cm))
            self._color_mode_combo.blockSignals(False)
        # Save dialog title (used by the Browse button).
        self._save_dialog_title = _("Export Plot")

    @classmethod
    def _format_label_text(cls, fmt: str) -> str:
        """Human-readable format label. Kept English-only intentionally."""
        return cls.FORMAT_LABELS.get(fmt, fmt.upper())

    @classmethod
    def _preset_button_label(cls, preset_id: str) -> str:
        """Translated preset button label."""
        return cls.PRESET_LABELS.get(preset_id, preset_id)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, default_path: Path) -> None:
        layout = QVBoxLayout(self)

        # ---- Top: file selector -----------------------------------------
        self._file_group = QGroupBox(_("File"))
        file_layout = QGridLayout(self._file_group)
        self._path_label = QLabel(_("Path:"))
        file_layout.addWidget(self._path_label, 0, 0)
        self._path_edit = QLineEdit(str(default_path))
        file_layout.addWidget(self._path_edit, 0, 1)
        self._browse_btn = QPushButton(_("Browse..."))
        file_layout.addWidget(self._browse_btn, 0, 2)

        self._format_label_widget = QLabel(_("Format:"))
        file_layout.addWidget(self._format_label_widget, 1, 0)
        self._format_combo = QComboBox()
        for fmt in ("svg", "pdf", "png", "jpg"):
            self._format_combo.addItem(self._format_label_text(fmt), fmt)
        # Default to whatever extension the caller passed in. We use
        # ``itemData`` rather than ``setCurrentText`` because the latter
        # is text-based and would only match the *current* language.
        # ``.lstrip("")`` stripped nothing (empty separator is a no-op),
        # so the dot survived and the combo always fell back to PNG.
        # Strip the dot so e.g. ``report.svg`` selects the SVG entry.
        default_fmt = default_path.suffix.lower().lstrip(".")
        if default_fmt not in self.FORMAT_LABELS:
            default_fmt = "png"
        idx = next(
            (
                i
                for i, fmt in enumerate(("svg", "pdf", "png", "jpg"))
                if fmt == default_fmt
            ),
            3,
        )
        self._format_combo.setCurrentIndex(idx)
        file_layout.addWidget(self._format_combo, 1, 1, 1, 2)
        layout.addWidget(self._file_group)

        # ---- Middle: format-aware options -------------------------------
        self._options_group = QGroupBox(_("Options"))
        form = QFormLayout(self._options_group)

        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(50, 2400)
        self._dpi_spin.setSingleStep(50)
        self._dpi_spin.setValue(300)
        self._dpi_label_widget = QLabel(_("Resolution (DPI):"))
        form.addRow(self._dpi_label_widget, self._dpi_spin)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(0.5, 60.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setSpecialValueText(_("auto"))
        self._width_spin.setValue(0.0)
        self._width_label_widget = QLabel(_("Width (in):"))
        form.addRow(self._width_label_widget, self._width_spin)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(0.5, 60.0)
        self._height_spin.setSingleStep(0.5)
        self._height_spin.setSpecialValueText(_("auto"))
        self._height_spin.setValue(0.0)
        self._height_label_widget = QLabel(_("Height (in):"))
        form.addRow(self._height_label_widget, self._height_spin)

        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 100)
        self._quality_spin.setSingleStep(5)
        self._quality_spin.setValue(95)
        self._quality_label_widget = QLabel(_("JPEG quality:"))
        form.addRow(self._quality_label_widget, self._quality_spin)

        self._bg_combo = QComboBox()
        self._bg_combo.addItems([_("White"), _("Transparent"), _("Theme")])
        self._bg_combo.setCurrentText(_("White"))
        self._bg_label_widget = QLabel(_("Background:"))
        form.addRow(self._bg_label_widget, self._bg_combo)

        self._color_mode_combo = QComboBox()
        self._color_mode_combo.addItems([_("Color"), _("Grayscale")])
        self._color_mode_combo.setCurrentText(_("Color"))
        self._color_mode_label_widget = QLabel(_("Color mode:"))
        form.addRow(self._color_mode_label_widget, self._color_mode_combo)

        self._transparent_checkbox = QCheckBox(_("Transparent background"))
        form.addRow("", self._transparent_checkbox)

        self._tight_checkbox = QCheckBox(_("Crop margins (tight)"))
        self._tight_checkbox.setChecked(True)
        form.addRow("", self._tight_checkbox)

        self._embed_fonts_checkbox = QCheckBox(_("Embed fonts (vector)"))
        self._embed_fonts_checkbox.setChecked(True)
        form.addRow("", self._embed_fonts_checkbox)

        layout.addWidget(self._options_group)

        # ---- Presets ----------------------------------------------------
        self._presets_group = QGroupBox(_("Presets"))
        preset_layout = QHBoxLayout(self._presets_group)
        self._preset_buttons: dict[str, QPushButton] = {}
        for preset_id in self.PRESET_LABELS:
            btn = QPushButton(self._preset_button_label(preset_id))
            btn.setProperty("preset_id", preset_id)
            btn.clicked.connect(lambda _checked=False, p=preset_id: self._apply_preset(p))
            preset_layout.addWidget(btn)
            self._preset_buttons[preset_id] = btn
        layout.addWidget(self._presets_group)

        # ---- Bottom: OK / Cancel ----------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._save_dialog_title = _("Export Plot")
        self._sync_format_dependent_widgets()

    def _wire_signals(self) -> None:
        self._format_combo.currentIndexChanged.connect(self._sync_format_dependent_widgets)
        self._browse_btn.clicked.connect(self._on_browse_clicked)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_clicked(self) -> None:
        """Show a native Save dialog with a filter matching the format."""
        fmt = self._current_format()
        filter_str = (
            f"{self.FORMAT_LABELS[fmt]} (*.{fmt});;All files (*)"
        )
        path, _ = QFileDialog.getSaveFileName(self, _("Export Plot"), self._path_edit.text(), filter_str)
        if path:
            self._path_edit.setText(path)

    def _current_format(self) -> str:
        return self._format_combo.currentData() or "png"

    def _sync_format_dependent_widgets(self) -> None:
        """Enable / disable inputs that depend on the chosen format."""
        fmt = self._current_format()
        is_raster = fmt in _RASTER_FORMATS
        is_jpeg = fmt == "jpg"
        is_pdf = fmt == "pdf"

        self._dpi_spin.setEnabled(is_raster)
        self._quality_spin.setEnabled(is_jpeg)
        # JPEG cannot honour transparency; disable to prevent the
        # user from creating a configuration that we will then refuse
        # on save.
        if is_jpeg:
            self._transparent_checkbox.setChecked(False)
            self._transparent_checkbox.setEnabled(False)
        else:
            self._transparent_checkbox.setEnabled(True)
        # PDF does not support transparency either.
        if is_pdf and self._transparent_checkbox.isChecked():
            self._transparent_checkbox.setChecked(False)

        # Font embedding is only meaningful for vector formats.
        self._embed_fonts_checkbox.setEnabled(fmt in _VECTOR_FORMATS)

        # Auto-correct the file extension when the format changes.
        current = self._path_edit.text().strip()
        if current:
            try:
                parsed = parse_options_from_path(current)
                if parsed.format != fmt:
                    new_path = str(Path(current).with_suffix(f".{fmt}"))
                    self._path_edit.setText(new_path)
            except ValueError:
                # Path with an unrecognised extension; just append one.
                p = Path(current)
                if p.suffix == "":
                    self._path_edit.setText(str(p.with_suffix(f".{fmt}")))

    def _apply_preset(self, preset_id: str) -> None:
        """Apply a preset by reading the current options and rewriting them."""
        try:
            current = self.get_options()
        except ValueError:
            current = PlotExportOptions()
        new_options = apply_preset(current, preset_id)
        self._set_options(new_options)

    def _set_options(self, options: PlotExportOptions) -> None:
        """Push ``options`` back into the dialog widgets."""
        # Format combo (find by data).
        idx = next(
            (
                i
                for i in range(self._format_combo.count())
                if self._format_combo.itemData(i) == options.format
            ),
            0,
        )
        self._format_combo.setCurrentIndex(idx)
        self._dpi_spin.setValue(options.dpi)
        self._quality_spin.setValue(options.jpeg_quality)
        self._bg_combo.setCurrentText(self._bg_label(options.background))
        self._color_mode_combo.setCurrentText(self._color_label(options.color_mode))
        self._transparent_checkbox.setChecked(options.transparent)
        self._tight_checkbox.setChecked(options.bbox_inches == "tight")
        self._embed_fonts_checkbox.setChecked(options.embed_fonts)
        self._width_spin.setValue(options.width_inches or 0.0)
        self._height_spin.setValue(options.height_inches or 0.0)
        # Update the path suffix to match the format.
        p = Path(self._path_edit.text())
        if p.suffix.lower().lstrip(".") != options.format:
            self._path_edit.setText(str(p.with_suffix(f".{options.format}")))
        self._sync_format_dependent_widgets()

    @staticmethod
    def _bg_label(value: str) -> str:
        return {
            "white": _("White"),
            "transparent": _("Transparent"),
            "theme": _("Theme"),
        }.get(value, _("White"))

    @staticmethod
    def _bg_value(label: str) -> str:
        return {
            _("White"): "white",
            _("Transparent"): "transparent",
            _("Theme"): "theme",
        }.get(label, "white")

    @staticmethod
    def _color_label(value: str) -> str:
        return {
            "color": _("Color"),
            "grayscale": _("Grayscale"),
        }.get(value, _("Color"))

    @staticmethod
    def _color_value(label: str) -> str:
        return {
            _("Color"): "color",
            _("Grayscale"): "grayscale",
        }.get(label, "color")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_options(self) -> PlotExportOptions:
        """Build a :class:`PlotExportOptions` from the dialog state.

        Raises ``ValueError`` when the path is empty, the extension
        does not match the chosen format, or the chosen configuration
        cannot be honoured (e.g. PDF with transparency).
        """
        path = self._path_edit.text().strip()
        if not path:
            raise ValueError(_("Please specify a destination path."))

        fmt = self._current_format()
        # Validate the extension; raise a clearer error than the facade
        # would so the UI can show it inline.
        p = Path(path)
        if p.suffix.lower().lstrip(".") != fmt:
            raise ValueError(
                _("File extension does not match format {0}. Expected .{0}.").format(fmt)
            )

        bg_label = self._bg_combo.currentText()
        background = self._bg_value(bg_label)
        transparent = self._transparent_checkbox.isChecked()
        # Map the dialog's background choice to a concrete option.
        if transparent:
            background = "transparent"
        elif background == "theme":
            background = "theme"
        else:
            background = "white"

        options = PlotExportOptions(
            format=fmt,  # type: ignore[arg-type]
            dpi=self._dpi_spin.value(),
            width_inches=self._width_spin.value() or None,
            height_inches=self._height_spin.value() or None,
            transparent=transparent,
            background=background,  # type: ignore[arg-type]
            color_mode=self._color_value(self._color_mode_combo.currentText()),  # type: ignore[arg-type]
            jpeg_quality=self._quality_spin.value(),
            bbox_inches="tight" if self._tight_checkbox.isChecked() else "standard",
            embed_fonts=self._embed_fonts_checkbox.isChecked(),
        )
        # Use the facade's own validation so the dialog and the file
        # writer share the same rules.
        from plot_export import _validate_options  # local import: avoid coupling

        _validate_options(options)
        return options

    def get_path(self) -> str:
        return self._path_edit.text().strip()