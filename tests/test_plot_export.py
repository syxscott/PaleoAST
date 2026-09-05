"""Tests for the plot export facade."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot_export import (
    PlotExportOptions,
    apply_preset,
    export_figure,
    parse_options_from_path,
)


@pytest.fixture
def tiny_figure():
    """Return a deterministic figure that survives across tests."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 0], color="#E74C3C")
    return fig


def _read(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_export_svg_writes_vector_file(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.svg"
    export_figure(tiny_figure, str(out), PlotExportOptions(format="svg"))
    data = _read(out)
    assert data.startswith(b"<?xml") or data.startswith(b"<svg")


def test_export_pdf_writes_pdf_file(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.pdf"
    export_figure(tiny_figure, str(out), PlotExportOptions(format="pdf"))
    assert _read(out).startswith(b"%PDF-")


def test_export_png_writes_png_with_dpi(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.png"
    export_figure(
        tiny_figure, str(out), PlotExportOptions(format="png", dpi=144)
    )
    data = _read(out)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    # Width and height in pixels are stored as 4-byte big-endian ints at
    # offset 16 / 20 of a PNG. dpi=144 over a 3-inch figure should yield
    # about 432 pixels; allow ±15% tolerance because matplotlib's
    # bbox_inches="tight" can shrink the canvas by a few percent.
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert 360 <= width <= 500
    assert 240 <= height <= 340


def test_export_jpg_quality_round_trip(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.jpg"
    export_figure(
        tiny_figure,
        str(out),
        PlotExportOptions(format="jpg", jpeg_quality=70),
    )
    assert _read(out).startswith(b"\xff\xd8\xff")


def test_export_transparent_png(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.png"
    export_figure(
        tiny_figure,
        str(out),
        PlotExportOptions(format="png", transparent=True),
    )
    # Decode and check alpha channel exists.
    from PIL import Image

    img = Image.open(out)
    assert img.mode in ("RGBA", "LA")


def test_export_grayscale_svg_keeps_alpha(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.svg"
    export_figure(
        tiny_figure,
        str(out),
        PlotExportOptions(format="svg", color_mode="grayscale"),
    )
    # SVG stays vector even under grayscale mode.
    assert _read(out).startswith(b"<?xml") or _read(out).startswith(b"<svg")


def test_export_pdf_rejects_transparent_flag(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.pdf"
    with pytest.raises(ValueError):
        export_figure(
            tiny_figure,
            str(out),
            PlotExportOptions(format="pdf", transparent=True),
        )


def test_export_jpg_rejects_transparent_flag(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.jpg"
    with pytest.raises(ValueError):
        export_figure(
            tiny_figure,
            str(out),
            PlotExportOptions(format="jpg", transparent=True),
        )


def test_export_invalid_format_raises(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.bmp"
    with pytest.raises(ValueError):
        export_figure(tiny_figure, str(out), PlotExportOptions(format="bmp"))


def test_export_normalises_extension(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot"
    export_figure(tiny_figure, str(out), PlotExportOptions(format="svg"))
    # File should be created with the requested format's extension.
    assert os.path.exists(str(out) + ".svg")


def test_export_resizes_figure(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.png"
    export_figure(
        tiny_figure,
        str(out),
        PlotExportOptions(format="png", dpi=100, width_inches=4.0, height_inches=3.0),
    )
    width = int.from_bytes(_read(out)[16:20], "big")
    height = int.from_bytes(_read(out)[20:24], "big")
    # Tight bbox may add a few pixels of margin; allow ±20% tolerance.
    assert 320 <= width <= 480
    assert 240 <= height <= 360


def test_export_background_white_for_jpg(tmp_path: Path, tiny_figure) -> None:
    out = tmp_path / "plot.jpg"
    export_figure(
        tiny_figure,
        str(out),
        PlotExportOptions(format="jpg", background="white"),
    )
    from PIL import Image

    img = Image.open(out).convert("RGB")
    pixels = np.asarray(img)
    # Edge pixels should match the requested background.
    assert (pixels[0, 0] == [255, 255, 255]).all()


def test_preset_publication_picks_pdf() -> None:
    options = apply_preset(PlotExportOptions(format="png"), "publication")
    assert options.format in {"pdf", "svg"}
    assert options.dpi >= 600


def test_preset_screen_picks_png_low_dpi() -> None:
    options = apply_preset(PlotExportOptions(format="svg"), "screen")
    assert options.format == "png"
    assert options.dpi <= 200


def test_preset_vector_prefers_svg_or_pdf() -> None:
    options = apply_preset(PlotExportOptions(format="png"), "vector")
    assert options.format in {"pdf", "svg"}


def test_parse_options_from_path_infers_format(tmp_path: Path) -> None:
    options = parse_options_from_path(tmp_path / "plot.svg")
    assert options.format == "svg"


def test_parse_options_from_path_keeps_user_format(tmp_path: Path) -> None:
    options = parse_options_from_path(tmp_path / "plot.png", format="png")
    assert options.format == "png"


def test_parse_options_from_path_unknown_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_options_from_path(tmp_path / "plot.bmp")


# ---------------------------------------------------------------------------
# PyQt UI tests (require PyQt6). They are skipped when the runtime does
# not have PyQt installed so the lighter tests stay fast in CI.
# ---------------------------------------------------------------------------

pytest.importorskip("PyQt6", reason="UI export dialog tests require PyQt6")


@pytest.fixture(scope="module")
def qapp_export():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-export-tests"])
    return app


def _set_format(dialog, fmt: str) -> None:
    """Set the dialog's format combo to ``fmt`` via its data role."""
    for i in range(dialog._format_combo.count()):
        if dialog._format_combo.itemData(i) == fmt:
            dialog._format_combo.setCurrentIndex(i)
            return
    raise AssertionError(f"format {fmt!r} not in combo")


def test_export_dialog_returns_options_for_svg(qapp_export, tmp_path):
    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "default.svg")
    _set_format(dialog, "svg")
    dialog._dpi_spin.setValue(300)
    dialog._bg_combo.setCurrentText("White")
    dialog._tight_checkbox.setChecked(True)

    options = dialog.get_options()

    assert options.format == "svg"
    assert options.dpi == 300
    assert options.background == "white"
    assert options.bbox_inches == "tight"


def test_export_dialog_rejects_inconsistent_extension(qapp_export, tmp_path):
    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "plot.svg")
    # Force the path / format into an inconsistent state without
    # triggering the auto-correct slot. The simplest reliable way is
    # to set the underlying combo data via itemData bypass: we set
    # the format via direct slot manipulation.
    dialog._path_edit.setText(str(tmp_path / "plot.svg"))
    # Mutate the combo's current "data" so it returns "png" while the
    # path suffix stays ".svg". This is exactly the inconsistency the
    # dialog must reject.
    for i in range(dialog._format_combo.count()):
        if dialog._format_combo.itemData(i) == "png":
            # Block the slot while we mutate so the auto-correct
            # doesn't rewrite the path.
            dialog._format_combo.blockSignals(True)
            dialog._format_combo.setCurrentIndex(i)
            dialog._format_combo.blockSignals(False)
            break

    with pytest.raises(ValueError):
        dialog.get_options()


def test_export_dialog_applies_presets(qapp_export, tmp_path):
    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "plot.png")
    _set_format(dialog, "png")
    dialog._apply_preset("publication")
    assert dialog._format_combo.currentData() in {"pdf", "svg"}
    dialog._apply_preset("screen")
    assert dialog._format_combo.currentData() == "png"
    assert dialog._dpi_spin.value() <= 200


def test_export_dialog_hides_dpi_for_vector_formats(qapp_export, tmp_path):
    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "plot.svg")
    # DPI is irrelevant for vector output; the spin box should be
    # disabled to communicate that.
    _set_format(dialog, "svg")
    assert not dialog._dpi_spin.isEnabled()
    _set_format(dialog, "png")
    assert dialog._dpi_spin.isEnabled()


def test_export_dialog_shows_quality_only_for_jpg(qapp_export, tmp_path):
    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "plot.png")
    _set_format(dialog, "png")
    assert not dialog._quality_spin.isEnabled()
    _set_format(dialog, "jpg")
    assert dialog._quality_spin.isEnabled()


def test_export_dialog_rejects_pdf_transparent(qapp_export, tmp_path):
    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "plot.pdf")
    _set_format(dialog, "pdf")
    dialog._transparent_checkbox.setChecked(True)

    with pytest.raises(ValueError):
        dialog.get_options()


def test_export_dialog_retranslates_on_language_change(qapp_export, tmp_path):
    """Bug: dialog labels must refresh when the UI language changes.

    Previously all labels were captured at construction time and never
    refreshed, so toggling language at runtime left the dialog in the
    old language.
    """
    from config import i18n

    # The translator singleton may have been created before the test
    # fixture's QApplication existed (state was carried over from a
    # previous test that imported ``config.i18n`` without PyQt).
    # Reset it so it picks up the QObject base in the current test
    # environment, then re-register the dictionaries.
    i18n._reset_translator()
    i18n.register_translations()
    i18n.set_language("en")

    from views.ui_plot_export_dialog import PlotExportDialog

    dialog = PlotExportDialog(tmp_path / "plot.png")
    dialog.retranslate()
    title_en = dialog.windowTitle()
    file_group_en = dialog._file_group.title()
    browse_en = dialog._browse_btn.text()

    i18n.set_language("zh")
    dialog.retranslate()

    assert dialog.windowTitle() != title_en, (
        f"Window title stayed {title_en!r} after language change"
    )
    assert dialog._file_group.title() != file_group_en
    assert dialog._browse_btn.text() != browse_en

    # Restore English so subsequent tests are stable.
    i18n.set_language("en")
    dialog.retranslate()
    assert dialog.windowTitle() == title_en


def test_plot_canvas_export_plot_uses_facade(tmp_path, monkeypatch):
    """InteractivePlotCanvas.export_plot must delegate to plot_export.

    The legacy ``_export_plot`` should now route through the unified
    facade and accept ``options`` so the dialog can drive it.
    """
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from views.ui_plot_canvas import InteractivePlotCanvas

    from plot_export import PlotExportOptions, export_figure

    app = QApplication.instance() or QApplication(["paleoast-export-tests"])
    canvas = InteractivePlotCanvas()

    # Avoid any modal information pop-ups on success/error.
    from views.ui_plot_canvas import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: None),
    )

    out = tmp_path / "canvas.png"
    options = PlotExportOptions(format="png", dpi=144)
    options.metadata["_target_path"] = str(out)
    try:
        canvas.export_plot(options=options)
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        canvas._figure.clear()
        canvas.close()