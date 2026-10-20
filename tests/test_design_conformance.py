# =============================================================================
# FILE: tests/test_design_conformance.py
# =============================================================================
"""
Regression tests for the design-system conformance defects found by the
2026-10-20 front-end review.

Each test pins one thing that used to be wrong and is user-visible:

* the spreadsheet painted its cell numbers in the light theme's text
  colour, so in the dark theme the numbers measured 1.63:1 against the
  background -- effectively invisible;
* the default (Run) button painted white text on the dark theme's light
  blue accent, 2.54:1, in all 22 analysis dialogs;
* the analysis canvas and the plot-menu plotters used different figure
  type scales, so the same axis label was 12pt in one and 8.8pt in the
  other;
* the canvas carried its own copy of a categorical palette, so the
  figure in the window was not the figure the export produced;
* the diagnostic console had five Chinese source strings, so its
  buttons read Chinese in the English interface too;
* the navigation tree's icon painters hard-coded the deprecated flat-UI
  palette and did not follow the theme at all.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIRS = ("views", "visualization")


# =============================================================================
# helpers
# =============================================================================
def _ui_files() -> list[Path]:
    out: list[Path] = []
    for sub in UI_DIRS:
        out.extend(sorted((REPO_ROOT / sub).rglob("*.py")))
    return out


# The files whose colours are UI FURNITURE -- chrome, dialogs, tree icons --
# as opposed to the colours of a plotted data series.  A data series may
# legitimately carry a categorical colour; the chrome may not, because the
# chrome has to follow the theme.  The plot modules are deliberately not
# in this list: converting their series colours to the registered palette
# is the remaining half of the palette work and is tracked as such.
CHROME_FILES = (
    "views/ui_navigation.py",
    "views/ui_dialogs.py",
    "views/ui_main_window.py",
    "views/ui_spreadsheet.py",
    "views/diagnostic_console.py",
    "views/ui_permutation_dialogs.py",
    "views/ui_imputation_dialog.py",
    "views/ui_allometry_dialogs.py",
    "views/ui_beta_diversity_dialogs.py",
    "views/ui_evolution_rate_dialogs.py",
    "views/ui_extinction_dialogs.py",
    "views/ui_macroevolution_dialogs.py",
    "views/ui_null_model_dialogs.py",
    "views/ui_pcm_dialogs.py",
    "views/ui_runlist_panel.py",
    "views/ui_plot_export_dialog.py",
    "views/script_console.py",
    "views/ui_navigation.py",
)


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """The file's lines with comments and docstring bodies removed.

    A comment may legitimately mention a colour it is explaining, so the
    scan has to skip them.
    """
    out: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append((i, line))
    return out


# The render check runs in a SUBPROCESS.
#
# This test file creates NO QApplication in the pytest process, on
# purpose.  Qt's teardown rules differ between platforms: on Linux an
# offscreen widget plus a QApplication that gets collected aborts the
# interpreter -- and because the collection happens at whatever later
# point the garbage collector runs, the crash lands far away from the
# test that caused it (it took out a Python 3.11 job on CI three
# percent before the end of the run, in an unrelated test).  A hard
# abort in the pytest process would take the whole suite down rather
# than failing one test, so the rendering happens out of process and the
# parent only asserts on the numbers it printed.
_RENDER_SCRIPT = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, {repo!r})
from PyQt6.QtWidgets import QApplication

app = QApplication(["paleoast-design-render"])
from views.ui_spreadsheet import ScientificSpreadsheet

sheet = ScientificSpreadsheet()
sheet.load_data([[1.5, 2.25], [3.125, 4.0]], row_labels=["r1", "r2"], col_labels=["c1", "c2"])
sheet.setDarkTheme(True)
sheet.resize(320, 160)
sheet.show()
app.processEvents()

image = sheet.grab().toImage()
counts = {{}}
for y in range(image.height()):
    for x in range(image.width()):
        name = image.pixelColor(x, y).name().upper()
        counts[name] = counts.get(name, 0) + 1
for name in ("#2C3E50", "#F1F5F9", "#0F172A"):
    print(name, counts.get(name, 0))
sheet.close()
sheet.deleteLater()
app.processEvents()
sys.stdout.flush()
os._exit(0)
"""


def _render_pixel_counts() -> dict[str, int]:
    """Render the dark-theme spreadsheet in a subprocess and count pixels."""
    import subprocess
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "paleo_design_render.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_RENDER_SCRIPT.format(repo=str(REPO_ROOT)))
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=300)
    out = proc.stdout or ""
    counts: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("#"):
            try:
                counts[parts[0].upper()] = int(parts[1])
            except ValueError:
                continue
    if not counts:
        raise AssertionError(f"the render subprocess produced no counts: {out!r} {proc.stderr!r}")
    return counts


def _srgb(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hexstr: str) -> float:
    h = hexstr.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb(r) + 0.7152 * _srgb(g) + 0.0722 * _srgb(b)


def _contrast(fg: str, bg: str) -> float:
    la, lb = _luminance(fg), _luminance(bg)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# =============================================================================
# the palette itself
# =============================================================================
def test_palette_text_colours_clear_the_4_5_to_1_bar():
    """The semantic text roles must be readable on their own surface.

    This is the check the hard-coded literals bypassed: "#2C3E50" on the
    dark background measured 1.63:1 and nothing in the suite noticed.
    """
    from config.design_system import ColorPalette, ColorPaletteDark

    for palette in (ColorPalette, ColorPaletteDark):
        for surface in (palette.bg_primary, palette.bg_secondary, palette.bg_tertiary):
            for role in ("text_primary", "text_secondary"):
                ratio = _contrast(getattr(palette, role), surface)
                assert ratio >= 4.5, f"{palette.__name__}.{role} on {surface} is {ratio:.2f}:1"


def test_accent_text_uses_on_primary_not_white():
    """Text on an accent fill must use the on_primary role."""
    from config.design_system import ColorPalette, ColorPaletteDark

    for palette in (ColorPalette, ColorPaletteDark):
        ratio = _contrast(palette.on_primary, palette.primary)
        assert ratio >= 4.5, f"{palette.__name__}: on_primary on primary is {ratio:.2f}:1"
        # The bug this pins: hard-coded white on the dark accent was 2.54:1.
        white = _contrast("#FFFFFF", palette.primary)
        if palette is ColorPaletteDark:
            assert white < 4.5, "this test only makes sense while white fails on the dark accent"


# =============================================================================
# the spreadsheet delegate
# =============================================================================
def test_spreadsheet_delegate_paints_with_the_theme_colour():
    """The cell painter must not hard-code a text colour."""
    from views.ui_spreadsheet import SpreadsheetDelegate

    source = (REPO_ROOT / "views" / "ui_spreadsheet.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    colours: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "QColor"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("#")
        ):
            colours.append(node.args[0].value)

    assert "#2C3E50" not in colours, "the spreadsheet still paints with the deprecated light-theme text colour"
    # The delegate must know the theme.
    assert hasattr(SpreadsheetDelegate, "set_dark_theme")


def test_spreadsheet_cell_text_is_readable_in_the_dark_theme():
    """Render the sheet in the dark theme and check the text pixels."""
    from config.design_system import ColorPaletteDark

    counts = _render_pixel_counts()

    # The regression guard: the old light-theme text colour must not be
    # painted at all in the dark theme.  Its absence is a stable
    # assertion; the exact pixel count of the correct colour is not,
    # because it depends on how much of the window is visible.
    assert counts.get("#2C3E50", 0) == 0, "the deprecated light-theme text colour is still painted"
    assert counts.get(ColorPaletteDark.bg_primary.upper(), 0) > 1000, "the dark background is not there"
    assert counts.get(ColorPaletteDark.text_primary.upper(), 0) > 10, (
        "the cell text is not painted in the theme's text colour"
    )


# =============================================================================
# the figure type scale
# =============================================================================
def test_both_figure_paths_use_the_same_type_scale():
    """The canvas and the plotters must agree on the figure font sizes.

    The plotters ran inside a matplotlib style library that sets
    axes.labelsize to 8.8, while the canvas used 12 -- a 27% difference
    between two figures in the same window.
    """
    from config.design_system import FigureTypography
    from visualization._style_scope import resolve_style_params

    scoped = resolve_style_params("seaborn-v0_8-paper")
    for key, value in FigureTypography.rc_overrides().items():
        assert scoped.get(key) == value, f"{key} is {scoped.get(key)} in the plotters, {value} in the scale"

    # And the canvas must read the same scale.
    canvas = (REPO_ROOT / "views" / "ui_plot_canvas.py").read_text(encoding="utf-8")
    assert "FigureTypography.rc_overrides()" in canvas


def test_figure_type_scale_is_the_documented_one():
    from config.design_system import FigureTypography

    overrides = FigureTypography.rc_overrides()
    assert overrides["axes.labelsize"] == 12
    assert overrides["axes.titlesize"] == 14
    assert overrides["xtick.labelsize"] == 10


# =============================================================================
# the categorical palette
# =============================================================================
def test_the_canvas_uses_the_registered_palette():
    """The canvas must not carry its own copy of a categorical palette."""
    from config.colors import current_palette

    canvas = (REPO_ROOT / "views" / "ui_plot_canvas.py").read_text(encoding="utf-8")
    assert "from config.colors import current_palette" in canvas
    # No hard-coded eight-colour list left.
    assert not re.search(r"COLORS\s*=\s*\[\s*[\"']#", canvas)

    palette = current_palette()
    assert len(palette) >= 8


def test_the_spreadsheet_uses_the_registered_palette():
    from config.colors import current_palette

    sheet = (REPO_ROOT / "views" / "ui_spreadsheet.py").read_text(encoding="utf-8")
    assert "from config.colors import current_palette" in sheet
    assert "GROUP_COLORS = [" not in sheet
    assert len(current_palette()) >= 8


# =============================================================================
# i18n
# =============================================================================
def test_no_chinese_source_strings_in_the_view_layer():
    """``_()`` takes the ENGLISH source text; a Chinese key never translates.

    The diagnostic console had five of them, so its buttons read Chinese
    in the English interface.
    """
    offenders: list[str] = []
    for path in _ui_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "_":
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if any("\u4e00" <= ch <= "\u9fff" for ch in arg.value):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {arg.value!r}")
    assert not offenders, "Chinese source strings in the view layer: " + "; ".join(offenders)


def test_the_diagnostic_console_strings_are_translated():
    from config.i18n import translations_en, translations_zh

    for key in ("Diagnostic Console", "Pause", "Resume", "Clear"):
        assert key in translations_zh.TRANSLATIONS, f"no Chinese entry for {key!r}"
        assert key in translations_en.TRANSLATIONS, f"no English entry for {key!r}"


# =============================================================================
# the navigation tree
# =============================================================================
def test_navigation_icons_follow_the_theme():
    """The icon painters must not hard-code the deprecated flat-UI palette."""
    source = (REPO_ROOT / "views" / "ui_navigation.py").read_text(encoding="utf-8")
    code = "\n".join(line for _i, line in _code_lines(REPO_ROOT / "views" / "ui_navigation.py"))
    for deprecated in ("#E74C3C", "#27AE60", "#F39C12", "#9B59B6", "#3498DB", "#E8F4F8"):
        assert deprecated not in code, f"{deprecated} is still hard-coded in the navigation tree"
    assert source  # the file exists and was read


def test_navigation_overlay_uses_the_palette_alphas():
    """The selected/hover overlays must come from the palette.

    The old literals were the light theme's alphas applied in both
    themes, so a selected row and a hovered row looked the same.
    """
    from config.design_system import ColorPalette, ColorPaletteDark

    source = (REPO_ROOT / "views" / "ui_navigation.py").read_text(encoding="utf-8")
    assert "QColor(59, 130, 246, 38)" not in source
    assert "QColor(30, 64, 175, 20)" not in source
    assert "c.selected_overlay" in source
    assert "c.hover_overlay" in source

    # And the two themes must actually differ, or the fix would be a no-op.
    assert ColorPalette.selected_overlay != ColorPaletteDark.selected_overlay
    assert ColorPalette.hover_overlay != ColorPaletteDark.hover_overlay


def test_the_overlay_helper_parses_the_palette_rgba():
    from views.ui_navigation import _overlay_color

    colour = _overlay_color("rgba(59, 130, 246, 0.3)")
    assert colour.red() == 59
    assert colour.green() == 130
    assert colour.blue() == 246
    # 0.3 * 255 = 76.5, and Python rounds halves to even -> 76.
    assert colour.alpha() == 76


# =============================================================================
# no literal "white" on an accent fill
# =============================================================================
def test_no_literal_white_text_on_an_accent_fill():
    """``color: white`` on an accent is 2.54:1 in the dark theme."""
    import re as _re

    offenders: list[str] = []
    for path in _ui_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _re.search(r"(color|selection-color)\s*:\s*white\s*;", line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}")
    assert not offenders, "literal white on an accent fill: " + "; ".join(offenders)


# =============================================================================
# the deprecated palette must not be in the UI layer
# =============================================================================
def test_the_deprecated_flat_ui_palette_is_gone_from_the_chrome():
    """The flat-UI palette in DESIGN_SYSTEM_SPECIFICATION.md is superseded.

    ``config/design_system.py`` says so in as many words: it, and not the
    docstring of any other file, is authoritative.  The document is the
    stale artefact.

    Scope: the UI chrome.  The colours of a plotted data series are a
    separate piece of work (the plot modules still carry hard-coded
    series colours); a series colour is not chrome and may legitimately
    be categorical.
    """
    deprecated = (
        "#3498DB",
        "#2980B9",
        "#5DADE2",
        "#27AE60",
        "#F39C12",
        "#E74C3C",
        "#16A085",
        "#E8F4F8",
        "#9B59B6",
        "#F1C40F",
    )
    offenders: list[str] = []
    for rel in CHROME_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        for i, line in _code_lines(path):
            for value in deprecated:
                if value in line:
                    offenders.append(f"{rel}:{i} {value}")
    assert not offenders, "deprecated flat-UI palette still in the chrome: " + "; ".join(offenders)


def test_the_design_specification_is_marked_as_superseded():
    """The stale document must not keep misleading the next reader."""
    spec = (REPO_ROOT / "DESIGN_SYSTEM_SPECIFICATION.md").read_text(encoding="utf-8")
    assert "config/design_system.py" in spec, "DESIGN_SYSTEM_SPECIFICATION.md must point at the authoritative module"


def test_no_chrome_module_hardcodes_the_light_theme_only_text_colour():
    """#2C3E50 is the old light-theme text colour; in the dark theme it is
    1.63:1 on the background."""
    offenders: list[str] = []
    for rel in CHROME_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        for i, line in _code_lines(path):
            if '"#2C3E50"' in line or "'#2C3E50'" in line:
                offenders.append(f"{rel}:{i}")
    assert not offenders, "hard-coded light-theme text colour: " + "; ".join(offenders)


def test_the_palette_is_the_one_the_export_reads():
    """A canvas figure and an exported figure must be the same figure."""
    from config.colors import current_palette

    canvas = (REPO_ROOT / "views" / "ui_plot_canvas.py").read_text(encoding="utf-8")
    assert "current_palette()" in canvas
    export = (REPO_ROOT / "visualization" / "r_export.py").read_text(encoding="utf-8")
    assert "get_color_scheme" in export
    assert len(current_palette()) >= 8
