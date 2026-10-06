"""Offscreen UI audit harness.

Instantiates the real MainWindow under QT_QPA_PLATFORM=offscreen, loads the
bundled example data, opens the analysis dialogs, and writes a PNG of each so
the layout can actually be looked at rather than guessed at.

Not a test - a diagnostic tool. Run:
    QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe scripts/ui_audit.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

OUT = ROOT / "_ui_audit"
OUT.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QApplication

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)

app = QApplication(sys.argv)
app.setStyle("Fusion")

# The offscreen platform plugin on Windows exposes an EMPTY font database
# (QFontDatabase.families() == 0), so every glyph renders as a tofu box and
# the layout cannot be judged. Register the on-disk font files by hand.
# Microsoft YaHei covers both Latin and CJK, which matters here because the UI
# is bilingual.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",  # Microsoft YaHei (Latin + CJK)
    r"C:\Windows\Fonts\msjh.ttc",  # Microsoft JhengHei
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]
from PyQt6.QtGui import QFont, QFontDatabase

_loaded: list[str] = []
for _path in _FONT_CANDIDATES:
    if os.path.exists(_path):
        _fid = QFontDatabase.addApplicationFont(_path)
        if _fid >= 0:
            _loaded += QFontDatabase.applicationFontFamilies(_fid)

if _loaded:
    app.setFont(QFont("Microsoft YaHei", 9))
else:
    print("WARNING: no fonts could be registered; all text will be tofu", flush=True)

REPORT: list[str] = []


def note(msg: str) -> None:
    REPORT.append(msg)
    print(msg, flush=True)


def pump(times: int = 6) -> None:
    """Let Qt settle: process events, let layout and paint finish."""
    for _ in range(times):
        app.processEvents()


def shoot(widget, name: str, resize=None) -> Path | None:
    """Grab a widget to PNG. Returns the path, or None if it failed."""
    try:
        if resize is not None:
            widget.resize(*resize)
        widget.show()
        pump(10)
        pump(10)
        path = OUT / f"{name}.png"
        pixmap = widget.grab()
        if pixmap.isNull() or pixmap.width() < 10:
            note(f"  [FAIL] {name}: grabbed a null/empty pixmap")
            return None
        pixmap.save(str(path))
        note(f"  [ok]   {name}.png  {pixmap.width()}x{pixmap.height()}{'  (resized)' if resize else ''}")
        return path
    except Exception as exc:
        note(f"  [ERR]  {name}: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. Main window, empty
# ---------------------------------------------------------------------------
note("=== main window ===")
note(f"  fonts registered: {len(_loaded)} -> {_loaded[:6]}")
from views.ui_main_window import MainWindow

win = MainWindow()
win.resize(1600, 950)
shoot(win, "01_main_empty")

note("")
note("=== main window geometry sanity ===")
for name in ("sizeHint", "minimumSizeHint", "minimumSize", "maximumSize"):
    try:
        v = getattr(win, name)()
        note(f"  {name:18s} {v.width()}x{v.height()}")
    except Exception as exc:
        note(f"  {name:18s} <{type(exc).__name__}>")
note(f"  actual size        {win.width()}x{win.height()}")
note(f"  central widget     {type(win.centralWidget()).__name__}")
note(f"  window title       {win.windowTitle()!r}")

# ---------------------------------------------------------------------------
# 2. Load the bundled examples so the window is not empty
# ---------------------------------------------------------------------------
note("")
note("=== loading bundled example data ===")
import data.loader as loader

for fn_name in ("load_community", "load_moth_wings", "load_primate_traits", "load_primate_tree"):
    try:
        value = getattr(loader, fn_name)()
        shape = getattr(value, "shape", None)
        note(f"  {fn_name:22s} ok, shape={shape}")
    except Exception as exc:
        note(f"  {fn_name:22s} FAILED: {type(exc).__name__}: {exc}")

try:
    from models.data_matrix import DataMatrix

    frame = loader.load_community()
    # community_abundance.csv has two string columns up front: `site` (row
    # label) and `group` (the grouping variable). Only the numeric taxon
    # columns belong in the matrix.
    numeric = frame.select_dtypes(include="number")
    matrix = DataMatrix(
        data=numeric.to_numpy(dtype=float),
        row_labels=[str(s) for s in frame["site"]],
        col_labels=[str(c) for c in numeric.columns],
        specimen_metadata=[{"group": str(g)} for g in frame["group"]],
    )
    win._state.set_data_matrix(matrix)
    pump(8)
    note(f"  state has_data -> {win._state.has_data}")
    note(f"  state matrix  -> {win._state.data_matrix.data.shape}")
    note(f"  groups seen   -> {sorted(set(frame['group']))}")

    # Expand the navigation tree so the categories are visible.
    for tree in win.findChildren(type(win._nav_tree)) if hasattr(win, "_nav_tree") else []:
        tree.expandAll()
    if hasattr(win, "_nav_tree"):
        win._nav_tree.expandAll()
except Exception:
    note("  could not push data into the window state:")
    note("    " + traceback.format_exc().strip().replace("\n", "\n    "))

shoot(win, "02_main_with_data")

# --- ribbon tabs, with data loaded so the buttons enable -------------------
try:
    ribbon = None
    for child in win.findChildren(object):
        if type(child).__name__ in {"Ribbon", "CustomRibbon", "RibbonWidget"}:
            ribbon = child
            break
    if ribbon is not None and hasattr(ribbon, "setCurrentIndex"):
        for i in range(min(5, ribbon.count())):
            ribbon.setCurrentIndex(i)
            pump(6)
            shoot(win, f"02b_ribbon_tab_{i}")
    else:
        note("  ribbon widget not located by class name")
except Exception as exc:
    note(f"  ribbon tab sweep failed: {type(exc).__name__}: {exc}")

# --- run a real analysis so the plotting path is exercised -----------------
# Go through MainWindow._on_pca_result_ready rather than calling the canvas
# directly: the labels/groups fix lives in the *wiring* between the state and
# the canvas, so bypassing it would test the wrong thing.
try:
    from stats.pca import PCAAnalyzer

    labels, groups, group_names = win._get_plot_labels_and_groups()
    note(f"  resolved labels      : {labels[:4] if labels else None} ...")
    note(f"  resolved groups      : {groups}")
    note(f"  resolved group names : {group_names}")

    result = PCAAnalyzer().analyze(win._state.data_matrix.data, n_components=3)
    win._on_pca_result_ready(result, ctx={"show_scree": True})
    pump(10)
    note("  ran PCA through the main-window handler")
    # The scores plot is the first workspace tab; the scree plot was added
    # last and is the one left visible, so step back to the scores.
    win._workspace.setCurrentIndex(2)
    pump(8)
    shoot(win, "03_main_with_pca_plot")
except Exception:
    note("  could not run PCA into the workspace:")
    note("    " + traceback.format_exc().strip().replace("\n", "\n    "))

# ---------------------------------------------------------------------------
# 3. Analysis dialogs - every QDialog subclass we can find
# ---------------------------------------------------------------------------
note("")
note("=== analysis dialogs ===")
DIALOGS = [
    # module, class, size
    ("views.ui_dialogs", "PCADialog", (760, 700)),
    ("views.ui_dialogs", "PCoADialog", (780, 740)),
    ("views.ui_dialogs", "NMDSOptionsDialog", (760, 700)),
    ("views.ui_dialogs", "DiversityDialog", (760, 700)),
    ("views.ui_dialogs", "RarefactionDialog", (760, 700)),
    ("views.ui_dialogs", "ImportDialog", (860, 640)),
    ("views.ui_dialogs", "SimperDialog", (740, 640)),
    ("views.ui_dialogs", "UnivariateDialog", (740, 660)),
    ("views.ui_dialogs", "LDADialog", (740, 660)),
    ("views.ui_dialogs", "ClusteringDialog", (740, 660)),
    ("views.ui_dialogs", "CONISSDialog", (740, 660)),
    ("views.ui_dialogs", "MarkovDialog", (740, 660)),
    ("views.ui_dialogs", "DirectionalDialog", (740, 660)),
    ("views.ui_dialogs", "EFADialog", (740, 660)),
    ("views.ui_dialogs", "TPSGridDialog", (760, 700)),
    ("views.ui_dialogs", "SpatialRipleyKDialog", (740, 660)),
    ("views.ui_dialogs", "BiostratigraphyDialog", (820, 700)),
    ("views.ui_dialogs", "WaveletDialog", (740, 660)),
    ("views.ui_dialogs", "CCADialog", (780, 700)),
    ("views.ui_dialogs", "IsotopeAnalysisDialog", (780, 700)),
    ("views.ui_dialogs", "StratigraphicCorrelationDialog", (820, 700)),
    ("views.ui_dialogs", "PaleoEnvironmentDialog", (780, 700)),
    ("views.ui_allometry_dialogs", "AllometryDialog", (800, 720)),
    ("views.ui_allometry_dialogs", "PLSDialog", (800, 720)),
    ("views.ui_beta_diversity_dialogs", "BetaDiversityDialog", (800, 680)),
    ("views.ui_beta_diversity_dialogs", "CoverageRarefactionDialog", (800, 680)),
    ("views.ui_imputation_dialog", "ImputationDialog", (800, 740)),
    ("views.ui_evolution_rate_dialogs", "EvolutionRateDialog", (820, 720)),
    ("views.ui_extinction_dialogs", "ExtinctionIntervalDialog", (780, 680)),
    ("views.ui_null_model_dialogs", "NullModelDialog", (780, 660)),
    ("views.ui_pcm_dialogs", "PICDialog", (760, 640)),
    ("views.ui_pcm_dialogs", "AncestralStateDialog", (760, 640)),
    ("views.ui_pcm_dialogs", "PhyloSignalDialog", (760, 640)),
    ("views.ui_pcm_dialogs", "PhyloANOVADialog", (760, 640)),
    ("views.ui_macroevolution_dialogs", "Morpho3DDialog", (780, 660)),
    # MacroevolutionDialog, AddRunDialog and PlotExportDialog need real
    # constructor arguments, so they are handled individually below rather
    # than by this (module, class, size) loop.
]

import importlib

controller = getattr(win, "_statistics_controller", None)
index = 3
for module_name, class_name, size in DIALOGS:
    index += 1
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as exc:
        note(f"  [ERR]  {class_name}: import failed: {type(exc).__name__}: {exc}")
        continue

    dialog = None
    # Bound explicitly: `cls` is rebound on every iteration, so a bare closure
    # would capture whichever class happened to be last. These are called
    # within the same iteration so it cannot bite today, but binding the
    # default makes that a property of the code rather than of the caller.
    for attempt in (
        lambda c=cls: c(parent=win, controller=controller),
        lambda c=cls: c(win, controller),
        lambda c=cls: c(parent=win),
        lambda c=cls: c(),
    ):
        try:
            dialog = attempt()
            break
        except TypeError:
            continue
        except Exception as exc:
            note(f"  [ERR]  {class_name}: construct: {type(exc).__name__}: {exc}")
            dialog = None
            break
    if dialog is None:
        note(f"  [ERR]  {class_name}: no working constructor signature")
        continue

    shoot(dialog, f"{index:02d}_{class_name}", resize=size)
    dialog.close()
    pump(2)

# MacroevolutionDialog's signature is (controller, parent) -- NOT (parent, controller).
# Passing (win, controller) made QDialog try to use the StatisticsController as
# its parent widget, which raised TypeError.
try:
    module = importlib.import_module("views.ui_macroevolution_dialogs")
    dialog = module.MacroevolutionDialog(controller, win)
    shoot(dialog, f"{index + 1:02d}_MacroevolutionDialog", resize=(840, 640))
    dialog.close()
except Exception as exc:
    note(f"  [ERR]  MacroevolutionDialog: {type(exc).__name__}: {exc}")

# PlotExportDialog's signature is (default_path, parent) -- the path comes
# first. Passing (win, path) made QDialog try to use the str as its parent.
try:
    module = importlib.import_module("views.ui_plot_export_dialog")
    dialog = module.PlotExportDialog(str(OUT / "figure.png"), win)
    shoot(dialog, f"{index + 2:02d}_PlotExportDialog", resize=(660, 640))
    dialog.close()
except Exception as exc:
    note(f"  [ERR]  PlotExportDialog: {type(exc).__name__}: {exc}")

# AddRunDialog's signature is (manager, parent); it needs a real PresetManager.
try:
    module = importlib.import_module("views.ui_runlist_panel")
    from presets.manager import PresetManager

    manager = getattr(win, "_preset_manager", None) or PresetManager()
    dialog = module.AddRunDialog(manager, win)
    shoot(dialog, "39_AddRunDialog", resize=(720, 560))
    dialog.close()
except Exception as exc:
    note(f"  [ERR]  AddRunDialog: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
note("")
note("=== summary ===")
ok = sum(1 for line in REPORT if "[ok]" in line)
bad = sum(1 for line in REPORT if "[ERR]" in line or "[FAIL]" in line)
note(f"  screenshots written: {ok}   failed: {bad}")
note(f"  output directory:    {OUT}")
(OUT / "report.txt").write_text("\n".join(REPORT), encoding="utf-8")
