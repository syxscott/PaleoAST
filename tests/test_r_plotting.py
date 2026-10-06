# =============================================================================
# FILE: tests/test_r_plotting.py
# =============================================================================
"""
Tests for the R plotting bridge: export to ``.R``, and run it back.

WHY THESE EXIST
---------------
The R feature was verified by hand -- scripts were exported, R really ran, and
the figures were looked at. That is the only verification that proves the
figures are right, and it is also the only one that does not repeat. These
tests pin the three properties that are cheap to check and expensive to get
wrong, plus one that only a real R can check:

  1. The numbers in the ``.csv`` next to the script are the numbers in the
     result object. Every other guarantee is cosmetic if this drifts, and it
     can drift silently: the CSV is written by hand-rolled formatting, so a
     rounding or column-offset change produces a script that still parses and
     still runs, just with the wrong points in the wrong place.

  2. A hand-edited script is never overwritten without asking. The user is
     expected to open the ``.R`` in RStudio; if ``_on_export_as_r_script``
     clobbered their edits, the feature would be actively harmful.

  3. "Re-run" runs the file on disk and never regenerates it. These are two
     separate actions on purpose (see views/ui_main_window.py).

  4. Figures are found by diffing the directory, not by guessing the name the
     generator happened to use -- the user may have renamed the output.

Tests that need a real R are skipped when none is installed, because R is
something the user provides, not a packaged dependency (see the module
docstring of visualization/r_render.py for why rpy2 is not used).
"""

from __future__ import annotations

import csv
import functools
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from visualization.r_export import (
    RPlotSpec,
    RScriptExporter,
    validate_r_script,
)
from visualization.r_render import (
    describe_missing_r,
    find_rscript,
    run_r_script,
)

pytest.importorskip("PyQt6", reason="the R actions live in the Qt main window")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pca_result():
    """A real PCAResult -- not a stub, so the CSV assertions test the truth."""
    from stats.pca import PCAAnalyzer

    rng = np.random.default_rng(20261003)
    # 5 variables, correlated by a known loading matrix, so PC1 carries real
    # variance instead of the noise-dominated PC1 a plain random matrix gives.
    loadings = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.3],
        ]
    )
    matrix = rng.normal(size=(24, 5)) @ loadings
    return PCAAnalyzer().analyze(matrix, n_components=3, method="correlation")


def _read_scores(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


# ---------------------------------------------------------------------------
# 1. The CSV is the result object
# ---------------------------------------------------------------------------


def test_exported_csv_matches_result_scores(pca_result, tmp_path):
    """The PC1/PC2 columns are the result's own scores, to float precision."""
    spec = RPlotSpec(output_format="pdf")
    export = RScriptExporter(stamp="test").export_pca_scores(
        pca_result,
        tmp_path,
        spec,
        labels=[f"T{i}" for i in range(len(pca_result.scores))],
        groups=["A"] * 12 + ["B"] * 12,
    )

    header, rows = _read_scores(export.data_path)
    assert header == ["sample", "PC1", "PC2", "group", "label"]
    assert len(rows) == len(pca_result.scores)

    expected = np.asarray(pca_result.get_scores(n_components=2), dtype=float)
    for i, row in enumerate(rows):
        assert int(row[0]) == i + 1
        assert float(row[1]) == pytest.approx(expected[i, 0], abs=1e-6)
        assert float(row[2]) == pytest.approx(expected[i, 1], abs=1e-6)
        assert row[3] == ("A" if i < 12 else "B")
        assert row[4] == f"T{i}"


def test_explained_variance_percentages_reach_the_script(pca_result, tmp_path):
    """Axis labels come from the result, not from a hard-coded number.

    ``PCAResult.explained_variance`` is ALREADY a percentage -- stats/pca.py
    multiplies by 100 before storing it -- so the script must use the value
    as-is. Multiplying again here would print "3469.2% variance" on the axis.
    """
    spec = RPlotSpec(output_format="pdf")
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, spec)
    text = export.script_path.read_text(encoding="utf-8")
    ev = np.asarray(pca_result.explained_variance, dtype=float)
    assert 0.0 < ev[0] <= 100.0
    assert f"PC1 ({ev[0]:.1f}% variance)" in text
    assert f"PC2 ({ev[1]:.1f}% variance)" in text


def test_scree_csv_matches_result(pca_result, tmp_path):
    spec = RPlotSpec(output_format="pdf")
    export = RScriptExporter(stamp="test").export_pca_scree(pca_result, tmp_path, spec)
    header, rows = _read_scores(export.data_path)
    assert header == ["component", "eigenvalue", "variance_pct", "cumulative_pct"]
    ev = np.asarray(pca_result.explained_variance, dtype=float)
    cum = np.asarray(pca_result.cumulative_variance, dtype=float)
    # One row per component, each matching the result's own numbers.
    assert len(rows) == ev.size
    for i, (row, value, running) in enumerate(zip(rows, ev, cum, strict=True)):
        assert row[0] == str(i + 1)
        assert float(row[2]) == pytest.approx(value, abs=1e-4)
        assert float(row[3]) == pytest.approx(running, abs=1e-4)


# ---------------------------------------------------------------------------
# More groups than the palette has colours
# ---------------------------------------------------------------------------


def test_more_groups_than_colours_never_index_past_the_palette(pca_result, tmp_path):
    """The bug this guards: 7+ groups and the points silently disappeared.

    The old line was

        values = PALETTE[seq_along(levels(factor(data$group)))]

    With a 9-colour palette and 9 groups that was fine, but PALETTE held 6
    entries, so groups 7-9 indexed to NA. ggplot then dropped those rows,
    printing "Removed 9 rows containing missing values" -- and still exited 0
    with a PDF on disk. A figure that quietly lost a third of its data is
    worse than one that failed.
    """
    n_groups = 9
    groups = [f"Horizon {i % n_groups}" for i in range(len(pca_result.scores))]
    export = RScriptExporter(stamp="test").export_pca_scores(
        pca_result, tmp_path, RPlotSpec(output_format="pdf"), groups=groups
    )
    script = export.script_path.read_text(encoding="utf-8")

    # No direct indexing into the palette by level count any more.
    assert "PALETTE[seq_along" not in script, (
        "the script indexes PALETTE by level count, which yields NA once the groups outnumber the colours"
    )
    assert "rep_len(PALETTE" in script
    assert "GROUP_COLOURS" in script and "GROUP_SHAPES" in script
    # And it says so out loud when the palette has to be recycled.
    assert "recycled" in script


def test_okabe_ito_is_the_default_palette(pca_result, tmp_path):
    """Colour-blind safe and greyscale-safe by default, not by luck.

    Okabe & Ito (2008) is the palette a biology figure is expected to use, and
    the previous default was a hand-picked set with no citation.
    """
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec(output_format="pdf"))
    script = export.script_path.read_text(encoding="utf-8")
    assert RPlotSpec().color_palette == "okabeito"
    assert 'PALETTE_NAME <- "okabeito"' in script
    assert "#E69F00" in script and "#009E73" in script  # Okabe-Ito orange/green
    # Every advertised palette must be selectable in the generated script.
    for name in ("okabeito", "dark2", "greyscale", "viridis"):
        assert f"\n  {name} =" in script or f"\n  {name} = " in script


def test_ellipses_are_computed_not_stated_by_ggplot(pca_result, tmp_path):
    """The ellipse must carry its group's colour.

    ``stat_ellipse`` does not preserve the colour aesthetic: naming it in the
    layer's aes draws every ellipse in one colour, and ggplot answers with
    "the following aesthetics were dropped during statistical transformation:
    colour". The figure then showed ellipses that did not match their points.
    """
    groups = [f"Horizon {i // 8}" for i in range(len(pca_result.scores))]
    export = RScriptExporter(stamp="test").export_pca_scores(
        pca_result, tmp_path, RPlotSpec(output_format="pdf"), groups=groups
    )
    script = export.script_path.read_text(encoding="utf-8")
    assert "stat_ellipse" not in script
    assert "geom_path" in script
    assert ".ellipse95" in script
    # The minimum is enforced where the ellipse is built, not only reported.
    assert "min_n = ELLIPSE_MIN_N" in script
    # The data-frame column is named, not positional: d[[1]] is `sample`.
    assert 'd[["PC1"]]' in script or 'd[["PC2"]]' in script


# ---------------------------------------------------------------------------
# 2. Generated scripts stay valid, and avoid the R traps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["pdf", "svg", "png"])
def test_generated_script_passes_structural_validation(pca_result, tmp_path, fmt):
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec(output_format=fmt))
    assert validate_r_script(export.script_path.read_text(encoding="utf-8")) == []


def test_no_leading_plus_continuations(pca_result, tmp_path):
    """A leading ``+`` is a unary plus in R, not a continuation.

    It parses, it runs, and it silently turns one layer into its own statement
    -- so this cannot be caught by ``validate_r_script`` and needs its own check.
    """
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec())
    text = export.script_path.read_text(encoding="utf-8")
    offenders = [line for line in text.splitlines() if re.match(r"^\s*\+", line)]
    assert offenders == []


def test_user_settings_reach_the_script(pca_result, tmp_path):
    spec = RPlotSpec(
        theme="bw",
        base_size=12.0,
        figsize=(8.0, 6.0),
        dpi=600,
        output_format="svg",
    )
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, spec)
    text = export.script_path.read_text(encoding="utf-8")
    assert "THEME <- theme_bw(base_size = 12)" in text
    assert 'ggsave("pca_scores.svg"' in text
    assert "width = 8, height = 6" in text
    assert "dpi = 600" in text


# ---------------------------------------------------------------------------
# 3. A hand-edited script is protected
# ---------------------------------------------------------------------------


def test_script_is_unmodified_tracks_a_hand_edit(pca_result, tmp_path):
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec())
    assert export.exists()
    assert export.script_is_unmodified() is True

    text = export.script_path.read_text(encoding="utf-8")
    export.script_path.write_text(text + "\n# my tweak\n", encoding="utf-8")
    assert export.script_is_unmodified() is False

    # Deleting it is also "not what we wrote".
    export.script_path.unlink()
    assert export.exists() is False
    assert export.script_is_unmodified() is False


def test_export_refuses_to_clobber_an_edited_script(pca_result, tmp_path, monkeypatch):
    """Answering "No" must leave the user's file byte-for-byte intact.

    The window here is a stub, not a real MainWindow. Those slots open modal
    dialogs, and a modal box in an offscreen run blocks forever -- the test
    would hang instead of failing, and a hang is worse than a failure because
    it just burns the CI timeout. Driving the real method body against a stub
    keeps the behaviour under test and removes the event loop entirely.
    """
    from PyQt6.QtWidgets import QMessageBox

    from views import ui_main_window as mw

    # Export first, then simulate the user editing the file: that is the state
    # the guard exists for. Writing the "user's" file first and exporting after
    # would leave it unmodified and the guard would never fire.
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec())
    script = export.script_path
    script.write_text(
        script.read_text(encoding="utf-8") + "\n# the user's own tweak\n",
        encoding="utf-8",
    )
    original = script.read_bytes()
    assert export.script_is_unmodified() is False

    # Pretend a previous export produced this file, then the user edited it.
    win = SimpleNamespace()
    win._r_output_dir = lambda: str(tmp_path)
    win._last_r_export = export
    win._logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    win._state = SimpleNamespace(has_data=True)
    win._run_r_script_and_show = lambda *a, **k: pytest.fail(
        "must not run anything after the user declines the overwrite"
    )

    asked = {}

    def _question(*args, **kwargs):
        asked["title"] = args[2]
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(mw.QMessageBox, "question", staticmethod(_question))
    monkeypatch.setattr(mw.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(mw.QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    mw.MainWindow._on_export_as_r_script(win)

    assert asked, "the user was never asked before overwriting an edit"
    assert script.read_bytes() == original, "the edited script was clobbered"


def test_rerun_never_regenerates_the_script(pca_result, tmp_path, monkeypatch):
    """ "Re-run" must execute the file on disk, not rewrite it.

    This is the whole reason the two actions are separate buttons.
    """
    from views import ui_main_window as mw

    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec())
    edited = export.script_path.read_text(encoding="utf-8") + "\n# edited\n"
    export.script_path.write_text(edited, encoding="utf-8")

    ran = {}
    win = SimpleNamespace()
    win._r_output_dir = lambda: str(tmp_path)
    win._logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    win._run_r_script_and_show = lambda path, title: ran.update(path=Path(path), title=title)

    monkeypatch.setattr(mw.QMessageBox, "information", staticmethod(lambda *a, **k: None))

    mw.MainWindow._on_rerun_r_script(win)

    assert ran["path"] == export.script_path
    assert export.script_path.read_text(encoding="utf-8") == edited, (
        "re-run rewrote the script; the user's edits would be destroyed"
    )


def test_rerun_without_a_script_reports_instead_of_guessing(tmp_path, monkeypatch):
    from views import ui_main_window as mw

    shown = {}

    def _info(parent, title, text):
        shown["title"] = title

    monkeypatch.setattr(mw.QMessageBox, "information", staticmethod(_info))

    win = SimpleNamespace()
    win._r_output_dir = lambda: str(tmp_path)
    win._logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    win._run_r_script_and_show = lambda *a, **k: pytest.fail("nothing exists to run")

    mw.MainWindow._on_rerun_r_script(win)
    assert "title" in shown


# ---------------------------------------------------------------------------
# 4. Finding figures: by diff, not by guessed name
# ---------------------------------------------------------------------------


def test_run_finds_a_renamed_output(monkeypatch, tmp_path):
    """A user who renames the output still gets the file reported.

    R is stubbed out here, so this asserts the diff logic alone: a file that
    existed before the run and was not touched is not reported, a file that
    appeared during the run is.
    """
    from visualization import r_render

    script = tmp_path / "user.R"
    script.write_text("p <- 1\n", encoding="utf-8")
    stale = tmp_path / "old_plot.pdf"
    stale.write_bytes(b"%PDF-1.4 stale")

    fake_r = tmp_path / "Rscript"
    fake_r.write_text("stub", encoding="utf-8")

    def _fake_run(argv, **kwargs):
        # The user renamed the output; nothing is called pca_scores.pdf.
        (tmp_path / "my_renamed_figure.pdf").write_bytes(b"%PDF-1.4 new")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(r_render.subprocess, "run", _fake_run)

    run = run_r_script(script, rscript=str(fake_r), make_preview=False)
    assert run.ok is True
    names = [p.name for p in run.produced]
    assert names == ["my_renamed_figure.pdf"]
    assert "old_plot.pdf" not in names, "a pre-existing untouched file was reported as produced by this run"


def test_run_reports_an_r_error_without_raising(monkeypatch, tmp_path):
    from visualization import r_render

    script = tmp_path / "broken.R"
    script.write_text("stop('boom')\n", encoding="utf-8")
    fake_r = tmp_path / "Rscript"
    fake_r.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        r_render.subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="Error: boom\n"),
    )

    run = run_r_script(script, rscript=str(fake_r), make_preview=False)
    assert run.ok is False
    assert "boom" in run.message()
    assert run.produced == []


def test_run_survives_an_r_script_that_prints_nothing(monkeypatch, tmp_path):
    """R can exit 0 while writing no figure. Say so; do not claim success+file."""
    from visualization import r_render

    script = tmp_path / "viewer_only.R"
    script.write_text("p <- 1\n", encoding="utf-8")
    fake_r = tmp_path / "Rscript"
    fake_r.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        r_render.subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    run = run_r_script(script, rscript=str(fake_r), make_preview=False)
    assert run.ok is True
    assert run.produced == []
    assert "no figure file appeared" in run.error


def test_missing_script_is_reported_not_raised(tmp_path):
    run = run_r_script(tmp_path / "nope.R", make_preview=False)
    assert run.ok is False
    assert "not found" in run.error.lower()


# ---------------------------------------------------------------------------
# 5. Locating R
# ---------------------------------------------------------------------------


def test_configured_path_wins(monkeypatch, tmp_path):
    target = tmp_path / "Rscript"
    target.write_text("stub", encoding="utf-8")
    monkeypatch.setattr("visualization.r_render.shutil.which", lambda name: None)
    assert find_rscript(str(target)) == target


def test_missing_r_explains_the_fix():
    text = describe_missing_r()
    assert "cran.r-project.org" in text
    assert "Preferences" in text
    # The generated script is useless to nobody even without R.
    assert "RStudio" in text


# ---------------------------------------------------------------------------
# 6. The dialog -> spec -> script chain
# ---------------------------------------------------------------------------


# The QApplication must outlive every widget built from it, and must NOT
# outlive this module. Two ways to get that wrong, both fatal:
#
#   * ``QApplication.instance() or QApplication([])`` as a plain local -- the
#     local dies, CPython collects the app, and the next widget construction
#     aborts the interpreter (0xC0000409) with the misleading message
#     "Must construct a QApplication before a QWidget" even though one WAS
#     created.
#   * holding it in a module-level global -- the app then survives until
#     interpreter shutdown, so it is torn down while widgets from other test
#     modules still exist. On CI that showed up as a "Fatal Python error:
#     Aborted" in views/diagnostic_console.py:479, when a late matplotlib font
#     warning reached a console QObject that had already been destroyed.
#
# A module-scoped fixture scopes the app to this module, which is the same
# arrangement tests/test_qss_supported_properties.py already uses.
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _r_plot_spec_from(values: dict):
    """Call the real ``_r_plot_spec`` without a full MainWindow.

    A stub rather than a real window: instantiating MainWindow pulls in the
    whole app, and these tests are about one small translation.
    """
    from views.ui_main_window import MainWindow

    win = SimpleNamespace()
    win._get_preferences_state = lambda: values
    return MainWindow._r_plot_spec(win)


def test_preferences_reach_the_script_through_the_real_spec(pca_result, tmp_path, qapp):
    """The dialog's values must survive the trip into the generated script.

    The preference keys (``r_theme``, ``r_base_size``, ...) and the
    :class:`RPlotSpec` field names (``theme``, ``base_size``, ...) are not the
    same strings, and the translation lives in ``_r_plot_spec``. Nothing else
    in the codebase connects those two, so a renamed key on either side would
    silently fall back to a default and produce a script that still runs --
    just not the one the user asked for.

    Every value here is deliberately NOT the default. With defaults on both
    sides, ``prefs.get("theme", "classic")`` and ``prefs.get("r_theme",
    "classic")`` return the same thing, so a wrongly-named key produces a
    passing test and a silently wrong script -- the exact failure this is
    meant to catch.
    """
    from views.ui_permutation_dialogs import PreferencesDialog

    chosen = {
        "r_rscript": r"D:\somewhere\Rscript.exe",
        "r_theme": "bw",
        "r_base_size": 13,
        "r_output_format": "svg",
        "r_timeout": 42,
    }
    dialog = PreferencesDialog(current=chosen)
    try:
        values = dialog.get_preferences()
    finally:
        dialog.deleteLater()

    for key, want in chosen.items():
        assert values[key] == want, f"the dialog lost {key}"

    spec = _r_plot_spec_from(values)

    assert spec.theme == chosen["r_theme"]
    assert spec.base_size == pytest.approx(float(chosen["r_base_size"]))
    assert spec.output_format == chosen["r_output_format"]
    assert spec.r_executable == chosen["r_rscript"]
    assert values["r_timeout"] == 42

    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, spec)
    text = export.script_path.read_text(encoding="utf-8")
    assert f"THEME <- theme_{spec.theme}(base_size = {spec.base_size:g})" in text
    assert f'ggsave("pca_scores.{spec.output_format}"' in text


def test_every_offered_theme_maps_to_a_ggplot2_theme(pca_result, tmp_path):
    """A choice in the dialog must not produce a script calling a missing function.

    ``theme_tufte()`` is not in ggplot2, so offering it would generate a script
    that fails at the first line of the plot body. ``RPlotSpec`` does not
    validate ``theme``, so this is where the two lists get reconciled.
    """
    from views.ui_permutation_dialogs import PreferencesDialog

    known = {"classic", "bw", "minimal", "grey", "void", "light", "dark", "default", "linedraw", "test", "ggplot2"}
    for theme in PreferencesDialog.R_THEMES:
        assert theme in known, f"'{theme}' is not a ggplot2 theme_*() function"
        spec = RPlotSpec(theme=theme)
        assert spec.theme_line() == f"THEME <- theme_{theme}(base_size = 9)"

    for fmt in PreferencesDialog.R_OUTPUT_FORMATS:
        assert fmt in {"pdf", "svg", "png"}, f"ggsave cannot write '{fmt}'"


# ---------------------------------------------------------------------------
# 7. Live R (skipped when R is absent)
# ---------------------------------------------------------------------------


# "R is installed" is NOT the same as "R can plot". The GitHub Windows and
# macOS runner images ship R but not ggplot2, so a skipif on find_rscript()
# alone let these tests run and fail there with "there is no package called
# 'ggplot2'". The precondition that actually matters is that the generated
# script can run, so that is what gets checked -- once, at import.
@functools.lru_cache(maxsize=1)
def _r_with_ggplot2() -> bool:
    exe = find_rscript()
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [str(exe), "-e", 'suppressMessages(library(ggplot2)); cat("ggplot2-ok")'],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "ggplot2-ok" in (proc.stdout or "")


needs_r = pytest.mark.skipif(not _r_with_ggplot2(), reason="Rscript with ggplot2 is not installed")


@needs_r
def test_real_r_runs_the_generated_script(pca_result, tmp_path):
    """The only test that would have caught the leading-``+`` trap.

    A script missing a ``+`` still exits 0 and still writes a file; the file is
    just wrong. So this asserts the figure really is a full-size raster with
    real content, not merely that R returned zero.
    """
    spec = RPlotSpec(output_format="png", dpi=110)
    export = RScriptExporter(stamp="test").export_pca_scores(
        pca_result,
        tmp_path,
        spec,
        groups=["A"] * 12 + ["B"] * 12,
    )

    run = run_r_script(export.script_path, make_preview=True)
    assert run.ok, run.message()

    figure = tmp_path / "pca_scores.png"
    assert figure.is_file(), f"R exited 0 but wrote no figure. {run.message()}"
    assert figure.stat().st_size > 5000, (
        f"figure is only {figure.stat().st_size} bytes -- a truncated plot still exits 0"
    )
    # The script already wrote a PNG, so no companion preview is needed. This
    # is the branch that avoids running the script twice.
    assert run.preview_png is None


@needs_r
def test_real_r_previews_a_vector_only_script(pca_result, tmp_path):
    """PDF output cannot be shown inline, so a PNG companion is generated.

    The companion ``source()``s the user's script and re-saves whatever plot it
    left behind, which is the only way a vector-only script can be displayed
    without assuming it still looks the way we generated it.
    """
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, RPlotSpec(output_format="pdf"))

    run = run_r_script(export.script_path, make_preview=True)
    assert run.ok, run.message()
    assert (tmp_path / "pca_scores.pdf").is_file()
    assert run.preview_png is not None, (
        "no PNG companion for a vector-only output; the user would have to open the PDF by hand"
    )
    assert run.preview_png.is_file()
    assert run.preview_png.stat().st_size > 5000
    # The companion script is scaffolding and must not be left behind.
    assert list(tmp_path.glob("*__paleoast_preview.R")) == []


@needs_r
def test_real_r_scree_script_renders(pca_result, tmp_path):
    spec = RPlotSpec(output_format="png", dpi=110)
    export = RScriptExporter(stamp="test").export_pca_scree(pca_result, tmp_path, spec)
    run = run_r_script(export.script_path, make_preview=True)
    assert run.ok, run.message()
    assert (tmp_path / "pca_scree.png").is_file()


@needs_r
def test_real_ggplot2_has_every_theme_the_dialog_offers(pca_result, tmp_path):
    """Ask the installed ggplot2, rather than trusting a hard-coded list.

    The static test pins the list against what ggplot2 has shipped historically.
    This one is authoritative for the R actually installed on this machine, and
    is the check that would fail if a theme is renamed upstream.
    """
    import subprocess

    from views.ui_permutation_dialogs import PreferencesDialog
    from visualization.r_render import find_rscript

    exe = find_rscript()
    assert exe is not None
    themes = ", ".join(f'"{t}"' for t in PreferencesDialog.R_THEMES)
    proc = subprocess.run(
        [
            str(exe),
            "-e",
            "suppressMessages(library(ggplot2));"
            f"cat(paste(sapply(c({themes}), "
            "function(t) paste0(t, '=', exists(paste0('theme_', t), "
            'mode="function"))), collapse=" "))',
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    found = dict(p.split("=", 1) for p in proc.stdout.split())
    missing = [t for t, ok in found.items() if ok != "TRUE"]
    assert missing == [], f"the dialog offers themes ggplot2 does not have: {missing}"

    # And the generated script must actually run with the chosen theme.
    spec = RPlotSpec(theme="bw", output_format="png", dpi=110)
    export = RScriptExporter(stamp="test").export_pca_scores(pca_result, tmp_path, spec)
    run = run_r_script(export.script_path, make_preview=False)
    assert run.ok, run.message()
    assert (tmp_path / "pca_scores.png").is_file()
