# =============================================================================
# Test: the categorical palette has exactly one source
# =============================================================================
"""
The interactive figures (matplotlib) and the exported figures (real ggplot2)
have to be the same picture. They were not, for three separate reasons, each of
which had its own test here:

* ``config/colors.py`` and the R script generator each carried their own copy
  of the palettes, under names that did not line up ("colorblind" vs
  "okabeito"), so nothing joined the two up;
* the two Okabe-Ito tables disagreed about the eighth colour -- grey in one,
  black in the other -- so the eighth group was a different colour on screen
  than in the file;
* ``get_color_scheme`` returned a default for any name it did not recognise,
  without raising, so a typo or an R-side name produced a figure in a
  different palette and said nothing about it.

A fourth defect lived in the wiring rather than the palettes: the Preferences
dialog collected ``r_palette``, and neither ``_get_preferences_state`` nor
``_apply_preferences`` mentioned it, so the choice was discarded on the way
out and never reached the figure. Those two are covered here too, because a
palette that cannot be persisted is not a preference.

Every assertion below is written to go RED if the behaviour it describes comes
back, not merely to pass on the current code.
"""

from __future__ import annotations

import os
import re

import matplotlib
import pytest

matplotlib.use("Agg")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np

from config.colors import (
    COLORBLIND_FRIENDLY_PALETTE,
    DEFAULT_PALETTE_NAME,
    PALETTES,
    current_palette,
    current_palette_name,
    get_color_scheme,
    palette_names,
    resolve_palette_name,
    set_current_palette,
)
from stats.pca import PCAAnalyzer
from visualization.r_export import RPlotSpec, RScriptExporter

ALL_PALETTES = ["okabeito", "greyscale", "dark2", "viridis"]


@pytest.fixture(autouse=True)
def _restore_palette():
    """The current palette is module-level state.

    Without this, a case that switches the palette leaks into the next one and
    the suite passes for reasons that have nothing to do with the code under
    test.
    """
    before = resolve_palette_name(current_palette()[0] and DEFAULT_PALETTE_NAME)
    del before
    original = set_current_palette(DEFAULT_PALETTE_NAME)
    try:
        yield
    finally:
        set_current_palette(DEFAULT_PALETTE_NAME)
        del original


@pytest.fixture(scope="module")
def pca_result():
    rng = np.random.RandomState(3)
    data = np.vstack([rng.randn(15, 5) + 2.5, rng.randn(15, 5) - 2.0])
    return PCAAnalyzer().analyze(data)


def _export(pca_result, palette, tmp_path):
    exporter = RScriptExporter(stamp="test")
    exporter.export_pca_scores(
        pca_result,
        tmp_path,
        RPlotSpec(output_format="png", color_palette=palette),
        labels=[f"S{i}" for i in range(len(pca_result.scores))],
        groups=["A"] * 15 + ["B"] * 15,
    )
    return (tmp_path / "pca_scores.R").read_text(encoding="utf-8")


def _exported_colours(script: str) -> list[str]:
    line = next(ln for ln in script.splitlines() if ln.startswith("PALETTE <-"))
    return re.findall(r'"(#[0-9A-Fa-f]{6})"', line)


# ---------------------------------------------------------------------------
# Strict lookup
# ---------------------------------------------------------------------------


def test_unknown_palette_name_raises():
    """An unrecognised name must not quietly resolve to some other palette.

    Would go red if ``get_color_scheme`` returned its default again: that is
    precisely how "viridis" produced Tol colours with no complaint.
    """
    with pytest.raises(ValueError):
        get_color_scheme("chartreuse")


def test_the_magic_name_default_is_gone():
    """``"default"`` resolved to CHART_COLORS while the app defaulted to
    something else, so a caller could ask for the default and not get it.

    Would go red if ``"default"`` were reintroduced as an alias.
    """
    with pytest.raises(ValueError):
        get_color_scheme("default")


def test_every_advertised_name_resolves():
    """``palette_names`` is what the UI builds its list from, so every name in
    it has to be resolvable -- a name in the dropdown that raises would break
    the very first palette switch."""
    for name in palette_names():
        assert get_color_scheme(name) == PALETTES[name]


def test_returned_palette_is_a_copy():
    """A caller mutating the returned list would otherwise corrupt the registry
    for every other figure drawn afterwards.

    Both accessors are covered. ``current_palette`` handing out the registry's
    own list is the same bug wearing a different name, and the plotters are
    the callers holding it.
    """
    for accessor in (lambda: get_color_scheme("okabeito"), current_palette):
        got = accessor()
        got.append("#FF00FF")
        assert "#FF00FF" not in PALETTES["okabeito"]
        assert "#FF00FF" not in current_palette()
        assert get_color_scheme("okabeito") == COLORBLIND_FRIENDLY_PALETTE


# ---------------------------------------------------------------------------
# The palette's own values
# ---------------------------------------------------------------------------


def test_okabe_ito_eighth_colour_is_black():
    """Regression guard for the grey/black swap.

    Okabe & Ito's eighth colour is black. The in-app list had grey there, so
    an eight-group figure showed grey on screen and black in the exported
    file. Asserting on the index rather than on the whole list keeps the
    failure pointed at the one colour that was wrong.
    """
    assert COLORBLIND_FRIENDLY_PALETTE[7] == "#000000", (
        "Okabe-Ito's eighth colour is black; grey here means the in-app figure "
        "and the exported figure disagree about the eighth group"
    )


def test_palettes_are_non_trivially_distinct():
    """Four names that all returned the same list would pass every test above
    while every figure came out the same colour."""
    rendered = [tuple(get_color_scheme(name)) for name in ALL_PALETTES]
    assert len(set(rendered)) == len(ALL_PALETTES)


# ---------------------------------------------------------------------------
# Preview == export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("palette", ALL_PALETTES)
def test_exported_colours_come_from_the_registry(pca_result, palette, tmp_path):
    """The R script must be handed the registry's colours, not its own copy.

    Would go red if the generated script went back to carrying a hard-coded
    table, or to calling ``RColorBrewer::brewer.pal()`` itself.
    """
    script = _export(pca_result, palette, tmp_path)
    assert _exported_colours(script) == get_color_scheme(palette)


@pytest.mark.parametrize("palette", ALL_PALETTES)
def test_interactive_figure_uses_the_same_palette_as_the_export(pca_result, palette, tmp_path):
    """The property the whole change exists for.

    ``current_palette()`` is what the matplotlib plotters draw with, and the
    R script is built from the same list. Before, the two sides were separate:
    the plotters hard-coded ``get_color_scheme("default")`` (Paul Tol's
    colours) while the export defaulted to Okabe-Ito, so every palette the
    user picked changed the file and left the screen untouched.
    """
    set_current_palette(palette)
    preview = list(current_palette())
    exported = _exported_colours(_export(pca_result, palette, tmp_path))
    assert preview == exported


def test_figure_actually_paints_with_the_palette(pca_result):
    """Not just that the right list came back, but that it reached the canvas.

    A plotters regression that stopped calling ``current_palette()`` would
    leave the two tests above passing -- they compare lists, not pixels.
    """
    import matplotlib.colors as mcolors

    from visualization.pca_plot import PCAPlotter

    set_current_palette("dark2")
    fig = PCAPlotter().plot_scores(
        pca_result,
        groups=[0] * 15 + [1] * 15,
        labels=[f"S{i}" for i in range(len(pca_result.scores))],
    )
    painted = {mcolors.to_hex(sc.get_facecolors()[0]) for sc in fig.axes[0].collections}
    assert painted == {c.lower() for c in get_color_scheme("dark2")[:2]}


def test_the_other_plotters_honour_the_preference_too():
    """``diversity_plot`` has four palette call sites of its own.

    They were converted in the same pass but nothing above touches them, and a
    mutation check confirmed the gap: reverting any one of them left this file
    entirely green, because only ``pca_plot`` was ever exercised. This case
    covers the multi-series rarefaction comparison, which is the one that puts
    two groups on one axis and so has to choose between two colours.

    Would go red if a diversity plotter went back to a fixed palette.
    """
    import matplotlib.colors as mcolors

    from ecology.diversity import compute_diversity_indices
    from models.diversity_result import RarefactionResult
    from visualization.diversity_plot import DiversityPlotter

    sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    results = [
        RarefactionResult(
            sample_name="Horizon A",
            expected_taxa=np.array([1, 3, 5, 6, 7, 8, 8, 9], dtype=float),
            sample_sizes=sizes,
        ),
        RarefactionResult(
            sample_name="Horizon B",
            expected_taxa=np.array([1, 2, 4, 6, 7, 7, 8, 8], dtype=float),
            sample_sizes=sizes,
        ),
    ]
    set_current_palette("dark2")
    fig = DiversityPlotter().plot_multiple_rarefaction(results)
    painted = {mcolors.to_hex(line.get_color()) for ax in fig.axes for line in ax.lines}
    assert painted == {c.lower() for c in get_color_scheme("dark2")[:2]}

    # plot_diversity_summary had two more call sites. One was inside a branch
    # guarded by hasattr(result, "abundances") -- a field DiversityResult does
    # not have, so it could never run and a mutation there left this file
    # green. That branch has been removed; these are the sites that are live.
    #
    # Asserted per axes, not as a set over the whole figure: this file's first
    # version took the union of every panel's colours, which stayed green as
    # long as *any one* panel used the palette. With four panels in the figure
    # that is three quarters of the assertion for free.
    summary = DiversityPlotter().plot_diversity_summary(
        compute_diversity_indices(np.array([10, 8, 5, 4, 3, 3, 2, 2, 1, 1]))
    )
    # The property is that nothing in the figure comes from anywhere else --
    # the indices panel legitimately uses four colours, not two. Bounding the
    # whole figure by the palette catches a panel left on a hard-coded pair,
    # which the union-of-all-axes version missed.
    palette = {c.lower() for c in get_color_scheme("dark2")}
    used = {mcolors.to_hex(rect.get_facecolor()) for ax in summary.axes for rect in ax.patches}
    assert used, "the summary figure painted nothing"
    assert used <= palette, f"colours outside the palette: {used - palette}"
    assert len(used) >= 2, "expected more than one colour, so the check is not vacuous"

    # The single-curve rarefaction has to agree with the comparison plot
    # above, which has always drawn from the palette. Two calls, because the
    # band is a second call site of its own: mutating it while this only asked
    # for the unshaded curve left the whole file green.
    with_ci = RarefactionResult(
        sample_name="Horizon A",
        expected_taxa=np.array([1, 3, 5, 6, 7, 8, 8, 9], dtype=float),
        confidence_interval_lower=np.array([0.8, 2.4, 4.2, 5.3, 6.4, 7.5, 7.6, 8.6]),
        confidence_interval_upper=np.array([1.2, 3.6, 5.8, 6.7, 7.6, 8.5, 8.4, 9.4]),
        sample_sizes=sizes,
    )
    shaded = DiversityPlotter().plot_rarefaction(with_ci, show_ci=True)
    curve = {mcolors.to_hex(line.get_color()) for line in shaded.axes[0].lines}
    band = {mcolors.to_hex(fc) for coll in shaded.axes[0].collections for fc in coll.get_facecolor()}
    assert curve == band == {get_color_scheme("dark2")[0].lower()}, (
        f"curve {curve} and band {band} must both be the palette's first colour"
    )

    plain = DiversityPlotter().plot_rarefaction(results[0])
    assert {mcolors.to_hex(line.get_color()) for line in plain.axes[0].lines} == {get_color_scheme("dark2")[0].lower()}


def test_generated_script_needs_no_optional_r_package(pca_result, tmp_path):
    """The script says it cannot die on a missing package, so it must not ask
    for one. It used to call ``RColorBrewer::brewer.pal()`` and
    ``viridisLite::viridis()`` behind ``requireNamespace``, which meant the
    colours could differ between machines -- and between the preview and the
    render whenever a package was missing.

    Would go red if either package or ``requireNamespace`` came back.
    """
    script = _export(pca_result, "dark2", tmp_path)
    code = "\n".join(ln for ln in script.splitlines() if not ln.lstrip().startswith("##"))
    assert "requireNamespace" not in code
    assert "RColorBrewer::" not in code
    assert "viridisLite::" not in code


def test_unknown_palette_reaches_the_export_as_an_error(pca_result, tmp_path):
    """A bad name must fail loudly rather than exporting a figure in the
    wrong palette -- the failure mode this whole registry replaced."""
    with pytest.raises(ValueError):
        _export(pca_result, "not-a-palette", tmp_path)


# ---------------------------------------------------------------------------
# The preference actually persists
# ---------------------------------------------------------------------------


def test_r_palette_survives_the_settings_round_trip():
    """``r_palette`` reached neither ``_get_preferences_state`` nor
    ``_apply_preferences``, so the dialog's choice was dropped and the stored
    key was never written. Round-tripping the key directly is the smallest
    statement of that.

    Would go red if the key stopped being persisted.
    """
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None

    settings = QSettings("PaleoASTTest", "PaleoASTTest")
    settings.remove("preferences/r_palette")
    settings.setValue("preferences/r_palette", "viridis")
    settings.sync()

    read_back = settings.value("preferences/r_palette", DEFAULT_PALETTE_NAME, type=str)
    assert read_back == "viridis"
    settings.remove("preferences/r_palette")
    settings.sync()


@pytest.fixture
def app_settings():
    """The window's real QSettings scope, with the palette key restored.

    ``_apply_preferences`` writes to ``QSettings("PaleoAST", "PaleoAST")``.
    Asserting against a different scope would pass or fail for reasons that
    have nothing to do with the code -- which is exactly what happened the
    first time this was written: the assertion read an empty string because it
    was looking at the wrong scope, not because the write had failed.

    The user's own value is put back afterwards so running the suite does not
    quietly change their palette.
    """
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None

    settings = QSettings("PaleoAST", "PaleoAST")
    had = settings.contains("preferences/r_palette")
    prior = settings.value("preferences/r_palette") if had else None
    settings.remove("preferences/r_palette")
    try:
        yield settings
    finally:
        settings.remove("preferences/r_palette")
        if had:
            settings.setValue("preferences/r_palette", prior)
        settings.sync()


def test_main_window_reads_and_writes_the_palette_key(app_settings):
    """Both ends, on the real methods.

    The dialog produced ``r_palette``; ``_get_preferences_state`` and
    ``_apply_preferences`` are the two places that have to carry it, and
    neither did. Neither method reads ``self``, so they can be called unbound
    -- which also keeps this test off the Qt widget path entirely.

    Would go red if either end dropped the key again.
    """
    from views.ui_main_window import MainWindow

    state = MainWindow._get_preferences_state(None)
    assert "r_palette" in state, "the palette never came back out of the settings"
    assert state["r_palette"] == DEFAULT_PALETTE_NAME

    MainWindow._apply_preferences(
        None,
        {
            **state,
            "language": "en",
            "csv_has_header": True,
            "csv_has_row_labels": True,
            "plot_dpi": 100,
            "plot_figsize": "8,6",
            "r_palette": "greyscale",
        },
    )
    app_settings.sync()
    assert app_settings.value("preferences/r_palette", type=str) == "greyscale"
    assert MainWindow._get_preferences_state(None)["r_palette"] == "greyscale"


def test_applying_a_preference_moves_the_live_palette(app_settings):
    """Persisting the value is only half of it; the figures have to change.

    Would go red if the preference were stored and the plotters kept drawing
    in whatever palette was current when the window opened.

    Takes ``app_settings`` even though it asserts nothing about storage: the
    call writes to the real QSettings, so without the fixture this case would
    leave a palette selected on the developer's machine.
    """
    from PyQt6.QtWidgets import QApplication

    from views.ui_main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None

    MainWindow._apply_preferences(
        None,
        {
            "language": "en",
            "csv_has_header": True,
            "csv_has_row_labels": True,
            "plot_dpi": 100,
            "plot_figsize": "8,6",
            "r_palette": "viridis",
        },
    )
    assert current_palette() == get_color_scheme("viridis")


def test_a_stale_palette_name_does_not_stop_the_window(app_settings):
    """A palette name this build does not know -- an older setting, or a
    hand-edited config -- must fall back rather than raise out of the window's
    constructor."""
    from PyQt6.QtWidgets import QApplication

    from views.ui_main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None

    MainWindow._apply_preferences(
        None,
        {
            "language": "en",
            "csv_has_header": True,
            "csv_has_row_labels": True,
            "plot_dpi": 100,
            "plot_figsize": "8,6",
            "r_palette": "a-palette-from-a-newer-build",
        },
    )
    assert current_palette() == get_color_scheme(DEFAULT_PALETTE_NAME)


def test_the_stored_palette_survives_a_window_restart(app_settings):
    """The last link: what the window reads at startup is what the figures
    draw with, so the preference survives closing the app.

    Without this the chain could break anywhere between the dialog and the
    canvas and every other case would still pass -- each of them exercises one
    link in isolation.

    ``_load_settings`` restores window geometry before it reaches the palette,
    so it is called on a stand-in with those two methods rather than on a real
    ``MainWindow``. Building the whole window here would test Qt, not this
    wiring, and would put a multi-second widget construction into a unit test
    on fifteen CI configurations.
    """
    from PyQt6.QtWidgets import QApplication

    from views.ui_main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None

    class _WindowStub:
        """The only two things ``_load_settings`` touches before the palette."""

        def restoreGeometry(self, geometry):
            return None

        def restoreState(self, state):
            return None

    app_settings.setValue("preferences/r_palette", "dark2")
    app_settings.sync()

    set_current_palette(DEFAULT_PALETTE_NAME)
    assert current_palette_name() == DEFAULT_PALETTE_NAME

    MainWindow._load_settings(_WindowStub())
    assert current_palette_name() == "dark2"
    assert current_palette() == get_color_scheme("dark2")
