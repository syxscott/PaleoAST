"""Regression tests for the defects found by OPERATING the application.

These four were not found by reading code; they were found by launching
PaleoAST, loading a dataset from the example picker and opening
Preferences:

1. The bundled ``primate_traits`` example dataset could not be loaded at
   all -- "could not convert string to float: 'Catarrhini'".  The sibling
   ``community_abundance`` branch dropped its text columns and this one did
   not.  (The loader-level fix is covered in tests/data/test_examples.py;
   this file pins the wiring that reaches it.)

2. The dark theme had a complete implementation -- a palette, a stylesheet
   and a propagation chain through every widget -- and no way to reach it.
   The application always started light, and nothing in the UI called
   MainWindow.setDarkTheme().  Every dark-theme contrast defect in the
   review was therefore unreachable in the product.

3. The Preferences dialog read the language from "preferences/language"
   while the language menu wrote "language", so the field showed the
   default "en" even when the whole interface was running in Chinese.

4. The Preferences dialog's OK / Cancel buttons were Qt's own, whose
   translations are English-only, so a Chinese interface showed "OK" and
   "Cancel" there while every other button read "帮助" / "取消" / "运行".
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6", reason="UI tests require PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-live-findings-tests"])
    return app


@pytest.fixture
def isolated_settings(tmp_path):
    """Point QSettings at a sandbox ini so each test starts clean.

    The default format is put back afterwards, so a test that runs later
    in the same process is not left writing its settings into a
    directory that has already been deleted.
    """
    previous_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    try:
        yield
    finally:
        QSettings.setDefaultFormat(previous_format)


def _close_without_prompting(window, qapp) -> None:
    """Close a window without letting it raise a modal "save changes?" box.

    ``MainWindow.closeEvent`` asks the user to save before closing whenever
    the SHARED StateManager is modified, and it asks with a MODAL
    ``QMessageBox``.  A headless test run has nobody to answer it, so the
    test would hang instead of failing -- which is how this file managed to
    stop the whole suite rather than report a defect.

    These tests never load anything, so there is nothing to save.  The
    shared state is marked saved first, which is what keeps the dialog
    from appearing even if another test left the state dirty.
    """
    from models.state_manager import get_state_manager

    get_state_manager().mark_saved()
    window.close()
    window.deleteLater()
    qapp.processEvents()


# =============================================================================
# 1. the example datasets the picker offers must all load
# =============================================================================
class TestExampleDatasetsLoad:
    def test_primate_traits_loads_through_the_shared_seam(self):
        """The bundled example must not fail with a raw numpy ValueError."""
        from data.loader import load_example_matrix

        matrix = load_example_matrix("primate_traits")
        assert matrix.data.shape == (13, 8)
        assert "group" not in matrix.col_labels

    def test_the_ui_path_uses_the_shared_seam(self):
        """The view must not rebuild the matrix itself.

        It used to, and its version dropped only "species" while the
        community dataset's version dropped "site" AND "group" -- the
        "same function, only one of them fixed" shape.
        """
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "views" / "ui_main_window.py").read_text(encoding="utf-8")
        assert "load_example_matrix(name)" in source, (
            "the view still builds the example matrix inline instead of using the shared seam"
        )
        # And it must not convert a frame with text columns to float itself.
        assert "to_numpy(dtype=float)" not in source, (
            "the view still calls to_numpy(dtype=float) on a frame that may carry text columns"
        )

    def test_every_matrix_example_loads_through_the_view(self, qapp):
        """The whole path, not just the loader.

        The picker was the one part of this flow with no test at all, and
        it was broken: it called ``QInputDialog.setCurrentIndex``, which
        is a ``QComboBox`` method and does not exist on ``QInputDialog``
        (and the ``InputMode.ComboBox`` it also used is not exposed by
        PyQt6 at all).  The example picker therefore raised an
        ``AttributeError`` the first time a user clicked "Load Example
        Data" -- and nothing in the suite noticed, because nothing in the
        suite ever opened it.
        """
        from data.loader import list_example_datasets
        from views.ui_main_window import MainWindow

        window = MainWindow()
        try:
            for dataset in list_example_datasets():
                name = dataset["name"]
                if name == "primate_tree":
                    continue  # a phylogeny, not a matrix -- it opens a text tab
                window._load_example_dataset(name)
                matrix = window._state.data_matrix
                assert matrix is not None, f"{name} did not load through the view"
                assert matrix.data.size > 0, f"{name} produced an empty matrix"
                assert "group" not in matrix.col_labels, f"{name} kept its text column in the matrix"
        finally:
            _close_without_prompting(window, qapp)

    def test_the_example_picker_is_built_with_translated_buttons(self, qapp):
        """The picker must be constructible, and its buttons translated.

        The construction is checked without opening the modal dialog --
        what is being pinned is that the dialog can be built at all, and
        that its OK / Cancel are the translated labels rather than Qt's
        English-only ones.
        """
        from PyQt6.QtWidgets import QDialogButtonBox

        from config.i18n import _
        from views.ui_main_window import MainWindow

        window = MainWindow()
        try:
            dialog, combo = window._build_example_picker(["first — a", "second — b"])
            assert combo.count() == 2
            assert combo.currentText() == "first — a"
            labels = {button.text() for box in dialog.findChildren(QDialogButtonBox) for button in box.buttons()}
            assert _("OK") in labels, f"the picker's OK button is not translated: {labels}"
            assert _("Cancel") in labels, f"the picker's Cancel button is not translated: {labels}"
        finally:
            _close_without_prompting(window, qapp)


# =============================================================================
# 2. the dark theme has to be reachable
# =============================================================================
class TestDarkThemeIsReachable:
    def test_preferences_offers_a_dark_theme_switch(self, qapp, isolated_settings):
        from views.ui_permutation_dialogs import PreferencesDialog

        dialog = PreferencesDialog(current={"dark_theme": True})
        assert hasattr(dialog, "_dark_theme_check"), "no dark-theme control in Preferences"
        assert dialog._dark_theme_check.isChecked() is True
        assert dialog.get_preferences()["dark_theme"] is True

    def test_applying_the_theme_switches_the_main_window(self, qapp, isolated_settings):
        from views.ui_main_window import MainWindow

        window = MainWindow()
        try:
            assert window._is_dark_theme is False
            window.apply_theme(True)
            assert window._is_dark_theme is True, "apply_theme did not switch the window"
            # The application-wide stylesheet has to change too, otherwise the
            # chrome outside the window stays light.
            stylesheet = QApplication.instance().styleSheet()
            assert stylesheet, "the application stylesheet was not set"
        finally:
            _close_without_prompting(window, qapp)

    def test_the_theme_is_restored_from_the_settings_at_startup(self, qapp, isolated_settings):
        """A persisted dark-theme preference must be honoured on the next start."""
        settings = QSettings("PaleoAST", "PaleoAST")
        settings.setValue("preferences/dark_theme", True)
        settings.sync()

        from main import PaleoASTApplication

        assert PaleoASTApplication._read_dark_theme_preference() is True

        settings.setValue("preferences/dark_theme", False)
        settings.sync()
        assert PaleoASTApplication._read_dark_theme_preference() is False

    def test_the_dark_palette_is_reachable_through_the_stylesheet(self):
        """The dark stylesheet must actually be the dark one."""
        from config.design_system import ColorPalette, ColorPaletteDark, get_stylesheet

        dark = get_stylesheet(True)
        light = get_stylesheet(False)
        assert dark != light, "the dark and light stylesheets are identical"
        # A colour that only the dark palette defines.
        assert ColorPaletteDark.bg_secondary in dark
        assert ColorPaletteDark.bg_secondary not in light
        assert ColorPalette.bg_secondary in light

    def test_a_partial_preferences_write_does_not_turn_the_theme_off(self, qapp, isolated_settings):
        """A caller that says nothing about the theme must not turn it off.

        ``_apply_preferences`` is deliberately callable with a partial
        state -- it writes the keys it is given and leaves the rest alone.
        Writing a *missing* key as its default meant that every partial
        write reset the stored dark theme back to light, so running the
        test suite silently undid the choice the user made in the
        Preferences dialog.  This is exactly how the theme preference was
        found to be "false" on a machine where the user had chosen dark.
        """
        from views.ui_main_window import MainWindow

        settings = QSettings("PaleoAST", "PaleoAST")
        settings.setValue("preferences/dark_theme", True)
        settings.sync()

        state = MainWindow._get_preferences_state(None)
        assert state["dark_theme"] is True, "the stored theme was not read back"

        partial = {key: value for key, value in state.items() if key != "dark_theme"}
        MainWindow._apply_preferences(None, partial)

        assert settings.value("preferences/dark_theme", False, type=bool) is True, (
            "a partial preferences write turned the stored dark theme off"
        )


# =============================================================================
# 3. the language field must show the language that is actually in force
# =============================================================================
class TestPreferencesLanguageField:
    def test_the_language_field_reads_the_key_the_menu_writes(self, qapp, isolated_settings):
        from views.ui_main_window import MainWindow

        window = MainWindow()
        try:
            settings = QSettings("PaleoAST", "PaleoAST")
            settings.setValue("language", "zh")
            settings.sync()
            state = window._get_preferences_state()
            assert state["language"] == "zh", "the preferences field does not show the language that is in force"
        finally:
            _close_without_prompting(window, qapp)

    def test_the_two_language_keys_cannot_drift_apart(self):
        """One key, one meaning.

        "language" is what the Language menu writes and what the running
        interface uses.  "preferences/language" is a mirror.  The dialog has
        to read the first, and a change made in the dialog has to reach the
        first.
        """
        import inspect

        from views.ui_main_window import MainWindow

        text = inspect.getsource(MainWindow._get_preferences_state)
        assert 'settings.value("language"' in text, "the preferences state must read the authoritative 'language' key"


# =============================================================================
# 4. the Preferences buttons must be translated
# =============================================================================
class TestPreferencesButtonsAreTranslated:
    def test_ok_and_cancel_use_the_translated_labels(self, qapp, isolated_settings):
        from PyQt6.QtWidgets import QDialogButtonBox

        from config.i18n import _
        from views.ui_permutation_dialogs import PreferencesDialog

        dialog = PreferencesDialog()
        try:
            boxes = dialog.findChildren(QDialogButtonBox)
            assert boxes, "no button box in the preferences dialog"
            labels = {button.text() for box in boxes for button in box.buttons()}
            assert _("OK") in labels, f"the OK button is not translated: {labels}"
            assert _("Cancel") in labels, f"the Cancel button is not translated: {labels}"
        finally:
            dialog.close()
            dialog.deleteLater()
            qapp.processEvents()

    def test_the_translated_labels_exist_in_both_catalogs(self):
        from config.i18n import translations_en, translations_zh

        for key in ("OK", "Cancel", "Dark theme"):
            assert key in translations_zh.TRANSLATIONS, f"no Chinese entry for {key!r}"
            assert key in translations_en.TRANSLATIONS, f"no English entry for {key!r}"


# =============================================================================
# 5. the dark theme has to reach the figures the user makes
# =============================================================================
class TestDarkThemeReachesNewFigures:
    def test_a_figure_built_after_the_theme_is_dark_too(self, qapp):
        """A canvas is built fresh for every analysis.

        It is therefore never a child of the window at the moment the
        theme is applied, and the propagation in ``setDarkTheme`` only
        reaches the plots that already existed.  Every figure the user
        produced in the dark theme came out on a white background -- a
        bright rectangle in a dark application -- until the workspace
        told the new plot which theme was in force.

        Found by running the application: the theme was dark, the PCA
        score plot was light.
        """
        import matplotlib.pyplot as plt

        from views.ui_main_window import MainWindow
        from views.ui_plot_canvas import InteractivePlotCanvas

        # The canvas themes matplotlib globally, so the global settings are
        # restored afterwards.  Left dark, every figure built by another
        # module in the rest of the session comes out dark too, and the
        # tests that assert on a light "paper" figure start failing for
        # reasons that have nothing to do with what they are testing.
        before = plt.rcParams.copy()
        window = MainWindow()
        try:
            window._is_dark_theme = True
            plot = InteractivePlotCanvas()
            window._add_plot_to_workspace(plot, "test figure")
            assert plot._is_dark_theme is True, (
                "a figure created while the dark theme was in force was not told about it"
            )
        finally:
            plt.rcParams.update(before)
            _close_without_prompting(window, qapp)


# =============================================================================
# the example picker's own description of its datasets
# =============================================================================
def test_every_offered_example_dataset_has_a_working_loader():
    """Whatever the picker offers has to load.

    The picker advertised four datasets; two of them carried a text column
    that the matrix builder could not digest.
    """
    from data.loader import list_example_datasets, load_example_matrix

    for dataset in list_example_datasets():
        name = dataset["name"]
        if name == "primate_tree":
            # A phylogeny, not a matrix -- it is rendered as text.
            from data.loader import load_primate_tree

            assert load_primate_tree().leaf_count == 13
            continue
        matrix = load_example_matrix(name)
        assert matrix.data.size > 0, f"{name} produced an empty matrix"
        assert os.path.basename(matrix.col_labels[0]) or matrix.col_labels
