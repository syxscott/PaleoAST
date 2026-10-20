"""Regression tests for the shared-state leak that stopped the test suite.

The symptom was a suite that HUNG rather than failed, and it cost a full
debugging round to find, so the mechanism is worth writing down.

``StateManager`` is a process-wide singleton.  Nothing reset it between
tests, so a matrix one test loaded stayed loaded for every test that ran
after it -- and stayed flagged as *modified*.

``MainWindow.closeEvent`` asks the user to save before closing whenever
that shared state is modified, and it asks with a MODAL ``QMessageBox``.
A headless test run has nobody to answer it, so the run stops dead
instead of failing.  ``tests/test_pca_dialog_entry_point.py`` loads a
matrix and leaves it dirty; every later test that closed a real main
window then waited forever on a dialog nobody could see.

Three things are pinned here:

1. the shared state is reset around every test (tests/conftest.py);
2. a window closed in one test does not inherit another test's data, so
   the close does not raise a modal dialog;
3. closing a window with nothing to save does not ask to save at all.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="UI tests require PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-shared-state-tests"])
    return app


@pytest.fixture(autouse=True)
def _isolate_qsettings(tmp_path):
    """Keep the test off the developer's own QSettings scope.

    ``MainWindow.close()`` saves the window geometry, so without this the
    suite would quietly rewrite the geometry of the machine it runs on.

    The default format is put back afterwards.  Leaving it pointing at
    the ini sandbox would send every later test's ``QSettings`` writes
    into a directory that no longer exists, which is the same class of
    leak as the shared state this file is about.
    """
    previous_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    try:
        yield
    finally:
        QSettings.setDefaultFormat(previous_format)


class _RecordingMessageBox(QMessageBox):
    """A ``QMessageBox`` that records the modal questions instead of asking.

    Stubbing the question is what turns the original hang into a failure.
    Without the stub the run would stop dead, which is the very behaviour
    this file is about.  ``Cancel`` is the reply, so a code path that
    reacts to it never reaches a file dialog either.
    """

    asked: list = []

    @staticmethod
    def question(*args, **kwargs):
        _RecordingMessageBox.asked.append(args)
        return QMessageBox.StandardButton.Cancel

    @staticmethod
    def warning(*args, **kwargs):
        return QMessageBox.StandardButton.Ok


# Set by the first test so the second one can tell that the file's tests
# ran in the order they are written.
_DIRTIED = False


# =============================================================================
# 1. the shared state does not leak between tests
# =============================================================================
def test_a_test_can_leave_the_shared_state_dirty():
    """A test that loads a matrix leaves the shared state modified."""
    global _DIRTIED

    from models.data_matrix import DataMatrix
    from models.state_manager import get_state_manager

    state = get_state_manager()
    state.set_data_matrix(DataMatrix(np.ones((3, 3))))
    assert state.has_data
    assert state.is_modified
    _DIRTIED = True


def test_the_shared_state_does_not_leak_into_the_next_test():
    """The state the previous test left behind is gone.

    This is the isolation in ``tests/conftest.py``.  It is what keeps a
    later ``MainWindow.close()`` from raising a modal "save changes?" box.
    """
    if not _DIRTIED:
        pytest.skip("needs the preceding test in this file to have run")

    from models.state_manager import get_state_manager

    state = get_state_manager()
    assert not state.has_data, "the shared state leaked from an earlier test"
    assert not state.is_modified, "the shared state stayed flagged as modified"


# =============================================================================
# 2. closing a window does not inherit another test's data
# =============================================================================
def test_closing_a_window_after_another_test_loaded_data_does_not_prompt(qapp, monkeypatch):
    """The original hang, reproduced with the dialog stubbed out.

    The test before this one in the file loads a matrix.  Without the
    shared-state isolation the window would find that matrix still loaded
    and still modified, and would ask to save it -- a modal dialog that
    blocks a headless run forever.  The stub turns that into a failure
    with a message, which is the only useful way for it to show up.
    """
    import views.ui_main_window as m

    _RecordingMessageBox.asked.clear()
    monkeypatch.setattr(m, "QMessageBox", _RecordingMessageBox)

    window = m.MainWindow()
    try:
        window.close()
    finally:
        window.deleteLater()
        qapp.processEvents()

    assert not _RecordingMessageBox.asked, (
        "closing a window asked to save data it never touched; in a headless run that modal dialog blocks forever"
    )


# =============================================================================
# 3. nothing to save means nothing to ask about
# =============================================================================
def test_closing_a_window_with_nothing_to_save_does_not_ask(qapp, monkeypatch):
    """No data loaded, so there is nothing to save and nothing to ask.

    The predicate used to be ``is_modified`` alone.  With no data loaded
    the prompt was not merely pointless, it was a dead end: "Save" was
    answered with "No data to save", which returned False, which made the
    window ignore the close event -- so the window refused to close and
    the user had to answer "Discard" to get out.
    """
    import views.ui_main_window as m
    from models.data_matrix import DataMatrix
    from models.state_manager import get_state_manager

    state = get_state_manager()
    state.set_data_matrix(DataMatrix(np.ones((3, 3))))
    state.clear_data()  # modified, but nothing left to save
    assert state.is_modified
    assert not state.has_data

    _RecordingMessageBox.asked.clear()
    monkeypatch.setattr(m, "QMessageBox", _RecordingMessageBox)

    window = m.MainWindow()
    try:
        window.close()
    finally:
        window.deleteLater()
        qapp.processEvents()

    assert not _RecordingMessageBox.asked, "the window asked to save data that does not exist"


def test_a_window_with_unsaved_data_still_asks(qapp, monkeypatch):
    """The prompt is not removed -- it is made correct.

    With data loaded and modified, the user still has to be asked, or
    closing the application would throw away their work.
    """
    import views.ui_main_window as m
    from models.data_matrix import DataMatrix
    from models.state_manager import get_state_manager

    state = get_state_manager()
    state.set_data_matrix(DataMatrix(np.ones((3, 3))))
    assert state.has_data
    assert state.is_modified

    _RecordingMessageBox.asked.clear()
    monkeypatch.setattr(m, "QMessageBox", _RecordingMessageBox)

    window = m.MainWindow()
    try:
        window.close()
    finally:
        window.deleteLater()
        qapp.processEvents()

    assert _RecordingMessageBox.asked, "the window closed without asking about unsaved data"
