"""Tests for controllers/data_controller.py — CSV loading and DataLoadTask."""

import pytest

# PyQt6 is required throughout the codebase (utils.event_bus imports it).
# All tests in this module are skipped if PyQt6 is not available.
pytest.importorskip("PyQt6", reason="PyQt6 is required throughout the PaleoAST codebase")

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """Create a small CSV file for testing."""
    content = """\
name,a,b,c
sample1,1.0,2.0,3.0
sample2,4.0,5.0,6.0
sample3,7.0,8.0,9.0
"""
    path = tmp_path / "small.csv"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def csv_with_missing(tmp_path) -> Path:
    """CSV file with missing-value markers."""
    content = """\
x,y,z
1.0,NA,3.0
4.0,5.0,NA
NA,8.0,9.0
"""
    path = tmp_path / "missing.csv"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def large_csv(tmp_path) -> Path:
    """Create a large CSV (10 000 rows) for performance/async testing."""
    n_rows = 10000
    n_cols = 20
    lines = ["col" + str(j) for j in range(n_cols)]
    lines = ",".join(lines) + "\n"
    for i in range(n_rows):
        vals = ",".join(str(round(i * j + 0.5, 4)) for j in range(n_cols))
        lines += vals + "\n"
    path = tmp_path / "large.csv"
    path.write_text(lines, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Synchronous load_csv
# ---------------------------------------------------------------------------


def test_load_csv_basic(sample_csv):
    """load_csv returns a DataMatrix with the correct shape and values."""
    from controllers.data_controller import DataController

    ctrl = DataController()
    matrix = ctrl.load_csv(str(sample_csv))

    assert matrix.n_samples == 3
    assert matrix.n_variables == 3
    np.testing.assert_allclose(matrix.data[0], [1.0, 2.0, 3.0])
    assert matrix.row_labels == ["sample1", "sample2", "sample3"]
    assert matrix.col_labels == ["a", "b", "c"]


def test_load_csv_no_header(tmp_path):
    """When has_header=False, pandas uses generic column names."""
    content = "1.0,2.0\n3.0,4.0\n"
    path = tmp_path / "nohdr.csv"
    path.write_text(content, encoding="utf-8")

    from controllers.data_controller import DataController

    ctrl = DataController()
    matrix = ctrl.load_csv(str(path), has_header=False)

    assert matrix.n_samples == 2
    assert matrix.n_variables == 2
    # Generic labels because no header was read
    assert matrix.col_labels == ["0", "1"]


def test_load_csv_with_row_labels(tmp_path):
    """First column is used as row labels when has_row_labels=True."""
    content = """\
name,a,b
s1,1.0,2.0
s2,3.0,4.0
"""
    path = tmp_path / "rowlabels.csv"
    path.write_text(content, encoding="utf-8")

    from controllers.data_controller import DataController

    ctrl = DataController()
    matrix = ctrl.load_csv(str(path), has_row_labels=True)

    assert matrix.row_labels == ["s1", "s2"]
    assert matrix.col_labels == ["a", "b"]
    np.testing.assert_allclose(matrix.data[0], [1.0, 2.0])


def test_load_csv_missing_values(csv_with_missing):
    """Missing-value string is correctly converted to NaN."""
    from controllers.data_controller import DataController

    ctrl = DataController()
    matrix = ctrl.load_csv(str(csv_with_missing), missing_value="NA")

    assert matrix.has_missing
    assert np.isnan(matrix.data[0, 1])
    assert np.isnan(matrix.data[1, 2])
    assert np.isnan(matrix.data[2, 0])
    assert matrix.data[0, 0] == 1.0


def test_load_csv_file_not_found():
    """Non-existent file raises FileOperationError."""
    from controllers.data_controller import DataController
    from utils.exceptions import FileOperationError

    ctrl = DataController()
    with pytest.raises(FileOperationError, match="not found"):
        ctrl.load_csv("/does/not/exist.csv")


def test_load_csv_empty_file(tmp_path):
    """Empty file raises FileOperationError."""
    from controllers.data_controller import DataController
    from utils.exceptions import FileOperationError

    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    ctrl = DataController()
    with pytest.raises(FileOperationError, match="empty"):
        ctrl.load_csv(str(path))


# ---------------------------------------------------------------------------
# DataLoadTask unit tests
# ---------------------------------------------------------------------------


def test_csv_load_task_cancel():
    """CsvLoadTask._cancelled flag is set by cancel()."""
    from controllers.data_controller import CsvLoadTask

    task = CsvLoadTask("/tmp/fake.csv")
    assert not task.is_cancelled
    task.cancel()
    assert task.is_cancelled


def test_data_load_task_task_attribute():
    """DataLoadTask exposes the underlying CsvLoadTask via .task property."""
    from controllers.data_controller import DataLoadTask

    task = DataLoadTask("/tmp/fake.csv", missing_value="NA")
    assert task.task.filepath == "/tmp/fake.csv"
    assert task.task.missing_value == "NA"
    assert task.task.is_cancelled is False


# ---------------------------------------------------------------------------
# Async load_csv_async (PyQt6 required — each test handles its own skip)
# ---------------------------------------------------------------------------

def _get_qapp():
    """Return a QApplication instance, creating one if needed."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-tests"])
    return app


def test_load_csv_async_emits_result_ready(sample_csv):
    """load_csv_async emits result_ready with a valid DataMatrix."""
    pyqt6 = pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QSignalSpy

    from controllers.data_controller import DataController

    qapp = _get_qapp()
    ctrl = DataController()
    task = ctrl.load_csv_async(str(sample_csv))

    spy_result = QSignalSpy(task.result_ready, qapp)
    spy_error = QSignalSpy(task.error_raised, qapp)

    # Wait up to 30 s for the result
    assert spy_result.wait(30000), "result_ready was never emitted"
    assert len(spy_error) == 0, "error_raised was unexpectedly emitted"

    matrix = spy_result[0][0]
    assert matrix.n_samples == 3
    assert matrix.n_variables == 3


def test_load_csv_async_does_not_block_main_thread(large_csv):
    """The async task must emit progress signals before completion.

    If load_csv_async ran on the main thread, the progress signal would only
    be emitted after the entire file is parsed, not during parsing.
    We verify that progress is emitted during loading by checking that at
    least two distinct progress signals are received.
    """
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QSignalSpy

    from controllers.data_controller import DataController

    qapp = _get_qapp()
    ctrl = DataController()
    task = ctrl.load_csv_async(str(large_csv))

    spy_progress = QSignalSpy(task.progress, qapp)

    # Wait up to 60 s for at least 2 progress signals (start + partial update)
    assert spy_progress.wait(60000), "progress signal was never emitted"
    assert len(spy_progress) >= 2, (
        f"Expected >= 2 progress signals during load, got {len(spy_progress)}"
    )

    # First signal should be indeterminate (0, -1)
    first = spy_progress[0]
    assert first[0] == 0 and first[1] == -1, "First progress should be indeterminate"

    # Subsequent signals should have real row counts
    for sig in spy_progress[1:]:
        assert sig[1] > 0, "Later progress signals should have real total row count"


def test_load_csv_async_cancelled_before_submit():
    """Cancelling before submit prevents result_ready from firing with data."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QSignalSpy

    from controllers.data_controller import DataController

    qapp = _get_qapp()
    ctrl = DataController()
    # Use a path that would succeed if loaded — but we cancel before submitting
    task = ctrl.load_csv_async("/tmp/never_load.csv")
    task.task.cancel()  # mark cancelled before submit

    spy_result = QSignalSpy(task.result_ready, qapp)
    spy_cancelled = QSignalSpy(task.cancelled, qapp)

    # The synchronous fallback path fires cancelled immediately
    assert spy_cancelled.wait(5000), "cancelled signal should be emitted"
    assert spy_result.wait(5000), "result_ready should be emitted with None"
    assert spy_result[0][0] is None
