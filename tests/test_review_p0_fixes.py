# =============================================================================
# FILE: tests/test_review_p0_fixes.py
# =============================================================================
"""
Regression tests for the five P0 defects found by the 2026-10-19 code review.

Each test pins one defect that used to produce a silently wrong result, a
crash, or a lost column.  They are written so that a future change that
reintroduces the defect fails here rather than in a published paper.

The five defects
----------------
P0-1  views/ui_spreadsheet.py
      A cell edit of "inf" reached the DataMatrix validator, whose
      DataValidationError escaped the Qt slot.  PyQt6 aborts the process on
      an unhandled exception in a slot, so typing "inf" in one cell lost
      every unsaved edit.
P0-2  parsers/dat_parser.py
      _is_label and _parse_numeric_value disagreed about thousand
      separators, so a thousand-separated number in the FIRST data column
      was classified as a row label and the whole column vanished from the
      data -- with no error at all.
P0-3  utils/statistics_core.py
      fit_nonlinear hard-coded absolute_sigma=True while sigma was None, so
      scipy returned an unscaled covariance and every growth-model standard
      error and 95% interval was too narrow by sqrt(RSS/(M-N)).
P0-4  stats/distribution_tests.py
      The Epps-Singleton permutation loop divided by a group mean that can
      be exactly zero, raising a bare ZeroDivisionError out of the test.
P0-5  morphometrics/allometry.py
      The isometry F-test paired a pooled sum of squares with the residual
      df of a single coordinate, so the p-value was systematically far too
      large and real allometry was reported as isometry.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# P0-1  the spreadsheet must survive an "inf" cell edit
# =============================================================================
_SHEET_SCRIPT = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, {repo!r})
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QTableWidgetItem

app = QApplication([])

def dismiss():
    w = QApplication.activeModalWidget()
    if w is not None:
        w.close()

QTimer.singleShot(150, dismiss)
QTimer.singleShot(400, dismiss)

from views.ui_spreadsheet import ScientificSpreadsheet

sheet = ScientificSpreadsheet()
sheet.load_data([[1.0, 2.0], [3.0, 4.0]], row_labels=["r1", "r2"], col_labels=["c1", "c2"])
item = sheet._table.item(0, 0)
item.setText({text!r})
try:
    sheet._on_item_changed(item)
except RuntimeError:
    pass  # the table may rebuild the item; the stored value is what matters
print("STORED", float(sheet._data[0, 0]))
print("SURVIVED")
sheet.close()
sheet.deleteLater()
app.processEvents()
sys.stdout.flush()
# Qt's interpreter teardown can abort when the QApplication is collected
# while a widget is still alive, which would turn a PASS into a crash and
# make this test's exit-code check fail for the wrong reason.  The
# assertions are already printed, so leave immediately.
os._exit(0)
"""


def _run_sheet_script(text: str) -> tuple[int, str]:
    path = os.path.join(tempfile.gettempdir(), "paleo_sheet_regression.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_SHEET_SCRIPT.format(repo=REPO_ROOT, text=text))
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=180)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@pytest.mark.slow
def test_inf_cell_edit_does_not_abort_the_process():
    """P0-1: typing "inf" must not kill the application."""
    code, out = _run_sheet_script("inf")
    assert "SURVIVED" in out, f"the process did not survive: {out[-800:]}"
    assert code == 0, f"the process aborted with exit code {code}: {out[-800:]}"
    # The rejected value must NOT have been stored.
    assert "STORED 1.0" in out, f"inf was stored: {out[-800:]}"


@pytest.mark.slow
def test_nan_cell_edit_is_still_accepted():
    """P0-1: NaN is a legal missing value and must still be storable.

    The P0-1 fix rejects infinities.  It must not reject NaN as well --
    typing "nan" is how a user records a missing measurement, and
    DataMatrix allows NaN (allow_nan=True).
    """
    code, out = _run_sheet_script("nan")
    assert "SURVIVED" in out, f"the process did not survive: {out[-800:]}"
    assert code == 0, f"the process aborted with exit code {code}: {out[-800:]}"
    assert "STORED nan" in out, f"NaN was rejected but must be accepted: {out[-800:]}"


# =============================================================================
# P0-2  a thousand separator in the FIRST data column is data, not a label
# =============================================================================
def _write_dat(tmp_path, text: str) -> str:
    path = tmp_path / "sample.dat"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_dat_thousand_separator_in_first_data_column_is_kept(tmp_path):
    """P0-2: the first data column must survive a thousand separator."""
    from parsers.dat_parser import DATParser

    path = _write_dat(
        tmp_path,
        "Length\tWeight\n1,234.56\t2,345.67\n2,345.67\t3,456.78\n",
    )
    parsed = DATParser().parse(path)
    data = np.asarray(parsed.data)
    assert data.shape == (2, 2), f"a column was lost: {data.shape}"
    assert abs(float(data[0, 0]) - 1234.56) < 1e-6
    # The numbers must not have become the row labels.
    assert not any("," in str(label) for label in parsed.row_labels)


def test_dat_dotted_identifier_is_still_a_label(tmp_path):
    """P0-2 guard: "1.2.3" is an identifier, not the number 123."""
    from parsers.dat_parser import DATParser

    path = _write_dat(
        tmp_path,
        "Specimen\tWeight\n1.2.3\t2.5\n2.3.4\t3.5\n",
    )
    parsed = DATParser().parse(path)
    data = np.asarray(parsed.data)
    assert data.shape == (2, 1), f"a dotted identifier was read as data: {data.shape}"
    assert list(parsed.row_labels) == ["1.2.3", "2.3.4"]


# =============================================================================
# P0-3  growth-model confidence intervals must not be too narrow
# =============================================================================
def test_fit_nonlinear_standard_error_matches_scipy():
    """P0-3: fit_nonlinear must scale the covariance by the residual variance."""
    import scipy.optimize as opt

    from utils.statistics_core import fit_nonlinear

    def model(t, linf, k):
        return linf * (1.0 - np.exp(-k * t))

    rng = np.random.default_rng(0)
    t = np.arange(1.0, 16.0)
    y = 50.0 * (1.0 - np.exp(-0.2 * t)) + rng.normal(0, 5, size=t.size)

    result = fit_nonlinear(model, t, y, p0=[45.0, 0.15])
    reported = float(result.stderr[0])
    _popt, pcov = opt.curve_fit(model, t, y, p0=[45.0, 0.15])
    expected = float(np.sqrt(pcov[0, 0]))

    assert expected > 0
    assert abs(reported - expected) / expected < 0.02, (
        f"reported stderr {reported:.5f} != scipy {expected:.5f}: "
        "the covariance was not scaled by the residual variance"
    )


# =============================================================================
# P0-4  Epps-Singleton must not raise ZeroDivisionError
# =============================================================================
def test_epps_singleton_permutation_does_not_divide_by_zero():
    """P0-4: a permutation with a zero-mean group must be skipped, not fatal."""
    from stats.distribution_tests import DistributionTestAnalyzer

    sample1 = np.array([-1.0, 1.0, 10.0, 11.0])
    sample2 = np.array([-2.0, 2.0, 12.0, 13.0])

    result = DistributionTestAnalyzer().epps_singleton(sample1, sample2, n_permutations=199, random_seed=7)
    assert 0.0 <= float(result.p_value) <= 1.0


# =============================================================================
# P0-5  the isometry test must detect real allometry
# =============================================================================
def _allometric_data(seed: int = 3, n: int = 20, with_allometry: bool = True):
    rng = np.random.default_rng(seed)
    log_cs = rng.normal(0, 0.3, n)
    coords = np.zeros((n, 6, 2))
    if with_allometry:
        coords[:, 0, 0] = log_cs * 1.5 + rng.normal(0, 0.2, n)
        coords[:, 1, 0] = log_cs * 1.2 + rng.normal(0, 0.2, n)
        coords[:, 0, 1] = log_cs * 1.1 + rng.normal(0, 0.2, n)
    for k in range(2, 6):
        coords[:, k, 0] = rng.normal(0, 0.2, n)
        coords[:, k, 1] = rng.normal(0, 0.2, n)
    return coords, np.exp(log_cs)


def test_isometry_test_detects_real_allometry():
    """P0-5: a real allometric signal must be rejected."""
    from morphometrics.allometry import AllometryAnalyzer

    coords, sizes = _allometric_data(with_allometry=True)
    result = AllometryAnalyzer().analyze_allometry(coords, centroid_sizes=sizes, n_permutations=999, random_seed=1)
    assert float(result.r_squared) > 0.4
    assert float(result.isometry_pvalue) < 0.05, f"p={result.isometry_pvalue} -- the isometry test is too conservative"


def test_isometry_test_does_not_reject_without_allometry():
    """P0-5: under H0 the test must keep its nominal error rate."""
    from morphometrics.allometry import AllometryAnalyzer

    rejects = 0
    trials = 8
    for i in range(trials):
        coords, sizes = _allometric_data(seed=100 + i, with_allometry=False)
        result = AllometryAnalyzer().analyze_allometry(coords, centroid_sizes=sizes, n_permutations=199, random_seed=5)
        if float(result.isometry_pvalue) < 0.05:
            rejects += 1
    assert rejects <= 2, f"{rejects}/{trials} false positives under H0"


def test_isometry_permutation_is_reproducible_with_a_seed():
    """P0-5: the same seed must give the same p-value."""
    from morphometrics.allometry import AllometryAnalyzer

    coords, sizes = _allometric_data(with_allometry=True)
    first = AllometryAnalyzer().analyze_allometry(coords, centroid_sizes=sizes, n_permutations=199, random_seed=42)
    second = AllometryAnalyzer().analyze_allometry(coords, centroid_sizes=sizes, n_permutations=199, random_seed=42)
    assert float(first.isometry_pvalue) == float(second.isometry_pvalue)
