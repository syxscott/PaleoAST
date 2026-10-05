"""Regression test for the PCA UI entry point, not the controller.

Why this file exists
--------------------
The earlier PCA min_variance work was verified by calling
``controller.run_pca()`` and then ``_on_pca_result_ready()`` directly. That
SKIPS ``MainWindow._on_run_pca``'s inner ``_work`` closure -- and the closure
is where a ``NameError`` was hiding: a ``@staticmethod`` was called without
``self.``. The full suite stayed green for exactly the same reason; every
existing PCA test drives the controller, never the dialog.

The first attempt at this file spun an event loop and deadlocked, because
``app.processEvents()`` and ``QEventLoop.exec()`` competed for the same
single-shot timer. Instead of reproducing that fragility, this test replaces
``_run_analysis_async`` with a synchronous shim. That is not a weakening: the
defect lives entirely inside the ``work`` callback, so invoking that callback
is precisely the code under test, and it becomes deterministic.
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"
sys.path.insert(0, r"D:\GIthub\PaleoAST")

import numpy as np
from PyQt6.QtWidgets import QApplication, QDialog, QStackedWidget

app = QApplication.instance() or QApplication([])

from models.data_matrix import DataMatrix


def _default_params():
    """Exactly what the shipped PCADialog produces with its defaults.

    min_variance 0.05 is the value that used to reach the broken branch.
    """
    return {
        "n_components": 3,
        "method": "correlation",
        "min_variance": 0.05,
        "show_loadings": True,
        "show_scores": True,
        "show_scree": True,
        "show_biplot": False,
        "biplot_scale": 1.0,
        "impute_missing": False,
        "parallel": False,
    }


def _build_window(monkeypatch, params):
    import views.ui_main_window as m

    rng = np.random.default_rng(3)
    n, p = 14, 5
    data = rng.normal(size=(n, p)) * 0.4 + 3.0 + np.arange(n)[:, None] * 0.05

    win = m.MainWindow()
    win.resize(1400, 900)
    try:
        # A dirty MainWindow's closeEvent opens a modal "save changes?" box,
        # which blocks forever with no user to dismiss it. Teardown is not
        # what this file is testing, so make close a no-op.
        win.closeEvent = lambda event: None
        win._state.set_data_matrix(
            DataMatrix(data,
                       row_labels=[str(i) for i in range(n)],
                       col_labels=[f"V{j}" for j in range(p)])
        )

        class FakeDialog:
            def __init__(self, parent=None):
                pass

            def setDarkTheme(self, v):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            def get_parameters(self):
                return dict(params)

        monkeypatch.setattr(m, "PCADialog", FakeDialog)

        # Run the analysis synchronously: the work callback is the code under
        # test, and this keeps the test free of thread/event-loop timing.
        seen = {}

        def sync_run(work, on_done, on_fail, label=None):
            seen["label"] = label
            try:
                result = work()
            except Exception as exc:            # the failure we are hunting
                seen["raised"] = exc
                on_fail(exc)
                return
            seen["result"] = result
            on_done(result)

        monkeypatch.setattr(win, "_run_analysis_async", sync_run)

        # The failure path ends in QMessageBox.critical(), a MODAL dialog that
        # blocks forever with nobody to dismiss it. Record instead, so a
        # regression FAILS the test rather than hanging the run.
        errors = []

        def record_error(exc):
            errors.append(exc)

        monkeypatch.setattr(win, "_on_pca_error", record_error)
        monkeypatch.setattr(win._status_bar, "setWarning", lambda *a, **k: None)
        return win, seen, errors
    except Exception:
        win.close()
        raise


def test_pca_dialog_path_executes_the_work_closure(monkeypatch):
    """The user's path: accept the dialog with defaults, run, get a plot."""
    win, seen, errors = _build_window(monkeypatch, _default_params())
    try:
        win._on_run_pca()

        assert errors == [], f"the error path was taken: {errors[0]!r}"

        assert "result" in seen, "the work closure never returned a result"

        stack = win._workspace.findChildren(QStackedWidget)[0]
        classes = [stack.widget(i).metaObject().className()
                   for i in range(stack.count())]
        assert classes.count("InteractivePlotCanvas") >= 1, f"no plot: {classes}"
        assert stack.currentWidget().metaObject().className() == "InteractivePlotCanvas", (
            f"workspace did not rest on the score plot; current="
            f"{stack.currentWidget().metaObject().className()} pages={classes}"
        )
    finally:
        win.close()


def test_pca_dialog_path_with_zero_min_variance(monkeypatch):
    """min_variance = 0 must mean 'no threshold', and must not raise."""
    params = _default_params()
    params["min_variance"] = 0.0
    win, seen, errors = _build_window(monkeypatch, params)
    try:
        win._on_run_pca()
        assert errors == [], f"the error path was taken: {errors[0]!r}"
        assert "result" in seen
    finally:
        win.close()


def test_pca_dialog_path_keeps_min_variance_semantics(monkeypatch):
    """A high threshold must retain fewer components than a low one."""
    low = _default_params()
    low["min_variance"] = 0.01
    high = _default_params()
    high["min_variance"] = 0.80

    def retained(params):
        win, seen, errors = _build_window(monkeypatch, params)
        try:
            win._on_run_pca()
            assert errors == [], f"the error path was taken: {errors[0]!r}"
            return int(seen["result"].n_components)
        finally:
            win.close()

    n_low = retained(low)
    n_high = retained(high)
    assert n_high <= n_low, (
        f"80% threshold kept {n_high} components but 1% kept {n_low}"
    )


def test_trim_helper_handles_falsy_n_components():
    """`n_avail` was referenced before assignment, masked by `or` short-circuit.

    The right side of ``getattr(...) or n_avail`` is only evaluated when the
    left side is falsy, so the bug stayed hidden until n_components was 0.
    """
    from views.ui_main_window import MainWindow

    class FalsyN:
        n_components = 0
        cumulative_variance = np.array([30.0, 60.0, 90.0])

    out = MainWindow._trim_components_to_variance(FalsyN(), 0.0)
    assert isinstance(out, int) and out >= 1, f"-> {out!r}"

    class NoSpec:
        n_components = 0
        explained_variance = np.array([30.0, 60.0, 90.0])

    out2 = MainWindow._trim_components_to_variance(NoSpec(), 0.0)
    assert isinstance(out2, int) and out2 >= 1, f"-> {out2!r}"

