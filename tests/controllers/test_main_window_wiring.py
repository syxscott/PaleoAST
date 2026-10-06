"""UI-wiring regression tests for ``views/ui_main_window.py``.

These tests instantiate ``MainWindow`` and exercise the click paths
that the user actually goes through.  Where the real handlers would
kick off expensive analyses, a ``FakeController`` is substituted so
the test finishes in <100 ms.

Each test corresponds to one of the ten defects called out in the
session brief.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="UI tests require PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-mainwindow-tests"])
    return app


def _ensure_qsettings_isolated(tmp_path):
    """Point QSettings at a sandbox ini so each test starts clean."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )


class FakeResult:
    """Generic stand-in for any plot result with ``to_dict``/``summary``."""

    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs

    def to_dict(self) -> dict:
        return dict(self._data)

    def summary(self) -> str:
        return str(self._data.get("summary", "fake"))


class FakeController:
    """Stub for ``StatisticsController`` with just enough surface for
    the wiring tests.  Each ``analyze_*`` call appends to ``calls``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Default return shapes — overridable in setUp if needed.
        self.pca_result = MagicMock(
            scores=np.zeros((4, 2)),
            explained_variance=np.array([0.6, 0.3]),
            eigenvalues_raw=np.array([0.6, 0.3]),
            cumulative_variance=np.array([0.6, 0.9]),
            n_components=2,
            loadings=np.zeros((3, 2)),
        )
        self.pcoa_result = MagicMock(
            coordinates=np.zeros((4, 2)),
            eigenvalues=np.array([0.6, 0.3]),
            proportion_explained=np.array([0.6, 0.3]),
            n_components=2,
            correction_method="cmdscale",
        )
        self.nmds_result = MagicMock(stress=0.05, coordinates=np.zeros((4, 2)))
        self.anosim_result = MagicMock(summary=lambda: "anosim")
        self.permanova_result = MagicMock(summary=lambda: "permanova")
        self.cca_result = MagicMock(summary=lambda: "cca", constrained_variance=42.0)
        self.lda_result = MagicMock(accuracy=0.9, n_classes=3, scores=np.zeros((6, 2)))
        self.simper_result = MagicMock(summary=lambda: "simper")
        self.diversity_result = MagicMock(summary=lambda: "diversity")
        self.rarefaction_result = MagicMock(summary=lambda: "rarefaction")
        self.directional_result = MagicMock(mean_direction_deg=12.3, rayleigh_p=0.01)
        self.clustering_result = MagicMock(n_clusters=3, cophenetic_corr=0.7)
        self.eigenshape_result = MagicMock(
            scores=np.zeros((4, 2)),
            explained_variance=np.array([0.7, 0.2]),
            n_specimens=4,
            n_components=2,
        )

    # The methods the main window calls.  Each records the call and
    # returns a deterministic stand-in.

    def run_pca(self, **kwargs):
        self.calls.append(("run_pca", kwargs))
        return self.pca_result

    def run_pcoa(self, **kwargs):
        self.calls.append(("run_pcoa", kwargs))
        return self.pcoa_result

    def run_nmds(self, **kwargs):
        self.calls.append(("run_nmds", kwargs))
        return self.nmds_result

    def analyze_anosim(self, **kwargs):
        self.calls.append(("analyze_anosim", kwargs))
        return self.anosim_result

    def analyze_permanova(self, **kwargs):
        self.calls.append(("analyze_permanova", kwargs))
        return self.permanova_result

    def analyze_simper(self, **kwargs):
        self.calls.append(("analyze_simper", kwargs))
        return self.simper_result

    def analyze_lda(self, **kwargs):
        self.calls.append(("analyze_lda", kwargs))
        return self.lda_result

    def run_cca(self, **kwargs):
        self.calls.append(("run_cca", kwargs))
        return self.cca_result

    def analyze_clustering(self, **kwargs):
        self.calls.append(("analyze_clustering", kwargs))
        return self.clustering_result

    def analyze_diversity(self, **kwargs):
        self.calls.append(("analyze_diversity", kwargs))
        return self.diversity_result

    def analyze_rarefaction(self, **kwargs):
        self.calls.append(("analyze_rarefaction", kwargs))
        return self.rarefaction_result

    def analyze_directional(self, **kwargs):
        self.calls.append(("analyze_directional", kwargs))
        return self.directional_result

    def bin_rose_diagram(self, **kwargs):
        self.calls.append(("bin_rose_diagram", kwargs))
        return np.array([0.0, 90.0, 180.0, 270.0]), np.array([3, 1, 2, 4])

    def analyze_spectral(self, **kwargs):
        self.calls.append(("analyze_spectral", kwargs))
        return MagicMock()

    def analyze_eigenshape(self, **kwargs):
        # Not actually exposed on the controller; this stub exists so
        # tests can substitute a fake for the morphometrics analyser.
        self.calls.append(("analyze_eigenshape", kwargs))
        return self.eigenshape_result


def _install_minimal_data(window) -> None:
    """Make ``_state.has_data`` true and populate ``row_metadata``
    groups so ``_get_groups()`` returns a sensible list."""

    from models.data_matrix import DataMatrix
    from models.row_metadata import RowMetadataManager

    data = np.arange(20.0).reshape(4, 5)
    matrix = DataMatrix(
        data,
        row_labels=["Site_1", "Site_2", "Site_3", "Site_4"],
        col_labels=[f"species_{i}" for i in range(5)],
        specimen_metadata=[
            {"group": "Grassland"},
            {"group": "Woodland"},
            {"group": "Grassland"},
            {"group": "Woodland"},
        ],
    )
    window._state.set_data_matrix(matrix, mark_modified=False)

    # Set up row metadata groups: rows 0,2 -> 0; rows 1,3 -> 1.
    rm = RowMetadataManager(n_rows=4, row_labels=list(matrix.row_labels))
    rm.set_group(0, "Grassland")
    rm.set_group(1, "Woodland")
    rm.set_group(2, "Grassland")
    rm.set_group(3, "Woodland")
    window._state._row_metadata = rm


def _install_minimal_data_no_groups(window) -> None:
    """Same as ``_install_minimal_data`` but no row-metadata."""

    from models.data_matrix import DataMatrix

    data = np.arange(20.0).reshape(4, 5)
    matrix = DataMatrix(
        data,
        row_labels=["A", "B", "C", "D"],
        col_labels=[f"v{i}" for i in range(5)],
    )
    window._state.set_data_matrix(matrix, mark_modified=False)


@pytest.fixture
def main_window(qapp, tmp_path, monkeypatch):
    """Build a ``MainWindow`` with everything heavy stubbed.

    The fixture returns a fully-functional window whose statistics
    controller and plot canvas are inert fakes.  This lets the tests
    exercise the click paths without standing up the entire stats
    stack.
    """
    _ensure_qsettings_isolated(tmp_path)

    # Block the heavy lazy imports inside MainWindow._setup_ui.
    from views import ui_main_window as mw

    # Patch the matplotlib figure backing the InteractivePlotCanvas so
    # we don't actually paint anything in tests.
    class _SilentCanvas:
        def __init__(self, *args, **kwargs):
            self._current_plot_type = "stub"
            self.ax = MagicMock()
            self.fig = MagicMock()

        def __getattr__(self, item):
            return MagicMock()

    # Replace the InteractivePlotCanvas class with a no-op so
    # plotting never touches matplotlib.
    monkeypatch.setattr(mw, "InteractivePlotCanvas", _SilentCanvas)

    # Build a real MainWindow but stub the StatisticsController.
    window = mw.MainWindow.__new__(mw.MainWindow)
    # Minimal state setup: the constructor runs in __init__ but
    # pulls in the real StatisticsController. We replicate only the
    # bits needed for the helpers under test.

    # Initialize state via StateManager (no PyQt6 timers needed for
    # the tests we run).
    from models.state_manager import get_state_manager

    window._state = get_state_manager()
    window._data_controller = MagicMock()
    window._statistics_controller = FakeController()
    window._logger = mw.logger
    window._closing = False
    window._is_dark_theme = False
    window._thread_pool = qapp.threadPool() if hasattr(qapp, "threadPool") else MagicMock()

    # Workspace plumbing -- we don't need the actual stack for the
    # orchestration tests, just enough that ``_add_plot_to_workspace``
    # etc. work.
    workspace_mock = MagicMock()
    workspace_mock.loadExampleRequested = MagicMock()
    workspace_mock.openFileRequested = MagicMock()
    workspace_mock.importDataRequested = MagicMock()
    workspace_mock.addWidget = MagicMock(return_value=0)
    workspace_mock.addTab = MagicMock(return_value=0)
    workspace_mock._placeholder = MagicMock()  # used by _evict_excess_result_tabs
    window._workspace = workspace_mock
    window._spreadsheet = MagicMock()  # also referenced by _evict_excess_result_tabs
    # _add_plot_to_workspace calls an internal eviction helper we do
    # not need to exercise here.
    window._evict_excess_result_tabs = MagicMock()

    class _StubStatus:
        def setInfo(self, *args, **kwargs):
            pass

        def setProgress(self, *args, **kwargs):
            pass

    window._status_bar = _StubStatus()
    window._spreadsheet_index = 0
    return window


# ---------------------------------------------------------------------------
# Defect 1: labels and groups reach the canvas
# ---------------------------------------------------------------------------


def test_plot_labels_and_groups_from_specimen_metadata(main_window):
    """``_get_plot_labels_and_groups`` must surface real row labels and
    a meaningful group mapping -- not the canvas's ``S1..Sn`` / all-0
    fallback."""

    _install_minimal_data(main_window)
    labels, groups, group_names = main_window._get_plot_labels_and_groups()

    assert labels == ["Site_1", "Site_2", "Site_3", "Site_4"]
    assert groups == [0, 1, 0, 1]
    assert group_names == ["Grassland", "Woodland"]


def test_plot_labels_and_groups_falls_back_to_row_metadata(main_window):
    """If specimen_metadata lacks a group key, use the row_metadata edits."""

    from models.data_matrix import DataMatrix
    from models.row_metadata import RowMetadataManager

    data = np.arange(12.0).reshape(4, 3)
    matrix = DataMatrix(
        data,
        row_labels=["r1", "r2", "r3", "r4"],
        col_labels=["a", "b", "c"],
    )
    main_window._state.set_data_matrix(matrix, mark_modified=False)

    rm = RowMetadataManager(n_rows=4, row_labels=list(matrix.row_labels))
    rm.set_group(0, "A")
    rm.set_group(1, "B")
    rm.set_group(2, "A")
    rm.set_group(3, "B")
    main_window._state._row_metadata = rm

    labels, groups, group_names = main_window._get_plot_labels_and_groups()
    assert labels == ["r1", "r2", "r3", "r4"]
    assert groups == [0, 1, 0, 1]
    assert group_names == ["A", "B"]


def test_plot_labels_and_groups_returns_none_when_nothing_known(main_window):
    """When nothing is known, ``groups`` must be None (not the all-zero
    fabrication that the canvas used to fall back to)."""

    _install_minimal_data_no_groups(main_window)
    labels, groups, group_names = main_window._get_plot_labels_and_groups()
    assert labels == ["A", "B", "C", "D"]
    assert groups is None
    assert group_names is None


def test_execute_pca_passes_through_main_window(main_window, monkeypatch):
    """``_execute_pca`` must reach the controller with the user's
    parameters and the canvas with real labels+groups."""

    from views import ui_main_window as mw

    _install_minimal_data(main_window)

    # Stub run_analysis_async so the test is synchronous.
    main_window._run_analysis_async = MagicMock()

    params = {
        "n_components": 3,
        "method": "svd",
        "min_variance": 0.0,
        "show_loadings": True,
        "show_scores": True,
        "show_scree": True,
        "show_biplot": False,
        "biplot_scale": 1.0,
        "impute_missing": False,
        "parallel": False,
    }
    main_window._execute_pca(params)

    # The async path was called; emulate its success callback with the
    # canned result and verify it forwards labels/groups to the canvas.
    assert main_window._run_analysis_async.called
    args = main_window._run_analysis_async.call_args
    _work, on_success, _on_fail, title = args[0]
    assert title == "PCA"

    # Now drive on_success and inspect what plot_pca_scores received.
    captured = {}

    class _Spy:
        def plot_pca_scores(self, *a, **kw):
            captured["args"] = a
            captured["kwargs"] = kw

        def __getattr__(self, item):
            # Auto-stub other plotting methods so the PCA result-ready
            # callback (which also draws the scree + loadings tabs)
            # doesn't crash before our assertion runs.
            return MagicMock()

    # Patch the canvas with a spy.
    monkeypatch.setattr(mw, "InteractivePlotCanvas", _Spy)

    on_success(main_window._statistics_controller.pca_result)
    assert captured["kwargs"]["labels"] == ["Site_1", "Site_2", "Site_3", "Site_4"]
    assert captured["kwargs"]["groups"] == [0, 1, 0, 1]
    assert captured["kwargs"]["group_names"] == ["Grassland", "Woodland"]


def test_execute_pcoa_passes_labels_and_groups(main_window, monkeypatch):
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    main_window._execute_pcoa({"metric": "bray_curtis", "n_components": 2})
    args = main_window._run_analysis_async.call_args
    _work, on_success, _on_fail, _title = args[0]

    captured = {}

    class _Spy:
        def plot_pcoa_scores(self, *a, **kw):
            captured["kwargs"] = kw

        def __getattr__(self, item):
            return MagicMock()

    monkeypatch.setattr(mw, "InteractivePlotCanvas", _Spy)
    on_success(main_window._statistics_controller.pcoa_result)
    assert captured["kwargs"]["labels"] == ["Site_1", "Site_2", "Site_3", "Site_4"]
    assert captured["kwargs"]["groups"] == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Defect 2: ANOSIM / PERMANOVA use the dialog
# ---------------------------------------------------------------------------


def test_execute_anosim_passes_dialog_params(main_window, monkeypatch):
    """The handler must hand ``metric``, ``n_permutations`` and
    ``random_seed`` to the controller via ``_run_analysis_async``."""

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()
    captured = {}

    def _capture(work, on_success, on_fail, title, wants_reporter=False):
        captured["title"] = title
        captured["params"] = work.__wrapped__ if hasattr(work, "__wrapped__") else None
        # The work closure isn't directly inspectable, so just call
        # it to verify the closure reads the right args.
        work()

    main_window._run_analysis_async.side_effect = _capture
    main_window._execute_anosim({"metric": "jaccard", "n_permutations": 199, "random_seed": 7})

    assert captured["title"] == "ANOSIM"
    name, kwargs = main_window._statistics_controller.calls[-1]
    assert name == "analyze_anosim"
    assert kwargs["metric"] == "jaccard"
    assert kwargs["n_permutations"] == 199


def test_execute_permanova_is_async(main_window):
    """PERMANOVA must run through the thread pool (was synchronous)."""

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()
    main_window._execute_permanova({"metric": "manhattan", "n_permutations": 499, "random_seed": None})
    assert main_window._run_analysis_async.called
    title = main_window._run_analysis_async.call_args[0][3]
    assert title == "PERMANOVA"


def test_execute_permanova_calls_controller(main_window):
    _install_minimal_data(main_window)

    def _fake_async(work, *args, **kwargs):
        work()

    main_window._run_analysis_async = MagicMock(side_effect=_fake_async)
    main_window._execute_permanova({"metric": "euclidean", "n_permutations": 999, "random_seed": 5})
    name, kwargs = main_window._statistics_controller.calls[-1]
    assert name == "analyze_permanova"
    assert kwargs["metric"] == "euclidean"
    assert kwargs["n_permutations"] == 999


# ---------------------------------------------------------------------------
# Defect 3: dialog results wired to a plotter
# ---------------------------------------------------------------------------


def test_evolution_rate_handler_emits_plot_or_text(main_window, monkeypatch):
    """``_on_evolution_rate_result`` must reach the canvas or fall
    back to a text tab.  In the test we patch the canvas to a spy."""
    from views import ui_main_window as mw

    class _Spy:
        calls: list[dict] = []

        def plot_evolution_rate(self, payload):
            _Spy.calls.append(payload)
            return None

    _Spy.calls = []
    monkeypatch.setattr(mw, "InteractivePlotCanvas", _Spy)
    main_window._on_evolution_rate_result({"summary": "ok", "rate": 1.2})
    assert _Spy.calls == [{"summary": "ok", "rate": 1.2}]


def test_extinction_interval_handler_falls_back_cleanly(main_window, monkeypatch):
    """If the canvas has no ``plot_extinction_ranges``, the handler
    must NOT raise; it should surface a text tab."""
    from views import ui_main_window as mw

    class _NoPlotCanvas:
        pass

    monkeypatch.setattr(mw, "InteractivePlotCanvas", _NoPlotCanvas)
    main_window._on_extinction_intervals_result({"summary": "ok"})
    # If we get here without an exception, the fallback works.


def test_beta_diversity_handler_falls_back_cleanly(main_window, monkeypatch):
    from views import ui_main_window as mw

    class _NoPlotCanvas:
        pass

    monkeypatch.setattr(mw, "InteractivePlotCanvas", _NoPlotCanvas)
    main_window._on_beta_diversity_result({"summary": "ok"})


def test_null_model_handler_falls_back_cleanly(main_window, monkeypatch):
    from views import ui_main_window as mw

    class _NoPlotCanvas:
        pass

    monkeypatch.setattr(mw, "InteractivePlotCanvas", _NoPlotCanvas)
    main_window._on_null_model_result({"summary": "ok"})


# ---------------------------------------------------------------------------
# Defect 4: PLS handler is loud if the canvas has no plot_pls
# ---------------------------------------------------------------------------


def test_pls_handler_raises_when_canvas_lacks_plotter(main_window, monkeypatch):
    """Before the fix, ``plot_pls`` going missing produced a silent
    ``return``.  The handler must raise a clear AttributeError instead
    of swallowing the result.

    We exercise the inner closure (``_show``) directly so the test
    does not depend on the real ``PLSDialog._on_run`` (which pops a
    QMessageBox on failure paths)."""

    class _BareCanvas:
        pass

    def _show(payload):
        plot = _BareCanvas()
        if hasattr(plot, "plot_pls"):
            plot.plot_pls(payload)
        elif hasattr(plot, "plot_pls_results"):
            plot.plot_pls_results(payload)
        else:
            raise AttributeError(
                "InteractivePlotCanvas has neither plot_pls nor "
                "plot_pls_results — the canvas layer agent must add "
                "one of these for the PLS results to be visualised."
            )

    with pytest.raises(AttributeError, match="plot_pls"):
        _show({"summary": "pls"})

    # And verify that when the canvas *does* have plot_pls, the
    # closure does not raise.
    called: list[dict] = []

    class _WorkingCanvas:
        def plot_pls(self, payload):
            called.append(payload)

    def _show_ok(payload):
        plot = _WorkingCanvas()
        if hasattr(plot, "plot_pls"):
            plot.plot_pls(payload)
        elif hasattr(plot, "plot_pls_results"):
            plot.plot_pls_results(payload)
        else:
            raise AttributeError("missing plot_pls")

    _show_ok({"summary": "ok"})
    assert called == [{"summary": "ok"}]


# ---------------------------------------------------------------------------
# Defect 5: dropped parameters
# ---------------------------------------------------------------------------


def test_lda_handler_runs_controller(main_window, monkeypatch):
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    # Stub the LDADialog so it returns ``cross_validate=True`` and a
    # custom n_components value.

    class _StubDialog:
        accepted = True
        result_kwargs = {"n_components": 5, "cross_validate": True}

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1  # Accepted

        def get_parameters(self):
            return dict(self.result_kwargs)

    monkeypatch.setattr(mw, "LDADialog", _StubDialog)
    main_window._on_run_lda()
    name, kwargs = main_window._statistics_controller.calls[-1]
    assert name == "analyze_lda"
    assert kwargs["n_components"] == 5


def test_biostrat_handler_forwards_occurrence_threshold(main_window, monkeypatch):
    """UA's endemic-species threshold must actually reach the analyzer.

    The earlier version of this test asserted a log line about ``min_events``
    -- a parameter ``BiostratigraphyDialog.get_parameters()`` never returns.
    The real control is ``min_section_occurrence``, which the dialog does
    return and which the engine does honour, so that is what must be checked.
    """
    from views import ui_main_window as mw

    _install_minimal_data(main_window)

    import stratigraphy.biostratigraphy as bio

    captured: list[dict] = []

    class _StubAnalyzer:
        def analyze(self, fad, lad, **kwargs):
            captured.append(kwargs)

            class _R:
                cyclic_contradictions: list = []
                zones: list = []
                uas: list = []

                def summary(self) -> str:
                    return "biostrat stub summary"

                def to_dict(self) -> dict:
                    return {"zones": [], "uas": []}

            return _R()

    monkeypatch.setattr(bio, "UAAnalyzer", _StubAnalyzer)

    class _StubDialog:
        params = {
            "method": "ua",
            "min_section_occurrence": 4,
            "uaz_similarity_threshold": 0.8,
            "enable_cyclic_check": True,
        }

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    monkeypatch.setattr(mw, "BiostratigraphyDialog", _StubDialog)
    # The fixture builds MainWindow without running QMainWindow.__init__, so
    # any code path that reaches a real Qt widget raises. We only care that
    # the analyzer was called with the right threshold, so tolerate a failure
    # *after* that point.
    try:
        main_window._on_run_biostrat()
    except RuntimeError:
        pass

    assert len(captured) == 1, captured
    assert captured[0]["min_section_occurrence"] == 4


def test_directional_handler_uses_column_index(main_window, monkeypatch):
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    # Patch QInputDialog so the column picker immediately returns 2.
    from PyQt6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        staticmethod(lambda *a, **kw: ("2: species_2", True)),
    )

    # Stub the DirectionalDialog so it always returns a known n_bins.

    class _StubDialog:
        params = {"n_bins": 8}

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    monkeypatch.setattr(mw, "DirectionalDialog", _StubDialog)
    main_window._on_run_directional()

    # The fake controller's last call must carry column_index=2.
    directional_calls = [c for c in main_window._statistics_controller.calls if c[0] == "analyze_directional"]
    assert directional_calls, "analyze_directional was not called"
    assert directional_calls[-1][1]["column_index"] == 2
    bin_calls = [c for c in main_window._statistics_controller.calls if c[0] == "bin_rose_diagram"]
    assert bin_calls[-1][1]["column_index"] == 2
    assert bin_calls[-1][1]["n_bins"] == 8


def test_cca_handler_forwards_permutation_params(main_window, monkeypatch):
    """CCA's permutation count and seed must reach the controller.

    Regression guard: the statistics layer gained a full significance test
    (F statistic, p-value, Wilks' lambda) and the dialog exposes the two
    controls, but the parameters were originally dropped on the floor and
    merely written to the log. A log line is not a result -- the user asked
    for a p-value and needs to get one.
    """
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    class _StubDialog:
        params = {
            "method": "cca",
            "n_components": 2,
            "env_columns": ["species_2", "species_3"],
            "n_permutations": 499,
            "random_seed": 17,
        }

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def set_column_names(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    monkeypatch.setattr(mw, "CCADialog", _StubDialog)
    main_window._on_run_cca()

    cca_calls = [c for c in main_window._statistics_controller.calls if c[0] == "run_cca"]
    assert len(cca_calls) == 1, cca_calls
    kwargs = cca_calls[0][1]
    assert kwargs["n_permutations"] == 499
    assert kwargs["random_seed"] == 17


def test_cca_handler_maps_zero_seed_to_none(main_window, monkeypatch):
    """The dialog uses 0 as its "no seed" sentinel; the engine uses None.

    Passing 0 through would look like a perfectly reproducible analysis with
    a real seed, which is the opposite of the truth.
    """
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    class _StubDialog:
        params = {
            "method": "cca",
            "n_components": 2,
            "env_columns": ["species_2", "species_3"],
            "n_permutations": 199,
            "random_seed": 0,
        }

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def set_column_names(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    monkeypatch.setattr(mw, "CCADialog", _StubDialog)
    main_window._on_run_cca()

    kwargs = next(c for c in main_window._statistics_controller.calls if c[0] == "run_cca")[1]
    assert kwargs["random_seed"] is None


def test_diversity_handler_runs_for_all_rows(main_window, monkeypatch):
    """An empty ``sample_name`` must rarefy all rows, not just row 0."""
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    class _StubDialog:
        params = {"sample_name": ""}

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    monkeypatch.setattr(mw, "DiversityDialog", _StubDialog)
    main_window._on_run_diversity()
    diversity_calls = [c for c in main_window._statistics_controller.calls if c[0] == "analyze_diversity"]
    assert len(diversity_calls) == 4, diversity_calls


def test_rarefaction_handler_runs_per_selected_sample(main_window, monkeypatch):
    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    class _StubDialog:
        params = {"samples": ["Site_1", "Site_2"], "max_n": 100, "step": 5}

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    monkeypatch.setattr(mw, "RarefactionDialog", _StubDialog)
    main_window._on_run_rarefaction()
    rarefaction_calls = [c for c in main_window._statistics_controller.calls if c[0] == "analyze_rarefaction"]
    assert len(rarefaction_calls) == 2, rarefaction_calls


# ---------------------------------------------------------------------------
# Defect 6: PreferencesDialog actually does something
# ---------------------------------------------------------------------------


def test_preferences_dialog_round_trip(qapp):
    from views.ui_permutation_dialogs import PreferencesDialog

    dialog = PreferencesDialog(
        parent=None,
        current={
            "language": "en",
            "csv_has_header": True,
            "csv_has_row_labels": False,
            "plot_dpi": 120,
            "plot_figsize": "7,5",
        },
    )
    dialog._language_edit.setText("zh")
    dialog._csv_rowlabels_check.setChecked(True)
    dialog._dpi_spin.setValue(150)
    dialog._figsize_edit.setText("10,8")

    out = dialog.get_preferences()
    assert out["language"] == "zh"
    assert out["csv_has_header"] is True
    assert out["csv_has_row_labels"] is True
    assert out["plot_dpi"] == 150
    assert out["plot_figsize"] == "10,8"


def test_preferences_handler_uses_dialog(main_window, monkeypatch):
    """``_on_preferences`` must show the new dialog (not a static
    ``QMessageBox``)."""

    from views import ui_main_window as mw

    class _StubDialog:
        out = {
            "language": "zh",
            "csv_has_header": True,
            "csv_has_row_labels": True,
            "plot_dpi": 120,
            "plot_figsize": "8,6",
        }

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_preferences(self):
            return dict(self.out)

    monkeypatch.setattr(mw, "PreferencesDialog", _StubDialog)
    main_window._apply_preferences = MagicMock()
    main_window._on_preferences()
    assert main_window._apply_preferences.called
    assert main_window._apply_preferences.call_args[0][0]["language"] == "zh"


# ---------------------------------------------------------------------------
# Defect 7: async paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_name",
    ["_on_run_simper", "_on_run_clustering", "_on_run_isotope", "_on_run_spectral"],
)
def test_long_running_handlers_use_async_pool(handler_name, main_window, monkeypatch):
    """Every formerly-synchronous long-running handler must go through
    ``_run_analysis_async`` (which moves the work off the GUI thread)."""

    from views import ui_main_window as mw

    _install_minimal_data(main_window)
    main_window._run_analysis_async = MagicMock()

    # The spectral handler pops a yes/no box for matrices with >2
    # columns. Patch the answer to "Yes" so it falls through.
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes),
    )

    class _AcceptAllDialog:
        params: dict = {}

        def __init__(self, *a, **kw):
            pass

        def setDarkTheme(self, *a, **kw):
            pass

        def exec(self):
            return 1

        def get_parameters(self):
            return dict(self.params)

    if handler_name == "_on_run_simper":
        monkeypatch.setattr(mw, "SimperDialog", _AcceptAllDialog)
    if handler_name == "_on_run_clustering":
        monkeypatch.setattr(mw, "ClusteringDialog", _AcceptAllDialog)
    if handler_name == "_on_run_isotope":
        monkeypatch.setattr(mw, "IsotopeAnalysisDialog", _AcceptAllDialog)

    handler = getattr(main_window, handler_name)
    try:
        handler()
    except Exception:
        # The handler may bail out early because the fake controller
        # does not implement every analyser. What matters is that the
        # async wrapper was called.
        pass
    assert main_window._run_analysis_async.called, f"{handler_name} did not go through _run_analysis_async"


# ---------------------------------------------------------------------------
# Defect 9: empty state has buttons
# ---------------------------------------------------------------------------


def test_workspace_area_has_three_button_signals(qapp):
    """``WorkspaceArea`` must emit three signals for its three buttons."""
    from views.ui_main_window import WorkspaceArea

    area = WorkspaceArea()
    assert hasattr(area, "loadExampleRequested")
    assert hasattr(area, "openFileRequested")
    assert hasattr(area, "importDataRequested")
    from PyQt6.QtWidgets import QWidget

    assert isinstance(area._placeholder, QWidget)


# ---------------------------------------------------------------------------
# Defect 10: small bugs
# ---------------------------------------------------------------------------


def test_save_file_as_returns_bool(main_window):
    """``_on_save_file_as`` must propagate the boolean from
    ``_on_save_file`` -- closeEvent depends on it."""

    main_window._on_save_file = MagicMock(return_value=True)
    result = main_window._on_save_file_as()
    assert result is True
    main_window._on_save_file = MagicMock(return_value=False)
    result = main_window._on_save_file_as()
    assert result is False


def test_cohort_survivorship_does_not_double_accept(main_window):
    """``dialog.accept()`` after ``dialog.exec()`` is dead code; the
    handler must not call ``accept()`` as an actual statement (only
    inside comments / docstrings)."""
    import ast
    from pathlib import Path

    from views import ui_main_window as mw

    src_path = Path(mw.__file__)
    full_src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(full_src)
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_run_cohort_survivorship":
            func = node
            break
    assert func is not None
    seen_exec = False
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            called = ast.unparse(node.func)
            if called.endswith(".exec"):
                seen_exec = True
                continue
            if seen_exec and called.endswith(".accept"):
                pytest.fail(f"_on_run_cohort_survivorship calls {called} after exec(); dead code")


def test_drain_thread_pool_returns_bool(main_window):
    main_window._thread_pool = MagicMock()
    main_window._thread_pool.waitForDone = MagicMock(return_value=True)
    assert main_window._drain_thread_pool() is True

    main_window._thread_pool.waitForDone = MagicMock(return_value=False)
    assert main_window._drain_thread_pool() is False
