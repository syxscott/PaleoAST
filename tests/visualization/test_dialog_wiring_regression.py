"""UI-level regression tests for the analysis-dialog wiring fixes.

These tests assert that the values the user sets in each dialog are
actually forwarded to the engine / controller, instead of being silently
dropped. Each test instantiates a dialog with a small ``FakeController``
(or stubs the analyzer module) and inspects the call the dialog makes
when ``_on_run`` is fired.

They cover:

* Defect 1 — ``EvolutionRateDialog`` forwards ``seed`` and
  ``n_bootstrap`` to ``EvolutionRateAnalyzer.analyze``.
* Defect 2 — ``EvolutionRateDialog`` exposes the engine's actual model
  names (``random_walk`` / ``directional`` / ``stasis``) and no longer
  the misleading BM/OU labels.
* Defect 3 — ``CCADialog`` exposes ``n_permutations`` and
  ``random_seed`` in ``get_parameters``.
* Defect 4 — every flag we exposed was either wired through to a real
  engine parameter, or its dead UI control was removed; the dialogs
  still return the parameters they advertise.

The tests use the same headless PyQt6 pattern as
``tests/controllers/test_allometry_pls_ui_wiring.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6", reason="UI dialog tests require PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-dialog-wiring-tests"])
    return app


@pytest.fixture(autouse=True)
def _suppress_messageboxes(monkeypatch):
    """Suppress modal ``QMessageBox`` popups in tests.

    Several dialogs end their ``_on_run`` with a modal
    ``QMessageBox.information(...)`` whose only purpose is to tell the
    user the analysis finished.  In a headless test that modal blocks
    the event loop forever (no human to click OK).  Replace every
    popup-style call with a no-op so the dialog can return and emit
    its signal.
    """
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.StandardButton.Ok)


# ---------------------------------------------------------------------------
# Defect 1 + 2 — EvolutionRateDialog forwards seed / n_bootstrap and uses
# the real model names.
# ---------------------------------------------------------------------------


def test_evolution_rate_dialog_forwards_seed_and_bootstrap(qapp, monkeypatch):
    """Defect 1: ``seed`` and ``n_bootstrap`` must reach the engine."""
    from views import ui_evolution_rate_dialogs

    captured: dict[str, Any] = {}

    class FakeAnalyzer:
        def analyze(self, **kwargs):
            captured.update(kwargs)
            result = MagicMock(summary=lambda: "fake-summary", to_dict=lambda: {"k": "fake"})
            return result

    fake_module = MagicMock()
    fake_module.EvolutionRateAnalyzer = FakeAnalyzer
    monkeypatch.setattr(ui_evolution_rate_dialogs, "EvolutionRateAnalyzer", FakeAnalyzer, raising=False)
    # The dialog does ``from morphometrics.evolution_rate import EvolutionRateAnalyzer``
    # inside the slot; patch the symbol in the import cache too.
    import sys

    sys.modules.setdefault("morphometrics", MagicMock())
    monkeypatch.setitem(sys.modules, "morphometrics.evolution_rate", fake_module)

    dialog = ui_evolution_rate_dialogs.EvolutionRateDialog(parent=None)
    dialog._seed_spin.setValue(777)
    dialog._bootstrap_spin.setValue(313)
    dialog._confidence_spin.setValue(0.9)
    dialog._trait_input.setPlainText("1\n2\n3\n4\n5\n6\n7\n8")
    dialog._on_run()

    assert captured, "EvolutionRateDialog._on_run did not call EvolutionRateAnalyzer.analyze"
    assert int(captured["seed"]) == 777, f"seed not forwarded: {captured}"
    assert int(captured["n_bootstrap"]) == 313, f"n_bootstrap not forwarded: {captured}"
    # confidence_level must still be forwarded (was already wired).
    assert abs(float(captured["confidence_level"]) - 0.9) < 1e-9


def test_evolution_rate_models_combo_uses_engine_names(qapp):
    """Defect 2: the models combo must advertise the real engine names."""
    from views.ui_evolution_rate_dialogs import EvolutionRateDialog

    dialog = EvolutionRateDialog(parent=None)
    items = [dialog._models_combo.itemText(i) for i in range(dialog._models_combo.count())]
    # Must NOT contain "BM" or "OU" anywhere — those are phylogenetic
    # model names, not what this stratigraphic analyzer fits.
    for text in items:
        assert "BM" not in text
        assert " OU " not in text
    # And must contain the actual engine model names.
    flat = " | ".join(items)
    assert "Random walk" in flat
    assert "Directional" in flat
    assert "Stasis" in flat


def test_evolution_rate_results_ready_emits_dict(qapp, monkeypatch):
    """Defect 5: ``resultsReady`` must emit a ``dict``."""
    from views import ui_evolution_rate_dialogs

    class FakeAnalyzer:
        def analyze(self, **kwargs):
            result = MagicMock(summary=lambda: "fake-summary")
            result.to_dict = MagicMock(return_value={"best_model": "random_walk"})
            return result

    import sys

    monkeypatch.setitem(
        sys.modules,
        "morphometrics.evolution_rate",
        MagicMock(EvolutionRateAnalyzer=FakeAnalyzer),
    )

    dialog = ui_evolution_rate_dialogs.EvolutionRateDialog(parent=None)
    captured: list[Any] = []
    dialog.resultsReady.connect(captured.append)
    dialog._trait_input.setPlainText("1\n2\n3\n4\n5\n6\n7")
    dialog._on_run()

    assert captured, "EvolutionRateDialog did not emit resultsReady"
    assert isinstance(captured[-1], dict)
    assert captured[-1] == {"best_model": "random_walk"}


# ---------------------------------------------------------------------------
# Defect 3 — CCADialog exposes permutation test parameters.
# ---------------------------------------------------------------------------


def test_cca_dialog_exposes_permutation_controls(qapp):
    from views.ui_dialogs import CCADialog

    dialog = CCADialog(parent=None)
    assert hasattr(dialog, "_n_perm_spin"), "CCADialog missing n_permutations spinbox"
    assert hasattr(dialog, "_seed_spin"), "CCADialog missing random_seed spinbox"
    params = dialog.get_parameters()
    assert "n_permutations" in params, f"get_parameters missing n_permutations: {params}"
    assert "random_seed" in params, f"get_parameters missing random_seed: {params}"
    assert isinstance(params["n_permutations"], int)
    assert isinstance(params["random_seed"], int)
    assert params["n_permutations"] == dialog._n_perm_spin.value()
    assert params["random_seed"] == dialog._seed_spin.value()


# ---------------------------------------------------------------------------
# Defect 4 — Dead UI controls: either removed, or marked; nothing silently
# dropped. We assert the *interface* (get_parameters) matches reality.
# ---------------------------------------------------------------------------


def test_lda_dialog_has_no_silent_cross_validate_flag(qapp):
    """Defect 4a: ``LDADialog.get_parameters`` must NOT advertise
    ``cross_validate`` — the controller does not consume it."""
    from views.ui_dialogs import LDADialog

    dialog = LDADialog(parent=None)
    params = dialog.get_parameters()
    assert "cross_validate" not in params, (
        f"LDADialog still exposes a cross_validate flag the controller ignores: {params}"
    )
    assert not hasattr(dialog, "_cv_check"), "LDADialog still has the dead _cv_check checkbox."


def test_biostrat_dialog_has_no_silent_min_events_flag(qapp):
    """Defect 4b: ``BiostratigraphyDialog.get_parameters`` must NOT
    advertise ``min_events`` — the UA analyzer has no such knob."""
    from views.ui_dialogs import BiostratigraphyDialog

    dialog = BiostratigraphyDialog(parent=None)
    assert not hasattr(dialog, "_min_events_spin"), "BiostratigraphyDialog still has the dead _min_events_spin."
    params = dialog.get_parameters()
    assert "min_events" not in params, (
        f"BiostratigraphyDialog still exposes a min_events flag the UA analyzer ignores: {params}"
    )


def test_markov_dialog_returns_useful_parameters(qapp):
    """Defect 4c: ``MarkovDialog`` used to return ``{}``.  It must now
    surface the facies-name knob that ``MarkovAnalyzer.analyze``
    actually accepts."""
    from views.ui_dialogs import MarkovDialog

    dialog = MarkovDialog(parent=None)
    dialog._facies_names_edit.setText("A,B ,C")
    params = dialog.get_parameters()
    assert params == {"facies_names": ["A", "B", "C"]}
    # Empty input collapses to None (engine default).
    dialog._facies_names_edit.setText("")
    assert dialog.get_parameters() == {"facies_names": None}


def test_pic_dialog_has_no_silent_use_branch_lengths_checkbox(qapp):
    """Defect 4d: ``PICDialog`` must not advertise a
    ``use_branch_lengths`` toggle — the PIC formula requires branch
    lengths, and the engine has no such switch."""
    from views.ui_pcm_dialogs import PICDialog

    dialog = PICDialog(parent=None)
    assert not hasattr(dialog, "_check_branch_lengths"), "PICDialog still has the dead _check_branch_lengths checkbox."


def test_allometry_dialog_has_no_silent_confidence_level_spin(qapp):
    """Defect 4e: ``AllometryDialog`` must not advertise a confidence-
    level control — ``analyze_allometry`` ignores the parameter."""
    from views.ui_allometry_dialogs import AllometryDialog

    class FakeController:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []
            self.result = MagicMock(summary=lambda: "ok", to_dict=lambda: {"k": "v"})

        def analyze_allometry(self, **kwargs):
            self.calls.append(kwargs)
            return self.result

    ctrl = FakeController()
    dialog = AllometryDialog(parent=None, controller=ctrl)
    assert not hasattr(dialog, "_ci_spin"), "AllometryDialog still has the dead _ci_spin."
    dialog._on_run()
    assert ctrl.calls, "AllometryDialog._on_run did not call the controller"
    assert "confidence_level" not in ctrl.calls[-1], (
        f"Forwarded a confidence_level argument the engine ignores: {ctrl.calls[-1]}"
    )


def test_beta_diversity_dialog_has_no_silent_transform_or_pairwise(qapp):
    """Defect 4f: ``BetaDiversityDialog`` must not advertise the
    transform / pairwise-display toggles — the engine ignores them."""
    from views.ui_beta_diversity_dialogs import BetaDiversityDialog

    dialog = BetaDiversityDialog(parent=None)
    assert not hasattr(dialog, "_transform_combo"), "BetaDiversityDialog still has the dead _transform_combo."
    assert not hasattr(dialog, "_show_pairwise_check"), "BetaDiversityDialog still has the dead _show_pairwise_check."


def test_coverage_rarefaction_dialog_routes_to_hill_analyzer(qapp, monkeypatch):
    """Defect 4g: ``CoverageRarefactionDialog`` must forward
    ``n_points``, ``n_bootstrap`` and ``seed`` to the
    ``coverage_rarefaction_hill`` engine (which actually consumes them).
    The legacy ``analyze`` wrapper silently drops the bootstrap count,
    so the dialog must NOT route through it."""
    from views import ui_beta_diversity_dialogs

    captured: dict[str, Any] = {}

    class FakeAnalyzer:
        analyze_called = False

        def coverage_rarefaction_hill(self, **kwargs):
            captured.update(kwargs)
            r = MagicMock(summary=lambda: "hill-summary", to_dict=lambda: {"k": "hill"})
            return r

        def analyze(self, **kwargs):
            FakeAnalyzer.analyze_called = True
            return MagicMock(summary=lambda: "analyze-summary", to_dict=lambda: {"k": "x"})

    fake_module = MagicMock(CoverageRarefactionAnalyzer=FakeAnalyzer)
    import sys

    monkeypatch.setitem(sys.modules, "ecology.beta_diversity", fake_module)

    dialog = ui_beta_diversity_dialogs.CoverageRarefactionDialog(parent=None)
    dialog._endpoint_spin.setValue(77)
    dialog._n_boot_spin.setValue(123)
    dialog._seed_spin.setValue(99)
    dialog._data_input.setPlainText("S1\t1\t2\t3\nS2\t4\t5\t0\nS3\t2\t3\t1")
    dialog._on_run()

    assert captured, "Dialog did not call coverage_rarefaction_hill"
    assert int(captured["n_points"]) == 77
    assert int(captured["n_bootstrap"]) == 123
    assert int(captured["seed"]) == 99
    assert FakeAnalyzer.analyze_called is False, (
        "Dialog routed through .analyze(), which silently ignores n_bootstrap and seed."
    )


# ---------------------------------------------------------------------------
# Defect 5 — resultsReady emits a dict on every dialog that advertises it.
# ---------------------------------------------------------------------------


def _make_fake_controller_with_result(result_dict: dict[str, Any]):
    class _Fake:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []
            self.result = MagicMock(summary=lambda: "ok")
            self.result.to_dict = MagicMock(return_value=result_dict)

        def analyze_pic(self, tree, traits):
            self.calls.append({"tree": tree, "traits": traits})
            return self.result

        def analyze_ancestral_states(self, tree, traits, model="bm"):
            self.calls.append({"tree": tree, "traits": traits, "model": model})
            return self.result

        def analyze_phylogenetic_signal(self, tree, traits, n_randomizations=999):
            self.calls.append({"tree": tree, "traits": traits, "n_randomizations": n_randomizations})
            return self.result

        def analyze_phylo_anova(self, tree, traits, groups, n_permutations=999):
            self.calls.append({"tree": tree, "traits": traits, "groups": groups, "n_permutations": n_permutations})
            return self.result

        def analyze_allometry(self, **kwargs):
            self.calls.append(kwargs)
            return self.result

        def analyze_pls(self, **kwargs):
            self.calls.append(kwargs)
            return self.result

    return _Fake()


def test_beta_diversity_emits_results_ready_dict(qapp, monkeypatch):
    """Defect 5: ``BetaDiversityDialog.resultsReady`` must emit a dict."""
    from views import ui_beta_diversity_dialogs

    captured_run: dict[str, Any] = {}

    class FakeAnalyzer:
        def decompose_beta_diversity(self, **kwargs):
            captured_run.update(kwargs)
            r = MagicMock(summary=lambda: "beta-summary", to_dict=lambda: {"k": "beta"})
            return r

    import sys

    monkeypatch.setitem(
        sys.modules,
        "ecology.beta_diversity",
        MagicMock(BetaDiversityAnalyzer=FakeAnalyzer, CoverageRarefactionAnalyzer=FakeAnalyzer),
    )

    dialog = ui_beta_diversity_dialogs.BetaDiversityDialog(parent=None)
    captured_signal: list[Any] = []
    dialog.resultsReady.connect(captured_signal.append)
    dialog._data_input.setPlainText("S1\t1\t2\t3\nS2\t4\t5\t0\nS3\t2\t3\t1")
    dialog._on_run()

    assert captured_signal, "BetaDiversityDialog did not emit resultsReady"
    assert isinstance(captured_signal[-1], dict)
    assert captured_signal[-1] == {"k": "beta"}


def test_null_model_emits_results_ready_dict(qapp, monkeypatch):
    from views import ui_null_model_dialogs

    class FakeAnalyzer:
        def analyze(self, **kwargs):
            r = MagicMock(summary=lambda: "null-summary", to_dict=lambda: {"k": "null"})
            return r

    import sys

    monkeypatch.setitem(
        sys.modules,
        "ecology.null_models",
        MagicMock(NullModelAnalyzer=FakeAnalyzer),
    )

    dialog = ui_null_model_dialogs.NullModelDialog(parent=None)
    captured: list[Any] = []
    dialog.resultsReady.connect(captured.append)
    dialog._data_input.setPlainText("sp1\t1\t0\t1\nsp2\t0\t1\t1\nsp3\t1\t1\t0")
    dialog._on_run()

    assert captured, "NullModelDialog did not emit resultsReady"
    assert isinstance(captured[-1], dict)
    assert captured[-1] == {"k": "null"}


def test_extinction_emits_results_ready_dict(qapp, monkeypatch):
    from views import ui_extinction_dialogs

    class FakeAnalyzer:
        def analyze(self, **kwargs):
            r = MagicMock(summary=lambda: "ext-summary", to_dict=lambda: {"k": "ext"})
            return r

    import sys

    monkeypatch.setitem(
        sys.modules,
        "stratigraphy.extinction",
        MagicMock(ExtinctionIntervalAnalyzer=FakeAnalyzer),
    )

    dialog = ui_extinction_dialogs.ExtinctionIntervalDialog(parent=None)
    captured: list[Any] = []
    dialog.resultsReady.connect(captured.append)
    dialog._lad_list.setPlainText("12\n9\n7\n4")
    dialog._on_run()

    assert captured, "ExtinctionIntervalDialog did not emit resultsReady"
    assert isinstance(captured[-1], dict)
    assert captured[-1] == {"k": "ext"}
