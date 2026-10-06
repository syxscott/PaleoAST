"""UI-level smoke tests for the Allometry / PLS dialog wiring (Defect 2).

Before the fix, ``AllometryDialog._on_run`` only showed a QMessageBox
saying "Allometry analysis requires GPA-aligned configurations…" and
``PLSDialog`` was never even instantiated from the main window.  After
the fix, clicking "Run Analysis" must actually invoke the controller
methods (``analyze_allometry`` / ``analyze_pls``) and forward the
user's parameter choices verbatim.

We exercise this with a tiny ``FakeController`` so the test does not
depend on the heavy statistics-engine import chain (PCA / PLS / GPA
engines all pull in PyQt6 through ``utils.event_bus``).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6", reason="UI dialog tests require PyQt6")

from PyQt6.QtWidgets import QApplication


class _FakeController:
    """Stub for ``StatisticsController`` that records the calls made on it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.allometry_result = MagicMock(summary=lambda: "allometry-summary", to_dict=lambda: {"k": "allometry"})
        self.pls_result = MagicMock(summary=lambda: "pls-summary", to_dict=lambda: {"k": "pls"})

    def analyze_allometry(self, **kwargs: Any) -> Any:
        self.calls.append(("analyze_allometry", kwargs))
        return self.allometry_result

    def analyze_pls(self, **kwargs: Any) -> Any:
        self.calls.append(("analyze_pls", kwargs))
        return self.pls_result


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-allometry-tests"])
    return app


def test_allometry_dialog_runs_controller(qapp):
    from views.ui_allometry_dialogs import AllometryDialog

    fake = _FakeController()
    dialog = AllometryDialog(parent=None, controller=fake)
    # Tick the "PCA reduction" checkbox and set the spin to 7 so we
    # can verify both branches propagate.
    dialog._use_pca_check.setChecked(True)
    dialog._n_components_spin.setValue(7)
    dialog._on_run()

    assert fake.calls, "AllometryDialog._on_run did not call the controller"
    name, kwargs = fake.calls[-1]
    assert name == "analyze_allometry"
    assert kwargs == {"n_components": 7}, (
        f"AllometryDialog forwarded the wrong kwargs: {kwargs}"
    )


def test_allometry_dialog_unchecks_pca_passes_none(qapp):
    from views.ui_allometry_dialogs import AllometryDialog

    fake = _FakeController()
    dialog = AllometryDialog(parent=None, controller=fake)
    dialog._use_pca_check.setChecked(False)
    dialog._n_components_spin.setValue(7)  # ignored when checkbox is off
    dialog._on_run()

    name, kwargs = fake.calls[-1]
    assert name == "analyze_allometry"
    assert kwargs == {"n_components": None}


def test_pls_dialog_runs_controller_with_division(qapp):
    from views.ui_allometry_dialogs import PLSDialog

    fake = _FakeController()
    dialog = PLSDialog(parent=None, controller=fake)
    dialog._division_combo.setCurrentIndex(2)  # Random Split
    dialog._n_components_spin.setValue(4)
    dialog._permutations_spin.setValue(50)
    dialog._seed_spin.setValue(99)
    dialog._on_run()

    name, kwargs = fake.calls[-1]
    assert name == "analyze_pls"
    assert kwargs["division"] == "random"
    assert kwargs["n_components"] == 4
    assert kwargs["permutations"] == 50
    assert kwargs["seed"] == 99


def test_allometry_dialog_emits_results_ready(qapp):
    from views.ui_allometry_dialogs import AllometryDialog

    fake = _FakeController()
    dialog = AllometryDialog(parent=None, controller=fake)
    captured: list[dict] = []
    dialog.resultsReady.connect(captured.append)

    dialog._on_run()

    assert captured, "AllometryDialog did not emit resultsReady after running"
    assert captured[0] == {"k": "allometry"}


def test_pls_dialog_emits_results_ready(qapp):
    from views.ui_allometry_dialogs import PLSDialog

    fake = _FakeController()
    dialog = PLSDialog(parent=None, controller=fake)
    captured: list[dict] = []
    dialog.resultsReady.connect(captured.append)

    dialog._on_run()

    assert captured, "PLSDialog did not emit resultsReady after running"
    assert captured[0] == {"k": "pls"}


def test_macroevolution_dialog_honours_tab_property(qapp):
    """``dialog.setProperty('tab', N)`` must switch the active tab."""
    from views.ui_macroevolution_dialogs import MacroevolutionDialog

    for tab_index in (0, 1, 2, 3):
        dialog = MacroevolutionDialog(controller=_FakeController(), parent=None)
        dialog.setProperty("tab", tab_index)
        # Re-run the setup so the property is read at show time.
        dialog._setup_ui()
        assert dialog._tabs.currentIndex() == tab_index, (
            f"tab={tab_index} did not switch the active tab "
            f"(got {dialog._tabs.currentIndex()})"
        )


def test_macroevolution_dialog_defaults_to_first_tab_without_property(qapp):
    from views.ui_macroevolution_dialogs import MacroevolutionDialog

    dialog = MacroevolutionDialog(controller=_FakeController(), parent=None)
    dialog._setup_ui()
    assert dialog._tabs.currentIndex() == 0


def test_evolution_rate_dialog_has_no_tree_input(qapp):
    """Defect 3: the analyzer does not support phylogenetic trees,
    so the dialog must not expose a Newick input that would be silently
    dropped."""
    from views.ui_evolution_rate_dialogs import EvolutionRateDialog

    dialog = EvolutionRateDialog(parent=None)
    # Look for the old tree widget name — it must be gone.
    assert not hasattr(dialog, "_tree_input"), (
        "EvolutionRateDialog still exposes a _tree_input widget that would "
        "be silently discarded by the analyzer."
    )
    # And a notice explaining the design.
    [dialog.findChild(type(lbl)) for lbl in [dialog]]
    # Robust check: read the dialog's children for a QLabel mentioning "phylogenetic".
    from PyQt6.QtWidgets import QLabel

    descendants = dialog.findChildren(QLabel)
    texts = [lbl.text() for lbl in descendants]
    assert any("phylogenetic" in t.lower() or "pcm" in t.lower() for t in texts), (
        f"EvolutionRateDialog has no notice about phylogenetic trees; labels: {texts[:5]}"
    )
