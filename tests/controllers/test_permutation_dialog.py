"""Regression tests for the new ``PermutationTestDialog``.

Covers Defect 2 (no parameter dialog for ANOSIM / PERMANOVA) and the
honest-degradation contract used by the new ``_on_*_result`` slots.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("PyQt6", reason="UI dialog tests require PyQt6")

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["paleoast-permutation-tests"])
    return app


def test_default_metric_is_bray_curtis(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    params = dialog.get_parameters()
    assert params["metric"] == "bray_curtis"
    assert params["n_permutations"] == 9999
    # Default seed sentinel is "empty" -> None.
    assert params["random_seed"] is None


def test_metric_choice_round_trip(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="PERMANOVA")
    # Move to the second item (Euclidean).
    dialog._metric_combo.setCurrentIndex(1)
    params = dialog.get_parameters()
    assert params["metric"] == "euclidean"


def test_n_permutations_clamped(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._n_perm_spin.setValue(500)
    params = dialog.get_parameters()
    assert params["n_permutations"] == 500


def test_seed_zero_translates_to_none(qapp):
    """The QSpinBox sentinel for "empty" is 0; it must become None so
    downstream code can match the no-seed convention used by stats/."""
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._seed_spin.setValue(0)
    params = dialog.get_parameters()
    assert params["random_seed"] is None


def test_seed_nonzero_passes_through(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._seed_spin.setValue(42)
    params = dialog.get_parameters()
    assert params["random_seed"] == 42


def test_accept_emits_runtime_warning_when_seed_is_empty(qapp):
    """Match the stats-layer convention: no seed ⇒ non-reproducible."""
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._seed_warning_check.setChecked(True)
    dialog._seed_spin.setValue(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dialog._on_accept()
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
        "Expected RuntimeWarning when no seed is set, got: " + ", ".join(str(w.message) for w in caught)
    )


def test_accept_suppresses_warning_when_seed_is_set(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._seed_spin.setValue(7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dialog._on_accept()
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught), (
        "Did not expect a RuntimeWarning when seed is set; got: " + ", ".join(str(w.message) for w in caught)
    )


def test_accept_suppresses_warning_when_checkbox_unchecked(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._seed_warning_check.setChecked(False)
    dialog._seed_spin.setValue(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dialog._on_accept()
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_slow_warning_appears_for_large_n(qapp):
    from views.ui_permutation_dialogs import PermutationTestDialog

    dialog = PermutationTestDialog(parent=None, title="ANOSIM")
    dialog._n_perm_spin.setValue(9999)
    assert "(may" in dialog._slow_warning_label.text() or "while" in dialog._slow_warning_label.text()
    dialog._n_perm_spin.setValue(500)
    assert dialog._slow_warning_label.text() == ""
