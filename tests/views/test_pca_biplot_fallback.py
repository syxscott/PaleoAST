"""Headless tests for two things the PCA path gets wrong.

``StatusBarWidget`` is exercised against a real Qt widget with
``QT_QPA_PLATFORM=offscreen``, so the assertions are about actual widget state
rather than about the source text.

What is NOT tested here, and cannot be without a display, is anything about how
a figure looks. That remains the largest unverified surface in this project.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
WINDOW = REPO / "views" / "ui_main_window.py"
CANVAS = REPO / "views" / "ui_plot_canvas.py"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestStatusBarWarningIsDistinct:
    """A request the app could not honour must not look like one it carried out.

    The "Show biplot" checkbox reaches the PCA slot, the canvas has no
    ``plot_pca_biplot``, and the user was getting a plain scatter with no
    message at all. These lock in that the warning path is visible *and* that
    the next informational message clears it -- a sequence that was easy to get
    wrong, since ``setInfo`` runs immediately after the plot is added.
    """

    def _bar(self, qapp):
        from views.ui_main_window import StatusBarWidget

        return StatusBarWidget()

    def test_setinfo_has_no_warning_prefix_or_style(self, qapp):
        bar = self._bar(qapp)
        bar.setInfo("PCA: 3 components, PC1+PC2 = 88.4%")
        assert not bar._info_label.text().startswith("⚠")
        assert bar._info_label.styleSheet() == ""

    def test_setwarning_marks_the_message(self, qapp):
        bar = self._bar(qapp)
        bar.setWarning("Show biplot was selected but there is no renderer")
        assert bar._info_label.text().startswith("⚠")
        assert "biplot" in bar._info_label.text()
        assert bar._info_label.styleSheet() != "", "a warning must be visually distinct"
        assert bar._info_label.toolTip() != ""

    def test_setinfo_clears_a_previous_warning(self, qapp):
        # The PCA slot sets the warning and then would otherwise have set the
        # component summary; the warning has to survive as the final state.
        bar = self._bar(qapp)
        bar.setWarning("something could not be honoured")
        bar.setInfo("PCA: 3 components")
        assert not bar._info_label.text().startswith("⚠")
        assert bar._info_label.styleSheet() == ""
        assert bar._info_label.toolTip() == ""


def _canvas_methods() -> set[str]:
    tree = ast.parse(CANVAS.read_text(encoding="utf-8"), filename=str(CANVAS))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(item.name)
    return out


def _guarded_calls(tree: ast.AST) -> set[int]:
    """Line numbers of ``plot.x(...)`` calls that sit inside a hasattr guard."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "hasattr" not in test_src:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "plot"
            ):
                guarded.add(inner.lineno)
    return guarded


class TestCanvasCallContract:
    """Every plot method the window calls must exist or be guarded.

    An unguarded call is an AttributeError the first time a user runs that
    analysis. A guarded call to a method that does not exist is a *silent*
    feature gap, which is the case worth knowing about, so both are reported.
    """

    def test_no_unguarded_calls_to_missing_canvas_methods(self):
        methods = _canvas_methods()
        tree = ast.parse(WINDOW.read_text(encoding="utf-8"), filename=str(WINDOW))
        guarded = _guarded_calls(tree)

        unguarded_missing = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "plot"
            ):
                continue
            name = node.func.attr
            if name in methods or node.lineno in guarded:
                continue
            unguarded_missing.append((node.lineno, name))

        assert not unguarded_missing, (
            "ui_main_window calls plot methods that InteractivePlotCanvas does "
            f"not define and does not hasattr-guard: {unguarded_missing}"
        )

    def test_show_biplot_option_is_reported_when_unrenderable(self):
        """The biplot fallback and the warning must live in the same function.

        Checking that ``setWarning`` is called *somewhere* in the file would
        pass even if the PCA slot stopped warning and some unrelated code path
        kept the method alive. The relationship is the thing worth pinning: the
        branch that falls back from a missing ``plot_pca_biplot`` is the branch
        that has to tell the user.
        """
        tree = ast.parse(WINDOW.read_text(encoding="utf-8"), filename=str(WINDOW))

        def function_containing(marker: str):
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)} | {
                    child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
                }
                if marker in names:
                    return node
            return None

        holder = function_containing("biplot_drawn")
        assert holder is not None, "no function tracks whether the biplot was drawn"
        assert "setWarning" in {
            child.func.attr
            for child in ast.walk(holder)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }, f"{holder.name} falls back to a score plot when the biplot renderer is missing but never warns the user"

    def test_dialog_still_offers_the_biplot_checkbox(self):
        """Records why the warning exists: the control is user-facing."""
        dialogs = (REPO / "views" / "ui_dialogs.py").read_text(encoding="utf-8")
        assert "Show biplot" in dialogs
        assert "_show_biplot_check" in dialogs
