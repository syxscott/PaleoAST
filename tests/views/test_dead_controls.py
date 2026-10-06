# =============================================================================
# Test: no dead interactive controls
# =============================================================================
"""
A ``QComboBox`` (or ``QCheckBox``, ``QSpinBox``...) that is constructed, given
its items and added to a layout, but whose value is never read, is worse than
no control at all: it looks like a choice and silently has none.

There have been two of these in this codebase, and they were the same defect:

* the R plotting **palette** dropdown. The dialog produced ``r_palette`` and
  handed it to ``_apply_preferences``, which looped over five other keys and
  did not include it, and ``_get_preferences_state`` did not read it back. So
  the key was never written to QSettings, never read, and choosing anything
  but the default did nothing at all;
* the allometry dialog's **"RMA (reduced major axis)"** combo.
  ``_method_combo`` was constructed, filled with OLS and RMA, added to the
  layout -- and then nothing ever read ``currentIndex()``. Picking RMA ran the
  OLS regression and labelled the output RMA.

The second is the worse of the two: the numbers come out looking correct.

Neither was caught by a behavioural test, because both dialogs still run and
still return a result. They are found here instead, structurally: a control
attribute that appears exactly once in its file, at its own construction, is
dead. That is checkable without running the GUI.

The allowlist is for controls that are genuinely decorative or that are read
through a mechanism this check cannot see. Add to it with a comment saying why
-- an entry without a reason is a dead control waiting to happen.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Widget classes whose selection/value a user is meant to change.
_WIDGET_TYPES = (
    "QComboBox",
    "QCheckBox",
    "QRadioButton",
    "QSpinBox",
    "QDoubleSpinBox",
    "QSlider",
    "QLineEdit",
)

# control attribute -> why reading it is not visible to this check
ALLOWLIST: dict[str, str] = {}


_REPO = REPO

# Methods whose call means "the user chose something". A control set up but
# never asked for its value through one of these is dead.
_VALUE_GETTERS = {
    # combo boxes
    "currentText",
    "currentIndex",
    "currentData",
    "currentTextChanged",
    "itemText",
    "itemData",
    "count",
    "isEditable",
    # check / radio
    "isChecked",
    "checkedState",
    "checkState",
    "isCheckable",
    "autoExclusive",
    # spin boxes and sliders
    "value",
    "minimum",
    "maximum",
    "singleStep",
    # line edits and plain widgets
    "text",
    "displayText",
    "isVisible",
    "isEnabled",
    "isHidden",
    "isReadOnly",
    "placeholderText",
}

# Calls that place a widget rather than ask it anything. A control passed here
# has been shown to the user and nothing more.
_STRUCTURAL_CALLS = {
    "addWidget",
    "addRow",
    "addLayout",
    "addItem",
    "setLayout",
    "insertWidget",
    "insertLayout",
    "setCellWidget",
    "setCellLayout",
    "addStretch",
    "addSpacing",
    "addSeparator",
    "append",
    "insert",
    "removeWidget",
}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "views/*.py"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout.split()
    return [REPO / rel for rel in out]


def _self_attr(node: ast.AST) -> str | None:
    """``self.foo`` -> ``"foo"``; anything else -> None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _dead_controls(path: Path) -> set[str]:
    """Widgets built in this file whose value is never queried in this file.

    The naive version of this check -- "the attribute name occurs only once" --
    does not work, and the mutation test is what showed it: a dead combo still
    occurs three times, in the assignment, in ``.addItems()`` and in the
    ``addRow`` that lays it out. What separates them from a live control is not
    how often the name appears but whether anything ever asks it what its value
    is.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    built: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            type_name = (
                call.func.attr
                if isinstance(call.func, ast.Attribute)
                else (call.func.id if isinstance(call.func, ast.Name) else "")
            )
            if type_name in _WIDGET_TYPES:
                for target in node.targets:
                    if (attr := _self_attr(target)) and attr.startswith("_"):
                        built.add(attr)

    read: set[str] = set()
    for node in ast.walk(tree):
        # self.<attr>.<getter>(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _VALUE_GETTERS:
                if attr := _self_attr(node.func.value):
                    read.add(attr)
        # self.<attr> handed to something else -- but NOT to a layout method.
        # ``opts_layout.addRow(_("Regression method:"), self._method_combo)``
        # is placement, not a question; counting it as a read is what made this
        # check report the allometry combo as alive in the first version.
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if not any(callee.endswith("." + m) for m in _STRUCTURAL_CALLS):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if attr := _self_attr(arg):
                        read.add(attr)
        # bound: handler = self.<attr>.valueChanged  (no call, still a read)
        if isinstance(node, ast.Attribute):
            if (attr := _self_attr(node.value)) and node.attr.endswith("Changed"):
                read.add(attr)

    return {a for a in built if a not in read}


def test_no_dead_interactive_controls():
    """Every self._* widget this file builds must have its value read here.

    Would go red if a control like the allometry RMA combo came back: it is
    built, filled and laid out, and nothing ever asks it what it holds.
    """
    dead: list[str] = []
    for path in _tracked_files():
        for attr in sorted(_dead_controls(path)):
            if attr in ALLOWLIST:
                continue
            dead.append(f"{path.relative_to(REPO)}: {attr}")
    assert not dead, "controls built but never queried:\n  " + "\n  ".join(dead)


def test_allowlist_has_no_stale_entries():
    """An allowlist entry for a control that no longer exists is either a
    leftover or a name that drifted; either way it hides a future hit."""
    live: set[str] = set()
    for path in _tracked_files():
        live |= set(_dead_controls(path)) | set(ALLOWLIST)
    stale = sorted(k for k in ALLOWLIST if k not in live)
    assert not stale, f"allowlist entries for controls that no longer exist: {stale}"
