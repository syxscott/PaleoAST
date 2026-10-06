# =============================================================================
# FILE: tests/test_ribbon_icon_variety.py
# =============================================================================
"""
Fail the build when ribbon buttons stop being distinguishable by icon.

WHAT WENT WRONG
---------------
Six adjacent data transforms -- Log, Sqrt, Hellinger, Z-Score, % Total and
Wisconsin -- were all created with the icon type ``"settings"``. Five icon
types covered 29 buttons across the whole ribbon. Within a tab, the icon
repeated the tab name and nothing else, so it carried no information while
still costing 24px per button; and six identical glyphs side by side are
worse than none, because they read as a mistake rather than as a choice.

Two properties are checked, and they are different in kind:

  1. every requested icon type is actually implemented. The engine's final
     ``else`` draws a plain filled circle, so an unrecognised name is not an
     error -- it is a silent fallback that looks like a real icon. That is
     why it needs an assertion.
  2. no ribbon group is mostly one icon. A group where every button shares a
     glyph is the exact shape of the original bug.

Parsed from the source rather than the live widgets: the ribbon is built
inside one large method, and instantiating a MainWindow to read it back
would pull the whole application into a unit test.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("PyQt6", reason="the ribbon is a Qt widget")

_MAIN = Path(__file__).resolve().parent.parent / "views" / "ui_main_window.py"
_SOURCE = _MAIN.read_text(encoding="utf-8")

ADDBUTTON = re.compile(r'addButton\(\s*"([a-z0-9_]+)"')
ADDGROUP = re.compile(r"addGroup\(")

# A group is only suspicious if its buttons collapse onto a single glyph. Two
# is allowed: a legitimately paired set (undo/redo, block A/block B) is not
# the problem being guarded against.
MAX_SHARE_OF_ONE_ICON = 2


def _implemented_icon_types() -> set[str]:
    tree = ast.parse(_SOURCE)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "create_icon":
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Compare)
                and isinstance(sub.left, ast.Name)
                and sub.left.id == "icon_type"
                and len(sub.ops) == 1
                and isinstance(sub.ops[0], ast.Eq)
                and isinstance(sub.comparators[0], ast.Constant)
                and isinstance(sub.comparators[0].value, str)
            ):
                found.add(sub.comparators[0].value)
    return found


def _glyph_icon_types() -> set[str]:
    from views.ui_main_window import _GLYPH_ICONS

    return set(_GLYPH_ICONS)


def _requested_icon_types() -> set[str]:
    return set(ADDBUTTON.findall(_SOURCE))


def test_every_requested_icon_is_implemented():
    """An unrecognised name silently becomes the fallback circle."""
    implemented = _implemented_icon_types() | _glyph_icon_types()
    missing = sorted(_requested_icon_types() - implemented)
    assert not missing, (
        f"ribbon requests icon types the engine does not implement: {missing}. "
        "create_icon ends in `else: drawEllipse(...)`, so these render as the "
        "same plain circle without any error."
    )


def test_no_ribbon_group_is_collapsed_onto_one_icon():
    """The shape of the original defect: a group of look-alike buttons."""
    # Group boundaries come from the variable each addGroup result is bound to;
    # buttons belong to the group whose variable they are passed to.
    groups: dict[str, list[str]] = {}
    for line in _SOURCE.splitlines():
        m = re.search(r"(\w+)\s*=\s*\w+\.addGroup\(", line)
        if m:
            groups[m.group(1)] = []
    for line in _SOURCE.splitlines():
        m = re.search(r"(\w+)\.addButton\(\s*\"([a-z0-9_]+)\"", line)
        if m and m.group(1) in groups:
            groups[m.group(1)].append(m.group(2))

    assert groups, "could not find any addGroup() calls; this check is stale"

    offenders = {}
    for name, icons in groups.items():
        if not icons:
            continue
        top, count = Counter(icons).most_common(1)[0]
        if count >= 3 and count > MAX_SHARE_OF_ONE_ICON:
            offenders[name] = (top, count, len(icons))

    assert not offenders, (
        "ribbon groups whose buttons are mostly one icon:\n  "
        + "\n  ".join(f"{name}: {count}/{total} use {icon!r}" for name, (icon, count, total) in offenders.items())
        + "\n\nEither draw a distinct glyph per action, or drop the icon from "
        "that group (RibbonStyle.TEXT_ONLY exists). An icon that repeats the "
        "tab name costs 24px per button and says nothing."
    )


def test_transform_buttons_have_distinct_glyphs():
    """Named explicitly, because these are the six that collided.

    A group-level count would also pass if a different set of six buttons
    collapsed, so the specific regression gets its own assertion.
    """
    from views.ui_main_window import _GLYPH_ICONS

    expected = {
        "tf_log",
        "tf_sqrt",
        "tf_hellinger",
        "tf_zscore",
        "tf_pct",
        "tf_wisconsin",
    }
    assert expected <= set(_GLYPH_ICONS), f"missing transform glyphs: {sorted(expected - set(_GLYPH_ICONS))}"
    glyphs = [_GLYPH_ICONS[k] for k in expected]
    assert len(set(glyphs)) == len(glyphs), f"transform glyphs are not distinct: {glyphs}"
    # Each must actually be wired to a button, or the table is decorative.
    requested = _requested_icon_types()
    unwired = sorted(expected - requested)
    assert not unwired, f"glyph icons defined but never used on a button: {unwired}"
