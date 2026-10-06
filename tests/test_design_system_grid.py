# =============================================================================
# FILE: tests/test_design_system_grid.py
# =============================================================================
"""
Fail the build when the stylesheet stops respecting the 4px grid.

WHAT THIS IS GUARDING
---------------------
``Spacing`` documents a 4px scale, and the stylesheet then hardcoded twelve
``padding`` values around it -- three of which (2, 6, 10) were not multiples
of four. A declared grid that nothing enforces is a comment. This turns it
into a constraint.

WHY THE TEMPLATE, NOT THE GENERATED STYLESHEET
-----------------------------------------------
Both halves of this check would be meaningless against
``get_modern_stylesheet()``'s *output*:

  * every ``{spacing.xs}`` has already become ``4px`` by then, so counting
    token references finds nothing;
  * every ``{colors.primary}`` has already become ``#1E40AF``, so scanning
    for hex literals finds the whole palette and cannot tell a token from a
    hardcoded value.

So the source of the f-string is what gets read. That is also the only place
the distinction exists.

SCOPE
-----
``padding`` and ``margin`` only, in the generated sheet:

  * a 1px border is a hairline and 4 is not an option;
  * an 18px radio indicator is a component size chosen to look right next to
    a 13px label, not a spacing decision;
  * a *negative* margin (``-6px`` on a slider handle) is a geometric
    compensation that centres a child on its groove, not a gap between
    elements. It is exempt, and the exemption is asserted below so it stays
    deliberate.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from config.design_system import (
    ColorPalette,
    ColorPaletteDark,
    Spacing,
    get_modern_stylesheet,
)

GRID = 4
_MODULE = Path(__file__).resolve().parent.parent / "config" / "design_system.py"

# Rungs the scale declares but the stylesheet does not currently need. A scale
# is a vocabulary: not every step has to have a user today, and inventing a
# use for one just to satisfy this check would be worse than leaving it
# reserved. Listed explicitly so adding a use is a deliberate edit rather than
# an accident, and so a stale entry is visible.
RESERVED_SPACING_TOKENS = {
    "xxl": "32px -- reserved for full-page spacing; no current rule needs it",
}

SPACING_PROPERTY = re.compile(
    r"^\s*(padding|margin)(-(top|right|bottom|left))?\s*:\s*([^;]+);",
    re.MULTILINE,
)
PX_VALUE = re.compile(r"(-?\d+)px")


def _placeholder_text(node: ast.FormattedValue) -> str:
    """Rebuild ``{spacing.sm}`` from the expression node.

    In an f-string a placeholder is a ``FormattedValue``, not part of the
    surrounding text -- so concatenating only the ``Constant`` chunks yields
    the stylesheet with every ``{token}`` missing, and a check for "is this
    token referenced?" then finds none. That is exactly the bug this test
    exists to catch, reproduced inside the test itself.
    """
    expr = node.value
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        return "{" + expr.value.id + "." + expr.attr + "}"
    if isinstance(expr, ast.Name):
        return "{" + expr.id + "}"
    return "{?}"


def _stylesheet_template() -> str:
    """The un-interpolated f-string body of get_modern_stylesheet."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "get_modern_stylesheet":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.JoinedStr):
                parts = []
                for part in sub.value.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        parts.append(part.value)
                    elif isinstance(part, ast.FormattedValue):
                        parts.append(_placeholder_text(part))
                return "".join(parts)
    raise AssertionError(
        "get_modern_stylesheet no longer returns an f-string; this check has to be taught where the template moved"
    )


@pytest.mark.parametrize("palette", [ColorPalette, ColorPaletteDark], ids=["light", "dark"])
def test_generated_stylesheet_respects_the_grid(palette):
    off_grid = []
    for match in SPACING_PROPERTY.finditer(get_modern_stylesheet(palette)):
        declaration = match.group(0).strip()
        for raw in PX_VALUE.findall(match.group(4)):
            value = int(raw)
            if value < 0:
                continue  # geometric compensation, see module docstring
            if value % GRID:
                off_grid.append(f"{declaration}  ({value}px)")
    assert not off_grid, f"{len(off_grid)} spacing values are off the {GRID}px grid:\n  " + "\n  ".join(
        sorted(set(off_grid))
    )


def test_spacing_scale_is_itself_on_the_grid():
    """The scale cannot be the thing that is off-grid."""
    for name in ("xs", "sm", "md", "lg", "xl", "xxl"):
        value = getattr(Spacing, name)
        assert value % GRID == 0, f"Spacing.{name} = {value} is off-grid"


def test_every_spacing_token_is_referenced():
    """Guard against the state this started from: declared, never used.

    A stylesheet that satisfies the grid check by hardcoding every value
    passes it while making ``Spacing`` dead code. Both halves have to hold:
    the literals are on-grid *and* the scale is load-bearing.
    """
    template = _stylesheet_template()
    unused = [
        name
        for name in ("xs", "sm", "md", "lg", "xl", "xxl")
        if "{spacing." + name + "}" not in template and name not in RESERVED_SPACING_TOKENS
    ]
    assert not unused, (
        f"Spacing tokens never interpolated: {unused}. A token nothing reads is documentation, not a scale."
    )


def test_reserved_spacing_tokens_still_exist():
    """A reservation for a token that no longer exists is stale.

    Without this, a renamed or deleted rung would leave its reservation
    quietly matching nothing, and the ratchet above would stop noticing.
    """
    for name in RESERVED_SPACING_TOKENS:
        assert hasattr(Spacing, name), (
            f"RESERVED_SPACING_TOKENS names {name!r}, but Spacing no longer defines it; drop the reservation"
        )


def test_template_has_no_hardcoded_colours():
    """Literal colours are how the icon engine ended up on a dead palette.

    The stylesheet moved to tokens and the vector icons did not, so the two
    drifted apart with nothing to notice. Anything the sheet paints has to
    come from a token. (Checked on the template, where a token is still
    ``{colors.primary}`` rather than the hex it resolves to.)
    """
    template = _stylesheet_template()
    literals = re.findall(r"#[0-9A-Fa-f]{6}\b", template)
    assert not literals, (
        f"hardcoded colours in the stylesheet template: {sorted(set(literals))} -- use a ColorPalette token"
    )


def test_negative_margin_exemption_is_still_justified():
    """The grid check skips negative margins; make sure that stays honest.

    A negative margin is legitimate only where it centres a child on a
    parent. If one appears somewhere else, it is an off-grid value wearing an
    exemption, so the list of allowed sites is asserted rather than left
    implicit.
    """
    sheet = get_modern_stylesheet()
    negatives = {
        m.group(0).strip()
        for m in SPACING_PROPERTY.finditer(sheet)
        if any(int(v) < 0 for v in PX_VALUE.findall(m.group(4)))
    }
    assert negatives <= {"margin: -6px 0;"}, (
        f"unexpected negative margin(s): {sorted(negatives - {'margin: -6px 0;'})}"
        " -- the slider handle is the only legitimate case; anything else is "
        "an off-grid value claiming the exemption"
    )
