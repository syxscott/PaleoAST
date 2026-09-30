# =============================================================================
# FILE: parsers/sentinels.py
# =============================================================================
"""
Shared missing-value sentinel tokens used by every file parser in PaleoAST.

Why a shared module
--------------------
``TpsDig`` (Rohlf's TPS editor), MorphoJ, the NEXUS format itself, and most
PAST exports all converge on the same handful of glyphs to say "this cell
is missing":

* ``?`` — NEXUS standard missing symbol; also the convention tpsDig writes
  in ``LM=`` blocks when a landmark was not digitised.
* ``*`` — MorphoJ / geomorph's missing-landmark convention; some
  TPS exporters use it in coordinates.
* ``-`` — NEXUS gap symbol; widely accepted as a missing-value alias.
* ``NA`` / ``N/A`` / ``NaN`` / ``None`` / ``NULL`` — text spellings used by
  R/pandas/SPSS exports.

Hard-coding this list in every parser invited drift: the DAT parser at one
point knew ``NAN`` but not ``?``, while the TPS parser knew ``nan`` but not
``*``, so the same data file round-tripped through both lost different
cells. A single source of truth means ``?`` always means missing everywhere.

Constants:
    MISSING_SENTINELS: case-insensitive set of strings that map to NaN.
    MISSING_SENTINELS_UPPER: pre-uppercased for fast lookup.
    is_missing_token: helper, accepts a value and returns true if it's a
        sentinel.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations


# Canonical set of strings that represent a missing value. This is a frozenset
# on purpose: it has no iteration order, so nothing downstream can come to
# depend on one. If a serialised form ever needs a stable order, sort
# explicitly at that boundary rather than relying on the literal below.
class _CaseInsensitiveFrozenSet(frozenset):
    """A ``frozenset`` whose ``in`` operator is case-insensitive.

    The sentinels are stored uppercase, but ``MISSING_SENTINELS`` is a public
    export, so a caller can reasonably write ``if cell in MISSING_SENTINELS:``.
    Against a plain ``frozenset`` that silently returns ``False`` for "na",
    "nan", "null" and "None" -- a missed missing-value, which is exactly the
    class of silent data corruption this module exists to prevent. Making
    ``__contains__`` fold case keeps the membership test honest instead of
    pushing the burden onto every future caller to remember to uppercase.
    """

    __slots__ = ()

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return frozenset.__contains__(self, item.upper())
        return frozenset.__contains__(self, item)


MISSING_SENTINELS: frozenset[str] = _CaseInsensitiveFrozenSet(
    {
        "?",  # NEXUS missing / tpsDig missing landmark
        "*",  # MorphoJ / geomorph missing landmark
        "-",  # NEXUS gap; SPSS user-missing
        "NA",
        "N/A",
        "NAN",
        "NONE",
        "NULL",
    }
)
"""Case-insensitive set of strings that map to ``NaN``.

Membership folds case, so ``"na" in MISSING_SENTINELS`` and
``"NA" in MISSING_SENTINELS`` are both ``True``. The stored spellings are
uppercase; use this attribute for the lookup itself and
``MISSING_SENTINELS_UPPER`` when you want a plain, pre-folded ``frozenset``
for the fastest possible hot-loop access.

The empty string ``""`` is deliberately NOT in this set — a blank cell is a
*layout* error (missing token in a row), not a *value* error, and silently
turning it into NaN would hide ragged files. Parsers must surface empty
cells as field-count errors instead.
"""

# Pre-uppercased snapshot. Callers that do ``value.upper() in MISSING_SENTINELS_UPPER``
# skip the per-cell allocation of a new upper-cased string.
MISSING_SENTINELS_UPPER: frozenset[str] = frozenset(s.upper() for s in MISSING_SENTINELS)


def is_missing_token(value: str) -> bool:
    """Return True if ``value`` represents a missing-data token.

    Whitespace around the token is stripped; comparison is
    case-insensitive.

    Examples:
        >>> is_missing_token("?")
        True
        >>> is_missing_token("  *  ")
        True
        >>> is_missing_token("nan")
        True
        >>> is_missing_token("")
        False
        >>> is_missing_token("0")
        False
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return stripped.upper() in MISSING_SENTINELS_UPPER


__all__ = [
    "MISSING_SENTINELS",
    "MISSING_SENTINELS_UPPER",
    "is_missing_token",
]
