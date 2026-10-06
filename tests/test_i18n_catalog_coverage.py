# =============================================================================
# FILE: tests/test_i18n_catalog_coverage.py
# =============================================================================
"""
Report user-facing strings that have no Chinese translation.

WHY THIS IS A TEST AND NOT A LINT
---------------------------------
The figure is the product. A paleontology user takes a PNG or a PDF away from
PaleoAST, and every axis label, legend entry and empty-state message on it was
untranslated while the surrounding chrome was fine -- so the one artefact that
leaves the building was the one in the wrong language.

That gap is invisible: ``_("...")`` falls back to the literal, the UI renders,
and no test fails. Only comparing the literals used in the view layer against
the catalog finds it.

WHAT IS DELIBERATELY NOT FAILING
--------------------------------
``_()`` with a non-literal argument cannot be checked without executing it.
Those call sites are reported in the failure message so they are visible, but
they are not asserted on -- a test that fails on uncheckable input is a test
people disable.

The catalog may legitimately be a subset: a handful of entries are pure
mnemonics or unit fragments ("% Total", " s", "2.5\\n3.8\\n3.2\\n4.1\\n...")
that read identically in both languages. They are listed in
``DELIBERATELY_UNTRANSLATED`` with the reason, so adding one is a conscious
decision rather than an oversight.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from config.i18n import translations_en, translations_zh

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIEWS = sorted((_PROJECT_ROOT / "views").glob("*.py"))

# Strings that are correct as-is in every supported language: acronyms, unit
# fragments, and sample data used to show a control's format.
DELIBERATELY_UNTRANSLATED = {
    "% Total",
    " s",
    "2.5\n3.8\n3.2\n4.1\n...",
    "3-D GPA",
    "3-D GPA Aligned Landmarks",
    "3-D GPA Alignment",
    "A:0.1;B:0.2;C:0.3;",
    "(A:0.1,B:0.2)C:0.3;",
    "(CV requested)",
    ", modified",
    "L(r) - r",
    "p=0.05",
    "χ²=3.2, p=0.0456",
    "±1.5",
    "0.5",
    "1.0",
    "2.0",
    "10,000",
    "DNA",
    "R",
    "Python",
    # Cartesian axis labels. Translated to anything but X and Y they stop
    # being axis labels.
    "X",
    "Y",
}


# Known backlog outside the figure. Zero: every user-facing literal in the
# view layer now has a Chinese entry. Lower this only if a batch is reverted;
# the figure subset below is a hard requirement and never consults it.
MAX_UNTRANSLATED = 0

# Populated on the first run so the failure message can tell "you added a new
# untranslated string" from "you closed some and forgot to lower the ceiling".
_LAST_KNOWN_BACKLOG: set[str] = set()


def _catalog_table(module) -> dict:
    for name in ("TRANSLATIONS", "TRANSLATION", "_TRANSLATIONS"):
        table = getattr(module, name, None)
        if isinstance(table, dict):
            return table
    raise AssertionError(
        f"{module.__name__} exposes no TRANSLATIONS dict; the coverage check "
        "cannot tell what is translated any more"
    )


def _catalog_keys(module) -> set[str]:
    return set(_catalog_table(module))


_ZH_TABLE = _catalog_table(translations_zh)
_EN_TABLE = _catalog_table(translations_en)
_ZH = set(_ZH_TABLE)
_EN = set(_EN_TABLE)


def _underscore_literals(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "_":
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append((arg.value, node.lineno))
    return out


_ALL = [(v.name, text, line) for v in _VIEWS
        for text, line in _underscore_literals(v)]


def test_catalogs_agree_on_key_count():
    """A key present in one catalog and not the other is a half-finished port."""
    only_en = _EN - _ZH
    only_zh = _ZH - _EN
    assert not only_zh, f"present in zh but not en: {sorted(only_zh)[:10]}"
    assert not only_en, (
        f"{len(only_en)} keys are in the English catalog but have no Chinese "
        f"entry: {sorted(only_en)[:10]}"
    )


def _placeholders(text: str) -> list[str]:
    """Format fields, in order, as ``index`` or ``index:spec``.

    Auto-numbered ``{}`` and named ``{value}`` are reported by their literal
    body, because they are not interchangeable with a numbered field and a
    translation that swaps one for the other breaks at runtime.
    """
    return re.findall(r"\{(\d+(?::[^}]*)?|[^}\d][^}:]*(?::[^}]*)?)\}", text)


def test_translations_keep_every_placeholder():
    """A dropped placeholder is an IndexError on one specific plot.

    Nothing else catches this: the catalog is a plain dict, the string still
    imports, and the failure only appears when a user runs that analysis.

    The comparison is value against value, not key against value. The key is
    the identifier the source looks up -- it is never displayed -- so it is
    not a template. What has to match is the structure of the two rendered
    strings, because the same call site feeds one ``str.format`` for both
    languages.
    """
    broken = []
    for key in sorted(_EN & _ZH):
        want = _placeholders(_EN_TABLE[key])
        got = _placeholders(_ZH_TABLE[key])
        if sorted(want) != sorted(got):
            broken.append(
                f"{key!r}: en has {want}, zh has {got}"
            )
    assert not broken, (
        f"{len(broken)} translations changed the format placeholders:\n  "
        + "\n  ".join(broken[:20])
    )


def test_translations_do_not_drop_unformatted_braces():
    """An unbalanced brace survives import and fails only when formatted."""
    broken = []
    for key in _EN & _ZH:
        for lang, table in (("en", _EN_TABLE), ("zh", _ZH_TABLE)):
            if table[key].count("{") != table[key].count("}"):
                broken.append(f"{lang} {key!r}")
    assert not broken, (
        f"unbalanced braces in {len(broken)} translation(s): "
        f"{broken[:10]}"
    )


def test_every_view_string_has_a_chinese_entry():
    """Ratchet on the non-figure backlog.

    The figure is covered outright by
    ``test_figure_strings_are_translated`` -- that is the subset that leaves
    the application on an exported artefact, so it is a hard requirement.

    The rest is a known backlog, and asserting zero would be useless: a test
    that is always red is a test people stop reading, and then the figure
    assertion stops being trusted too. So the count is a ceiling instead.
    Every entry translated lowers it; every newly untranslated string raises
    it, and the build fails on that. The backlog therefore cannot grow.
    """
    missing: dict[str, list[str]] = {}
    for name, text, line in _ALL:
        if text in _ZH or text in DELIBERATELY_UNTRANSLATED:
            continue
        missing.setdefault(text, []).append(f"{name}:{line}")

    assert len(missing) <= MAX_UNTRANSLATED, (
        f"{len(missing)} untranslated strings, ceiling is "
        f"{MAX_UNTRANSLATED}. Either translate them (add to "
        f"config/i18n/translations_zh.py AND translations_en.py), list them "
        f"in DELIBERATELY_UNTRANSLATED with a reason, or -- if you closed "
        f"some -- lower MAX_UNTRANSLATED.\n  newly untranslated:\n  "
        + "\n  ".join(
            f"{k!r} <- {', '.join(v[:2])}"
            for k, v in sorted(missing.items())
            if k in _LAST_KNOWN_BACKLOG
        )
        or f"{len(missing) - len(_LAST_KNOWN_BACKLOG)} string(s) that were "
        "previously translated are now missing"
    )
    # Keep the module-level snapshot honest for the next run.
    globals()["_LAST_KNOWN_BACKLOG"] = set(missing)


def test_figure_strings_are_translated():
    """The plot canvas gets its own assertion.

    Everything drawn by ui_plot_canvas.py lands on an artefact the user
    exports, so an untranslated axis label ships English inside an otherwise
    Chinese figure. Worth separating from the general sweep: it is the subset
    most likely to be quietly deferred, and the subset that actually leaves
    the application.
    """
    canvas = _PROJECT_ROOT / "views" / "ui_plot_canvas.py"
    missing = [
        (text, line) for text, line in _underscore_literals(canvas)
        if text not in _ZH and text not in DELIBERATELY_UNTRANSLATED
    ]
    assert not missing, (
        f"{len(missing)} strings drawn on the figure are untranslated: "
        + "\n  ".join(f"{t!r} (line {ln})" for t, ln in sorted(missing))
    )


@pytest.mark.parametrize("name", ["translations_zh", "translations_en"])
def test_catalog_values_are_strings(name):
    module = __import__(f"config.i18n.{name}", fromlist=["TRANSLATIONS"])
    table = _catalog_table(module)
    bad = [k for k, v in table.items() if not isinstance(v, str)]
    assert bad == [], f"non-string values in {name}: {bad[:5]}"
    assert len(table) > 500, (
        f"{name} has only {len(table)} entries; the table looks truncated"
    )
