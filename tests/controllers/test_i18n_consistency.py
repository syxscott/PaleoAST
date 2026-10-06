"""i18n table consistency regression tests.

The translation tables in ``config/i18n/`` are the only bridge between
the rest of the code and ``config.i18n._()``.  A subtle mismatch here
silently breaks user-facing copy: a missing key falls back to the key
text, a duplicate key is last-wins, and a mismatched placeholder
(``{0}`` vs ``{name}``) raises ``IndexError`` / ``KeyError`` at the
call site instead of producing a translated string.

These tests assert:

* ``translations_en.py`` and ``translations_zh.py`` declare exactly
  the same set of keys (no orphans on either side);
* no key is duplicated within either file;
* every value is a non-empty string;
* for any key whose value contains a ``str.format`` placeholder, the
  set of placeholders matches across the two files (English ``{0}``
  cannot be paired with Chinese ``{name}``);
* both tables load via ``config.i18n.register_translations()`` and
  the translator returns the expected value for a handful of round-trip
  probes.

A failure here means: a developer added a ``_("foo")`` somewhere and
forgot to add the matching entry in one of the two tables, or the two
tables drifted apart.  The fix is mechanical — extend ``en``/``zh`` so
the assertion holds again.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "i18n"
_PLACEHOLDER_RE = re.compile(r"\{[^}]*\}")


def _read_translations(path: Path) -> dict[str, str]:
    """Parse the ``TRANSLATIONS = {...}`` literal out of ``path``.

    Using :func:`ast.literal_eval` keeps the loader free of side effects
    and avoids executing any other top-level statements in the file
    (e.g. the i18n machinery's ``QObject`` setup, which fails without a
    live Qt event loop).
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TRANSLATIONS":
                if not isinstance(node.value, ast.Dict):
                    raise AssertionError(f"TRANSLATIONS in {path} is not a dict literal")
                result: dict[str, str] = {}
                for k, v in zip(node.value.keys, node.value.values):
                    if not isinstance(k, ast.Constant):
                        raise AssertionError(f"Non-constant key in {path}: {ast.dump(k)}")
                    result[k.value] = ast.literal_eval(v)
                return result
    raise AssertionError(f"No TRANSLATIONS dict found in {path}")


@pytest.fixture(scope="module")
def en_translations() -> dict[str, str]:
    return _read_translations(CONFIG_DIR / "translations_en.py")


@pytest.fixture(scope="module")
def zh_translations() -> dict[str, str]:
    return _read_translations(CONFIG_DIR / "translations_zh.py")


# ---------------------------------------------------------------------------
# Key-set parity
# ---------------------------------------------------------------------------


def test_key_sets_are_identical(
    en_translations: dict[str, str],
    zh_translations: dict[str, str],
) -> None:
    """Every key in either table must exist in the other.

    The previous bug report had 46 keys present in ZH but missing from
    EN, so English users saw raw key strings on screen; this guard
    keeps the two tables in lockstep.
    """
    missing_in_en = sorted(set(zh_translations) - set(en_translations))
    missing_in_zh = sorted(set(en_translations) - set(zh_translations))
    assert not missing_in_en, f"Keys missing from translations_en.py: {missing_in_en}"
    assert not missing_in_zh, f"Keys missing from translations_zh.py: {missing_in_zh}"


# ---------------------------------------------------------------------------
# Duplicate / empty value guards
# ---------------------------------------------------------------------------


def _duplicates_in(path: Path) -> list[str]:
    """Return keys declared more than once in ``TRANSLATIONS``.

    Python's dict literal silently keeps the LAST assignment, so a
    duplicate key is invisible at runtime — the earlier value vanishes
    without warning, leaving the user with whichever value the parser
    kept.  The fix is to remove the earlier copy.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TRANSLATIONS":
                keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                return sorted({k for k in keys if keys.count(k) > 1})
    return []


def test_en_has_no_duplicate_keys() -> None:
    assert not _duplicates_in(CONFIG_DIR / "translations_en.py"), (
        "Duplicate keys in translations_en.py: "
        f"{_duplicates_in(CONFIG_DIR / 'translations_en.py')}"
    )


def test_zh_has_no_duplicate_keys() -> None:
    assert not _duplicates_in(CONFIG_DIR / "translations_zh.py"), (
        "Duplicate keys in translations_zh.py: "
        f"{_duplicates_in(CONFIG_DIR / 'translations_zh.py')}"
    )


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_no_empty_values(
    lang: str,
    en_translations: dict[str, str],
    zh_translations: dict[str, str],
) -> None:
    """An empty value is as bad as a missing key — both render the key."""
    table = en_translations if lang == "en" else zh_translations
    empties = [k for k, v in table.items() if not v or not v.strip()]
    assert not empties, f"Empty values in translations_{lang}.py: {empties}"


# ---------------------------------------------------------------------------
# Placeholder consistency
# ---------------------------------------------------------------------------


def _placeholder_set(value: str) -> set[str]:
    """Extract the set of ``str.format`` placeholders in ``value``.

    Only ``re.findall`` of the basic ``{...}`` syntax — name-form
    placeholders (``{name}``) and indexed placeholders (``{0}``,
    ``{1:.2f}``) both appear, and we treat them as opaque tokens; the
    goal is "do the two languages reference the same fields?" not
    "are the field names identical across the two sides?".
    """
    return set(_PLACEHOLDER_RE.findall(value))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_placeholders_match_across_languages(
    lang: str,
    en_translations: dict[str, str],
    zh_translations: dict[str, str],
) -> None:
    """A key whose English value uses ``{0}`` must use ``{0}`` in ZH too.

    Otherwise ``str.format`` raises ``IndexError`` (or ``KeyError`` for
    name-form placeholders) the moment the i18n lookup succeeds in one
    language and fails in the other.  The previous table silently
    contained a few drift points; this test pins them down.
    """
    mismatches: list[str] = []
    other = zh_translations if lang == "en" else en_translations
    table = en_translations if lang == "en" else zh_translations
    for key, value in table.items():
        en_phs = _placeholder_set(value)
        zh_phs = _placeholder_set(other[key])
        if en_phs != zh_phs:
            mismatches.append(
                f"  {key!r}: {lang}={sorted(en_phs)} vs other={sorted(zh_phs)}"
            )
    assert not mismatches, (
        "Placeholder mismatch between translations_en.py and translations_zh.py:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# Round-trip through the live translator
# ---------------------------------------------------------------------------


def test_register_translations_round_trip(zh_translations: dict[str, str]) -> None:
    """``register_translations()`` must register both tables and the
    translator must return the ZH value for a known ZH key when the
    language is switched to ``zh``.

    This catches the case where the module-level
    ``register_translations()`` function accidentally stops loading one
    of the two tables (e.g. an import typo after a rename).
    """
    pytest.importorskip("PyQt6", reason="PyQt6 is required for config.i18n QObject base class")
    from PyQt6.QtWidgets import QApplication

    from config.i18n import _reset_translator, get_translator, register_translations

    app = QApplication.instance() or QApplication([])
    try:
        _reset_translator()
        register_translations()
        translator = get_translator()

        # Probe with a key whose EN and ZH values both differ from the key
        # (the i18n module returned English-as-key for proper nouns).
        en_probe = "Op: file loading"  # EN: "file loading", ZH: "文件加载"
        previous_lang = translator.get_language()
        try:
            translator.set_language("zh")
            zh_value_open = translator.translate(en_probe)
            translator.set_language("en")
            en_value_open = translator.translate(en_probe)
        finally:
            translator.set_language(previous_lang)

        # Both languages must look up successfully — if the EN value
        # comes back as the raw key, the EN table was not loaded; if the
        # ZH value matches the raw key, the ZH table was not loaded.
        assert en_value_open != en_probe, (
            f"Translator returned the raw key for {en_probe!r} in en mode — "
            "the EN table is not being loaded."
        )
        assert zh_value_open != en_probe, (
            f"Translator returned the raw key for {en_probe!r} in zh mode — "
            "the ZH table is not being loaded."
        )
        # And confirm the ZH value matches the ZH table entry verbatim
        # — this catches a partial load (e.g. a re-import that only
        # picked up a subset of the table).
        assert zh_value_open == zh_translations[en_probe], (
            f"Translator returned {zh_value_open!r} for {en_probe!r} in zh mode, "
            f"expected {zh_translations[en_probe]!r}"
        )
    finally:
        _reset_translator()
