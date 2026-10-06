"""Parity test between translations_en.py and translations_zh.py.

The previous baseline left 46 keys present in the Chinese translation
file but absent from the English one, so English users saw raw key
names on screen.  Six duplicate keys in the Chinese file (``Visualization``,
``Method: {0}`` etc.) were silently merged by Python's dict literal, so
the user never saw a load-time error but every re-parse of the file
produced a different "last value wins" view.

This test reads both translation files via :mod:`ast`, expands their
``TRANSLATIONS`` dict literal, and asserts:

* every key in ``translations_zh.py`` is also in ``translations_en.py``;
* no key is duplicated within either file;
* values are non-empty strings.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "i18n"


def _read_translations(path: Path) -> dict[str, str]:
    """Parse the ``TRANSLATIONS = {...}`` literal out of ``path``.

    Using ``ast.literal_eval`` keeps the loader free of side effects and
    avoids executing any other top-level statements in the file.
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
                for k, v in zip(node.value.keys, node.value.values, strict=False):
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


def test_zh_has_no_duplicate_keys() -> None:
    """The Chinese file used to declare six keys twice (last-wins)."""
    src = (CONFIG_DIR / "translations_zh.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TRANSLATIONS":
                    keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                    assert len(keys) == len(set(keys)), (
                        f"Duplicate keys detected in translations_zh.py: {sorted(k for k in keys if keys.count(k) > 1)}"
                    )


def test_en_has_no_duplicate_keys() -> None:
    src = (CONFIG_DIR / "translations_en.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TRANSLATIONS":
                    keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                    assert len(keys) == len(set(keys)), (
                        f"Duplicate keys detected in translations_en.py: {sorted(k for k in keys if keys.count(k) > 1)}"
                    )


def test_key_sets_are_identical(
    en_translations: dict[str, str],
    zh_translations: dict[str, str],
) -> None:
    """English and Chinese must cover the same key set."""
    missing_in_en = sorted(set(zh_translations) - set(en_translations))
    missing_in_zh = sorted(set(en_translations) - set(zh_translations))
    assert not missing_in_en, f"Keys missing from translations_en.py: {missing_in_en}"
    assert not missing_in_zh, f"Keys missing from translations_zh.py: {missing_in_zh}"


def test_no_empty_values(
    en_translations: dict[str, str],
    zh_translations: dict[str, str],
) -> None:
    """A blank value is as bad as a missing key from the user's perspective."""
    for label, table in (("EN", en_translations), ("ZH", zh_translations)):
        empties = [k for k, v in table.items() if not v or not v.strip()]
        assert not empties, f"Empty values in translations_{label.lower()}.py: {empties}"


def test_46_original_missing_keys_now_present(en_translations: dict[str, str]) -> None:
    """Regression: the original 46 keys reported in the bug report."""
    expected = {
        "Abundance (log)",
        "Apply on Next Start",
        "Average Contribution (%)",
        "Biozone",
        "Biozones identified: {0}",
        "CA Axis",
        "Contour points: ",
        "Cross-validation (leave-one-out)",
        "Distance",
        "Events: {0}",
        "Group Comparison",
        "Harmonics",
        "Hierarchical Clustering",
        "Input",
        "Isotope",
        "Kruskal-Wallis (non-parametric)",
        "Language will change to {0} on next start.",
        "Linear Discriminant Analysis",
        "Linkage Method",
        "Linkage:",
        "Normality Test (Shapiro-Wilk)",
        "Number of LD axes: ",
        "Number of bins: ",
        "Number of clusters: ",
        "Number of harmonics: ",
        "Number of zones: ",
        "Observed",
        "One-way ANOVA (3+ groups)",
        "Original",
        "Paleo-Env. CA Reconstruction",
        "Paleo-Environment",
        "Rank",
        "Reconstructed",
        "Resampling",
        "Rose Diagram",
        "SHE Analysis",
        "SIMPER: Top Contributing Variables",
        "Sample Size",
        "Spatial",
        "Species-Abundance Models",
        "Stratigraphic Correlation",
        "Summary Statistics",
        "Test Type",
        "Transform",
        "Wavelet",
        "t-test (2 groups)",
    }
    missing = sorted(expected - set(en_translations))
    assert not missing, f"These 46 keys are missing from translations_en.py: {missing}"
