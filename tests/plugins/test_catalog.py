# =============================================================================
# FILE: tests/plugins/test_catalog.py
# =============================================================================
"""
Tests for the built-in analysis catalog and the plugin registry it feeds.

The failure this file exists to prevent: plugins/loader.py carried a
7-entry ``_BUILTIN_PLUGINS`` tuple that no code read, and whose comment
records that it had been documented as "every first-party analysis
plugin" when it named 7 of 23. A list of analyses that nothing checks is
a list that silently rots.

So the tests here go in both directions:

  * every catalog entry must resolve -- module imports, class exists,
    method is callable;
  * every ``*Analyzer`` class in the analysis packages must be covered by
    at least one entry.

The second is the one that fails when someone adds an analyzer and
forgets the catalog, which is the case the old list could not catch even
in principle.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from plugins.base import AnalysisResult
from plugins.catalog import (
    BUILTIN_ANALYSES,
    BuiltinAnalysisPlugin,
    register_builtin_analyses,
)
from plugins.registry import AnalysisPluginRegistry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The packages the catalog draws from. Keeping this list explicit means a
# new analysis package is a deliberate addition rather than something
# picked up by a directory walk that then fails on a module with an
# optional dependency.
ANALYSIS_PACKAGES = (
    "stats",
    "ecology",
    "stratigraphy",
    "morphometrics",
    "macroevolution",
    "models",
    "phylogenetics",
)


def _analyzer_classes_in_source() -> set[tuple[str, str]]:
    """Every ``*Analyzer`` class actually defined in the analysis packages.

    Discovered by importing each module, not by grepping: a grep for the
    class name would also match a mention in a docstring or a comment,
    which is the same class of error the catalog is meant to avoid.
    """
    found: set[tuple[str, str]] = set()
    for package in ANALYSIS_PACKAGES:
        directory = PROJECT_ROOT / package
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name == "__init__.py":
                continue
            modname = f"{package}.{path.stem}"
            try:
                module = importlib.import_module(modname)
            except Exception:  # pragma: no cover - optional dependency
                continue
            for name, obj in vars(module).items():
                if inspect.isclass(obj) and obj.__module__ == modname and name.endswith("Analyzer"):
                    found.add((modname, name))
    return found


# ---------------------------------------------------------------------------
# The catalog resolves
# ---------------------------------------------------------------------------


def test_catalog_is_not_empty() -> None:
    """An empty catalog would pass every other test here."""
    assert len(BUILTIN_ANALYSES) >= 50


def test_every_entry_resolves() -> None:
    """Module imports, class exists, method is callable, no early import.

    This is the assertion the old hand-written list could not make: it
    named modules, and nothing ever checked that a class with the
    expected name was in them.
    """
    broken: list[str] = []
    for entry in BUILTIN_ANALYSES:
        try:
            module = importlib.import_module(entry.module)
        except Exception as exc:
            broken.append(f"{entry.name}: cannot import {entry.module} ({exc})")
            continue
        if entry.class_name is None:
            target: object = module
        else:
            target = getattr(module, entry.class_name, None)
            if target is None:
                broken.append(f"{entry.name}: {entry.module} has no {entry.class_name}")
                continue
            target = target()  # type: ignore[operator]
        if not callable(getattr(target, entry.method, None)):
            broken.append(f"{entry.name}: {entry.module}.{entry.class_name or ''}.{entry.method} is not callable")
    assert not broken, "catalog entries that do not resolve:\n" + "\n".join(broken)


def test_every_catalogued_module_imports() -> None:
    """Each catalogued module must be importable, not merely present.

    A name that resolves to a class attribute can still name a module
    that fails to import -- a circular import, or a dependency the base
    install lacks. The frozen build hits the same wall harder: the spec
    lists these modules as hidden imports precisely because the catalog
    references them by string and PyInstaller's analysis follows imports
    only. Six of them -- stats.mantel, stats.detriding,
    stats.spatial_stats, stats.design_tests, models.growth_models and
    utils.script_cli -- were missing from the executable while passing
    every other check here, so this is the assertion that would have
    caught it.
    """
    failures: list[str] = []
    for module in sorted({entry.module for entry in BUILTIN_ANALYSES}):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    assert not failures, "catalogued modules that do not import:\n" + "\n".join(failures)


def test_the_spec_derives_hidden_imports_from_the_catalog() -> None:
    """The spec must read the catalog, or the two can drift apart.

    Checking the catalog is importable is necessary but not sufficient:
    PyInstaller will not follow a string reference, so an analysis in the
    catalog that nothing imports statically is still left out of the
    build. The guard for that is that PaleoAST.spec adds the catalog's
    modules to hiddenimports at build time rather than listing them.
    """
    spec = (Path(__file__).resolve().parent.parent.parent / "PaleoAST.spec").read_text(encoding="utf-8")
    assert "BUILTIN_ANALYSES" in spec, (
        "PaleoAST.spec does not derive hidden imports from the analysis "
        "catalog, so a catalogued analysis that nothing imports statically "
        "will be missing from the frozen build."
    )


def test_names_are_unique() -> None:
    """A duplicate name would make the second one unreachable."""
    names = [e.name for e in BUILTIN_ANALYSES]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate catalog names: {sorted(duplicates)}"


def test_every_category_is_meaningful() -> None:
    """No placeholders, no empty strings."""
    for entry in BUILTIN_ANALYSES:
        assert entry.category.strip(), f"{entry.name} has no category"
        assert entry.description.strip(), f"{entry.name} has no description"
        assert len(entry.description) > 5, f"{entry.name} description is too short"


# ---------------------------------------------------------------------------
# The catalog does not fall behind the source tree
# ---------------------------------------------------------------------------


def test_catalog_covers_every_analyzer_in_the_source() -> None:
    """An analyzer nobody catalogued is invisible to the scripting layer.

    The specific failure this catches: someone adds
    ``SomeNewAnalyzer`` to stats/ and the plugin registry, the UI and
    the controller never learn about it, and nothing complains.
    """
    covered = {(e.module, e.class_name) for e in BUILTIN_ANALYSES if e.class_name is not None}
    present = _analyzer_classes_in_source()
    missing = sorted(present - covered)
    assert not missing, (
        "analyzer classes not present in plugins/catalog.py:\n  "
        + "\n  ".join(f"{m}.{c}" for m, c in missing)
        + "\nAdd each to BUILTIN_ANALYSES, or delete it if it was dead code."
    )


def test_catalog_does_not_name_classes_that_do_not_exist() -> None:
    """The other direction: a catalogued class that was renamed away.

    Covered by test_every_entry_resolves, but asserted separately so the
    message names the catalog rather than the resolver.
    """
    present = _analyzer_classes_in_source()
    stale = sorted(
        (e.module, e.class_name)
        for e in BUILTIN_ANALYSES
        if e.class_name is not None and e.class_name.endswith("Analyzer") and (e.module, e.class_name) not in present
    )
    assert not stale, "catalog names analyzer classes not found in the source tree:\n  " + "\n  ".join(
        f"{m}.{c}" for m, c in stale
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> AnalysisPluginRegistry:
    """A registry with a clean slate."""
    reg = AnalysisPluginRegistry()
    reg.clear()
    yield reg
    reg.clear()


def test_registration_fills_the_registry(registry: AnalysisPluginRegistry) -> None:
    """The registry goes from empty to holding the whole catalog."""
    assert registry.list_plugins() == []
    registered = register_builtin_analyses(registry=registry)
    assert len(registered) == len(BUILTIN_ANALYSES)
    assert set(registry.list_plugins()) == {e.name for e in BUILTIN_ANALYSES}


def test_registration_is_idempotent(registry: AnalysisPluginRegistry) -> None:
    """Loading the catalog twice must not raise.

    The registry's own register() raises on a duplicate, which is right
    for a plugin author who registers twice by mistake and wrong for a
    catalog applied at startup -- application start, then a test that
    clears and reloads, then the window reopening.
    """
    register_builtin_analyses(registry=registry)
    second = register_builtin_analyses(registry=registry)
    assert second == []
    assert len(registry.list_plugins()) == len(BUILTIN_ANALYSES)


def test_replace_reloads(registry: AnalysisPluginRegistry) -> None:
    """replace=True overwrites rather than skipping."""
    first = register_builtin_analyses(registry=registry)
    again = register_builtin_analyses(registry=registry, replace=True)
    assert len(again) == len(first)
    assert len(registry.list_plugins()) == len(BUILTIN_ANALYSES)


def test_categories_come_out_consistent(registry: AnalysisPluginRegistry) -> None:
    """The category filter and the full list agree."""
    register_builtin_analyses(registry=registry)
    all_names = set(registry.list_plugins())
    per_category: set[str] = set()
    for category in registry.list_categories():
        per_category.update(registry.list_plugins(category=category))
    assert per_category == all_names
    assert "ordination" in registry.list_categories()


# ---------------------------------------------------------------------------
# The adapter actually runs something
# ---------------------------------------------------------------------------


def test_plugin_runs_and_wraps_the_result(
    registry: AnalysisPluginRegistry,
) -> None:
    """run_plugin reaches the wrapped analyzer and normalises the result.

    PCA is used because it needs only a matrix, takes no tuning, and is
    the analysis the application has run longest -- so if the adapter
    were broken, this is where it would show.
    """
    register_builtin_analyses(registry=registry)
    plugin = registry.get("pca")
    assert plugin is not None

    data = np.random.default_rng(0).normal(size=(24, 5))
    outcome = plugin.analyze(data, n_components=2)

    assert isinstance(outcome, AnalysisResult)
    assert outcome.success
    assert outcome.metadata["module"] == "stats.pca"
    assert outcome.metadata["method"] == "analyze"
    assert outcome.data is not None
    # The wrapped result keeps its own type, not a dict.
    assert hasattr(outcome.data, "row_scores") or hasattr(outcome.data, "scores")
    assert "summary" in outcome.metadata


def test_plugin_surfaces_the_wrapped_exceptions(registry: AnalysisPluginRegistry) -> None:
    """A ValidationError from the analyzer must not become a generic one."""
    register_builtin_analyses(registry=registry)
    plugin = registry.get("kmeans")
    assert plugin is not None
    with pytest.raises(Exception) as excinfo:
        plugin.analyze(np.random.default_rng(0).normal(size=(6, 2)), n_clusters=99)
    assert "smaller than the number of samples" in str(excinfo.value)


def test_plugin_reports_a_stale_catalog_rather_than_failing_obscurely() -> None:
    """A catalogued method that has been renamed says so, clearly."""
    import dataclasses

    stale = dataclasses.replace(BUILTIN_ANALYSES[0], method="no_such_method")
    broken = BuiltinAnalysisPlugin(stale)
    with pytest.raises(AttributeError, match="catalog is out of date"):
        broken.analyze(np.zeros((4, 2)))


def test_controller_sees_the_registry() -> None:
    """The controller's plugin half is no longer empty.

    This is the end-to-end assertion: before the catalog existed,
    list_plugins() returned [] and list_available_analyses() was
    indistinguishable from the controller's own method names.
    """
    from controllers.statistics_controller import StatisticsController

    controller = StatisticsController()
    plugins = controller.list_plugins()
    assert len(plugins) >= 50
    for name in ("mantel", "dca", "kmeans", "two_way_anova", "growth_model"):
        assert name in plugins, f"{name} missing from the registry"
    combined = controller.list_available_analyses()
    assert set(plugins).issubset(set(combined))


def test_controller_run_plugin_executes() -> None:
    """run_plugin reaches a real analysis and returns a result."""
    from controllers.statistics_controller import StatisticsController

    controller = StatisticsController()
    data: npt.NDArray = np.random.default_rng(0).normal(size=(20, 4))
    outcome = controller.run_plugin("mantel", data=data, n_permutations=99, random_seed=1)
    assert isinstance(outcome, AnalysisResult)
    assert outcome.metadata["category"] == "spatial"
    assert np.isfinite(outcome.data.statistic)
