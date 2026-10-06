"""Regression tests for plugins/loader.py and the analysis catalog.

Covers defect 3: the old ``_BUILTIN_PLUGINS`` list pointed at ``statistics.*``
modules that no longer exist after the ``0ffa12a`` rename; the dead
``load_builtin_plugins`` function imported each one and silently counted
only the survivor (``ecology.diversity``); an external ``SyntaxError`` in
a third-party plugin crashed the whole loader.

The list itself is gone -- plugins/catalog.py replaced it, because a
tuple no code reads cannot be kept honest. The two assertions below are
unchanged in intent and now run against that catalog, so the stdlib
shadowing regression stays locked down against whatever the current
source of truth happens to be. The catalog has its own, stricter tests in
tests/plugins/test_catalog.py.
"""

from __future__ import annotations

import importlib


def _catalog_modules() -> list[str]:
    """Every module the built-in analysis catalog names."""
    from plugins.catalog import BUILTIN_ANALYSES

    return [entry.module for entry in BUILTIN_ANALYSES]


class TestBuiltinPluginNames:
    """Every module the catalog names must actually import."""

    def test_every_builtin_plugin_imports(self):
        for module_name in _catalog_modules():
            # importlib.import_module raises ModuleNotFoundError on the
            # old ``statistics.pca`` form -- that's exactly the regression
            # we want to lock down.
            importlib.import_module(module_name)

    def test_no_statistics_stdlib_shadowing(self):
        """The catalog MUST NOT contain ``statistics.*`` -- that's the Python
        stdlib module and shadows our own analysis package."""
        for module_name in _catalog_modules():
            assert not module_name.startswith("statistics."), (
                f"{module_name} collides with the Python standard library "
                f"'statistics' module and the project's 'stats' package was "
                f"renamed in commit 0ffa12a."
            )

    def test_stale_list_is_gone(self):
        """The unconsumed tuple must not linger beside its replacement."""
        loader = importlib.import_module("plugins.loader")
        assert not hasattr(loader, "_BUILTIN_PLUGINS"), (
            "plugins/loader.py grew a new hard-coded analysis list; add the "
            "entry to plugins/catalog.py instead, where the tests can check "
            "it against the source tree."
        )


class TestLoadBuiltinPluginsRemoved:
    """``load_builtin_plugins`` was a zero-caller function with a broken
    module list. It must be gone."""

    def test_load_builtin_plugins_is_removed(self):
        loader = importlib.import_module("plugins.loader")
        assert not hasattr(loader, "load_builtin_plugins")

    def test_plugins_package_does_not_export_it(self):
        plugins_pkg = importlib.import_module("plugins")
        assert not hasattr(plugins_pkg, "load_builtin_plugins")
