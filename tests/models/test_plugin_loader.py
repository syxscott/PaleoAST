"""Regression tests for plugins/loader.py.

Covers defect 3: the old ``_BUILTIN_PLUGINS`` list pointed at ``statistics.*``
modules that no longer exist after the ``0ffa12a`` rename; the dead
``load_builtin_plugins`` function imported each one and silently counted
only the survivor (``ecology.diversity``); an external ``SyntaxError`` in
a third-party plugin crashed the whole loader.
"""

from __future__ import annotations

import importlib


class TestBuiltinPluginNames:
    """Every name in ``_BUILTIN_PLUGINS`` must actually import."""

    def test_every_builtin_plugin_imports(self):
        loader = importlib.import_module("plugins.loader")
        for module_name in loader._BUILTIN_PLUGINS:
            # importlib.import_module raises ModuleNotFoundError on the
            # old ``statistics.pca`` form — that's exactly the regression
            # we want to lock down.
            importlib.import_module(module_name)

    def test_no_statistics_stdlib_shadowing(self):
        """The list MUST NOT contain ``statistics.*`` — that's the Python
        stdlib module and shadows our own analysis package."""
        loader = importlib.import_module("plugins.loader")
        for module_name in loader._BUILTIN_PLUGINS:
            assert not module_name.startswith("statistics."), (
                f"{module_name} collides with the Python standard library "
                f"'statistics' module and the project's 'stats' package was "
                f"renamed in commit 0ffa12a."
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

