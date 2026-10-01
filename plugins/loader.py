# =============================================================================
# FILE: plugins/loader.py
# =============================================================================
"""
Plugin Loader for PaleoAST

Utilities for discovering and loading analysis plugins.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Built-in analysis modules, kept as a testable list of names.
#
# This is a hand-curated list, not a complete inventory: ``stats/`` has 15
# modules and ``ecology/`` 8, and this names 7. The earlier version of this
# comment claimed the list let a caller "find every first-party analysis
# plugin", which was wrong twice over -- it is not every one of them, and
# ``discover_plugins_in_package`` does not read this list at all (it scans a
# directory path). The names are kept because they are what the stale-rename
# test asserts against: they used to say ``statistics.*`` after the package was
# renamed to ``stats/``, and ``statistics`` is also a stdlib name, so the list
# pointed at modules that could never import.
_BUILTIN_PLUGINS: tuple[str, ...] = (
    "stats.pca",
    "stats.pcoa",
    "stats.nmds",
    "stats.anosim",
    "stats.permanova",
    "stats.simper",
    "ecology.diversity",
)


def discover_plugins_in_package(package_path: Path) -> list[str]:
    """
    List the importable module names directly inside a package directory.

    Returns every non-underscore **module** found in ``package_path``
    (subpackages are skipped, as they were before). This is a *directory
    listing*, not a plugin filter: nothing here imports the candidates or
    inspects them, so a module with no ``AnalysisPlugin`` subclass in it is
    still returned. Deciding what is actually a plugin means importing the
    modules and looking for subclasses, which is left to the caller precisely
    because importing a module runs it.

    The previous docstring claimed this "looks for Python files that define
    AnalysisPlugin subclasses", which it never did. A caller who believed that
    would take the result as a list of plugins and try to instantiate entries
    that are not plugins.

    Parameters:
        package_path: Path to the package directory. A path that is not a
            directory yields an empty list.

    Returns:
        Module names, in the order ``pkgutil`` reports them.
    """
    if not package_path.is_dir():
        return []

    modules = []
    for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
        if not ispkg and not modname.startswith("_"):
            modules.append(modname)
    return modules


__all__ = ["discover_plugins_in_package"]
