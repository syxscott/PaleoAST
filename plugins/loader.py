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

# The first-party analyses used to be listed here, as a 7-entry tuple
# that no code read and whose docstring claimed it was a complete
# inventory when it named 7 of 23. The real list is
# ``plugins.catalog.BUILTIN_ANALYSES``, which IS consumed, is checked
# against the source tree by tests/plugins/test_catalog.py, and records
# the class and method as well as the module.
#
# What remains here is the directory-walk helper, which is a different
# job: it finds module NAMES in a package and deliberately does not
# import them, because importing a module runs it.
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
