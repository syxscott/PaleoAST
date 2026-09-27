#!/usr/bin/env python3
"""
Import smoke test for a **non-editable** (wheel) install.

Why this exists
---------------
Every other CI job does ``pip install -e .``, which puts the repository root on
``sys.path``. That makes a whole class of packaging bug invisible:

* a package missing from ``[tool.setuptools.packages.find] include`` still
  imports, because the source tree happens to be right there;
* a package whose *subpackages* are not all discovered (e.g. only
  ``views`` is found but ``views/some_sub/`` is not) still imports;
* ``py-modules`` omissions are invisible -- ``main`` and ``plot_export`` are
  top-level modules, and ``packages.find`` alone does not ship them, yet the
  editable install happily resolves ``from plot_export import ...``;
* ``[tool.setuptools.package-data]`` omissions are invisible -- a wheel
  without ``data/examples/*`` still lets ``import data.loader`` succeed, and
  only fails later inside ``load_*()`` at runtime.

This script is meant to run with the CWD **outside** the source tree, against a
clean venv holding only the built wheel. It asserts the opposite of the above:
that every declared package really is importable, that each one resolved to the
installed copy rather than the checkout, and that the package data and the
console-script entry point survived packaging.

Usage
-----
    python scripts/import_smoke.py            # from anywhere; uses installed copy
    python scripts/import_smoke.py --verbose  # list every module's __file__

Exit codes: 0 = all checks passed, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The declared packaging surface, mirrored from [tool.setuptools] in
# pyproject.toml. ``_check_matches_pyproject`` below fails loudly if the two
# ever drift apart, so this list cannot silently rot.
# ---------------------------------------------------------------------------
DECLARED_PACKAGES = [
    "config",
    "controllers",
    "data",
    "ecology",
    "hpc",
    "macroevolution",
    "models",
    "morpho3d",
    "morphometrics",
    "parsers",
    "phylogenetics",
    "plugins",
    "presets",
    "reporting",
    "state_machine",
    "stats",
    "stratigraphy",
    "utils",
    "views",
    "visualization",
]

DECLARED_PY_MODULES = ["main", "plot_export"]

#: ``data/loader.py`` resolves its bundled corpora through
#: ``importlib.resources.files("data")``. Without these in the wheel, every
#: ``load_*()`` call raises FileNotFoundError at runtime.
REQUIRED_PACKAGE_DATA = {
    "data": ["examples", "golden"],
}

CONSOLE_SCRIPT = "paleoast"
CONSOLE_SCRIPT_TARGET = "main:main"

_FAILURES: list[str] = []
_VERBOSE = False


def _fail(message: str) -> None:
    _FAILURES.append(message)
    print(f"  FAIL  {message}")


def _ok(message: str) -> None:
    if _VERBOSE:
        print(f"  ok    {message}")


def _is_inside(child: Path, parent: Path) -> bool:
    """True if ``child`` is ``parent`` or lives under it (both resolved)."""
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_imports_resolve_outside_source_tree() -> None:
    """Every declared package and py-module must import, and must come from
    the installed location -- not from the checkout.

    The source-tree half of this is the whole point: if ``sys.path`` still
    contains the repository (editable install, or a stray ``PYTHONPATH``), a
    package missing from the wheel would import anyway and the wheel would be
    silently broken.
    """
    source_root = _locate_source_root()
    # The script itself lives at <root>/scripts/import_smoke.py.
    for name in DECLARED_PACKAGES + DECLARED_PY_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            _fail(f"import {name!r} raised {type(exc).__name__}: {exc}")
            continue

        module_file = getattr(module, "__file__", None)
        if module_file is None:
            # Namespace package: has no single file, so check its search
            # locations instead.
            locations = [Path(p) for p in (getattr(module, "__path__", None) or [])]
            if not locations:
                _fail(f"{name!r} imported but has no __file__ and no __path__")
                continue
            if source_root and any(_is_inside(loc, source_root) for loc in locations):
                _fail(f"{name!r} resolved into the source tree: {locations[0]}")
                continue
            _ok(f"{name!r} -> {locations[0]}")
            continue

        path = Path(module_file)
        if source_root and _is_inside(path, source_root):
            _fail(
                f"{name!r} resolved to the source checkout, not the installed "
                f"package: {path}\n"
                f"          This masks packaging bugs. Run this from OUTSIDE "
                f"{source_root} against a wheel install."
            )
            continue
        _ok(f"{name!r} -> {path}")


def check_not_on_source_path() -> None:
    """The repository root must not be on ``sys.path``.

    ``pip install -e .`` injects it; a stray ``PYTHONPATH=.`` does too. Either
    makes the rest of this script meaningless, so it is checked explicitly and
    reported first.
    """
    source_root = _locate_source_root()
    if not source_root:
        return
    for entry in sys.path:
        if not entry:
            continue
        try:
            candidate = Path(entry).resolve()
        except OSError:
            continue
        if candidate == source_root:
            _fail(
                f"repository root {source_root} is on sys.path (index "
                f"{sys.path.index(entry)}). The wheel under test is being "
                f"shadowed by the checkout -- run from a different directory."
            )
            return
    _ok("repository root is not on sys.path")


def check_package_data() -> None:
    """Bundled data files must exist inside the installed package."""
    for package_name, required_dirs in REQUIRED_PACKAGE_DATA.items():
        try:
            package_root = importlib.resources.files(package_name)
        except (ImportError, ModuleNotFoundError, TypeError) as exc:
            _fail(f"cannot resolve package data root for {package_name!r}: {exc}")
            continue

        for required in required_dirs:
            try:
                target = package_root / required
                exists = target.is_dir()
            except (OSError, NotADirectoryError, TypeError) as exc:
                _fail(f"{package_name}/{required}: cannot stat ({exc})")
                continue
            if not exists:
                _fail(
                    f"missing package data {package_name}/{required}/ -- "
                    f"add it to [tool.setuptools.package-data] in pyproject.toml"
                )
                continue
            try:
                count = sum(1 for p in target.iterdir() if p.is_file())
            except OSError as exc:
                _fail(f"{package_name}/{required}: cannot list ({exc})")
                continue
            if count == 0:
                _fail(f"package data {package_name}/{required}/ is present but empty")
                continue
            _ok(f"{package_name}/{required}/ -> {count} file(s)")


def check_console_script() -> None:
    """The ``paleoast`` entry point must be installed and importable.

    ``[project.scripts] paleoast = "main:main"`` depends on ``main`` being
    shipped via ``py-modules``; without that the generated wrapper raises
    ModuleNotFoundError only when a user actually runs the command.
    """
    try:
        scripts = importlib.metadata.distribution("paleoast").entry_points
    except importlib.metadata.PackageNotFoundError:
        _fail(
            "distribution 'paleoast' is not installed -- this script must run "
            "against an installed wheel, not the checkout"
        )
        return

    console = [ep for ep in scripts if ep.group == "console_scripts"]
    if not console:
        _fail("installed distribution declares no console_scripts")
        return

    by_name = {ep.name: ep for ep in console}
    if CONSOLE_SCRIPT not in by_name:
        _fail(f"console script {CONSOLE_SCRIPT!r} missing; found {sorted(by_name) or 'none'}")
        return

    ep = by_name[CONSOLE_SCRIPT]
    if f"{ep.module}:{ep.attr}" != CONSOLE_SCRIPT_TARGET:
        _fail(f"console script {CONSOLE_SCRIPT!r} points at {ep.module}:{ep.attr}, expected {CONSOLE_SCRIPT_TARGET}")
        return

    if not callable(getattr(ep, "load", lambda: None)()):
        _fail(f"console script {CONSOLE_SCRIPT!r} target {CONSOLE_SCRIPT_TARGET} is not callable")
        return
    _ok(f"{CONSOLE_SCRIPT} -> {CONSOLE_SCRIPT_TARGET}")


def check_manifest() -> None:
    """Every declared package must appear in the installed distribution's
    top-level file list.

    ``importlib`` succeeding is strong evidence, but a package can import and
    still be missing subpackages, so cross-check the recorded manifest too.
    """
    try:
        dist = importlib.metadata.distribution("paleoast")
    except importlib.metadata.PackageNotFoundError:
        return  # already reported by check_console_script

    top_level = set()
    for f in dist.files or ():
        parts = Path(str(f)).parts
        if not parts:
            continue
        root = parts[0]
        if root.endswith((".dist-info", ".egg-info", ".data")):
            continue
        top_level.add(root.split(".")[0])

    for name in DECLARED_PACKAGES + DECLARED_PY_MODULES:
        if name not in top_level:
            _fail(
                f"{name!r} is declared in pyproject.toml but absent from the "
                f"installed manifest (top-level entries: {sorted(top_level)})"
            )
        else:
            _ok(f"manifest contains {name}")


def _locate_source_root() -> Path | None:
    """The repository root, derived from this script's own location.

    Returns ``None`` when the script is run from somewhere that is not the
    checkout, in which case the shadowing checks degrade to a no-op rather
    than reporting a bogus failure.
    """
    here = Path(__file__).resolve()
    candidate = here.parent.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def _check_matches_pyproject() -> None:
    """Fail if DECLARED_PACKAGES drifted from pyproject.toml's ``include``."""
    source_root = _locate_source_root()
    if not source_root:
        return
    pyproject = source_root / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot read {pyproject}: {exc}")
        return

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            print("  note  tomllib unavailable; skipping pyproject sync check")
            return

    try:
        data = tomllib.loads(text)
    except Exception as exc:
        _fail(f"cannot parse {pyproject}: {exc}")
        return

    find = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
    include = find.get("include", [])
    py_modules = data.get("tool", {}).get("setuptools", {}).get("py-modules", [])

    # Compare against the package roots, ignoring the trailing "*" wildcard
    # and any nested subpackage pattern.
    declared_roots = sorted({p.rstrip("*").rstrip(".") for p in include})
    if declared_roots != sorted(DECLARED_PACKAGES):
        _fail(
            "DECLARED_PACKAGES is out of sync with pyproject.toml\n"
            f"        pyproject: {declared_roots}\n"
            f"        script:    {sorted(DECLARED_PACKAGES)}"
        )
    else:
        _ok("DECLARED_PACKAGES matches pyproject.toml include")

    if sorted(py_modules) != sorted(DECLARED_PY_MODULES):
        _fail(
            "DECLARED_PY_MODULES is out of sync with pyproject.toml\n"
            f"        pyproject: {sorted(py_modules)}\n"
            f"        script:    {sorted(DECLARED_PY_MODULES)}"
        )
    else:
        _ok("DECLARED_PY_MODULES matches pyproject.toml py-modules")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    global _VERBOSE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print every successful check, not just failures",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable summary on stdout as the last line",
    )
    args = parser.parse_args()
    _VERBOSE = args.verbose

    source_root = _locate_source_root()
    print("PaleoAST wheel import smoke test")
    print(f"  python     {sys.version.split()[0]}")
    print(f"  prefix     {sys.prefix}")
    print(f"  source root{'' if source_root else ' (not found)':<12}{source_root or '-'}")
    print(f"  cwd        {Path.cwd()}")
    print(f"  QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '<unset>')}")
    print()

    # Order matters: the sys.path check explains the import failures that
    # follow, so it has to run first.
    check_not_on_source_path()
    _check_matches_pyproject()
    check_imports_resolve_outside_source_tree()
    check_package_data()
    check_console_script()
    check_manifest()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s)")
        for failure in _FAILURES:
            print(f"  - {failure}")
        if args.json:
            print(json.dumps({"ok": False, "failures": _FAILURES}))
        return 1

    print(
        f"PASSED: {len(DECLARED_PACKAGES)} packages + "
        f"{len(DECLARED_PY_MODULES)} py-modules, package data, console script"
    )
    if args.json:
        print(json.dumps({"ok": True, "failures": []}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
