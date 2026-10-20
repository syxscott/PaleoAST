"""Scan for functions where ``_`` shadows the translation helper.

``_`` is this project's translation function (``from config.i18n import _``).
If a function also binds ``_`` as a local -- through a loop variable, a
tuple-unpacking target, a plain assignment, a ``with`` target or an
``except ... as _`` -- Python makes ``_`` a LOCAL for the entire function
scope, so every ``_("...")`` inside it resolves to that binding instead of
the translator.

Two failure modes, both invisible until the branch is reached:
  * the binding comes first -> ``UnboundLocalError`` on the error path
  * the binding comes later -> ``_("msg")`` calls the last value of the
    loop variable -> ``TypeError: 'numpy.ndarray' object is not callable``

This is mechanically enumerable, so it does not need a human reading the
code. Run it as a gate:

    python scripts/scan_underscore_shadow.py

NOTE when extending the detector: tuple-unpacking targets
(``a, _, _, _ = f()``) count. A first version that only looked at
``ast.For`` and ``ast.Assign`` missed them and reported only the one
site its author had just written, hiding two that were already in the
tree.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "__pycache__",
    "build",
    "dist",
    "paleoast.egg-info",
    ".claude",
    ".github",
    "node_modules",
    ".venv",
    "venv",
}
TEST_ROOTS = {"tests", "tests_morpho3d_macroevolution"}


def binds_underscore(node: ast.AST) -> bool:
    """Does this statement bind the name ``_``?"""
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_":
                return True
            if isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    if isinstance(element, ast.Name) and element.id == "_":
                        return True
    if isinstance(node, ast.AnnAssign):
        target = node.target
        if isinstance(target, ast.Name) and target.id == "_":
            return True
    if isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            optional = item.optional_vars
            if isinstance(optional, ast.Name) and optional.id == "_":
                return True
    if isinstance(node, ast.ExceptHandler) and node.name == "_":
        return True
    return False


def calls_translator(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not node.args:
        return False
    func = node.func
    name = getattr(func, "id", None) or getattr(func, "attr", None)
    return name == "_" and isinstance(node.args[0], ast.Constant)


def scan(path: Path) -> list[tuple[str, int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    hits: list[tuple[str, int, str]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(func.body)
        has_bind = False
        bind_line = 0
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.For, ast.AsyncFor)):
                    target = node.target
                    if isinstance(target, ast.Name) and target.id == "_":
                        has_bind = True
                        bind_line = node.lineno
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for element in target.elts:
                            if isinstance(element, ast.Name) and element.id == "_":
                                has_bind = True
                                bind_line = node.lineno
                elif binds_underscore(node):
                    has_bind = True
                    bind_line = getattr(node, "lineno", bind_line)
        if not has_bind:
            continue
        for stmt in body:
            for node in ast.walk(stmt):
                if calls_translator(node):
                    hits.append(
                        (
                            str(path.relative_to(ROOT)).replace("\\", "/"),
                            func.lineno,
                            f"{func.name}() binds '_' at line {bind_line} and calls _() at line {node.lineno}",
                        )
                    )
                    break
            else:
                continue
            break
    return hits


def main() -> int:
    files = []
    for path in ROOT.rglob("*.py"):
        if set(path.parts) & SKIP_DIR_NAMES:
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in TEST_ROOTS:
            continue
        files.append(path)

    all_hits: list[tuple[str, int, str]] = []
    for path in sorted(files):
        all_hits.extend(scan(path))

    print(f"scanned {len(files)} product files")
    print(f"\n=== FUNCTIONS WHERE '_' SHADOWS THE TRANSLATOR ({len(all_hits)}) ===")
    for rel, line, detail in all_hits:
        print(f"{rel}:{line}  {detail}")
    if not all_hits:
        print("(none)")
    return 1 if all_hits else 0


if __name__ == "__main__":
    sys.exit(main())
