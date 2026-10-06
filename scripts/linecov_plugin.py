"""Minimal line-coverage collector (stdlib only).

`coverage` and `pytest-cov` are not installed in this venv and installing
them is not permitted here, and the stdlib `trace` module aborts on a binary
file in the tree. So this is a small, self-contained line tracer, scoped to
the scientific packages so the slowdown stays tolerable.

Enable with:  -p linecov_plugin   (run from the repo root)
Results land in .linecov.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = (
    "stats",
    "ecology",
    "phylogenetics",
    "stratigraphy",
    "morphometrics",
    "morpho3d",
    "macroevolution",
    "parsers",
    "models",
    "reporting",
    "state_machine",
    "plugins",
    "hpc",
)

# Which lines are executable. Derived from the AST so that docstrings,
# comments and multi-line expressions are not counted as missed lines.
import ast

_EXECUTABLE: dict[str, set[int]] = {}


def _build_line_index() -> None:
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, ValueError):
                continue
            lines: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.stmt, ast.expr)) or (isinstance(node, ast.ExceptHandler) and node.lineno):
                    lines.add(node.lineno)
            _EXECUTABLE[str(path)] = lines


_build_line_index()
_HIT: dict[str, set[int]] = {}


def pytest_configure(config):
    if os.environ.get("LINECOV_DISABLE"):
        return
    target = str(ROOT)

    def _tracer(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if not filename.startswith(target) or filename not in _EXECUTABLE:
                return None
            return _line_tracer
        return None

    def _line_tracer(frame, event, arg):
        if event == "line":
            filename = frame.f_code.co_filename
            if filename in _EXECUTABLE:
                _HIT.setdefault(filename, set()).add(frame.f_lineno)
        return _line_tracer

    sys.settrace(_tracer)
    threading_profile = getattr(sys, "setprofile", None)
    if threading_profile:
        threading_profile(_tracer)


def pytest_sessionfinish(session, exitstatus):
    report = {}
    for filename, executable in _EXECUTABLE.items():
        if not executable:
            continue
        hit = _HIT.get(filename, set())
        covered = len(executable & hit)
        report[filename] = {"covered": covered, "total": len(executable)}

    out = ROOT / ".linecov.json"
    out.write_text(json.dumps(report), encoding="utf-8")
    sys.settrace(None)
