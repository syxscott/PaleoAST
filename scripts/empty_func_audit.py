"""Mechanical audit: empty functions, stubs, duplicate definitions, mojibake.

Enumeration only. It answers "what is empty?" exactly, so the semantic
review can focus on "is the empty thing wrong?" rather than hunting for
trivia.

Categories
----------
empty_pass          body is exactly ``pass``
empty_docstring     body is a docstring only
empty_ellipsis      body is ``...`` only
empty_return_none   body is ``return None`` only
not_implemented     body raises NotImplementedError only
dup_def             same name defined twice in one scope (accessors excluded)
dup_method          same method name twice in one class body
replacement_char    U+FFFD, i.e. mojibake / lost bytes

A property getter and its ``@x.setter`` share a name on purpose, so
accessors are excluded from the duplicate-definition check; counting them
flags every property in the codebase.

Run:
    python scripts/empty_func_audit.py
    python scripts/empty_func_audit.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
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

MARKER_WORDS = ("TODO", "FIXME", "XXX", "HACK")
PROTOCOL_MARKERS = ("not implemented", "todo", "fixme", "override", "subclass", "abstract")


def is_docstring_stmt(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def classify_body(body: list[ast.stmt]) -> str | None:
    """Return the empty-category name for a function body, else None."""
    if not body:
        return "empty_body"
    if len(body) == 1:
        only = body[0]
        if isinstance(only, ast.Pass):
            return "empty_pass"
        if is_docstring_stmt(only):
            return "empty_docstring"
        if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) and only.value.value is Ellipsis:
            return "empty_ellipsis"
        if isinstance(only, ast.Return):
            return "empty_return_none" if only.value is None else None
        if isinstance(only, ast.Raise):
            exc = only.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                return "not_implemented"
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return "not_implemented"
        return None
    if is_docstring_stmt(body[0]) and len(body) == 2 and isinstance(body[1], ast.Pass):
        return "empty_pass"
    return None


def scan_file(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return {"_syntax_error": f"{path}:{exc.lineno}: {exc.msg}"}

    out: dict = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "empty": [],
        "dup_def": [],
        "dup_method": [],
        "protocol": [],
        "replacement_char": [],
        "markers": [],
    }
    lines = text.splitlines()

    for i, line in enumerate(lines, 1):
        if "\ufffd" in line:
            out["replacement_char"].append({"line": i, "text": line.strip()[:160]})

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node
            seen: dict[str, int] = {}
            for child in scope.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                is_accessor = False
                for dec in child.decorator_list:
                    if isinstance(dec, ast.Attribute) and dec.attr in {"setter", "deleter", "getter"}:
                        is_accessor = True
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        if dec.func.attr in {"setter", "deleter", "getter"}:
                            is_accessor = True
                if is_accessor:
                    continue
                if child.name in seen:
                    out["dup_method" if isinstance(scope, ast.ClassDef) else "dup_def"].append(
                        {
                            "name": child.name,
                            "first_line": seen[child.name],
                            "dup_line": child.lineno,
                            "scope": getattr(scope, "name", "<module>"),
                        }
                    )
                else:
                    seen[child.name] = child.lineno

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cat = classify_body(node.body)
            if cat is None:
                continue
            doc = ast.get_docstring(node) or ""
            entry = {
                "name": node.name,
                "line": node.lineno,
                "category": cat,
                "decorated": bool(node.decorator_list),
                "doc": doc.strip().splitlines()[0][:160] if doc.strip() else "",
            }
            if any(m in doc.lower() for m in PROTOCOL_MARKERS):
                out["protocol"].append(entry)
            else:
                out["empty"].append(entry)

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        upper = stripped.upper()
        for word in MARKER_WORDS:
            if word in upper:
                out["markers"].append({"line": i, "word": word, "text": stripped[:160]})
                break

    return {k: v for k, v in out.items() if v}


def iter_py_files(include_tests: bool) -> list[Path]:
    files = []
    for path in ROOT.rglob("*.py"):
        if set(path.parts) & SKIP_DIR_NAMES:
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in TEST_ROOTS and not include_tests:
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--include-tests", action="store_true")
    args = ap.parse_args()

    files = iter_py_files(args.include_tests)
    empty: list[dict] = []
    dup: list[dict] = []
    dupm: list[dict] = []
    markers: list[dict] = []
    protocol: list[dict] = []
    repl: list[dict] = []
    syntax: list[str] = []

    for path in files:
        res = scan_file(path)
        if not res:
            continue
        if "_syntax_error" in res:
            syntax.append(res["_syntax_error"])
            continue
        for key, bucket in (
            ("empty", empty),
            ("dup_def", dup),
            ("dup_method", dupm),
            ("markers", markers),
            ("protocol", protocol),
            ("replacement_char", repl),
        ):
            for entry in res.get(key, []):
                entry["file"] = res["path"]
                bucket.append(entry)

    if args.json:
        print(
            json.dumps(
                {
                    "empty": empty,
                    "dup_def": dup,
                    "dup_method": dupm,
                    "markers": markers,
                    "protocol": protocol,
                    "replacement_char": repl,
                    "syntax_errors": syntax,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"scanned {len(files)} files")
    print(f"\n=== EMPTY / STUB FUNCTIONS ({len(empty)}) ===")
    for e in sorted(empty, key=lambda x: (x["file"], x["line"])):
        print(f"{e['file']}:{e['line']}  {e['category']:<18} {e['name']}")
        if e["doc"]:
            print(f"      doc: {e['doc']}")
    print(f"\n=== DOCUMENTED PROTOCOL STUBS ({len(protocol)}) ===")
    for e in sorted(protocol, key=lambda x: (x["file"], x["line"])):
        print(f"{e['file']}:{e['line']}  {e['category']:<18} {e['name']}")
    print(f"\n=== DUPLICATE TOP-LEVEL DEFS ({len(dup)}) ===")
    for e in dup:
        print(f"{e['file']}:{e['dup_line']}  {e['name']} (first at line {e['first_line']})")
    print(f"\n=== DUPLICATE METHODS ({len(dupm)}) ===")
    for e in dupm:
        print(f"{e['file']}:{e['dup_line']}  {e['scope']}.{e['name']} (first at line {e['first_line']})")
    print(f"\n=== U+FFFD REPLACEMENT CHAR ({len(repl)}) ===")
    for e in repl:
        print(f"{e['file']}:{e['line']}  {e['text']}")
    print(f"\n=== TODO/FIXME/XXX/HACK COMMENTS ({len(markers)}) ===")
    for e in markers:
        print(f"{e['file']}:{e['line']}  [{e['word']}] {e['text']}")
    if syntax:
        print(f"\n=== SYNTAX ERRORS ({len(syntax)}) ===")
        for s in syntax:
            print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
