#!/usr/bin/env python
"""Fill the per-analysis tables in ``docs/methods/``.

WHY THIS EXISTS
~~~~~~~~~~~~~~~
A method document that says "Shapiro-Wilk assumes the sample is drawn
from a continuous distribution" is worth a great deal to a reader -- and
worth nothing if it quietly falls out of date with the code.

So the two halves are separated by responsibility:

  * the PROSE (assumptions, applicability, references) is written by
    hand, because it is a statement about the science;
  * the TABLE (which analyses exist, where they live, and what evidence
    exists for each) is generated, because it is a statement about the
    code.

The table lives between ``<!-- matrix:start -->`` and
``<!-- matrix:end -->`` in each file, and this script rewrites what is
between them.  Run it after adding or renaming an analysis.

The evidence column comes from the same matrix that
``scripts/validation_matrix.py`` builds, so the two can never disagree.

Usage:
    python scripts/method_docs.py            # rewrite the tables
    python scripts/method_docs.py --check    # fail if they are stale
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plugins.catalog import BUILTIN_ANALYSES
from scripts.validation_matrix import build

DOCS = REPO / "docs" / "methods"
START = "<!-- matrix:start -->"
END = "<!-- matrix:end -->"

#: Which method document each analysis category belongs to.
DOMAINS: dict[str, str] = {
    "univariate": "univariate.md",
    "distribution": "univariate.md",
    "group-comparison": "group-comparison.md",
    "ordination": "ordination.md",
    "spatial": "ordination.md",
    "diversity": "diversity.md",
    "morphometrics": "morphometrics.md",
    "stratigraphy": "stratigraphy.md",
    "time-series": "stratigraphy.md",
    "directional": "stratigraphy.md",
    "macroevolution": "evolution.md",
    "growth": "evolution.md",
    "clustering": "clustering.md",
    "plot": "plot.md",
}

EVIDENCE_LABEL = {
    "R": "R",
    "gold": "真值",
    "prop": "性质",
    "unit": "单元",
}


def _evidence_cell(row: dict[str, object]) -> str:
    marks = [EVIDENCE_LABEL[k] for k in ("R", "gold", "prop") if row[k]]
    return " + ".join(marks) if marks else "单元"


def build_tables() -> dict[str, str]:
    """The generated table for each method document."""
    rows = {str(r["name"]): r for r in build()}
    by_domain: dict[str, list[tuple[str, dict[str, object]]]] = {name: [] for name in DOMAINS.values()}
    for entry in BUILTIN_ANALYSES:
        domain = DOMAINS.get(entry.category)
        if domain is None:
            raise SystemExit(f"no method document for category {entry.category!r} (analysis {entry.name!r})")
        by_domain[domain].append((entry.name, rows[entry.name]))

    tables: dict[str, str] = {}
    for domain, entries in by_domain.items():
        lines = [
            "| 分析 | 模块 | 验证证据 | 说明 |",
            "|---|---|---|---|",
        ]
        for name, row in sorted(entries):
            description = str(row.get("description", "")).replace("|", "\\|")
            lines.append(f"| `{name}` | `{row['module']}` | {_evidence_cell(row)} | {description} |")
        tables[domain] = "\n".join(lines)
    return tables


def _rewrite(path: pathlib.Path, table: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{path} is missing the {START} / {END} markers")
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    updated = f"{head}{START}\n{table}\n{END}{tail}"
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting")
    args = parser.parse_args()

    tables = build_tables()
    stale: list[str] = []
    for domain, table in sorted(tables.items()):
        path = DOCS / domain
        if not path.exists():
            raise SystemExit(f"missing method document: {path.relative_to(REPO)}")
        if args.check:
            text = path.read_text(encoding="utf-8")
            head, _, rest = text.partition(START)
            _, _, tail = rest.partition(END)
            if f"{head}{START}\n{table}\n{END}{tail}" != text:
                stale.append(domain)
        else:
            changed = _rewrite(path, table)
            print(f"{'rewrote' if changed else 'unchanged'}  {path.relative_to(REPO)}")

    if stale:
        print(
            "\nthese method documents have stale tables -- run `python scripts/method_docs.py`: " + ", ".join(stale),
            file=sys.stderr,
        )
        return 1
    if args.check:
        print("every method document is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
