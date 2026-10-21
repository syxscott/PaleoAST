#!/usr/bin/env python
"""Build the validation-coverage matrix for every first-party analysis.

WHAT THIS IS FOR
~~~~~~~~~~~~~~~~
The review found five defects that would each have put a wrong number in a
paper -- an isometry test whose degrees of freedom did not match its
statistic, a growth-model interval that was 3.3x too narrow, a PCA
residual sum-of-squares decomposition that was simply wrong.  Every one of
them was found by reading the code.  None of them was found by a test.

A professional research package closes that gap by being able to say, for
every analysis it offers, HOW the implementation was validated:

  R      -- compared against a reference implementation in R (vegan, ape,
            geomorph, iNext, base R, MASS).  The strongest evidence there
            is: two independent implementations agreeing.
  gold   -- checked against a published or independently derived ground
            truth (Palaeoverse values, the hummingbird and scallop
            datasets, exhaustive Procrustes searches).
  prop   -- a property test: an invariant that must hold for every input,
            not just the ones in the example.
  unit   -- the function returns the expected value on a chosen example.
            Necessary, and not sufficient.

The matrix makes the shape of the gap visible instead of leaving it as a
vague worry, and it gives the next batch of work a ranked worklist.

HOW TO READ IT
~~~~~~~~~~~~~~
A row with an empty "R" column is not a defect by itself -- plenty of
analyses have no R counterpart.  A row with an empty "unit" column IS a
defect: the analysis is reachable from the interface and nothing tests it.

Usage:
    python scripts/validation_matrix.py            # print the table
    python scripts/validation_matrix.py --markdown  # write docs/validation-matrix.md
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plugins.catalog import BUILTIN_ANALYSES

TESTS = REPO / "tests"
GIVEN = re.compile(r"@given")

# The matrix's own test file mentions analysis names, and would otherwise
# count as evidence for every one of them.
SELF = "tests/test_validation_coverage.py"


def _test_files() -> list[pathlib.Path]:
    return sorted(p for p in TESTS.rglob("test_*.py") if p.relative_to(REPO).as_posix() != SELF)


def _classify(path: pathlib.Path) -> str:
    parts = set(path.parts)
    if "cross_validation" in parts:
        return "R"
    if "golden" in parts:
        return "gold"
    if "property" in parts:
        return "prop"
    return "unit"


def _mentions(path: pathlib.Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _declared_analyses(text: str) -> set[str] | None:
    """The analyses a test file says it covers, if it says.

    A test file may declare ``ANALYSES_COVERED = {"pca", "nmds"}`` at module
    level.  When it does, that declaration is authoritative: the file is
    evidence for those analyses and for nothing else.

    This matters because the fallback is a NAME MATCH, and a name match
    over-reports badly.  A file that imports ``compute_diversity_indices``
    mentions "diversity", so every analysis whose name starts with
    "diversity" looks validated -- fourteen of them -- when in fact the
    file compared one.  A matrix that says "R" for an analysis nobody
    compared against R is worse than no matrix: it makes the gap look
    smaller than it is, and the ratchet built on it would let the gap
    grow in peace.
    """
    match = re.search(r"^ANALYSES_COVERED\s*(?::\s*[^=]+)?=\s*\{(.*?)\}", text, re.MULTILINE | re.DOTALL)
    if not match:
        return None
    return set(re.findall(r"[\"']?([a-z0-9_]+)[\"']?", match.group(1)))


def build() -> list[dict[str, object]]:
    files = _test_files()
    cache = {p: p.read_text(encoding="utf-8", errors="ignore") for p in files}
    rows: list[dict[str, object]] = []
    for entry in BUILTIN_ANALYSES:
        # A few entries are module-level functions rather than a class
        # (ProcD, the orbital-forcing helpers), so the class name is None
        # and the method name is the thing to look for.
        needles = [n for n in (entry.class_name, entry.method) if n]
        kinds: set[str] = set()
        hits: list[str] = []
        for path, text in cache.items():
            declared = _declared_analyses(text)
            if declared is not None:
                # The file says what it covers; believe it.
                if entry.name in declared:
                    kinds.add(_classify(path))
                    hits.append(path.relative_to(REPO).as_posix())
                continue
            # The fallback: the class (or method) name has to appear AND
            # the module has to appear.  Either alone is far too loose --
            # a class name credits every analysis that shares that class,
            # and a module name credits every analysis in it.
            #
            # This rule UNDER-counts on purpose.  A ratchet that credits
            # coverage that is not there would let the gap be declared
            # closed on a false positive; a ratchet that misses coverage
            # only makes the number look worse than it is.
            module_tail = entry.module.rsplit(".", 1)[-1]
            if any(needle in text for needle in needles) and module_tail in text:
                kinds.add(_classify(path))
                hits.append(path.relative_to(REPO).as_posix())
        rows.append(
            {
                "name": entry.name,
                "category": entry.category,
                "module": entry.module,
                "class": entry.class_name,
                "method": entry.method,
                "description": entry.description or "",
                "R": "R" in kinds,
                "gold": "gold" in kinds,
                "prop": "prop" in kinds,
                "unit": "unit" in kinds,
                "any": bool(kinds),
                "hits": hits,
            }
        )
    return rows


def _summary(rows: list[dict[str, object]]) -> str:
    total = len(rows)
    r = sum(1 for x in rows if x["R"])
    gold = sum(1 for x in rows if x["gold"])
    prop = sum(1 for x in rows if x["prop"])
    unit = sum(1 for x in rows if x["unit"])
    none = sum(1 for x in rows if not x["any"])
    both = sum(1 for x in rows if x["R"] and (x["gold"] or x["prop"]))
    # The number that matters: an analysis whose only evidence is one
    # worked example.  A unit test is necessary and not sufficient.
    thin = sum(1 for x in rows if x["unit"] and not (x["R"] or x["gold"] or x["prop"]))
    lines = [
        f"analyses:            {total}",
        f"  compared with R:   {r:>3}  ({100 * r / total:.0f}%)   EXACT -- declared per file",
        f"  golden reference:  {gold:>3}  ({100 * gold / total:.0f}%)   inferred",
        f"  property test:     {prop:>3}  ({100 * prop / total:.0f}%)   inferred",
        f"  unit test:         {unit:>3}  ({100 * unit / total:.0f}%)   inferred",
        f"  no test at all:    {none:>3}  ({100 * none / total:.0f}%)   inferred",
        f"  R + (gold|prop):   {both:>3}  ({100 * both / total:.0f}%)",
        f"  UNIT TEST ONLY:    {thin:>3}  ({100 * thin / total:.0f}%)   <- the gap (lower bound)",
        "",
        "The R row is exact: every cross-validation file declares the analyses",
        "it covers.  The other rows are inferred from the test tree, and they",
        "UNDER-count on purpose -- a ratchet that credited coverage that is not",
        "there would let the gap be declared closed on a false positive.",
    ]
    by_cat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for row in rows:
        entry = by_cat[str(row["category"])]
        entry[0] += 1
        if row["R"]:
            entry[1] += 1
        if row["unit"] and not (row["R"] or row["gold"] or row["prop"]):
            entry[2] += 1
    lines.append("")
    lines.append(f"{'category':<22}{'n':>4}{'R':>5}{'thin':>8}")
    for cat in sorted(by_cat):
        n, r, thin = by_cat[cat]
        lines.append(f"{cat:<22}{n:>4}{r:>5}{thin:>8}")
    return "\n".join(lines)


def _table(rows: list[dict[str, object]]) -> str:
    out = [
        "| analysis | category | module | R | gold | prop | unit |",
        "|---|---|---|:--:|:--:|:--:|:--:|",
    ]
    for row in rows:
        mark = {k: "x" if row[k] else "" for k in ("R", "gold", "prop", "unit")}
        out.append(
            f"| `{row['name']}` | {row['category']} | `{row['module']}` "
            f"| {mark['R']} | {mark['gold']} | {mark['prop']} | {mark['unit']} |"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", metavar="PATH", help="write the matrix to PATH")
    parser.add_argument(
        "--thin",
        action="store_true",
        help="list only the analyses whose sole evidence is a unit test",
    )
    args = parser.parse_args()

    rows = build()
    if args.thin:
        thin = [r for r in rows if r["unit"] and not (r["R"] or r["gold"] or r["prop"])]
        by_category: dict[str, list[str]] = collections.defaultdict(list)
        for row in thin:
            by_category[str(row["category"])].append(str(row["name"]))
        print(f"{len(thin)} analyses rest on a unit test alone\n")
        for category in sorted(by_category, key=lambda c: (-len(by_category[c]), c)):
            print(f"{category} ({len(by_category[category])}):")
            for name in sorted(by_category[category]):
                print(f"    {name}")
            print()
        return 0

    print(_summary(rows))
    if args.markdown:
        target = pathlib.Path(args.markdown)
        if not target.is_absolute():
            target = REPO / target
        target.parent.mkdir(parents=True, exist_ok=True)
        body = [
            "# Validation coverage matrix",
            "",
            "Generated by `scripts/validation_matrix.py`. Do not edit by hand:",
            "the table is rebuilt from the catalog and the test tree.",
            "",
            "What each column means:",
            "",
            "- **R** -- compared against a reference implementation in R.",
            "- **gold** -- checked against published or independently derived truth.",
            "- **prop** -- a property test: an invariant for every input, not one example.",
            "- **unit** -- returns the expected value on a chosen example.",
            "",
            "```",
            _summary(rows),
            "```",
            "",
            _table(rows),
            "",
        ]
        target.write_text("\n".join(body), encoding="utf-8")
        print(f"\nwrote {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
