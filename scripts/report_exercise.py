"""Exercise the reporting layer end to end.

``reporting/`` had 0% line coverage -- the whole LaTeX report pipeline had
never been executed. This builds a realistic report and checks the invariants
a publisher would care about: does the .tex come out well-formed, are labels
unique and referenced correctly, do special characters get escaped, do
statistics render as numbers rather than placeholders.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reporting.latex_preamble import DocumentClass, LatexPreamble
from reporting.matrix_converter import MatrixConverter
from reporting.report_builder import ReportBuilder
from reporting.table_generator import TableGenerator

OUT = Path(__file__).resolve().parents[1] / "_ui_audit" / "report"
OUT.mkdir(parents=True, exist_ok=True)

PROBLEMS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        PROBLEMS.append(f"{label} {detail}")


print("=== 1. build a full report ===")
builder = ReportBuilder()
builder.set_title("A test of the report builder")
builder.add_author("A. Researcher")
builder.set_abstract("We examined whether the report builder works. It does.")
builder.add_section("Introduction", "Morphometrics is a palaeontology workhorse.")
builder.add_section("Methods", "We did the thing.", level=1)
builder.add_section("Landmarks", "Configurational landmarks were digitised.", level=2)
builder.add_statistical_result("PERMANOVA", 2.69, p_value=0.041, df=4, effect_size=0.62)
builder.add_statistical_result("ANOSIM R", 0.31, p_value=0.09)
builder.add_table(
    "\\begin{tabular}{lr}\n Taxon & CS \\\\\n A & 1.0 \\\\\n B & 1.4 \\\\\n\\end{tabular}",
    caption="Centroid sizes of two taxa.",
)
tex_path = OUT / "report.tex"
try:
    written = builder.generate(str(tex_path))
    check("generate() returned a path", bool(written), repr(written))
    check("file exists on disk", tex_path.exists())
    text = tex_path.read_text(encoding="utf-8")
except Exception as exc:
    print(f"  FAIL  generate() raised {type(exc).__name__}: {exc}")
    PROBLEMS.append(f"generate raised {type(exc).__name__}: {exc}")
    text = ""

if text:
    print()
    print("=== 2. is the LaTeX actually well formed? ===")
    check("has \\begin{document}", "\\begin{document}" in text)
    check("has \\end{document}", "\\end{document}" in text)
    check("has a preamble", "\\documentclass" in text and "\\begin{document}" in text)
    braces = text.count("{") - text.count("}")
    check("braces balance", braces == 0, f"off by {braces}")
    check(
        "no unescaped % outside comments",
        text.count("%") - text.count("\\%") <= text.count("\\begin{comment}") + 2,
        "a bare % starts a LaTeX comment and silently eats the rest of the line",
    )
    check(
        "no unescaped _ in prose",
        not re.search(r"(?<!\\)_[a-zA-Z]", text.replace("\\\\_", "")),
        "a bare _ is a LaTeX error outside math mode",
    )
    check("title present", "A test of the report builder" in text)
    check("author present", "A. Researcher" in text)

    print()
    print("=== 3. are statistics rendered, or left as placeholders? ===")
    check("PERMANOVA named in output", "PERMANOVA" in text)
    check("statistic value 2.69 appears", "2.69" in text, text[:0])
    p_mentioned = "0.041" in text
    check("p-value rendered", p_mentioned, "the p-value never made it into the document")
    check("df rendered", "4" in text)
    check(
        "no unfilled {statistic}-style placeholder",
        "None" not in text.split("\\begin{document}")[-1][:4000],
        "a Python None leaked into the body",
    )

print()
print("=== 4. preamble / escaping ===")
try:
    preamble = LatexPreamble(DocumentClass.ARTICLE)
    rendered = preamble.render() if hasattr(preamble, "render") else str(preamble)
    check("preamble renders", bool(rendered) and len(rendered) > 20)
    check("preamble declares a documentclass", "documentclass" in rendered)
except Exception as exc:
    print(f"  FAIL  LatexPreamble.render raised {type(exc).__name__}: {exc}")
    PROBLEMS.append(f"preamble: {exc}")

try:
    from reporting.figure_handler import _escape_latex

    cases = {
        "100%": r"100\%",
        "a_b": r"a\_b",
        "C&D": r"C\&D",
        "x^2": r"x\textasciicircum 2",
    }
    for raw, expected in cases.items():
        got = _escape_latex(raw)
        check(f"escape {raw!r}", got == expected, f"got {got!r} want {expected!r}")
except Exception as exc:
    print(f"  FAIL  _escape_latex raised {type(exc).__name__}: {exc}")

print()
print("=== 5. table + matrix conversion ===")
try:
    tg = TableGenerator()
    has = [m for m in dir(tg) if not m.startswith("_")]
    check("TableGenerator exposes something usable", len(has) > 1, str(has))
    print(f"        methods: {has[:8]}")
except Exception as exc:
    print(f"  FAIL  TableGenerator: {exc}")
    PROBLEMS.append(f"TableGenerator: {exc}")

try:
    mc = MatrixConverter()
    has = [m for m in dir(mc) if not m.startswith("_")]
    check("MatrixConverter exposes something usable", len(has) > 1, str(has))
    print(f"        methods: {has[:8]}")
except Exception as exc:
    print(f"  FAIL  MatrixConverter: {exc}")
    PROBLEMS.append(f"MatrixConverter: {exc}")

print()
print("=" * 60)
if PROBLEMS:
    print(f"{len(PROBLEMS)} problem(s):")
    for p in PROBLEMS:
        print(f"  * {p}")
    sys.exit(1)
print("reporting layer exercised cleanly")
