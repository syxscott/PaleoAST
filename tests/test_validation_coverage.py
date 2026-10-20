# =============================================================================
# FILE: tests/test_validation_coverage.py
# =============================================================================
"""
The validation-coverage ratchet.

WHAT THIS GUARDS
~~~~~~~~~~~~~~~~
The review found five defects that would each have put a wrong number in a
paper: an isometry test whose degrees of freedom did not match its
statistic, a growth-model interval that was 3.3x too narrow, a PCA residual
sum-of-squares decomposition that was simply wrong, a thousand-separated
number in the first data column that swallowed the whole column, and a
permutation that divided by zero.

Every one of them was found by reading the code.  None of them was found
by a test.  That is the gap this file measures and holds shut.

``scripts/validation_matrix.py`` builds the picture: for every analysis in
the first-party catalog, what kind of evidence exists for it --

  R      compared against a reference implementation in R
  gold   checked against published or independently derived truth
  prop   a property test: an invariant for every input, not one example
  unit   returns the expected value on a chosen example

An analysis whose only evidence is ``unit`` rests on a single worked
example.  That is where the next defect is most likely to live.

THE RATCHET
~~~~~~~~~~~
``THIN_BASELINE`` is the number of analyses that rest on a unit test
alone.  It may go DOWN (when something is validated properly) and it may
not go UP.  Lowering it is a deliberate act, with the validation that
earned it recorded in the commit that does it.

THE MATRIX DOCUMENT
~~~~~~~~~~~~~~~~~~~
``docs/validation-matrix.md`` is regenerated and compared, so the table
in the repository cannot drift away from the code it describes.

A NOTE ON THE HEURISTIC
~~~~~~~~~~~~~~~~~~~~~~~~
The matrix finds a test by looking for the analysis' class name in the
test tree.  That is a NAME match, not a call graph: a test that exercises
an analysis without naming its class is not counted.  So the numbers are
a floor, not a ceiling -- the true coverage is no worse than this, and
possibly better.  The ratchet therefore measures a lower bound, which is
the safe direction for a thing that must not grow.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.validation_matrix import build

# 2026-10-20: 74 of 113 analyses rested on a unit test alone.
# Lower it when an analysis gains an R comparison, a golden reference or
# a property test -- and say in the commit message which one.
THIN_BASELINE = 74

# The ratchet for the other end: every analysis the interface offers must
# be reachable by at least one test.
UNTESTED_BASELINE = 0


def _thin(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [r for r in rows if r["unit"] and not (r["R"] or r["gold"] or r["prop"])]


def test_every_analysis_has_some_test():
    """An analysis nobody tests is a defect, not a gap."""
    rows = build()
    untested = [str(r["name"]) for r in rows if not r["any"]]
    assert len(untested) <= UNTESTED_BASELINE, f"{len(untested)} analyses have no test at all: {untested}"


def test_the_gap_does_not_grow():
    """The number of analyses resting on one worked example must not rise."""
    rows = build()
    thin = _thin(rows)
    assert len(thin) <= THIN_BASELINE, (
        f"{len(thin)} analyses rest on a unit test alone (baseline {THIN_BASELINE}). "
        "The gap may shrink; it may not widen.  Analyses that grew: "
        + ", ".join(
            sorted(str(r["name"]) for r in thin if str(r["name"]) not in {str(x["name"]) for x in _thin(build())})
        )
    )


def test_the_matrix_document_is_current():
    """The committed table must be the one the code produces."""
    import subprocess

    script = REPO / "scripts" / "validation_matrix.py"
    document = REPO / "docs" / "validation-matrix.md"
    assert document.exists(), "docs/validation-matrix.md is missing"

    before = document.read_text(encoding="utf-8")
    subprocess.run(
        [sys.executable, str(script), "--markdown", str(document)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    after = document.read_text(encoding="utf-8")
    assert before == after, (
        "docs/validation-matrix.md is out of date -- regenerate it with "
        "`python scripts/validation_matrix.py --markdown docs/validation-matrix.md`"
    )
