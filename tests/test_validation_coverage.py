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

# 2026-10-20: 74 of 113 analyses rested on a unit test alone, on the
# first count -- which turned out to be too generous.  The count matched a
# test file by the analysis' CLASS name, so one file importing
# ``compute_diversity_indices`` credited all fourteen diversity analyses
# with R coverage, and the R column read 38 when only 10 analyses have
# ever been compared against R.
#
# The rule is now: a test file may DECLARE the analyses it covers, and
# that declaration is authoritative; otherwise a file counts only if it
# mentions both the class and the module, which under-counts on purpose --
# a ratchet that credits coverage that is not there would let the gap be
# declared closed on a false positive.
#
# The number is a conservative lower bound on the gap.  Lower it when
# an analysis gains an R comparison, a golden reference or a property
# test -- and say in the commit message which one.
#
# 2026-10-21: 97 -> 87.  Ten univariate analyses gained a live comparison
# against base R: the two-way ANOVA, the repeated-measures ANOVA, the
# intraclass correlation, the exact binomial proportion test, the
# Fligner-Kileen test, the descriptive statistics, the outlier screen,
# the Brunner-Munzel test, Mood's median test and the
# Cochran-Mantel-Haenszel test.  The univariate domain now has all
# sixteen of its analyses compared against R, and none of them rests on a
# single worked example.
#
# 2026-10-21: 87 -> 78.  Nine of the twenty plot analyses gained one: the
# XY scatter, the XY scatter with error bars, the histogram, the bar
# chart, the box plot, the pie chart, the stacked chart, the normal
# probability plot and the percentile plot.  What is compared is the data
# behind the figure -- the counts in the bars, the quartiles in the box,
# the angles in the pie -- read back out of the matplotlib artists and
# checked against R.
#
# 2026-10-21: 78 -> 71.  Fifty-eight new property tests -- the univariate
# tests, the design tests, the PCA, the diversity indices, Moran's I and
# the clustering -- each giving an analysis an invariant that holds for
# every input of a shape rather than for one worked example.
THIN_BASELINE = 71

#: The number of analyses compared against a reference implementation in
#: R.  This one is EXACT, because the cross-validation files declare it.
#:
#: 2026-10-21: 10 -> 20, for the ten univariate analyses.
#: 2026-10-21: 20 -> 29, for the nine plot analyses.
R_BASELINE = 29

#: The number of analyses with a PROPERTY test -- an invariant that must
#: hold for every input, not for one example.
#:
#: This one is INFERRED, like the unit column, so it under-counts on
#: purpose.  It is held to a floor because a property test is the
#: strongest evidence the matrix knows about: it says the answer is
#: right for the whole class of inputs and not only for the one in the
#: test file.
PROP_BASELINE = 30

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
        "The gap may shrink; it may not widen."
    )


def test_the_r_comparison_count_does_not_shrink():
    """The R column is exact, so it can be held to a floor.

    Every analysis compared against a reference implementation in R is
    evidence that cannot go away by accident, and the cross-validation
    files declare what they cover, so this number is exact rather than
    inferred.
    """
    rows = build()
    with_r = [str(r["name"]) for r in rows if r["R"]]
    assert len(with_r) >= R_BASELINE, (
        f"only {len(with_r)} analyses are compared against R (was {R_BASELINE}): {sorted(with_r)}. "
        "A cross-validation file must declare the analyses it covers in ANALYSES_COVERED."
    )


def test_the_property_test_count_does_not_shrink():
    """The property column is the strongest evidence the matrix knows.

    An invariant that holds for every input of a shape says the answer
    is right for the whole class of inputs, and not only for the one in
    the test file.  The count is inferred, so it under-counts on purpose,
    but it must not go backwards.
    """
    rows = build()
    with_property = [str(r["name"]) for r in rows if r["prop"]]
    assert len(with_property) >= PROP_BASELINE, (
        f"only {len(with_property)} analyses have a property test (was {PROP_BASELINE}): {sorted(with_property)}. "
        "A property test is an invariant that must hold for every input, not for one example."
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
