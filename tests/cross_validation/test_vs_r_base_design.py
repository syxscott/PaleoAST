# =============================================================================
# FILE: tests/cross_validation/test_vs_r_base_design.py
# =============================================================================
"""
Cross-validation of the design tests against base R's ``stats``.

Three analyses are compared here: the two-way factorial ANOVA, the
repeated-measures ANOVA, and the intraclass correlation coefficient.

Why these three
----------------
They are the analyses where the *decomposition* is the result.  A one-way
ANOVA has one sum of squares and one denominator; a two-way ANOVA has three
terms that must add up to the total, and a repeated-measures ANOVA has two
strata whose degrees of freedom have to be read off different error terms.
Get any of those wrong and the F ratio is wrong, and an F ratio is a number
that goes into a paper.

What R is asked, and why each choice
------------------------------------
``stats::aov`` reports the **Type I** (sequential) decomposition.  PaleoAST's
two-way ANOVA is the same decomposition -- its own docstring says so, and
the comparison here is made on a deliberately **unbalanced** design as well
as a balanced one, because on a balanced design the Type I, Type II and
Type III sums of squares coincide and a comparison there cannot tell which
one the application computes.  The unbalanced case is the one that
discriminates.

``stats::aov`` with an ``Error()`` term gives the repeated-measures
analysis its two strata, and the within-subject stratum only appears if the
measurement factor is in the model formula -- ``y ~ grp * meas +
Error(subj/meas)`` and not ``y ~ grp + Error(subj/meas)``, which silently
leaves the measurement effect out of the analysis entirely.

The intraclass correlation is compared through R's own ANOVA table and the
McGraw & Wong (1996) formulae, which are the ones ``irr::icc`` implements.
Those formulae were checked against ``irr::icc`` (0.85) locally on the data
below and agree to twelve digits.  The *interval* is not compared: PaleoAST
inverts the F ratio for both mean squares with a Bonferroni split while
``irr`` inverts a single ratio, so comparing them would compare two
interval constructions rather than two analyses.

How the comparison is run
-------------------------
Each reference is a small R program that ends by returning a **named
numeric vector**; the test reads the names back one by one.  Going through
a program rather than through R's objects directly is deliberate: the
summary tables of an ``aov`` come in two different shapes (a matrix for a
plain model, a list of named vectors per stratum when there is an
``Error()`` term), and rpy2's conversion of those shapes loses attributes
that the row names live in.  Letting R do the indexing keeps the row names
on the R side of the fence, where they are reliable.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import r_labels, r_numbers, r_program, require

pytestmark = pytest.mark.cross_validation

#: The analyses this file compares against R.  ``scripts/validation_matrix.py``
#: reads this declaration rather than guessing from imported class names.
ANALYSES_COVERED = {
    "two_way_anova",
    "repeated_measures_anova",
    "intraclass_correlation",
}

R_STATS = require("stats")

#: Tolerances.  These are the same quantities, computed by two independent
#: implementations, so they agree to the last bits of a double.
RTOL = 1e-9
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Running R
# ---------------------------------------------------------------------------
#: An R helper that turns any shape of an ``aov`` summary into a matrix with
#: clean row and column names.
#:
#: Two things force this helper to exist.  First, R's summary of a model
#: with an ``Error()`` term nests the table one level deeper than the
#: summary of a plain model -- ``summary(fit)[[1]]`` is the table for a
#: plain model but a *list holding* the table for an ``Error()`` model, and
#: the two are easy to confuse.  Second, and less obvious: the rows of an
#: ``anova`` data frame are padded with trailing spaces to a common width,
#: so the row called "a" is really called "a          ".  Looking it up as
#: "a" then matches BOTH "a          " and "a:b        ", and R resolves
#: that ambiguity to ``NA`` -- a silent wrong answer rather than an error.
#: Trimming the names first makes every lookup exact and unambiguous.
_R_TAB_HELPER = """
tab <- function(x) {
  while (is.list(x) && !is.data.frame(x) && length(x) == 1L) x <- x[[1L]]
  m <- as.matrix(x)
  rownames(m) <- trimws(rownames(m))
  colnames(m) <- trimws(colnames(m))
  m
}
"""


def _r_program(code: str, **substitutions: str) -> dict[str, float]:
    """Run a reference program with the ``tab`` helper in scope."""
    return r_program(_R_TAB_HELPER + code, **substitutions)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _two_way_balanced() -> tuple[np.ndarray, list[str], list[str]]:
    """A balanced 2x2 factorial: 6 observations in each of the 4 cells."""
    rng = np.random.default_rng(101)
    values = np.concatenate(
        [
            rng.normal(10.0, 1.0, 6),
            rng.normal(12.0, 1.0, 6),
            rng.normal(11.0, 1.2, 6),
            rng.normal(13.5, 1.0, 6),
        ]
    )
    factor_a = ["A1"] * 12 + ["A2"] * 12
    factor_b = ["B1"] * 6 + ["B2"] * 6 + ["B1"] * 6 + ["B2"] * 6
    return values, factor_a, factor_b


def _two_way_unbalanced() -> tuple[np.ndarray, list[str], list[str]]:
    """An unbalanced 2x2 factorial: 5, 4, 3 and 6 observations in the cells.

    The cell sizes are unequal on purpose.  On a balanced design the Type I,
    Type II and Type III decompositions give the same sum of squares, so a
    comparison there cannot show *which* decomposition the application
    computes -- it can only show that the arithmetic agrees.  Here the three
    differ, and R's ``aov`` reports the sequential one.
    """
    rng = np.random.default_rng(202)
    values = np.concatenate(
        [
            rng.normal(10.0, 1.0, 5),
            rng.normal(12.0, 1.4, 4),
            rng.normal(11.0, 0.8, 3),
            rng.normal(13.5, 1.1, 6),
        ]
    )
    factor_a = ["A1"] * 9 + ["A2"] * 9
    factor_b = ["B1"] * 5 + ["B2"] * 4 + ["B1"] * 3 + ["B2"] * 6
    return values, factor_a, factor_b


def _repeated_measures() -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """8 subjects x 3 measurements, 4 subjects in each of 2 groups."""
    rng = np.random.default_rng(303)
    group_shift = np.repeat([0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5], 3)
    values = rng.normal(20.0, 1.0, 24) + group_shift
    subjects = [f"S{i}" for i in range(8) for _ in range(3)]
    groups = ["g1"] * 12 + ["g2"] * 12
    measurements = [1, 2, 3] * 8
    return values, subjects, groups, measurements


def _icc_table() -> np.ndarray:
    """A complete 8 targets x 3 raters measurement table."""
    rng = np.random.default_rng(404)
    target_effect = rng.normal(0.0, 3.0, 8)
    rater_effect = np.array([0.0, 0.4, -0.3])
    noise = rng.normal(0.0, 1.0, (8, 3))
    return 10.0 + target_effect[:, None] + rater_effect[None, :] + noise


# ---------------------------------------------------------------------------
# Two-way ANOVA
# ---------------------------------------------------------------------------
_TWO_WAY_PROGRAM = """
d <- data.frame(
  y = @@values@@,
  a = factor(@@factor_a@@),
  b = factor(@@factor_b@@)
)
m <- tab(summary(aov(y ~ a * b, data = d))[[1]])
out <- c(
  a_df = m["a", "Df"], a_ss = m["a", "Sum Sq"], a_ms = m["a", "Mean Sq"],
  a_f = m["a", "F value"], a_p = m["a", "Pr(>F)"],
  b_df = m["b", "Df"], b_ss = m["b", "Sum Sq"], b_ms = m["b", "Mean Sq"],
  b_f = m["b", "F value"], b_p = m["b", "Pr(>F)"],
  ab_df = m["a:b", "Df"], ab_ss = m["a:b", "Sum Sq"], ab_ms = m["a:b", "Mean Sq"],
  ab_f = m["a:b", "F value"], ab_p = m["a:b", "Pr(>F)"],
  res_df = m["Residuals", "Df"], res_ss = m["Residuals", "Sum Sq"],
  res_ms = m["Residuals", "Mean Sq"]
)
out
"""


class TestTwoWayAnova:
    """``stats::aov(y ~ a * b)`` -- the Type I factorial decomposition."""

    @pytest.mark.parametrize("dataset", ["balanced", "unbalanced"])
    def test_the_three_terms_match_r(self, dataset: str):
        """Every term's sum of squares, F and p-value matches R's.

        The unbalanced design is the discriminating one: on a balanced
        design the sum-of-squares types coincide, so agreeing there says
        less than it appears to.
        """
        from stats.design_tests import DesignTestAnalyzer

        if dataset == "balanced":
            values, factor_a, factor_b = _two_way_balanced()
        else:
            values, factor_a, factor_b = _two_way_unbalanced()

        mine = DesignTestAnalyzer().two_way_anova(values, factor_a, factor_b, n_permutations=0)
        theirs = _r_program(
            _TWO_WAY_PROGRAM,
            values=r_numbers(values),
            factor_a=r_labels(factor_a),
            factor_b=r_labels(factor_b),
        )

        for term, prefix in ((mine.term_a, "a"), (mine.term_b, "b"), (mine.term_interaction, "ab")):
            assert term.df == int(theirs[f"{prefix}_df"])
            assert_allclose(term.ss, theirs[f"{prefix}_ss"], rtol=RTOL, atol=ATOL)
            assert_allclose(term.ms, theirs[f"{prefix}_ms"], rtol=RTOL, atol=ATOL)
            assert_allclose(term.f_statistic, theirs[f"{prefix}_f"], rtol=RTOL, atol=ATOL)
            assert_allclose(term.p_value, theirs[f"{prefix}_p"], rtol=RTOL, atol=ATOL)

        assert mine.df_error == int(theirs["res_df"])
        assert_allclose(mine.ss_error, theirs["res_ss"], rtol=RTOL, atol=ATOL)

    def test_the_balanced_design_is_reported_as_balanced(self):
        """The balanced case is the one a reader is most likely to try.

        Kept as its own test because it is the case where a reader can
        reproduce the comparison by hand, and because it is the case where
        the application reports ``balanced=True``.
        """
        from stats.design_tests import DesignTestAnalyzer

        values, factor_a, factor_b = _two_way_balanced()
        mine = DesignTestAnalyzer().two_way_anova(values, factor_a, factor_b, n_permutations=0)
        assert mine.balanced

        theirs = _r_program(
            _TWO_WAY_PROGRAM,
            values=r_numbers(values),
            factor_a=r_labels(factor_a),
            factor_b=r_labels(factor_b),
        )
        assert_allclose(mine.term_interaction.f_statistic, theirs["ab_f"], rtol=RTOL, atol=ATOL)

    def test_the_terms_add_up_to_the_total(self):
        """The three terms and the residual must reconstruct the total.

        A decomposition that does not add up is the classic way an ANOVA
        table goes wrong, and it is invisible if only the F ratios are
        compared.
        """
        from stats.design_tests import DesignTestAnalyzer

        values, factor_a, factor_b = _two_way_unbalanced()
        mine = DesignTestAnalyzer().two_way_anova(values, factor_a, factor_b, n_permutations=0)
        total = mine.term_a.ss + mine.term_b.ss + mine.term_interaction.ss + mine.ss_error
        assert_allclose(mine.ss_total, total, rtol=RTOL, atol=ATOL)
        assert mine.df_total == mine.term_a.df + mine.term_b.df + mine.term_interaction.df + mine.df_error


# ---------------------------------------------------------------------------
# Repeated-measures ANOVA
# ---------------------------------------------------------------------------
_REPEATED_PROGRAM = """
d <- data.frame(
  y = @@values@@,
  grp = factor(@@groups@@),
  meas = factor(@@measurements@@),
  subj = factor(@@subjects@@)
)
s <- summary(aov(y ~ grp * meas + Error(subj/meas), data = d))
bt <- tab(s[[1]])
wt <- tab(s[[2]])
out <- c(
  grp_df = bt["grp", "Df"], grp_ss = bt["grp", "Sum Sq"], grp_ms = bt["grp", "Mean Sq"],
  grp_f = bt["grp", "F value"], grp_p = bt["grp", "Pr(>F)"],
  sub_df = bt["Residuals", "Df"], sub_ss = bt["Residuals", "Sum Sq"],
  sub_ms = bt["Residuals", "Mean Sq"],
  meas_df = wt["meas", "Df"], meas_ss = wt["meas", "Sum Sq"],
  meas_ms = wt["meas", "Mean Sq"], meas_f = wt["meas", "F value"],
  meas_p = wt["meas", "Pr(>F)"],
  gxm_df = wt["grp:meas", "Df"], gxm_ss = wt["grp:meas", "Sum Sq"],
  gxm_ms = wt["grp:meas", "Mean Sq"], gxm_f = wt["grp:meas", "F value"],
  gxm_p = wt["grp:meas", "Pr(>F)"],
  err_df = wt["Residuals", "Df"], err_ss = wt["Residuals", "Sum Sq"],
  err_ms = wt["Residuals", "Mean Sq"]
)
out
"""


class TestRepeatedMeasuresAnova:
    """``stats::aov(y ~ grp * meas + Error(subj/meas))`` -- the two strata."""

    def test_between_and_within_strata_match_r(self):
        from stats.design_tests import DesignTestAnalyzer

        values, subjects, groups, measurements = _repeated_measures()
        mine = DesignTestAnalyzer().repeated_measures_anova(values, subjects, groups, measurement=measurements)
        theirs = _r_program(
            _REPEATED_PROGRAM,
            values=r_numbers(values),
            groups=r_labels(groups),
            measurements=r_labels(measurements),
            subjects=r_labels(subjects),
        )

        # Between-subject stratum: the group effect is tested against the
        # subjects within group.
        assert mine.term_groups.df == int(theirs["grp_df"])
        assert_allclose(mine.term_groups.ss, theirs["grp_ss"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_groups.ms, theirs["grp_ms"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_groups.f_statistic, theirs["grp_f"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_groups.p_value, theirs["grp_p"], rtol=RTOL, atol=ATOL)

        assert mine.term_subjects.df == int(theirs["sub_df"])
        assert_allclose(mine.term_subjects.ss, theirs["sub_ss"], rtol=RTOL, atol=ATOL)

        # Within-subject stratum: the measurement effect and the interaction
        # are both tested against the subject-by-measurement error.
        assert mine.term_measurement.df == int(theirs["meas_df"])
        assert_allclose(mine.term_measurement.ss, theirs["meas_ss"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_measurement.f_statistic, theirs["meas_f"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_measurement.p_value, theirs["meas_p"], rtol=RTOL, atol=ATOL)

        assert mine.term_interaction.df == int(theirs["gxm_df"])
        assert_allclose(mine.term_interaction.ss, theirs["gxm_ss"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_interaction.f_statistic, theirs["gxm_f"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.term_interaction.p_value, theirs["gxm_p"], rtol=RTOL, atol=ATOL)

        assert mine.term_error.df == int(theirs["err_df"])
        assert_allclose(mine.term_error.ss, theirs["err_ss"], rtol=RTOL, atol=ATOL)

    def test_the_two_strata_add_up_to_the_total(self):
        """The within-subject and between-subject variation must be complete.

        A repeated-measures analysis that reports one stratum without the
        other is how a df total gets misread, so the decomposition is checked
        as a whole rather than term by term.
        """
        from stats.design_tests import DesignTestAnalyzer

        values, subjects, groups, measurements = _repeated_measures()
        mine = DesignTestAnalyzer().repeated_measures_anova(values, subjects, groups, measurement=measurements)
        total = (
            mine.term_groups.ss
            + mine.term_subjects.ss
            + mine.term_measurement.ss
            + mine.term_interaction.ss
            + mine.term_error.ss
        )
        assert_allclose(mine.ss_total, total, rtol=RTOL, atol=ATOL)
        assert mine.df_total == (
            mine.term_groups.df
            + mine.term_subjects.df
            + mine.term_measurement.df
            + mine.term_interaction.df
            + mine.term_error.df
        )


# ---------------------------------------------------------------------------
# Intraclass correlation
# ---------------------------------------------------------------------------
_ICC_PROGRAM = """
d <- data.frame(
  y = @@values@@,
  rater = factor(@@raters@@),
  target = factor(@@targets@@)
)
m <- tab(summary(aov(y ~ rater + target, data = d))[[1]])
ms_t <- m["target", "Mean Sq"]
ms_r <- m["rater", "Mean Sq"]
ms_e <- m["Residuals", "Mean Sq"]
out <- c(
  target_ms = ms_t, rater_ms = ms_r, error_ms = ms_e,
  error_df = m["Residuals", "Df"],
  target_f = m["target", "F value"],
  target_ss = m["target", "Sum Sq"],
  rater_ss = m["rater", "Sum Sq"],
  error_ss = m["Residuals", "Sum Sq"],
  icc21 = (ms_t - ms_e) / (ms_t + (@@n_raters@@ - 1) * ms_e + @@n_raters@@ * (ms_r - ms_e) / @@n_targets@@),
  icc31 = (ms_t - ms_e) / (ms_t + (@@n_raters@@ - 1) * ms_e)
)
out
"""


class TestIntraclassCorrelation:
    """The McGraw & Wong (1996) forms, computed from R's own ANOVA table."""

    @pytest.mark.parametrize("form", ["2,1", "3,1"])
    def test_icc_matches_the_reference_formulas(self, form: str):
        from stats.design_tests import DesignTestAnalyzer

        matrix = _icc_table()
        mine = DesignTestAnalyzer().intraclass_correlation(matrix, form=form)

        n_targets, n_raters = matrix.shape
        theirs = _r_program(
            _ICC_PROGRAM,
            values=r_numbers(matrix.ravel()),
            raters=r_labels([f"R{j}" for _ in range(n_targets) for j in range(n_raters)]),
            targets=r_labels([f"T{i}" for i in range(n_targets) for _ in range(n_raters)]),
            n_targets=n_targets,
            n_raters=n_raters,
        )

        assert_allclose(mine.ms_targets, theirs["target_ms"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.ms_raters, theirs["rater_ms"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.ms_error, theirs["error_ms"], rtol=RTOL, atol=ATOL)
        assert mine.df_error == int(theirs["error_df"])
        assert_allclose(mine.f_statistic, theirs["target_f"], rtol=RTOL, atol=ATOL)

        key = "icc21" if form == "2,1" else "icc31"
        assert_allclose(mine.icc, theirs[key], rtol=RTOL, atol=ATOL)

    def test_the_two_forms_differ_when_the_raters_disagree(self):
        """ICC(2,1) and ICC(3,1) are not the same statistic.

        They coincide only when the rater effect is negligible.  The table
        below has a rater offset on purpose, and a comparison that collapsed
        the two forms would hide a rater effect that changes the answer.
        """
        from stats.design_tests import DesignTestAnalyzer

        matrix = _icc_table()
        agreement = DesignTestAnalyzer().intraclass_correlation(matrix, form="2,1")
        consistency = DesignTestAnalyzer().intraclass_correlation(matrix, form="3,1")
        assert not np.isclose(agreement.icc, consistency.icc, rtol=1e-6)

    def test_the_icc_is_inside_the_range_it_can_take(self):
        """A coefficient outside [-1, 1] is not an ICC.

        The point estimate is deliberately left unclipped when a variance
        component came out negative, so this is a property of a *sane*
        estimate rather than a tautology.
        """
        from stats.design_tests import DesignTestAnalyzer

        matrix = _icc_table()
        for form in ("2,1", "3,1"):
            result = DesignTestAnalyzer().intraclass_correlation(matrix, form=form)
            assert -1.0 <= result.icc <= 1.0
            assert result.ci_lower <= result.icc <= result.ci_upper
