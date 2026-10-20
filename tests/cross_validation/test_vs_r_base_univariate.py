# =============================================================================
# FILE: tests/cross_validation/test_vs_r_base_univariate.py
# =============================================================================
"""
Cross-validation of the univariate tests against base R's ``stats``.

These are the tests a palaeontologist runs most often, and the ones where
a wrong number changes a conclusion rather than a digit: a Wilcoxon test
that lands on the wrong side of 0.05, a Kruskal-Wallis that does, a
chi-square that says "independent" when the table says otherwise.

The comparison is LIVE: R computes its answer during the test run, so
this is a check against the reference implementation rather than a
restated formula.  Every value compared here was also checked against a
local R 4.5.2 before it was committed.

What R is being asked, and why
------------------------------
``wilcox.test`` (two-sample) reports its statistic as "the sum of the
ranks of the first sample MINUS n1(n1+1)/2", which is exactly scipy's
U.  The two are therefore directly comparable; no conversion, and a
mismatch means a real difference.

``oneway.test`` is the exception, and it is worth spelling out.  R's
default is ADAPTIVE: it tests for equal variances first and then picks
the classical or the Welch test accordingly.  PaleoAST computes the
CLASSICAL (pooled-variance) ANOVA -- which is what
``scipy.stats.f_oneway`` does, and what the name "one-way ANOVA" means
to most people.  On data whose group variances differ, R's default
therefore reports a DIFFERENT p-value: on the data below, R's default
gives p = 0.2467 and PaleoAST gives p = 0.1764, while R's
``var.equal = TRUE`` agrees with PaleoAST to twelve digits.

That is not a bug -- it is an unstated choice, and an unstated choice is
what makes a reader's own R session disagree with the application.  So
the comparison is made against ``var.equal = TRUE``, which is the test
PaleoAST actually performs, and the difference from R's default is
recorded here where a reader will meet it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import as_float, r, r_data_frame, r_matrix, r_vector, require

pytestmark = pytest.mark.cross_validation

R_STATS = require("stats")

#: Tolerances.  These are the same statistics, so they agree to the last
#: bit of a double in most cases; 1e-9 is loose enough to survive a
#: difference in the last ulp and tight enough that a wrong statistic
#: cannot hide behind it.
RTOL = 1e-9
ATOL = 1e-12


def _groups_and_values() -> tuple[np.ndarray, list[int]]:
    """A deterministic three-group sample with unequal variances.

    The unequal variances are on purpose: they are what make R's default
    ``oneway.test`` choose the Welch test, which is the difference this
    file documents.
    """
    rng = np.random.default_rng(11)
    groups = [0] * 8 + [1] * 8 + [2] * 8
    values = np.concatenate(
        [
            rng.normal(loc=10.0, scale=2.0, size=8),
            rng.normal(loc=12.5, scale=2.4, size=8),
            rng.normal(loc=9.2, scale=1.8, size=8),
        ]
    )
    rng.shuffle(values)
    return values, groups


def _paired() -> tuple[np.ndarray, np.ndarray]:
    """Two paired samples, as R's ``wilcox.test(x, y, paired = TRUE)`` sees them."""
    rng = np.random.default_rng(23)
    before = rng.normal(loc=50.0, scale=5.0, size=16)
    after = before + rng.normal(loc=3.0, scale=2.0, size=16)
    return before, after


def _contingency() -> np.ndarray:
    return np.array([[30, 12, 8], [14, 22, 11], [9, 7, 19]])


def _factor(groups: list[int]):
    """An R factor from integer group labels."""
    return r("factor")(r_vector([float(g) for g in groups]))


class TestKruskalWallis:
    """``stats::kruskal.test`` -- the non-parametric alternative to ANOVA."""

    def test_statistic_and_p_value_match(self):
        from stats.univariate import UnivariateAnalyzer

        values, groups = _groups_and_values()
        mine = UnivariateAnalyzer().kruskal_wallis(values, groups)

        result = R_STATS["kruskal.test"](r_vector(values), _factor(groups))
        assert_allclose(mine.statistic, as_float(result.rx2("statistic")), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, as_float(result.rx2("p.value")), rtol=RTOL, atol=ATOL)


class TestShapiroWilk:
    """``stats::shapiro.test`` -- what the normality test reports."""

    def test_statistic_and_p_value_match(self):
        from stats.univariate import UnivariateAnalyzer

        values, _groups = _groups_and_values()
        mine = UnivariateAnalyzer().normality_test(values)

        result = R_STATS["shapiro.test"](r_vector(values))
        assert_allclose(mine.shapiro_stat, as_float(result.rx2("statistic")), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.shapiro_p, as_float(result.rx2("p.value")), rtol=RTOL, atol=ATOL)


class TestMannWhitney:
    """``stats::wilcox.test`` -- R's W is scipy's U (see the module docstring)."""

    def test_statistic_and_p_value_match(self):
        from stats.univariate import UnivariateAnalyzer

        values, groups = _groups_and_values()
        # Two groups: the first two merged against the third.
        labels = [0 if g < 2 else 1 for g in groups]
        mine = UnivariateAnalyzer().mann_whitney(values, labels)

        first = r_vector(values[[i for i, g in enumerate(groups) if g < 2]])
        second = r_vector(values[[i for i, g in enumerate(groups) if g == 2]])
        result = R_STATS["wilcox.test"](first, second)
        assert_allclose(mine.statistic, as_float(result.rx2("statistic")), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, as_float(result.rx2("p.value")), rtol=RTOL, atol=ATOL)


class TestPairedRankTest:
    """``stats::wilcox.test(paired = TRUE)`` -- the signed-rank test.

    PaleoAST takes ONE column of 2n values and splits it with group
    labels: the i-th member of group 0 is paired with the i-th member of
    group 1.  Laying the data out that way makes it R's two vectors.
    """

    def test_statistic_and_p_value_match(self):
        from stats.univariate import UnivariateAnalyzer

        before, after = _paired()
        stacked = np.concatenate([before, after])
        labels = [0] * len(before) + [1] * len(after)
        mine = UnivariateAnalyzer().paired_rank_test(stacked, column=0, groups=labels, method="wilcoxon")

        result = R_STATS["wilcox.test"](r_vector(before), r_vector(after), paired=True)
        assert_allclose(mine.statistic, as_float(result.rx2("statistic")), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, as_float(result.rx2("p.value")), rtol=RTOL, atol=ATOL)


class TestContingencyChiSquare:
    """``stats::chisq.test`` -- independence in a contingency table."""

    def test_statistic_p_value_and_dof_match(self):
        from stats.design_tests import DesignTestAnalyzer

        table = _contingency()
        mine = DesignTestAnalyzer().contingency_chi_square(table)

        result = R_STATS["chisq.test"](r_matrix(table))
        assert_allclose(mine.chi_square, as_float(result.rx2("statistic")), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, as_float(result.rx2("p.value")), rtol=RTOL, atol=ATOL)
        assert mine.dof == int(as_float(result.rx2("parameter")))


class TestOneWayAnova:
    """``stats::oneway.test`` -- and the one place the two do not agree.

    PaleoAST computes the CLASSICAL (pooled-variance) ANOVA.  R's
    default is adaptive and picks the Welch test when the group variances
    differ, so on this data R's default gives a different p-value.

    The comparison below is against ``var.equal = TRUE``, which is the
    test PaleoAST performs.  The second test pins the difference so that
    it cannot be mistaken for a regression: if a future change makes
    PaleoAST's ANOVA equal to R's default, that change is a DECISION
    about which test to run, and it has to be made deliberately.
    """

    def test_matches_r_with_equal_variances(self):
        from stats.univariate import UnivariateAnalyzer

        values, groups = _groups_and_values()
        mine = UnivariateAnalyzer().one_way_anova(values, groups, tukey=False)

        frame = r_data_frame({"v": values, "g": groups}, factors=("g",))
        formula = r("as.formula")("v ~ g")
        # The R argument is ``var.equal``, with the dot.  Passed as a dict so
        # the dot survives -- the same reason this suite spells prcomp's
        # ``scale.`` that way.
        result = R_STATS["oneway.test"](formula, data=frame, **{"var.equal": True})
        assert_allclose(mine.f_statistic, as_float(result.rx2("statistic")), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, as_float(result.rx2("p.value")), rtol=RTOL, atol=ATOL)

    def test_the_difference_from_r_s_default_is_the_one_we_mean(self):
        """R's default would give a different answer, and that is on record.

        If this test ever fails because the two now AGREE, the ANOVA
        changed which test it performs -- which changes published
        numbers, so it has to be a deliberate, documented change rather
        than a side effect of a dependency upgrade.
        """
        from stats.univariate import UnivariateAnalyzer

        values, groups = _groups_and_values()
        mine = UnivariateAnalyzer().one_way_anova(values, groups, tukey=False)

        frame = r_data_frame({"v": values, "g": groups}, factors=("g",))
        formula = r("as.formula")("v ~ g")
        result = R_STATS["oneway.test"](formula, data=frame)
        assert not np.isclose(mine.p_value, as_float(result.rx2("p.value")), rtol=1e-6), (
            "PaleoAST's one-way ANOVA now agrees with R's adaptive default; "
            "if the test is meant to change, say so in the method documentation"
        )
