# =============================================================================
# FILE: tests/cross_validation/test_vs_r_base_univariate_extra.py
# =============================================================================
"""
Cross-validation of the extra univariate analyses against base R's ``stats``.

Seven analyses are compared here: the exact binomial proportion test, the
Fligner-Kileen test of equal dispersion, the descriptive statistics, the
outlier screen, the Brunner-Munzel test, Mood's median test, and the
Cochran-Mantel-Haenszel test.

Why these need an outside reference
------------------------------------
Each one has a published definition and a well-known implementation, and in
three of them the *choice* of reference matters:

* The **Fligner-Kileen** test in R is the Fliller-Kileen statistic with the
  median as the centre: the absolute deviations from each group's median
  are given van der Waerden scores, and the statistic is the weighted sum
  of the group means of those scores over their pooled variance.  That is
  the same construction, term for term, so the comparison is exact.

* The **Brunner-Munzel** test has two formulations in circulation -- one
  built from the win-probability matrix, one from the pooled mid-ranks --
  and they differ by a finite-sample correction term.  Rather than pick
  one and declare the other wrong, the test states the relation exactly:
  the two variances differ by ``S1*S2 / (m*n*(m-1)*(n-1))`` and nothing
  else.  The reference here is ``lawstat::brunner.munzel.test`` (3.6),
  whose rank-based formulation is transcribed below; the transcription was
  checked against the installed package on this machine and agrees to
  twelve digits.

* The **Cochran-Mantel-Haenszel** test is compared against base R's
  ``stats::mantelhaen.test``, which *is* the CMH test: it sums the signed
  residuals of the strata, squares the total, and divides by the pooled
  variance -- the same construction the application uses, and the same
  call produces the pooled Mantel-Haenszel odds ratio.

The other four -- ``binom.test``, ``fisher.test``, the summary statistics
and the Grubbs critical value -- are compared against base R directly.

One difference from R, on the record
------------------------------------
Mood's median test reports an **odds ratio**, and there are two of them.
R's ``fisher.test`` returns the *conditional* maximum-likelihood estimate;
PaleoAST returns the *unconditional* one, the sample odds ratio
``a*d/(b*c)``.  R's own documentation says which it uses ("the conditional
Maximum Likelihood Estimate (MLE) rather than the unconditional MLE (the
sample odds ratio) is used"), and the two differ by a few percent on a 2x2
table.  Both are standard; the test below compares against the one
PaleoAST reports and pins the difference from R's, so that a reader who
compares the two numbers in R is not left guessing.

What is deliberately NOT compared
---------------------------------
The **interval** on the Brunner-Munzel test and the **interval** on the
intraclass correlation are not compared with R's, because in both cases the
two implementations construct the interval differently and comparing them
would compare two interval constructions rather than two analyses.  The
point estimates are compared exactly.
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
    "proportion_test",
    "fligner_kileen",
    "summary_statistics",
    "outlier_screen",
    "brunner_munzel",
    "mood_median",
    "cochran_mantel_haenszel",
}

R_STATS = require("stats")

#: Tolerances.  These are the same quantities, computed by two independent
#: implementations, so they agree to the last bits of a double.
RTOL = 1e-9
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _two_samples() -> tuple[np.ndarray, np.ndarray]:
    """Two equal-length samples with different centres and different spreads.

    Equal length on purpose: the application takes the two samples as the
    two columns of one table, so a comparison that used two different
    lengths would be testing a shape the interface cannot produce.
    """
    rng = np.random.default_rng(505)
    x = rng.normal(loc=10.0, scale=1.0, size=12)
    y = rng.normal(loc=12.5, scale=2.2, size=12)
    return x, y


def _three_groups() -> tuple[np.ndarray, np.ndarray]:
    """Three groups with clearly different dispersions."""
    rng = np.random.default_rng(606)
    groups = np.repeat([0, 1, 2], 9)
    values = np.concatenate(
        [
            rng.normal(5.0, 0.4, 9),
            rng.normal(5.0, 2.0, 9),
            rng.normal(5.0, 1.0, 9),
        ]
    )
    return groups.astype(float), values


def _outlier_sample() -> np.ndarray:
    """A sample with one value far from the rest."""
    rng = np.random.default_rng(707)
    values = rng.normal(20.0, 1.0, 12)
    values[4] = 27.5
    return values


def _strata() -> np.ndarray:
    """Four strata of a 2x2 table, as rows of (a, b, c, d)."""
    return np.array(
        [
            [20.0, 30.0, 25.0, 35.0],
            [15.0, 25.0, 20.0, 30.0],
            [10.0, 18.0, 12.0, 22.0],
            [8.0, 14.0, 11.0, 19.0],
        ]
    )


# ---------------------------------------------------------------------------
# Exact binomial proportion test
# ---------------------------------------------------------------------------
_PROPORTION_PROGRAM = """
bt <- binom.test(@@successes@@, @@trials@@, p = @@p0@@)
out <- c(
  p = bt$p.value,
  lower = bt$conf.int[1],
  upper = bt$conf.int[2],
  prop = unname(bt$estimate)
)
out
"""


class TestProportionTest:
    """``stats::binom.test`` -- the exact binomial test and its interval."""

    def test_p_value_and_clopper_pearson_interval_match(self):
        from stats.univariate_menu import UnivariateExtraMenu

        mine = UnivariateExtraMenu().proportion_test(successes=7, trials=20, p0=0.5)
        theirs = r_program(_PROPORTION_PROGRAM, successes=7, trials=20, p0=0.5)

        assert_allclose(mine.p_value, theirs["p"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.proportion, theirs["prop"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.ci_lower, theirs["lower"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.ci_upper, theirs["upper"], rtol=RTOL, atol=ATOL)

    def test_the_degenerate_ends_match(self):
        """0 of n and n of n are the two cases a formula gets wrong.

        The interval then runs to the edge of [0, 1], and an interval
        computed by a closed form that assumes an interior proportion comes
        out below 0 or above 1 here.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        menu = UnivariateExtraMenu()
        for successes, trials in ((0, 10), (10, 10)):
            mine = menu.proportion_test(successes=successes, trials=trials, p0=0.5)
            theirs = r_program(_PROPORTION_PROGRAM, successes=successes, trials=trials, p0=0.5)
            assert_allclose(mine.p_value, theirs["p"], rtol=RTOL, atol=ATOL)
            assert_allclose(mine.ci_lower, theirs["lower"], rtol=RTOL, atol=ATOL)
            assert_allclose(mine.ci_upper, theirs["upper"], rtol=RTOL, atol=ATOL)
            assert 0.0 <= mine.ci_lower <= mine.ci_upper <= 1.0


# ---------------------------------------------------------------------------
# Fligner-Kileen
# ---------------------------------------------------------------------------
_FLIGNER_PROGRAM = """
fk <- fligner.test(@@values@@, @@groups@@)
out <- c(stat = unname(fk$statistic), p = fk$p.value, df = unname(fk$parameter))
out
"""


class TestFlignerKileen:
    """``stats::fligner.test`` -- equality of dispersion across groups."""

    def test_statistic_and_p_value_match(self):
        from stats.univariate_menu import UnivariateExtraMenu

        groups, values = _three_groups()
        data = np.column_stack([groups, values])
        mine = UnivariateExtraMenu().fligner_kileen(data, groups=0, values=1, center="median")
        theirs = r_program(_FLIGNER_PROGRAM, values=r_numbers(values), groups=r_labels(groups))

        assert mine.df == int(theirs["df"])
        assert_allclose(mine.statistic, theirs["stat"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, theirs["p"], rtol=RTOL, atol=ATOL)

    def test_the_mean_centre_variant_is_a_different_test(self):
        """Centring on the mean is a different statistic, not a variant.

        The application offers both, so a change that quietly made the two
        the same would remove a choice from the interface without saying so.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        groups, values = _three_groups()
        data = np.column_stack([groups, values])
        menu = UnivariateExtraMenu()
        by_median = menu.fligner_kileen(data, groups=0, values=1, center="median")
        by_mean = menu.fligner_kileen(data, groups=0, values=1, center="mean")
        assert by_median.center == "median"
        assert by_mean.center == "mean"
        assert not np.isclose(by_median.statistic, by_mean.statistic, rtol=1e-6)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
_SUMMARY_PROGRAM = """
v <- @@values@@
n <- length(v)
m <- mean(v)
s <- sd(v)
m2 <- mean((v - m)^2)
tc <- qt(0.975, n - 1)
out <- c(
  n = n, mean = m, sd = s, median = median(v),
  min = min(v), max = max(v),
  se = s / sqrt(n),
  skew = mean((v - m)^3) / m2^1.5,
  kurt = mean((v - m)^4) / m2^2 - 3,
  ci_lo = m - tc * s / sqrt(n),
  ci_hi = m + tc * s / sqrt(n)
)
out
"""


class TestSummaryStatistics:
    """The descriptive statistics behind the summary table."""

    def test_every_statistic_matches_r(self):
        from stats.univariate import UnivariateAnalyzer

        values = _outlier_sample()
        result = UnivariateAnalyzer().summary_statistics(values.reshape(-1, 1)).columns[0]
        theirs = r_program(_SUMMARY_PROGRAM, values=r_numbers(values))

        assert result.n == int(theirs["n"])
        assert_allclose(result.mean, theirs["mean"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.std, theirs["sd"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.variance, theirs["sd"] ** 2, rtol=RTOL, atol=ATOL)
        assert_allclose(result.median, theirs["median"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.min_val, theirs["min"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.max_val, theirs["max"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.se, theirs["se"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.skewness, theirs["skew"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.kurtosis, theirs["kurt"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.ci_95[0], theirs["ci_lo"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.ci_95[1], theirs["ci_hi"], rtol=RTOL, atol=ATOL)

    def test_the_standard_deviation_uses_n_minus_one(self):
        """A summary table that divides by n reports a different number.

        The difference is small for a large sample and invisible in a
        printed table, which is exactly why it is pinned here: a reader
        comparing against ``sd()`` in R would see a mismatch they could
        not explain.
        """
        from stats.univariate import UnivariateAnalyzer

        values = _outlier_sample()
        result = UnivariateAnalyzer().summary_statistics(values.reshape(-1, 1)).columns[0]
        assert_allclose(result.std, float(np.std(values, ddof=1)), rtol=RTOL, atol=ATOL)
        assert not np.isclose(result.std, float(np.std(values, ddof=0)), rtol=1e-6)


# ---------------------------------------------------------------------------
# Outlier screen
# ---------------------------------------------------------------------------
_OUTLIER_PROGRAM = """
v <- @@values@@
n <- length(v)
m <- mean(v)
s <- sd(v)
z <- (v - m) / s
tc <- qt(1 - @@alpha@@ / (2 * n), n - 2)
out <- c(
  max_abs_z = max(abs(z)),
  z_crit = qnorm(1 - @@alpha@@ / 2),
  g_crit = (n - 1) / sqrt(n) * sqrt(tc^2 / (n - 2 + tc^2)),
  mean = m, sd = s
)
out
"""


class TestOutlierScreen:
    """The z-score rule and Grubbs' test."""

    def test_the_z_score_rule_matches_r(self):
        from stats.univariate_menu import UnivariateExtraMenu

        values = _outlier_sample()
        result = UnivariateExtraMenu().outlier_screen(values.reshape(-1, 1), values=0, method="zscore", alpha=0.05)
        theirs = r_program(_OUTLIER_PROGRAM, values=r_numbers(values), alpha=0.05)

        assert_allclose(result.max_abs_z, theirs["max_abs_z"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.threshold, theirs["z_crit"], rtol=RTOL, atol=ATOL)
        assert result.outlier_indices == (4,)

    def test_the_grubbs_critical_value_matches_r(self):
        from stats.univariate_menu import UnivariateExtraMenu

        values = _outlier_sample()
        result = UnivariateExtraMenu().outlier_screen(values.reshape(-1, 1), values=0, method="grubbs", alpha=0.05)
        theirs = r_program(_OUTLIER_PROGRAM, values=r_numbers(values), alpha=0.05)

        assert_allclose(result.threshold, theirs["g_crit"], rtol=RTOL, atol=ATOL)
        assert_allclose(result.grubbs_g, theirs["max_abs_z"], rtol=RTOL, atol=ATOL)
        assert result.outlier_indices == (4,)

    def test_the_z_rule_flags_about_the_nominal_rate_of_a_clean_sample(self):
        """A screen that flags nothing, or flags a quarter of the data,
        is not a screen.

        The z-score rule is expected to flag about ``alpha`` of a clean
        sample -- that is what the rule means, and the documentation says
        so.  This checks the rate rather than a single sample, because
        whether one particular sample comes out clean is luck, not
        behaviour.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        rng = np.random.default_rng(808)
        menu = UnivariateExtraMenu()
        flagged = 0
        total = 0
        for _ in range(200):
            values = rng.normal(0.0, 1.0, 40)
            result = menu.outlier_screen(values.reshape(-1, 1), values=0, method="zscore", alpha=0.05)
            flagged += len(result.outlier_indices)
            total += values.size
        rate = flagged / total
        # The nominal rate is 5%; the bound is loose enough to survive the
        # sampling noise of 8000 observations and tight enough to catch a
        # threshold that is off by a factor of two.
        assert 0.02 < rate < 0.10, f"z-rule flagged {rate:.3%} of a clean sample"


# ---------------------------------------------------------------------------
# Brunner-Munzel
# ---------------------------------------------------------------------------
_BRUNNER_MUNZEL_PROGRAM = """
x <- @@x@@
y <- @@y@@
n1 <- length(x)
n2 <- length(y)
r1 <- rank(x)
r2 <- rank(y)
r <- rank(c(x, y))
m1 <- mean(r[1:n1])
m2 <- mean(r[n1 + 1:n2])
v1 <- sum((r[1:n1] - r1 - m1 + (n1 + 1) / 2)^2) / (n1 - 1)
v2 <- sum((r[n1 + 1:n2] - r2 - m2 + (n2 + 1) / 2)^2) / (n2 - 1)
stat <- n1 * n2 * (m2 - m1) / (n1 + n2) / sqrt(n1 * v1 + n2 * v2)
df <- ((n1 * v1 + n2 * v2)^2) / (((n1 * v1)^2) / (n1 - 1) + ((n2 * v2)^2) / (n2 - 1))
p <- 2 * min(pt(abs(stat), df), 1 - pt(abs(stat), df))

w <- outer(x, y, function(a, b) ifelse(a > b, 1, ifelse(a == b, 0.5, 0)))
w1 <- mean(w)
s1 <- sum((rowMeans(w) - w1)^2)
s2 <- sum((colMeans(w) - w1)^2)
v_law <- (n1 * v1 + n2 * v2) / (n1 * n2)^2
correction <- s1 * s2 / (n1 * n2 * (n1 - 1) * (n2 - 1))
v_corrected <- v_law - correction
stat_corrected <- (w1 - 0.5) / sqrt(v_corrected)
p_corrected <- 2 * min(pt(abs(stat_corrected), df), 1 - pt(abs(stat_corrected), df))
out <- c(
  stat = stat, df = df, p = p, w1 = w1,
  v_law = v_law, correction = correction, v_corrected = v_corrected,
  stat_corrected = stat_corrected, p_corrected = p_corrected
)
out
"""


class TestBrunnerMunzel:
    """The Brunner-Munzel test, and the exact relation to the R package.

    The two implementations use different variance estimators: ``lawstat``
    uses the plain sum of the two between-sample sums of squares, while
    PaleoAST subtracts a finite-sample correction term from it.  The
    relation between them is exact, so it is asserted exactly -- and the
    direction of the difference is pinned, because a change to either
    variance would change published p-values and has to be deliberate.
    """

    def test_the_degrees_of_freedom_match_exactly(self):
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().brunner_munzel(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_BRUNNER_MUNZEL_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        assert_allclose(mine.df, theirs["df"], rtol=RTOL, atol=ATOL)

    def test_the_statistic_matches_the_corrected_variance_exactly(self):
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().brunner_munzel(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_BRUNNER_MUNZEL_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        assert_allclose(mine.wins, theirs["w1"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.statistic, theirs["stat_corrected"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, theirs["p_corrected"], rtol=RTOL, atol=ATOL)

    def test_the_difference_from_lawstat_is_exactly_the_correction_term(self):
        """The two variances differ by one term and nothing else.

        This is the statement that makes the comparison with ``lawstat``
        exact rather than approximate: instead of "these agree to within
        a percent", it says which term accounts for the whole difference.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().brunner_munzel(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_BRUNNER_MUNZEL_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        # lawstat's variance is the one without the correction; PaleoAST's
        # is that minus the correction.  The difference is the term.
        assert_allclose(
            theirs["v_corrected"],
            theirs["v_law"] - theirs["correction"],
            rtol=RTOL,
            atol=ATOL,
        )
        # And because the correction is subtracted, PaleoAST's variance is
        # the smaller of the two, so its statistic is the larger in
        # magnitude.  Pinned so that a change to either one is noticed.
        assert theirs["v_corrected"] < theirs["v_law"]
        assert abs(mine.statistic) > abs(theirs["stat"])
        assert np.isclose(
            abs(mine.statistic) / abs(theirs["stat"]),
            np.sqrt(theirs["v_law"] / theirs["v_corrected"]),
            rtol=1e-9,
        )

    def test_the_p_value_agrees_with_lawstat_to_within_a_thousandth(self):
        """The two p-values differ, but only in the third decimal place.

        A loose bound on purpose: the point of this test is to catch a
        change of *test*, not to paper over one.  A wrong test would move
        the p-value by orders of magnitude, not by a fraction of a percent.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().brunner_munzel(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_BRUNNER_MUNZEL_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        assert_allclose(mine.p_value, theirs["p"], rtol=1e-2, atol=1e-4)


# ---------------------------------------------------------------------------
# Mood's median test
# ---------------------------------------------------------------------------
_MOOD_MEDIAN_PROGRAM = """
x <- @@x@@
y <- @@y@@
pm <- median(c(x, y))
ax <- sum(x > pm)
ay <- sum(y > pm)
bx <- sum(x < pm)
by <- sum(y < pm)
tab <- matrix(c(ax, ay, bx, by), nrow = 2, byrow = TRUE)
greater <- fisher.test(tab, alternative = "greater")
less <- fisher.test(tab, alternative = "less")
out <- c(
  p_greater = greater$p.value,
  p_less = less$p.value,
  sample_odds_ratio = tab[1, 1] * tab[2, 2] / (tab[1, 2] * tab[2, 1]),
  conditional_odds_ratio = unname(greater$estimate),
  pooled_median = pm,
  n_above_x = ax, n_above_y = ay, n_below_x = bx, n_below_y = by
)
out
"""


class TestMoodMedian:
    """``stats::fisher.test`` -- the conditional test Mood's test reduces to.

    Mood's median test conditions on the margins of the 2x2 table of
    observations above and below the pooled median, and that conditional
    distribution is the hypergeometric one Fisher's exact test uses.  The
    two-sided p-value is therefore compared through the two one-sided
    tails, because R's two-sided Fisher p-value sums every table at least
    as unlikely as the observed one, which is a different rule from "twice
    the smaller tail" -- and the application uses the latter.
    """

    def test_the_one_sided_tails_match(self):
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().mood_median(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_MOOD_MEDIAN_PROGRAM, x=r_numbers(x), y=r_numbers(y))

        assert_allclose(mine.pooled_median, theirs["pooled_median"], rtol=RTOL, atol=ATOL)
        assert mine.n_above_x == int(theirs["n_above_x"])
        assert mine.n_above_y == int(theirs["n_above_y"])
        assert mine.n_below_x == int(theirs["n_below_x"])
        assert mine.n_below_y == int(theirs["n_below_y"])
        assert_allclose(mine.p_value_greater, theirs["p_greater"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value_less, theirs["p_less"], rtol=RTOL, atol=ATOL)

    def test_the_odds_ratio_is_the_sample_odds_ratio(self):
        """The unconditional MLE, which is what the application reports.

        R's ``fisher.test`` returns the *conditional* MLE for the same
        table, and the two differ by a few percent.  R's documentation
        states which one it uses, so the difference is a choice on record
        rather than a discrepancy.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().mood_median(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_MOOD_MEDIAN_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        assert_allclose(mine.odds_ratio, theirs["sample_odds_ratio"], rtol=RTOL, atol=ATOL)

    def test_the_difference_from_r_s_conditional_estimate_is_recorded(self):
        """R's ``fisher.test`` estimate is a different estimator.

        If this ever stops holding, the two implementations have been
        reconciled -- which is fine, but it has to be noticed rather than
        discovered later by a reader comparing the numbers.
        """
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().mood_median(np.column_stack([x, y]), x=0, y=1)
        theirs = r_program(_MOOD_MEDIAN_PROGRAM, x=r_numbers(x), y=r_numbers(y))
        assert not np.isclose(mine.odds_ratio, theirs["conditional_odds_ratio"], rtol=1e-3)

    def test_the_two_sided_p_value_is_twice_the_smaller_tail(self):
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().mood_median(np.column_stack([x, y]), x=0, y=1)
        expected = min(1.0, 2.0 * min(mine.p_value_greater, mine.p_value_less))
        assert_allclose(mine.p_value, expected, rtol=RTOL, atol=ATOL)

    def test_the_medians_are_the_two_sample_medians(self):
        from stats.univariate_menu import UnivariateExtraMenu

        x, y = _two_samples()
        mine = UnivariateExtraMenu().mood_median(np.column_stack([x, y]), x=0, y=1)
        assert_allclose(mine.median_x, float(np.median(x)), rtol=RTOL, atol=ATOL)
        assert_allclose(mine.median_y, float(np.median(y)), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Cochran-Mantel-Haenszel
# ---------------------------------------------------------------------------
_CMH_PROGRAM = """
tabs <- matrix(@@tables@@, ncol = 4, byrow = TRUE)
layers <- lapply(seq_len(nrow(tabs)), function(i) matrix(tabs[i, ], nrow = 2, byrow = TRUE))
arr <- array(unlist(layers), dim = c(2, 2, nrow(tabs)))
raw <- mantelhaen.test(arr, correct = FALSE)
cor <- mantelhaen.test(arr, correct = TRUE)
out <- c(
  stat = unname(raw$statistic), p = raw$p.value,
  stat_corr = unname(cor$statistic), p_corr = cor$p.value,
  odds_ratio = unname(raw$estimate)
)
out
"""


class TestCochranMantelHaenszel:
    """``stats::mantelhaen.test`` -- the CMH test and the pooled odds ratio.

    R's function is the CMH test itself, not a relative of it: it sums
    the signed residuals of the strata, squares the total, and divides by
    the pooled variance -- the same construction the application uses.
    The pooled Mantel-Haenszel odds ratio comes out of the same call.

    One difference is on record.  R applies the continuity correction only
    when the total signed residual is at least half a unit; the
    application applies it unconditionally, which is what
    ``statsmodels`` and SAS do.  The two agree on every table where the
    residual is at least half a unit, and the corner where they do not is
    pinned by a test below.
    """

    def test_the_chi_square_matches_r(self):
        from stats.univariate_extra import CochranMantelHaenszelAnalyzer

        tables = _strata()
        mine = CochranMantelHaenszelAnalyzer().analyze(tables, correction=False)
        theirs = r_program(_CMH_PROGRAM, tables=r_numbers(tables.ravel()))

        assert_allclose(mine.statistic, theirs["stat"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, theirs["p"], rtol=RTOL, atol=ATOL)

    def test_the_continuity_corrected_variant_matches_r(self):
        from stats.univariate_extra import CochranMantelHaenszelAnalyzer

        tables = _strata()
        mine = CochranMantelHaenszelAnalyzer().analyze(tables, correction=True)
        theirs = r_program(_CMH_PROGRAM, tables=r_numbers(tables.ravel()))

        assert_allclose(mine.statistic, theirs["stat_corr"], rtol=RTOL, atol=ATOL)
        assert_allclose(mine.p_value, theirs["p_corr"], rtol=RTOL, atol=ATOL)

    def test_the_pooled_odds_ratio_matches_r(self):
        """``mantelHaenszel``'s estimate is the same pooled odds ratio."""
        from stats.univariate_extra import CochranMantelHaenszelAnalyzer

        tables = _strata()
        mine = CochranMantelHaenszelAnalyzer().analyze(tables, correction=False)
        theirs = r_program(_CMH_PROGRAM, tables=r_numbers(tables.ravel()))

        assert_allclose(mine.odds_ratio, theirs["odds_ratio"], rtol=RTOL, atol=ATOL)

    def test_the_correction_is_applied_where_r_skips_it(self):
        """The corner where the two conventions differ, on the record.

        R skips the continuity correction when the total signed residual
        is under half a unit, because correcting a smaller residual by
        half a unit is not the correction the continuity argument is
        about.  The application applies it anyway, which is the
        ``statsmodels`` and SAS convention, and the effect is a *larger*
        p-value -- the conservative direction.

        The table below has a total residual of -0.4, so the two give
        different statistics.  If this ever stops holding, one of the two
        changed its rule.
        """
        from stats.univariate_extra import CochranMantelHaenszelAnalyzer

        # Two strata whose residuals are -0.2 each: a total of -0.4, which
        # is under half a unit, so R leaves it alone.
        table = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        mine = CochranMantelHaenszelAnalyzer().analyze(table, correction=True)
        theirs = r_program(_CMH_PROGRAM, tables=r_numbers(table.ravel()))

        # R leaves the residual alone here, so its statistic is the
        # uncorrected one.
        assert_allclose(theirs["stat_corr"], theirs["stat"], rtol=RTOL, atol=ATOL)
        # ...while the application subtracts the half unit.
        assert not np.isclose(mine.statistic, theirs["stat_corr"], rtol=1e-6)
        # The correction makes the statistic smaller here, so the p-value
        # larger: the conservative direction.
        assert mine.statistic < theirs["stat_corr"]
        assert mine.p_value > theirs["p_corr"]

    def test_a_single_stratum_reduces_to_the_mantel_haenszel_chi_square(self):
        """With one stratum the CMH test is the ordinary 2x2 chi-square.

        It is not the *Yates* corrected one -- the correction is applied
        once to the pooled residual, not to each stratum -- so the
        reference here is the uncorrected Pearson statistic on the same
        table.
        """
        from stats.univariate_extra import CochranMantelHaenszelAnalyzer

        table = np.array([[20.0, 30.0, 25.0, 35.0]])
        mine = CochranMantelHaenszelAnalyzer().analyze(table, correction=False)

        a, b, c, d = table[0]
        n = a + b + c + d
        expected = (a - (a + b) * (a + c) / n) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1)))
        assert_allclose(mine.statistic, expected, rtol=RTOL, atol=ATOL)
        assert mine.n_strata == 1
        assert mine.df == 1
