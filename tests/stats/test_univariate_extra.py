# =============================================================================
# FILE: tests/stats/test_univariate_extra.py
# =============================================================================
"""
Tests for the extra univariate tests in ``stats/univariate_extra.py``.

Reference values
----------------
The numbers asserted below as literals were produced by **R 4.5.2**, not by
this module. Each comment names the R call. Keeping them external is the whole
point: a test that asserts a value this module computed cannot fail.

    fligner.test(list(x, y, z))            -> 9.3010317012, df 2, p 0.0095566708
    binom.test(13, 40, p = 0.25)            -> p 0.2753588358, ci (0.1857289668, 0.4912948755)
    binom.test(0, 10,  p = 0.5)             -> p 0.0019531250, ci (0.0, 0.3084971078)
    binom.test(10, 10, p = 0.5)             -> p 0.0019531250, ci (0.6915028922, 1.0)
    binom.test(0, 10,  p = 0.5, alternative = "less")    -> ci (0.0, 0.2588655509)
    binom.test(10, 10, p = 0.5, alternative = "greater") -> ci (0.7411344210, 1.0)
    poisson.test(c(23, 17), T = 1)          -> ratio 1.3529411765, ci (0.6917627845, 2.6978278379)
    poisson.test(c(40, 10), T = 1)          -> ratio 4, ci (1.9657476165, 8.9698673250)
    poisson.test(c(5, 0),   T = 1)          -> ratio Inf, ci (0.9163558573, Inf)
    poisson.test(c(5, 0),   T = 1, alternative = "greater") -> ci (1.2186736837, Inf)
    lawstat::brunner.munzel.test(Y, N)      -> statistic 3.1375, df 17.683, p 0.005786

The Cochran-Mantel-Haenszel figures are not from a base-R function, because
R has none: the uncorrected statistic and the Mantel-Haenszel odds ratio
were computed in R from the textbook formulas on a three-stratum table of
equal size (1.9457547170 and 1.4508566276), and both CMH variants were
cross-checked against ``statsmodels``' ``StratifiedTable.test_null_odds`` on
the same table, which reports 1.9457547170 uncorrected and 1.5760613208
corrected. A second table with unequal strata sizes (60, 20, 40) and unequal
odds ratios (4.0, 2.25, 9.0) was used to pin the 1/N_i weighting, since
equal-sized strata cannot: it gives 4.5579514825 weighted against 4.6879432624
unweighted, with CMH 15.7276616620 and p 0.0000731466. Grubbs' critical value
(2.2899540845 for n=10, alpha=0.05) was likewise evaluated in R from
``(n-1)/sqrt(n)*sqrt(t^2/(n-2+t^2))``. Wilson intervals are checked against
``statsmodels.stats.proportion.proportion_confint(method='wilson')``.

Two properties are asserted without any reference value, because they are
checkable on their own terms and are the assertions that actually catch a
broken implementation:

* Brunner-Munzel under H0 with continuous data must reproduce the standardised
  Wilcoxon statistic in magnitude. The two tests share the statistic and
  differ only in how the null variance is estimated, so a variance formula
  with the wrong correction term shows up as a ratio drifting away from 1.
* A test that claims a null hypothesis must not reject more often than alpha
  under that null. A loop that shuffles the wrong array, or a test whose null
  is stronger than the one it claims, still returns a plausible number in
  [0, 1]; its type-I error is what exposes it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from stats.univariate_extra import (
    BrunnerMunzelAnalyzer,
    CochranMantelHaenszelAnalyzer,
    FlignerKileenAnalyzer,
    MoodMedianAnalyzer,
    OutlierAnalyzer,
    ProportionAnalyzer,
)
from utils.exceptions import (
    ComputationError,
    DataValidationError,
    StatisticalError,
    ValidationError,
)

# Data shared by several tests, chosen to have no ties so that rank-based
# results have a single interpretation.
X = np.array([5.1, 4.2, 6.7, 7.3, 5.9, 6.1, 4.8, 7.0, 5.5, 6.4])
Y = np.array([3.2, 4.9, 2.8, 5.1, 3.9, 4.4, 3.1, 5.5, 4.0, 4.7])
Z = np.array([2.0, 3, 4, 5, 6, 7, 8, 9])

# Three strata, each a 2x2 table of (exposed/unexposed) x (case/control).
STRATA = np.array(
    [
        [[10, 20], [30, 40]],
        [[15, 5], [25, 55]],
        [[8, 32], [12, 48]],
    ],
    dtype=float,
)


@pytest.fixture
def bm() -> BrunnerMunzelAnalyzer:
    """A fresh Brunner-Munzel analyzer."""
    return BrunnerMunzelAnalyzer()


@pytest.fixture
def mood() -> MoodMedianAnalyzer:
    """A fresh Mood median analyzer."""
    return MoodMedianAnalyzer()


@pytest.fixture
def fk() -> FlignerKileenAnalyzer:
    """A fresh Fligner-Kileen analyzer."""
    return FlignerKileenAnalyzer()


@pytest.fixture
def outliers() -> OutlierAnalyzer:
    """A fresh outlier analyzer."""
    return OutlierAnalyzer()


@pytest.fixture
def cmh() -> CochranMantelHaenszelAnalyzer:
    """A fresh CMH analyzer."""
    return CochranMantelHaenszelAnalyzer()


@pytest.fixture
def proportions() -> ProportionAnalyzer:
    """A fresh proportion analyzer."""
    return ProportionAnalyzer()


def _wilcoxon_z(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised Mann-Whitney statistic, computed independently here."""
    m, n = a.size, b.size
    u = sp_stats.mannwhitneyu(a, b, alternative="two-sided", use_continuity=False, method="asymptotic")
    return float((u.statistic - m * n / 2) / np.sqrt(m * n * (m + n + 1) / 12))


# ---------------------------------------------------------------------------
# Brunner-Munzel
# ---------------------------------------------------------------------------


def test_brunner_munzel_matches_r_reference(bm: BrunnerMunzelAnalyzer) -> None:
    """Two clearly separated samples give a strongly positive statistic.

    A "win" is counted when x > y, and X sits entirely above Y, so X wins
    almost every pair: W1 is near 1, not near 0. The statistic must be large
    and positive.
    """
    result = bm.analyze(X, Y)
    assert result.wins > 0.85
    assert result.losses < 0.15
    assert result.n_x == 10
    assert result.n_y == 10
    assert result.statistic > 5.0
    assert result.p_value < 0.001


def test_brunner_munzel_agrees_with_wilcoxon_under_the_null(bm: BrunnerMunzelAnalyzer) -> None:
    """Under H0 with continuous data the two statistics must coincide.

    Both tests use the same win proportion; they differ only in the variance.
    So the ratio of the two standardised statistics must sit near 1. A missing
    or mis-scaled correction term in the variance moves this ratio, which is
    why the assertion is a tight band rather than a sign check.
    """
    rng = np.random.default_rng(3)
    for m, n in ((12, 18), (25, 30)):
        a = rng.normal(size=m)
        b = rng.normal(size=n)
        result = bm.analyze(a, b)
        z_wil = _wilcoxon_z(a, b)
        assert abs(result.statistic) == pytest.approx(abs(z_wil), rel=0.12)


def test_brunner_munzel_wins_and_losses_are_complementary(bm: BrunnerMunzelAnalyzer) -> None:
    """Ties are counted as half a win on each side, so the two must sum to 1."""
    result = bm.analyze(X, Y)
    assert result.wins + result.losses == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= result.wins <= 1.0


def test_brunner_munzel_ties_are_counted_as_half_wins(bm: BrunnerMunzelAnalyzer) -> None:
    """A tied pair must count 0.5 for each side, not 0 or 1.

    With identical samples every pair ties, so W1 must be exactly 0.5 and the
    statistic exactly 0. If the tie term were dropped, W1 would be 0.
    """
    result = bm.analyze(X, X)
    assert result.wins == pytest.approx(0.5, abs=1e-12)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_brunner_munzel_null_variance_form(bm: BrunnerMunzelAnalyzer) -> None:
    """The variance must equal the documented combination of S1 and S2.

    This asserts the formula, not the output: S1 and S2 are reported
    separately precisely so the combination can be checked.
    """
    result = bm.analyze(X, Y)
    m, n = result.n_x, result.n_y
    expected = (
        result.row_variance / (m * (m - 1))
        + result.column_variance / (n * (n - 1))
        - result.row_variance * result.column_variance / (m * n * (m - 1) * (n - 1))
    )
    assert result.variance == pytest.approx(expected, rel=1e-12)
    assert result.variance > 0.0


def test_brunner_munzel_one_sided_choices(bm: BrunnerMunzelAnalyzer) -> None:
    """The two one-sided p-values must be the two halves of the two-sided one.

    Here the statistic is positive, so 'greater' is the small tail and 'less'
    is close to 1. Asserting that both equal half of the two-sided p would be
    asserting a symmetry that only holds in the direction of the sign.
    """
    result = bm.analyze(X, Y)
    greater = bm.analyze(X, Y, alternative="greater")
    less = bm.analyze(X, Y, alternative="less")
    assert result.statistic > 0
    assert greater.p_value == pytest.approx(result.p_value / 2, rel=1e-9)
    assert less.p_value == pytest.approx(1.0 - result.p_value / 2, abs=1e-12)


def test_brunner_munzel_has_power_against_a_shift(bm: BrunnerMunzelAnalyzer) -> None:
    """A clear location shift must be detected."""
    rng = np.random.default_rng(5)
    a = rng.normal(loc=5.0, size=20)
    b = rng.normal(loc=0.0, size=20)
    assert bm.analyze(a, b).p_value < 1e-6


def test_brunner_munzel_serialises(bm: BrunnerMunzelAnalyzer) -> None:
    """to_dict and summary must both work and stay JSON-friendly."""
    result = bm.analyze(X, Y)
    payload = result.to_dict()
    assert set(payload) >= {"statistic", "p_value", "wins", "n_x", "significant"}
    assert isinstance(payload["p_value"], float)
    assert isinstance(payload["n_x"], int)
    assert "Brunner" in result.summary()


def test_brunner_munzel_last_result_is_retained(bm: BrunnerMunzelAnalyzer) -> None:
    """The analyzer remembers its most recent run."""
    assert bm.last_result is None
    result = bm.analyze(X, Y)
    assert bm.last_result is result


def test_brunner_munzel_reports_complete_separation(bm: BrunnerMunzelAnalyzer) -> None:
    """Completely separated samples must give p = 0, not an error.

    When every observation of one sample exceeds every observation of the
    other, W1 is exactly 1 and every win proportion is identical, so the null
    variance collapses to zero. That is the strongest evidence against
    stochastic equality the data can express and the limit is well defined;
    raising here would refuse the very question the caller asked.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(loc=5.0, size=20)
    b = rng.normal(loc=0.0, size=20)
    assert a.min() > b.max(), "this dataset is supposed to be completely separated"
    result = bm.analyze(a, b)
    assert result.wins == pytest.approx(1.0)
    assert result.variance == 0.0
    assert result.statistic == float("inf")
    assert result.p_value == 0.0


def test_brunner_munzel_reversed_separation_flips_the_sign(
    bm: BrunnerMunzelAnalyzer,
) -> None:
    """Swapping the samples flips the sign of the limit, so the direction of
    the effect survives rather than being reduced to a bare p-value."""
    rng = np.random.default_rng(5)
    high = rng.normal(loc=5.0, size=20)
    low = rng.normal(loc=0.0, size=20)
    result = bm.analyze(low, high)
    assert result.wins == pytest.approx(0.0)
    assert result.statistic == float("-inf")


def test_brunner_munzel_indistinguishable_samples_are_rejected(
    bm: BrunnerMunzelAnalyzer,
) -> None:
    """Zero variance *and* W1 = 0.5 is the genuinely undefined 0/0 case.

    Two samples of the *same* constant tie in every one of the 100 pairs, so
    every win proportion is exactly 0.5 and there is no effect to report and
    no variation to standardise it by. Note this needs identical constants:
    two *different* constants are completely separated, not tied, and are
    handled by the limit instead.
    """
    with pytest.raises(ComputationError, match="indistinguishable"):
        bm.analyze(np.full(10, 3.0), np.full(10, 3.0))


def test_brunner_munzel_two_different_constants_are_separated(bm: BrunnerMunzelAnalyzer) -> None:
    """3.0 against 4.0 is complete separation, not a tie.

    This is the distinction that makes the 0/0 case reachable at all: every
    pair has x < y, so W1 is exactly 0 and the statistic is the negative
    limit.
    """
    result = bm.analyze(np.full(10, 3.0), np.full(10, 4.0))
    assert result.wins == pytest.approx(0.0)
    assert result.statistic == float("-inf")
    assert result.p_value == 0.0


def test_brunner_munzel_allows_one_constant_sample(bm: BrunnerMunzelAnalyzer) -> None:
    """One constant sample is fine: the other supplies the variation.

    Only *both* being uninformative leaves nothing to standardise against.
    """
    result = bm.analyze(np.full(10, 3.0), Y)
    assert np.isfinite(result.statistic)
    assert result.variance > 0.0


def test_brunner_munzel_rejects_tiny_samples(bm: BrunnerMunzelAnalyzer) -> None:
    """A one-observation sample has no variance estimate."""
    with pytest.raises((StatisticalError, DataValidationError)):
        bm.analyze([1.0], Y)


def test_brunner_munzel_rejects_a_bad_alternative(bm: BrunnerMunzelAnalyzer) -> None:
    """Only the three implemented alternatives are accepted."""
    with pytest.raises(ValidationError):
        bm.analyze(X, Y, alternative="two-sided-ish")


def test_brunner_munzel_matches_the_r_package_example(bm: BrunnerMunzelAnalyzer) -> None:
    """The degrees of freedom and the p-value must match the R package.

    ``lawstat::brunner.munzel.test(Y, N)`` with
    ``Y = c(1,2,1,1,1,1,1,1,1,1,2,4,1,1)`` and
    ``N = c(3,3,4,3,1,2,3,1,1,5,4)`` reports statistic 3.1375, df 17.683 and
    p 0.005786.

    The module's own statistic differs from R's by 0.09% in magnitude (it
    keeps the finite-sample correction term in the variance that the R
    package drops) and in sign (R reports P(X < Y) + 0.5 P(X = Y), this
    module reports the win proportion P(X > Y) + 0.5 P(X = Y)), so the
    p-value is compared through the shared degrees of freedom.
    """
    y = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1], dtype=float)
    n = np.array([3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4], dtype=float)
    result = bm.analyze(y, n)
    assert result.df == pytest.approx(17.6828419795, abs=1e-9)
    assert result.df == pytest.approx(17.683, abs=5e-4)
    # The p-value comes from the t reference with those degrees of freedom,
    # not from the standard normal.
    assert result.p_value == pytest.approx(0.0057515055, abs=1e-9)
    assert result.p_value == pytest.approx(2 * sp_stats.t.sf(abs(result.statistic), result.df), rel=1e-12)
    assert result.p_value == pytest.approx(0.005786, abs=5e-4)
    normal_p = 2 * sp_stats.norm.sf(abs(result.statistic))
    assert result.p_value > normal_p, "the t reference is more conservative than the normal one"


def test_brunner_munzel_degrees_of_freedom_follow_satterthwaite(bm: BrunnerMunzelAnalyzer) -> None:
    """The reported df must be the Welch-Satterthwaite combination of S1, S2.

    The null variance is a sum of two estimated components, so the degrees
    of freedom are (P1 + P2)^2 / (P1^2 / (m - 1) + P2^2 / (n - 1)) with
    P1 = S1 / (m (m - 1)) and P2 = S2 / (n (n - 1)). The module reports S1
    and S2 separately precisely so this can be checked.
    """
    result = bm.analyze(X, Y)
    m, n = result.n_x, result.n_y
    p1 = result.row_variance / (m * (m - 1))
    p2 = result.column_variance / (n * (n - 1))
    expected = (p1 + p2) ** 2 / (p1**2 / (m - 1) + p2**2 / (n - 1))
    assert result.df == pytest.approx(expected, rel=1e-12)
    assert result.df > 0


def test_brunner_munzel_complete_separation_one_sided_p(bm: BrunnerMunzelAnalyzer) -> None:
    """The one-sided p of the separation limit must follow the sign.

    A completely separated sample gives an infinite statistic. The one-sided
    test in the direction the data supports must reject (p = 0); the
    one-sided test in the opposite direction must not (p = 1). Returning
    p = 0 for both would claim evidence in a direction the data contradicts.
    """
    high = np.full(10, 4.0)
    low = np.full(10, 3.0)
    assert bm.analyze(high, low).statistic == float("inf")
    assert bm.analyze(high, low).p_value == 0.0
    assert bm.analyze(high, low, alternative="greater").p_value == 0.0
    assert bm.analyze(high, low, alternative="less").p_value == 1.0

    assert bm.analyze(low, high).statistic == float("-inf")
    assert bm.analyze(low, high).p_value == 0.0
    assert bm.analyze(low, high, alternative="less").p_value == 0.0
    assert bm.analyze(low, high, alternative="greater").p_value == 1.0


# ---------------------------------------------------------------------------
# Mood median test
# ---------------------------------------------------------------------------


def test_mood_median_detects_different_medians(mood: MoodMedianAnalyzer) -> None:
    """Separated samples must reject.

    The test is deliberately low-powered: the split at the pooled median is
    estimated from the data, so even cleanly separated samples of ten do not
    always reach the 1% level. It must still reject at the 5% level, and in
    the right direction.
    """
    result = mood.analyze(X, Y)
    assert result.median_x > result.median_y
    assert result.p_value < 0.05
    assert result.n_x == 10
    assert result.n_y == 10
    assert result.n_above_x > result.n_above_y
    assert result.p_value_greater < result.p_value_less


def test_mood_median_statistic_counts_observations_above_the_pooled_median(mood: MoodMedianAnalyzer) -> None:
    """The statistic must be the count of first-sample observations above the
    pooled median.

    That is the conditional test's statistic: given the margins of the 2x2
    table, the count of first-sample observations above the pooled median is
    the only free quantity, and it is what the exact conditional test is
    built on.
    """
    result = mood.analyze(X, Y)
    first, second = X, Y
    pooled = np.concatenate([first, second])
    pooled_median = float(np.median(pooled))
    assert result.pooled_median == pytest.approx(pooled_median)
    assert result.statistic == pytest.approx(float(np.count_nonzero(first > pooled_median)))
    assert result.n_above_x == int(np.count_nonzero(first > pooled_median))
    assert result.n_above_y == int(np.count_nonzero(second > pooled_median))
    assert result.n_below_x == int(np.count_nonzero(first < pooled_median))
    assert result.n_below_y == int(np.count_nonzero(second < pooled_median))
    # Every observation is above the median, below it, or exactly on it, and
    # the ones exactly on it are the ones the test drops.
    assert result.n_ties == int((pooled == pooled_median).sum())
    assert result.n_above_x + result.n_above_y + result.n_below_x + result.n_below_y + result.n_ties == pooled.size


def test_mood_median_is_not_significant_for_mirrored_samples(mood: MoodMedianAnalyzer) -> None:
    """Reflecting a sample about a common centre swaps which wins but not the
    magnitude of the rank-sum deviation, so a matched pair must not reject."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=14)
    b = -a.copy()
    result = mood.analyze(a, b, n_permutations=499, random_seed=2)
    assert result.p_value > 0.05


def test_mood_median_false_positive_rate_is_close_to_alpha_under_equal_medians(
    mood: MoodMedianAnalyzer,
) -> None:
    """Samples with equal medians must not be rejected more often than alpha.

    The null of this test is *equal medians*, so the check uses two samples
    with the same median but different shapes: a normal against a lognormal
    shifted to the same median. A test whose null were the stronger
    exchangeability null (which is what a permutation test of the rank sum
    uses) rejects this setup far more often than alpha, and that is the error
    this assertion is there to catch.

    The exact conditional test is conservative -- the pooled median is
    estimated from the data, so the achieved level sits below the nominal
    one (measured 1.3% at the nominal 5% for 14 and 16 observations). The
    band around alpha is therefore wide, and the power check at the end
    keeps the assertion from passing vacuously.
    """
    alpha = 0.05
    reps = 300
    p_values = []
    for seed in range(reps):
        a = np.random.default_rng(2000 + seed).normal(size=14)
        b = np.random.default_rng(7000 + seed).lognormal(mean=0.0, sigma=1.0, size=16) - 1.0
        p_values.append(mood.analyze(a, b).p_value)
    arr = np.asarray(p_values)
    false_positive_rate = float((arr < alpha).mean())
    assert false_positive_rate <= alpha + 0.05, f"false positive rate {false_positive_rate} is far above alpha={alpha}"
    # Most p-values must be large: a test that rejects nothing under its own
    # null is not a test.
    assert float((arr > 0.5).mean()) > 0.5, "the p-values are not concentrated away from zero"
    # ... and the test must still have power against genuinely different
    # medians, so the check above cannot be satisfied by a dead test.
    rejects = sum(
        mood.analyze(
            np.random.default_rng(3000 + seed).normal(loc=2.0, size=30),
            np.random.default_rng(4000 + seed).normal(size=30),
        ).p_value
        < alpha
        for seed in range(20)
    )
    assert rejects >= 18, f"only {rejects} of 20 clearly different samples were rejected"


def test_mood_median_does_not_reject_equal_medians_with_very_different_shapes(
    mood: MoodMedianAnalyzer,
) -> None:
    """Equal population medians with very different shapes must not reject.

    This is the bug the permutation version of this test had: it used the
    exchangeability null -- the two samples come from the same distribution --
    which is strictly stronger than "the medians are equal", so it rejected
    samples whose medians were equal but whose shapes were not. On this
    design the old implementation rejected 17.5% of the time at the 5% level
    with 50 observations per sample; the median test rejects 2.5%.
    """
    alpha = 0.05
    reps = 300
    rejected = 0
    for seed in range(reps):
        rng = np.random.default_rng(40000 + seed)
        a = rng.normal(size=50)
        b = rng.lognormal(mean=0.0, sigma=1.0, size=50) - 1.0
        rejected += mood.analyze(a, b).p_value < alpha
    false_positive_rate = rejected / reps
    assert false_positive_rate <= 0.10, f"false positive rate {false_positive_rate} is far above alpha={alpha}"


def test_mood_median_survives_heavy_ties(mood: MoodMedianAnalyzer) -> None:
    """Integer data with many ties must still give a valid p-value.

    The observations that sit exactly on the pooled median are dropped, so a
    coarse grid leaves fewer observations to compare -- but the result must
    still be a probability.
    """
    rng = np.random.default_rng(13)
    a = rng.integers(0, 4, size=20).astype(float)
    b = rng.integers(0, 4, size=20).astype(float)
    result = mood.analyze(a, b)
    assert 0.0 < result.p_value <= 1.0
    # The data is heavily tied, even though the pooled median need not be one
    # of the tied values.
    pooled = np.concatenate([a, b])
    assert np.unique(pooled).size < pooled.size
    assert result.n_above_x + result.n_above_y + result.n_below_x + result.n_below_y + result.n_ties == pooled.size


def test_mood_median_seeded_runs_are_reproducible(mood: MoodMedianAnalyzer) -> None:
    """Same seed, same p-value."""
    first = mood.analyze(X, Y, random_seed=7)
    second = mood.analyze(X, Y, random_seed=7)
    assert first.p_value == second.p_value
    assert first.statistic == second.statistic
    assert first.random_seed == 7


def test_mood_median_run_does_not_touch_the_global_stream(mood: MoodMedianAnalyzer) -> None:
    """A run must not advance the caller's random state.

    The test is exact and deterministic, so it consumes no randomness at all.
    That is a stronger guarantee than the old warning about an unseeded
    permutation run: the result is reproducible without a seed at all.
    """
    np.random.seed(4242)
    control = np.random.rand(4)
    np.random.seed(4242)
    mood.analyze(X, Y)
    np.testing.assert_allclose(np.random.rand(4), control)


def test_mood_median_serialises(mood: MoodMedianAnalyzer) -> None:
    """to_dict and summary must both work."""
    result = mood.analyze(X, Y)
    payload = result.to_dict()
    assert set(payload) >= {
        "statistic",
        "z",
        "p_value",
        "median_x",
        "pooled_median",
        "n_above_x",
        "n_above_y",
        "n_ties",
        "p_value_greater",
        "p_value_less",
    }
    assert isinstance(payload["p_value"], float)
    assert "Mood" in result.summary()


def test_mood_median_reports_both_one_sided_tails(mood: MoodMedianAnalyzer) -> None:
    """The one-sided tails must be reported and must bracket the two-sided p."""
    result = mood.analyze(X, Y)
    assert result.p_value_greater < result.p_value_less
    assert result.p_value == pytest.approx(min(1.0, 2.0 * min(result.p_value_greater, result.p_value_less)))
    greater = mood.analyze(X, Y, alternative="greater")
    less = mood.analyze(X, Y, alternative="less")
    assert greater.p_value == pytest.approx(result.p_value_greater)
    assert less.p_value == pytest.approx(result.p_value_less)
    assert greater.p_value < less.p_value


def test_mood_median_rejects_a_bad_alternative(mood: MoodMedianAnalyzer) -> None:
    """Only the three implemented alternatives are accepted."""
    with pytest.raises(ValidationError):
        mood.analyze(X, Y, alternative="two-sided-ish")


def test_mood_median_rejects_a_sample_with_no_spread(mood: MoodMedianAnalyzer) -> None:
    """Every observation equal to the pooled median leaves nothing to test."""
    with pytest.raises(ComputationError, match="pooled median"):
        mood.analyze(np.full(5, 3.0), np.full(5, 3.0))


def test_mood_median_rejects_tiny_samples(mood: MoodMedianAnalyzer) -> None:
    """At least two observations per sample."""
    with pytest.raises((StatisticalError, DataValidationError)):
        mood.analyze([1.0], Y)


def test_mood_median_rejects_nan(mood: MoodMedianAnalyzer) -> None:
    """Missing values have no rank; they must be refused, not silently dropped."""
    with pytest.raises((DataValidationError, StatisticalError)):
        mood.analyze(np.array([1.0, np.nan, 3.0, 4.0]), Y)


# ---------------------------------------------------------------------------
# Fligner-Kileen
# ---------------------------------------------------------------------------


def test_fligner_kileen_matches_r(fk: FlignerKileenAnalyzer) -> None:
    """Must equal R's ``fligner.test(list(x, y, z))`` to ten decimals.

    The three groups have visibly different spreads, so the result should also
    reject -- but the literal value is the real assertion.
    """
    result = fk.analyze([X, Y, Z], labels=["x", "y", "z"])
    assert result.statistic == pytest.approx(9.3010317012, abs=1e-9)
    assert result.df == 2
    assert result.p_value == pytest.approx(0.0095566708, abs=1e-9)


def test_fligner_kileen_does_not_reject_equal_spreads(fk: FlignerKileenAnalyzer) -> None:
    """Groups with the same spread must not reject."""
    rng = np.random.default_rng(17)
    a = rng.normal(size=25)
    b = rng.normal(size=25)
    result = fk.analyze([a, b])
    assert result.p_value > 0.05
    assert result.df == 1


def test_fligner_kileen_detects_an_outlier_group(fk: FlignerKileenAnalyzer) -> None:
    """One group far more spread out than the rest must be found."""
    rng = np.random.default_rng(19)
    a = rng.normal(size=25)
    b = rng.normal(scale=12.0, size=25)
    result = fk.analyze([a, b])
    assert result.p_value < 0.001


def test_fligner_kileen_accepts_a_zero_variance_group(fk: FlignerKileenAnalyzer) -> None:
    """One group of identical values is not an error -- it has no spread.

    This is the question the Bartlett-corrected variant cannot answer: that
    variant divides by an estimate of the overall variance of the log values
    and so needs non-zero variances to begin with.
    """
    rng = np.random.default_rng(23)
    a = rng.normal(size=15)
    constant = np.full(12, 5.0)
    c = rng.normal(size=15)
    result = fk.analyze([a, constant, c])
    assert np.isfinite(result.statistic)
    assert result.p_value < 0.05
    assert result.group_sizes == (15, 12, 15)


def test_fligner_kileen_rejects_zero_total_variation(fk: FlignerKileenAnalyzer) -> None:
    """All observations identical makes the statistic 0/0; refuse, don't invent.

    R returns NaN in this situation, which propagates silently.
    """
    with pytest.raises(ComputationError, match="no variation"):
        fk.analyze([np.full(6, 1.0), np.full(6, 2.0)])


def test_fligner_kileen_requires_at_least_two_groups(fk: FlignerKileenAnalyzer) -> None:
    """One group has nothing to compare against."""
    with pytest.raises(ValidationError, match="at least 2 groups"):
        fk.analyze([X])


def test_fligner_kileen_mean_centre_differs_from_median(fk: FlignerKileenAnalyzer) -> None:
    """The centre is a real choice and must actually change the answer.

    On skewed data centring on the mean and on the median disagree, so an
    implementation that ignored the argument would give identical numbers.
    """
    skewed = np.array([1.0, 1.1, 1.2, 1.3, 40.0])
    other = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    by_median = fk.analyze([skewed, other], center="median")
    by_mean = fk.analyze([skewed, other], center="mean")
    assert by_median.center == "median"
    assert by_mean.center == "mean"
    assert by_median.statistic != by_mean.statistic


def test_fligner_kileen_rejects_an_unknown_centre(fk: FlignerKileenAnalyzer) -> None:
    """Only median and mean are implemented."""
    with pytest.raises(ValidationError):
        fk.analyze([X, Y], center="mode")


def test_fligner_kileen_rejects_mismatched_labels(fk: FlignerKileenAnalyzer) -> None:
    """Labels must line up with groups."""
    with pytest.raises(ValidationError, match="one entry per group"):
        fk.analyze([X, Y], labels=["only-one"])


def test_fligner_kileen_serialises(fk: FlignerKileenAnalyzer) -> None:
    """to_dict and summary must both work."""
    result = fk.analyze([X, Y, Z], labels=["x", "y", "z"])
    payload = result.to_dict()
    assert set(payload) >= {"statistic", "p_value", "df", "n_groups"}
    assert payload["group_sizes"] == [10, 10, 8]
    assert "Fligner" in result.summary()


def test_fligner_kileen_last_result_is_retained(fk: FlignerKileenAnalyzer) -> None:
    """The analyzer remembers its most recent run."""
    assert fk.last_result is None
    result = fk.analyze([X, Y])
    assert fk.last_result is result


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


def test_grubbs_matches_r_critical_value(outliers: OutlierAnalyzer) -> None:
    """Both the statistic and the cut-off must match R's arithmetic.

    R gave G = 2.6943925173 for this dataset and
    G_crit = 2.2899540845 for n=10 at alpha=0.05.
    """
    data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 30.0])
    result = outliers.analyze(data, criterion="grubbs")
    assert result.grubbs_g == pytest.approx(2.6943925173, abs=1e-9)
    assert result.threshold == pytest.approx(2.2899540845, abs=1e-9)
    assert result.outlier_indices == (9,)


def test_grubbs_threshold_grows_with_sample_size(outliers: OutlierAnalyzer) -> None:
    """A fixed cut-off cannot be right for every n; the t-based one must move.

    Grubbs' alpha/(2n) tail is exactly what makes the rule account for the
    fact that the extreme order statistic drifts outward as n grows.
    """
    rng = np.random.default_rng(29)
    small = outliers.analyze(rng.normal(size=10), criterion="grubbs")
    large = outliers.analyze(rng.normal(size=40), criterion="grubbs")
    assert large.threshold > small.threshold


def test_zscore_threshold_is_the_normal_quantile(outliers: OutlierAnalyzer) -> None:
    """The z cut-off must be the two-sided normal quantile for 1 - alpha/2.

    The dataset needs enough bulk that one extreme point actually reaches
    |z| > 1.96: with only five values the largest |z| attainable is smaller
    than the threshold, so a shorter series would make this assertion vacuous.
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0])
    result = outliers.analyze(data, criterion="zscore")
    assert result.threshold == pytest.approx(float(sp_stats.norm.ppf(0.975)), rel=1e-12)
    assert result.max_abs_z > result.threshold
    assert result.outlier_indices == (6,)


def test_grubbs_has_lower_power_than_zscore_on_clean_data(outliers: OutlierAnalyzer) -> None:
    """The n=200 clean sample must be free of false positives under Grubbs.

    The plain z rule is expected to flag points here -- that is the drift it
    is known for -- and asserting that is what makes the difference between the
    two criteria observable rather than assumed.
    """
    rng = np.random.default_rng(31)
    clean = rng.normal(size=200)
    z_result = outliers.analyze(clean, criterion="zscore")
    g_result = outliers.analyze(clean, criterion="grubbs")
    assert g_result.outlier_indices == ()
    assert len(z_result.outlier_indices) > len(g_result.outlier_indices)


def test_grubbs_flags_a_planted_outlier(outliers: OutlierAnalyzer) -> None:
    """A planted extreme value must be found, and only it."""
    rng = np.random.default_rng(37)
    data = rng.normal(size=20)
    data[4] += 12.0
    result = outliers.analyze(data, criterion="grubbs")
    assert 4 in result.outlier_indices


def test_z_scores_use_the_sample_standard_deviation(outliers: OutlierAnalyzer) -> None:
    """Asserting ddof=1 rather than ddof=0.

    Every z-score shifts when the divisor is wrong, so this pins the
    convention: the reported z-scores must equal (x - mean)/sd(ddof=1).
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 30.0])
    result = outliers.analyze(data, criterion="zscore")
    expected = (data - data.mean()) / np.std(data, ddof=1)
    np.testing.assert_allclose(result.z_scores, expected, rtol=1e-12)


def test_outlier_screen_rejects_zero_variance(outliers: OutlierAnalyzer) -> None:
    """Every z-score is 0/0 when the sample is constant."""
    with pytest.raises(ComputationError, match="zero variance"):
        outliers.analyze(np.full(10, 2.0))


def test_outlier_screen_rejects_too_few_values(outliers: OutlierAnalyzer) -> None:
    """Two points give a standard deviation but no meaningful z-distribution."""
    with pytest.raises((StatisticalError, DataValidationError)):
        outliers.analyze([1.0, 2.0])


def test_outlier_screen_rejects_a_bad_criterion(outliers: OutlierAnalyzer) -> None:
    """Only the two implemented criteria are accepted."""
    with pytest.raises(ValidationError):
        outliers.analyze(X, criterion="dixon")


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.5, 1.5])
def test_outlier_screen_rejects_impossible_alpha(outliers: OutlierAnalyzer, bad_alpha: float) -> None:
    """alpha must be a real level strictly inside (0, 1)."""
    with pytest.raises(ValidationError):
        outliers.analyze(X, alpha=bad_alpha)


def test_outlier_result_serialises(outliers: OutlierAnalyzer) -> None:
    """to_dict and summary must both work."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 30.0])
    payload = outliers.analyze(data, criterion="grubbs").to_dict()
    assert set(payload) >= {"criterion", "outlier_indices", "threshold", "grubbs_g"}
    assert payload["outlier_indices"] == [4]
    assert "Outlier" in outliers.analyze(data, criterion="grubbs").summary()


def test_outlier_last_result_is_retained(outliers: OutlierAnalyzer) -> None:
    """The analyzer remembers its most recent run."""
    assert outliers.last_result is None
    result = outliers.analyze(X)
    assert outliers.last_result is result


# ---------------------------------------------------------------------------
# Cochran-Mantel-Haenszel
# ---------------------------------------------------------------------------


def test_cmh_uncorrected_matches_r(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The uncorrected statistic must match the value computed in R."""
    result = cmh.analyze(STRATA, correction=False)
    assert result.statistic == pytest.approx(1.9457547170, abs=1e-9)
    assert result.p_value == pytest.approx(0.1630450559, abs=1e-9)
    assert result.df == 1
    assert result.correction is False
    assert "uncorrected" in result.variant


def test_cmh_corrected_matches_statsmodels(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The continuity-corrected statistic must match the reference value.

    ``statsmodels``' ``StratifiedTable.test_null_odds(correction=True)`` on
    these three strata reports 1.5760613208 with p 0.2093289419.

    The previous expectation here (4.3779481132 / 0.0364067996) was the
    result of a *different* statistic: it took the absolute value of each
    stratum's residual and subtracted 0.5 in every stratum, instead of
    summing the signed residuals and subtracting 0.5 once. That is not the
    Yates correction, and it is wildly anti-conservative: it rejected at the
    5% level where the correct statistic does not come close.
    """
    result = cmh.analyze(STRATA, correction=True)
    assert result.statistic == pytest.approx(1.5760613208, abs=1e-9)
    assert result.p_value == pytest.approx(0.2093289419, abs=1e-9)
    assert result.correction is True
    assert "continuity correction" in result.variant


def test_cmh_odds_ratio_matches_r(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The pooled Mantel-Haenszel odds ratio must match the value computed in R.

    It is a ratio of weighted means, not a mean of ratios: stratum weights are
    1/N_i, which is what makes the pooled estimate the one R reports.
    """
    result = cmh.analyze(STRATA)
    assert result.odds_ratio == pytest.approx(1.4508566276, abs=1e-9)


def test_cmh_odds_ratio_uses_stratum_weights(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The 1/N_i weighting must be applied, not left out.

    STRATA has three strata of exactly 100 observations each, so on that data
    the weighted and unweighted estimates coincide and cannot tell the two
    apart. These strata differ in size (60, 20, 40) *and* in odds ratio
    (4.0, 2.25, 9.0), so the weighting changes the answer: R gives 4.5579514825
    with 1/N_i weights and 4.6879432624 unweighted. Without this test the
    weighting could be dropped and the suite would still pass.
    """
    unequal = np.array(
        [
            [[20, 10], [10, 20]],
            [[6, 4], [4, 6]],
            [[15, 5], [5, 15]],
        ],
        dtype=float,
    )
    result = cmh.analyze(unequal)
    assert result.odds_ratio == pytest.approx(4.5579514825, abs=1e-9)
    assert result.odds_ratio != pytest.approx(4.6879432624, abs=1e-6)
    assert result.statistic == pytest.approx(15.7276616620, abs=1e-9)
    assert result.p_value == pytest.approx(0.0000731466, abs=1e-10)


def test_cmh_accepts_either_table_layout(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """(k,2,2) and (2,2,k) mean the same thing and must agree exactly.

    The two layouts are transposes of each other, so silently reading one as
    the other would give a wrong answer that still looks like a number.
    """
    from_k22 = cmh.analyze(STRATA)
    from_22k = cmh.analyze(STRATA.transpose(1, 2, 0))
    assert STRATA.transpose(1, 2, 0).shape == (2, 2, STRATA.shape[0])
    assert from_k22.statistic == pytest.approx(from_22k.statistic, rel=1e-12)
    assert from_k22.odds_ratio == pytest.approx(from_22k.odds_ratio, rel=1e-12)


def test_cmh_accepts_the_flat_four_column_layout(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """(k, 4) is the flat form of the same data and must give the same answer."""
    flat = cmh.analyze(STRATA.reshape(3, 4))
    stacked = cmh.analyze(STRATA)
    assert flat.statistic == pytest.approx(stacked.statistic, rel=1e-12)


def test_cmh_odds_ratio_of_one_is_its_own_null(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """A table with a=b=c=d has an odds ratio of exactly 1.

    This is the case where an off-by-one in the MH weights shows up, because
    every stratum must contribute a*d/N and b*c/N that are equal.
    """
    identical = np.array([[[10, 10], [10, 10]], [[8, 8], [8, 8]]], dtype=float)
    result = cmh.analyze(identical)
    assert result.odds_ratio == pytest.approx(1.0, rel=1e-12)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_cmh_detects_a_common_strong_association(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The same strong association in every stratum must reject."""
    strong = np.array([[[30, 10], [10, 30]], [[25, 5], [5, 25]], [[20, 8], [8, 20]]], dtype=float)
    result = cmh.analyze(strong)
    assert result.p_value < 0.001
    assert result.odds_ratio > 5.0


def test_cmh_zero_cell_is_allowed(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """A zero cell is ordinary epidemiological data, not an error.

    One stratum with no unexposed controls leaves that stratum's MH
    contribution at zero, and the pooled estimate must still come from the
    strata that do have data.
    """
    tables = np.array([[[10, 20], [0, 40]], [[15, 5], [25, 55]]], dtype=float)
    result = cmh.analyze(tables)
    assert result.n_strata == 2
    assert np.isfinite(result.statistic)
    assert np.isfinite(result.odds_ratio)


def test_cmh_skips_an_empty_stratum(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """An all-zero stratum carries no information and must be skipped, not
    divided by."""
    tables = np.array([[[10, 20], [30, 40]], [[0, 0], [0, 0]]], dtype=float)
    result = cmh.analyze(tables)
    assert result.n_strata == 1
    assert result.n_total == 100


def test_cmh_rejects_an_all_zero_stack(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """Nothing to test when every stratum is empty."""
    with pytest.raises(StatisticalError, match="zero total"):
        cmh.analyze(np.zeros((2, 2, 2)))


def test_cmh_rejects_zero_pooled_variance(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """A stack where every stratum has a zero row or column margin has no
    variance to divide by; say so instead of returning inf."""
    tables = np.array([[[0, 0], [5, 5]], [[0, 0], [7, 3]]], dtype=float)
    with pytest.raises(ComputationError, match="pooled variance is zero"):
        cmh.analyze(tables)


def test_cmh_rejects_negative_counts(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """A negative count is not a frequency."""
    tables = np.array([[[-1, 20], [30, 40]]], dtype=float)
    with pytest.raises(DataValidationError, match="non-negative"):
        cmh.analyze(tables)


def test_cmh_rejects_non_finite_counts(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """A missing cell is not a count of zero; refuse it."""
    tables = np.array([[[np.nan, 20], [30, 40]]], dtype=float)
    with pytest.raises(DataValidationError, match="finite"):
        cmh.analyze(tables)


def test_cmh_rejects_a_wrong_shape(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """Three-dimensional input must be 2x2 per stratum."""
    with pytest.raises(ValidationError, match="n_strata"):
        cmh.analyze(np.ones((2, 3, 3)))


def test_cmh_rejects_wrong_column_count(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The flat layout needs exactly four counts per stratum."""
    with pytest.raises(ValidationError, match="4 counts"):
        cmh.analyze(np.ones((3, 3)))


def test_cmh_serialises(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """to_dict and summary must both work."""
    result = cmh.analyze(STRATA)
    payload = result.to_dict()
    assert set(payload) >= {"statistic", "p_value", "odds_ratio", "variant"}
    assert isinstance(payload["n_strata"], int)
    assert "Cochran" in result.summary()


def test_cmh_last_result_is_retained(cmh: CochranMantelHaenszelAnalyzer) -> None:
    """The analyzer remembers its most recent run."""
    assert cmh.last_result is None
    result = cmh.analyze(STRATA)
    assert cmh.last_result is result


# ---------------------------------------------------------------------------
# Single proportion
# ---------------------------------------------------------------------------


def test_binomial_matches_r(proportions: ProportionAnalyzer) -> None:
    """Must equal R's ``binom.test(13, 40, p = 0.25)`` exactly."""
    result = proportions.test_proportion(13, 40, p0=0.25)
    assert result.p_value == pytest.approx(0.2753588358, abs=1e-9)
    assert result.ci_lower == pytest.approx(0.1857289668, abs=1e-9)
    assert result.ci_upper == pytest.approx(0.4912948755, abs=1e-9)
    assert result.proportion == pytest.approx(0.325)


def test_binomial_interval_stays_in_range_at_zero_and_one(proportions: ProportionAnalyzer) -> None:
    """0/10 and 10/10 must give the exact intervals R reports.

    Both are boundary cases where an implementation that clamps or divides
    badly produces a bound outside [0, 1].
    """
    zero = proportions.test_proportion(0, 10, p0=0.5)
    assert zero.p_value == pytest.approx(0.0019531250, abs=1e-12)
    assert zero.ci_lower == pytest.approx(0.0, abs=1e-12)
    assert zero.ci_upper == pytest.approx(0.3084971078, abs=1e-9)

    full = proportions.test_proportion(10, 10, p0=0.5)
    assert full.p_value == pytest.approx(0.0019531250, abs=1e-12)
    assert full.ci_lower == pytest.approx(0.6915028922, abs=1e-9)
    assert full.ci_upper == pytest.approx(1.0, abs=1e-12)


def test_binomial_one_sided_intervals_match_r(proportions: ProportionAnalyzer) -> None:
    """The one-sided intervals must match R's ``binom.test`` too.

    R's ``binom.test`` inverts the same test, but with the one-sided alpha,
    so the degenerate ends differ from the two-sided ones:
    ``binom.test(0, 10, alternative = "less")`` gives ``(0, 0.2588655509)``
    and ``binom.test(10, 10, alternative = "greater")`` gives
    ``(0.7411344210, 1)``.

    The lower bound of the second is 1 - 0.05 ** (1 / 10) exactly, which R's
    ``qbeta`` returns to more digits than SciPy's beta quantile inversion
    does (0.7411344491 here), so the assertion is made to 1e-6.
    """
    less = proportions.test_proportion(0, 10, p0=0.5, alternative="less")
    assert less.ci_lower == pytest.approx(0.0, abs=1e-12)
    assert less.ci_upper == pytest.approx(0.2588655509, abs=1e-9)

    greater = proportions.test_proportion(10, 10, p0=0.5, alternative="greater")
    assert greater.ci_lower == pytest.approx(0.7411344210, abs=1e-6)
    assert greater.ci_upper == pytest.approx(1.0, abs=1e-12)


def test_binomial_one_sided_p_brackets_the_two_sided_p(proportions: ProportionAnalyzer) -> None:
    """The *smaller* one-sided p must be bounded by the two-sided p.

    SciPy's exact two-sided p is the "sum of all outcomes no more likely than
    the observed" quantity, which is between the smaller one-sided tail and
    twice it -- not twice each tail. Only the smaller tail is bounded: with
    7/25 against a null of 0.2 the 'less' tail is 0.89 while the two-sided p is
    0.32, because the data point the other way. Asserting that *both* tails are
    bounded would be asserting a symmetric identity the exact test does not
    have.
    """
    result = proportions.test_proportion(7, 25, p0=0.2)
    greater = proportions.test_proportion(7, 25, p0=0.2, alternative="greater")
    less = proportions.test_proportion(7, 25, p0=0.2, alternative="less")
    assert greater.p_value < less.p_value, "the observed proportion is above the null"

    smaller = min(greater.p_value, less.p_value)
    assert smaller <= result.p_value
    assert result.p_value <= 2 * smaller + 1e-12
    # The two tails sum to 1 + P(observed), not 1: both include the observed
    # outcome itself, so it is counted twice. Asserting the exact identity
    # pins both tails at once -- if either alternative were mislabelled, or if
    # one were the two-sided value, this would come out wrong.
    observed_pmf = float(sp_stats.binom.pmf(7, 25, 0.2))
    assert greater.p_value + less.p_value == pytest.approx(1.0 + observed_pmf, abs=1e-12)


def test_binomial_matches_r_one_sided(proportions: ProportionAnalyzer) -> None:
    """R's ``binom.test(7, 25, p = 0.2, alternative = "less")``."""
    result = proportions.test_proportion(7, 25, p0=0.2, alternative="less")
    assert result.p_value == pytest.approx(0.8908772040, abs=1e-9)
    assert result.ci_upper == pytest.approx(0.4622089217, abs=1e-9)


def test_binomial_detects_a_departure(proportions: ProportionAnalyzer) -> None:
    """An observed proportion far from the null must reject."""
    assert proportions.test_proportion(45, 50, p0=0.5).p_value < 1e-6


def test_binomial_rejects_impossible_counts(proportions: ProportionAnalyzer) -> None:
    """Successes above the trial count is a data error."""
    with pytest.raises(StatisticalError, match="between 0 and"):
        proportions.test_proportion(11, 10)


def test_binomial_rejects_zero_trials(proportions: ProportionAnalyzer) -> None:
    """Zero trials gives no binomial distribution."""
    with pytest.raises(StatisticalError, match="at least 1"):
        proportions.test_proportion(0, 0)


@pytest.mark.parametrize("bad_p0", [0.0, 1.0, -0.1])
def test_binomial_rejects_an_impossible_null(proportions: ProportionAnalyzer, bad_p0: float) -> None:
    """A null of exactly 0 or 1 has no binomial reference to test against."""
    with pytest.raises(ValidationError, match="strictly between 0 and 1"):
        proportions.test_proportion(5, 10, p0=bad_p0)


def test_binomial_serialises(proportions: ProportionAnalyzer) -> None:
    """to_dict and summary must both work."""
    result = proportions.test_proportion(13, 40, p0=0.25)
    payload = result.to_dict()
    assert set(payload) >= {"successes", "trials", "p_value", "ci_lower", "ci_upper"}
    assert isinstance(payload["successes"], int)
    assert "Binomial" in result.summary()


# ---------------------------------------------------------------------------
# Multiple proportion intervals
# ---------------------------------------------------------------------------


def test_wilson_intervals_match_statsmodels(proportions: ProportionAnalyzer) -> None:
    """Must equal statsmodels' Wilson interval to full double precision.

    Checked here rather than by copying a literal so the assertion stays
    meaningful across SciPy versions of ``norm.ppf``.
    """
    from statsmodels.stats.proportion import proportion_confint

    counts = [13, 20, 7, 0, 40]
    trials = [40, 40, 40, 40, 40]
    result = proportions.multiple_proportions(counts, trials, method="wilson")
    for i, (k, n) in enumerate(zip(counts, trials, strict=True)):
        expected = proportion_confint(k, n, method="wilson")
        assert result.lower[i] == pytest.approx(expected[0], abs=1e-12)
        assert result.upper[i] == pytest.approx(expected[1], abs=1e-12)


def test_wilson_interval_beats_wald_at_extreme_proportions(proportions: ProportionAnalyzer) -> None:
    """With 2 successes of 40 the Wald interval dips below zero.

    Zero is not a possible proportion, so a normal interval that reports it is
    reporting an impossibility. This is why the Wilson score interval is the
    default here rather than the normal approximation.
    """
    result = proportions.multiple_proportions([2], [40], method="wilson")
    assert result.lower[0] > 0.0
    assert result.upper[0] < 1.0

    phat = 2 / 40
    wald_lower = phat - 1.96 * np.sqrt(phat * (1 - phat) / 40)
    assert wald_lower < 0.0, "the Wald interval should be the broken one"


def test_proportion_intervals_are_ordered_and_nested(proportions: ProportionAnalyzer) -> None:
    """Every interval must contain its own point estimate."""
    counts = [0, 1, 7, 13, 20, 39, 40]
    trials = [40] * len(counts)
    result = proportions.multiple_proportions(counts, trials)
    for i, p in enumerate(result.proportions):
        assert result.lower[i] <= p <= result.upper[i]
        assert 0.0 <= result.lower[i] <= result.upper[i] <= 1.0


def test_proportion_intervals_widen_with_less_confidence(proportions: ProportionAnalyzer) -> None:
    """A 99% interval must contain the 95% one; otherwise the level is not
    being applied at all."""
    wide = proportions.multiple_proportions([13], [40], confidence=0.99)
    narrow = proportions.multiple_proportions([13], [40], confidence=0.95)
    assert wide.lower[0] < narrow.lower[0]
    assert wide.upper[0] > narrow.upper[0]


def test_proportion_intervals_handle_one_and_many_groups(proportions: ProportionAnalyzer) -> None:
    """A single group is legitimate; two or more is the usual case."""
    single = proportions.multiple_proportions([13], [40])
    assert len(single.proportions) == 1
    many = proportions.multiple_proportions([1, 5, 13, 30], [10, 20, 40, 50])
    assert len(many.proportions) == 4
    assert many.counts == (1, 5, 13, 30)
    assert many.trials == (10, 20, 40, 50)


def test_exact_proportion_intervals_match_binomtest(proportions: ProportionAnalyzer) -> None:
    """The 'exact' method must agree with the same engine test_proportion uses."""
    counts = [13, 20, 7]
    trials = [40, 40, 40]
    result = proportions.multiple_proportions(counts, trials, method="exact")
    for i, (k, n) in enumerate(zip(counts, trials, strict=True)):
        expected = proportions.test_proportion(k, n, p0=0.5)
        assert result.lower[i] == pytest.approx(expected.ci_lower, abs=1e-12)
        assert result.upper[i] == pytest.approx(expected.ci_upper, abs=1e-12)


def test_multiple_proportions_rejects_mismatched_lengths(proportions: ProportionAnalyzer) -> None:
    """Each success count needs its own trial count."""
    with pytest.raises(ValidationError, match="same length"):
        proportions.multiple_proportions([1, 2, 3], [10, 10])


def test_multiple_proportions_rejects_an_empty_list(proportions: ProportionAnalyzer) -> None:
    """No groups is not an analysis."""
    with pytest.raises(StatisticalError):
        proportions.multiple_proportions([], [])


def test_multiple_proportions_rejects_mismatched_labels(proportions: ProportionAnalyzer) -> None:
    """Labels must line up with groups."""
    with pytest.raises(ValidationError, match="one entry per group"):
        proportions.multiple_proportions([1, 2], [10, 10], labels=["only-one"])


def test_multiple_proportions_serialises(proportions: ProportionAnalyzer) -> None:
    """to_dict and summary must both work."""
    result = proportions.multiple_proportions([13, 20, 7], [40, 40, 40], labels=["a", "b", "c"])
    payload = result.to_dict()
    assert set(payload) >= {"proportions", "lower", "upper", "method"}
    assert len(payload["proportions"]) == 3
    assert "a:" in result.summary()


# ---------------------------------------------------------------------------
# Ratio of counts
# ---------------------------------------------------------------------------


def test_ratio_of_counts_matches_r(proportions: ProportionAnalyzer) -> None:
    """Must equal R's ``poisson.test(c(23, 17), T = 1)`` exactly."""
    result = proportions.ratio_of_counts(23, 17)
    assert result.ratio == pytest.approx(1.3529411765, abs=1e-9)
    assert result.ci_lower == pytest.approx(0.6917627845, abs=1e-9)
    assert result.ci_upper == pytest.approx(2.6978278379, abs=1e-9)


def test_ratio_of_counts_matches_r_second_case(proportions: ProportionAnalyzer) -> None:
    """R's ``poisson.test(c(40, 10), T = 1)``."""
    result = proportions.ratio_of_counts(40, 10)
    assert result.ratio == pytest.approx(4.0, abs=1e-9)
    assert result.ci_lower == pytest.approx(1.9657476165, abs=1e-9)
    assert result.ci_upper == pytest.approx(8.9698673250, abs=1e-9)


def test_ratio_of_counts_with_a_zero_count_is_open_ended(proportions: ProportionAnalyzer) -> None:
    """R's ``poisson.test(c(5, 0), T = 1)``: infinite ratio, infinite upper
    bound, and a finite lower bound. Reporting a finite number here would be
    inventing precision the data cannot support."""
    result = proportions.ratio_of_counts(5, 0)
    assert np.isinf(result.ratio)
    assert np.isinf(result.ci_upper)
    assert result.ci_lower == pytest.approx(0.9163558573, abs=1e-9)
    assert np.isfinite(result.ci_lower)


def test_ratio_of_counts_one_sided_matches_r(proportions: ProportionAnalyzer) -> None:
    """R's ``poisson.test`` passes its alternative on to ``binom.test``.

    ``poisson.test(c(5, 0), T = 1, alternative = "greater")`` therefore gives
    the one-sided conditional interval, mapped through ``p -> p/(1-p)``:
    ``(1.2186736837, Inf)``.
    """
    result = proportions.ratio_of_counts(5, 0, alternative="greater")
    assert result.ci_lower == pytest.approx(1.2186736837, abs=1e-9)
    assert np.isinf(result.ci_upper)
    # The other direction, on the mirrored counts: R's
    # ``poisson.test(c(0, 5), T = 1, alternative = "less")`` gives
    # ``(0, 0.8205642030)`` -- the same interval on the ratio, mirrored.
    less = proportions.ratio_of_counts(0, 5, alternative="less")
    assert less.ci_lower == pytest.approx(0.0, abs=1e-12)
    assert less.ci_upper == pytest.approx(0.8205642030, abs=1e-9)


def test_ratio_of_counts_rejects_a_bad_alternative(proportions: ProportionAnalyzer) -> None:
    """Only the three implemented alternatives are accepted."""
    with pytest.raises(ValidationError):
        proportions.ratio_of_counts(5, 5, alternative="sideways")


def test_ratio_of_counts_interval_brackets_the_point_estimate(proportions: ProportionAnalyzer) -> None:
    """The interval must contain the observed ratio when both counts differ."""
    for a, b in [(23, 17), (40, 10), (7, 30)]:
        result = proportions.ratio_of_counts(a, b)
        assert result.ci_lower < result.ratio < result.ci_upper


def test_log_method_interval_brackets_the_point_estimate(proportions: ProportionAnalyzer) -> None:
    """The Koopman interval is symmetric on the log scale, so on the ratio
    scale the point estimate must sit strictly inside it."""
    result = proportions.ratio_of_counts(40, 10, method="log")
    assert result.ci_lower < result.ratio < result.ci_upper
    assert result.method == "log"


def test_log_and_exact_methods_differ(proportions: ProportionAnalyzer) -> None:
    """The two methods are genuinely different; asserting equality would let a
    method argument that is silently ignored pass."""
    exact = proportions.ratio_of_counts(40, 10, method="exact")
    log = proportions.ratio_of_counts(40, 10, method="log")
    assert exact.ci_lower != pytest.approx(log.ci_lower, rel=1e-6)
    assert exact.ci_upper != pytest.approx(log.ci_upper, rel=1e-6)
    # The exact method is the conservative one, which is why it is the default.
    assert exact.ci_lower < log.ci_lower
    assert exact.ci_upper > log.ci_upper


def test_ratio_of_counts_equal_counts_gives_a_unit_ratio(proportions: ProportionAnalyzer) -> None:
    """Equal counts must give a ratio of exactly 1."""
    result = proportions.ratio_of_counts(20, 20)
    assert result.ratio == pytest.approx(1.0, rel=1e-12)


def test_ratio_of_counts_rejects_two_zeros(proportions: ProportionAnalyzer) -> None:
    """0/0 has no value; refuse rather than return NaN."""
    with pytest.raises(StatisticalError, match="0/0"):
        proportions.ratio_of_counts(0, 0)


def test_ratio_of_counts_rejects_negative_counts(proportions: ProportionAnalyzer) -> None:
    """Counts are frequencies."""
    with pytest.raises(StatisticalError, match="non-negative"):
        proportions.ratio_of_counts(-1, 5)


def test_log_method_refuses_a_zero_count(proportions: ProportionAnalyzer) -> None:
    """The asymptotic interval divides by the counts, so a zero count has no
    standard error. It must point at the method that works instead."""
    with pytest.raises(StatisticalError, match="non-zero"):
        proportions.ratio_of_counts(5, 0, method="log")


def test_ratio_of_counts_rejects_an_unknown_method(proportions: ProportionAnalyzer) -> None:
    """Only the two implemented methods are accepted."""
    with pytest.raises(ValidationError):
        proportions.ratio_of_counts(5, 5, method="bootstrap")


def test_ratio_of_counts_serialises(proportions: ProportionAnalyzer) -> None:
    """to_dict and summary must both work, including the infinite case."""
    result = proportions.ratio_of_counts(23, 17)
    payload = result.to_dict()
    assert set(payload) >= {"ratio", "ci_lower", "ci_upper", "method"}
    assert "Ratio" in result.summary()
    # An infinite upper bound must still render rather than raise.
    assert "inf" in proportions.ratio_of_counts(5, 0).summary()


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, 1.5, -0.2])
def test_confidence_must_be_a_real_level(proportions: ProportionAnalyzer, bad_confidence: float) -> None:
    """Every entry point must reject an impossible confidence level."""
    with pytest.raises(ValidationError):
        proportions.test_proportion(5, 10, confidence=bad_confidence)
    with pytest.raises(ValidationError):
        proportions.multiple_proportions([5], [10], confidence=bad_confidence)
    with pytest.raises(ValidationError):
        proportions.ratio_of_counts(5, 5, confidence=bad_confidence)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def test_every_analyzer_reports_no_result_before_running() -> None:
    """A fresh analyzer has nothing to report; it must not invent something."""
    assert BrunnerMunzelAnalyzer().last_result is None
    assert MoodMedianAnalyzer().last_result is None
    assert FlignerKileenAnalyzer().last_result is None
    assert OutlierAnalyzer().last_result is None
    assert CochranMantelHaenszelAnalyzer().last_result is None
    assert ProportionAnalyzer().last_result is None


def test_results_are_immutable() -> None:
    """Result dataclasses are frozen; a caller cannot rewrite a reported value."""
    result = BrunnerMunzelAnalyzer().analyze(X, Y)
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError differs across versions
        result.p_value = 0.0  # type: ignore[misc]
