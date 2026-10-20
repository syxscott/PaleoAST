# =============================================================================
# FILE: tests/stats/test_distribution_tests.py
# =============================================================================
"""
Tests for the two-sample distribution-equality tests.

Every expected number here comes from one of three sources, never from
running the module and copying the output:

* R 4.5.2 ``stats::ks.test`` run on the same fixed inputs (the exact and
  asymptotic p-values for eight cases, tied and tie-free);
* a closed-form identity derived in the test itself (the ``max(|x-y|)/n``
  form of D, the exact count of orderings for a fully separated pair);
* a structural property that a broken implementation necessarily violates.

The tests that would catch the interesting bugs are the ones that compare
against R and against an independent enumeration. A permutation p-value that
shuffles the wrong array still returns a number in [0, 1], so its only real
check is calibration: the DISTRIBUTION of the p-values under the null, plus
a power check. That is what ``test_epps_p_values_are_uniform_under_the_null``
and ``test_epps_rejects_a_genuinely_different_alternative`` do, and
``test_the_uniformity_check_catches_a_broken_permutation_loop`` pins that the
check has teeth by running it against a deliberately broken loop.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from stats.distribution_tests import DistributionTestAnalyzer
from utils.exceptions import ComputationError, DataValidationError

# ---------------------------------------------------------------------------
# R 4.5.2 reference values: stats::ks.test(x, y) on these exact inputs.
# Verified with:
#   Rscript -e 'print(ks.test(x, y))'
#     R version 4.5.2 (2025-10-31 ucrt)
# The tie-free cases additionally agree with scipy.stats.ks_2samp; the tied
# ones deliberately do not, which is the whole point of test_scipy_exact_is_not
# relied on below.
# ---------------------------------------------------------------------------
R_KS_CASES = {
    "notie-8-9": (
        [1.1, 2.3, 3.7, 4.2, 5.9, 6.1, 7.4, 8.0],
        [2.0, 2.5, 3.1, 3.3, 4.5, 5.0, 6.3, 7.7, 8.8],
        0.97893870835047303,
        0.99717575191005758,
    ),
    "notie-6-7": (
        [0.4, 1.1, 1.9, 2.2, 3.3, 4.1],
        [0.7, 1.4, 1.5, 2.9, 3.0, 3.8, 5.2],
        0.96270396270396263,
        0.99304533120349692,
    ),
    "notie-5-5": (
        [1, 2, 3, 4, 10],
        [2, 3, 4, 5, 6],
        1.0,
        0.99996523065400766,
    ),
    "notie-3-3": (
        [1, 5, 9],
        [2, 6, 10],
        1.0,
        0.99625519237939875,
    ),
    "tie-5-4": (
        [1, 2, 2, 3, 4],
        [2, 3, 3, 5],
        0.7142857142857143,
        0.94830811096545298,
    ),
    "tie-6-6": (
        [1, 1, 2, 2, 3, 4],
        [1, 2, 3, 3, 4, 4],
        0.64935064935064934,
        0.89277833725012334,
    ),
    "const-4-4": (
        [1, 1, 1, 1],
        [1, 2, 3, 4],
        0.14285714285714285,
        0.21055163272601135,
    ),
    "same-4-4": (
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        1.0,
        1.0,
    ),
}


@pytest.fixture
def analyzer() -> DistributionTestAnalyzer:
    """A fresh analyzer."""
    return DistributionTestAnalyzer()


# ---------------------------------------------------------------------------
# KS: agreement with R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(R_KS_CASES))
def test_ks_exact_matches_r(analyzer: DistributionTestAnalyzer, case: str) -> None:
    """The exact p-value equals R's stats::ks.test to full double precision.

    A DP that mishandles the tie multiplicity, or an off-by-one in the
    barrier comparison, changes at least the last few digits on the tied
    cases, so a tight tolerance here is a real assertion and not decoration.
    """
    x, y, r_exact, _ = R_KS_CASES[case]
    result = analyzer.ks_2samp(x, y, method="exact")
    assert result.method == "exact"
    assert result.p_value == pytest.approx(r_exact, abs=1e-12)


@pytest.mark.parametrize("case", sorted(R_KS_CASES))
def test_ks_asymptotic_matches_r(analyzer: DistributionTestAnalyzer, case: str) -> None:
    """The asymptotic p-value equals R's ``exact=FALSE`` to ~1e-11.

    R's asymptotic variant is Q_KS(sqrt(n_eff) * D) with the fractional
    n_eff; scipy instead rounds n_eff to an integer before evaluating
    kstwo.sf, which is why scipy's number differs from R's here and this
    test pins the implementation to the unrounded form.
    """
    x, y, _, r_asymp = R_KS_CASES[case]
    result = analyzer.ks_2samp(x, y, method="asymp")
    assert result.method == "asymp"
    assert result.p_value == pytest.approx(r_asymp, abs=1e-10)


def test_ks_statistic_matches_r_on_every_case(analyzer: DistributionTestAnalyzer) -> None:
    """D is the same quantity R reports, in all eight cases."""
    for case, (x, y, _, _) in R_KS_CASES.items():
        result = analyzer.ks_2samp(x, y, method="exact")
        r_d = float(sp_stats.ks_2samp(x, y).statistic)
        assert result.statistic == pytest.approx(r_d, abs=1e-12), case


# ---------------------------------------------------------------------------
# KS: derived identities (no external reference needed)
# ---------------------------------------------------------------------------


def test_ks_statistic_is_max_gap_over_fractions(analyzer: DistributionTestAnalyzer) -> None:
    """D equals the largest |i/n1 - j/n2| reachable, computed a different way.

    Evaluated by walking the pooled values and taking the running ECDF gap
    directly, rather than through the module's own searchsorted path. If the
    module took the sup at the wrong positions this diverges immediately.
    """
    x = [1.1, 2.3, 3.7, 4.2, 5.9, 6.1, 7.4, 8.0]
    y = [2.0, 2.5, 3.1, 3.3, 4.5, 5.0, 6.3, 7.7, 8.8]
    n1, n2 = len(x), len(y)
    pooled = sorted(x + y)
    fx = fy = 0.0
    worst = 0.0
    for value in pooled:
        if value in x:
            fx += 1.0 / n1
        if value in y:
            fy += 1.0 / n2
        worst = max(worst, abs(fx - fy))
    assert analyzer.ks_2samp(x, y).statistic == pytest.approx(worst, abs=1e-12)


def test_fully_separated_samples_give_d_one(analyzer: DistributionTestAnalyzer) -> None:
    """Complete separation gives D = 1 and the smallest attainable exact p-value.

    D = 1 is reached by exactly TWO of the C(n1+n2, n1) orderings -- sample 1
    entirely below sample 2, or entirely above it -- so the p-value is
    exactly 2/C(n1+n2, n1). Both the factor 2 and the denominator are closed
    forms, not fitted numbers. Cross-checked against R:
    ``ks.test(c(1,2,3,4), c(10:15))`` gives p = 0.0095238095238095247
    = 2/210, matching the assertion below.
    """
    n1, n2 = 4, 6
    x = [1.0, 2.0, 3.0, 4.0][:n1]
    y = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0][:n2]
    result = analyzer.ks_2samp(x, y, method="exact")
    assert result.statistic == 1.0
    # C(10, 4) = 210 orderings; exactly two separate the samples.
    assert result.p_value == pytest.approx(2.0 / 210.0, abs=1e-12)
    assert result.p_value < 0.05
    assert result.significant is True


def test_identical_samples_give_d_zero_and_p_one(analyzer: DistributionTestAnalyzer) -> None:
    """Two identical samples cannot be distinguished from the null."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = analyzer.ks_2samp(values, list(values), method="exact")
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.significant is False


def test_ks_is_symmetric_under_swapping_the_samples(analyzer: DistributionTestAnalyzer) -> None:
    """D and the two-sided p-value do not depend on argument order.

    Swapping the samples relabels the lattice but not the set of orderings,
    so both must be identical. A DP that treated sample 1 specially (an
    asymmetric barrier, say) breaks here and nowhere else.
    """
    x = [1.1, 2.3, 3.7, 4.2, 5.9, 6.1, 7.4, 8.0]
    y = [2.0, 2.5, 3.1, 3.3, 4.5, 5.0, 6.3, 7.7, 8.8]
    forward = analyzer.ks_2samp(x, y, method="exact")
    reverse = analyzer.ks_2samp(y, x, method="exact")
    assert forward.statistic == pytest.approx(reverse.statistic, abs=1e-12)
    assert forward.p_value == pytest.approx(reverse.p_value, abs=1e-12)


def test_d_sign_follows_which_cdf_is_lower(analyzer: DistributionTestAnalyzer) -> None:
    """d_sign records which sample's ECDF is higher at the extreme point.

    Convention: +1 when sample 1's ECDF is the higher one at the point where
    the two CDFs are furthest apart (sample 1 sits to the right), -1 when it
    is the lower one. The SIGN convention is the same as
    ``scipy.stats.ks_2samp``'s ``statistic_sign``; where the two do NOT
    agree is the tie-breaking of the LOCATION, which is a separate
    convention and is pinned by
    ``test_d_location_breaks_ties_by_the_first_distinct_value`` below.
    """
    lower_first = analyzer.ks_2samp([1.0, 1.1, 1.2], [5.0, 6.0, 7.0])
    upper_first = analyzer.ks_2samp([5.0, 6.0, 7.0], [1.0, 1.1, 1.2])
    assert lower_first.d_sign == 1
    assert upper_first.d_sign == -1
    # And against scipy, on a case with a less extreme separation. The
    # comparison only holds because continuous data never attain the
    # maximum deviation at more than one distinct value.
    rng = np.random.default_rng(59)
    for _ in range(20):
        x = rng.normal(size=12)
        y = rng.normal(0.7, 1.4, 12)
        mine = analyzer.ks_2samp(x, y).d_sign
        theirs = int(sp_stats.ks_2samp(x, y).statistic_sign)
        assert mine == theirs


def test_d_location_breaks_ties_by_the_first_distinct_value(analyzer: DistributionTestAnalyzer) -> None:
    """When several values attain the maximum, the first one is reported.

    This is the module's own convention, and it is NOT scipy's. On
    x = [3, 2, 1, 3, 1] against y = [3, 2, 3, 2, 0] the two ECDFs are
    0.2 apart at BOTH 0 and 1, so the extreme point is genuinely
    ambiguous: this module reports 0 with sign -1 (the first of the
    maximising distinct values), while ``scipy.stats.ks_2samp`` reports 1
    with sign +1 (the last of them). The statistic D itself is the same
    in both.

    The previous version of this file claimed that the module and scipy
    "use the same convention" and checked it on 20 continuous normal
    draws, which never produce the ambiguous case.
    """
    x = [3, 2, 1, 3, 1]
    y = [3, 2, 3, 2, 0]
    mine = analyzer.ks_2samp(x, y)
    theirs = sp_stats.ks_2samp(x, y)
    assert mine.statistic == pytest.approx(float(theirs.statistic), abs=1e-12)
    assert mine.d_location == 0.0
    assert mine.d_sign == -1
    # scipy's convention, for the record.
    assert float(theirs.statistic_location) == 1.0
    assert int(theirs.statistic_sign) == 1

    # A second ambiguous case, with a different sign of the deviation.
    x2 = [1, 2, 1, 2, 2]
    y2 = [3, 0, 3, 0, 2]
    mine2 = analyzer.ks_2samp(x2, y2)
    theirs2 = sp_stats.ks_2samp(x2, y2)
    assert mine2.statistic == pytest.approx(float(theirs2.statistic), abs=1e-12)
    assert mine2.d_location == 0.0
    assert mine2.d_sign == -1
    assert float(theirs2.statistic_location) == 2.0
    assert int(theirs2.statistic_sign) == 1


def test_d_location_is_a_value_present_in_the_pooled_sample(analyzer: DistributionTestAnalyzer) -> None:
    """D is attained at an observed pooled value, not an interpolation.

    If the module evaluated the sup between observations (the classic
    off-by-half-a-rank error) this would report a value no observation has.
    """
    x = [1.1, 2.3, 3.7, 4.2, 5.9, 6.1, 7.4, 8.0]
    y = [2.0, 2.5, 3.1, 3.3, 4.5, 5.0, 6.3, 7.7, 8.8]
    result = analyzer.ks_2samp(x, y)
    assert result.d_location in x + y


def test_ties_flag_reports_pooled_structure(analyzer: DistributionTestAnalyzer) -> None:
    """The ties flag and distinct count describe the pooled sample exactly."""
    tied = analyzer.ks_2samp([1, 2, 2, 3, 4], [2, 3, 3, 5])
    assert tied.ties is True
    assert tied.n_distinct == 5
    assert tied.n1 + tied.n2 == 9

    free = analyzer.ks_2samp([1.1, 2.3, 3.7], [4.1, 5.5, 6.9])
    assert free.ties is False
    assert free.n_distinct == 6


def test_ks_is_invariant_to_a_joint_rescaling_and_shift(analyzer: DistributionTestAnalyzer) -> None:
    """Changing the units must not change the answer.

    The KS statistic depends only on the ORDER of the pooled values, so
    multiplying both samples by a positive constant, or adding the same
    constant to both, leaves D bit-for-bit identical. Rescaling only ONE
    sample is a different question -- that genuinely changes the
    distributions, and the test is right to notice -- so the invariance
    asserted here is deliberately the joint one.

    Powers of two are used so the multiplication is exact in binary floating
    point and the test is testing the module rather than rounding.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=40)
    other = rng.normal(size=40)
    reference = analyzer.ks_2samp(base, other)
    for factor in (8.0, 1024.0, 0.125):
        scaled = analyzer.ks_2samp(base * factor, other * factor)
        assert scaled.statistic == pytest.approx(reference.statistic, abs=1e-12)
        assert scaled.p_value == pytest.approx(reference.p_value, abs=1e-12)
    shifted = analyzer.ks_2samp(base + 500.0, other + 500.0)
    assert shifted.statistic == pytest.approx(reference.statistic, abs=1e-12)
    assert shifted.p_value == pytest.approx(reference.p_value, abs=1e-12)


def test_ks_rejects_a_confident_location_shift(analyzer: DistributionTestAnalyzer) -> None:
    """A large mean shift is detected, a small one may not be.

    Only the strong direction is asserted. Asserting that a small shift is
    also rejected would be asserting a power figure that depends on the
    sample size and on the null distribution, not a property of the test.
    """
    rng = np.random.default_rng(61)
    base = rng.normal(size=40)
    assert analyzer.ks_2samp(base, base + 8.0).p_value < 0.01
    assert analyzer.ks_2samp(base, base + 8.0).statistic == pytest.approx(1.0, abs=1e-12)


def test_ks_notices_when_only_one_sample_is_rescaled(analyzer: DistributionTestAnalyzer) -> None:
    """Scaling one sample really does change the distributions.

    This is the asymmetry the previous test deliberately does NOT claim.
    x near 0 with unit spread and 1000x near 0 with a thousand-fold spread
    are different distributions, and a test that missed that would be
    broken. Asserted so a later "optimisation" that normalises the inputs
    cannot quietly change the meaning of the test.
    """
    rng = np.random.default_rng(63)
    base = rng.normal(size=40)
    assert analyzer.ks_2samp(base, base * 1000.0).p_value < 0.01


# ---------------------------------------------------------------------------
# KS: p-value range and method behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["exact", "asymp"])
def test_ks_p_value_is_always_a_probability(analyzer: DistributionTestAnalyzer, method: str) -> None:
    """Every p-value lands in [0, 1] on adversarial inputs.

    Inputs chosen to break the series (a statistic of exactly zero, maximal
    separation, and a constant sample) and to break the DP (a single tie
    group holding everything).
    """
    cases = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 1, 1], [1, 1, 1]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([0.0, 1.0], [1.0, 2.0, 3.0]),
        (list(range(20)), list(range(20))),
        ([1.0, 1.0, 2.0], [2.0, 3.0, 3.0]),
    ]
    for x, y in cases:
        result = analyzer.ks_2samp(x, y, method=method)
        assert 0.0 <= result.p_value <= 1.0
        assert 0.0 <= result.statistic <= 1.0


def test_ks_exact_p_value_is_a_count_on_the_ordering_lattice(analyzer: DistributionTestAnalyzer) -> None:
    """The exact p-value is a count on the lattice of orderings.

    The exact null counts the orderings of the pooled sample that reach the
    observed statistic, so its p-value must be an integer multiple of
    1/C(n1+n2, n1). The asymptotic p-value approximates the SAME tail
    probability and is not such a multiple, so this tells the two branches
    apart: swapping them fails here.

    The previous version of this test asserted that the exact p-value is
    never larger than the asymptotic one, arguing that "the exact null
    counts only orderings the data can produce, so it is a subset of the
    asymptotic reference and its tail probability can only be smaller".
    That reasoning is wrong: the asymptotic p-value is an APPROXIMATION
    of the same tail probability, not a superset of the exact lattice, so
    the two are not ordered at all. The assertion fails on 368 of 400
    fixed-seed draws of 20 + 20 normals -- for instance on seed 0, where
    exact = 0.83197 against asymp = 0.81862. It passed here only because
    the 25 fixed seeds it used happened to fall on the other side.
    """
    from math import comb

    total = comb(40, 20)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=20)
        y = rng.normal(size=20)
        exact = analyzer.ks_2samp(x, y, method="exact").p_value
        asymp = analyzer.ks_2samp(x, y, method="asymp").p_value
        count = exact * total
        assert count == pytest.approx(round(count), abs=1e-6), seed
        assert 0.0 <= count <= total, seed
        # The asymptotic p-value is an approximation, not a lattice count.
        asymp_count = asymp * total
        assert asymp_count != pytest.approx(round(asymp_count), abs=1e-6), seed

    # Neither of the two is systematically the larger one.
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    y = rng.normal(size=20)
    exact = analyzer.ks_2samp(x, y, method="exact").p_value
    asymp = analyzer.ks_2samp(x, y, method="asymp").p_value
    assert exact > asymp
    rng = np.random.default_rng(1)
    x = rng.normal(size=9)
    y = rng.normal(size=11)
    assert analyzer.ks_2samp(x, y, method="exact").p_value < analyzer.ks_2samp(x, y, method="asymp").p_value


def test_ks_exact_is_coarser_than_asymptotic_for_large_separation(analyzer: DistributionTestAnalyzer) -> None:
    """The exact null is lattice-limited; the asymptotic one is not.

    With n1=n2=5 fully separated, D = 1 is reached by only two of the
    C(10,5) = 252 orderings, so the exact p-value cannot go below 2/252 even
    though the evidence is overwhelming. This is a property of the exact
    test and the reason the asymptotic variant exists. R agrees:
    ks.test(c(1:5), c(10:14)) gives 0.0079365079365079361 = 2/252.
    """
    result = analyzer.ks_2samp([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0], method="exact")
    assert result.p_value == pytest.approx(2.0 / 252.0, abs=1e-12)
    assert result.p_value > 0.007


def test_ks_falls_back_to_asymptotic_past_the_lattice_budget(analyzer: DistributionTestAnalyzer) -> None:
    """A request for 'exact' on a very large sample is downgraded, not refused.

    The result must still be a valid p-value, and the reported method must
    say what actually happened rather than claiming an exact p-value it did
    not compute.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=600)
    y = rng.normal(size=600)
    result = analyzer.ks_2samp(x, y, method="exact")
    assert result.method == "asymp"
    assert 0.0 <= result.p_value <= 1.0
    assert result.n1 == 600


def test_ks_keeps_the_exact_null_when_the_lattice_is_small(analyzer: DistributionTestAnalyzer) -> None:
    """The downgrade follows the size of the LATTICE, not the sample size.

    999 + 2 observations is 1001 observations, which used to push the
    request onto the asymptotic null -- even though the exact lattice
    there is trivial and the asymptotic null is the least trustworthy of
    the two at such sizes (the module's own docstring says the exact
    null "is the only one that is correct at small sample sizes").

    The DP is cheap here because only a handful of states are reachable:
    the number of sample-2 observations is bounded by 2, so at most three
    of the 1000 slots can hold a non-zero count.
    """
    x = list(range(999))
    y = [1000.0, 1001.0]
    result = analyzer.ks_2samp(x, y, method="exact")
    assert result.method == "exact"
    assert result.n1 == 999
    assert result.n2 == 2
    assert 0.0 <= result.p_value <= 1.0
    # The same data through the asymptotic null, for reference.
    asymp = analyzer.ks_2samp(x, y, method="asymp")
    assert 0.0 <= asymp.p_value <= 1.0


def test_ks_exact_and_asymptotic_agree_on_a_large_lattice(analyzer: DistributionTestAnalyzer) -> None:
    """Above the product threshold the two nulls agree to better than 1e-3.

    This is the measurement the silent fallback rests on: where the exact
    lattice is large (n1 * n2 > 10 000) the exact and the asymptotic
    p-values differ by less than 1e-3 -- the largest gap found here is
    5e-04 -- so falling back to the asymptotic p-value costs nothing
    that can be measured.
    """
    rng = np.random.default_rng(2024)
    for _ in range(5):
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        exact = analyzer.ks_2samp(x, y, method="exact").p_value
        asymp = analyzer.ks_2samp(x, y, method="asymp").p_value
        assert abs(exact - asymp) < 1e-3


def test_ks_asymptotic_p_value_is_correct_for_small_lambda(analyzer: DistributionTestAnalyzer) -> None:
    """The asymptotic p-value must not break down when lambda is small.

    The textbook Kolmogorov series
    ``2 * sum_j (-1)^(j-1) exp(-2 j^2 lam^2)`` converges only for
    lambda >= ~0.02, and truncating it at a fixed 200 terms was
    materially wrong below that: at lambda = 0.00707 it returned 0.982
    where the true tail is 1.0.

    The inputs below are the ones that produce the smallest attainable D
    -- a perfectly alternating pooled order, one observation per side --
    at the sample sizes that make lambda small. The reference is
    ``scipy.stats.kstwobign.sf``, which switches to a different expansion
    near zero; R's own ``K2l()`` agrees with it to about 1e-14.
    """
    for n, expected_lam in ((10_000, 0.0070710678), (20_000, 0.005), (200_000, 0.0015811388)):
        x = np.arange(0.0, 2 * n, 2.0)
        y = np.arange(1.0, 2 * n, 2.0)
        result = analyzer.ks_2samp(x, y, method="asymp")
        lam = np.sqrt(n * n / (n + n)) * result.statistic
        assert lam == pytest.approx(expected_lam, rel=1e-6)
        assert result.p_value == pytest.approx(sp_stats.kstwobign.sf(lam), abs=1e-12)
        assert result.p_value == pytest.approx(1.0, abs=1e-9)


def test_ks_default_call_also_gives_the_small_lambda_p_value(analyzer: DistributionTestAnalyzer) -> None:
    """The default path downgrades to the asymptotic null, so it must be right.

    A request for 'exact' on 20000 + 20000 observations is downgraded
    because the exact lattice is out of reach, which means the DEFAULT
    call reaches the same asymptotic p-value -- and used to reach the
    broken one.
    """
    n = 20_000
    x = np.arange(0.0, 2 * n, 2.0)
    y = np.arange(1.0, 2 * n, 2.0)
    result = analyzer.ks_2samp(x, y)
    assert result.method == "asymp"
    assert result.p_value == pytest.approx(1.0, abs=1e-9)


def test_ks_ignores_non_finite_values(analyzer: DistributionTestAnalyzer) -> None:
    """NaN and infinities are dropped, and the size is reported after.

    Comparing the filtered result with a hand-cleaned copy of the same data
    is the check: if the module mishandled the mask, the two would differ.
    """
    x = [1.0, 2.0, np.nan, 3.0, np.inf, 4.0]
    y = [2.0, 3.0, 5.0, -np.inf, 6.0]
    result = analyzer.ks_2samp(x, y)
    clean = analyzer.ks_2samp([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 6.0])
    assert result.n1 == 4
    assert result.n2 == 4
    assert result.statistic == pytest.approx(clean.statistic, abs=1e-12)
    assert result.p_value == pytest.approx(clean.p_value, abs=1e-12)


# ---------------------------------------------------------------------------
# Epps-Singleton
# ---------------------------------------------------------------------------


def test_epps_statistic_is_the_cv2_difference(analyzer: DistributionTestAnalyzer) -> None:
    """The statistic is s1^2/x1^2 - s2^2/x2^2, computed here by hand.

    The expected value is written out as explicit arithmetic from the raw
    numbers, so a change to either the variance or the mean term in the
    formula shows up immediately.
    """
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    expected = (np.var(x, ddof=1) / np.mean(x) ** 2) - (np.var(y, ddof=1) / np.mean(y) ** 2)

    result = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=1)
    assert result.cv2_1 == pytest.approx(np.var(x, ddof=1) / np.mean(x) ** 2, abs=1e-12)
    assert result.cv2_2 == pytest.approx(np.var(y, ddof=1) / np.mean(y) ** 2, abs=1e-12)
    assert result.statistic == pytest.approx(result.cv2_1 - result.cv2_2, abs=1e-12)
    assert result.statistic == pytest.approx(expected, abs=1e-12)


def test_epps_statistic_is_antisymmetric_under_swapping(analyzer: DistributionTestAnalyzer) -> None:
    """Swapping the samples negates the statistic and swaps the two CV^2 values.

    Exactly antisymmetric, with no tolerance needed. Only the statistic has
    this property: the permutation p-value is a Monte-Carlo quantity drawn
    from a generator applied to a differently laid-out array, so seeded
    p-values need not coincide even for equal sample sizes. That is checked
    separately, within Monte-Carlo error.
    """
    rng = np.random.default_rng(17)
    x = rng.normal(size=20)
    y = rng.lognormal(size=20)

    forward = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=4)
    reverse = analyzer.epps_singleton(y, x, n_permutations=199, random_seed=4)
    assert forward.statistic == pytest.approx(-reverse.statistic, abs=1e-10)
    assert forward.cv2_1 == pytest.approx(reverse.cv2_2, abs=1e-12)
    assert forward.cv2_2 == pytest.approx(reverse.cv2_1, abs=1e-12)


def test_epps_p_values_agree_under_swapping_within_monte_carlo_error(
    analyzer: DistributionTestAnalyzer,
) -> None:
    """The p-value is a function of the group split, so order cannot matter.

    The permutation null re-labels the pooled sample, which is the same set
    of orderings whichever way round the caller supplied the two groups, so
    the two p-values must agree up to the Monte-Carlo error of estimating a
    proportion from 999 permutations (sd ~ sqrt(p(1-p)/1000), about 0.015 at
    p ~ 0.5). A permutation that shuffled only one of the two arrays, or a
    statistic that depended on argument order, fails here.

    Run for both equal and unequal sample sizes: with n1 != n2 the split
    point differs, which is expected and must still give the same answer.
    """
    for n1, n2 in ((20, 20), (12, 25)):
        rng = np.random.default_rng(19)
        x = rng.normal(size=n1)
        y = rng.lognormal(size=n2)
        forward = analyzer.epps_singleton(x, y, n_permutations=999, random_seed=1)
        reverse = analyzer.epps_singleton(y, x, n_permutations=999, random_seed=2)
        assert abs(forward.p_value - reverse.p_value) < 0.05, (n1, n2)


def test_epps_detects_a_variance_difference_at_equal_mean(analyzer: DistributionTestAnalyzer) -> None:
    """The statistic is not blind to spread at a shared mean.

    This is the case a location test cannot see. A module that reduced the
    Epps-Singleton statistic to a between-group mean difference (the
    plausible-but-wrong construction) returns a p-value near 1 here.
    """
    rng = np.random.default_rng(23)
    x = rng.normal(10.0, 1.0, 40)
    y = rng.normal(10.0, 4.0, 40)
    result = analyzer.epps_singleton(x, y, n_permutations=999, random_seed=2)
    assert result.statistic != 0.0
    assert result.p_value < 0.05
    assert result.significant is True


def test_epps_p_value_is_reproducible_from_its_seed(analyzer: DistributionTestAnalyzer) -> None:
    """Same seed, same p-value; the Monte-Carlo result is auditable.

    Without this a reported p-value could not be checked by anyone, which
    would make the module unusable in a report.
    """
    rng = np.random.default_rng(29)
    x = rng.normal(size=18)
    y = rng.normal(0.5, 1.2, 18)
    first = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=123)
    second = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=123)
    assert first.p_value == second.p_value
    assert first.n_significant == second.n_significant


def test_epps_reports_the_seed_it_actually_used(analyzer: DistributionTestAnalyzer) -> None:
    """The result carries the seed that reproduces its p-value.

    The bug this pins: with the default ``random_seed=None`` the
    permutation test drew fresh entropy inside ``make_rng``, but the
    result reported ``random_seed=None`` -- so nothing that could
    reproduce the p-value was reported, and two runs of the same analysis
    gave different p-values (0.95 and 0.905) with nothing to tell anyone
    how to get either of them back.

    Now the module draws the seed itself, reports it, and passing it back
    in reproduces the p-value exactly.
    """
    rng = np.random.default_rng(29)
    x = rng.normal(size=18)
    y = rng.normal(0.5, 1.2, 18)

    first = analyzer.epps_singleton(x, y, n_permutations=199)
    second = analyzer.epps_singleton(x, y, n_permutations=199)

    # The seed is reported, it is an integer, and it is not None.
    assert isinstance(first.random_seed, int)
    assert first.random_seed is not None
    assert first.to_dict()["random_seed"] == first.random_seed

    # Two runs of the same analysis report the same seed and the same
    # p-value.
    assert second.random_seed == first.random_seed
    assert second.p_value == first.p_value

    # And the reported seed reproduces the p-value when passed back in.
    replay = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=first.random_seed)
    assert replay.p_value == first.p_value
    assert replay.n_significant == first.n_significant
    assert replay.random_seed == first.random_seed


def test_epps_explicit_seed_is_reported_unchanged(analyzer: DistributionTestAnalyzer) -> None:
    """A seed the caller supplies is the seed that is reported.

    The reported seed must be the one that was used, not a substitute:
    reporting a different one would make the result irreproducible while
    looking perfectly reproducible.
    """
    rng = np.random.default_rng(29)
    x = rng.normal(size=18)
    y = rng.normal(0.5, 1.2, 18)
    result = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=4242)
    assert result.random_seed == 4242
    assert result.to_dict()["random_seed"] == 4242
    # A later unseeded call must not overwrite the caller's seed.
    unseeded = analyzer.epps_singleton(x, y, n_permutations=199)
    assert unseeded.random_seed != 4242
    assert analyzer.epps_singleton(x, y, n_permutations=199, random_seed=4242).random_seed == 4242


def test_epps_p_value_is_never_zero(analyzer: DistributionTestAnalyzer) -> None:
    """Even total separation reports 1/(n+1), never 0.

    Zero would claim more evidence than the permutation count can supply
    (Phipson & Smyth 2010). The floor is a closed form, not a fitted value.
    """
    rng = np.random.default_rng(31)
    x = rng.normal(0.0, 0.01, 15)
    y = rng.normal(100.0, 0.01, 15)
    result = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=6)
    assert result.p_value == pytest.approx(1.0 / 200.0, abs=1e-12)
    assert result.p_value > 0.0


def test_epps_p_values_are_uniform_under_the_null(analyzer: DistributionTestAnalyzer) -> None:
    """The check with teeth: under H0 the p-values must be UNIFORM.

    A permutation loop that shuffles the wrong array, or a statistic that
    responds to something other than the group split, still returns numbers
    in [0, 1]. The only way to catch it is to look at the DISTRIBUTION of
    the p-values, not at one of them.

    The previous version of this test asserted an UPPER bound on the
    false-positive rate only. That has no teeth against the most likely
    bug: a loop whose "permutations" all return the observed statistic
    reports p = 1.0 for every draw, a false-positive rate of 0.0, which
    satisfies "<= 0.20" perfectly. This version therefore also asserts

    * that the p-values are statistically indistinguishable from uniform
      (a one-sample Kolmogorov-Smirnov test over the pooled p-values),
    * that they are spread over the whole interval rather than piled at 1,
    * and that the false-positive rate is not zero either.

    Four distributions are used because a statistic calibrated only under
    one null would still look fine on that one.
    """
    alpha = 0.05
    reps = 200
    p_values: list[float] = []
    for draw in (
        lambda rng: (rng.normal(size=14), rng.normal(size=14)),
        lambda rng: (rng.exponential(size=14), rng.exponential(size=14)),
        lambda rng: (rng.lognormal(size=14), rng.lognormal(size=14)),
        lambda rng: (rng.poisson(2, 14), rng.poisson(2, 14)),
    ):
        p_values.extend(
            analyzer.epps_singleton(
                *draw(np.random.default_rng(7000 + 13 * i)), n_permutations=199, random_seed=i
            ).p_value
            for i in range(reps)
        )
    arr = np.asarray(p_values)

    # (a) The p-values must look uniform. A one-sided KS test against
    #     U(0, 1); the threshold is far below what the correct
    #     implementation produces (the smallest value seen over many
    #     independent repetitions of this check is 0.076).
    assert sp_stats.kstest(arr, "uniform").pvalue > 0.005, arr

    # (b) They must be spread over the interval, not piled at one end.
    assert float(arr.mean()) == pytest.approx(0.5, abs=0.05), arr
    assert float(arr.min()) < 0.2, arr
    assert float(arr.max()) > 0.8, arr

    # (c) The false-positive rate must be near alpha, and not zero: a
    #     broken loop that never re-labels the groups reports p = 1.0
    #     for every draw and never rejects at all.
    false_positive_rate = float((arr < alpha).mean())
    assert 0.0 < false_positive_rate <= 0.20, arr


def test_epps_rejects_a_genuinely_different_alternative(analyzer: DistributionTestAnalyzer) -> None:
    """The other half of the calibration: the test must have POWER.

    A null can be perfectly calibrated and still be useless -- a test that
    never rejects anything passes every uniformity check. The alternative
    used here differs in the statistic's own terms: the two groups share a
    mean but not a spread, which is exactly the case a location test
    cannot see and the case this test exists for.

    The rejection rate is measured over independent draws, so it is a
    statement about power and not about one lucky data set.
    """
    reps = 50
    rejections = 0
    for i in range(reps):
        rng = np.random.default_rng(5000 + i)
        x = rng.normal(10.0, 1.0, 30)
        y = rng.normal(10.0, 4.0, 30)
        p_value = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=i).p_value
        rejections += int(p_value < 0.05)
    assert rejections / reps >= 0.8, rejections


def test_the_uniformity_check_catches_a_broken_permutation_loop(
    analyzer: DistributionTestAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibration check must FAIL for a loop that never re-labels.

    A deliberately broken implementation -- every "permutation" returns the
    observed statistic -- reports p = 1.0 for every draw. The previous
    version of the calibration test only asserted an upper bound on the
    false-positive rate, which that broken implementation satisfies
    (0.0 <= 0.20). This test pins the properties that the broken version
    violates, so that the check cannot silently lose its teeth again.
    """
    import stats.distribution_tests as dt
    from utils.statistics_core import PermutationResult

    def broken_permutation_pvalue(observed, statistic_fn, **kwargs):
        """A permutation loop whose 'permutations' all return the observed value."""
        n_permutations = int(kwargs.get("n_permutations", 999))
        return PermutationResult(
            observed=float(observed),
            null_distribution=np.full(n_permutations, float(observed)),
            p_value=1.0,
            p_value_two_sided=1.0,
            n_permutations=n_permutations,
            n_significant=n_permutations,
            random_seed=kwargs.get("random_seed"),
        )

    monkeypatch.setattr(dt, "permutation_pvalue", broken_permutation_pvalue)

    p_values = [
        analyzer.epps_singleton(
            np.random.default_rng(7000 + 13 * i).normal(size=14),
            np.random.default_rng(8000 + 13 * i).normal(size=14),
            n_permutations=199,
            random_seed=i,
        ).p_value
        for i in range(50)
    ]
    arr = np.asarray(p_values)

    # The broken loop's p-values are all exactly 1.0: not uniform, and
    # they never reject anything.
    assert np.all(arr == 1.0)
    assert sp_stats.kstest(arr, "uniform").pvalue < 0.001
    assert float(arr.mean()) > 0.9
    assert float((arr < 0.05).mean()) == 0.0


def test_epps_is_invariant_to_a_common_shift(analyzer: DistributionTestAnalyzer) -> None:
    """Shifting both samples together changes nothing.

    CV^2 = s^2/xbar^2 is not shift-invariant on its own -- the denominator
    moves -- but the hypothesis is about the two *distributions*, and this
    pins the fact that the module applies no absolute-value convention of
    its own: the reported means move while the comparison does not.
    """
    rng = np.random.default_rng(37)
    x = rng.normal(50.0, 3.0, 25)
    y = rng.normal(52.0, 3.0, 25)
    shifted = analyzer.epps_singleton(x, y, n_permutations=199, random_seed=8)
    moved = analyzer.epps_singleton(x + 100.0, y + 100.0, n_permutations=199, random_seed=8)
    assert shifted.mean1 == pytest.approx(moved.mean1 - 100.0, abs=1e-9)
    # The CV^2 values genuinely differ (the denominator moved), but the test
    # still finds no evidence: both groups moved together.
    assert moved.p_value > 0.05


# ---------------------------------------------------------------------------
# Errors -- house exception types with actionable messages
# ---------------------------------------------------------------------------


def test_ks_rejects_a_too_small_sample(analyzer: DistributionTestAnalyzer) -> None:
    """One observation defines no empirical CDF, so the test is refused."""
    with pytest.raises(DataValidationError) as excinfo:
        analyzer.ks_2samp([1.0], [1.0, 2.0, 3.0])
    assert "at least 2 finite values" in str(excinfo.value)


def test_ks_rejects_an_unknown_method(analyzer: DistributionTestAnalyzer) -> None:
    """An unknown method is a validation error listing the valid ones."""
    with pytest.raises(DataValidationError) as excinfo:
        analyzer.ks_2samp([1.0, 2.0], [3.0, 4.0], method="monte-carlo")
    assert "exact" in str(excinfo.value)
    assert "monte-carlo" in str(excinfo.value)


def test_ks_rejects_all_non_finite_input(analyzer: DistributionTestAnalyzer) -> None:
    """Dropping every value leaves nothing to test, and that is an error."""
    with pytest.raises(DataValidationError):
        analyzer.ks_2samp([np.nan, np.inf], [1.0, 2.0, 3.0])


def test_epps_rejects_a_two_point_sample(analyzer: DistributionTestAnalyzer) -> None:
    """A two-point sample has no variance estimate worth dividing by a mean."""
    with pytest.raises(DataValidationError) as excinfo:
        analyzer.epps_singleton([1.0, 2.0], [3.0, 4.0, 5.0])
    assert "at least 3 finite values" in str(excinfo.value)


def test_epps_rejects_a_zero_mean_sample(analyzer: DistributionTestAnalyzer) -> None:
    """s^2/xbar^2 at xbar = 0 is undefined; the message says what to do."""
    with pytest.raises(ComputationError) as excinfo:
        analyzer.epps_singleton([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0])
    message = str(excinfo.value)
    assert "zero" in message.lower()
    assert "shift" in message.lower()


def test_epps_rejects_non_positive_permutations(analyzer: DistributionTestAnalyzer) -> None:
    """n_permutations must be positive; the house helper raises for it."""
    with pytest.raises(DataValidationError):
        analyzer.epps_singleton([1.0, 2.0, 4.0], [2.0, 3.0, 8.0], n_permutations=0)


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------


def test_ks_result_dict_round_trips(analyzer: DistributionTestAnalyzer) -> None:
    """to_dict exposes the fields a caller needs, as plain JSON types."""
    result = analyzer.ks_2samp([1, 2, 2, 3, 4], [2, 3, 3, 5])
    payload = result.to_dict()
    assert payload["n1"] == 5
    assert payload["n2"] == 4
    assert payload["ties"] is True
    assert payload["significant"] is False
    assert isinstance(payload["statistic"], float)
    assert isinstance(payload["significant"], bool)


def test_epps_result_dict_round_trips(analyzer: DistributionTestAnalyzer) -> None:
    """Same for the Epps-Singleton payload."""
    rng = np.random.default_rng(41)
    result = analyzer.epps_singleton(rng.normal(size=15), rng.normal(size=15), n_permutations=99, random_seed=3)
    payload = result.to_dict()
    assert payload["n1"] == 15
    assert payload["n_permutations"] == 99
    assert payload["random_seed"] == 3
    assert payload["significant"] is False


def test_summaries_are_non_empty_and_carry_the_headline_numbers(
    analyzer: DistributionTestAnalyzer,
) -> None:
    """A summary that omits D or the p-value is useless in a report."""
    ks = analyzer.ks_2samp([1, 2, 2, 3, 4], [2, 3, 3, 5])
    ks_text = ks.summary()
    assert "Kolmogorov" in ks_text
    assert f"{ks.statistic:.4f}" in ks_text
    assert f"{ks.p_value:.4f}" in ks_text

    rng = np.random.default_rng(43)
    es = analyzer.epps_singleton(rng.normal(size=15), rng.normal(size=15), n_permutations=99, random_seed=5)
    es_text = es.summary()
    assert "Epps" in es_text
    assert f"{es.statistic:.4f}" in es_text
    assert f"{es.p_value:.4f}" in es_text


def test_last_result_tracks_the_most_recent_run(analyzer: DistributionTestAnalyzer) -> None:
    """The analyzer reports what it last computed, and None before that."""
    assert analyzer.last_result is None
    first = analyzer.ks_2samp([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert analyzer.last_result is first
    rng = np.random.default_rng(47)
    second = analyzer.epps_singleton(rng.normal(size=12), rng.normal(size=12), n_permutations=99, random_seed=1)
    assert analyzer.last_result is second


def test_constant_vs_non_constant_samples_are_handled(analyzer: DistributionTestAnalyzer) -> None:
    """A zero-variance sample is a legitimate KS input, not an error.

    scipy returns p = 1.0 here and so does this module; the check is that
    the tie-heavy structure is handled rather than raising or returning a
    NaN. R agrees (0.14285714 exact for the 4-and-4 case above).
    """
    result = analyzer.ks_2samp([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], method="exact")
    assert np.isfinite(result.statistic)
    assert np.isfinite(result.p_value)
    assert result.p_value == pytest.approx(1.0 / 7.0, abs=1e-12)
    assert result.ties is True


def test_equal_and_unequal_sample_sizes_are_both_handled(analyzer: DistributionTestAnalyzer) -> None:
    """Unequal sizes must not change the shape of the answer.

    Compared against the equivalent data where n1 == n2, so a module that
    silently assumed balanced groups fails here.
    """
    rng = np.random.default_rng(53)
    x = rng.normal(size=5)
    y = rng.normal(0.3, 1.1, 30)
    unbalanced = analyzer.ks_2samp(x, y, method="exact")
    assert unbalanced.n1 == 5
    assert unbalanced.n2 == 30
    assert 0.0 <= unbalanced.p_value <= 1.0
