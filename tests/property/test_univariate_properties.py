# =============================================================================
# FILE: tests/property/test_univariate_properties.py
# =============================================================================
"""
Property-based tests for the univariate tests.

A worked example shows that one answer was right once.  These show that
the answer is right for *every* input of a shape -- which is where a
statistical test actually earns its place, because the input a
palaeontologist brings is not the one in the test file.

The invariants used here are the ones a test must have to be a test:

  * **A positive affine transform** ``y -> a*y + b`` (``a > 0``) changes
    the numbers but not the conclusions: a mean shift and a rescaling
    leave every test statistic and every p-value alone.  A test that
    changes under one is not measuring what it claims to measure.
  * **A strictly increasing transform** changes the numbers and leaves
    the *rank*-based statistics alone, because the ranks are unchanged.
  * **Permuting the values inside a group**, or renaming the groups,
    cannot change a p-value: the design is the same design.
  * **Symmetry**: the Mann-Whitney statistic of two samples and the
    statistic of the two samples the other way round add up to the
    number of pairs.
  * **Independence of the confidence level**: the p-value of a binomial
    test is about the test, not about the interval.

Two traps this file walks into on purpose, and steps around:

  * Shuffling the values of a sample *globally* moves observations from
    one group to another, which is a different design, not a different
    order.  The shuffle here is inside each group.
  * A transform applied to the *values* can create ties that the
    original sample did not have -- ``tanh`` saturates, and two large
    values become the same double -- and a new tie is a new rank.  The
    strictly increasing transform used here is built from the ranks, so
    it is exact and it introduces no tie that was not there.

References:
    Conover, W.J. (1999). Statistical Methods Using Nonparametrics.
    2nd ed. SAS Institute.
    Ruxt, R.L. (2006). The Intuition of Statistics. 3rd ed. Wiley.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stats.univariate import UnivariateAnalyzer
from stats.univariate_menu import UnivariateExtraMenu

from ._helpers import (
    affine,
    call_or_none,
    has_variation,
    integer_strategy,
    monotone,
    odd_strategy,
    scale_strategy,
    shift_strategy,
    split_in_two,
    two_group_strategy,
    two_samples,
    values_strategy,
)

#: The tolerances.  A statistic computed on the transformed data is not
#: bit-identical to the one computed on the original -- the arithmetic
#: goes through different intermediate values -- so the comparison is
#: made at the precision the arithmetic actually has.  A wrong
#: implementation is wrong by a factor, not by the last few bits.
RTOL = 1e-6
ATOL = 1e-8

_SETTING = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _shuffle_within_groups(values: np.ndarray, groups: list[int], seed: int) -> np.ndarray:
    """Shuffle the values inside each group, leaving the groups alone.

    A global shuffle would move observations from one group to another,
    which is a different design rather than a different order.
    """
    rng = np.random.default_rng(seed)
    out = values.copy()
    for label in set(groups):
        indices = [i for i, g in enumerate(groups) if g == label]
        block = out[indices]
        rng.shuffle(block)
        out[indices] = block
    return out


# ---------------------------------------------------------------------------
# Affine invariance
# ---------------------------------------------------------------------------
@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_one_way_anova_is_invariant_under_a_positive_affine_transform(values, scale, shift):
    """A mean shift and a rescaling cannot change an F ratio.

    The F ratio is a ratio of two quantities that both scale by
    ``a**2``, so the scale cancels; the shift moves numerator and
    denominator by the same amount and cancels too.  If the statistic
    moves, the implementation is doing something else.
    """
    array, groups = split_in_two(values)
    analyzer = UnivariateAnalyzer()
    base = call_or_none(analyzer.one_way_anova, array, groups, tukey=False)
    moved = call_or_none(analyzer.one_way_anova, affine(array, scale, shift), groups, tukey=False)
    if base is None or moved is None:
        return
    assert np.isclose(base.f_statistic, moved.f_statistic, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_normality_is_invariant_under_a_positive_affine_transform(values, scale, shift):
    """The Shapiro-Wilk statistic is a function of the *ranks* of the
    standardised deviations, so a positive affine transform leaves it
    alone."""
    array = np.asarray(values, dtype=float)
    if not has_variation(array):
        return
    analyzer = UnivariateAnalyzer()
    base = call_or_none(analyzer.normality_test, array)
    moved = call_or_none(analyzer.normality_test, affine(array, scale, shift))
    if base is None or moved is None:
        return
    assert np.isclose(base.shapiro_stat, moved.shapiro_stat, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.shapiro_p, moved.shapiro_p, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_fligner_kileen_is_invariant_under_a_positive_affine_transform(values, scale, shift):
    """The Fliller-Kileen statistic is built on the *ranks* of the
    absolute deviations, and a positive scaling multiplies every
    deviation by the same factor, so the ranks are unchanged."""
    array = np.asarray(values, dtype=float)
    if array.size < 6:
        return
    third = array.size // 3
    groups = [0] * third + [1] * third + [2] * (array.size - 2 * third)
    if array.size - 2 * third < 2:
        return
    labels = np.asarray(groups, dtype=float)
    menu = UnivariateExtraMenu()
    base = call_or_none(menu.fligner_kileen, np.column_stack([labels, array]), groups=0, values=1)
    moved = call_or_none(
        menu.fligner_kileen,
        np.column_stack([labels, affine(array, scale, shift)]),
        groups=0,
        values=1,
    )
    if base is None or moved is None:
        return
    assert np.isclose(base.statistic, moved.statistic, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_the_outlier_z_scores_are_invariant_under_a_positive_affine_transform(values, scale, shift):
    """A z-score is a standardised deviation, and standardising removes
    both the shift and the scale."""
    array = np.asarray(values, dtype=float)
    if not has_variation(array):
        return
    menu = UnivariateExtraMenu()
    base = call_or_none(menu.outlier_screen, array.reshape(-1, 1), values=0, method="zscore")
    moved = call_or_none(menu.outlier_screen, affine(array, scale, shift).reshape(-1, 1), values=0, method="zscore")
    if base is None or moved is None:
        return
    assert np.allclose(base.z_scores, moved.z_scores, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.max_abs_z, moved.max_abs_z, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert base.outlier_indices == moved.outlier_indices


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_brunner_munzel_is_invariant_under_a_positive_affine_transform(values, scale, shift):
    """The Brunner-Munzel test counts wins, and a positive affine
    transform does not change which observation is bigger."""
    array = np.asarray(values, dtype=float)
    first, second = two_samples(array)
    if first.size < 2 or second.size < 2:
        return
    data = np.column_stack([first, second])
    moved = np.column_stack([affine(first, scale, shift), affine(second, scale, shift)])
    menu = UnivariateExtraMenu()
    base = call_or_none(menu.brunner_munzel, data, x=0, y=1)
    changed = call_or_none(menu.brunner_munzel, moved, x=0, y=1)
    if base is None or changed is None:
        return
    assert np.isclose(base.wins, changed.wins, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.statistic, changed.statistic, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_mood_median_is_invariant_under_a_positive_affine_transform(values, scale, shift):
    """Mood's test asks how many observations are above the pooled
    median, and a positive affine transform preserves both the order and
    which side of the median each observation falls on."""
    array = np.asarray(values, dtype=float)
    first, second = two_samples(array)
    if first.size < 2 or second.size < 2:
        return
    data = np.column_stack([first, second])
    moved = np.column_stack([affine(first, scale, shift), affine(second, scale, shift)])
    menu = UnivariateExtraMenu()
    base = call_or_none(menu.mood_median, data, x=0, y=1)
    changed = call_or_none(menu.mood_median, moved, x=0, y=1)
    if base is None or changed is None:
        return
    assert base.n_above_x == changed.n_above_x
    assert base.n_above_y == changed.n_above_y
    assert base.n_below_x == changed.n_below_x
    assert base.n_below_y == changed.n_below_y
    assert np.isclose(base.p_value, changed.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


# ---------------------------------------------------------------------------
# Rank invariance
# ---------------------------------------------------------------------------
@given(values=values_strategy)
@_SETTING
def test_kruskal_wallis_is_invariant_under_a_strictly_increasing_transform(values):
    """Kruskal-Wallis is a test on ranks, and a strictly increasing
    transform leaves the ranks alone."""
    array, groups = split_in_two(values)
    analyzer = UnivariateAnalyzer()
    base = call_or_none(analyzer.kruskal_wallis, array, groups)
    moved = call_or_none(analyzer.kruskal_wallis, monotone(array), groups)
    if base is None or moved is None:
        return
    assert np.isclose(base.statistic, moved.statistic, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=values_strategy)
@_SETTING
def test_mann_whitney_is_invariant_under_a_strictly_increasing_transform(values):
    """The Mann-Whitney U counts wins, and a strictly increasing
    transform does not change which observation is bigger."""
    array, groups = split_in_two(values)
    analyzer = UnivariateAnalyzer()
    base = call_or_none(analyzer.mann_whitney, array, groups)
    moved = call_or_none(analyzer.mann_whitney, monotone(array), groups)
    if base is None or moved is None:
        return
    assert np.isclose(base.statistic, moved.statistic, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=values_strategy)
@_SETTING
def test_the_mann_whitney_statistic_of_two_samples_adds_up_to_the_number_of_pairs(values):
    """U(x, y) + U(y, x) = n1 * n2.

    Every pair of observations is either a win for the first sample or a
    win for the second, and there are ``n1 * n2`` pairs.  A statistic
    that does not add up is counting something else.
    """
    array = np.asarray(values, dtype=float)
    half = array.size // 2
    if half < 1 or array.size - half < 1:
        return
    first = [0] * half + [1] * (array.size - half)
    second = [1] * half + [0] * (array.size - half)
    analyzer = UnivariateAnalyzer()
    forward = call_or_none(analyzer.mann_whitney, array, first)
    backward = call_or_none(analyzer.mann_whitney, array, second)
    if forward is None or backward is None:
        return
    assert np.isclose(
        forward.statistic + backward.statistic,
        half * (array.size - half),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------
@given(values=two_group_strategy)
@_SETTING
def test_the_one_way_anova_p_value_does_not_depend_on_the_group_labels(values):
    """Renaming the groups is the same design with a different name."""
    array, groups = split_in_two(values)
    renamed = [1 - g for g in groups]
    analyzer = UnivariateAnalyzer()
    base = call_or_none(analyzer.one_way_anova, array, groups, tukey=False)
    moved = call_or_none(analyzer.one_way_anova, array, renamed, tukey=False)
    if base is None or moved is None:
        return
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.f_statistic, moved.f_statistic, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=two_group_strategy, seed=st.integers(min_value=0, max_value=2**16))
@_SETTING
def test_the_one_way_anova_p_value_does_not_depend_on_the_order_of_the_values(values, seed):
    """Shuffling the values inside a group is the same data.

    The order of the rows is not part of the design, so a p-value that
    changes when the rows are shuffled is reading the row order.
    """
    array, groups = split_in_two(values)
    shuffled = _shuffle_within_groups(array, groups, seed)
    analyzer = UnivariateAnalyzer()
    base = call_or_none(analyzer.one_way_anova, array, groups, tukey=False)
    moved = call_or_none(analyzer.one_way_anova, shuffled, groups, tukey=False)
    if base is None or moved is None:
        return
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.f_statistic, moved.f_statistic, rtol=RTOL, atol=ATOL, equal_nan=True)


# ---------------------------------------------------------------------------
# The summary statistics
# ---------------------------------------------------------------------------
@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_the_summary_statistics_transform_predictably(values, scale, shift):
    """The mean moves by ``a*x + b`` and the standard deviation scales
    by ``a``."""
    array = np.asarray(values, dtype=float)
    if not has_variation(array):
        return
    analyzer = UnivariateAnalyzer()
    base = analyzer.summary_statistics(array.reshape(-1, 1)).columns[0]
    moved = analyzer.summary_statistics(affine(array, scale, shift).reshape(-1, 1)).columns[0]
    assert np.isclose(moved.mean, base.mean * scale + shift, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(moved.std, base.std * scale, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(moved.variance, base.variance * scale**2, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(moved.se, base.se * scale, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=odd_strategy)
@_SETTING
def test_the_summary_median_is_the_middle_observation(values):
    """For an odd-sized sample the median is the middle observation.

    The sample is odd on purpose: with an even one the median is the
    average of the two middle values, and then the question "is the
    median one of the observations?" has a different answer.
    """
    array = np.asarray(values, dtype=float)
    analyzer = UnivariateAnalyzer()
    base = analyzer.summary_statistics(array.reshape(-1, 1)).columns[0]
    ordered = np.sort(array)
    assert np.isclose(base.median, ordered[array.size // 2], rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=odd_strategy)
@_SETTING
def test_the_summary_median_survives_a_strictly_increasing_transform(values):
    """A strictly increasing transform carries the median along.

    The transformed sample has the same order as the original, so the
    middle observation is still the middle one -- and the summary has to
    report it.
    """
    array = np.asarray(values, dtype=float)
    analyzer = UnivariateAnalyzer()
    moved = analyzer.summary_statistics(monotone(array).reshape(-1, 1)).columns[0]
    ordered = np.sort(monotone(array))
    assert np.isclose(moved.median, ordered[array.size // 2], rtol=RTOL, atol=ATOL, equal_nan=True)


# ---------------------------------------------------------------------------
# The binomial test
# ---------------------------------------------------------------------------
@given(
    successes=st.integers(min_value=0, max_value=30),
    extra=st.integers(min_value=0, max_value=30),
    confidence=st.floats(min_value=0.5, max_value=0.999, allow_nan=False),
)
@_SETTING
def test_the_binomial_p_value_does_not_depend_on_the_confidence_level(successes, extra, confidence):
    """The confidence level is about the *interval*, not the test.

    A p-value that moved with the confidence level would be reporting
    something other than the p-value of the test.
    """
    trials = successes + extra
    if trials < 1:
        return
    menu = UnivariateExtraMenu()
    base = menu.proportion_test(successes=successes, trials=trials, p0=0.5, confidence=0.95)
    moved = menu.proportion_test(successes=successes, trials=trials, p0=0.5, confidence=float(confidence))
    assert np.isclose(base.p_value, moved.p_value, rtol=1e-12, atol=1e-14)


@given(successes=st.integers(min_value=0, max_value=30), extra=st.integers(min_value=0, max_value=30))
@_SETTING
def test_the_binomial_test_is_symmetric_about_one_half(successes, extra):
    """``binom.test(k, n, p)`` and ``binom.test(n - k, n, 1 - p)`` are
    the same test read from the other end."""
    trials = successes + extra
    if trials < 1:
        return
    menu = UnivariateExtraMenu()
    base = menu.proportion_test(successes=successes, trials=trials, p0=0.5)
    moved = menu.proportion_test(successes=trials - successes, trials=trials, p0=0.5)
    assert np.isclose(base.p_value, moved.p_value, rtol=1e-10, atol=1e-12)
