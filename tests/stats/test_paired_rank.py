# =============================================================================
# FILE: tests/stats/test_paired_rank.py
# =============================================================================
"""
Tests for the paired nonparametric tests: sign and Wilcoxon signed-rank.

The sign test's exact answer is computable by hand, so that is what the
main assertions use: for n non-zero differences all in one direction, the
two-sided binomial p-value is 2 * (1/2)^n, which can be checked against a
literal rather than against a library.

The Wilcoxon tests use the properties that hold regardless of the
algorithm -- reversibility, symmetry of p under sign flip, and invariance
to the scale of the differences -- plus one comparison against scipy,
which is an independent implementation rather than a self-check.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as sp_stats

from stats.univariate import UnivariateAnalyzer
from utils.exceptions import ComputationError, ValidationError


@pytest.fixture
def analyzer() -> UnivariateAnalyzer:
    """A fresh analyzer."""
    return UnivariateAnalyzer()


def _paired(after: list[float], before: list[float]) -> npt.NDArray:
    """One column, 2n rows: the "after" block then the "before" block.

    ``t_test`` and ``paired_rank_test`` both split ROWS by group and then
    read one column, so a paired design puts group 0's n values in the
    first n rows and group 1's n values in the next n. The i-th member of
    each group is then row i within its own block, which is the pairing.
    """
    values = np.asarray(list(after) + list(before), dtype=float)
    return values.reshape(-1, 1)


def _groups(n: int) -> list[int]:
    """Group 0 for the first n rows, group 1 for the next n."""
    return [0] * n + [1] * n


# ---------------------------------------------------------------------------
# Sign test -- exact binomial
# ---------------------------------------------------------------------------


def test_sign_test_matches_the_exact_binomial(analyzer: UnivariateAnalyzer) -> None:
    """All differences positive: p = 2 * 0.5^n, computed here by hand."""
    n = 12
    before = [1.0] * n
    after = [2.0] * n
    result = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(n), method="sign"
    )
    expected = 2.0 * (0.5**n)
    assert result.p_value == pytest.approx(expected, rel=1e-9)
    assert result.statistic == float(n)
    assert result.n_positive == n
    assert result.n_negative == 0
    assert result.n_ties == 0
    assert result.n_pairs == n
    assert result.median_difference == pytest.approx(1.0)
    assert result.significant


def test_sign_test_with_ties_discards_them(analyzer: UnivariateAnalyzer) -> None:
    """Zero differences carry no direction, so they are not counted."""
    before = [1.0, 1.0, 1.0, 1.0]
    after = [2.0, 1.0, 3.0, 1.0]  # two ties
    result = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(4), method="sign"
    )
    assert result.n_ties == 2
    assert result.n_positive == 2
    assert result.n_negative == 0
    # n_used = 2, all positive -> p = 2 * 0.25 = 0.5
    assert result.p_value == pytest.approx(0.5, rel=1e-9)
    # The ties are still reported, so n_pairs covers the whole design.
    assert result.n_pairs == 4


def test_sign_test_is_symmetric(analyzer: UnivariateAnalyzer) -> None:
    """Flipping the direction of every difference must not move the p-value."""
    before = [1.0] * 8
    forward = [2.0] * 6 + [1.0] * 2
    result_a = analyzer.paired_rank_test(
        _paired(forward, before), groups=_groups(8), method="sign"
    )
    result_b = analyzer.paired_rank_test(
        _paired(before, forward), groups=_groups(8), method="sign"
    )
    assert result_a.p_value == pytest.approx(result_b.p_value, rel=1e-12)
    assert (result_a.n_positive, result_a.n_negative) == (
        result_b.n_negative,
        result_b.n_positive,
    )


def test_sign_test_ignores_the_magnitude_of_differences(
    analyzer: UnivariateAnalyzer,
) -> None:
    """One huge outlier does not make the sign test significant.

    This is the property that separates it from the rank-based test, and
    the reason both are offered.
    """
    n = 10
    before = [0.0] * n
    after = [0.1] * 5 + [-0.1] * 4 + [1000.0]
    result = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(n), method="sign"
    )
    assert result.n_positive == 6
    assert result.n_negative == 4
    # 6 of 10 is nowhere near significant.
    assert result.p_value > 0.5
    assert not result.significant


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------


def test_wilcoxon_reverses_with_the_data(analyzer: UnivariateAnalyzer) -> None:
    """Swapping the two columns must mirror the statistic, not change it."""
    before = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    after = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    forward = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(6), method="wilcoxon"
    )
    reverse = analyzer.paired_rank_test(
        _paired(before, after), groups=_groups(6), method="wilcoxon"
    )
    assert forward.statistic == pytest.approx(reverse.statistic, abs=1e-9)
    assert forward.p_value == pytest.approx(reverse.p_value, rel=1e-12)
    assert forward.median_difference == pytest.approx(
        -reverse.median_difference
    )


def test_wilcoxon_is_affected_by_an_outlier_that_the_sign_test_ignores(
    analyzer: UnivariateAnalyzer,
) -> None:
    """A single large difference is noise to a sign test and signal to ranks."""
    n = 10
    before = [0.0] * n
    after = [0.1] * 5 + [-0.1] * 4 + [100.0]
    sign = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(n), method="sign"
    )
    wilcoxon = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(n), method="wilcoxon"
    )
    assert not sign.significant
    assert wilcoxon.p_value < sign.p_value


def test_wilcoxon_agrees_with_scipy(analyzer: UnivariateAnalyzer) -> None:
    """Independent implementation, same answer."""
    rng = np.random.default_rng(0)
    before = rng.normal(10, 2, 20)
    after = before + rng.normal(0.5, 0.3, 20)
    result = analyzer.paired_rank_test(
        _paired(after.tolist(), before.tolist()),
        groups=_groups(20),
        method="wilcoxon",
    )
    expected_w, expected_p = sp_stats.wilcoxon(after, before, zero_method="wilcox")
    assert result.statistic == pytest.approx(expected_w, abs=1e-9)
    assert result.p_value == pytest.approx(expected_p, rel=1e-12)





# ---------------------------------------------------------------------------
# Pairing behaviour shared by both
# ---------------------------------------------------------------------------


def test_a_pair_is_dropped_when_either_side_is_missing(
    analyzer: UnivariateAnalyzer,
) -> None:
    """A missing measurement invalidates its pair, not just its own side.

    Filtering each side independently would shift the survivors against
    each other and test a different pairing entirely.
    """
    before = [1.0, 2.0, 3.0, 4.0]
    after = [2.0, np.nan, 4.0, 5.0]
    result = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(4), method="sign"
    )
    # Pair 1 has a missing "after", so it goes even though its "before" is
    # present. The survivors are pairs 0, 2 and 3, all of which differ by
    # exactly +1 -- which is the assertion that matters: if the survivors
    # had been realigned instead of dropped, the differences would be a
    # different set of numbers.
    assert result.n_pairs == 3
    assert result.n_ties == 0
    assert result.n_positive == 3
    assert result.n_negative == 0
    assert result.median_difference == pytest.approx(1.0)


def test_unequal_group_sizes_are_rejected(analyzer: UnivariateAnalyzer) -> None:
    """A paired design needs one partner per observation.

    Four rows and four labels, so the label count is consistent; the
    groups themselves are the problem (3 vs 1), and that is the error
    under test here.
    """
    data = np.arange(4, dtype=float).reshape(-1, 1)
    with pytest.raises(ComputationError, match="equal sample sizes"):
        analyzer.paired_rank_test(data, groups=[0, 0, 0, 1], method="wilcoxon")


def test_mismatched_label_count_is_rejected(analyzer: UnivariateAnalyzer) -> None:
    """Labels and rows have to agree before anything else is considered."""
    data = np.arange(2, dtype=float).reshape(-1, 1)
    with pytest.raises((ComputationError, ValidationError)):
        analyzer.paired_rank_test(data, groups=[0, 1, 1], method="wilcoxon")


def test_three_groups_are_rejected(analyzer: UnivariateAnalyzer) -> None:
    """A paired test is a two-group test."""
    data = np.random.default_rng(0).normal(size=(9, 2))
    with pytest.raises(ComputationError):
        analyzer.paired_rank_test(data, groups=[0, 0, 0, 1, 1, 1, 2, 2, 2])


def test_missing_groups_are_rejected(analyzer: UnivariateAnalyzer) -> None:
    """Group labels are required."""
    data = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ComputationError):
        analyzer.paired_rank_test(data, groups=None, method="wilcoxon")


def test_unknown_method_is_rejected(analyzer: UnivariateAnalyzer) -> None:
    """Only the two implemented tests."""
    data = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ValidationError):
        analyzer.paired_rank_test(data, groups=_groups(3), method="mannwhitney")


def test_all_zero_differences_are_rejected(analyzer: UnivariateAnalyzer) -> None:
    """Every difference zero: there is no direction to test.

    Both methods reject this. scipy would return p = 1.0 for the Wilcoxon
    case, but a "no difference detected" verdict computed from data that
    contains no differences at all is a statement about the input, not
    about the specimens, and is better made loudly.
    """
    values = [1.0, 2.0, 3.0, 4.0]
    for method in ("sign", "wilcoxon"):
        with pytest.raises(ComputationError):
            analyzer.paired_rank_test(
                _paired(values, values), groups=_groups(4), method=method
            )


def test_wilcoxon_counts_ties_without_discarding_the_rest(
    analyzer: UnivariateAnalyzer,
) -> None:
    """A few ties do not stop the Wilcoxon test, unlike the sign test."""
    before = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    after = [2.0, 2.0, 4.0, 4.0, 6.0, 6.0]  # three ties, three doubles
    result = analyzer.paired_rank_test(
        _paired(after, before), groups=_groups(6), method="wilcoxon"
    )
    assert result.n_ties == 3
    assert result.n_positive == 3
    assert result.n_negative == 0
    # n_pairs is the design size after NaN filtering, not the number of
    # non-zero differences; the ties are what the Wilcoxon test drops, and
    # they are reported separately as n_ties.
    assert result.n_pairs == 6
    assert result.p_value == pytest.approx(2.0 * 0.5**3, rel=1e-9)


def test_too_few_pairs_are_rejected(analyzer: UnivariateAnalyzer) -> None:
    """One pair cannot support a rank test."""
    data = np.array([[1.0], [2.0]])
    with pytest.raises(ComputationError):
        analyzer.paired_rank_test(data, groups=[0, 1], method="wilcoxon")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_result_serialises(analyzer: UnivariateAnalyzer) -> None:
    """to_dict and summary both work for either test."""
    before = [1.0] * 10
    after = [2.0] * 10
    for method in ("sign", "wilcoxon"):
        result = analyzer.paired_rank_test(
            _paired(after, before), groups=_groups(10), method=method
        )
        payload = result.to_dict()
        assert payload["test_type"] == method
        assert set(payload) >= {"statistic", "p_value", "n_positive", "n_ties"}
        assert "Paired Nonparametric Test" in result.summary()


def test_paired_t_test_still_shares_the_pairing_rule(
    analyzer: UnivariateAnalyzer,
) -> None:
    """The t-test refactored onto the shared splitter must behave as before."""
    before = [1.0, 2.0, 3.0, 4.0]
    after = [2.0, 3.0, 4.0, 5.0]
    data = _paired(after, before)
    t_result = analyzer.t_test(data, column=0, groups=_groups(4), paired=True)
    expected = sp_stats.ttest_rel(after, before)
    assert t_result.statistic == pytest.approx(expected.statistic, abs=1e-9)
    assert t_result.p_value == pytest.approx(expected.pvalue, rel=1e-12)
    assert t_result.test_type == "paired"
    assert t_result.n1 == 4
