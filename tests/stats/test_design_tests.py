# =============================================================================
# FILE: tests/stats/test_design_tests.py
# =============================================================================
"""
Tests for the factorial, repeated-measures and reliability designs.

The properties asserted here are chosen so that a broken implementation
fails rather than merely returning a different number:

  * every ANOVA sum of squares is checked against an arithmetic derivation
    written out in the test, not against a value this module produced;
  * the main-effect sums of squares are checked against the *one-way*
    between-group sums of squares of the same data, which is an identity
    (SS_total = SS_A + within|A) rather than a restatement;
  * the ANOVA p-values are checked against ``scipy.stats.f.sf``, an
    independent implementation of the tail probability;
  * every repeated-measures denominator is checked explicitly, including
    an assertion that the group effect is *not* divided by the
    within-subject error -- the error that returns a plausible number;
  * the ICC confidence interval is checked by coverage against a known
    ICC, not by comparing the interval to the point estimate;
  * a permutation loop that reshuffles the wrong array still returns a
    number in [0, 1], so the seeded permutation tests check calibration
    and determinism rather than a stored p-value.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as sp_stats

from stats.design_tests import DesignTestAnalyzer
from utils.exceptions import (
    ComputationError,
    DataValidationError,
    MatrixDimensionError,
    ValidationError,
)


@pytest.fixture
def analyzer() -> DesignTestAnalyzer:
    """A fresh analyzer."""
    return DesignTestAnalyzer()


# ---------------------------------------------------------------------------
# Two-way ANOVA -- a design small enough to work out on paper
# ---------------------------------------------------------------------------


def _factorial_hand_case() -> tuple[npt.NDArray, list[str], list[str]]:
    """A 2x2 design with two replicates, chosen so every SS is an integer.

    Cell layout (row = factor A, column = factor B):

        A1B1 = 10, 14      A1B2 = 20, 24
        A2B1 = 30, 34      A2B2 = 12, 16
    """
    values = np.array([10.0, 14.0, 20.0, 24.0, 30.0, 34.0, 12.0, 16.0])
    factor_a = ["a1", "a1", "a1", "a1", "a2", "a2", "a2", "a2"]
    factor_b = ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"]
    return values, factor_a, factor_b


def test_two_way_anova_matches_the_hand_derived_decomposition(
    analyzer: DesignTestAnalyzer,
) -> None:
    """The sums of squares, computed by hand, from Winer (1971) ch. 5.

    Grand mean = (10+14+20+24+30+34+12+16)/8 = 160/8 = 20.
    Cell means: A1B1 = 12, A1B2 = 22, A2B1 = 32, A2B2 = 14.
    A means: 17, 23.  B means: 22, 18.

    SS_total = sum (y - 20)^2 = 100+36+0+16+100+196+64+16 = 528
    SS_A     = 4(17-20)^2 + 4(23-20)^2 = 4*9 + 4*9 = 72
    SS_B     = 4(22-20)^2 + 4(18-20)^2 = 16 + 16 = 32
    Additive fit m_ij = M + (m_i. - M) + (m_.j - M):
        A1B1 = 19, A1B2 = 15, A2B1 = 25, A2B2 = 21
    Interaction contrast m_ij - fitted:
        12-19 = -7, 22-15 = +7, 32-25 = +7, 14-21 = -7
    SS_AB    = 2*(49 + 49 + 49 + 49) = 392
    SS_error = 4 cells x (1^2 + 1^2) = 8 + 8 = 32
    Closure: 72 + 32 + 392 + 32 = 528 = SS_total.

    df: A = 1, B = 1, AB = 1, error = 8 - 4 = 4.
    MS_error = 32/4 = 8, so
        F_A = (72/1)/8 = 9, F_B = (32/1)/8 = 4, F_AB = (392/1)/8 = 49.
    """
    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    assert result.ss_total == pytest.approx(528.0, abs=1e-9)
    assert result.term_a.ss == pytest.approx(72.0, abs=1e-9)
    assert result.term_b.ss == pytest.approx(32.0, abs=1e-9)
    assert result.term_interaction.ss == pytest.approx(392.0, abs=1e-9)
    assert result.ss_error == pytest.approx(32.0, abs=1e-9)

    assert (result.term_a.df, result.term_b.df, result.term_interaction.df) == (1, 1, 1)
    assert result.df_error == 4
    assert result.df_total == 7

    assert result.term_a.f_statistic == pytest.approx(9.0, abs=1e-9)
    assert result.term_b.f_statistic == pytest.approx(4.0, abs=1e-9)
    assert result.term_interaction.f_statistic == pytest.approx(49.0, abs=1e-9)


def test_two_way_anova_effect_sizes_follow_their_definitions(
    analyzer: DesignTestAnalyzer,
) -> None:
    """eta^2 is the share of the total, partial eta^2 the share against error.

    eta^2     = SS_term / SS_total = 72/528
    partial   = SS_term / (SS_term + SS_error) = 72/104
    The two differ here precisely because a factorial design has a total
    that is more than one term plus the error -- which is the case where
    telling them apart matters.
    """
    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    assert result.term_a.eta_squared == pytest.approx(72.0 / 528.0, abs=1e-12)
    assert result.term_a.partial_eta_squared == pytest.approx(72.0 / 104.0, abs=1e-12)
    assert result.term_interaction.eta_squared == pytest.approx(392.0 / 528.0, abs=1e-12)
    assert result.term_interaction.partial_eta_squared == pytest.approx(392.0 / 424.0, abs=1e-12)


def test_two_way_anova_main_effects_equal_a_one_way_anova(
    analyzer: DesignTestAnalyzer,
) -> None:
    """SS_A must equal the one-way between-group SS of the same data.

    For any grouping of the observations, SS_total = SS_between +
    SS_within, and SS_A = SS_total - within|A. So computing the one-way
    between-group sum of squares from the individual observations is an
    independent route to SS_A: different code, different grouping
    operation, same number. (This identity holds for unbalanced designs
    as well, which is why the check is repeated on one below.)
    """

    def one_way_between(response: npt.NDArray, labels: list[str]) -> float:
        grand = float(np.mean(response))
        total = 0.0
        for level in set(labels):
            chunk = response[[i for i, v in enumerate(labels) if v == level]]
            total += len(chunk) * (float(np.mean(chunk)) - grand) ** 2
        return total

    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    assert result.term_a.ss == pytest.approx(one_way_between(values, factor_a), abs=1e-9)
    assert result.term_b.ss == pytest.approx(one_way_between(values, factor_b), abs=1e-9)


def test_two_way_anova_holds_for_an_unbalanced_design(
    analyzer: DesignTestAnalyzer,
) -> None:
    """The same identities must survive unequal cell sizes.

    Cell sizes and means: a1b1 = 2 @ 12, a1b2 = 2 @ 22, a2b1 = 3 @ 82/3,
    a2b2 = 2 @ 14, grand mean M = 178/9, A means 17 and 22.

    SS_A = 4*(17 - 178/9)^2 + 5*(22 - 178/9)^2
         = 4*2500/81 + 5*1600/81 = (10000 + 8000)/81 = 500/9.

    The weighted marginal means are *not* orthogonal here, so the naive
    closed forms for B and AB give 22.7556 and 297.6889, which sum with A
    and the error term to 538.67 against SS_total = 531.56. The nested
    least-squares decomposition used instead closes exactly.
    """
    values = np.array([10.0, 14.0, 20.0, 24.0, 30.0, 34.0, 12.0, 16.0, 18.0])
    factor_a = ["a1", "a1", "a1", "a1", "a2", "a2", "a2", "a2", "a2"]
    factor_b = ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2", "b1"]
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    assert not result.balanced
    assert result.df_error == 9 - 4
    assert (result.term_a.df, result.term_b.df, result.term_interaction.df) == (1, 1, 1)
    assert result.term_a.ss == pytest.approx(500.0 / 9.0, abs=1e-9)
    assert result.ss_total == pytest.approx(531.5555555555555, abs=1e-9)
    # Closure exact, which the naive weighted-marginal terms cannot do here.
    assert (
        result.term_a.ss
        + result.term_b.ss
        + result.term_interaction.ss
        + result.ss_error
    ) == pytest.approx(result.ss_total, abs=1e-9)
    # And the one-way identity still holds for the main effect.
    grand = float(np.mean(values))
    expected_a = 0.0
    for level in {"a1", "a2"}:
        chunk = values[[i for i, v in enumerate(factor_a) if v == level]]
        expected_a += len(chunk) * (float(np.mean(chunk)) - grand) ** 2
    assert result.term_a.ss == pytest.approx(expected_a, abs=1e-9)


def test_two_way_anova_error_is_the_within_cell_sum(analyzer: DesignTestAnalyzer) -> None:
    """SS_error must be sum (y - cell mean)^2, not a leftover by subtraction."""
    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    within = 0.0
    for a in {"a1", "a2"}:
        for b in {"b1", "b2"}:
            chunk = values[
                [i for i in range(len(values)) if factor_a[i] == a and factor_b[i] == b]
            ]
            within += float(np.sum((chunk - np.mean(chunk)) ** 2))
    assert result.ss_error == pytest.approx(within, abs=1e-12)


def test_two_way_anova_uses_the_residual_in_every_denominator(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Each F divides by MS_error -- not by MS_A, MS_B or MS_AB.

    The classic factorial bug is to build a "sequential" denominator for
    the interaction out of MS_B + MS_AB, which still returns a plausible
    looking F. Asserting the denominator explicitly is the check.
    """
    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    ms_error = result.ss_error / result.df_error
    for term in result.terms:
        expected = (term.ss / term.df) / ms_error
        assert term.f_statistic == pytest.approx(expected, abs=1e-9)
    assert result.term_interaction.f_statistic != pytest.approx(
        (result.term_interaction.ss / result.term_interaction.df)
        / ((result.term_b.ss / result.term_b.df)
           + (result.term_interaction.ss / result.term_interaction.df)),
        abs=1e-6,
    )


def test_two_way_anova_p_values_match_scipy(analyzer: DesignTestAnalyzer) -> None:
    """The upper-tail probability comes from scipy, which is independent."""
    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    for term in result.terms:
        expected = float(sp_stats.f.sf(term.f_statistic, term.df, result.df_error))
        assert term.p_value == pytest.approx(expected, abs=1e-12)


def test_two_way_anova_interaction_vanishes_for_additive_cells(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Cell means that are exactly additive leave no interaction at all.

    Cell means 10, 20, 30, 40 with grand mean 25, A means 15/35 and B
    means 20/30 give a fitted additive surface of 10, 20, 30, 40 -- the
    observed means -- so every interaction contrast is exactly zero and
    SS_AB must be 0, not merely small.
    """
    values = np.array([9.0, 11.0, 19.0, 21.0, 29.0, 31.0, 39.0, 41.0])
    factor_a = ["a1", "a1", "a1", "a1", "a2", "a2", "a2", "a2"]
    factor_b = ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"]
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)

    assert result.term_interaction.ss == pytest.approx(0.0, abs=1e-12)
    assert result.term_interaction.f_statistic == pytest.approx(0.0, abs=1e-12)
    assert result.term_interaction.p_value == pytest.approx(1.0, abs=1e-12)
    # The main effects survive: SS_A = 4*100 + 4*100 = 800,
    # SS_B = 4*25 + 4*25 = 200, SS_error = 4 cells * 2 = 8, MS_error = 2.
    assert result.term_a.f_statistic == pytest.approx(400.0, abs=1e-9)
    assert result.term_b.f_statistic == pytest.approx(100.0, abs=1e-9)


def test_two_way_anova_reports_a_missing_response_row(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A non-finite response is dropped, and the drop is reported."""
    values, factor_a, factor_b = _factorial_hand_case()
    values = np.append(values, np.nan)
    factor_a = [*factor_a, "a1"]
    factor_b = [*factor_b, "b1"]
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)
    assert result.n_missing == 1
    assert result.n_obs == 8


def test_two_way_anova_permutation_is_reproducible(analyzer: DesignTestAnalyzer) -> None:
    """Same seed, same permutation p-values."""
    values, factor_a, factor_b = _factorial_hand_case()
    first = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=99, random_seed=7)
    second = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=99, random_seed=7)
    for a, b in zip(first.terms, second.terms, strict=True):
        assert a.permutation_p_value == b.permutation_p_value


def test_two_way_anova_permutation_p_values_are_calibrated(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Under the null the permutation p-value must not reject too often.

    A permutation that reshuffles the wrong array, or compares against
    the wrong tail, still returns a number in [0, 1]; this is the check
    that has teeth.
    """
    rng = np.random.default_rng(21)
    p_values = []
    for seed in range(20):
        values = rng.normal(size=32)
        factor_a = ["a1"] * 8 + ["a2"] * 8 + ["a1"] * 8 + ["a2"] * 8
        factor_b = ["b1"] * 16 + ["b2"] * 16
        result = analyzer.two_way_anova(
            values, factor_a, factor_b, n_permutations=99, random_seed=seed
        )
        p_values.append(result.term_a.permutation_p_value)
    assert float(np.mean(np.asarray(p_values) < 0.05)) <= 0.25
    assert min(p_values) >= 1.0 / 100 - 1e-12


def test_two_way_anova_unseeded_permutation_leaves_the_global_stream_alone(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Running the analysis must not consume the caller's random numbers."""
    values, factor_a, factor_b = _factorial_hand_case()
    np.random.seed(1234)
    expected = np.random.rand(4)
    np.random.seed(1234)
    with pytest.warns(RuntimeWarning):
        analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=49)
    np.testing.assert_allclose(np.random.rand(4), expected)


# ---------------------------------------------------------------------------
# Two-way ANOVA -- input that cannot produce an answer
# ---------------------------------------------------------------------------


def test_two_way_anova_rejects_mismatched_label_counts(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A short label array would silently drop the tail of the data."""
    values, factor_a, factor_b = _factorial_hand_case()
    with pytest.raises(MatrixDimensionError):
        analyzer.two_way_anova(values, factor_a[:-2], factor_b, n_permutations=0)


def test_two_way_anova_rejects_a_single_level_factor(analyzer: DesignTestAnalyzer) -> None:
    """One level has no degrees of freedom to test."""
    values, _, factor_b = _factorial_hand_case()
    with pytest.raises(ValidationError, match="at least 2 levels"):
        analyzer.two_way_anova(values, ["a1"] * 8, factor_b, n_permutations=0)


def test_two_way_anova_rejects_a_design_without_residual_df(
    analyzer: DesignTestAnalyzer,
) -> None:
    """One observation per cell cannot estimate the denominator of F."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    factor_a = ["a1", "a1", "a2", "a2"]
    factor_b = ["b1", "b2", "b1", "b2"]
    with pytest.raises((DataValidationError, MatrixDimensionError, ValidationError)):
        analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)


def test_two_way_anova_rejects_constant_data(analyzer: DesignTestAnalyzer) -> None:
    """No variation means nothing to decompose."""
    values = np.full(8, 3.0)
    factor_a = ["a1"] * 4 + ["a2"] * 4
    factor_b = ["b1"] * 2 + ["b2"] * 2 + ["b1"] * 2 + ["b2"] * 2
    with pytest.raises(ComputationError, match="constant"):
        analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)


def test_two_way_anova_rejects_a_missing_factor_label(analyzer: DesignTestAnalyzer) -> None:
    """A NaN label is a data error, not an extra level."""
    values, factor_a, factor_b = _factorial_hand_case()
    factor_a[0] = float("nan")
    with pytest.raises(DataValidationError):
        analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)


def test_two_way_anova_rejects_a_multi_column_response(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Pooling several variables into one response would change the analysis."""
    values = np.ones((8, 3))
    factor_a = ["a1"] * 4 + ["a2"] * 4
    factor_b = ["b1"] * 2 + ["b2"] * 2 + ["b1"] * 2 + ["b2"] * 2
    with pytest.raises(MatrixDimensionError):
        analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)


def test_two_way_anova_result_serialises(analyzer: DesignTestAnalyzer) -> None:
    """to_dict and summary must both work and stay JSON-friendly."""
    values, factor_a, factor_b = _factorial_hand_case()
    result = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=49, random_seed=3)
    payload = result.to_dict()
    assert payload["term_a"]["f_statistic"] == pytest.approx(9.0, abs=1e-9)
    assert isinstance(payload["ss_total"], float)
    assert isinstance(payload["balanced"], bool)
    assert "Two-Way ANOVA" in result.summary()


# ---------------------------------------------------------------------------
# Repeated measures -- a design small enough to work out on paper
# ---------------------------------------------------------------------------


def _repeated_measures_hand_case() -> tuple[npt.NDArray, list[str], list[str]]:
    """2 groups x 2 subjects x 3 measurements, arranged so the SS are simple.

    Subject means are 10, 20 (group 1) and 30, 40 (group 2); the
    subject-centred profiles z are
        A = (3, -1, -2)   B = (0, 1, -1)   C = (-2, 0, 2)   D = (1, 0, -1)
    so the reconstruction is y = offset + z.
    """
    matrix = np.array(
        [
            [13.0, 9.0, 8.0],
            [20.0, 21.0, 19.0],
            [28.0, 30.0, 32.0],
            [41.0, 40.0, 39.0],
        ]
    )
    subjects = [s for s in "ABCD" for _ in range(3)]
    groups = ["g1"] * 6 + ["g2"] * 6
    return matrix.ravel(), subjects, groups


def test_repeated_measures_matches_the_hand_derived_decomposition(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Every sum of square, worked out by hand from Howell (2013) ch. 12.

    Between-subject stratum, on the subject means 10, 20, 30, 40:
      group means 15 and 35, grand mean 25
      SS_groups   = k * n_per_group * [(15-25)^2 + (35-25)^2] = 3*2*200 = 1200
      SS_subjects = k * [(10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2]
                  = 3 * 500 = 1500
      SS_subjects within groups = 1500 - 1200 = 300

    Within-subject stratum, on z = y - subject mean:
      measurement means over all 4 subjects: (0.5, 0, -0.5)
      SS_measurement = 4 * (0.25 + 0 + 0.25) = 2
      group means of z: g1 = (1.5, 0, -1.5), g2 = (-0.5, 0, 0.5)
      SS_interaction = 2 * [(1.5-0.5)^2 + 0 + (-1.5+0.5)^2
                            + (-0.5-0.5)^2 + 0 + (0.5+0.5)^2]
                     = 2 * 4 = 8
      sum of z^2 = 14 + 2 + 8 + 2 = 26
      SS_error = 26 - 2 - 8 = 16

    df: groups = 1, subjects = g*(n-1) = 2, measurement = k-1 = 2,
        interaction = (g-1)(k-1) = 2, error = g*(n-1)*(k-1) = 4,
        total = 2*2*3 - 1 = 11.
    MS: groups 1200, subjects 150, measurement 1, interaction 4, error 4.
    F: groups = 1200/150 = 8, measurement = 1/4 = 0.25,
       interaction = 4/4 = 1.
    """
    values, subjects, groups = _repeated_measures_hand_case()
    result = analyzer.repeated_measures_anova(values, subjects, groups)

    assert result.term_groups.ss == pytest.approx(1200.0, abs=1e-9)
    assert result.term_subjects.ss == pytest.approx(300.0, abs=1e-9)
    assert result.term_measurement.ss == pytest.approx(2.0, abs=1e-9)
    assert result.term_interaction.ss == pytest.approx(8.0, abs=1e-9)
    assert result.term_error.ss == pytest.approx(16.0, abs=1e-9)
    assert result.ss_total == pytest.approx(1526.0, abs=1e-9)

    assert result.term_groups.df == 1
    assert result.term_subjects.df == 2
    assert result.term_measurement.df == 2
    assert result.term_interaction.df == 2
    assert result.term_error.df == 4
    assert result.df_total == 11
    assert sum(t.df for t in result.tested_terms) + result.term_subjects.df + result.term_error.df == 11

    assert result.term_groups.f_statistic == pytest.approx(8.0, abs=1e-9)
    assert result.term_measurement.f_statistic == pytest.approx(0.25, abs=1e-9)
    assert result.term_interaction.f_statistic == pytest.approx(1.0, abs=1e-9)


def test_repeated_measures_group_effect_uses_the_between_subject_error(
    analyzer: DesignTestAnalyzer,
) -> None:
    """The group F must divide by MS_subjects, not by the within-subject error.

    In the hand case MS_subjects = 150 and MS_error = 4, so a design that
    put the wrong denominator in place returns 300 instead of 8. Asserting
    the value the right way round is what catches it.
    """
    values, subjects, groups = _repeated_measures_hand_case()
    result = analyzer.repeated_measures_anova(values, subjects, groups)

    assert result.term_subjects.ms == pytest.approx(150.0, abs=1e-9)
    assert result.term_error.ms == pytest.approx(4.0, abs=1e-9)
    assert result.term_groups.f_statistic == pytest.approx(
        result.term_groups.ms / result.term_subjects.ms, abs=1e-12
    )
    assert result.term_groups.f_statistic != pytest.approx(
        result.term_groups.ms / result.term_error.ms, rel=0.01
    )


def test_repeated_measures_reduces_to_the_textbook_one_factor_case(
    analyzer: DesignTestAnalyzer,
) -> None:
    """With a single group the design collapses to Howell's one-way form.

    Four subjects measured three times:
        subject means 2, 2, 6, 6; grand mean 4
        SS_total      = sum (y - 4)^2 = 56
        SS_subjects   = 3 * [2*(2-4)^2 + 2*(6-4)^2] = 48   (between-subject
                       stratum, all of it "error" because there is no group
                       effect to test)
        SS_measurement= 4 * [(3.5-4)^2 + (4.5-4)^2 + (4-4)^2] = 4 * 0.5 = 2
        SS_error      = 56 - 48 - 2 = 6,  df = 3*2 = 6
        F             = (2/2)/(6/6) = 1
    """
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 5.0],
        ]
    )
    subjects = [f"s{i}" for i in range(4) for _ in range(3)]
    groups = ["g"] * 12
    result = analyzer.repeated_measures_anova(matrix.ravel(), subjects, groups)

    assert result.term_groups.df == 0
    assert not np.isfinite(result.term_groups.f_statistic)
    assert result.term_subjects.df == 3
    assert result.term_subjects.ss == pytest.approx(48.0, abs=1e-9)
    assert result.term_measurement.ss == pytest.approx(2.0, abs=1e-9)
    assert result.term_error.df == 6
    assert result.term_error.ss == pytest.approx(6.0, abs=1e-9)
    assert result.term_measurement.f_statistic == pytest.approx(1.0, abs=1e-9)
    assert result.df_total == 11
    # The within-subject residual, computed here straight from the raw
    # values, must equal the within-subject stratum in full.
    residual = float(np.sum((matrix - matrix.mean(axis=1, keepdims=True)) ** 2))
    assert result.term_measurement.ss + result.term_error.ss == pytest.approx(
        residual, abs=1e-9
    )


def test_repeated_measures_accepts_an_explicit_measurement_label(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Shuffled rows plus measurement labels must give the same answer."""
    values, subjects, groups = _repeated_measures_hand_case()
    expected = analyzer.repeated_measures_anova(values, subjects, groups)

    order = np.array([7, 0, 3, 11, 2, 5, 8, 1, 10, 4, 9, 6])
    measurement = [i % 3 for i in order]
    shuffled = analyzer.repeated_measures_anova(
        values[order],
        [subjects[i] for i in order],
        [groups[i] for i in order],
        measurement=measurement,
    )
    assert shuffled.term_measurement.f_statistic == pytest.approx(
        expected.term_measurement.f_statistic, abs=1e-9
    )
    assert shuffled.term_interaction.f_statistic == pytest.approx(
        expected.term_interaction.f_statistic, abs=1e-9
    )


def test_repeated_measures_p_values_match_scipy(analyzer: DesignTestAnalyzer) -> None:
    """Both strata's p-values come from scipy's F tail."""
    values, subjects, groups = _repeated_measures_hand_case()
    result = analyzer.repeated_measures_anova(values, subjects, groups)

    assert result.term_groups.p_value == pytest.approx(
        float(sp_stats.f.sf(result.term_groups.f_statistic, result.term_groups.df, 2)), abs=1e-12
    )
    assert result.term_measurement.p_value == pytest.approx(
        float(sp_stats.f.sf(result.term_measurement.f_statistic, 2, result.term_error.df)), abs=1e-12
    )
    assert result.term_interaction.p_value == pytest.approx(
        float(sp_stats.f.sf(result.term_interaction.f_statistic, 2, result.term_error.df)), abs=1e-12
    )


def test_repeated_measures_result_serialises(analyzer: DesignTestAnalyzer) -> None:
    """to_dict and summary must both work and stay JSON-friendly."""
    values, subjects, groups = _repeated_measures_hand_case()
    result = analyzer.repeated_measures_anova(values, subjects, groups)
    payload = result.to_dict()
    assert payload["term_groups"]["f_statistic"] == pytest.approx(8.0, abs=1e-9)
    assert payload["n_subjects"] == 4
    assert isinstance(payload["significant_terms"], list)
    assert "Repeated-Measures ANOVA" in result.summary()


# ---------------------------------------------------------------------------
# Repeated measures -- the unbalanced design is refused, not repaired
# ---------------------------------------------------------------------------


def test_repeated_measures_rejects_an_incomplete_subject(
    analyzer: DesignTestAnalyzer,
) -> None:
    """One missing measurement must be an error, not a shorter average.

    Silently averaging what the subject does have would change both
    strata while leaving the row count alone, and the resulting F values
    look entirely ordinary -- which is exactly the failure this test
    exists to prevent.
    """
    values, subjects, groups = _repeated_measures_hand_case()
    # Index 4 is subject B's second measurement (rows are subject-major:
    # A at 0-2, B at 3-5, C at 6-8, D at 9-11).
    keep = [i for i in range(len(values)) if i != 4]
    assert subjects[4] == "B"
    with pytest.raises(MatrixDimensionError, match="all 3 measurements"):
        analyzer.repeated_measures_anova(
            values[keep], [subjects[i] for i in keep], [groups[i] for i in keep]
        )


def test_repeated_measures_rejects_a_subject_in_two_groups(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A subject cannot straddle the between-subject factor."""
    values, subjects, groups = _repeated_measures_hand_case()
    groups = list(groups)
    groups[0] = "g2"
    with pytest.raises(MatrixDimensionError, match="cannot belong"):
        analyzer.repeated_measures_anova(values, subjects, groups)


def test_repeated_measures_rejects_unequal_group_sizes(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A complete but unbalanced design is still not a Type I decomposition."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    subjects = ["s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3", "s3"]
    groups = ["g1"] * 6 + ["g2"] * 3
    with pytest.raises(MatrixDimensionError, match="balanced"):
        analyzer.repeated_measures_anova(values, subjects, groups)


def test_repeated_measures_rejects_mismatched_label_counts(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A short label array would silently drop observations."""
    values, subjects, groups = _repeated_measures_hand_case()
    with pytest.raises(MatrixDimensionError):
        analyzer.repeated_measures_anova(values, subjects[:-1], groups)


def test_repeated_measures_rejects_a_single_measurement(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Without repeated measurements there is nothing to repeat."""
    with pytest.raises(ValidationError, match="at least 2 measurements"):
        analyzer.repeated_measures_anova(np.array([1.0, 2.0, 3.0, 4.0]), ["a", "b", "c", "d"], ["g"] * 4)


def test_repeated_measures_rejects_constant_data(analyzer: DesignTestAnalyzer) -> None:
    """No variation means nothing to decompose."""
    subjects = [s for s in "ABCD" for _ in range(3)]
    with pytest.raises(ComputationError, match="constant"):
        analyzer.repeated_measures_anova(np.full(12, 5.0), subjects, ["g1"] * 6 + ["g2"] * 6)


def test_repeated_measures_rejects_missing_values(analyzer: DesignTestAnalyzer) -> None:
    """A missing measurement is reported, not dropped from the series."""
    values, subjects, groups = _repeated_measures_hand_case()
    values = values.copy()
    values[0] = np.nan
    with pytest.raises(DataValidationError):
        analyzer.repeated_measures_anova(values, subjects, groups)


# ---------------------------------------------------------------------------
# ICC -- a table small enough to work out on paper
# ---------------------------------------------------------------------------


def _icc_hand_case() -> npt.NDArray:
    """3 targets x 2 raters.

    Grand mean = (1+2+2+3+4+6)/6 = 3.
    Row (target) means 1.5, 2.5, 5;  column (rater) means 3.5 and 11/3.
    SS_total   = 4+1+1+0+1+9 = 16
    SS_targets = 2 * [(-1.5)^2 + (-0.5)^2 + 2^2] = 2 * 6.5 = 13
    SS_raters  = 3 * [(3.5-3)^2 + (11/3-3)^2] = 3 * 8/9 = 8/3
    SS_error   = 16 - 13 - 8/3 = 1/3
    MS_targets = 13/2 = 6.5,  MS_raters = 8/3,  MS_error = (1/3)/2 = 1/6.
    """
    return np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 6.0]])


def test_icc_matches_the_hand_derived_mean_squares(
    analyzer: DesignTestAnalyzer,
) -> None:
    """The mean squares and both ICC forms, worked out by hand.

    ICC(3,1) = (6.5 - 1/6)/(6.5 + 1/6) = (38/6)/(40/6) = 0.95
    ICC(2,1) = (38/6)/(6.5 + 1/6 + 2*(8/3 - 1/6)/3)
             = (38/6)/(40/6 + 5/3) = (38/6)/(50/6) = 0.76

    The formulas are Shrout & Fleiss (1979) eq. 6 and eq. 7; the forms are
    named as in McGraw & Wong (1996).
    """
    table = _icc_hand_case()
    result_31 = analyzer.intraclass_correlation(table, form="3,1")
    result_21 = analyzer.intraclass_correlation(table, form="2,1")

    assert result_31.ms_targets == pytest.approx(6.5, abs=1e-12)
    assert result_31.ms_raters == pytest.approx(8 / 3, abs=1e-12)
    assert result_31.ms_error == pytest.approx(1 / 6, abs=1e-12)
    assert result_31.df_error == 2
    assert result_31.icc == pytest.approx(0.95, abs=1e-12)
    assert result_21.icc == pytest.approx(0.76, abs=1e-12)


def test_icc_equals_the_variance_component_ratio(analyzer: DesignTestAnalyzer) -> None:
    """Both forms must equal sigma^2_targets / (sigma^2_targets + ...).

    This is the definitional cross-check: the moment estimates are

        sigma^2_t = (MS_targets - MS_error) / k
        sigma^2_c = (MS_raters  - MS_error) / n
        sigma^2_e = MS_error

    and the two closed forms in the module must reproduce the ratio they
    estimate, computed here by a different route.
    """
    table = _icc_hand_case()
    result = analyzer.intraclass_correlation(table, form="2,1")
    ms_targets, ms_raters, ms_error = result.ms_targets, result.ms_raters, result.ms_error
    n_targets, n_raters = result.n_targets, result.n_raters

    sigma_t = (ms_targets - ms_error) / n_raters
    sigma_c = (ms_raters - ms_error) / n_targets
    expected = sigma_t / (sigma_t + sigma_c + ms_error)
    assert result.icc == pytest.approx(expected, abs=1e-12)
    assert result.icc == pytest.approx(0.76, abs=1e-12)

    consistency = analyzer.intraclass_correlation(table, form="3,1")
    sigma_t31 = (ms_targets - ms_error) / n_raters
    assert consistency.icc == pytest.approx(sigma_t31 / (sigma_t31 + ms_error), abs=1e-12)


def test_icc_error_ss_equals_the_within_cell_sum(analyzer: DesignTestAnalyzer) -> None:
    """SS_error must be sum (y - target mean - rater mean + grand)^2."""
    table = _icc_hand_case()
    result = analyzer.intraclass_correlation(table, form="3,1")
    grand = float(np.mean(table))
    residuals = table - table.mean(axis=1, keepdims=True) - table.mean(axis=0, keepdims=True) + grand
    assert result.ms_error == pytest.approx(float(np.sum(residuals**2)) / 2, abs=1e-12)


def test_icc_is_one_without_any_within_target_error(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Identical repeats leave nothing for the error term, so ICC = 1.

    y = [[1,1],[2,2],[3,3],[4,4]]: MS_error = 0 exactly, so the moment
    estimator has no sampling error and the interval collapses onto the
    point estimate rather than being given an invented width.
    """
    table = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    for form in ("2,1", "3,1"):
        result = analyzer.intraclass_correlation(table, form=form)
        assert result.icc == pytest.approx(1.0, abs=1e-12)
        assert result.ci_lower == pytest.approx(1.0, abs=1e-12)
        assert result.ci_upper == pytest.approx(1.0, abs=1e-12)


def test_icc_is_zero_without_between_target_signal(analyzer: DesignTestAnalyzer) -> None:
    """A table whose target means equal its error gives ICC = 0 exactly.

    y = [[0,0],[1,0]]: grand mean 1/4, target means 0 and 1/2,
    MS_targets = 2*[(-1/4)^2 + (1/4)^2]/1 = 1/4, and the residual sum of
    squares is 1/4, so MS_error = 1/4 too. Then ICC(3,1) = 0/(1/4+1/4) = 0
    exactly, not "close to zero".
    """
    table = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = analyzer.intraclass_correlation(table, form="3,1")
    assert result.ms_targets == pytest.approx(0.25, abs=1e-12)
    assert result.ms_error == pytest.approx(0.25, abs=1e-12)
    assert result.icc == pytest.approx(0.0, abs=1e-12)
    assert not result.significant


def test_icc_absolute_agreement_penalises_a_rater_offset(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A constant rater bias lowers ICC(2,1) and leaves ICC(3,1) alone.

    Adding 10 to the second column does not change the target means'
    deviations, nor the residuals, so MS_targets and MS_error -- and
    therefore the consistency coefficient -- are unchanged. MS_raters
    absorbs the bias, which is exactly what the consistency form ignores.
    """
    table = _icc_hand_case()
    biased = table + np.array([0.0, 10.0])

    consistent = analyzer.intraclass_correlation(table, form="3,1")
    consistent_biased = analyzer.intraclass_correlation(biased, form="3,1")
    assert consistent_biased.icc == pytest.approx(consistent.icc, abs=1e-12)

    agreement = analyzer.intraclass_correlation(table, form="2,1")
    agreement_biased = analyzer.intraclass_correlation(biased, form="2,1")
    assert agreement_biased.ms_targets == pytest.approx(agreement.ms_targets, abs=1e-12)
    assert agreement_biased.ms_error == pytest.approx(agreement.ms_error, abs=1e-12)
    assert agreement_biased.icc < agreement.icc


def test_icc_long_form_matches_the_matrix_form(analyzer: DesignTestAnalyzer) -> None:
    """The long form is the same table, and must give the same numbers."""
    table = _icc_hand_case()
    values = table.ravel()
    targets = [f"t{i}" for i in range(3) for _ in range(2)]
    groups = ["r1", "r2"] * 3
    for form in ("2,1", "3,1"):
        direct = analyzer.intraclass_correlation(table, form=form)
        long_form = analyzer.intraclass_correlation(values, targets, groups, form=form)
        assert long_form.icc == pytest.approx(direct.icc, abs=1e-12)
        assert long_form.ci_lower == pytest.approx(direct.ci_lower, abs=1e-12)
        assert long_form.ci_upper == pytest.approx(direct.ci_upper, abs=1e-12)


def test_icc_interval_contains_the_point_estimate(analyzer: DesignTestAnalyzer) -> None:
    """A coefficient reported without a usable interval is not reportable."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        table = rng.normal(scale=2.0, size=(9, 3)) + rng.normal(size=(9, 1)) * 3.0
        for form in ("2,1", "3,1"):
            result = analyzer.intraclass_correlation(table, form=form)
            assert result.ci_lower <= result.icc <= result.ci_upper
            assert -1.0 <= result.ci_lower <= 1.0
            assert -1.0 <= result.ci_upper <= 1.0


def test_icc_interval_covers_a_known_coefficient(analyzer: DesignTestAnalyzer) -> None:
    """Coverage simulation: the interval must contain the truth ~95 % of runs.

    Data are drawn from the two-way random-effects model the coefficient
    is defined for, y_ij = alpha_i + beta_j + e_ij with all three variances
    equal to 1, so the population values are
        ICC(3,1) = 1/(1+1)     = 1/2
        ICC(2,1) = 1/(1+1+1)   = 1/3
    and the interval is checked against those, not against the estimate.
    The 2,1 form is built from two mean-square ratios that share MS_error,
    so it is Bonferroni-combined and allowed to be conservative -- the
    bound is one-sided for that reason.
    """
    rng = np.random.default_rng(20240906)
    reps = 400
    covered_31 = 0
    covered_21 = 0
    for _ in range(reps):
        table = rng.normal(size=(8, 1)) + rng.normal(size=(1, 3)) + rng.normal(size=(8, 3))
        r31 = analyzer.intraclass_correlation(table, form="3,1")
        covered_31 += r31.ci_lower <= 0.5 <= r31.ci_upper
        r21 = analyzer.intraclass_correlation(table, form="2,1")
        covered_21 += r21.ci_lower <= 1 / 3 <= r21.ci_upper
    # Monte-Carlo error at 400 reps is about +/-0.02 at the 95% level.
    assert covered_31 / reps >= 0.90
    assert covered_21 / reps >= 0.90


def test_icc_result_serialises(analyzer: DesignTestAnalyzer) -> None:
    """to_dict and summary must both work and stay JSON-friendly."""
    result = analyzer.intraclass_correlation(_icc_hand_case(), form="2,1")
    payload = result.to_dict()
    assert payload["icc"] == pytest.approx(0.76, abs=1e-12)
    assert isinstance(payload["variance_components"], dict)
    assert "ICC" in result.summary()


# ---------------------------------------------------------------------------
# ICC -- input that cannot produce an answer
# ---------------------------------------------------------------------------


def test_icc_rejects_an_unknown_form(analyzer: DesignTestAnalyzer) -> None:
    """Only the two implemented forms are accepted."""
    with pytest.raises(ValidationError):
        analyzer.intraclass_correlation(_icc_hand_case(), form="3,k")


def test_icc_rejects_a_degenerate_confidence_level(
    analyzer: DesignTestAnalyzer,
) -> None:
    """An interval at 0 or 100 % is not an interval."""
    with pytest.raises(ValidationError):
        analyzer.intraclass_correlation(_icc_hand_case(), confidence=1.0)


def test_icc_rejects_a_table_that_is_too_small(analyzer: DesignTestAnalyzer) -> None:
    """One rater leaves no residual mean square."""
    with pytest.raises((MatrixDimensionError, ValidationError)):
        analyzer.intraclass_correlation(np.array([1.0, 2.0, 3.0, 4.0]))


def test_icc_rejects_a_constant_table(analyzer: DesignTestAnalyzer) -> None:
    """Zero total variation leaves nothing to attribute."""
    with pytest.raises(ComputationError, match="constant"):
        analyzer.intraclass_correlation(np.ones((4, 3)))


def test_icc_rejects_a_duplicated_target_rater_cell(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A cell must hold exactly one measurement, or the table is ambiguous."""
    values = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 6.0])
    targets = ["t1", "t1", "t2", "t2", "t3", "t3"]
    groups = ["r1", "r2", "r1", "r2", "r1", "r1"]
    with pytest.raises(MatrixDimensionError, match="more than once"):
        analyzer.intraclass_correlation(values, targets, groups)


def test_icc_rejects_missing_measurements(analyzer: DesignTestAnalyzer) -> None:
    """An incomplete table is a data error, not something to impute."""
    table = _icc_hand_case()
    table[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        analyzer.intraclass_correlation(table)


def test_icc_long_form_needs_both_label_arrays(
    analyzer: DesignTestAnalyzer,
) -> None:
    """A long-form ICC is meaningless without knowing what was measured."""
    with pytest.raises(ValidationError):
        analyzer.intraclass_correlation(np.array([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# Contingency chi-square
# ---------------------------------------------------------------------------


def test_contingency_chi_square_matches_a_closed_form(
    analyzer: DesignTestAnalyzer,
) -> None:
    """Expected counts and chi-square computed by hand on a 2x2 table.

    Observed [[10, 20], [30, 40]], N = 100, row totals 30 and 70, column
    totals 40 and 60, so the expected counts are 12, 18, 28, 42.
    chi-square = (10-12)^2/12 + (20-18)^2/18 + (30-28)^2/28 + (40-42)^2/42
               = 4/12 + 4/18 + 4/28 + 4/42 = 0.793650...
    Cramer's V = sqrt(chi-square / (N * (min(r,c) - 1))) = sqrt(0.79365/100).
    """
    table = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = analyzer.contingency_chi_square(table)

    expected = 4 / 12 + 4 / 18 + 4 / 28 + 4 / 42
    assert result.chi_square == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(result.expected, [[12.0, 18.0], [28.0, 42.0]], atol=1e-12)
    assert result.dof == 1
    assert result.cramers_v == pytest.approx(float(np.sqrt(expected / 100.0)), abs=1e-12)
    assert result.n_obs == 100
    assert result.min_expected == pytest.approx(12.0, abs=1e-12)


def test_contingency_chi_square_matches_scipy(analyzer: DesignTestAnalyzer) -> None:
    """scipy's uncorrected chi-square is an independent implementation."""
    rng = np.random.default_rng(17)
    table = rng.integers(5, 60, size=(3, 4)).astype(float)
    result = analyzer.contingency_chi_square(table)
    reference = sp_stats.chi2_contingency(table, correction=False)
    assert result.chi_square == pytest.approx(float(reference[0]), abs=1e-9)
    assert result.p_value == pytest.approx(float(reference[1]), abs=1e-12)


def test_contingency_chi_square_warns_on_small_expected_counts(
    analyzer: DesignTestAnalyzer,
) -> None:
    """The approximation is weak below 5 and the result has to say so."""
    result = analyzer.contingency_chi_square(np.array([[1.0, 2.0], [1.0, 3.0]]))
    assert result.min_expected < 5.0
    assert "expected count" in result.summary()


def test_contingency_chi_square_rejects_a_bad_table(analyzer: DesignTestAnalyzer) -> None:
    """A 1xN table has no degrees of freedom to test."""
    with pytest.raises(MatrixDimensionError):
        analyzer.contingency_chi_square(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(DataValidationError):
        analyzer.contingency_chi_square(np.array([[-1.0, 2.0], [3.0, 4.0]]))


# ---------------------------------------------------------------------------
# Analyzer housekeeping
# ---------------------------------------------------------------------------


def test_last_result_is_retained(analyzer: DesignTestAnalyzer) -> None:
    """The analyzer remembers its most recent run."""
    assert analyzer.last_result is None
    result = analyzer.intraclass_correlation(_icc_hand_case(), form="3,1")
    assert analyzer.last_result is result


def test_all_three_designs_share_one_analyzer(analyzer: DesignTestAnalyzer) -> None:
    """One engine, three designs, one last_result slot."""
    values, factor_a, factor_b = _factorial_hand_case()
    two_way = analyzer.two_way_anova(values, factor_a, factor_b, n_permutations=0)
    assert analyzer.last_result is two_way

    rm_values, subjects, groups = _repeated_measures_hand_case()
    repeated = analyzer.repeated_measures_anova(rm_values, subjects, groups)
    assert analyzer.last_result is repeated

    icc = analyzer.intraclass_correlation(_icc_hand_case())
    assert analyzer.last_result is icc


def test_the_stochastic_free_paths_are_reproducible(analyzer: DesignTestAnalyzer) -> None:
    """Repeated measures and the ICC draw no random numbers at all.

    Both are exact arithmetic, so repeating the call must reproduce the
    result to the last bit -- which is the property that makes them safe
    to quote in a table without a seed.
    """
    values, subjects, groups = _repeated_measures_hand_case()
    first = analyzer.repeated_measures_anova(values, subjects, groups)
    second = analyzer.repeated_measures_anova(values, subjects, groups)
    assert first.to_dict() == second.to_dict()

    table = _icc_hand_case()
    for form in ("2,1", "3,1"):
        a = analyzer.intraclass_correlation(table, form=form)
        b = analyzer.intraclass_correlation(table, form=form)
        assert a.to_dict() == b.to_dict()
