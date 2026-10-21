# =============================================================================
# FILE: tests/property/test_design_properties.py
# =============================================================================
"""
Property-based tests for the design tests: the two-way ANOVA, the
repeated-measures ANOVA, the intraclass correlation and the contingency
chi-square.

These are the analyses where the *decomposition* is the result, so the
invariants are about the decomposition:

  * the terms and the residual must add up to the total, exactly -- a
    decomposition that does not add up is wrong in a way no single F
    ratio would reveal;
  * the degrees of freedom must add up to ``n - 1``;
  * a positive affine transform of the response changes the numbers and
    not the conclusions;
  * renaming the factor levels, or permuting the rows inside a cell,
    cannot change a p-value: the design is the same design;
  * the intraclass correlation is a correlation, so it lives in
    ``[-1, 1]``, and its variance components are the total variance;
  * the chi-square of a contingency table cannot depend on the order of
    the rows or the columns.

References:
    Scheffe, H. (1959). The Analysis of Variance. Wiley.
    McGraw, K.O. & Wong, S.P. (1996). Forming inferences about some
    intraclass correlation coefficients. Psychological Methods 1, 30-46.
    Agresti, A. (2018). Categorical Data Analysis. 3rd ed. Wiley.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stats.design_tests import DesignTestAnalyzer

from ._helpers import affine, call_or_none, has_variation, integer_strategy, scale_strategy, shift_strategy

#: The tolerances.  The arithmetic on the transformed data goes through
#: different intermediate values, so the comparison is made at the
#: precision the arithmetic actually has.  A wrong implementation is
#: wrong by a factor, not by the last few bits.
RTOL = 1e-7
ATOL = 1e-9

_SETTING = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _two_by_two(values: list[int]) -> tuple[np.ndarray, list[str], list[str]]:
    """A 2x2 design with the same number of observations in every cell."""
    array = np.asarray(values, dtype=float)
    per_cell = array.size // 4
    if per_cell < 2:
        raise ValueError("not enough observations for a 2x2 design")
    used = array[: 4 * per_cell]
    factor_a = ["A1"] * (2 * per_cell) + ["A2"] * (2 * per_cell)
    factor_b = ["B1"] * per_cell + ["B2"] * per_cell + ["B1"] * per_cell + ["B2"] * per_cell
    return used, factor_a, factor_b


def _repeated(values: list[int]) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Two groups of four subjects, three measurements each."""
    array = np.asarray(values, dtype=float)
    if array.size < 24:
        raise ValueError("not enough observations")
    used = array[:24]
    subjects = [f"S{i}" for i in range(8) for _ in range(3)]
    groups = ["g1"] * 12 + ["g2"] * 12
    measurements = [1, 2, 3] * 8
    return used, subjects, groups, measurements


def _shuffle_within_cells(values: np.ndarray, factor_a: list[str], factor_b: list[str], seed: int) -> np.ndarray:
    """Shuffle the rows inside each cell, leaving the design alone.

    A global shuffle would move observations from one cell to another,
    which is a different design rather than a different order.
    """
    rng = np.random.default_rng(seed)
    out = values.copy()
    cells = {(a, b) for a, b in zip(factor_a, factor_b, strict=True)}
    for a, b in sorted(cells):
        indices = [i for i, (x, y) in enumerate(zip(factor_a, factor_b, strict=True)) if (x, y) == (a, b)]
        block = out[indices]
        rng.shuffle(block)
        out[indices] = block
    return out


# ---------------------------------------------------------------------------
# Two-way ANOVA
# ---------------------------------------------------------------------------
@given(values=integer_strategy)
@_SETTING
def test_the_two_way_anova_terms_add_up_to_the_total(values):
    """A decomposition that does not add up is not a decomposition."""
    try:
        array, factor_a, factor_b = _two_by_two(values)
    except ValueError:
        return
    result = call_or_none(DesignTestAnalyzer().two_way_anova, array, factor_a, factor_b, n_permutations=0)
    if result is None:
        return
    total = result.term_a.ss + result.term_b.ss + result.term_interaction.ss + result.ss_error
    assert np.isclose(result.ss_total, total, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy)
@_SETTING
def test_the_two_way_anova_degrees_of_freedom_add_up(values):
    """The degrees of freedom of a factorial ANOVA are ``n - 1``."""
    try:
        array, factor_a, factor_b = _two_by_two(values)
    except ValueError:
        return
    result = call_or_none(DesignTestAnalyzer().two_way_anova, array, factor_a, factor_b, n_permutations=0)
    if result is None:
        return
    assert result.term_a.df + result.term_b.df + result.term_interaction.df + result.df_error == result.df_total
    assert result.df_total == array.size - 1


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_the_two_way_anova_is_invariant_under_a_positive_affine_transform(values, scale, shift):
    """A mean shift and a rescaling of the response change no F ratio."""
    try:
        array, factor_a, factor_b = _two_by_two(values)
    except ValueError:
        return
    analyzer = DesignTestAnalyzer()
    base = call_or_none(analyzer.two_way_anova, array, factor_a, factor_b, n_permutations=0)
    moved = call_or_none(analyzer.two_way_anova, affine(array, scale, shift), factor_a, factor_b, n_permutations=0)
    if base is None or moved is None:
        return
    for term, changed in (
        (base.term_a, moved.term_a),
        (base.term_b, moved.term_b),
        (base.term_interaction, moved.term_interaction),
    ):
        assert np.isclose(term.f_statistic, changed.f_statistic, rtol=RTOL, atol=ATOL, equal_nan=True)
        assert np.isclose(term.p_value, changed.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy)
@_SETTING
def test_the_two_way_anova_does_not_depend_on_the_factor_labels(values):
    """Renaming the levels of a factor is the same design."""
    try:
        array, factor_a, factor_b = _two_by_two(values)
    except ValueError:
        return
    renamed_a = ["A2" if label == "A1" else "A1" for label in factor_a]
    analyzer = DesignTestAnalyzer()
    base = call_or_none(analyzer.two_way_anova, array, factor_a, factor_b, n_permutations=0)
    moved = call_or_none(analyzer.two_way_anova, array, renamed_a, factor_b, n_permutations=0)
    if base is None or moved is None:
        return
    assert np.isclose(base.term_a.p_value, moved.term_a.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.term_b.p_value, moved.term_b.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(
        base.term_interaction.p_value, moved.term_interaction.p_value, rtol=RTOL, atol=ATOL, equal_nan=True
    )


@given(values=integer_strategy, seed=st.integers(min_value=0, max_value=2**16))
@_SETTING
def test_the_two_way_anova_does_not_depend_on_the_row_order_inside_a_cell(values, seed):
    """The order of the rows is not part of the design."""
    try:
        array, factor_a, factor_b = _two_by_two(values)
    except ValueError:
        return
    shuffled = _shuffle_within_cells(array, factor_a, factor_b, seed)
    analyzer = DesignTestAnalyzer()
    base = call_or_none(analyzer.two_way_anova, array, factor_a, factor_b, n_permutations=0)
    moved = call_or_none(analyzer.two_way_anova, shuffled, factor_a, factor_b, n_permutations=0)
    if base is None or moved is None:
        return
    assert np.isclose(base.term_a.p_value, moved.term_a.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.term_b.p_value, moved.term_b.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(
        base.term_interaction.p_value, moved.term_interaction.p_value, rtol=RTOL, atol=ATOL, equal_nan=True
    )


# ---------------------------------------------------------------------------
# Repeated-measures ANOVA
# ---------------------------------------------------------------------------
@given(values=integer_strategy)
@_SETTING
def test_the_repeated_measures_strata_add_up_to_the_total(values):
    """The between-subject and the within-subject strata are a
    decomposition too, and they must be complete."""
    try:
        array, subjects, groups, measurements = _repeated(values)
    except ValueError:
        return
    result = call_or_none(
        DesignTestAnalyzer().repeated_measures_anova,
        array,
        subjects,
        groups,
        measurement=measurements,
    )
    if result is None:
        return
    total = (
        result.term_groups.ss
        + result.term_subjects.ss
        + result.term_measurement.ss
        + result.term_interaction.ss
        + result.term_error.ss
    )
    assert np.isclose(result.ss_total, total, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(values=integer_strategy)
@_SETTING
def test_the_repeated_measures_degrees_of_freedom_add_up(values):
    """The degrees of freedom of the two strata are ``n - 1``."""
    try:
        array, subjects, groups, measurements = _repeated(values)
    except ValueError:
        return
    result = call_or_none(
        DesignTestAnalyzer().repeated_measures_anova,
        array,
        subjects,
        groups,
        measurement=measurements,
    )
    if result is None:
        return
    assert (
        result.term_groups.df
        + result.term_subjects.df
        + result.term_measurement.df
        + result.term_interaction.df
        + result.term_error.df
        == result.df_total
    )


@given(values=integer_strategy, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_the_repeated_measures_group_effect_is_invariant_under_an_affine_transform(values, scale, shift):
    """The group effect is a between-subject F ratio, and an affine
    transform of the response changes no ratio."""
    try:
        array, subjects, groups, measurements = _repeated(values)
    except ValueError:
        return
    analyzer = DesignTestAnalyzer()
    base = call_or_none(analyzer.repeated_measures_anova, array, subjects, groups, measurement=measurements)
    moved = call_or_none(
        analyzer.repeated_measures_anova,
        affine(array, scale, shift),
        subjects,
        groups,
        measurement=measurements,
    )
    if base is None or moved is None:
        return
    assert np.isclose(base.term_groups.f_statistic, moved.term_groups.f_statistic, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.term_groups.p_value, moved.term_groups.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


# ---------------------------------------------------------------------------
# Intraclass correlation
# ---------------------------------------------------------------------------
#: A measurement table with a FIXED number of raters.
#:
#: The number of columns is fixed because a ragged list is not a
#: measurement table: the rows of one are the same raters.
_icc_table = st.lists(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
    min_size=3,
    max_size=10,
)


@given(table=_icc_table)
@_SETTING
def test_the_intraclass_correlation_is_a_correlation(table):
    """An ICC lives in ``[-1, 1]``.

    The point estimate is deliberately left unclipped when a variance
    component came out negative, so this is a property of a *sane*
    estimate rather than a tautology.

    The interval is built by inverting the F ratio, and that inversion
    only produces NON-NEGATIVE component ratios.  So when a variance
    component is estimated below zero the interval starts at zero while
    the point estimate, which is left unclipped because that bias is
    part of the estimator, is negative -- and the result says so.  The
    containment therefore holds unless a component was clipped.
    """
    array = np.asarray(table, dtype=float)
    if not has_variation(array):
        return
    analyzer = DesignTestAnalyzer()
    for form in ("2,1", "3,1"):
        result = call_or_none(analyzer.intraclass_correlation, array, form=form)
        if result is None:
            continue
        # The coefficient is undefined -- 0/0 -- when the targets carry no
        # variation at all, and undefined is not a value outside the
        # range; it is no value at all.
        if not np.isfinite(result.icc):
            continue
        assert -1.0 <= result.icc <= 1.0
        clipped = any(value == 0.0 for value in result.variance_components.values())
        if not clipped:
            assert result.ci_lower - 1e-9 <= result.icc <= result.ci_upper + 1e-9


@given(table=_icc_table, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_the_intraclass_correlation_is_invariant_under_an_affine_transform(table, scale, shift):
    """A rescaling of the whole table changes no ratio."""
    array = np.asarray(table, dtype=float)
    if not has_variation(array):
        return
    analyzer = DesignTestAnalyzer()
    for form in ("2,1", "3,1"):
        base = call_or_none(analyzer.intraclass_correlation, array, form=form)
        moved = call_or_none(analyzer.intraclass_correlation, affine(array, scale, shift), form=form)
        if base is None or moved is None:
            continue
        assert np.isclose(base.icc, moved.icc, rtol=RTOL, atol=1e-6, equal_nan=True)
        assert np.isclose(moved.ms_targets, base.ms_targets * scale**2, rtol=RTOL, atol=1e-6, equal_nan=True)
        assert np.isclose(moved.ms_error, base.ms_error * scale**2, rtol=RTOL, atol=1e-6, equal_nan=True)


@given(table=_icc_table, seed=st.integers(min_value=0, max_value=2**16))
@_SETTING
def test_the_intraclass_correlation_does_not_depend_on_the_order_of_targets_or_raters(table, seed):
    """Neither the targets nor the raters carry an order.

    Permuting the targets is permuting the rows; permuting the raters
    is permuting the columns.  Neither changes the variation between
    targets, the variation between raters or the residual, so the
    coefficient cannot move.
    """
    array = np.asarray(table, dtype=float)
    if not has_variation(array):
        return
    rng = np.random.default_rng(seed)
    row_order = rng.permutation(array.shape[0])
    column_order = rng.permutation(array.shape[1])
    shuffled = array[np.ix_(row_order, column_order)]

    analyzer = DesignTestAnalyzer()
    for form in ("2,1", "3,1"):
        base = call_or_none(analyzer.intraclass_correlation, array, form=form)
        moved = call_or_none(analyzer.intraclass_correlation, shuffled, form=form)
        if base is None or moved is None:
            continue
        assert np.isclose(base.icc, moved.icc, rtol=RTOL, atol=1e-6, equal_nan=True)
        assert np.isclose(base.ms_targets, moved.ms_targets, rtol=RTOL, atol=1e-6, equal_nan=True)
        assert np.isclose(base.ms_error, moved.ms_error, rtol=RTOL, atol=1e-6, equal_nan=True)


@given(table=_icc_table)
@_SETTING
def test_the_intraclass_variance_components_are_non_negative(table):
    """A variance component is a variance: it cannot be negative.

    A negative moment estimate is reported as zero, so the component
    the application reports is never negative -- which is the whole
    point of reporting it as zero rather than as the negative number it
    was estimated to be.
    """
    array = np.asarray(table, dtype=float)
    if not has_variation(array):
        return
    result = call_or_none(DesignTestAnalyzer().intraclass_correlation, array, form="2,1")
    if result is None:
        return
    for name, value in result.variance_components.items():
        assert value >= 0.0, f"the {name} variance component is negative"


# ---------------------------------------------------------------------------
# Contingency chi-square
# ---------------------------------------------------------------------------
#: A 3x3 contingency table.  The shape is fixed for the same reason as
#: the measurement table above: a contingency table has one shape.
_table = st.lists(
    st.lists(
        st.integers(min_value=0, max_value=50),
        min_size=3,
        max_size=3,
    ),
    min_size=3,
    max_size=3,
)


@given(table=_table)
@_SETTING
def test_the_contingency_chi_square_does_not_depend_on_the_row_order(table):
    """The order of the rows of a contingency table is not part of the
    table."""
    array = np.asarray(table, dtype=float)
    if array.sum() < 2:
        return
    analyzer = DesignTestAnalyzer()
    base = call_or_none(analyzer.contingency_chi_square, array)
    moved = call_or_none(analyzer.contingency_chi_square, array[::-1])
    if base is None or moved is None:
        return
    assert np.isclose(base.chi_square, moved.chi_square, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert base.dof == moved.dof


@given(table=_table)
@_SETTING
def test_the_contingency_chi_square_does_not_depend_on_the_column_order(table):
    """The order of the columns is not part of the table either."""
    array = np.asarray(table, dtype=float)
    if array.sum() < 2:
        return
    analyzer = DesignTestAnalyzer()
    base = call_or_none(analyzer.contingency_chi_square, array)
    moved = call_or_none(analyzer.contingency_chi_square, array[:, ::-1])
    if base is None or moved is None:
        return
    assert np.isclose(base.chi_square, moved.chi_square, rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(base.p_value, moved.p_value, rtol=RTOL, atol=ATOL, equal_nan=True)


@given(table=_table)
@_SETTING
def test_the_contingency_cramers_v_is_a_proportion(table):
    """Cramer's V is a correlation-like quantity: it lives in [0, 1].

    It is *undefined* -- NaN -- for a table with an empty row or column,
    because the expected count in that row is zero and the formula
    divides by it.  An undefined value is not a value outside the range;
    it is no value at all.
    """
    array = np.asarray(table, dtype=float)
    if array.sum() < 2:
        return
    result = call_or_none(DesignTestAnalyzer().contingency_chi_square, array)
    if result is None:
        return
    assert not np.isfinite(result.cramers_v) or 0.0 <= result.cramers_v <= 1.0
