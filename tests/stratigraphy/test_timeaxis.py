# =============================================================================
# FILE: tests/stratigraphy/test_timeaxis.py
# =============================================================================
"""
Tests for the canonical time axis in ``stratigraphy.timeaxis``.

The point of the module is that one place owns the Ma / kyr / years
conversion and the period / frequency conversion, so every numeric
expectation here is a definition rather than a measurement:

* 1 Ma = 1000 kyr = 1e6 years, exactly;
* 405 kyr = 1000 / 405 = 2.4691 cycles per Ma, exactly, and the round trip
  is the identity;
* the frequency grid runs from the record's fundamental ``1 / span`` to its
  Nyquist ``1 / (2 dt)``, which for a 4-point axis of 0.01 Ma steps is
  0.25 to 50 cycles/Ma and is checkable by hand;
* the log-spaced period grid is the inverse of the frequency grid, so its
  endpoints are the reciprocals of the frequency grid's endpoints.

The construction tests use the three existing representations the module
adapts -- ``IsotopeData``, ``StratigraphicSection`` and a ``time_bins`` row
table -- and check that the axis is sorted ascending regardless of the order
the input arrives in, because "oldest last" is the module's canonical order
and getting it wrong is the failure that silently flips every spectral
frequency.

This file deliberately tests no analysis: it is a value type, and the
analyses that consume it are covered in ``test_cycles.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratigraphy.correlation import StratigraphicSection
from stratigraphy.isotope_analysis import IsotopeData
from stratigraphy.timeaxis import UNIT_TO_MA, TimeAxis
from utils.exceptions import DataValidationError

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_axis_is_sorted_ascending_whatever_order_the_input_arrives_in() -> None:
    """Canonical order is ages ascending, i.e. oldest last.

    ``IsotopeData.age`` arrives deepest-first, because depth is the primary
    axis of a core; a time axis that preserved that order would make every
    ``np.diff`` negative and every spectral frequency negative.
    """
    data = IsotopeData(depth=np.arange(4.0), age=np.array([3.0, 2.0, 1.0, 0.0]))
    axis = TimeAxis.from_isotope_data(data)
    assert axis.ages_ma.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert np.all(np.diff(axis.ages_ma) > 0)
    assert axis.n == 4
    assert len(axis) == 4
    assert "IsotopeData" in axis.source


def test_axis_from_a_section_uses_its_ages_and_reports_a_missing_age_model() -> None:
    """A section without an age model has no time axis, and must say so.

    Inventing one (falling back to the height axis, say) would produce a
    plausible-looking axis in metres and every frequency on it would be
    wrong by a factor of a million.
    """
    with_ages = StratigraphicSection(
        name="Section A", heights=np.array([10.0, 20.0, 30.0]), thicknesses=np.ones(3),
        lithologies=["s", "sh", "s"], ages=np.array([30.0, 20.0, 10.0]),
    )
    axis = TimeAxis.from_section(with_ages)
    assert axis.ages_ma.tolist() == [10.0, 20.0, 30.0]
    assert "Section A" in axis.source

    without = StratigraphicSection(
        name="Section B", heights=np.array([10.0, 20.0]), thicknesses=np.ones(2),
        lithologies=["s", "sh"],
    )
    with pytest.raises(DataValidationError, match="Section B"):
        TimeAxis.from_section(without)


def test_axis_from_bin_rows_collapses_intervals_the_way_it_documents() -> None:
    """A bin table is an interval table, so ``auto`` has to pick a rule.

    ``time_bins`` rows carry ``max_ma``/``min_ma`` and normally ``mid_ma``
    too. ``auto`` prefers the stored midpoint, falls back to the interval
    midpoint, and only then to a bound -- three rows here, one for each
    branch, so the precedence is tested rather than described.
    """
    rows = [
        {"bin": 1, "max_ma": 3.0, "min_ma": 2.0, "mid_ma": 2.9},
        {"bin": 2, "max_ma": 2.0, "min_ma": 1.0},
        {"bin": 3, "min_ma": 0.0, "max_ma": 1.0},
    ]
    # The default uses the stored midpoint, and refuses a row without one
    # rather than quietly falling back.
    assert TimeAxis.from_bin_rows(rows[:1]).ages_ma.tolist() == [2.9]
    with pytest.raises(DataValidationError, match="missing 'mid_ma'"):
        TimeAxis.from_bin_rows(rows)
    # key="auto" is the three-branch rule, one branch per row above.
    assert TimeAxis.from_bin_rows(rows, key="auto").ages_ma.tolist() == [0.5, 1.5, 2.9]
    # The explicit key overrides auto, and a key no row has is an error.
    assert TimeAxis.from_bin_rows(rows, key="max_ma").ages_ma.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(DataValidationError, match="missing 'duration_myr'"):
        TimeAxis.from_bin_rows(rows, key="duration_myr")
    with pytest.raises(DataValidationError, match="empty bin table"):
        TimeAxis.from_bin_rows([])
    with pytest.raises(DataValidationError, match="needs mid_ma"):
        TimeAxis.from_bin_rows([{"bin": 1}], key="auto")


def test_axis_reports_span_step_and_uniformity() -> None:
    """Uniformity is a diagnostic to report, not an error to raise.

    Uneven sampling is the normal case in cyclostratigraphy, so
    ``is_uniform`` answers a question rather than gating entry. The 1 %
    tolerance is stated in the property: a 0.01 Ma step perturbed by 1 % is
    uniform, one perturbed by 50 % is not.
    """
    regular = TimeAxis.from_ages(np.arange(0.0, 0.5, 0.01))
    assert regular.is_uniform
    assert regular.dt_ma == pytest.approx(0.01, rel=0.01)
    assert regular.span_ma == pytest.approx(0.49, abs=1e-12)

    jittered = TimeAxis.from_ages(np.array([0.0, 0.01, 0.025, 0.03, 0.031]))
    assert not jittered.is_uniform
    assert "Uniform" in jittered.summary()
    assert "not a time axis" not in jittered.summary()


def test_axis_rejects_degenerate_input() -> None:
    """Too short, non-finite and unknown units are three different errors."""
    with pytest.raises(DataValidationError, match="empty"):
        TimeAxis.from_ages([])
    with pytest.raises(DataValidationError, match="non-finite"):
        TimeAxis.from_ages([0.0, np.nan, 2.0, 3.0])
    with pytest.raises(DataValidationError, match="units must be one of"):
        TimeAxis.from_ages([0.0, 1.0, 2.0], units="furlongs")


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit,factor_to_ma", [("Ma", 1.0), ("kyr", 0.001), ("yr", 1e-6), ("KYA", 0.001)]
)
def test_input_units_are_all_accepted_and_normalised_to_ma(unit: str, factor_to_ma: float) -> None:
    """The declared multipliers are the contract; assert them all at once.

    ``UNIT_TO_MA`` is the table every conversion goes through, so checking
    the value and checking the round trip are the same test: 1 kyr entering
    as 1.0 must leave as 0.001 Ma.
    """
    assert UNIT_TO_MA[unit.lower()] == pytest.approx(factor_to_ma, rel=1e-15)
    axis = TimeAxis.from_ages([0.0, 1.0, 2.0], units=unit)
    np.testing.assert_allclose(axis.ages_ma, [0.0, factor_to_ma, 2 * factor_to_ma], atol=1e-15)


def test_output_unit_conversions_are_exact_multiples() -> None:
    """1 Ma = 1000 kyr = 1e6 yr, with no rounding anywhere.

    The calendar-year conversion defaults to a present year of 2020 because
    ``stratigraphy/time_bins.py`` documents its ages as "Myr before 2020 AD";
    defaulting to the conventional 1950 BP datum instead would put a 70-year
    offset into every date a user prints, which is invisible at Ma and
    visible at ka.
    """
    axis = TimeAxis.from_ages([0.0, 1.0, 2.5])
    np.testing.assert_allclose(axis.to_kyr(), [0.0, 1000.0, 2500.0], atol=1e-9)
    np.testing.assert_allclose(axis.to_bp_years(), [0.0, 1e6, 2.5e6], rtol=1e-12)
    np.testing.assert_allclose(axis.calendar_years(), [2020.0, -997980.0, -2497980.0], atol=1.0)
    np.testing.assert_allclose(
        axis.calendar_years(present_year=1950), [1950.0, -998050.0, -2498050.0], atol=1.0
    )
    # Round trip through the advertised units.
    np.testing.assert_allclose(TimeAxis.from_ages(axis.to_kyr(), units="kyr").ages_ma, axis.ages_ma)


def test_period_and_frequency_conversion_is_the_hundredfold_rule() -> None:
    """f = 1000 / P_kyr in cycles per Ma, and the round trip is the identity.

    405 kyr is the canonical long-period cycle, so it is the number a reader
    will check first: 1000 / 405 = 2.46914 cycles per Ma. The dropped factor
    of 1000 is the classic unit slip in this domain, and both directions are
    asserted so neither can be wrong on its own.
    """
    axis = TimeAxis.from_ages(np.arange(0.0, 1.0, 0.01))
    assert axis.frequency_per_ma(405.0) == pytest.approx(2.469135802, rel=1e-9)
    assert axis.period_kyr(2.4691358024691358) == pytest.approx(405.0, rel=1e-9)
    periods = np.array([19.0, 23.0, 41.0, 100.0, 400.0])
    np.testing.assert_allclose(
        axis.period_kyr(axis.frequency_per_ma(periods)), periods, rtol=1e-12
    )
    # Vector and scalar paths must agree.
    assert axis.frequency_per_ma(100.0) == pytest.approx(float(axis.frequency_per_ma([100.0])[0]))


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def test_frequency_grid_spans_the_fundamental_to_the_nyquist() -> None:
    """The default grid is the pair the periodogram routine picks by hand.

    For 0 <= t <= 0.99 in steps of 0.01, span = 0.99 Ma and dt = 0.01, so
    the fundamental is 1 / 0.99 = 1.0101 cycles/Ma and the Nyquist limit is
    1 / (2 * 0.01) = 50. The period grid is the reciprocal, so it runs from
    1 / 50 = 0.02 kyr to 1 / 1.0101 = 990 kyr, and it is log-spaced because
    a 405 kyr cycle and a 19 kyr cycle differ by more than an order of
    magnitude.
    """
    axis = TimeAxis.from_ages(np.arange(0.0, 1.0, 0.01))
    freq = axis.frequency_axis(n=5)
    assert freq[0] == pytest.approx(1.0 / axis.span_ma)
    assert freq[-1] == pytest.approx(1.0 / (2.0 * axis.dt_ma))
    assert freq[-1] == pytest.approx(50.0)
    assert np.all(np.diff(freq) > 0)

    period = axis.period_axis(n=5)
    # The period grid is the reciprocal OF THE UNIT CONVERTED frequency grid,
    # in kyr, so it runs from 1000 / 50 = 20 kyr up to 1000 / 1.0101 = 990
    # kyr. It is log-spaced because a 405 kyr cycle and a 19 kyr cycle differ
    # by more than an order of magnitude, and a linear frequency grid would
    # resolve the long end at the short end's expense.
    assert period[0] == pytest.approx(axis.period_kyr(freq[-1]))
    assert period[0] == pytest.approx(20.0, rel=1e-6)
    assert period[-1] == pytest.approx(axis.period_kyr(freq[0]))
    assert period[-1] == pytest.approx(990.0, rel=1e-6)
    ratios = period[1:] / period[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
    assert ratios[0] > 1.0


def test_grids_reject_an_axis_that_cannot_support_them() -> None:
    """Too few points, a negative request and a flat axis must all be refused."""
    short = TimeAxis.from_ages([0.0, 1.0])
    with pytest.raises(DataValidationError, match="at least 3 ages"):
        short.frequency_axis()
    with pytest.raises(DataValidationError, match="at least 3 ages"):
        short.period_axis()
    with pytest.raises(DataValidationError, match="n must be"):
        TimeAxis.from_ages(np.arange(0.0, 1.0, 0.01)).frequency_axis(n=1)
    with pytest.raises(DataValidationError, match="n must be"):
        TimeAxis.from_ages(np.arange(0.0, 1.0, 0.01)).period_axis(n=1)
    # Two identical ages have a zero span, so no fundamental exists.
    flat = TimeAxis.from_ages([0.0, 0.0])
    with pytest.raises(DataValidationError, match="at least 3 ages"):
        flat.frequency_axis()
    # A single interval longer than twice its own step has a fundamental
    # above its own Nyquist limit, which is not a grid, it is a contradiction.
    coarse = TimeAxis.from_ages([0.0, 10.0, 20.0])
    with pytest.raises(DataValidationError, match="not above the fundamental"):
        coarse.frequency_axis()


def test_repr_and_summary_carry_the_source() -> None:
    """A result has to be traceable to the object it came from."""
    axis = TimeAxis.from_ages(np.arange(0.0, 0.1, 0.01), source="core-7 d18O")
    assert "core-7 d18O" in repr(axis)
    assert "core-7 d18O" in axis.summary()
    assert "Time Axis" in axis.summary()
    assert repr(axis).startswith("TimeAxis(n=10")
