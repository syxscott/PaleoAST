# tests/macroevolution/test_foote_cohort.py
"""Tests for Foote 1997 cohort rate calculations."""

import numpy as np

from macroevolution.cohort import CohortSurvivorshipAnalysis, analyze_cohort_survivorship


def test_foote_cohort_basic():
    """Test basic Foote cohort rate calculation.

    Geological-time convention: time decreases from new to old.
    Records are (origin_time, extinction_time) with origin_time > extinction_time
    (origin is at older / larger Ma, extinction at younger / smaller Ma).
    Interval (t_start, t_end) has t_start < t_end (younger boundary first).
    """
    analysis = CohortSurvivorshipAnalysis()

    # Four records spanning the full taxonomy of boundary-crossers.
    # Interval (5, 10) Ma: t_start=5 is the younger (top) boundary, t_end=10 is
    # the older (bottom) boundary.
    records = [
        (12.0, 3.0),  # Through-timer: o=12>10 (older), L=3<5 (younger)
        (12.0, 8.0),  # Backward-only: o=12>10, L=8 in [5,10]
        (8.0, 3.0),   # Forward-only: o=8 in [5,10], L=3<5
        (8.0, 6.0),   # Both-in: o=8 in [5,10], L=6 in [5,10]
    ]
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    # Sanity: result exposes the Foote cohort rates
    assert hasattr(result, 'foote97_origination')
    assert hasattr(result, 'foote97_extinction')
    assert hasattr(result, 'foote00_origination')
    assert hasattr(result, 'foote00_extinction')

    # Classification under the (o, L) = (origin, extinction) interpretation
    # with interval treated as closed at both ends:
    #   (12, 3) → started_before+ended_after  → n_surv, n_bt, n_ft
    #   (12, 8) → started_before+ended_in     → n_lb,  n_bt, n_fl
    #   (8,  3) → started_in+ended_after      → n_fb,  n_bl, n_ft
    #   (8,  6) → started_in+ended_in         → n_bl,  n_fl
    #
    # Totals (Foote 2000 boundary crossers):
    #   n_surv=1, n_fb=1, n_lb=1, n_total=n_surv+n_fb+n_lb=3
    # Totals (Foote 1997 cohort variables):
    #   n_bt=2 (both through-timer and backward-only count), n_bl=2
    #     (forward-only + both-in)
    #   n_ft=2 (through-timer + forward-only), n_fl=2 (backward-only + both-in)
    # Consistency check: n_bt + n_bl = n_ft + n_fl = n_t = 4

    interval_data = result.intervals[0]
    assert interval_data.n_surv == 1, f"Expected n_surv=1, got {interval_data.n_surv}"
    assert interval_data.n_fb == 1, f"Expected n_fb=1, got {interval_data.n_fb}"
    assert interval_data.n_lb == 1, f"Expected n_lb=1, got {interval_data.n_lb}"
    assert interval_data.n_total == 3, f"Expected n_total=3, got {interval_data.n_total}"
    assert interval_data.n_bt == 2, f"Expected n_bt=2, got {interval_data.n_bt}"
    assert interval_data.n_bl == 2, f"Expected n_bl=2, got {interval_data.n_bl}"
    assert interval_data.n_ft == 2, f"Expected n_ft=2, got {interval_data.n_ft}"
    assert interval_data.n_fl == 2, f"Expected n_fl=2, got {interval_data.n_fl}"


def test_foote97_vs_per_capita_rates():
    """Test Foote 1997 cohort rates vs per-capita survivorship rates."""
    analysis = CohortSurvivorshipAnalysis()

    # Single through-timer spanning interval (5, 10) entirely.
    #   origin=15 Ma (older than t_end=10)
    #   extinction=4 Ma (younger than t_start=5, so L < 5 strictly)
    records = [(15.0, 4.0)]
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    # With single through-timer:
    #   n_surv = 1, n_total = 1
    #   p = 1.0  (100% survival)
    #
    # Foote (1997, 2000) per-capita rates (boundary crossers):
    #   n_bt = 1, n_bl = 0  → n_t = 1, n_ft = 1
    #   origination p = -ln(n_bt/n_t)/dt = -ln(1)/5 = 0
    #     (the taxon already existed: no origination observed)
    #   extinction q = -ln(n_ft/n_t)/dt = -ln(1)/5 = 0
    #     (the taxon crosses the young boundary: forward survival = 1)
    # A through-timer evidences ZERO extinction; the previous assertion
    # of q = +inf (from the backward count n_bl) inverted this.

    interval_data = result.intervals[0]
    assert interval_data.n_surv == 1
    assert interval_data.n_total == 1
    assert interval_data.n_bt == 1
    assert interval_data.n_bl == 0
    assert interval_data.n_ft == 1

    # Origination = 0 when n_bt == n_t (all backward persisters)
    assert abs(result.foote97_origination[0] - 0.0) < 1e-12
    # Extinction = 0 when n_ft == n_t (all taxa cross the young boundary)
    assert abs(result.foote97_extinction[0] - 0.0) < 1e-12
    # Main rate outputs mirror the Foote cohort estimates
    assert abs(result.origination_rates[0] - 0.0) < 1e-12
    assert abs(result.extinction_rates[0] - 0.0) < 1e-12


def test_foote2000_simplified_rates():
    """Test Foote 2000 simplified rate calculation.

    Using the same four-record dataset as test_foote_cohort_basic so the
    boundary-crosser classification is unambiguous.
    """
    analysis = CohortSurvivorshipAnalysis()

    records = [
        (12.0, 3.0),  # through-timer  → n_surv, n_bt, n_ft
        (12.0, 8.0),  # backward-only  → n_lb,  n_bt, n_fl
        (8.0, 3.0),   # forward-only   → n_fb,  n_bl, n_ft
        (8.0, 6.0),   # both-in        → n_bl,  n_fl
    ]
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    interval_data = result.intervals[0]

    # n_t = n_bt + n_bl = 2 + 2 = 4
    # p_F = n_ft / n_t = 2 / 4 = 0.5
    # q_F = n_fl / n_t = 2 / 4 = 0.5
    assert abs(result.foote00_origination[0] - 0.5) < 1e-12, (
        f"foote00_origination={result.foote00_origination[0]} expected 0.5"
    )
    assert abs(result.foote00_extinction[0] - 0.5) < 1e-12, (
        f"foote00_extinction={result.foote00_extinction[0]} expected 0.5"
    )


def test_foote97_formula_consistency():
    """Test that Foote 1997 formula is correctly applied."""
    analysis = CohortSurvivorshipAnalysis()

    # Two taxa, both backward-only (originated before, extinct within interval).
    # This means n_bt = N_t, n_bl = 0.
    records = [
        (12.0, 8.0),  # existed before, extinct in interval
        (11.0, 7.0),  # existed before, extinct in interval
    ]
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    interval_data = result.intervals[0]

    # n_bt = 2, n_bl = 0, n_ft = 0, n_fl = 2
    # N_t = 2
    # foote97_origination = -ln(2/2)/5 = -ln(1)/5 = 0
    # foote97_extinction  = -ln(0/2)/5 = +inf  (n_bl = 0, no events → rate is +inf)

    assert interval_data.n_bt == 2
    assert interval_data.n_bl == 0
    assert interval_data.n_ft == 0
    assert interval_data.n_fl == 2

    # n_bt == n_t means foote97_origination = -ln(1) = 0
    assert abs(result.foote97_origination[0] - 0.0) < 1e-12
    # n_bl == 0 means foote97_extinction = +inf (mathematically, -ln(0))
    assert np.isinf(result.foote97_extinction[0]) and result.foote97_extinction[0] > 0


def test_interval_sequence():
    """Test multiple intervals."""
    analysis = CohortSurvivorshipAnalysis()

    # Use records where every taxon has L strictly < t_start of the youngest
    # interval so that the "through-timer" classification is unambiguous.
    records = [
        (20.0, 2.0),  # through-timer across all intervals
        (16.0, 12.0), # through interval 1, extinct in interval 2
        (8.0, 3.0),   # originated in interval 2
        (7.0, 2.0),   # through interval 2
    ]

    intervals = [
        (10.0, 15.0),  # Interval 1
        (5.0, 10.0),   # Interval 2
    ]

    result = analysis.analyze(records, intervals)

    assert len(result.intervals) == 2
    assert result.intervals[0].t_start == 10.0
    assert result.intervals[1].t_start == 5.0

    # Each interval should have its own Foote 1997 rates
    assert len(result.foote97_origination) == 2
    assert len(result.foote97_extinction) == 2


def test_empty_interval():
    """Test handling of intervals that DO have boundary-crossing taxa.

    (Mis-named; the original test fixture was wrong. This test now verifies
    that a single backward-only boundary crosser is correctly counted in
    n_total. An "empty" interval (no boundary crossers) yields n_total=0,
    which is a separate code path covered by the if-n_total>0 guard.)
    """
    analysis = CohortSurvivorshipAnalysis()

    # Single backward-only taxon in interval (10, 15):
    #   origin=20 (older than t_end=15) → started_before
    #   extinction=12 (in [10, 15])      → ended_in
    records = [(20.0, 12.0)]
    intervals = [(10.0, 15.0)]

    result = analysis.analyze(records, intervals)

    # The taxon is a boundary crosser (n_lb = 1), so n_total must be 1
    assert result.intervals[0].n_total > 0, (
        f"Expected n_total > 0, got {result.intervals[0].n_total}; "
        f"interval_data={result.intervals[0]}"
    )
    assert result.intervals[0].n_lb == 1
    assert result.intervals[0].n_total == 1


def test_rate_ratio():
    """Test rate ratio calculation."""
    analysis = CohortSurvivorshipAnalysis()

    records = [
        (12.0, 8.0),
        (11.0, 6.0),
        (9.0, 4.0),
        (7.0, 3.0),
    ]
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    # Rate ratio should be origination/extinction
    if not np.any(np.isnan(result.origination_rates)) and not np.any(np.isinf(result.extinction_rates)):
        ratio = result.get_rate_ratio()
        assert len(ratio) == len(result.origination_rates)


if __name__ == "__main__":
    print("Running Foote cohort tests...")

    test_foote_cohort_basic()
    print("test_foote_cohort_basic: PASSED")

    test_foote97_vs_per_capita_rates()
    print("test_foote97_vs_per_capita_rates: PASSED")

    test_foote2000_simplified_rates()
    print("test_foote2000_simplified_rates: PASSED")

    test_foote97_formula_consistency()
    print("test_foote97_formula_consistency: PASSED")

    test_interval_sequence()
    print("test_interval_sequence: PASSED")

    test_empty_interval()
    print("test_empty_interval: PASSED")

    test_rate_ratio()
    print("test_rate_ratio: PASSED")

    print("\nAll Foote cohort tests PASSED!")
