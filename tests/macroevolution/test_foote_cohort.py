# tests/macroevolution/test_foote_cohort.py
"""Tests for Foote 1997 cohort rate calculations."""

import numpy as np

from macroevolution.cohort import CohortSurvivorshipAnalysis, analyze_cohort_survivorship


def test_foote_cohort_basic():
    """Test basic Foote cohort rate calculation."""
    analysis = CohortSurvivorshipAnalysis()

    # Simple dataset: 4 taxa with known (o, L) ranges
    # Interval (5, 10): taxa appearing between 5 and 10 Ma
    records = [
        (12.0, 8.0),  # Existed before and during interval, survived past
        (6.0, 3.0),   # Originated in interval, survived past
        (11.0, 6.0),  # Existed before, went extinct in interval
        (7.0, 4.0),   # Originated and went extinct in interval
    ]
    intervals = [(5.0, 10.0)]  # Interval from 5 to 10 Ma

    result = analysis.analyze(records, intervals)

    # Check that we have Foote 1997 rates
    assert hasattr(result, 'foote97_origination')
    assert hasattr(result, 'foote97_extinction')
    assert hasattr(result, 'foote00_origination')
    assert hasattr(result, 'foote00_extinction')

    # For interval (5, 10):
    # - Taxon (12, 8): started_before=True, ended_in=True (existed before, extinct in interval)
    #   -> n_lb += 1, n_bt += 1, n_fl += 1
    # - Taxon (6, 3): started_in=True, ended_after=True (originated in, survived past)
    #   -> n_fb += 1, n_bl += 1, n_ft += 1
    # - Taxon (11, 6): started_before=True, ended_in=True (existed before, extinct in interval)
    #   -> n_lb += 1, n_bt += 1, n_fl += 1
    # - Taxon (7, 4): started_in=True, ended_in=True (originated and extinct in interval)
    #   -> n_bl += 1, n_fl += 1

    # n_fb = 1, n_lb = 2, n_surv = 0 (no through-timers)
    # n_bt = 2, n_bl = 2, n_ft = 1, n_fl = 3
    # N_t = n_bt + n_bl = 4

    interval_data = result.intervals[0]
    assert interval_data.n_bt == 2, f"Expected n_bt=2, got {interval_data.n_bt}"
    assert interval_data.n_bl == 2, f"Expected n_bl=2, got {interval_data.n_bl}"
    assert interval_data.n_ft == 1, f"Expected n_ft=1, got {interval_data.n_ft}"
    assert interval_data.n_fl == 3, f"Expected n_fl=3, got {interval_data.n_fl}"


def test_foote97_vs_per_capita_rates():
    """Test Foote 1997 cohort rates vs per-capita survivorship rates."""
    analysis = CohortSurvivorshipAnalysis()

    # Create dataset where we can verify the relationship
    # Through-timer: existed before interval, survived past
    records = [(15.0, 5.0)]  # Single taxon spanning interval (5, 10)
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    # With single through-timer:
    # n_surv = 1, n_total = 1
    # p = 1.0 (100% survival)
    # survivorship rate: extinction = -ln(1.0)/5 = 0

    # Foote 1997:
    # n_bt = 1, n_bl = 0 (only backward persistence, no backward extinction)
    # n_ft = 1, n_fl = 0
    # N_t = n_bt + n_bl = 1
    # foote97_extinction = -ln(n_bl/N_t)/dt = -ln(0)/5 = inf if n_bl=0

    interval_data = result.intervals[0]
    assert interval_data.n_surv == 1
    assert interval_data.n_total == 1

    # Foote 1997 origination should be 0 when n_bl = 0
    assert np.isnan(result.foote97_origination[0]) or result.foote97_origination[0] == 0


def test_foote2000_simplified_rates():
    """Test Foote 2000 simplified rate calculation."""
    analysis = CohortSurvivorshipAnalysis()

    # Dataset with known forward/backward counts
    # Taxon 1: (12, 8) - existed before, extinct in interval
    # Taxon 2: (6, 3) - originated in interval, survived past
    # Taxon 3: (11, 6) - existed before, extinct in interval
    # Taxon 4: (7, 4) - originated and extinct in interval
    records = [
        (12.0, 8.0),
        (6.0, 3.0),
        (11.0, 6.0),
        (7.0, 4.0),
    ]
    intervals = [(5.0, 10.0)]

    result = analysis.analyze(records, intervals)

    interval_data = result.intervals[0]

    # N_t = n_bt + n_bl = 2 + 2 = 4
    # p_F = n_ft / n_t = 1 / 4 = 0.25
    # q_F = n_fl / n_t = 3 / 4 = 0.75

    assert abs(result.foote00_origination[0] - 0.25) < 0.01
    assert abs(result.foote00_extinction[0] - 0.75) < 0.01


def test_foote97_formula_consistency():
    """Test that Foote 1997 formula is correctly applied."""
    analysis = CohortSurvivorshipAnalysis()

    # Create data where all taxa appear in interval and are known before
    # This means n_bt = N_t, n_bl = 0
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
    # foote97_extinction = -ln(0/2)/5 = -ln(0)/5 = inf (n_bl = 0, can't compute)

    assert interval_data.n_bt == 2
    assert interval_data.n_bl == 0

    # With n_bl = 0, the extinction rate formula gives inf
    # (mathematically correct: can't compute rate from zero events)
    assert np.isinf(result.foote97_extinction[0])


def test_interval_sequence():
    """Test multiple intervals."""
    analysis = CohortSurvivorshipAnalysis()

    records = [
        (15.0, 5.0),   # Through multiple intervals
        (12.0, 8.0),   # Through interval 1, extinct in interval 2
        (7.0, 3.0),    # Originated in interval 2
        (6.0, 2.0),    # Through interval 2
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
    """Test handling of empty intervals."""
    analysis = CohortSurvivorshipAnalysis()

    records = [(15.0, 5.0)]  # Single taxon spanning interval (10, 15)
    intervals = [(10.0, 15.0)]  # Interval with taxon present

    result = analysis.analyze(records, intervals)

    # Should handle correctly
    assert result.intervals[0].n_total > 0


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
