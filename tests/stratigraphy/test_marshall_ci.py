# =============================================================================
# Test: Marshall 1990 CI (stratigraphy/extinction.py)
# =============================================================================
"""
Tests for Marshall (1990) and Strauss & Sadler (1989) confidence intervals.

Coordinate convention (stratigraphy/extinction.py):
    LAD positions are counted from the TOP of the section; larger numbers
    are deeper/older. The true extinction lies at or YOUNGER than the LAD
    (Signor-Lipps), so the confidence interval is [LAD - gap, LAD] in
    these coordinates (ci_lower = younger bound).

References:
    Marshall, C.R. (1990). Confidence intervals on stratigraphic ranges.
        Paleobiology, 16(1), 1-24.
    Strauss, D. & Sadler, P.M. (1989). Classical confidence intervals and
        Bayesian probability estimates for ends of local taxon ranges.
        Mathematical Geology, 21(4), 411-427.
"""

from __future__ import annotations

import numpy as np

from stratigraphy.extinction import ExtinctionIntervalAnalyzer


class TestMarshallCI:
    """Verify Marshall (1990) CI direction, width and level."""

    def test_ci_extends_younger_not_older(self):
        """CI must extend toward the YOUNGER side (smaller layer numbers)."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0, 12.0, 15.0])
        n_above = np.array([2, 1, 0])

        ci_low, ci_up, _ = analyzer._marshall_ci(lad, n_above, 0.5, 0.95)

        assert np.all(ci_up == lad), "Upper bound must equal the LAD"
        assert np.all(ci_low < lad), "Lower bound must be younger than the LAD"
        assert np.all(ci_low >= 0), "Bounds must stay within the section"

    def test_known_gap_value(self):
        """95% gap = -ln(0.05) / -ln(1 - 0.5) = 4.321 layers."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above = np.array([0])

        ci_low, ci_up, _ = analyzer._marshall_ci(lad, n_above, 0.5, 0.95)

        expected_gap = -np.log(0.05) / -np.log(0.5)
        assert abs((ci_up[0] - ci_low[0]) - expected_gap) < 1e-6

    def test_perfect_detection_gives_degenerate_ci(self):
        """With p = 1 (certain detection) the gap is 0: extinction = LAD."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        ci_low, ci_up, _ = analyzer._marshall_ci(lad, np.array([0]), 1.0, 0.95)
        assert ci_low[0] == lad[0] == ci_up[0]

    def test_higher_detection_narrower_ci(self):
        """Better recovery should narrow the interval."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above = np.array([3])

        ci_low_hi, ci_up_hi, _ = analyzer._marshall_ci(lad, n_above, 0.9, 0.95)
        ci_low_lo, ci_up_lo, _ = analyzer._marshall_ci(lad, n_above, 0.3, 0.95)

        gap_hi = ci_up_hi[0] - ci_low_hi[0]
        gap_lo = ci_up_lo[0] - ci_low_lo[0]
        assert gap_hi < gap_lo

    def test_99_wider_than_95(self):
        """99% CI should be wider than 95% CI."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above = np.array([3])

        ci_low_95, ci_up_95, _ = analyzer._marshall_ci(lad, n_above, 0.5, 0.95)
        ci_low_99, ci_up_99, _ = analyzer._marshall_ci(lad, n_above, 0.5, 0.99)

        assert (ci_up_99[0] - ci_low_99[0]) > (ci_up_95[0] - ci_low_95[0])

    def test_youngest_lad_gets_ci(self):
        """The topmost (youngest) LAD must receive an interval - it is the
        classic Marshall use case and previously got a degenerate CI."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above = np.array([0])

        ci_low, ci_up, true_ext = analyzer._marshall_ci(lad, n_above, 0.5, 0.95)

        assert ci_low[0] < ci_up[0]
        assert true_ext[0] == lad[0]

    def test_analyze_end_to_end_direction(self):
        """End-to-end analyze(): intervals at or younger of the LAD."""
        analyzer = ExtinctionIntervalAnalyzer()
        result = analyzer.analyze(
            lad_positions=np.array([3.0, 7.0, 12.0]),
            detection_probability=0.5,
            confidence_level=0.95,
            method="marshall",
        )
        assert np.all(result.confidence_interval_upper == result.lad_positions)
        assert np.all(result.confidence_interval_lower <= result.confidence_interval_upper)


class TestStraussSadlerCI:
    """Verify Strauss & Sadler (1989) endpoint CI (last-spacing form)."""

    def test_95_multiplier(self):
        """95% gap = 1.736 x last spacing, toward the younger side."""
        analyzer = ExtinctionIntervalAnalyzer()
        # Subject is the middle LAD (35): last spacing to the next younger
        # LAD (20) is 15.
        lad = np.array([40.0, 35.0, 20.0])
        n_above = np.array([2, 1, 0])

        ci_low, ci_up, _ = analyzer._strauss_sadler_ci(lad, n_above, 0.95)

        expected_gap = (0.05 ** (-0.5) - 1.0) / 2.0 * 15.0  # 1.736 * 15
        assert abs((ci_up[1] - ci_low[1]) - expected_gap) < 1e-9
        assert abs(ci_low[1] - (35.0 - expected_gap)) < 1e-9

    def test_50_multiplier(self):
        """50% gap = 0.207 x last spacing."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([40.0, 35.0, 20.0])
        n_above = np.array([2, 1, 0])

        ci_low, ci_up, _ = analyzer._strauss_sadler_ci(lad, n_above, 0.50)

        expected_gap = (0.50 ** (-0.5) - 1.0) / 2.0 * 15.0  # 0.207 * 15
        assert abs((ci_up[1] - ci_low[1]) - expected_gap) < 1e-9

    def test_younger_direction(self):
        """All intervals must lie at or younger than the LAD."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([14.0, 9.0, 4.0])
        n_above = np.array([2, 1, 0])

        ci_low, ci_up, _ = analyzer._strauss_sadler_ci(lad, n_above, 0.95)

        assert np.all(ci_up == lad)
        assert np.all(ci_low <= lad)
        assert np.all(ci_low >= 0)
