# =============================================================================
# Test: Marshall 1990 CI (stratigraphy/extinction.py)
# =============================================================================
"""
Tests for Marshall (1990) confidence interval implementation.

Reference:
    Marshall, C.R. (1990). Confidence intervals on stratigraphic ranges:
    constraining the position of origin and extinction of taxa by
    avoiding biases of fossil recovery. Paleobiology, 16(1), 1-24.

Formula:
    t_upper = t_LAD + chi2_{2*alpha, df=2} / (2 * r)
where r = Poisson sampling rate (per layer),
      chi2_{0.10, 2} = 4.605 (for 95% CI, alpha=0.05)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from stratigraphy.extinction import ExtinctionIntervalAnalyzer


class TestMarshallCI:
    """Verify Marshall 1990 CI uses chi-square formula correctly."""

    def test_chi2_quantile_95(self):
        """Verify chi-square 95% CI quantile value (4.605 for 2 df).
        Marshall 1990 Eq. (3) uses chi-square upper-tail quantile at 2*alpha
        level: ppf(1 - 2*alpha, df=2). For alpha=0.05, this is ppf(0.90, df=2)
        = 4.605. (NOT chi2_{0.95, 2} = 5.991 which is two-sided 5% critical.)"""
        q = 0.05  # 95% CI -> alpha = 0.05, 2*alpha = 0.10 upper tail
        chi2_val = stats.chi2.ppf(1.0 - 2.0 * q, df=2)
        assert abs(chi2_val - 4.605) < 0.01, f"Expected ~4.605, got {chi2_val}"

    def test_marshall_upper_offset_positive(self):
        """Upper bound offset must be positive (CI extends older than LAD)."""
        # Toy data: 5 taxa, last appearing at LAD layer 10
        analyzer = ExtinctionIntervalAnalyzer()
        lad_positions = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        n_layers_above = np.array([5, 5, 5, 5, 5])
        detection_prob = 0.5
        confidence_level = 0.95

        ci_lower, ci_upper, true_ext = analyzer._marshall_ci(
            lad_positions, n_layers_above, detection_prob, confidence_level
        )

        # Upper bound must be strictly greater than LAD
        assert np.all(ci_upper > lad_positions), (
            f"CI upper must extend older than LAD. "
            f"lad={lad_positions}, ci_upper={ci_upper}"
        )

    def test_marshall_n_eff_dependence(self):
        """Higher effective sample size should give narrower CI."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above_high = np.array([100])  # Many layers above
        n_above_low = np.array([5])     # Few layers above

        ci_low_h, ci_up_h, _ = analyzer._marshall_ci(
            lad, n_above_high, detection_prob=0.5, confidence_level=0.95
        )
        ci_low_l, ci_up_l, _ = analyzer._marshall_ci(
            lad, n_above_low, detection_prob=0.5, confidence_level=0.95
        )

        # More samples -> narrower CI (smaller offset)
        offset_high = ci_up_h[0] - lad[0]
        offset_low = ci_up_l[0] - lad[0]
        assert offset_high < offset_low, (
            f"High n_eff should give narrower CI. "
            f"offset_high={offset_high}, offset_low={offset_low}"
        )

    def test_marshall_zero_layers_returns_degenerate_ci(self):
        """When k=0 (taxon at top layer), CI should be degenerate at LAD."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_layers_above = np.array([0])

        ci_low, ci_up, true_ext = analyzer._marshall_ci(
            lad, n_layers_above, detection_prob=0.5, confidence_level=0.95
        )

        # Degenerate: ci_upper == ci_lower == lad
        assert ci_up[0] == lad[0]
        assert ci_low[0] == lad[0]
        assert true_ext[0] == lad[0]

    def test_marshall_99_vs_95_ci(self):
        """99% CI should be wider than 95% CI (more conservative)."""
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above = np.array([20])

        ci_low_95, ci_up_95, _ = analyzer._marshall_ci(
            lad, n_above, detection_prob=0.5, confidence_level=0.95
        )
        ci_low_99, ci_up_99, _ = analyzer._marshall_ci(
            lad, n_above, detection_prob=0.5, confidence_level=0.99
        )

        offset_95 = ci_up_95[0] - lad[0]
        offset_99 = ci_up_99[0] - lad[0]
        assert offset_99 > offset_95, (
            f"99% CI should be wider than 95% CI. "
            f"offset_95={offset_95}, offset_99={offset_99}"
        )

    def test_marshall_known_numerical(self):
        """Verify exact chi-square computation."""
        # Manual computation:
        # r = n_eff = 20 (layers above) / 0.5 (detection prob) = 40
        # chi2_{0.10, 2} = 4.605
        # offset = 4.605 / (2 * 40) = 0.0576
        # ci_upper = 10 + 0.0576 = 10.0576
        analyzer = ExtinctionIntervalAnalyzer()
        lad = np.array([10.0])
        n_above = np.array([20])
        det_prob = 0.5

        ci_low, ci_up, _ = analyzer._marshall_ci(
            lad, n_above, detection_prob=det_prob, confidence_level=0.95
        )

        expected_offset = stats.chi2.ppf(0.90, df=2) / (2.0 * 40)
        expected_upper = 10.0 + expected_offset
        assert abs(ci_up[0] - expected_upper) < 1e-6, (
            f"Expected upper={expected_upper}, got {ci_up[0]}"
        )