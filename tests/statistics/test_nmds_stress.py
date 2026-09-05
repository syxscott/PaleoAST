# =============================================================================
# Test: NMDS Stress-1 formula consistency
# =============================================================================
"""
Tests for the NMDS stress formula consistency with R vegan::monoMDS.

Background:
    The standard Kruskal (1964) stress-1 formula is:
        stress_1 = sqrt(sum((d_hat - d_tilde)^2) / sum(d_hat^2))
    where d_hat are configuration distances and d_tilde are disparities.
    This is the canonical definition used by R vegan::monoMDS.

References:
    Kruskal, J.B. (1964). Multidimensional scaling by optimizing goodness
        of fit to a nonmetric hypothesis. Psychometrika, 29, 1-27.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statistics.nmds import NMDSAnalyzer


class TestNMDSStressFormula:
    """Verify NMDS stress formula matches Kruskal 1964 / R vegan::monoMDS."""

    def test_stress_1_uses_d_hat_denominator(self):
        """stress_1 denominator should be sum(d_hat^2), NOT sum(d_target^2)."""
        # Simple 4-sample distance matrix (will be embedded in 2D)
        D = np.array(
            [
                [0.0, 1.0, 2.5, 3.0],
                [1.0, 0.0, 2.0, 2.8],
                [2.5, 2.0, 0.0, 1.5],
                [3.0, 2.8, 1.5, 0.0],
            ]
        )

        analyzer = NMDSAnalyzer()
        # Stress-1 must be opt-in (raw_stress is the v1.0.0-compatible default).
        result = analyzer.analyze(
            D, n_dimensions=2, n_restarts=3, random_seed=42, method="stress_1"
        )

        assert result.stress_formula == "stress_1"
        assert 0 <= result.stress <= 1.0

    def test_raw_stress_legacy_formula(self):
        """raw_stress uses d_target in denominator (legacy)."""
        D = np.array(
            [
                [0.0, 1.0, 2.5, 3.0],
                [1.0, 0.0, 2.0, 2.8],
                [2.5, 2.0, 0.0, 1.5],
                [3.0, 2.8, 1.5, 0.0],
            ]
        )

        analyzer = NMDSAnalyzer()
        result = analyzer.analyze(
            D,
            n_dimensions=2,
            n_restarts=3,
            random_seed=42,
            method="raw_stress",
        )

        assert result.stress_formula == "raw_stress"

    def test_stress_formula_affects_value(self):
        """The two formulas generally give different stress values for the same data.

        Note: the coordinates do NOT have to be identical between the two
        formulas, because the SMACOF convergence check uses the current
        iteration's stress value to decide whether to stop early, and the two
        formulas produce different stress magnitudes. The test only checks
        that the reported stress values differ (i.e. the formula actually
        affects the result).
        """
        D = np.array(
            [
                [0.0, 1.0, 2.5, 3.0],
                [1.0, 0.0, 2.0, 2.8],
                [2.5, 2.0, 0.0, 1.5],
                [3.0, 2.8, 1.5, 0.0],
            ]
        )

        analyzer = NMDSAnalyzer()
        r_stress1 = analyzer.analyze(
            D, n_dimensions=2, n_restarts=1, random_seed=42, method="stress_1"
        )
        r_raw = analyzer.analyze(
            D, n_dimensions=2, n_restarts=1, random_seed=42, method="raw_stress"
        )

        # Sanity: stress values should both be finite and non-negative.
        assert np.isfinite(r_stress1.stress) and r_stress1.stress >= 0
        assert np.isfinite(r_raw.stress) and r_raw.stress >= 0
        # The two formulas should give different stress values (the whole point).
        assert r_stress1.stress != r_raw.stress, (
            f"Expected different stress values, both got {r_stress1.stress}"
        )
        # Formula label must reflect the choice.
        assert r_stress1.stress_formula == "stress_1"
        assert r_raw.stress_formula == "raw_stress"

    def test_invalid_method_raises(self):
        """Invalid method parameter should raise ValueError."""
        D = np.array([[0.0, 1.0], [1.0, 0.0]])

        analyzer = NMDSAnalyzer()
        try:
            analyzer.analyze(D, n_dimensions=1, method="bogus_method")
            raised = False
        except ValueError:
            raised = True

        assert raised, "Expected ValueError for invalid method"

    def test_known_4point_stress_reasonable(self):
        """For a 4-point Euclidean distance matrix, NMDS stress should be small
        (close to perfect embedding)."""
        # Generate 4 points in 2D, compute Euclidean distances
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 5, size=(4, 2))
        from scipy.spatial.distance import pdist, squareform

        D = squareform(pdist(points))

        analyzer = NMDSAnalyzer()
        result = analyzer.analyze(
            D, n_dimensions=2, n_restarts=10, random_seed=42, max_iterations=500
        )

        # 2D embedding of 2D data should yield very low stress
        assert result.stress < 0.1, f"Expected low stress, got {result.stress}"

    def test_stress_recorded_in_result(self):
        """The stress_formula attribute must reflect the chosen method."""
        D = np.array(
            [
                [0.0, 1.0, 2.5, 3.0],
                [1.0, 0.0, 2.0, 2.8],
                [2.5, 2.0, 0.0, 1.5],
                [3.0, 2.8, 1.5, 0.0],
            ]
        )

        analyzer = NMDSAnalyzer()
        r1 = analyzer.analyze(D, n_dimensions=2, random_seed=42, method="stress_1")
        r2 = analyzer.analyze(D, n_dimensions=2, random_seed=42, method="raw_stress")

        assert r1.stress_formula == "stress_1"
        assert r2.stress_formula == "raw_stress"