# =============================================================================
# FILE: tests/stats/test_simper_euclidean.py
# =============================================================================
"""
Tests for SIMPER ``metric='euclidean'`` semantics.

Background: SIMPER's per-variable contributions are additively
decomposable for two metrics -- Bray-Curtis (sum of |diff|/sum of sums)
and the *squared* Euclidean distance (sum of squared diffs).  PaleoAST's
euclidean path computes the per-pair *squared* Euclidean and averages,
but historically stuffed that squared value into the
``overall_dissimilarity`` field.  Users reading that field expected the
plain Euclidean distance and silently got numbers a factor of ~10
larger (or ~sqrt(D) smaller, depending on the scale).

We now expose both:
  - ``overall_dissimilarity`` = sqrt(mean squared Euclidean distance)
    (the additive per-variable decomposition is still in the
    contribution rows; we just take the square root at the end so the
    reported number is the conventional Euclidean distance).
  - ``overall_squared_distance`` = the raw mean of the per-pair squared
    Euclidean distances (the additively-decomposable quantity Clarke's
    table is built on).

For Bray-Curtis the behaviour is unchanged: ``overall_dissimilarity`` is
the average Bray-Curtis dissimilarity, and no
``overall_squared_distance`` is exposed (it would equal 0 by definition
in BC).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.testing import assert_allclose

from stats.simper import SimperAnalyzer


class TestSimperEuclidean:
    """Verify SIMPER's ``metric='euclidean'`` semantics."""

    def test_euclidean_overall_dissimilarity_is_sqrt_of_squared(self):
        """
        ``overall_dissimilarity`` for metric='euclidean' is the *plain*
        Euclidean distance, i.e. sqrt of the mean squared distance.
        """
        # Two distinct points in 2D: d = 3, d^2 = 9
        data = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
            ]
        )
        groups = [0, 0, 1, 1]
        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups, metric="euclidean")
        # The mean of squared distances across the 4 cross-pairs is 9,
        # so the Euclidean distance is 3.
        assert_allclose(result.overall_dissimilarity, 3.0, atol=1e-6)

    def test_euclidean_squared_distance_exposed(self):
        """
        The raw mean of squared Euclidean distances is exposed under
        ``overall_squared_distance`` (only meaningful for metric='euclidean').
        """
        data = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
            ]
        )
        groups = [0, 0, 1, 1]
        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups, metric="euclidean")
        assert hasattr(result, "overall_squared_distance")
        # Mean of squared distances (4 pairs, each d^2 = 9) = 9
        assert_allclose(result.overall_squared_distance, 9.0, atol=1e-6)
        # overall_dissimilarity is the sqrt of that
        assert_allclose(
            result.overall_dissimilarity,
            np.sqrt(result.overall_squared_distance),
            atol=1e-6,
        )

    def test_euclidean_squared_field_absent_for_bray_curtis(self):
        """``overall_squared_distance`` is only defined for metric='euclidean'."""
        rng = np.random.default_rng(0)
        data = rng.random((10, 4))
        groups = [0] * 5 + [1] * 5
        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups, metric="bray_curtis")
        # For BC the squared-distance field is meaningless; we don't
        # expose it.
        assert not hasattr(result, "overall_squared_distance") or result.overall_squared_distance is None

    def test_euclidean_per_variable_decomposition_sums_to_squared(self):
        """
        The per-variable contributions must still sum to the *squared*
        distance (the additive decomposition is on squared Euclidean).
        ``overall_squared_distance`` == sum of avg_vec.
        """
        rng = np.random.default_rng(1)
        data = rng.random((12, 5))
        groups = [0] * 6 + [1] * 6
        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups, metric="euclidean")
        total_avg = sum(c.average for c in result.contributions)
        assert_allclose(total_avg, result.overall_squared_distance, rtol=1e-10)
