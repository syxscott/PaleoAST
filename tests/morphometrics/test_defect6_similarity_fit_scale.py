"""
Regression tests for defect 6: ``_similarity_fit`` used the wrong scale
denominator.

The function rescales A to B's norm and then divides by ``sum(A**2)`` —
but A has already been rescaled, so the scale ``s`` is off by the ratio
``||A||² / ||B||²``.  The inverse map ``to_raw`` then has to compensate
with the same factor.

The fix uses the un-rescaled A for the SVD and divides by
``trace(A_origᵀ A_orig)``, which is the textbook formula for the
similarity scale (Schönemann 1966; Bookstein 1989).

The test verifies the missing-landmark estimator is **scale equivariant**:
scaling the entire landmark configuration by a uniform factor MUST leave
the filled-in missing coordinates identical (the bug made the filled
coordinates drift with the scale).
"""

import numpy as np
import pytest

from morphometrics.missing import estimate_missing


def _two_with_missing(seed: int = 11, scale: float = 1.0, n_complete: int = 4):
    """Several complete specimens and one with two landmarks missing.
    Used to test the missing-landmark estimator."""
    rng = np.random.default_rng(seed)
    complete = rng.normal(size=(n_complete, 5, 2))
    incomplete = rng.normal(size=(1, 5, 2))
    # hide landmarks 2 and 3 on the incomplete specimen
    incomplete[0, 2] = np.nan
    incomplete[0, 3] = np.nan
    cfgs = np.concatenate([complete, incomplete], axis=0)
    if scale != 1.0:
        cfgs = cfgs * scale
    return cfgs


class TestSimilarityFitScaleEquivariance:
    def test_missing_estimate_invariant_to_uniform_scale(self):
        """Scaling every landmark by a uniform factor MUST leave the
        estimated missing coordinates (in the original frame) identical
        — only the relative geometry matters."""
        cfgs_1 = _two_with_missing(scale=1.0, n_complete=4)
        cfgs_3 = _two_with_missing(scale=3.0, n_complete=4)

        r1 = estimate_missing(cfgs_1, method="tps")
        r3 = estimate_missing(cfgs_3, method="tps")

        # Both fill in landmarks 2 and 3 of specimen 2.  Their absolute
        # positions are scaled, so the relative POSITION must match when
        # one is rescaled.
        miss = (2, 3)
        est1 = r1.filled_configurations[-1, miss]
        est3 = r3.filled_configurations[-1, miss]
        np.testing.assert_allclose(est1, est3 / 3.0, atol=1e-8)

    def test_missing_estimate_invariant_to_uniform_scale_regression(self):
        """Same as above with the regression method (least squares)."""
        cfgs_1 = _two_with_missing(scale=1.0, n_complete=4)
        cfgs_2 = _two_with_missing(scale=2.5, n_complete=4)

        r1 = estimate_missing(cfgs_1, method="reg")
        r2 = estimate_missing(cfgs_2, method="reg")

        miss = (2, 3)
        est1 = r1.filled_configurations[-1, miss]
        est2 = r2.filled_configurations[-1, miss]
        np.testing.assert_allclose(est1, est2 / 2.5, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])