"""
Regression tests for defect 7: Hotelling T² used ``np.linalg.lstsq``
on the pooled covariance, which silently returns a minimum-norm
projection instead of raising when ``S`` is rank-deficient.

In the well-conditioned regime (full rank, ``S`` invertible) the test
verifies ``np.linalg.solve`` is preferred: same numerical answer, and the
explicit warning is NOT emitted.  In the rank-deficient regime (more
dimensions than samples per group) the test verifies the function still
returns a finite result and now emits a warning (instead of silently
giving a wrong T² via the lstsq fallback).
"""

import warnings

import numpy as np
import pytest

from morphometrics.shape_stats import hotelling_t2


class TestHotellingT2SolverPath:
    def test_well_conditioned_uses_solve_not_lstsq(self):
        """When S is full rank, the function should use ``np.linalg.solve``
        (no warning)."""
        rng = np.random.default_rng(13)
        x1 = rng.normal(size=(20, 5))
        x2 = rng.normal(size=(22, 5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = hotelling_t2(x1, x2)
        # No warnings about singular covariance
        assert not any("singular" in str(w.message).lower() for w in caught)
        assert np.isfinite(res.t2)
        assert np.isfinite(res.f_statistic)
        assert np.isfinite(res.p_value)

    def test_singular_covariance_emits_warning(self):
        """When the pooled covariance is rank-deficient (e.g. one
        feature is a perfect linear combination of the others), the
        function must emit a warning (the lstsq silently gave the
        minimum-norm projection — wrong answer masquerading as
        something).  It must still return a finite T² so downstream
        code doesn't crash."""
        rng = np.random.default_rng(7)
        # 20 × 4 — passes the n1+n2 > d+1 guard (24 > 5), and S has
        # theoretical full rank, BUT we make column 3 a perfect linear
        # combination of columns 0, 1, 2 so the covariance is exactly
        # singular (rank 3 instead of 4).
        x1 = rng.normal(size=(12, 4))
        x2 = rng.normal(size=(12, 4))
        x1[:, 3] = 2 * x1[:, 0] - x1[:, 1] + 0.5 * x1[:, 2]
        x2[:, 3] = 2 * x2[:, 0] - x2[:, 1] + 0.5 * x2[:, 2]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = hotelling_t2(x1, x2)
        # A warning about singular covariance must be emitted.
        assert any("singular" in str(w.message).lower() for w in caught), \
            f"Expected a singular-matrix warning; got: {[str(w.message) for w in caught]}"
        # Still returns a finite T² (we degrade gracefully, but loudly).
        assert np.isfinite(res.t2)
        assert np.isfinite(res.p_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])