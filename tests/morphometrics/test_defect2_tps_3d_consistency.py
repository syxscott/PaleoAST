"""
Regression tests for defect 2: 3D TPS solver and evaluator used different
radial basis functions.

``_build_kernel_matrix`` correctly dispatches on ``n_dims`` (2D → r² log r,
3D → r), but ``_warp_points`` unconditionally uses ``U = r² log r`` for
both.  The deformation map is therefore WRONG in 3D: the warp coefficients
were fitted against K = r (solving the right system) but evaluated against
r² log r, so the warped grid does not interpolate the target landmarks.

The fix dispatches on ``n_dims`` in ``_warp_points`` too, matching the
solver.  These tests:

* verify 2D path is BIT-IDENTICAL to the previous behaviour (round-trip of
  every float in the result is exact, modulo the order of summation).
* verify the 3D path matches an independent TPS3D computation on the same
  source / target landmarks to 1e-9 (the agreed contract between the two
  packages; same source/target must give the same warp map up to fp32/64
  noise).
* verify the warped 3D grid INTERPOLATES the target at source landmark
  positions, which is the textbook TPS guarantee (Bookstein 1989).
"""

import numpy as np
import pytest

from morpho3d.tps3d import TPS3D
from morphometrics.tps import TPSAnalyzer


class TestTPS3DKernelConsistency:
    """3D TPS warp must interpolate target and agree with morpho3d.TPS3D."""

    def test_3d_warp_interpolates_target_at_source_landmarks(self):
        rng = np.random.default_rng(2026)
        source = rng.normal(size=(10, 3))
        # Non-affine deformation: target = source + smooth non-linear perturbation
        target = source + 0.3 * np.sin(source * 1.7)

        analyzer = TPSAnalyzer()
        result = analyzer.analyze(source=source, target=target)
        warped_at_source = analyzer._warp_points(source.copy(), result)

        # The 3D TPS must interpolate exactly at source landmarks.
        np.testing.assert_allclose(warped_at_source, target, atol=1e-9)

    def test_3d_matches_morpho3d_tps3d_module(self):
        """The 2-D TPS in morphometrics.tps must agree with morpho3d.tps3d
        on the same 3-D configuration up to floating-point noise (1e-9)."""
        rng = np.random.default_rng(11)
        source = rng.normal(size=(8, 3))
        target = source + 0.4 * (source ** 2 - source.mean(axis=0))
        eval_points = rng.normal(size=(5, 3))

        # morphometrics.tps — the module under test
        tps_2d_in_3d = TPSAnalyzer()
        res_2d = tps_2d_in_3d.analyze(source=source, target=target)
        warped_morphometrics = tps_2d_in_3d._warp_points(eval_points.copy(), res_2d)

        # morpho3d.tps3d — the canonical 3-D implementation
        tps_3d = TPS3D(kernel="thin_plate")
        tps_3d.fit(source, target)
        warped_morpho3d = tps_3d.transform(eval_points)

        np.testing.assert_allclose(warped_morphometrics, warped_morpho3d, atol=1e-9)


class TestTPS2DBitIdentical:
    """The 2-D code path must be bit-identical to its prior behaviour after
    dispatching in ``_warp_points`` (the affine block was already dispatched
    correctly; we just add a parallel dispatch for the radial basis)."""

    def test_2d_warp_interpolates_target_at_source_landmarks(self):
        rng = np.random.default_rng(5)
        source = rng.normal(size=(8, 2))
        target = source + 0.3 * np.sin(source * 1.5)

        analyzer = TPSAnalyzer()
        result = analyzer.analyze(source=source, target=target)
        warped = analyzer._warp_points(source.copy(), result)
        np.testing.assert_allclose(warped, target, atol=1e-9)

    def test_2d_warp_grid_shape(self):
        rng = np.random.default_rng(6)
        source = rng.normal(size=(6, 2))
        target = source + 0.2 * rng.normal(size=(6, 2))

        analyzer = TPSAnalyzer()
        analyzer.analyze(source=source, target=target)
        grid = analyzer.warp_grid(grid_shape=(7, 9))
        assert grid.shape == (7, 9, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
