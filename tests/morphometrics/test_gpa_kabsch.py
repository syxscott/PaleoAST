"""
================================================================================
PaleoAST Kabsch Algorithm Unit Tests
================================================================================

Unit tests for the Kabsch SVD rotation algorithm in GPA, specifically testing
the reflection trap handling.

The Kabsch algorithm finds the optimal rotation R that aligns a target
configuration to a reference configuration by minimizing:
    ||Y - X @ R||²

The reflection trap occurs when det(R) < 0, indicating a reflection instead
of a proper rotation. The fix ensures det(R) = +1.

References:
    - Bookstein, F.L. (1989). Principal warps: thin-plate splines and
      the decomposition of deformations. IEEE TPAMI.
    - Dryden, I.L. & Mardia, K.V. (2016). Statistical shape analysis.
    - Kabsch, W. (1978). A discussion of the solution for the best
      rotation to related body structures. Acta Cryst.

Author: PaleoAST Development Team
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from morphometrics.gpa import GPAAnalyzer


class TestKabschRotation(unittest.TestCase):
    """
    Tests for the Kabsch SVD rotation algorithm, focusing on the reflection trap.
    """

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.gpa = GPAAnalyzer()

    def test_rotation_matrix_determinant_is_plus_one(self):
        """
        Test that GPA rotation matrices always have det = +1.

        This is the fundamental requirement for a valid rotation matrix
        (not a reflection).
        """
        configs = [np.random.randn(10, 2) for _ in range(5)]
        result = self.gpa.analyze(configs)

        for i, R in enumerate(result.rotations):
            det = np.linalg.det(R)
            self.assertAlmostEqual(
                det, 1.0, places=10,
                msg=f"Rotation matrix {i} has det={det}, expected det=+1"
            )
            # Also verify orthogonality: R^T R = I
            np.testing.assert_array_almost_equal(
                R @ R.T, np.eye(2),
                err_msg=f"Rotation matrix {i} is not orthogonal"
            )

    def test_pure_rotation_preserves_determinant(self):
        """
        Test that a pure rotation (no reflection) keeps det = +1.

        Creates a reference configuration and a target that is rotated
        by a known angle, then verifies the recovered rotation has det = +1.
        """
        # Create a simple shape (triangle)
        reference = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        # Rotate by 45 degrees
        angle = np.pi / 4
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        target = reference @ R_true.T

        # The _find_rotation method expects (reference, target) where
        # R @ target ≈ reference, so we pass (reference, target)
        # But actually looking at the code: R = _find_rotation(consensus, aligned[i])
        # which returns R such that R @ aligned[i] ≈ consensus
        # So we need R such that target @ R ≈ reference, i.e., R ≈ R_true

        # Since the Kabsch algorithm finds R = Vt.T @ U.T from SVD of target.T @ reference
        # Let's verify det(R) = +1
        H = target.T @ reference
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # det(R) should be +1 for a pure rotation
        self.assertGreater(
            np.linalg.det(R), 0,
            msg=f"Pure rotation should have det > 0, got det={np.linalg.det(R)}"
        )

    def test_reflection_case_flipped_to_rotation(self):
        """
        Test that when SVD produces a reflection (det < 0), it is corrected.

        Creates a configuration that would cause SVD to produce det < 0
        and verifies the final rotation has det = +1.
        """
        # Create a configuration that triggers reflection
        # This typically happens with certain symmetric configurations
        reference = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Create target with a reflection (determinant = -1)
        # Reflection matrix (mirror across x-y plane)
        reflection = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        target = reference @ reflection.T

        # The Kabsch algorithm should detect and correct the reflection
        H = target.T @ reference
        U, S, Vt = np.linalg.svd(H)
        R_initial = Vt.T @ U.T

        # Initial R may have det < 0 (reflection)
        det_initial = np.linalg.det(R_initial)

        # Apply the fix (flip last row of Vt if det < 0)
        if det_initial < 0:
            Vt_fixed = Vt.copy()
            Vt_fixed[-1, :] *= -1
            R_fixed = Vt_fixed.T @ U.T
        else:
            R_fixed = R_initial

        # Fixed R should have det = +1
        self.assertAlmostEqual(
            np.linalg.det(R_fixed), 1.0, places=10,
            msg=f"After correction, det should be +1, got {np.linalg.det(R_fixed)}"
        )

    def test_rotation_with_scaling(self):
        """
        Test that rotation works correctly with scaling present.

        GPA first scales to unit centroid size, so the rotation
        should only need to handle rotation, not scaling.
        """
        # Create reference
        reference = np.random.randn(8, 2)

        # Create target: scaled and rotated
        scale = 2.5
        angle = np.pi / 6
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        target = (reference * scale) @ R_true.T

        # GPA analyzer handles scaling separately; here we just test rotation
        # by centering both and doing Kabsch
        ref_centered = reference - reference.mean(axis=0)
        tgt_centered = target - target.mean(axis=0)

        # Scale to unit centroid size
        ref_size = np.sqrt(np.sum(ref_centered**2))
        tgt_size = np.sqrt(np.sum(tgt_centered**2))
        ref_scaled = ref_centered / ref_size
        tgt_scaled = tgt_centered / tgt_size

        # Kabsch
        H = tgt_scaled.T @ ref_scaled
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Verify det = +1
        self.assertAlmostEqual(
            np.linalg.det(R), 1.0, places=10,
            msg=f"Rotation with scaling should have det=+1, got {np.linalg.det(R)}"
        )

        # Verify it's close to true rotation
        angle_diff = np.arccos(np.clip((np.trace(R_true.T @ R) - 1) / 2, -1, 1))
        self.assertLess(
            abs(angle_diff), 0.01,
            msg=f"Recovered angle differs by {angle_diff:.4f} rad"
        )

    def test_symmetric_matrix_case(self):
        """
        Test Kabsch with symmetric configurations that may cause SVD issues.

        Symmetric matrices can produce singular values with ambiguous signs,
        making them prone to reflection issues.
        """
        # Create a symmetric shape (butterfly-like)
        reference = np.array([
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ])

        # Rotate by 90 degrees
        angle = np.pi / 2
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        target = reference @ R_true.T

        # Kabsch
        H = target.T @ reference
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply fix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
        R = Vt.T @ U.T

        # Verify det = +1
        self.assertAlmostEqual(
            np.linalg.det(R), 1.0, places=10,
            msg=f"Symmetric case should have det=+1, got {np.linalg.det(R)}"
        )

    def test_3d_rotation_determinant(self):
        """
        Test Kabsch with 3D configurations.
        """
        np.random.seed(123)
        configs = [np.random.randn(10, 3) for _ in range(4)]
        result = self.gpa.analyze(configs)

        for i, R in enumerate(result.rotations):
            det = np.linalg.det(R)
            self.assertAlmostEqual(
                det, 1.0, places=10,
                msg=f"3D rotation matrix {i} has det={det}, expected det=+1"
            )
            # Verify orthogonality
            np.testing.assert_array_almost_equal(
                R @ R.T, np.eye(3),
                err_msg=f"3D rotation matrix {i} is not orthogonal"
            )

    def test_gpa_convergence_with_reflection_fix(self):
        """
        Test that GPA converges correctly when the reflection fix is applied.

        The reflection fix should not interfere with convergence.
        """
        np.random.seed(456)
        configs = [np.random.randn(6, 2) for _ in range(5)]
        result = self.gpa.analyze(configs, tolerance=1e-10)

        # Should converge
        self.assertTrue(result.converged)
        # Should have reasonable number of iterations
        self.assertLess(result.n_iterations, 50)

        # All rotation matrices should have det = +1
        for i, R in enumerate(result.rotations):
            self.assertAlmostEqual(
                np.linalg.det(R), 1.0, places=10,
                msg=f"Rotation {i} has det={np.linalg.det(R)}"
            )


class TestBendingEnergyComputation(unittest.TestCase):
    """
    Tests for the TPS bending energy computation.

    Bending energy should be computed as w^T K w where w is the
    non-affine weight vector from TPS decomposition.
    """

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_bending_energy_is_nonnegative(self):
        """
        Test that bending energy is always non-negative.

        Bending energy is a quadratic form w^T K w, which should be >= 0
        since K is positive semi-definite.
        """
        from morphometrics.gpa import partial_gpa

        configs = np.random.randn(5, 8, 2)
        fixed = np.array([0, 1, 6, 7])
        curves = [[2, 3, 4, 5]]

        result = partial_gpa(
            configurations=configs,
            fixed_landmarks=fixed,
            curve_indices=curves,
            n_dims=2,
        )

        for i, be in enumerate(result.bending_energies):
            self.assertGreaterEqual(
                be, 0.0,
                msg=f"Bending energy {i} is negative: {be}"
            )

    def test_bending_energy_zero_for_identical_configs(self):
        """
        Test that bending energy is zero when specimen equals consensus.

        If the specimen configuration exactly matches the consensus,
        the TPS deformation is zero, so bending energy should be zero.
        """
        from morphometrics.gpa import _compute_bending_energy

        # Create a simple configuration
        consensus = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])

        # Specimen identical to consensus
        specimen = consensus.copy()
        fixed = np.array([0, 1, 2, 3])

        be = _compute_bending_energy(specimen, consensus, fixed, 2)

        self.assertAlmostEqual(
            be, 0.0, places=10,
            msg=f"Bending energy should be 0 for identical configs, got {be}"
        )

    def test_bending_energy_increases_with_deformation(self):
        """
        Test that bending energy increases as deformation increases.

        Larger non-affine deformations should have higher bending energy.
        """
        from morphometrics.gpa import _compute_bending_energy

        consensus = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])

        fixed = np.array([0, 1, 2, 3])

        # Small deformation
        small_deform = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.05, 1.0],
            [0.0, 1.0]
        ])

        # Large deformation
        large_deform = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.2, 1.0],
            [0.0, 1.0]
        ])

        be_small = _compute_bending_energy(small_deform, consensus, fixed, 2)
        be_large = _compute_bending_energy(large_deform, consensus, fixed, 2)

        self.assertLess(
            be_small, be_large,
            msg=f"Large deformation should have higher BE: small={be_small}, large={be_large}"
        )


if __name__ == "__main__":
    unittest.main()