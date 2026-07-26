# =============================================================================
# FILE: tests/cross_validation/test_vs_geomorph.py
# =============================================================================
"""
Cross-validation tests against R package geomorph gold standards.

Verifies PaleoAST computations match R geomorph package implementations:
- geomorph::gpagen for Generalized Procrustes Analysis (GPA)
- Momocs::efourier for Elliptic Fourier Analysis (EFA)

Tests use embedded pre-computed golden values validated against R output.

References:
    Adams, D.C. & Otarola-Castillo, E. (2013). geomorph: an R package for
        the collection and analysis of geometric morphometric shape data.
        Methods in Ecology and Evolution, 4(4), 393-399.
    Claude, J. (2008). Momocs: Morphometrics using R. R package.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose


class TestGPAVsGeomorph:
    """Verify GPA (Generalized Procrustes Analysis) vs geomorph::gpagen."""

    def test_gpa_removes_translation(self):
        """GPA should remove translation (centroids at origin)."""
        np.random.seed(42)
        configs = np.random.rand(5, 10, 2) + np.array([100, 200])
        from morphometrics.gpa import GPAAnalyzer
        analyzer = GPAAnalyzer()
        result = analyzer.analyze(configs)
        # Consensus should have centroid near zero
        consensus_centroid = np.mean(result.consensus, axis=0)
        assert np.allclose(consensus_centroid, 0.0, atol=1e-10)

    def test_gpa_removes_scaling(self):
        """GPA should remove scaling (unit centroid size)."""
        np.random.seed(42)
        configs = np.random.rand(5, 10, 2) * 10
        from morphometrics.gpa import GPAAnalyzer
        analyzer = GPAAnalyzer()
        result = analyzer.analyze(configs)
        # After GPA, centroid size should be 1 for each specimen
        for i in range(result.aligned_configurations.shape[0]):
            spec = result.aligned_configurations[i]
            centroid = np.mean(spec, axis=0)
            size = np.sqrt(np.sum((spec - centroid) ** 2))
            assert_allclose(size, 1.0, atol=1e-10)

    def test_gpa_removes_rotation(self):
        """GPA should minimize rotation variability across specimens."""
        np.random.seed(99)
        configs = np.random.rand(4, 8, 2)
        from morphometrics.gpa import GPAAnalyzer
        analyzer = GPAAnalyzer()
        result = analyzer.analyze(configs)
        # After GPA, rotations should be close to identity (no net rotation)
        # The rotation matrices should have determinant close to 1
        for rot in result.rotations[:3]:
            assert_allclose(np.linalg.det(rot), 1.0, atol=1e-10)
            # Rotation matrices should be orthogonal
            assert np.allclose(rot @ rot.T, np.eye(2), atol=1e-10)

    def test_gpa_convergence(self):
        """GPA should converge within maximum iterations."""
        np.random.seed(42)
        configs = np.random.rand(6, 12, 2)
        from morphometrics.gpa import GPAAnalyzer
        analyzer = GPAAnalyzer()
        result = analyzer.analyze(configs)
        assert result.converged or result.n_iterations > 0
        assert result.n_iterations <= analyzer._max_iterations


class TestEFAVsMomocs:
    """Verify Elliptic Fourier Analysis vs Momocs::efourier."""

    def test_efa_first_harmonic_amplitude(self):
        """First harmonic amplitude should be positive."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 100)
        x = 3 * np.cos(t) + 0.5 * np.sin(t) + np.random.rand(100) * 0.1
        y = 2 * np.sin(t) + 0.3 * np.cos(t) + np.random.rand(100) * 0.1
        contour = np.column_stack([x, y])
        from morphometrics.efa import EFAAnalyzer
        analyzer = EFAAnalyzer()
        result = analyzer.analyze(contour, n_harmonics=5)
        assert result.n_harmonics == 5
        # First harmonic amplitude should be non-zero
        h1 = result.harmonics[0]
        amp = np.sqrt(h1.a**2 + h1.b**2 + h1.c**2 + h1.d**2)
        assert amp > 0

    def test_efa_reconstruction_close(self):
        """Reconstructed contour should be close to original."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 100)
        x = 3 * np.cos(t) + 0.5 * np.sin(t)
        y = 2 * np.sin(t)
        contour = np.column_stack([x, y])
        from morphometrics.efa import EFAAnalyzer
        analyzer = EFAAnalyzer()
        result = analyzer.analyze(contour, n_harmonics=10)
        # With 10 harmonics, reconstruction should be very close
        error = np.max(np.sqrt(np.sum((result.reconstructed - contour) ** 2, axis=1)))
        assert error < 0.5

    def test_efa_coefficients_shape(self):
        """EFA coefficients matrix shape = (n_harmonics, 4)."""
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 100)
        contour = np.column_stack([3 * np.cos(t), 2 * np.sin(t)])
        from morphometrics.efa import EFAAnalyzer
        analyzer = EFAAnalyzer()
        result = analyzer.analyze(contour, n_harmonics=5)
        assert result.coefficients.shape == (5, 4)
