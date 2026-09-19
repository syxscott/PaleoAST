# =============================================================================
# FILE: tests/golden/test_real_datasets.py
# =============================================================================
"""
Golden regression tests on real published datasets (data/golden/*.npz).

Fixtures are converted from the geomorph R package:
    - hummingbirds: Berns & Adams (2010), 25 specimens, 44 2D points,
      15 semilandmark triples (geomorph curvepts rows = anchor, slider,
      anchor; slider is the middle column).
    - scallops: Serb et al. (2011), 5 specimens (one per species),
      46 3D points, curve sliders (curvslide) + surface outline 17-46.

Unlike the synthetic analytic tests in tests/morphometrics, these lock the
GPA/partial-GPA pipeline against real-shape invariances: rigid-motion and
rotation invariance, reflection sensitivity, unit centroid size, det = +1
rotations, convergence and idempotence.

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphometrics.gpa import GPAAnalyzer, partial_gpa

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"


def _centroid_sizes(x: np.ndarray) -> np.ndarray:
    """Centroid size per specimen for (n, k, m) input."""
    centred = x - x.mean(axis=1, keepdims=True)
    return np.sqrt((centred**2).sum(axis=(1, 2)))


def _rotation(theta: float = 0.7) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rotation3d(seed: int = 42) -> np.ndarray:
    q, _r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q = q @ np.diag([1.0, 1.0, -1.0])
    return q


@pytest.fixture(scope="module")
def hummingbirds() -> dict[str, np.ndarray]:
    with np.load(GOLDEN_DIR / "hummingbirds.npz") as z:
        return {k: z[k].copy() for k in z.files}


@pytest.fixture(scope="module")
def scallops() -> dict[str, np.ndarray]:
    with np.load(GOLDEN_DIR / "scallops.npz") as z:
        return {k: z[k].copy() for k in z.files}


class TestFixtureIntegrity:
    def test_hummingbirds_layout(self, hummingbirds):
        cfg = hummingbirds["configurations"]
        assert cfg.shape == (25, 44, 2)
        assert cfg.dtype == np.float64
        assert np.isfinite(cfg).all()
        triples = hummingbirds["curve_triples_1based"]
        assert triples.shape == (15, 3)
        assert triples.min() >= 1 and triples.max() <= 44
        sliders = np.unique(triples[:, 1])
        assert len(sliders) == 15, "each row names exactly one slider (middle column)"

    def test_scallops_layout(self, scallops):
        cfg = scallops["configurations"]
        assert cfg.shape == (5, 46, 3)
        assert cfg.dtype == np.float64
        assert np.isfinite(cfg).all()
        assert sorted(scallops["species"].tolist()) == [1, 2, 3, 4, 5]
        surf = scallops["surface_slide_1based"]
        assert surf.tolist() == list(range(17, 47))
        curv = scallops["curvslide_triples_1based"]
        assert curv.shape == (11, 3)


class TestHummingbirdsGPA:
    def test_gpa_invariants(self, hummingbirds):
        result = GPAAnalyzer().analyze(hummingbirds["configurations"])
        assert result.converged
        aligned = result.aligned_configurations
        assert aligned.shape == (25, 44, 2)
        assert_allclose(aligned.mean(axis=1), 0.0, atol=1e-10)
        assert_allclose(_centroid_sizes(aligned), 1.0, atol=1e-10)
        for rot in result.rotations:
            assert_allclose(np.linalg.det(rot), 1.0, atol=1e-10)

    def test_centroid_sizes_match_manual(self, hummingbirds):
        cfg = hummingbirds["configurations"]
        result = GPAAnalyzer().analyze(cfg)
        assert_allclose(result.centroid_sizes, _centroid_sizes(cfg), rtol=1e-10)

    def test_rigid_motion_invariance(self, hummingbirds):
        cfg = hummingbirds["configurations"]
        base = GPAAnalyzer().analyze(cfg)
        q = _rotation(0.7)
        moved = (cfg - np.array([12.5, -33.0])) @ q.T * 4.7 + np.array([5.0, 9.0])
        again = GPAAnalyzer().analyze(moved)
        # Iterative GPA fixes the overall orientation only up to a rotation,
        # so invariance is asserted equivariantly: spun input == base spun.
        assert_allclose(again.aligned_configurations, base.aligned_configurations @ q.T, atol=1e-12)
        assert_allclose(again.procrustes_distances, base.procrustes_distances, atol=1e-12)
        assert again.final_sse == pytest.approx(base.final_sse, rel=1e-12)

    def test_reflection_sensitivity(self, hummingbirds):
        cfg = hummingbirds["configurations"]
        base = GPAAnalyzer().analyze(cfg)
        mirrored = cfg.copy()
        mirrored[0, :, 0] *= -1.0
        reflect_off = GPAAnalyzer().analyze(mirrored, no_reflect=True)
        reflect_on = GPAAnalyzer().analyze(mirrored, no_reflect=False)
        # With proper rotations only, the mirror-imaged bird cannot snap back
        # onto the consensus: its fit strictly degrades.
        assert reflect_off.final_sse > base.final_sse * 1.01
        assert reflect_off.procrustes_distances[0] > base.procrustes_distances[0]
        assert_allclose(np.linalg.det(reflect_off.rotations[0]), 1.0, atol=1e-10)
        # Allowing the improper move flips that one specimen back exactly.
        assert reflect_on.final_sse == pytest.approx(base.final_sse, rel=1e-8)

    def test_partial_gpa_sliding(self, hummingbirds):
        cfg = hummingbirds["configurations"]
        triples0 = hummingbirds["curve_triples_1based"] - 1
        curves = [[int(a), int(b), int(c)] for a, b, c in triples0]
        sliders = sorted(set(int(s) for s in triples0[:, 1]))
        fixed = sorted(set(range(44)) - set(sliders))
        plain = GPAAnalyzer().analyze(cfg)
        result = partial_gpa(cfg, fixed_landmarks=fixed, curves=curves, n_dims=2)
        assert np.isfinite(result.aligned_configurations).all()
        assert_allclose(result.aligned_configurations.mean(axis=1), 0.0, atol=1e-10)
        assert_allclose(_centroid_sizes(result.aligned_configurations), 1.0, atol=1e-10)
        assert np.all(result.bending_energies >= 0.0)
        assert result.sliding_iterations >= 1
        # On real data the sliders genuinely move off their digitised spots.
        moved = np.abs(result.aligned_configurations[:, sliders, :]
                       - plain.aligned_configurations[:, sliders, :]).max()
        assert moved > 1e-6


class TestScallopsGPA:
    def test_gpa_invariants_3d(self, scallops):
        result = GPAAnalyzer().analyze(scallops["configurations"])
        assert result.converged
        aligned = result.aligned_configurations
        assert aligned.shape == (5, 46, 3)
        assert_allclose(aligned.mean(axis=1), 0.0, atol=1e-10)
        assert_allclose(_centroid_sizes(aligned), 1.0, atol=1e-10)
        for rot in result.rotations:
            assert rot.shape == (3, 3)
            assert_allclose(np.linalg.det(rot), 1.0, atol=1e-10)
            assert_allclose(rot @ rot.T, np.eye(3), atol=1e-10)

    def test_idempotence(self, scallops):
        first = GPAAnalyzer().analyze(scallops["configurations"])
        second = GPAAnalyzer().analyze(first.aligned_configurations)
        # Re-alignment is (near) a fixed point; residual drift is bounded by
        # the convergence tolerance of the first pass.
        assert_allclose(second.aligned_configurations, first.aligned_configurations, atol=1e-5)
        assert_allclose(second.consensus, first.consensus, atol=1e-7)
        assert second.final_sse == pytest.approx(first.final_sse, rel=1e-6)

    def test_rotation_invariance_3d(self, scallops):
        cfg = scallops["configurations"]
        base = GPAAnalyzer().analyze(cfg)
        q = _rotation3d()
        spun = (cfg - cfg.mean(axis=1, keepdims=True)) @ q.T * 2.3 + np.array([7.0, -2.0, 4.0])
        again = GPAAnalyzer().analyze(spun)
        assert_allclose(again.aligned_configurations, base.aligned_configurations @ q.T, atol=1e-12)
        assert again.final_sse == pytest.approx(base.final_sse, rel=1e-12)

    def test_centroid_sizes_match_manual(self, scallops):
        cfg = scallops["configurations"]
        result = GPAAnalyzer().analyze(cfg)
        assert_allclose(result.centroid_sizes, _centroid_sizes(cfg), rtol=1e-10)

    def test_partial_gpa_surface_sliding(self, scallops):
        cfg = scallops["configurations"]
        curv0 = scallops["curvslide_triples_1based"] - 1
        surf0 = [int(i) - 1 for i in scallops["surface_slide_1based"]]
        sliders = set(int(s) for s in curv0[:, 1]) | set(surf0[1:-1])
        fixed = sorted(set(range(46)) - sliders)
        result = partial_gpa(
            cfg,
            fixed_landmarks=fixed,
            curves=[[int(a), int(b), int(c)] for a, b, c in curv0],
            surfaces=[surf0],
            n_dims=3,
        )
        assert np.isfinite(result.aligned_configurations).all()
        assert_allclose(result.aligned_configurations.mean(axis=1), 0.0, atol=1e-10)
        assert_allclose(_centroid_sizes(result.aligned_configurations), 1.0, atol=1e-10)
        # 3D bending energy uses the ||x-y|| TPS kernel and is only sign-
        # conventional, so assert finiteness rather than a positive bound.
        assert np.all(np.isfinite(result.bending_energies))
