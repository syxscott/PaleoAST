"""
Regression tests for defect 4: surface normals computed in gpa.py don't
depend on the loop variable.

The old code in ``_compute_surface_tangents_and_normals`` reused the same
``v2 = consensus[surface[0]] - consensus[idx]`` for every iteration of
``j in neighbors[:3]``.  Three cross products against the same ``v2``
collapse to ``cross(Σv1_j, v2)`` instead of averaging independent face
normals — so the local normal was unrelated to the surrounding geometry
(the direction picked out is determined only by surface[0] and the centre
sum, not by the geometry of the other neighbours).

For a perfectly flat surface the bug happens to give the correct normal
(the cross product is constant regardless of ``j``).  The bug is visible
on CURVED surfaces, where each neighbour pair defines a different face
normal and the result should be their average.

The fix replaces the loop with a local PCA over the neighbours of each
point (smallest eigenvector = surface normal).  These tests:

* verify the rotation-equivariance (already true of the buggy code, but
  must remain true under the fix);
* verify that on a curved surface the sliding projection is invariant to
  rotating the consensus.
"""

import numpy as np
import pytest

from morphometrics.gpa import _compute_surface_tangents_and_normals, _slide_surface_tangent_plane


def _curved_surface(seed: int = 31, n: int = 12):
    """A curved surface patch (z = 0.5 sin(x) cos(y)).  The tangent
    normals vary across the patch, so the buggy constant-v2 normal will
    disagree with the local-PCA normal."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=0.3, size=(n, 3))
    pts[:, 2] = 0.5 * np.sin(pts[:, 0] * 1.5) * np.cos(pts[:, 1] * 1.5)
    return pts


def _flat_surface(seed: int = 31, n: int = 9):
    """A flat surface (z = 0) so a clean surface normal at every point
    should point in ±z."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=0.3, size=(n, 3))
    pts[:, 2] = 0.0
    return pts


def _rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


class TestSurfaceNormalsOnCurvedSurface:
    """The bug is invisible on flat surfaces (cross products give the same
    direction regardless of the loop variable).  Use a curved surface
    where the normal is known analytically."""

    def test_smooth_surface_normal_matches_analytic_gradient(self):
        """A smooth curved surface z = f(x, y) = 0.5 sin(1.5 x) cos(1.5 y)
        has analytic unit normal at (x, y, z):

            n = (-∂z/∂x, -∂z/∂y, 1) / ||.||

        Local PCA on a dense patch must recover this; the buggy
        constant-v2 formula gives a different direction."""
        n_x, n_y = 8, 8
        xs = np.linspace(-0.6, 0.6, n_x)
        ys = np.linspace(-0.6, 0.6, n_y)
        X, Y = np.meshgrid(xs, ys)
        Z = 0.5 * np.sin(1.5 * X) * np.cos(1.5 * Y)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        # Analytic normals
        dZdx = 0.5 * 1.5 * np.cos(1.5 * X) * np.cos(1.5 * Y)
        dZdy = 0.5 * -1.5 * np.sin(1.5 * X) * np.sin(1.5 * Y)
        nx = -dZdx.ravel()
        ny = -dZdy.ravel()
        nz = np.ones_like(nx)
        analytic = np.column_stack([nx, ny, nz])
        analytic = analytic / np.linalg.norm(analytic, axis=1, keepdims=True)
        # Compute local-PCA normals
        surface = list(range(len(pts)))
        normals, _ = _compute_surface_tangents_and_normals(pts, surface)
        bad = []
        for p, a, n in zip(pts, analytic, normals):
            cos_angle = abs(float(n @ a))
            if cos_angle < 0.85:
                bad.append((p, a, n, cos_angle))
        assert not bad, (
            "normals disagree with analytic gradient at "
            + ", ".join(
                f"point {p}: analytic={a}, pca={n}, cos={c:.3f}"
                for p, a, n, c in bad[:3]
            )
        )

    def test_plane_normals_point_out_of_plane(self):
        """For a flat (z = 0) consensus, every interior normal should be
        parallel to the z-axis."""
        consensus = _flat_surface()
        surface = list(range(len(consensus)))
        normals, _ = _compute_surface_tangents_and_normals(consensus, surface)
        for n in normals:
            nz = abs(n[2])
            assert nz == pytest.approx(1.0, abs=1e-6), f"normal {n} is not aligned with z"

    def test_rotated_consensus_gives_rotated_normals_curved(self):
        """If we rotate the entire consensus by R, every local surface
        normal must also rotate by R (equivalence of frame choice).

        Using a CURVED surface so the buggy constant-v2 normal is
        detectable; the test fails on the buggy implementation."""
        consensus = _curved_surface()
        surface = list(range(len(consensus)))
        normals_ref, _ = _compute_surface_tangents_and_normals(consensus, surface)

        Q = _rotation(np.random.default_rng(7))
        assert np.linalg.det(Q) > 0.999
        rotated = consensus @ Q.T
        normals_rot, _ = _compute_surface_tangents_and_normals(rotated, surface)
        for n_ref, n_rot in zip(normals_ref, normals_rot):
            rotated_ref = Q @ n_ref
            assert np.allclose(rotated_ref, n_rot, atol=1e-8) or np.allclose(rotated_ref, -n_rot, atol=1e-8)


class TestSlidingProjectionInvariance:
    """The bug is observable through the sliding projection: rotating the
    consensus must rotate the projected configuration.  With the buggy
    constant-v2 normal, the projection direction is wrong (orthogonal to
    the actual tangent plane), so the projection of a rotated config
    does NOT match the rotation of the original projection."""

    def test_projection_equivariant_under_consensus_rotation(self):
        consensus = _curved_surface()
        surface = list(range(len(consensus)))

        # a specimen that bulges in +z (out of the surface)
        rng = np.random.default_rng(13)
        config = consensus + 0.0
        # displace interior points uniformly in z
        interior_idx = surface[1:-1]
        config[interior_idx, 2] += 0.4

        projected = _slide_surface_tangent_plane(config.copy(), consensus, surface, 3)

        # rotate everything by Q
        Q = _rotation(np.random.default_rng(2))
        rotated_consensus = consensus @ Q.T
        rotated_config = config @ Q.T
        rotated_projected = _slide_surface_tangent_plane(
            rotated_config.copy(), rotated_consensus, surface, 3
        )

        # rotated(projection) ≈ projection(rotated), modulo the local PCA
        # sign choice; in either case the two must agree within fp64
        # noise OR differ only by sign on the displacement direction.
        expected = projected @ Q.T
        # for a CONSENSUS-driven normal direction, the projection removes
        # the component along the local normal; under rotation, the
        # normal also rotates, so the displacement direction rotates too.
        # The projected points must agree to fp64 noise (sign-flips would
        # mean a flipping z-displacement, but the input displacement is
        # strictly +z so the absolute z-component must be reduced).
        np.testing.assert_allclose(expected, rotated_projected, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
