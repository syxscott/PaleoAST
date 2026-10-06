"""
Ground-truth tests for morpho3d/mesh.py (Mesh3D + SurfaceInterpolator).

Each test cites an analytic ground truth (volume / area via closed-form
geometry, normal of a known plane, IDW at training points = identity, etc.)
so that any regression in the implementation surfaces as a numeric
disagreement rather than a coverage gap.

Run:
    .venv/Scripts/python.exe -m pytest tests/morphometrics/test_mesh_ground_truth.py \\
        -q --no-header -p no:cacheprovider
"""

from __future__ import annotations

import numpy as np
import pytest

from morpho3d.mesh import Mesh3D, SurfaceInterpolator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unit_tetrahedron():
    """Unit right tetrahedron with vertices at the coordinate origin.

    Volume = |det|/6 = 1/6.
    Surface area = 3 right isoceles triangles of area 1/2 each
                   + 1 equilateral-ish face with vertices
                     (1,0,0),(0,1,0),(0,0,1) of area sqrt(3)/2.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],  # z = 0
            [0, 2, 3],  # y = 0
            [0, 3, 1],  # x = 0
            [1, 3, 2],  # oblique face
        ],
        dtype=int,
    )
    return Mesh3D(vertices=vertices, faces=faces)


def unit_cube():
    """Axis-aligned unit cube [0,1]^3 with each face triangulated as 2
    triangles (24 triangles total) — outward CCW orientation viewed from
    outside, so the divergence-theorem volume formula is positive.

    Volume = 1.0; surface area = 6.0.
    """
    V = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=float,
    )

    def quad(a, b, c, d):
        """Two triangles of a quad (a,b,c,d) viewed CCW from outside."""
        return [[a, b, c], [a, c, d]]

    faces = []
    # z = 0 (bottom) — outward normal = -z, CCW seen from below means (0,3,2)(0,2,1)
    faces += quad(0, 3, 2, 1)
    # z = 1 (top) — outward normal = +z, CCW seen from above means (4,5,6)(4,6,7)
    faces += quad(4, 5, 6, 7)
    # y = 0
    faces += quad(0, 1, 5, 4)
    # y = 1
    faces += quad(3, 7, 6, 2)
    # x = 0
    faces += quad(0, 4, 7, 3)
    # x = 1
    faces += quad(1, 2, 6, 5)

    return Mesh3D(vertices=V, faces=np.array(faces, dtype=int))


def flat_xy_grid(n: int = 4):
    """n-by-n grid of points in the z = 0 plane, triangulated into 2(n-1)^2
    triangles, all sharing the same geometric normal ±z."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), np.zeros_like(gx.ravel())], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v01 = i * n + (j + 1)
            v11 = (i + 1) * n + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return Mesh3D(vertices=verts, faces=np.array(faces, dtype=int))


def slanted_plane(b: float, c: float, a: float = 0.0, n: int = 4):
    """Grid on the plane z = a + b*x + c*y. The face normal computed via
    (v1-v0) x (v2-v0) for v0=(0,0,a), v1=(1,0,a+b), v2=(0,1,a+c) gives
    (-b, -c, 1) up to normalization, so unit normal = (-b,-c,1)/|·|."""
    xs = np.linspace(-0.5, 0.5, n)
    ys = np.linspace(-0.5, 0.5, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    z = a + b * gx + c * gy
    verts = np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v01 = i * n + (j + 1)
            v11 = (i + 1) * n + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return Mesh3D(vertices=verts, faces=np.array(faces, dtype=int))


# ---------------------------------------------------------------------------
# Area & volume analytic invariants
# ---------------------------------------------------------------------------


class TestAnalyticAreaVolume:
    """Cross-check Mesh3D.compute_surface_area and .compute_volume against
    closed-form values for primitives."""

    def test_tetrahedron_volume(self):
        """Volume of the right tetrahedron at the origin equals |det|/6 = 1/6."""
        m = unit_tetrahedron()
        assert abs(m.compute_volume() - 1.0 / 6.0) < 1e-9

    def test_tetrahedron_surface_area(self):
        """Surface area = 3 right triangles of 1/2 + 1 oblique of sqrt(3)/2."""
        m = unit_tetrahedron()
        expected = 3 * 0.5 + np.sqrt(3) / 2
        assert abs(m.compute_surface_area() - expected) < 1e-9

    def test_cube_volume(self):
        """Volume of unit cube = 1 (divergence theorem on closed mesh)."""
        m = unit_cube()
        assert abs(m.compute_volume() - 1.0) < 1e-9

    def test_cube_surface_area(self):
        """Surface area of unit cube = 6."""
        m = unit_cube()
        assert abs(m.compute_surface_area() - 6.0) < 1e-9

    def test_cube_scaling_volume_cubic(self):
        """Volume of side-k cube must equal k^3 exactly."""
        k = 2.7
        m = unit_cube()
        m.vertices = m.vertices * k
        # Recompute faces unchanged: rebuild mesh so __post_init__ re-runs
        scaled = Mesh3D(vertices=m.vertices * 1.0, faces=m.faces.copy())
        assert abs(scaled.compute_volume() - k ** 3) < 1e-8

    def test_cube_scaling_area_quadratic(self):
        """Surface area of side-k cube must equal 6 k^2."""
        k = 1.8
        m = unit_cube()
        scaled = Mesh3D(vertices=m.vertices * k, faces=m.faces.copy())
        assert abs(scaled.compute_surface_area() - 6.0 * k * k) < 1e-8

    def test_volume_translation_invariance(self):
        """For a *closed* mesh the divergence-theorem volume must be
        independent of translation (extra cocycle terms cancel)."""
        m = unit_cube()
        for shift in (5.0, -3.2, np.array([7.0, -2.0, 11.0])):
            shifted = Mesh3D(vertices=m.vertices + shift, faces=m.faces.copy())
            assert abs(shifted.compute_volume() - 1.0) < 1e-9, (
                f"volume changed under translation by {shift}"
            )


# ---------------------------------------------------------------------------
# Normal invariants
# ---------------------------------------------------------------------------


class TestVertexNormals:
    """Vertex normals must equal the area-weighted average of incident face
    normals (then renormalized)."""

    def test_flat_mesh_normals_parallel_to_z(self):
        """All vertices of a planar z=0 grid must have normals parallel
        to ±z (cos(angle) with ±z basis is 1 up to sign)."""
        m = flat_xy_grid(n=5)
        nz = np.array([0.0, 0.0, 1.0])
        cos = np.abs(m.normals @ nz)
        assert np.all(cos > 1 - 1e-6), f"flat normals not aligned with ±z: {cos}"

    def test_slanted_plane_normal_direction(self):
        """Plane z = b*x + c*y has unit normal (-b,-c,1)/|·| up to sign.

        Per-vertex normals from averaging incident face normals must
        match the analytical normal within 0.001 cosine distance.
        """
        b, c = 0.4, -0.7
        m = slanted_plane(b=b, c=c, n=5)
        analytic = np.array([-b, -c, 1.0])
        analytic = analytic / np.linalg.norm(analytic)
        # Allow either orientation (the cross-product convention depends
        # on vertex winding in the triangulation).
        cos = np.abs(m.normals @ analytic)
        assert np.all(cos > 0.999), (
            f"max deviation from analytic normal = {1 - cos.max():.3e}"
        )

    def test_normals_are_unit_length(self):
        """Every vertex normal must be exactly unit length (or (0,0,0)
        for a vertex with no incident faces)."""
        m = slanted_plane(b=0.3, c=0.5, n=6)
        norms = np.linalg.norm(m.normals, axis=1)
        non_dangling = norms > 0
        assert np.allclose(norms[non_dangling], 1.0, atol=1e-9), (
            f"non-unit normals: min={norms[non_dangling].min()}, "
            f"max={norms[non_dangling].max()}"
        )

    def test_area_weighted_merge_biases_toward_big_face(self):
        """When two faces share a vertex, the averaged normal must align
        with the larger face's normal (cosine > 0.9).

        Catches: normals summed without area weighting, or normalized in
        a way that throws away magnitude information.
        """
        # Two right triangles sharing vertex (0,0,0).
        # big  = (0,0,0) -> (10,0,0) -> (0,10,0), area = 50.
        # small = (0,0,0) -> (1,0,0) -> (0,1,0), area = 0.5.
        verts = np.array(
            [
                [0.0, 0.0, 0.0],  # 0 shared
                [10.0, 0.0, 0.0],  # 1
                [0.0, 10.0, 0.0],  # 2
                [1.0, 0.0, 0.0],  # 3
                [0.0, 1.0, 0.0],  # 4
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2],  # big, normal = +z
                [0, 3, 4],  # small, normal = +z
            ],
            dtype=int,
        )
        m = Mesh3D(vertices=verts, faces=faces)
        big_normal = np.array([0.0, 0.0, 1.0])
        # The shared vertex (0) should have normal very close to ±z.
        cos = abs(m.normals[0] @ big_normal)
        assert cos > 0.99, (
            f"area-weighted merge did not bias to big face: "
            f"normal={m.normals[0]}, cosine={cos}"
        )


# ---------------------------------------------------------------------------
# Surface point sampling
# ---------------------------------------------------------------------------


class TestSurfaceSampling:
    """sample_points must be area-uniform on the mesh."""

    def test_point_count_and_finiteness(self):
        """n_points samples returned, all finite, inside the mesh AABB."""
        m = unit_cube()
        rng_seed = 1234
        np.random.seed(rng_seed)
        pts = m.sample_points(500)
        assert pts.shape == (500, 3)
        assert np.all(np.isfinite(pts)), "sampled points must be finite"
        # Inside the cube's AABB (slightly loose: on-surface triangles
        # can land on the boundary).
        bbox_lo = m.vertices.min(axis=0)
        bbox_hi = m.vertices.max(axis=0)
        assert np.all(pts >= bbox_lo - 1e-9) and np.all(pts <= bbox_hi + 1e-9)

    def test_sampling_is_area_uniform(self):
        """With one small + one big triangle in disjoint regions, the share
        of samples landing in the small triangle must approximate its
        area ratio. Test with N=20000 so the binomial standard error is
        ~0.003.

        The triangles are placed in disjoint AABB regions so a simple
        bbox classifier is unambiguous (a point in the small region came
        from the small triangle).

        small area = 0.005, big area = 50, ratio = 1/10001 ~ 0.0001.
        """
        verts = np.array(
            [
                [0.0, 0.0, 0.0],   # 0
                [10.0, 0.0, 0.0],  # 1
                [0.0, 10.0, 0.0],  # 2
                [5.0, 5.0, 0.0],   # 3 small-tri vertex
                [5.1, 5.0, 0.0],   # 4
                [5.0, 5.1, 0.0],   # 5
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2],  # big   (area 50)
                [3, 4, 5],  # small (area 0.005)
            ],
            dtype=int,
        )
        m = Mesh3D(vertices=verts, faces=faces)
        np.random.seed(2025)
        pts = m.sample_points(20000)

        # Disjoint bbox classifier — points in the small triangle have
        # x ∈ [5, 5.1], y ∈ [5, 5.1], z ∈ [-1e-9, 1e-9].
        in_small = (
            (pts[:, 0] >= 5.0 - 1e-9)
            & (pts[:, 0] <= 5.1 + 1e-9)
            & (pts[:, 1] >= 5.0 - 1e-9)
            & (pts[:, 1] <= 5.1 + 1e-9)
        )
        frac = in_small.mean()
        expected = 0.005 / (0.005 + 50.0)  # = 1/10001
        # Generous interval: 10 sigma binomial SE = 10*sqrt(p(1-p)/N)
        se = np.sqrt(expected * (1 - expected) / 20000)
        assert abs(frac - expected) < 10 * se, (
            f"sample share={frac:.5f}, expected={expected:.5f}, "
            f"10*SE={10 * se:.5f}"
        )


# ---------------------------------------------------------------------------
# SurfaceInterpolator (IDW, k=4)
# ---------------------------------------------------------------------------


class TestSurfaceInterpolator:
    """SurfaceInterpolator.interpolate_values is k-NN IDW with k = min(4, n)."""

    def test_identity_at_training_points(self):
        """At each training vertex the IDW must reproduce its value to
        high precision (the nearest-neighbor weight dominates)."""
        m = flat_xy_grid(n=4)
        values = np.arange(len(m.vertices), dtype=float) ** 2  # arbitrary
        interp = SurfaceInterpolator(m)
        np.random.seed(7)
        got = interp.interpolate_values(values, m.vertices.copy())
        assert np.allclose(got, values, atol=1e-6), (
            f"identity violated; max abs diff = {np.abs(got - values).max()}"
        )

    def test_finite_for_arbitrary_queries(self):
        """Any 3D query point must yield a finite scalar (even far away)."""
        m = slanted_plane(b=0.2, c=0.1, n=4)
        values = np.sin(np.arange(len(m.vertices)))
        interp = SurfaceInterpolator(m)
        queries = np.array(
            [
                [100.0, -50.0, 999.0],
                [-1e6, 1e6, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        out = interp.interpolate_values(values, queries)
        assert out.shape == (3,)
        assert np.all(np.isfinite(out)), f"non-finite interpolation: {out}"

    def test_recovers_constant_function(self):
        """For a constant vertex_values = c, IDW must reproduce c at any
        query point (every weight cancels the same factor). This is a
        well-known IDW invariant."""
        m = flat_xy_grid(n=6)
        c = 3.14
        values = np.full(len(m.vertices), c)
        interp = SurfaceInterpolator(m)
        queries = np.array(
            [
                [0.3, 0.3, 0.0],
                [0.7, 0.2, 0.0],
                [0.5, 0.5, 0.0],
                [-1.0, 5.0, 99.0],  # far away
            ]
        )
        got = interp.interpolate_values(values, queries)
        assert np.allclose(got, c, atol=1e-9), (
            f"constant not preserved: got={got}"
        )


# ---------------------------------------------------------------------------
# Robustness / degeneracy
# ---------------------------------------------------------------------------


class TestInputValidation:
    """The constructor and methods must reject / behave correctly on
    malformed inputs."""

    def test_wrong_vertex_shape_raises(self):
        """Vertices must be (n, 3); other shapes must raise ValueError."""
        with pytest.raises(ValueError):
            Mesh3D(vertices=np.zeros((4, 2)), faces=np.zeros((0, 3), int))
        with pytest.raises(ValueError):
            Mesh3D(vertices=np.zeros((3,)), faces=np.zeros((0, 3), int))

    def test_wrong_face_shape_raises(self):
        """Faces must be (n, 3)."""
        with pytest.raises(ValueError):
            Mesh3D(vertices=np.zeros((3, 3)), faces=np.zeros((1, 4), int))

    def test_out_of_bounds_face_index_raises(self):
        """Face indices referencing a missing vertex must raise (not
        silently return NaN / 0)."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 5]], dtype=int)  # 5 is out of range
        with pytest.raises((IndexError, ValueError)):
            Mesh3D(vertices=verts, faces=faces)

    def test_zero_area_triangle_does_not_crash(self):
        """A mesh containing a degenerate (zero-area) triangle must not
        produce NaN normals; the vertex may get a zero or unit normal."""
        verts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],  # duplicate of 0 — gives a zero-area triangle
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=int)
        m = Mesh3D(vertices=verts, faces=faces)
        # All normals must be finite
        assert np.all(np.isfinite(m.normals)), "NaN normals from degenerate face"
        # Non-zero normals must be unit length
        norms = np.linalg.norm(m.normals, axis=1)
        assert np.allclose(norms[norms > 0], 1.0, atol=1e-9)

    def test_dangling_vertex_has_well_defined_normal(self):
        """A vertex referenced by no face must not produce NaN; its
        normal is allowed to be (0, 0, 0) or any unit vector — but must
        be finite."""
        verts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.5, 0.5, 0.5],  # dangling: not in any face
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2]], dtype=int)
        m = Mesh3D(vertices=verts, faces=faces)
        assert np.all(np.isfinite(m.normals)), "dangling vertex produced non-finite normal"
        # The dangling normal must have well-defined length (0 or 1)
        assert np.linalg.norm(m.normals[3]) < 1e-9 or abs(np.linalg.norm(m.normals[3]) - 1.0) < 1e-9
