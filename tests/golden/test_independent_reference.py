# =============================================================================
# FILE: tests/golden/test_independent_reference.py
# =============================================================================
"""
Independent-reference cross-checks for geometric morphometrics.

Why this file exists
--------------------
``tests/golden/test_real_datasets.py`` loads the real published datasets
(Berns & Adams 2010 hummingbirds; Serb et al. 2011 scallops) but every
assertion there is a *self-consistency* invariant -- "GPA output is centred",
"centroid sizes equal 1", "the rotation matrix is orthogonal", "running it
twice gives the same answer". Those properties hold for a wrong implementation
just as happily as for a right one.

The only tests that compare PaleoAST against an outside authority live in
``tests/cross_validation/``, which skips entirely when R/rpy2 is absent. So a
default ``pytest`` run never checks whether the *science* is right.

This file closes that gap without needing R: it contains a second, deliberately
naive implementation of generalized Procrustes analysis written straight from
Dryden & Mardia (2016, *Statistical Shape Analysis*, Ch. 4) and Bookstein
(1991), sharing no code with ``morphometrics/``.

What can and cannot be asserted against it
------------------------------------------
GPA is minimised over a non-convex problem and a single-start textbook
iteration is *not* guaranteed to find the global optimum. On the hummingbird
beaks the reference below converges to a genuine fixed point (identical at
200, 2 000 and 20 000 iterations) whose sum of squares is HIGHER than the
production result. So the honest claim is not "the two agree element-wise" --
it is "the production GPA reaches an optimum at least as good as an
independent textbook implementation, and satisfies the defining property of a
Procrustes fit". Those are the assertions below, and both are falsifiable.

Conventions (verified numerically before use):
  * A configuration is (n_landmarks, n_dims) float.
  * ``_best_fit_rotation(A, B)`` returns R with ``A @ R.T ~= B`` and det(R)=+1.
  * Procrustes distance is the distance from an optimally aligned
    configuration to the consensus (Dryden & Mardia 2016, Prop. 2.5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphometrics.gpa import GPAAnalyzer
from morphometrics.shape_stats import procrustes_distance

_GOLDEN = Path(__file__).resolve().parents[2] / "data" / "golden"

# Procrustes distances are unitless shape quantities of order 1, but the
# comparisons go through centring and scaling of coordinates of order 1e1-1e2,
# so a few 1e-8 of cancellation noise is expected. Far tighter than any real
# algorithmic error, which shows up at 1e-2 or worse.
_TOL = 1e-7


# =============================================================================
# Independent reference implementation (Dryden & Mardia 2016; Bookstein 1991)
# =============================================================================


def _center(config: np.ndarray) -> np.ndarray:
    """Translate a configuration so its centroid is the origin."""
    return config - config.mean(axis=0)


def _centroid_size(config: np.ndarray) -> float:
    """Centroid size CS = sqrt(sum ||x_i - centroid||^2) (Bookstein 1991)."""
    return float(np.sqrt(np.sum(_center(config) ** 2)))


def _scale(config: np.ndarray) -> np.ndarray:
    """Centre a configuration and divide by its centroid size (CS = 1)."""
    return _center(config) / _centroid_size(config)


def _best_fit_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal *proper* rotation R minimising ||source @ R.T - target||.

    Orthogonal Procrustes: maximise tr(R source' target) subject to R' R = I
    and det(R) = +1. From the SVD source' target = U S V', the solution is
    R = V D U' with the smallest singular value's sign flipped when the raw
    solution is a reflection. A reflection is *not* a rigid motion of a
    landmark configuration, so the det = +1 constraint is mandatory.
    """
    n_dims = source.shape[1]
    u, _singular, vt = np.linalg.svd(source.T @ target)
    handedness = float(np.sign(np.linalg.det(vt.T @ u.T)))
    return vt.T @ np.diag([1.0] * (n_dims - 1) + [handedness]) @ u.T


def _reference_gpa(
    configurations: np.ndarray, n_iter: int = 200, tol: float = 1e-14
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generalized Procrustes analysis, written out longhand, single start.

    Iterates: estimate each specimen's rotation onto the current consensus,
    re-align, re-centre, re-scale, re-average, until the summed Procrustes
    distances stop improving. Returns (consensus, distances, final_sse).
    """
    aligned = np.array([_scale(c) for c in configurations], dtype=float)
    consensus = _scale(aligned.mean(axis=0))
    previous_sse = np.inf

    for _ in range(n_iter):
        for i in range(len(aligned)):
            rotation = _best_fit_rotation(aligned[i], consensus)
            aligned[i] = _scale(aligned[i] @ rotation.T)

        consensus = _scale(aligned.mean(axis=0))
        distances = np.array([np.sqrt(np.sum((aligned[i] - consensus) ** 2)) for i in range(len(aligned))])
        final_sse = float(np.sum(distances**2))

        if previous_sse - final_sse <= tol:
            break
        previous_sse = final_sse

    return consensus, distances, final_sse


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def hummingbirds() -> np.ndarray:
    """Berns & Adams (2010) *Archilochus* beaks: 25 specimens x 44 points x 2D."""
    with np.load(_GOLDEN / "hummingbirds.npz") as data:
        return np.asarray(data["configurations"], dtype=float)


@pytest.fixture(scope="module")
def scallops() -> np.ndarray:
    """Serb et al. (2011) *Argopecten*: 5 specimens x 46 points x 3D."""
    with np.load(_GOLDEN / "scallops.npz") as data:
        return np.asarray(data["configurations"], dtype=float)


# =============================================================================
# 1. The production GPA is at least as well optimised as a from-scratch one
# =============================================================================


class TestAgainstIndependentGPA:
    @pytest.mark.parametrize("fixture_name", ["hummingbirds", "scallops"])
    def test_production_reaches_an_at_least_as_good_optimum(self, fixture_name, request):
        """Production's objective must not be worse than the reference's.

        This is the falsifiable half of the cross-check. An implementation
        with a sign error, a transposed eigenvector, or a bad scale factor
        produces a *higher* sum of squared Procrustes distances and fails
        here, even though it would sail through every self-consistency test.
        """
        data = request.getfixturevalue(fixture_name)
        result = GPAAnalyzer().analyze(data)
        _consensus, _distances, reference_sse = _reference_gpa(data)

        assert result.final_sse <= reference_sse + _TOL, (
            f"production SSE {result.final_sse:.10f} exceeds the independent "
            f"reference {reference_sse:.10f} -- it is stuck in a worse optimum"
        )

    @pytest.mark.parametrize("fixture_name", ["hummingbirds", "scallops"])
    def test_each_specimen_lies_on_the_consensus(self, fixture_name, request):
        """After alignment, the mean configuration *is* the consensus.

        So each specimen's residual against the mean equals its Procrustes
        distance. A transposed eigenvector, a sign error, or a bad scale factor
        all break this, and none of them are caught by a self-consistency test.
        """
        data = request.getfixturevalue(fixture_name)
        result = GPAAnalyzer().analyze(data)
        mean_shape = result.aligned_configurations.mean(axis=0)
        residuals = np.linalg.norm(result.aligned_configurations - mean_shape, axis=(1, 2))
        assert_allclose(residuals, result.procrustes_distances, atol=_TOL)

    @pytest.mark.parametrize("fixture_name", ["hummingbirds", "scallops"])
    def test_consensus_is_a_fixed_point(self, fixture_name, request):
        """The consensus must be the CS-normalised mean of the alignment.

        Re-aligning every specimen to this consensus and re-averaging has to
        reproduce it, otherwise the reported distances describe a shape the
        analysis does not actually sit at.

        The tolerance here is looser than ``_TOL`` because the production
        iteration stops at its own convergence threshold rather than running
        to machine precision. The observed residual is ~7e-07 on the
        hummingbirds (CS = 1 shape units), which is that threshold, not a
        failure to converge.
        """
        data = request.getfixturevalue(fixture_name)
        result = GPAAnalyzer().analyze(data)

        consensus = _scale(result.aligned_configurations.mean(axis=0))
        refit = np.array(
            [_scale(specimen @ _best_fit_rotation(specimen, consensus)) for specimen in result.aligned_configurations]
        )
        assert_allclose(_scale(refit.mean(axis=0)), consensus, atol=1e-5)


# =============================================================================
# 2. Procrustes distance has the invariances the method promises
# =============================================================================


class TestProcrustesInvariances:
    """A similarity-shape distance must be blind to how the object is posed.

    These are properties of the *definition* (Dryden & Mardia 2016), not of this
    implementation, and they catch sign or scaling mistakes that a
    self-consistency test cannot see.
    """

    @pytest.mark.parametrize("fixture_name", ["hummingbirds", "scallops"])
    def test_translation_invariance(self, fixture_name, request):
        """A rigid translation of every landmark must change nothing.

        Note the offset must be the SAME vector for every landmark. Adding a
        different offset per landmark is a deformation, not a translation, and
        correctly yields a non-zero distance.
        """
        data = request.getfixturevalue(fixture_name)
        base = data[0]
        shift = np.array([13.7, -4.2, 0.5][: base.shape[1]])
        assert procrustes_distance(base, base + shift) == pytest.approx(0.0, abs=_TOL)

    def test_rotation_invariance_2d(self, hummingbirds):
        base = hummingbirds[0]
        theta = 0.7341
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        assert procrustes_distance(base, base @ rotation.T) == pytest.approx(0.0, abs=_TOL)

    def test_rotation_invariance_3d(self, scallops):
        base = scallops[0]
        # A rotation with no eigenvalue 1, so it is genuinely not the identity.
        theta = 0.9123
        rotation = rodrigues([1.0, 2.0, 3.0], theta)
        assert procrustes_distance(base, base @ rotation.T) == pytest.approx(0.0, abs=_TOL)

    @pytest.mark.parametrize("factor", [0.5, 2.0, 137.0])
    def test_scaling_invariance(self, factor, scallops):
        base = scallops[0]
        assert procrustes_distance(base, base * factor) == pytest.approx(0.0, abs=_TOL)

    def test_per_landmark_perturbation_is_not_invariant(self):
        """The mirror image of that property: deformation must register.

        A uniform scale is invisible; a different offset at each landmark is a
        real shape change. If this returned zero the metric would be
        discarding the data.
        """
        generator = np.random.default_rng(20260929)
        base = generator.normal(size=(30, 2))
        deformation = generator.normal(scale=0.4, size=base.shape)
        assert procrustes_distance(base, base + deformation) > 1e-3

    def test_distance_matrix_is_a_symmetric_metric(self, hummingbirds):
        """Symmetry, zero diagonal, non-negativity, triangle inequality."""
        aligned = GPAAnalyzer().analyze(hummingbirds).aligned_configurations
        n = len(aligned)
        grid = np.array([[procrustes_distance(aligned[i], aligned[j]) for j in range(n)] for i in range(n)])
        assert_allclose(grid, grid.T, atol=_TOL)
        assert_allclose(np.diag(grid), 0.0, atol=_TOL)
        assert np.all(grid >= -_TOL), "distances must be non-negative"
        for i, j, k in [(0, 7, 19), (1, 2, 3), (4, 11, 23), (0, 5, 10)]:
            assert grid[i, k] <= grid[i, j] + grid[j, k] + _TOL


# =============================================================================
# 3. Analytic cases with hand-checkable answers
# =============================================================================


class TestAnalyticCases:
    def test_identical_configurations_are_zero(self, scallops):
        assert procrustes_distance(scallops[0], scallops[0]) == pytest.approx(0.0, abs=1e-12)

    def test_tiny_real_displacement_is_not_clamped_to_zero(self, scallops):
        """The zero-clamp must not swallow a genuine difference.

        ``procrustes_distance`` rounds ``d²`` to 0 when it is pure float
        noise, because on this fixture identical input leaves ``d²`` at
        4.4e-16 (the singular values of a unit-norm Gram matrix sum to 1 only
        to within rounding) and the *positive* residue survives a
        ``max(d2, 0.0)`` clamp as a spurious 1.49e-8 distance.

        That fix is only safe while the threshold stays far below any real
        displacement. Displacing one landmark coordinate by 1e-4 moves ``d²``
        to 2.7e-13 — three orders of magnitude above the noise floor — so it
        must still be reported as a small nonzero distance. A threshold set
        at 1e-12 (a natural-looking "plenty of headroom" choice) silently
        rounds this to 0.0, which is the failure this test exists to catch.
        """
        base = scallops[0]
        moved = base.copy()
        moved[0, 0] += 1e-4

        d = procrustes_distance(base, moved)

        assert d > 0.0, "a real 1e-4 landmark displacement was clamped to zero"
        # d is ~5.2e-7 for this displacement; a loose bound is deliberate so
        # the test pins the *sign* (nonzero), not the exact float value.
        assert d < 1e-5, f"unexpectedly large distance for a 1e-4 shift: {d}"

    def test_displacement_is_bounded_and_nonzero(self):
        """Moving one corner of a square is neither congruent nor degenerate.

        A crash, a NaN, or a value outside [0, sqrt(2) * max displacement]
        would all indicate a broken fit; a value of exactly 0 would indicate a
        fit that is ignoring the displacement entirely.
        """
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        moved = square.copy()
        moved[3] = [0.0, 1.5]
        d = procrustes_distance(square, moved)
        assert 0.0 < d < np.sqrt(2) * 1.5

    def test_reference_implementation_is_itself_sound(self, hummingbirds):
        """Guard the guard.

        Two configurations of the same specimen are congruent, so any correct
        GPA must place both at distance zero. If this fails, every comparison
        above is meaningless and the reference implementation is the bug.
        """
        duplicated = np.stack([hummingbirds[0], hummingbirds[0].copy()])
        _consensus, distances, sse = _reference_gpa(duplicated)
        assert_allclose(distances, 0.0, atol=1e-9)
        assert sse == pytest.approx(0.0, abs=1e-12)

    def test_reference_rotation_recovers_a_known_rotation(self):
        """The reference's rotation solver must be exact on a known answer."""
        generator = np.random.default_rng(11)
        source = generator.normal(size=(25, 3))
        truth = rodrigues([0.3, -0.7, 1.1], 1.234)
        target = source @ truth.T
        recovered = _best_fit_rotation(source, target)
        assert_allclose(recovered, truth, atol=1e-10)
        assert np.linalg.det(recovered) == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# Helpers
# =============================================================================


def rodrigues(axis: list[float] | np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix about ``axis`` by ``theta`` (Rodrigues' formula)."""
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    skew = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
