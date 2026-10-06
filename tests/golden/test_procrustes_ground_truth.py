# =============================================================================
# FILE: tests/golden/test_procrustes_ground_truth.py
# =============================================================================
"""
Brute-force ground truth for ``morphometrics.shape_stats.procrustes_distance``.

Why this file exists
--------------------
``procrustes_distance`` has now been "fixed" twice, and both fixes were wrong
in a way that a self-consistency test could not see:

  1. It multiplied the whole singular-value sum by ``sign(det)``, which made
     a reflected configuration *further* from its mirror image rather than
     closer -- the opposite of what a full Procrustes distance should do.
  2. The repair flipped the *smallest* singular value whenever reflections
     were **allowed**. That is backwards: allowing reflections means taking
     the unconstrained optimum, which needs no flip at all. The flip is only
     needed to *force* a proper rotation.

The second version passed a hand-written test that asserted the reflected
distance was 1.0, when an exhaustive search over every orthogonal transform
puts it at 0.0. Both numbers are "stable" and both look plausible, so no
amount of re-running the implementation will reveal which is right.

The only way to settle it is to stop using the SVD identity and minimise
||A M - B|| directly over the transform group, which is what this file does.

The 2-D case is settled **exhaustively** (a closed-form sweep over every
possible rotation angle, for each determinant class). The 3-D case is settled
by dense multi-start optimisation over axis-angle parameters, which reaches
the global optimum on these smooth low-dimensional problems.

Landmark correspondence is held fixed throughout: landmark *i* of A must map
to landmark *i* of B. Permuting landmarks is not part of the similarity group
and is never allowed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from morphometrics.shape_stats import procrustes_distance

_TOL = 1e-9


# =============================================================================
# Ground truth by direct minimisation over the transform group
# =============================================================================


def _normalise(a: np.ndarray) -> np.ndarray:
    """Centre a configuration and scale it to centroid size 1."""
    centred = a - a.mean(axis=0)
    return centred / np.sqrt(np.sum(centred**2))


def _residual(a: np.ndarray, b: np.ndarray, transform: np.ndarray) -> float:
    return float(np.sum((_normalise(a) @ transform - _normalise(b)) ** 2))


def _min_2d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Exact minima over proper and improper 2-D transforms.

    Every element of O(2) is either R(theta) or R(theta) @ diag(1, -1), so a
    sweep over theta covers the group exactly. Returns (proper, improper).
    """
    best = {1: np.inf, -1: np.inf}
    # 0.0001 rad resolution is ~7 orders of magnitude finer than the
    # tolerances used by the shapes this is checked against.
    for theta in np.linspace(0.0, 2.0 * np.pi, 100_001):
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for transform in (rotation, rotation @ np.diag([1.0, -1.0])):
            handedness = int(np.sign(np.linalg.det(transform)))
            value = _residual(a, b, transform)
            if value < best[handedness]:
                best[handedness] = value
    return float(best[1]), float(best[-1])


def _rotation_3d(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula for a 3-D rotation."""
    unit = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _min_3d(a: np.ndarray, b: np.ndarray, n_starts: int = 300) -> tuple[float, float]:
    """Minima over proper and improper 3-D transforms by multi-start search.

    Parameterised as (unit axis, angle) -> rotation, minimised by L-BFGS-B
    from many random starts. The improper class is reached by composing the
    proper optimum with a fixed reflection, so both classes are searched over
    the same parameterisation.
    """
    generator = np.random.default_rng(20260929)
    reflection = np.diag([1.0, 1.0, -1.0])
    best = {1: np.inf, -1: np.inf}

    for _ in range(n_starts):
        seed_axis = generator.normal(size=3)
        norm = np.linalg.norm(seed_axis)
        if norm < 1e-9:
            continue
        seed = np.concatenate([seed_axis / norm, [generator.uniform(0.0, 2.0 * np.pi)]])

        def objective(params, reflect: bool):
            rotation = _rotation_3d(params[:3], params[3])
            if reflect:
                rotation = rotation @ reflection
            return _residual(a, b, rotation)

        for reflect, handedness in ((False, 1), (True, -1)):
            result = minimize(objective, seed, args=(reflect,), method="L-BFGS-B")
            best[handedness] = min(float(result.fun), best[handedness])
    return float(best[1]), float(best[-1])


def _ground_truth(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Returns (proper_rotation_min, full_group_min) as SQUARED distances.

    The full (reflection-allowed) minimum is the smaller of the two
    determinant classes -- it is the minimum over a SUPERSET of the proper
    rotations, so it can never be larger. Note this is *not* the same as
    "the improper branch": when det(A'B) > 0 the unconstrained optimum is
    itself a proper rotation, and the improper branch is strictly worse.
    """
    proper, improper = _min_2d(a, b) if a.shape[1] == 2 else _min_3d(a, b)
    return proper, min(proper, improper)


# =============================================================================
# Test cases with configurations chosen to exercise each branch
# =============================================================================


def _right_triangle() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _mirrored(config: np.ndarray) -> np.ndarray:
    """Reflect across the first coordinate axis."""
    out = config.copy()
    out[:, 0] *= -1.0
    return out


def _asymmetric_2d() -> np.ndarray:
    """A shape with no symmetry, so reflection genuinely changes it."""
    return np.array([[0.0, 0.0], [2.3, 0.4], [3.1, 1.9], [1.2, 2.7], [-0.6, 1.5]])


def _asymmetric_3d() -> np.ndarray:
    generator = np.random.default_rng(5)
    points = generator.normal(size=(9, 3))
    # Break any accidental symmetry.
    points[:, 0] += np.arange(9) * 0.37
    return points


class TestTwoDimensional:
    def test_mirror_pair_needs_reflection_to_match(self):
        """The canonical case that both broken versions got wrong.

        The mirrored right triangle is reachable from the original by a
        reflection but not by any proper rotation, because Procrustes holds
        the landmark correspondence fixed (landmark i of A must map to
        landmark i of B -- permuting landmarks is not part of the similarity
        group). So the full distance must be ~0 and the proper-rotation
        distance must be clearly positive.
        """
        a = _right_triangle()
        b = _mirrored(a)
        proper, full = _ground_truth(a, b)

        assert full == pytest.approx(0.0, abs=1e-6)
        assert proper > 0.5, "a proper rotation cannot reach the mirror image"

        assert procrustes_distance(a, b, no_reflect=False) == pytest.approx(np.sqrt(full), abs=1e-6)
        assert procrustes_distance(a, b, no_reflect=True) == pytest.approx(np.sqrt(proper), abs=1e-6)

    def test_default_is_the_reflection_allowed_minimum(self):
        """The default must never exceed the proper-rotation-only distance.

        The reflection-allowed set is a superset, so its minimum can only be
        smaller or equal. Any implementation that violates this has the two
        branches swapped.
        """
        generator = np.random.default_rng(11)
        for _ in range(12):
            a = generator.normal(size=(7, 2))
            b = generator.normal(size=(7, 2))
            full = procrustes_distance(a, b, no_reflect=False)
            proper = procrustes_distance(a, b, no_reflect=True)
            assert full <= proper + _TOL

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
    def test_matches_exhaustive_search(self, seed):
        generator = np.random.default_rng(seed)
        a = generator.normal(size=(6, 2))
        b = generator.normal(size=(6, 2))
        proper, full = _ground_truth(a, b)

        assert procrustes_distance(a, b, no_reflect=True) == pytest.approx(np.sqrt(proper), abs=1e-6)
        assert procrustes_distance(a, b, no_reflect=False) == pytest.approx(np.sqrt(full), abs=1e-6)

    def test_asymmetric_shape_mirror(self):
        a = _asymmetric_2d()
        b = _mirrored(a)
        proper, full = _ground_truth(a, b)
        assert procrustes_distance(a, b, no_reflect=False) == pytest.approx(np.sqrt(full), abs=1e-6)
        assert procrustes_distance(a, b, no_reflect=True) == pytest.approx(np.sqrt(proper), abs=1e-6)


class TestThreeDimensional:
    def test_matches_multistart_optimisation(self):
        a = _asymmetric_3d()
        b = a[::-1].copy() + 0.4  # reordering + shift: a genuinely different fit
        proper, full = _ground_truth(a, b)

        assert procrustes_distance(a, b, no_reflect=True) == pytest.approx(np.sqrt(proper), abs=1e-4)
        assert procrustes_distance(a, b, no_reflect=False) == pytest.approx(np.sqrt(full), abs=1e-4)

    def test_default_is_never_larger_than_proper_only(self):
        generator = np.random.default_rng(3)
        for _ in range(4):
            a = generator.normal(size=(6, 3))
            b = generator.normal(size=(6, 3))
            assert procrustes_distance(a, b, no_reflect=False) <= (procrustes_distance(a, b, no_reflect=True) + _TOL)


class TestInvariancesUnaffectedByTheBranchChoice:
    """Reflections aside, the distance must still be pose-invariant."""

    @pytest.mark.parametrize("no_reflect", [True, False])
    def test_identical_is_zero(self, no_reflect):
        a = _asymmetric_2d()
        assert procrustes_distance(a, a, no_reflect=no_reflect) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("no_reflect", [True, False])
    def test_similarity_transform_is_zero(self, no_reflect):
        a = _asymmetric_2d()
        theta = 0.6
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        b = 3.7 * (a @ rotation.T) + np.array([12.0, -5.0])
        assert procrustes_distance(a, b, no_reflect=no_reflect) == pytest.approx(0.0, abs=1e-9)
