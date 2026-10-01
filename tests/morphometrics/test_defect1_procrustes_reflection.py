"""
Regression tests for defect 1: Procrustes distance reflection handling.

History of this defect, because it was "fixed" twice and was wrong both times:

  1. The original implementation multiplied the entire singular-value sum by
     ``np.sign(det(Vt.T @ U.T))``, so a reflection made the distance LARGER
     instead of smaller -- the opposite of a full Procrustes distance.
  2. The first repair flipped the smallest singular value whenever
     reflections were *allowed*. That is backwards: allowing reflections means
     taking the unconstrained optimum, which needs no flip. The flip is only
     needed to *force* a proper rotation. That version had the two branches
     swapped and passed a test that asserted the wrong number.
  3. The current implementation flips the smallest singular value only when
     ``no_reflect=True`` and det < 0.

The mirror pair below pins which branch is which. Both earlier versions fail
it, and so would any future one that gets the branches the wrong way round.

Ground truth for these numbers is established by exhaustive search over every
orthogonal transform in ``tests/golden/test_procrustes_ground_truth.py`` --
not by re-deriving the SVD identity, which is what made the wrong answers
look plausible in the first place.
"""

import numpy as np
import pytest

from morphometrics.shape_stats import procrustes_distance


def _mirror_pair() -> tuple[np.ndarray, np.ndarray]:
    """A right triangle and its reflection about the y-axis.

    Procrustes holds the landmark correspondence fixed -- landmark i of A must
    map to landmark i of B, never to a different landmark -- so this pair is
    reachable from itself by a reflection but by no proper rotation. That makes
    it the cleanest possible probe of the two branches.
    """
    a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    b = a.copy()
    b[:, 0] *= -1.0
    return a, b


class TestFullProcrustesReflection:
    def test_reflection_allowed_is_zero(self):
        """Default: reflections allowed, so the mirror is matched exactly.

        A'B has singular values (0.75, 0.25) and negative determinant. With no
        flip, sum = 1.0 and d^2 = 2 - 2*1.0 = 0.
        """
        a, b = _mirror_pair()
        d = procrustes_distance(a, b)
        assert d == pytest.approx(0.0, abs=1e-7)

    def test_proper_rotation_only_is_one(self):
        """no_reflect=True forbids the reflection, so the distance opens up.

        Flipping the smallest singular value gives sum = 0.75 - 0.25 = 0.5 and
        d^2 = 2 - 2*0.5 = 1.
        """
        a, b = _mirror_pair()
        d = procrustes_distance(a, b, no_reflect=True)
        assert d == pytest.approx(1.0, abs=1e-8)
        assert d * d == pytest.approx(1.0, abs=1e-8)

    def test_reflection_allowed_is_never_the_larger_one(self):
        """The reflection-allowed set is a superset, so its minimum is smaller.

        This is the property that pins the branch order without depending on
        any particular fixture, and it fails loudly if the two branches are
        ever swapped again.
        """
        a, b = _mirror_pair()
        assert procrustes_distance(a, b) <= procrustes_distance(a, b, no_reflect=True)

    def test_buggy_whole_sum_flip_would_give_two(self):
        """The original bug returned d = 2 for this pair; guard against it."""
        a, b = _mirror_pair()
        assert abs(procrustes_distance(a, b) - 2.0) > 0.1

    def test_branches_are_not_swapped(self):
        """Both earlier "fixes" had these two answers exchanged."""
        a, b = _mirror_pair()
        full, proper = procrustes_distance(a, b), procrustes_distance(a, b, no_reflect=True)
        assert full < 0.5, "full Procrustes must reach the mirror image"
        assert proper > 0.5, "a proper rotation cannot reach the mirror image"

    def test_distance_non_negative(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(6, 2))
        b = rng.normal(size=(6, 2))
        d = procrustes_distance(a, b)
        assert d >= 0.0
        assert np.isfinite(d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
