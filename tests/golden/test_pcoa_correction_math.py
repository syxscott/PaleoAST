# =============================================================================
# FILE: tests/golden/test_pcoa_correction_math.py
# =============================================================================
"""
Ground truth for the negative-eigenvalue corrections in ``stats.pcoa``.

The correction machinery was implemented twice before it was correct, and both
times it passed its own tests:

  1. The UI offered four correction radio buttons and the caller silently
     dropped the choice, so the user got cmdscale whatever they picked.
  2. The parameter was then wired all the way through -- and the correction
     still did nothing, because it added the constant to *every* entry of the
     squared-distance matrix. Adding a constant everywhere is an exact no-op:
     with B = -1/2 J D^2 J and J 1 = 0,

         -1/2 J (D^2 + c*11') J = B - (c/2) J 11' J = B

     since J 11' J = (J1)(1'J) = 0. The log line claimed the shift was being
     applied; the eigenvalues were bit-for-bit unchanged.

So "are there negative eigenvalues left?" is NOT a sufficient test -- cmdscale
truncates them and also reports none. These tests instead pin the *shift
itself*, which is the only thing that distinguishes a real correction from a
no-op.

Mathematics under test
----------------------
Replacing D^2 by D^2 + c(11' - I) (off-diagonals only) gives

    B' = -1/2 J (D^2 + c(11' - I)) J = B + (c/2) J

Every eigenvector of B is orthogonal to the all-ones vector, so J acts as the
identity on it and each non-trivial eigenvalue moves by exactly **+c/2**.
With c = 2|lambda_min| the most negative eigenvalue lands precisely on zero,
which is the Lingoes (1971) correction.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist, squareform

from stats.pcoa import PCoAAnalyzer

_CORRECTING = ("lingoes", "wickoff", "torgerson")
_TOL = 1e-10


def _bray_curtis_case() -> np.ndarray:
    """A distance matrix with genuinely negative eigenvalues."""
    generator = np.random.default_rng(0)
    counts = generator.random((14, 10))
    counts[counts < 0.5] = 0.0
    return squareform(pdist(counts, "braycurtis"))


def _raw_eigenvalues(distance_matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of B = -1/2 J D^2 J, uncorrected, descending."""
    n = len(distance_matrix)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (distance_matrix**2) @ j
    return np.linalg.eigvalsh(b)[::-1]


def _corrected_spectrum(raw: np.ndarray, shift: float) -> np.ndarray:
    """The full corrected spectrum of B, descending.

    Every non-trivial eigenvalue moves by +shift (c/2). The trivial direction
    does not move: B' 1 = B 1 + (c/2) J 1 = 0, because J 1 = 0.
    """
    corrected = raw + shift
    corrected[int(np.argmin(np.abs(raw)))] = 0.0
    return np.sort(corrected)[::-1]


class TestCorrectionActuallyShifts:
    def test_correction_is_not_a_no_op(self):
        """A real correction must move the leading eigenvalues.

        This is the assertion that the previous implementation failed: it
        added the constant everywhere, so B -- and therefore every reported
        eigenvalue -- came out bit-identical to the uncorrected run.
        """
        distance_matrix = _bray_curtis_case()
        raw = _raw_eigenvalues(distance_matrix)
        assert (raw < 0).any(), "this fixture must have negative eigenvalues to be meaningful"

        baseline = np.asarray(PCoAAnalyzer().analyze(distance_matrix, 3, correction="cmdscale").eigenvalues)
        corrected = np.asarray(PCoAAnalyzer().analyze(distance_matrix, 3, correction="lingoes").eigenvalues)
        assert not np.allclose(baseline, corrected), (
            "lingoes produced the same eigenvalues as cmdscale -- the shift is a no-op"
        )

    @pytest.mark.parametrize("method", _CORRECTING)
    def test_shift_equals_c_over_two(self, method):
        """Each corrected eigenvalue must equal lambda + c/2 exactly."""
        distance_matrix = _bray_curtis_case()
        n = len(distance_matrix)
        raw = _raw_eigenvalues(distance_matrix)
        shift = abs(raw.min())  # c = 2|lambda_min|  =>  c/2 = |lambda_min|
        expected = _corrected_spectrum(raw, shift)[: n - 1]

        actual = np.asarray(PCoAAnalyzer().analyze(distance_matrix, n - 1, correction=method).eigenvalues)
        assert_allclose(actual, expected, atol=1e-9)

    @pytest.mark.parametrize("method", _CORRECTING)
    def test_smallest_eigenvalue_lands_exactly_on_zero(self, method):
        """The defining property: c = 2|lambda_min| puts the minimum at 0."""
        distance_matrix = _bray_curtis_case()
        eigenvalues = np.asarray(
            PCoAAnalyzer().analyze(distance_matrix, len(distance_matrix) - 1, correction=method).eigenvalues
        )
        assert eigenvalues.min() == pytest.approx(0.0, abs=1e-9)
        assert (eigenvalues >= -1e-12).all()


class TestCmdscaleRemainsTheDefault:
    @pytest.mark.parametrize("method", ["none", "cmdscale"])
    def test_default_applies_no_shift(self, method):
        """cmdscale must report the raw spectrum, negatives included.

        R's ``cmdscale`` leaves the negative eigenvalues in the reported
        values and excludes them only from the *coordinates* via
        ``sqrt(max(lambda, 0))``. Truncating them here would hide exactly the
        information the warning is telling the user about.
        """
        distance_matrix = _bray_curtis_case()
        n = len(distance_matrix)
        raw = _raw_eigenvalues(distance_matrix)
        actual = np.asarray(PCoAAnalyzer().analyze(distance_matrix, n - 1, correction=method).eigenvalues)
        assert_allclose(actual, raw[: n - 1], atol=1e-12)

    def test_omitting_the_argument_equals_cmdscale(self):
        """Backwards compatibility: the default must not change any number."""
        distance_matrix = _bray_curtis_case()
        implicit = PCoAAnalyzer().analyze(distance_matrix, 4)
        explicit = PCoAAnalyzer().analyze(distance_matrix, 4, correction="cmdscale")
        assert_allclose(implicit.coordinates, explicit.coordinates, atol=0.0)
        assert_allclose(implicit.eigenvalues, explicit.eigenvalues, atol=0.0)

    def test_positive_eigenvalue_case_is_untouched(self):
        """A Euclidean matrix needs no correction, so all methods agree."""
        generator = np.random.default_rng(7)
        points = generator.normal(size=(20, 4))
        euclidean = squareform(pdist(points))
        raw = _raw_eigenvalues(euclidean)
        assert (raw > -1e-9).all(), "Euclidean distances must be positive semi-definite"

        baseline = PCoAAnalyzer().analyze(euclidean, 3, correction="cmdscale").coordinates
        for method in _CORRECTING:
            corrected = PCoAAnalyzer().analyze(euclidean, 3, correction=method).coordinates
            assert_allclose(baseline, corrected, atol=1e-10)


class TestCorrectionDoesNotDistortOrdering:
    @pytest.mark.parametrize("method", _CORRECTING)
    def test_first_axis_keeps_the_most_variance(self, method):
        """A uniform shift preserves ordering, so axis 1 stays the largest."""
        distance_matrix = _bray_curtis_case()
        eigenvalues = np.asarray(
            PCoAAnalyzer().analyze(distance_matrix, len(distance_matrix) - 1, correction=method).eigenvalues
        )
        positive = eigenvalues[eigenvalues > 0]
        assert np.all(np.diff(positive) <= 1e-12), "eigenvalues must stay sorted descending"
