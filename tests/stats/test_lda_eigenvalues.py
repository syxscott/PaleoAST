# =============================================================================
# FILE: tests/stats/test_lda_eigenvalues.py
# =============================================================================
"""
Tests for the LDA ``eigenvalues`` field semantics.

Background: ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis``
exposes ``explained_variance_ratio_`` but not raw eigenvalues. PaleoAST's
historical code stored the explained-variance-ratio under both the
``explained_variance_ratio`` field AND the ``eigenvalues`` field, which
silently misnamed a [0, 1] proportion as a canonical root. R's
``lda::lda`` returns eigenvalues on the variance scale (sum of squares
between / within), so the numbers a user gets from PaleoAST were off by
orders of magnitude and carried a different interpretation.

The fix:
  - Add a new field ``eigenvalue_proportions`` (the previous
    ``explained_variance_ratio`` content).
  - Keep ``eigenvalues`` as an alias of the same array for backward
    compatibility, but document explicitly in the docstring that the
    values are the explained-variance *proportions*, not canonical roots.
  - Best-effort: also compute and expose the real canonical roots
    (eigenvalues of S_W^{-1} S_B) under ``eigenvalues_canonical``.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pytest

# Skip the whole module if scikit-learn isn't available -- the
# production code raises ComputationError if sklearn is missing and we
# don't want to add a hard dependency just for this defect fix.
pytest.importorskip("sklearn", reason="scikit-learn required for LDA")

import numpy as np
from numpy.testing import assert_allclose

from stats.lda import LDAAnalyzer


class TestLDAFieldSemantics:
    """Verify the field semantics of LDAResult."""

    def test_eigenvalue_proportions_field_exists_and_in_unit_interval(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((30, 4))
        groups = [0] * 10 + [1] * 10 + [2] * 10
        analyzer = LDAAnalyzer()
        result = analyzer.analyze(data, groups, n_components=2)
        assert hasattr(result, "eigenvalue_proportions")
        assert np.all(result.eigenvalue_proportions >= 0)
        assert np.all(result.eigenvalue_proportions <= 1.0)
        # Sum of proportions across axes is <= 1
        assert np.sum(result.eigenvalue_proportions) <= 1.0 + 1e-6

    def test_eigenvalues_alias_matches_proportions(self):
        """``eigenvalues`` is now an alias of ``eigenvalue_proportions``."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((30, 4))
        groups = [0] * 10 + [1] * 10 + [2] * 10
        analyzer = LDAAnalyzer()
        result = analyzer.analyze(data, groups, n_components=2)
        # The two fields must hold identical numbers (alias).
        assert_allclose(result.eigenvalues, result.eigenvalue_proportions)

    def test_canonical_eigenvalues_exist(self):
        """The real canonical roots of S_W^{-1} S_B are also exposed."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal((30, 4))
        groups = [0] * 10 + [1] * 10 + [2] * 10
        analyzer = LDAAnalyzer()
        result = analyzer.analyze(data, groups, n_components=2)
        # Either ``eigenvalues_canonical`` exists, or -- on degenerate
        # data where S_W is singular -- the array exists but is empty.
        assert hasattr(result, "eigenvalues_canonical")
        if len(result.eigenvalues_canonical) > 0:
            # Canonical eigenvalues are non-negative
            assert np.all(result.eigenvalues_canonical >= -1e-10)

    def test_canonical_eigenvalues_for_strong_signal(self):
        """
        When groups are far apart, the canonical roots are large; when they
        overlap, the roots are small. We don't pin a specific number, just
        verify the roots are finite and well-defined.
        """
        rng = np.random.default_rng(3)
        # Group 0 at (-3, -3), Group 1 at (3, 3), Group 2 at (-3, 3) -- well separated
        centers = np.array([[-3, -3], [3, 3], [-3, 3]])
        data = np.vstack([c + 0.1 * rng.standard_normal((10, 2)) for c in centers])
        groups = [0] * 10 + [1] * 10 + [2] * 10
        analyzer = LDAAnalyzer()
        result = analyzer.analyze(data, groups, n_components=2)
        # Canonical roots should be substantially > 0 (strong separation)
        assert np.all(result.eigenvalues_canonical > 0.1), (
            f"Expected canonical roots > 0.1 for separated groups, "
            f"got {result.eigenvalues_canonical}"
        )
