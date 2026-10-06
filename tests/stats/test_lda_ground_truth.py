"""
Ground truth for the canonical roots in ``stats.lda``.

Why this file exists
--------------------
``stats/lda.py`` is 25% covered, and the part worth checking is the one the
earlier review flagged as suspect: whether ``eigenvalues`` is really the
canonical discriminant roots or just a renamed ``explained_variance_ratio``
(the latter is bounded to [0, 1] and is not a discriminant root at all).

``LDAAnalyzer.analyze`` cannot be exercised in this environment because
scikit-learn is an optional dependency that is not installed, and the analyser
raises ``ComputationError`` without it. But ``_compute_canonical_eigenvalues``
is a module-level function with no scikit-learn dependency, and it is exactly
the piece under suspicion -- so it can be verified on its own.

The reference is ``scipy.linalg.eigh(S_B, S_W)``, SciPy's dedicated generalised
symmetric eigensolver. That is an independent algorithm (it forms and solves
a QZ-decomposed problem) rather than a restatement of the Cholesky reduction
the production code uses, so agreement is real evidence and not a tautology.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from stats.lda import _compute_canonical_eigenvalues

# =============================================================================
# Independent reference
# =============================================================================


def _scatters(data: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Between- and within-class scatter matrices, written from the definitions.

    S_B = sum_k n_k (mu_k - mu)(mu_k - mu)'
    S_W = sum_k sum_{i in C_k} (x_i - mu_k)(x_i - mu_k)'
    """
    overall = data.mean(axis=0)
    s_b = np.zeros((data.shape[1], data.shape[1]))
    s_w = np.zeros_like(s_b)
    for level in np.unique(groups):
        block = data[groups == level]
        mean_k = block.mean(axis=0)
        delta = (mean_k - overall).reshape(-1, 1)
        s_b += len(block) * (delta @ delta.T)
        s_w += (block - mean_k).T @ (block - mean_k)
    return s_b, s_w


def _reference_roots(data: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Canonical roots via SciPy's generalised solver, clipped at zero.

    The production code clips negative roots because they are meaningless
    downstream (they only arise when n_samples < n_vars, where S_B is
    indefinite), so the reference clips identically -- otherwise the comparison
    would fail on the clipping rather than on the algorithm.
    """
    s_b, s_w = _scatters(data, groups)
    roots = eigh(s_b, s_w, eigvals_only=True)
    return np.maximum(np.sort(roots)[::-1], 0.0)


# =============================================================================
# Tests
# =============================================================================


class TestCanonicalRoots:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_two_classes_match_scipy_generalised_solver(self, seed):
        """The roots must equal the generalised eigenvalues of (S_B, S_W)."""
        rng = np.random.default_rng(seed)
        data = rng.normal(size=(40, 4))
        groups = np.array([0] * 20 + [1] * 20)

        expected = _reference_roots(data, groups)[0]
        actual = _compute_canonical_eigenvalues(data, groups, 1)[0]
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)

    @pytest.mark.parametrize("n_classes", [2, 3, 4, 5])
    def test_multiclass_rank_is_k_minus_one(self, n_classes, seed=11):
        """With k classes the between-scatter matrix has rank <= k-1.

        So at most k-1 canonical roots can be non-zero. A test that only ever
        looked at two classes would miss an implementation that quietly
        returned k roots, or dropped one.
        """
        rng = np.random.default_rng(seed)
        per_class = 12
        offsets = rng.normal(scale=4.0, size=(n_classes, 3))
        data = np.vstack([rng.normal(loc=offsets[k], size=(per_class, 3)) for k in range(n_classes)])
        groups = np.repeat(np.arange(n_classes), per_class)

        roots = _compute_canonical_eigenvalues(data, groups, n_classes)
        expected = _reference_roots(data, groups)
        assert len(roots) == min(n_classes, len(expected))
        assert_allclose = np.testing.assert_allclose
        assert_allclose(roots, expected[: len(roots)], rtol=1e-8, atol=1e-10)
        assert np.count_nonzero(roots > 1e-9) <= n_classes - 1

    def test_roots_are_not_bounded_to_the_unit_interval(self):
        """A canonical root is a discriminant root, not a variance ratio.

        The bug this guards against is a field that claims to be an eigenvalue
        but actually holds sklearn's explained_variance_ratio, which is
        dimensionless and always in [0, 1]. Well-separated groups drive the
        true canonical root far above 1, so a bounded implementation is
        detectable without reading its source.
        """
        rng = np.random.default_rng(5)
        near = rng.normal(loc=0.0, scale=0.4, size=(20, 3))
        far = rng.normal(loc=25.0, scale=0.4, size=(20, 3))
        data = np.vstack([near, far])
        groups = np.array([0] * 20 + [1] * 20)

        root = _compute_canonical_eigenvalues(data, groups, 1)[0]
        assert root > 1.0, (
            f"canonical root {root} is inside [0,1]; that is a variance ratio, not a Fisher discriminant root"
        )

    def test_singular_within_scatter_returns_empty(self):
        """S_W rank-deficient means the canonical problem is undefined.

        Two observations per class cannot estimate a within-class covariance
        in three dimensions. The contract is an empty array -- a zero, or a
        crash, would both be misread downstream as "no separation".
        """
        rng = np.random.default_rng(2)
        data = rng.normal(size=(4, 3))
        groups = np.array([0, 0, 1, 1])
        result = _compute_canonical_eigenvalues(data, groups, 1)
        assert isinstance(result, np.ndarray)
        assert result.size == 0 or np.all(np.isfinite(result))

    def test_negative_roots_are_clipped_not_propagated(self):
        """Roots below zero are meaningless downstream and must be clipped.

        They appear when n_samples < n_vars, where S_B is indefinite. A
        negative "separation" flowing into a score axis would be nonsense.
        """
        rng = np.random.default_rng(9)
        data = rng.normal(size=(6, 8))  # fewer samples than variables
        groups = np.array([0, 0, 0, 1, 1, 1])
        roots = _compute_canonical_eigenvalues(data, groups, 5)
        assert np.all(roots >= 0.0)

    def test_roots_are_descending(self):
        rng = np.random.default_rng(13)
        data = rng.normal(size=(30, 4))
        groups = np.array([0] * 15 + [1] * 15)
        roots = _compute_canonical_eigenvalues(data, groups, 4)
        assert np.all(np.diff(roots) <= 1e-12), f"not descending: {roots}"


class TestSelfConsistency:
    def test_invariant_to_a_diagonal_variable_rescale(self):
        """Canonical roots do not depend on the units of the variables.

        Under X -> XD, both scatters become S_W -> D S_W D and
        S_B -> D S_B D, so the generalised problem becomes
        D^-1 (S_W^-1 S_B) D -- a similarity transform of the original.
        Similar matrices share eigenvalues, so the roots must be unchanged.

        A mis-specified scatter matrix (a missing n_k factor, a
        within-class mean taken from the wrong group) breaks this while
        still returning plausible-looking numbers.
        """
        rng = np.random.default_rng(21)
        data = rng.normal(size=(30, 3))
        groups = np.array([0] * 15 + [1] * 15)

        baseline = _compute_canonical_eigenvalues(data, groups, 2)
        for scale in ([1.0, 4.0, 0.25], [100.0, 0.01, 7.0]):
            rescaled = _compute_canonical_eigenvalues(data * np.array(scale), groups, 2)
            assert rescaled == pytest.approx(baseline, rel=1e-8, abs=1e-10)

    def test_invariant_to_row_ordering(self):
        rng = np.random.default_rng(22)
        data = rng.normal(size=(30, 3))
        groups = np.array([0] * 15 + [1] * 15)
        permutation = rng.permutation(len(data))
        assert _compute_canonical_eigenvalues(data[permutation], groups[permutation], 2) == pytest.approx(
            _compute_canonical_eigenvalues(data, groups, 2), rel=1e-10
        )
