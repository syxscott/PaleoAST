# =============================================================================
# FILE: tests/cross_validation/test_vs_r_base.py
# =============================================================================
"""
Cross-validation against base R's ``stats`` package.

These compare PaleoAST's PCA against ``stats::prcomp`` -- the function every R
user reaches for first, and the one whose numbers a reviewer is most likely to
have reproduced by hand. The comparison is live: R computes its answer during
the test run, so this is a real check against the reference implementation
rather than a restated formula.

Not compared
------------
Loadings are compared after sign alignment, because the sign of a principal
component is arbitrary (PC1 and -PC1 describe the same axis). Comparing raw
signs would fail half the time for no reason.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import as_array, as_float, r, r_matrix, require

pytestmark = pytest.mark.cross_validation

R_STATS = require("stats")


def _align_signs(paleo: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip each column of ``paleo`` so it points the same way as ``reference``.

    A principal component is a direction, not a vector: PC_k and -PC_k span the
    same axis and explain the same variance. Comparison therefore has to be
    sign-invariant, but it must still be *component*-specific -- so the sign is
    chosen from the largest-magnitude entry of the reference column, which is
    stable, rather than by dot product, which flips whenever two components
    happen to be near-orthogonal.
    """
    out = np.array(paleo, dtype=float, copy=True)
    for col in range(out.shape[1]):
        ref = reference[:, col]
        pivot = int(np.argmax(np.abs(ref)))
        if np.sign(ref[pivot]) * np.sign(out[pivot, col]) < 0:
            out[:, col] *= -1.0
    return out


def _dataset() -> np.ndarray:
    """A deterministic, correlated 30x5 matrix.

    Seeded and moderately correlated on purpose: an identity-like correlation
    matrix would make every component equally large and a sign/ordering mistake
    much harder to notice.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(size=(30, 3))
    loadings = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.3, 0.1],
            [0.7, 0.4, 0.2],
            [0.2, 0.8, 0.3],
            [0.1, 0.7, 0.5],
        ]
    )
    return base @ loadings.T + np.array([10.0, 5.0, 3.0, 20.0, 8.0])


class TestBridgeIsLive:
    """Guard that this suite still reaches R at all.

    The previous version of ``tests/cross_validation/`` imported rpy2, set a
    flag nobody read, and compared PaleoAST against hand-written constants --
    31 collected tests that never called R. Nothing failed, because nothing was
    wrong as far as pytest was concerned.

    These tests are the tripwire for that failure mode. If the bridge ever
    degrades to returning local constants, ``test_r_evaluates_r_code`` fails,
    because R has to actually execute a string and return a value only R knows.
    """

    def test_r_evaluates_r_code(self):
        """R is running and returns a value computed in R."""
        from ._rbridge import as_float, r

        # 7 * 6 is arithmetic R performs; no local constant could satisfy this
        # if the bridge were not really talking to R.
        #
        # The parentheses matter: `as_float(r["*"])(7, 6)` parses as
        # `(as_float(r["*"]))(7, 6)`, which hands the *function* to as_float and
        # raises `TypeError: object of type 'SignatureTranslatedFunction' has no
        # len()` before R ever multiplies anything. This test is the tripwire
        # for the bridge degrading to local constants, so it has to actually
        # call R.
        assert as_float(r["*"](7, 6)) == 42.0

    def test_r_lies_about_nothing(self):
        """A deliberately wrong expectation from R must not pass.

        A weaker check would be to assert R returns *something*. Asserting it
        returns the wrong number proves the value came from R rather than from a
        stub, and that the helper is not echoing its own input.
        """
        from ._rbridge import as_float, r

        assert as_float(r("sum")(r("c")(1, 2, 3, 4))) == 10.0
        # Same input, different function, different answer.
        assert as_float(r("max")(r("c")(1, 2, 3, 4))) == 4.0

    def test_reference_packages_load(self):
        """The four reference packages used here are installed and importable.

        Skips (rather than fails) per package, because a partial CRAN install is
        an environment problem, not a defect in PaleoAST.
        """
        for package in ("vegan", "ape", "geomorph", "iNEXT"):
            require(package)
        # iNEXT installs as a lowercase 'iNEXT'; require() keys on the exact
        # name, so this also confirms the spelling used in the other modules.


class TestPCAVsPrcomp:
    """Verify PCA against ``stats::prcomp``."""

    def test_covariance_eigenvalues_match(self):
        """Eigenvalues of the covariance PCA match ``prcomp``'s ``sdev^2``."""
        from stats.pca import PCAAnalyzer

        x = _dataset()
        result = PCAAnalyzer().analyze(x, n_components=5, method="covariance")

        r_prcomp = R_STATS.prcomp(r_matrix(x), center=True, scale_=False)
        r_eigenvalues = as_array(r_prcomp.rx2("sdev")) ** 2

        assert_allclose(
            np.asarray(result.eigenvalues),
            r_eigenvalues,
            rtol=1e-8,
            atol=1e-10,
            err_msg="covariance PCA eigenvalues disagree with stats::prcomp",
        )

    def test_correlation_eigenvalues_match(self):
        """Eigenvalues of the correlation PCA match ``prcomp(scale.=TRUE)``."""
        from stats.pca import PCAAnalyzer

        x = _dataset()
        result = PCAAnalyzer().analyze(x, n_components=5, method="correlation")

        r_prcomp = R_STATS.prcomp(r_matrix(x), center=True, scale_=True)
        r_eigenvalues = as_array(r_prcomp.rx2("sdev")) ** 2

        assert_allclose(
            np.asarray(result.eigenvalues),
            r_eigenvalues,
            rtol=1e-8,
            atol=1e-10,
            err_msg="correlation PCA eigenvalues disagree with stats::prcomp",
        )

    def test_variance_proportions_match(self):
        """``explained_variance`` (as a percentage) matches prcomp's share."""
        from stats.pca import PCAAnalyzer

        x = _dataset()
        result = PCAAnalyzer().analyze(x, n_components=5, method="covariance")

        r_prcomp = R_STATS.prcomp(r_matrix(x), center=True, scale_=False)
        sdev = as_array(r_prcomp.rx2("sdev"))
        r_share = sdev**2 / np.sum(sdev**2) * 100.0

        assert_allclose(
            np.asarray(result.explained_variance),
            r_share,
            rtol=1e-8,
            atol=1e-8,
            err_msg="variance proportions disagree with stats::prcomp",
        )
        # The percentages must also add to 100 across all components, which is a
        # cheap independent check that the slicing did not drop a component.
        assert_allclose(float(np.sum(result.explained_variance)), 100.0, atol=1e-8)

    def test_loadings_match_after_sign_alignment(self):
        """Loadings match ``prcomp``'s rotation, once the conventions agree.

        PaleoAST reports *factor-analysis* loadings, ``V * sqrt(lambda)``
        (see ``stats/pca.py``: ``loadings = V * np.sqrt(eigenvalues)``), which
        is what ``scores = X_centered @ loadings`` requires. ``prcomp``'s
        ``rotation`` is instead the unit-norm eigenvector matrix. Comparing
        the two directly fails by a factor of ``sqrt(lambda)`` per component
        and looks like a numerical disagreement when nothing is wrong. R's
        equivalent is ``prcomp$rotation %*% diag(prcomp$sdev)``, so that is
        what gets compared here.
        """
        from stats.pca import PCAAnalyzer

        x = _dataset()
        result = PCAAnalyzer().analyze(x, n_components=3, method="covariance")

        r_prcomp = R_STATS.prcomp(r_matrix(x), center=True, scale_=False)
        sdev = as_array(r_prcomp.rx2("sdev"))
        r_loadings = np.array(r_prcomp.rx2("rotation"), dtype=float)[:, :3] * sdev[:3]

        paleo = _align_signs(np.asarray(result.loadings, dtype=float), r_loadings)
        assert_allclose(
            paleo,
            r_loadings,
            rtol=1e-6,
            atol=1e-8,
            err_msg="PCA loadings disagree with stats::prcomp (after sign alignment)",
        )
        # Also pin the convention itself, so a future change to `loadings`
        # cannot silently redefine what this comparison means: a loadings
        # matrix of the form V*sqrt(lambda) has column norms sqrt(lambda).
        assert_allclose(
            np.linalg.norm(np.asarray(result.loadings), axis=0),
            np.sqrt(np.asarray(result.eigenvalues)),
            rtol=1e-10,
            err_msg="loadings are not V*sqrt(lambda): column norms should equal sqrt(eigenvalue)",
        )

    def test_scores_are_centred(self):
        """Scores are zero-centred, as ``prcomp$x`` is.

        This is a property R also guarantees, so it is checked against R's own
        output rather than against a hardcoded zero.

        Only the first ``n_components`` columns are compared. R returns a score
        for every variable (5 here) while the analyzer was asked for 3, so
        comparing the full vectors is a shape mismatch, not a numerical one.
        """
        from stats.pca import PCAAnalyzer

        x = _dataset()
        n_components = 3
        result = PCAAnalyzer().analyze(x, n_components=n_components, method="covariance")

        r_prcomp = R_STATS.prcomp(r_matrix(x), center=True, scale_=False)
        r_score_means = as_array(r("colMeans")(r_prcomp.rx2("x")))[:n_components]

        assert_allclose(
            np.mean(np.asarray(result.scores), axis=0),
            r_score_means,
            atol=1e-8,
        )
        # A vector of zeros would pass the comparison above trivially, so also
        # assert the scores are not degenerate.
        assert float(np.std(np.asarray(result.scores))) > 1e-6
        assert as_float(as_array(r("sum")(r_prcomp.rx2("x") ** 2)) > 0.0)
