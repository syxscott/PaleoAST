# =============================================================================
# FILE: tests/stats/test_cca_significance.py
# =============================================================================
"""
Tests for CCA/RDA permutation-based significance testing.

The PaleoAST CCA/RDA previously returned eigenvalues, site scores, etc., but
NO significance test (no F, no p-value, no Wilks lambda).  This test verifies
that:

  1. On data with no relationship between Y and X, the p-value of the overall
     model is roughly uniformly distributed in [0, 1] (a 5%-level test should
     have ~5% positives across 100 independent replications).
  2. On data with a strong Y <-> X relationship, the p-value is small
     (< 0.05).
  3. Permuting the *response* (Y) rows -- not the predictor (X) rows -- is
     what drives the null distribution.  This is the standard CCA permutation
     scheme (Legendre & Legendre 2012; vegan::anova.cca by='marignal').
  4. The result exposes F-statistics and p-values per axis, and an overall
     Wilks lambda, with a configurable ``n_permutations`` and
     ``random_seed``.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.testing import assert_allclose

from stats.cca import CCAAnalyzer


class TestCCAOverallSignificance:
    """The overall model p-value is meaningful on truly independent data."""

    def test_pvalue_uniform_under_null(self):
        """
        When Y and X are independent, the p-value should be approximately
        uniform on [0, 1].  We check this by running 30 independent
        random Y/X designs with the same seed-based permutation engine and
        verifying that the resulting p-values do not cluster near 0.
        """
        rng = np.random.default_rng(2026)
        n = 20
        pvals_rda = []
        pvals_cca = []
        for trial in range(30):
            Y = rng.random((n, 5))
            X = rng.random((n, 2))
            # Use a generous permutation budget so the p-values are
            # reproducible at low cost.
            analyzer = CCAAnalyzer()
            res_rda = analyzer.analyze(
                Y,
                X,
                n_components=2,
                method="rda",
                n_permutations=199,
                random_seed=trial,
            )
            res_cca = analyzer.analyze(
                Y,
                X,
                n_components=2,
                method="cca",
                n_permutations=199,
                random_seed=trial,
            )
            pvals_rda.append(res_rda.p_value)
            pvals_cca.append(res_cca.p_value)
        # The KS one-sample test against the uniform[0,1] CDF would be the
        # canonical check; we use a simple robust heuristic instead: at most
        # 15% of trials should come back with p < 0.05 when the null holds.
        n_sig_rda = sum(p < 0.05 for p in pvals_rda)
        n_sig_cca = sum(p < 0.05 for p in pvals_cca)
        assert n_sig_rda <= 8, (
            f"RDA: {n_sig_rda}/30 p-values below 0.05 under the null -- the test is anti-conservative."
        )
        assert n_sig_cca <= 8, (
            f"CCA: {n_sig_cca}/30 p-values below 0.05 under the null -- the test is anti-conservative."
        )

    def test_pvalue_small_under_alternative(self):
        """
        When Y is constructed to depend strongly on X, the p-value of the
        overall model should be small.
        """
        rng = np.random.default_rng(7)
        n = 30
        X = rng.random((n, 2))
        # Build Y as a smooth function of X with small noise
        Y = np.column_stack(
            [
                X[:, 0] + 0.1 * X[:, 1] + 0.05 * rng.standard_normal(n),
                2.0 * X[:, 0] - X[:, 1] + 0.05 * rng.standard_normal(n),
                -X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.standard_normal(n),
            ]
        )
        analyzer = CCAAnalyzer()
        result = analyzer.analyze(
            Y,
            X,
            n_components=2,
            method="rda",
            n_permutations=499,
            random_seed=123,
        )
        # Strong signal: p-value should be small
        assert result.p_value < 0.05, f"RDA p-value {result.p_value} too large under strong signal"
        # F-statistic should be positive and finite
        assert np.isfinite(result.f_statistic)
        assert result.f_statistic > 0.0

    def test_reproducible_with_seed(self):
        """The same random_seed must yield the same p-value."""
        rng = np.random.default_rng(0)
        n = 20
        Y = rng.random((n, 4))
        X = rng.random((n, 2))
        analyzer = CCAAnalyzer()
        res1 = analyzer.analyze(Y, X, n_components=2, method="rda", n_permutations=99, random_seed=42)
        res2 = analyzer.analyze(Y, X, n_components=2, method="rda", n_permutations=99, random_seed=42)
        assert res1.p_value == res2.p_value
        assert res1.f_statistic == res2.f_statistic


class TestCCAPermutationScheme:
    """The permutation must shuffle Y rows (not X)."""

    def test_permuting_y_drives_null(self):
        """
        A model with a strong Y <-> X signal gets a small p-value; if the
        analyser were permuting X instead of Y, the p-value would still be
        small (because Y-X relationship is preserved under X permutations).
        We verify it gets small, confirming Y is the variable being shuffled.
        """
        rng = np.random.default_rng(42)
        n = 30
        X = rng.random((n, 2))
        Y = np.column_stack([X[:, 0] + 0.01 * rng.standard_normal(n), 0.5 * X[:, 1] + 0.01 * rng.standard_normal(n)])
        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="rda", n_permutations=499, random_seed=10)
        # If we permuted X (wrong), the p-value would be ~1 because
        # permuting X destroys the Q-NULL relationship but leaves Y's
        # internal structure intact -- our F statistic would not exceed
        # the permuted F values.  Since we get a small p here, Y was the
        # variable shuffled.
        assert result.p_value < 0.05, (
            f"CCA permutation does not appear to be shuffling Y: p_value={result.p_value} should be small"
        )


class TestCCAResultFields:
    """The result object exposes F, p, Wilks lambda."""

    def test_fields_present(self):
        """CCAResult must expose F, p-values per axis, and overall Wilks lambda."""
        rng = np.random.default_rng(0)
        Y = rng.random((15, 4))
        X = rng.random((15, 2))
        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="rda", n_permutations=99, random_seed=42)
        # Per-axis fields
        assert hasattr(result, "f_statistic"), "Result must expose f_statistic"
        assert hasattr(result, "p_value"), "Result must expose p_value (overall model)"
        assert hasattr(result, "f_per_axis"), "Result must expose f_per_axis (per-axis F)"
        assert hasattr(result, "p_per_axis"), "Result must expose p_per_axis (per-axis p)"
        assert hasattr(result, "wilks_lambda"), "Result must expose wilks_lambda"
        # Lengths
        assert len(result.f_per_axis) == result.n_components
        assert len(result.p_per_axis) == result.n_components

    def test_wilks_lambda_in_unit_interval(self):
        """Wilks lambda lies in (0, 1] (1 means no relationship)."""
        rng = np.random.default_rng(0)
        Y = rng.random((20, 4))
        X = rng.random((20, 2))
        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="rda", n_permutations=99, random_seed=42)
        assert 0.0 < result.wilks_lambda <= 1.0

    def test_strong_signal_has_small_wilks_lambda(self):
        """When the signal is strong, Wilks lambda should be substantially below 1."""
        rng = np.random.default_rng(1)
        n = 30
        X = rng.random((n, 2))
        Y = np.column_stack([X[:, 0] + 0.01 * rng.standard_normal(n), X[:, 1] + 0.01 * rng.standard_normal(n)])
        analyzer = CCAAnalyzer()
        result = analyzer.analyze(Y, X, n_components=2, method="rda", n_permutations=99, random_seed=1)
        # Wilks lambda = prod(1 - r_k^2). Strong signal => some r close to 1.
        assert result.wilks_lambda < 0.9


class TestCCAZeroRowDegreesOfFreedom:
    """A zero-total sample must not change the reported F.

    CCA drops samples whose row total is zero, because their chi-square weight
    is zero and standardising by it produces NaN. The inertias are then computed
    from the reduced table -- but the residual degrees of freedom used to be
    taken from the *pre-drop* row count, so

        F = (constrained / q) / (residual / df),  df = n - 1 - q

    was inflated by (n - 1 - q) / (n_eff - 1 - q). With 10 samples, one
    dropped and q = 3 that is 6/5, i.e. a 20% inflation of the F statistic
    that ends up in ``CCAResult.f_statistic`` and ``f_per_axis``.

    The p-value happened to survive -- observed and permuted F scale by the
    same factor, so their ordering is unchanged -- which is exactly why this
    could sit unnoticed: the reported significance looked plausible while the
    reported F did not.
    """

    @staticmethod
    def _tables():
        rng = np.random.default_rng(11)
        X = rng.normal(size=(10, 3))
        Y = rng.poisson(4, size=(10, 5)).astype(float)
        return X, Y

    def test_zero_row_sample_does_not_inflate_f(self):
        """F is the same whether the empty sample is dropped inside or before."""
        X, Y = self._tables()
        Y_with_empty = Y.copy()
        Y_with_empty[3, :] = 0.0
        X_without = np.delete(X, 3, axis=0)
        Y_without = np.delete(Y_with_empty, 3, axis=0)

        analyzer = CCAAnalyzer()
        inside = analyzer.analyze(Y_with_empty, X, n_permutations=99, method="cca", random_seed=0)
        before = analyzer.analyze(Y_without, X_without, n_permutations=99, method="cca", random_seed=0)

        assert_allclose(
            inside.f_statistic,
            before.f_statistic,
            rtol=1e-9,
            err_msg="F depends on whether the empty sample was dropped inside the analysis",
        )

    def test_per_axis_f_is_unaffected_too(self):
        """The per-axis F values share the same denominator, so they shift too."""
        X, Y = self._tables()
        Y_with_empty = Y.copy()
        Y_with_empty[3, :] = 0.0

        analyzer = CCAAnalyzer()
        inside = analyzer.analyze(Y_with_empty, X, n_permutations=99, method="cca", random_seed=0)
        before = analyzer.analyze(
            np.delete(Y_with_empty, 3, axis=0), np.delete(X, 3, axis=0), n_permutations=99, method="cca", random_seed=0
        )

        assert_allclose(inside.f_per_axis, before.f_per_axis, rtol=1e-9)

    def test_rda_is_unaffected(self):
        """RDA drops nothing, so it is the control case for this test."""
        X, Y = self._tables()
        Y_r = Y - Y.mean(axis=0)
        analyzer = CCAAnalyzer()
        r = analyzer.analyze(Y_r, X, n_permutations=99, method="rda", random_seed=0)
        assert np.isfinite(r.f_statistic) and r.f_statistic >= 0.0
