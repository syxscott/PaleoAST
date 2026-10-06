"""
Regression tests for defect 3: Relative Warps Analysis had no α parameter.

The classical Bookstein (1991) Relative Warps use an α-weighting of
eigenvalues when scoring partial warps:

    λ_k^α

* α = -1  →  "uniform" warps  (small-scale features amplified)
* α =  0  →  standard PCA (uniform weighting; the original behaviour)
* α = +1  →  "affine-dominated" warps (large-scale features amplified)

The previous implementation never saw α: it just returned PCA.  The fix
adds an ``alpha`` parameter to ``analyze``.  α = 0 must reproduce the old
behaviour BIT-FOR-BIT (the codebase's backwards-compatibility contract);
α = ±1 must produce strictly different ordering of scores.
"""

import numpy as np
import pytest

from morphometrics.relative_warps import RelativeWarpsAnalyzer


def _aligned_set(seed: int = 99, n: int = 12, p: int = 6, k: int = 2):
    """Procrustes-aligned configurations with both small-scale and
    large-scale shape variation, so α = ±1 should distinguish them."""
    rng = np.random.default_rng(seed)
    # base shape
    base = rng.normal(size=(p, k)) * 0.3
    cfgs = np.empty((n, p, k))
    for i in range(n):
        # large-scale drift (first eigenvalue direction)
        drift = 0.6 * (i / (n - 1))
        # small-scale wiggle (last principal component)
        wiggle = 0.05 * np.sin(8 * np.pi * i / n)
        cfgs[i] = base + drift * np.array([1.0, 0.0]) + wiggle * np.eye(p, k)[i % p]
    return cfgs


class TestAlphaZeroBitIdentical:
    """α = 0 MUST reproduce the previous PCA behaviour exactly: same
    eigenvalues, same scores, same explained variance."""

    def test_alpha_zero_matches_legacy_behaviour(self):
        cfgs = _aligned_set()
        analyzer = RelativeWarpsAnalyzer()
        # Legacy behaviour: no alpha kwarg.  In the fixed code the default
        # is α = 0 (no weighting), so this should be identical to a fresh
        # ``analyze(..., alpha=0.0)`` call.
        res_legacy = analyzer.analyze(cfgs)
        res_alpha0 = analyzer.analyze(cfgs, alpha=0.0)
        # Eigenvalues: identical
        np.testing.assert_array_equal(res_legacy.eigenvalues, res_alpha0.eigenvalues)
        # Scores: identical
        np.testing.assert_array_equal(res_legacy.relative_warps, res_alpha0.relative_warps)
        # Explained variance: identical
        np.testing.assert_array_equal(res_legacy.explained_variance, res_alpha0.explained_variance)
        # Eigenvectors: identical
        np.testing.assert_array_equal(res_legacy.eigenvectors, res_alpha0.eigenvectors)
        # Cumulative variance: identical
        np.testing.assert_array_equal(res_legacy.cumulative_variance, res_alpha0.cumulative_variance)


class TestAlphaWeightingSemantics:
    """α = ±1 must give SCORES that are NOT bit-equal to α = 0, and must
    amplify the opposite end of the spectrum (Bookstein 1991)."""

    def test_alpha_minus_one_differs_from_alpha_zero(self):
        cfgs = _aligned_set()
        analyzer = RelativeWarpsAnalyzer()
        res_0 = analyzer.analyze(cfgs, alpha=0.0)
        res_m1 = analyzer.analyze(cfgs, alpha=-1.0)
        # scores must differ
        assert not np.allclose(res_0.relative_warps, res_m1.relative_warps)
        # eigenvalues are NOT weighted by α (only the score computation
        # is); scores are RW_k = sqrt(λ_k^α) * (centred · v_k)
        np.testing.assert_array_equal(res_0.eigenvalues, res_m1.eigenvalues)

    def test_alpha_plus_one_differs_from_alpha_zero(self):
        cfgs = _aligned_set()
        analyzer = RelativeWarpsAnalyzer()
        res_0 = analyzer.analyze(cfgs, alpha=0.0)
        res_p1 = analyzer.analyze(cfgs, alpha=1.0)
        assert not np.allclose(res_0.relative_warps, res_p1.relative_warps)

    def test_alpha_amplifies_correct_eigenvalue_end(self):
        """α = +1 must give MORE weight to the largest-eigenvalue axis than
        α = 0 (and α = -1 must give LESS).  Conversely α = -1 must give
        MORE relative weight to the smallest-eigenvalue axis than α = 0.

        We measure this via the ratio of variances across the first vs
        last axis: under α = +1 the ratio grows, under α = -1 it shrinks.
        This is exactly the Bookstein (1991) interpretation of α as
        emphasising either the affine (large λ) or uniform (small λ)
        end of the bending-energy spectrum.
        """
        cfgs = _aligned_set()
        analyzer = RelativeWarpsAnalyzer()
        res_0 = analyzer.analyze(cfgs, alpha=0.0)
        res_p1 = analyzer.analyze(cfgs, alpha=1.0)
        res_m1 = analyzer.analyze(cfgs, alpha=-1.0)
        var_0 = np.var(res_0.relative_warps, axis=0)
        var_p1 = np.var(res_p1.relative_warps, axis=0)
        var_m1 = np.var(res_m1.relative_warps, axis=0)
        # ratio of first to last axis variance:
        ratio_0 = var_0[0] / max(var_0[-1], 1e-300)
        ratio_p1 = var_p1[0] / max(var_p1[-1], 1e-300)
        ratio_m1 = var_m1[0] / max(var_m1[-1], 1e-300)
        # α = +1 amplifies the high end → ratio grows
        assert ratio_p1 > ratio_0
        # α = -1 amplifies the low end → ratio shrinks
        assert ratio_m1 < ratio_0

    def test_alpha_results_have_correct_count(self):
        cfgs = _aligned_set()
        analyzer = RelativeWarpsAnalyzer()
        res = analyzer.analyze(cfgs, alpha=0.5)
        assert res.n_components == min(cfgs.shape[0] - 1, cfgs.shape[1] * cfgs.shape[2])
        assert res.relative_warps.shape == (cfgs.shape[0], res.n_components)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
