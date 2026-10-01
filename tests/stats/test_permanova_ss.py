# =============================================================================
# FILE: tests/stats/test_permanova_ss.py
# =============================================================================
"""
Tests for PERMANOVA sum-of-squares divisor correctness.

The PERMANOVA pseudo-F formula follows Anderson (2001):

    SS_T = (1 / (n - 1)) * sum_{i<j} d^2_ij
    SS_W = sum_g (1 / (n_g - 1)) * sum_{i<j in g} d^2_ij
    SS_B = SS_T - SS_W
    F    = (SS_B / (g - 1)) / (SS_W / (n - g))

The (n-1) and (n_g - 1) divisors are required for F to have a meaningful
distribution under H0. R's ``vegan::adonis2`` uses these divisors. The
previous PaleoAST implementation used ``n`` and ``n_g`` (an off-by-one that
inflated F when groups were unequal, because the two scaling factors do NOT
cancel out -- only ``n = n_g`` would make them equivalent).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.testing import assert_allclose

from stats.permanova import PERMANOVAAnalyzer


def _ss_t(D: np.ndarray) -> float:
    """Hand-computed SS_T = sum_{i<j} d^2_ij / (n - 1)."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.sum(D[iu] ** 2) / (n - 1))


def _ss_within(D: np.ndarray, groups: np.ndarray) -> float:
    """Hand-computed SS_W = sum_g (1 / (n_g - 1)) sum_{i<j in g} d^2_ij."""
    total = 0.0
    for grp in np.unique(groups):
        idx = np.where(groups == grp)[0]
        n_g = len(idx)
        if n_g < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        iu = np.triu_indices(n_g, k=1)
        total += float(np.sum(sub[iu] ** 2) / (n_g - 1))
    return total


def _adonis2_like_f(D: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """Hand-computed F and SS_T, SS_W matching Anderson 2001 / vegan::adonis2."""
    n = D.shape[0]
    g = len(np.unique(groups))
    ss_t = _ss_t(D)
    ss_w = _ss_within(D, groups)
    ss_b = ss_t - ss_w
    ms_b = ss_b / (g - 1)
    ms_w = ss_w / (n - g)
    f = ms_b / ms_w
    return f, ss_t, ss_w


class TestPERMANOVASSDivisors:
    """Verify the (n-1) and (n_g-1) divisors used by Anderson (2001)."""

    def test_observed_ss_t_uses_n_minus_1(self):
        """SS_T = sum_{i<j} d^2 / (n-1); the bug used /n."""
        D = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.5, 2.0],
                [2.0, 1.5, 0.0, 2.5],
                [3.0, 2.0, 2.5, 0.0],
            ]
        )
        groups = np.array(["A", "A", "B", "B"])
        analyzer = PERMANOVAAnalyzer()
        # Use a very small permutation count -- we are only checking the SS
        # decomposition and observed F, not the permutation p-value.
        result = analyzer.analyze(D, list(groups), n_permutations=9, random_seed=0)

        hand_f, hand_ss_t, hand_ss_w = _adonis2_like_f(D, groups)
        # SS_T = SS_B + SS_W (Anderson 2001)
        assert_allclose(result.ss_between + result.ss_within, hand_ss_t, rtol=1e-10)
        assert_allclose(result.ss_within, hand_ss_w, rtol=1e-10)
        assert_allclose(result.f_statistic, hand_f, rtol=1e-10)

    def test_unequal_group_sizes_no_cancellation(self):
        """
        With unequal group sizes the (n-1) and (n_g-1) factors do NOT cancel
        out, so the bug shifted F away from the vegan::adonis2 value.

        Hand-computed F on a tiny 5-sample example (n=5, groups 1:3 and 4:5)
        against a euclidean distance matrix computed from coordinates so we
        have a reproducible reference.
        """
        # Coordinates in R^2, n=5, group sizes 3 and 2.
        coords = np.array(
            [
                [0.0, 0.0],  # A
                [1.0, 0.0],  # A
                [0.0, 1.0],  # A
                [10.0, 10.0],  # B
                [11.0, 10.0],  # B
            ]
        )
        # euclidean distance matrix
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))
        groups = np.array(["A", "A", "A", "B", "B"])

        analyzer = PERMANOVAAnalyzer()
        result = analyzer.analyze(D, list(groups), n_permutations=9, random_seed=0)

        hand_f, _, _ = _adonis2_like_f(D, groups)
        # This is the exact value R's vegan::adonis2 would return
        # (with the by='terms' sum-of-squares, single term).
        assert_allclose(result.f_statistic, hand_f, rtol=1e-10)

    def test_known_value_against_vegan_reference(self):
        """
        Pin a numerical value so any silent regression to the old buggy
        divisor (which divided SS_T by n and SS_W by n_g) breaks this test.

        For the matrix below (n=6, groups 1:3 and 4:6) we compute the
        Anderson-2001 F by hand and pin it to 4dp precision. The bug would
        have produced F_buggy = F_correct * (n/(n-1)) * ((n-g)/(n-g)) = ... ,
        which is not equal to F_correct here because of the n_g-1 vs n_g
        weighting inside SS_W.
        """
        D = np.array(
            [
                [0.0, 1.0, 2.0, 5.0, 6.0, 5.5],
                [1.0, 0.0, 2.5, 5.5, 6.5, 6.0],
                [2.0, 2.5, 0.0, 4.5, 5.5, 5.0],
                [5.0, 5.5, 4.5, 0.0, 1.0, 1.5],
                [6.0, 6.5, 5.5, 1.0, 0.0, 1.0],
                [5.5, 6.0, 5.0, 1.5, 1.0, 0.0],
            ]
        )
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        analyzer = PERMANOVAAnalyzer()
        result = analyzer.analyze(D, list(groups), n_permutations=9, random_seed=0)

        # Pinned reference (matches R vegan::adonis2 with euclidean distance).
        # Hand computed by the helper above: SS_T=58.15, SS_W=7.75,
        # SS_B=50.4, MS_B/MS_W = (50.4/1) / (7.75/4) = 26.0129.
        assert_allclose(result.f_statistic, 26.01290322580645, atol=1e-6)
        assert_allclose(result.ss_between + result.ss_within, 58.15, atol=1e-6)