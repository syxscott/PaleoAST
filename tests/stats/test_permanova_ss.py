# =============================================================================
# FILE: tests/stats/test_permanova_ss.py
# =============================================================================
"""
Tests for the PERMANOVA sum-of-squares divisors.

The pseudo-F of Anderson (2001), computed over UNORDERED pairs:

    SS_T = (1 / n) * sum_{i<j} d^2_ij
    SS_W = sum_g (1 / n_g) * sum_{i<j in g} d^2_ij
    SS_B = SS_T - SS_W
    F    = (SS_B / (g - 1)) / (SS_W / (n - g))

WHY THE UNORDERED FORM
----------------------
The full squared matrix holds every pair twice, so ``sum(D**2)/(2n)`` and
``sum over upper-triangle pairs / n`` are the same number. The divisor is
the sample count.

This file previously asserted the OPPOSITE -- ``n - 1`` and ``n_g - 1`` --
and cited ``vegan::adonis2`` as doing the same. It does not. Checked
against vegan 2.7.6 on this repository's cross-validation fixture
(16 samples, two groups of eight, five variates, seed 11):

                            this file claimed   adonis2     this file now
    SS_T (total)            436.39789701        409.12302844 409.12302844
    SS_W (residual)          63.75686326         55.78725536  55.78725536
    R^2                      0.8539019924        0.8636418596 0.8636418596

The distance matrix is identical on both sides -- ``sum(D**2)/(2n)`` agrees
to the last digit -- so the difference was never the distances. It was the
divisors, and because the two terms were scaled by different factors they
did not cancel.

The permutation p-value would never have shown this. A uniform scaling of
both terms leaves the permutation distribution's shape alone; only the
R-squared and F carry the evidence, which is why the cross-validation
job exists.

WHAT GUARDS IT NOW
------------------
test_total_ss_is_the_distance_matrix_total is the assertion that does not
depend on anyone's memory of the formula. If someone changes a divisor
again, that test fails regardless of which divisor the author believed.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pytest
from numpy.testing import assert_allclose

from stats.permanova import PERMANOVAAnalyzer


def _reference(D: np.ndarray, groups: np.ndarray) -> tuple[float, float, float, float]:
    """(SS_T, SS_W, SS_B, F) from the definitions, computed independently."""
    n = D.shape[0]
    g = len(np.unique(groups))
    upper = np.triu_indices(n, k=1)
    ss_t = float(np.sum(D[upper] ** 2) / n)
    ss_w = 0.0
    for grp in np.unique(groups):
        idx = np.where(groups == grp)[0]
        n_g = len(idx)
        if n_g < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        ss_w += float(np.sum(sub[np.triu_indices(n_g, k=1)] ** 2) / n_g)
    ss_b = ss_t - ss_w
    f = (ss_b / (g - 1)) / (ss_w / (n - g))
    return ss_t, ss_w, ss_b, f


D6 = np.array(
    [
        [0.0, 1.0, 2.0, 5.0, 6.0, 5.5],
        [1.0, 0.0, 2.5, 5.5, 6.5, 6.0],
        [2.0, 2.5, 0.0, 4.5, 5.5, 5.0],
        [5.0, 5.5, 4.5, 0.0, 1.0, 1.5],
        [6.0, 6.5, 5.5, 1.0, 0.0, 1.0],
        [5.5, 6.0, 5.0, 1.5, 1.0, 0.0],
    ]
)
G6 = np.array(["A", "A", "A", "B", "B", "B"])


class TestPERMANOVADivisors:
    """The divisors are the sample and group counts."""

    def test_total_ss_is_the_distance_matrix_total(self):
        """SS_T must equal sum(D**2) / (2n).

        This is the assertion that does not depend on remembering the
        formula: it says the total is the total of the matrix that was
        handed in. Any divisor other than n fails it. It is the check the
        previous version of this file lacked, which is why a wrong
        divisor sat here being asserted as correct.
        """
        result = PERMANOVAAnalyzer().analyze(D6, list(G6), n_permutations=9, random_seed=0)
        n = D6.shape[0]
        expected_total = float(np.sum(D6**2)) / (2 * n)
        assert_allclose(
            float(result.ss_between) + float(result.ss_within),
            expected_total,
            rtol=1e-12,
            err_msg=("ss_between + ss_within must equal the total sum of squares of the distance matrix"),
        )

    def test_divisors_match_the_reference(self):
        """SS_T, SS_W and F match a computation from the definitions."""
        result = PERMANOVAAnalyzer().analyze(D6, list(G6), n_permutations=9, random_seed=0)
        _ss_t, ss_w, _ss_b, f = _reference(D6, G6)
        assert_allclose(float(result.ss_within), ss_w, rtol=1e-12)
        assert_allclose(float(result.f_statistic), f, rtol=1e-12)

    def test_pinned_values(self):
        """Exact values, so a divisor change cannot hide inside a tolerance.

        SS_T = 48.4583333, SS_W = 5.1666667, SS_B = 43.2916667,
        F = 33.51612903. The divisor this file previously asserted
        gives SS_T = 58.15 and F = 26.0129.
        """
        result = PERMANOVAAnalyzer().analyze(D6, list(G6), n_permutations=9, random_seed=0)
        ss_t, ss_w, ss_b, f = _reference(D6, G6)
        assert_allclose(float(result.ss_between) + float(result.ss_within), 48.458333333333336, atol=1e-10)
        assert_allclose(float(result.ss_within), 5.166666666666667, atol=1e-10)
        assert_allclose(float(result.ss_between), 43.291666666666664, atol=1e-10)
        assert_allclose(float(result.f_statistic), 33.516129032258064, atol=1e-10)
        assert (ss_t, ss_w, ss_b, f) == pytest.approx(
            (48.458333333333336, 5.166666666666667, 43.291666666666664, 33.516129032258064)
        )

    def test_unequal_group_sizes(self):
        """n = 5 with groups of 3 and 2.

        Unequal sizes are where a divisor error shows most clearly,
        because the two terms are then scaled by different factors and
        the error cannot cancel.
        """
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0], [11.0, 10.0]])
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt((diff**2).sum(axis=2))
        groups = np.array(["A", "A", "A", "B", "B"])

        result = PERMANOVAAnalyzer().analyze(D, list(groups), n_permutations=9, random_seed=0)
        _ss_t, _ss_w, _ss_b, f = _reference(D, groups)
        assert_allclose(float(result.f_statistic), f, rtol=1e-12)

    def test_small_matrix(self):
        """n = 4 with groups of 2 and 2."""
        D = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.5, 2.0],
                [2.0, 1.5, 0.0, 2.5],
                [3.0, 2.0, 2.5, 0.0],
            ]
        )
        groups = np.array(["A", "A", "B", "B"])
        result = PERMANOVAAnalyzer().analyze(D, list(groups), n_permutations=9, random_seed=0)
        ss_t, ss_w, _ss_b, f = _reference(D, groups)
        assert_allclose(float(result.ss_within), ss_w, rtol=1e-12)
        assert_allclose(float(result.f_statistic), f, rtol=1e-12)
        assert_allclose(float(result.ss_between) + float(result.ss_within), ss_t, rtol=1e-12)
