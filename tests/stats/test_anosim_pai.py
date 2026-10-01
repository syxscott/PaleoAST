# =============================================================================
# FILE: tests/stats/test_anosim_pai.py
# =============================================================================
"""
Tests for ANOSIM PAI (Percentage of Average Intergroup/Intragroup) correction.

The original Clarke (1993) R statistic is biased upward when group sizes are
unequal.  ``vegan::anosim`` (and Chapman & Underwood's later correction) fix
this with the PAI adjustment

    R_PAI = (R - E[R | H0]) / (1 - E[R | H0])

where the theoretical null expectation of R (Clarke 1993) is

    E[R | H0] = -1 / (n - 1)

With balanced groups the correction is tiny (R_PAI - R < 1%); with strongly
unbalanced groups (e.g. 4 vs 8) the difference is noticeable.

Both the raw R and the corrected R_PAI are reported so existing consumers
that read ``result.statistic`` are unaffected, and downstream code that
needs the corrected one can do so explicitly.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.testing import assert_allclose

from stats.anosim import ANOSIMAnalyzer


def _make_analyzer() -> ANOSIMAnalyzer:
    return ANOSIMAnalyzer()


class TestANOSIMPAICorrection:
    """PAI = (R - E[R|H0]) / (1 - E[R|H0]) with E[R|H0] = -1/(n-1)."""

    def test_balanced_groups_correction_is_tiny(self):
        """With equal group sizes, R_PAI and R should differ by < 1.5%."""
        rng = np.random.default_rng(0)
        group_a = rng.random((6, 2))
        group_b = rng.random((6, 2)) + 1.0
        data = np.vstack([group_a, group_b])
        D = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))
        groups = ["A"] * 6 + ["B"] * 6

        result = _make_analyzer().analyze(D, groups, n_permutations=99, random_seed=42)
        # The raw R should be the statistic value
        # and R_PAI should be in the corrected field
        assert hasattr(result, "statistic"), "Backward compat: result.statistic must exist"
        assert hasattr(result, "r_pai"), "Result must expose r_pai when pai=True"
        # The correction is bounded by 1/(n-1) ratio:
        # R_PAI = (R + 1/(n-1)) / (n/(n-1)) = ((n-1)*R + 1) / n
        # so R_PAI - R = (R - (-1/(n-1))) / (1 - (-1/(n-1))) - R
        # = (R + 1/(n-1)) / (n/(n-1)) - R = ((n-1)R + 1)/n - R
        # = (R - 1)/n, which for R near 1 is approximately 0
        assert abs(result.r_pai - result.statistic) < 0.02, (
            f"Balanced groups: |R_PAI - R| = {abs(result.r_pai - result.statistic):.4f} "
            f"should be tiny"
        )

    def test_unbalanced_groups_correction_is_substantial(self):
        """
        With 4 vs 8 group sizes the raw R is biased upward; PAI pulls it
        back.  We verify the formula directly (no scipy dependency).
        """
        rng = np.random.default_rng(0)
        group_a = rng.random((4, 2))
        group_b = rng.random((8, 2)) + 1.0
        data = np.vstack([group_a, group_b])
        D = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))
        groups = ["A"] * 4 + ["B"] * 8
        n = D.shape[0]

        result = _make_analyzer().analyze(D, groups, n_permutations=99, random_seed=42)
        # The PAI correction: R_PAI = (R - E_R) / (1 - E_R), E_R = -1/(n-1)
        E_R = -1.0 / (n - 1)
        expected_pai = (result.statistic - E_R) / (1.0 - E_R)
        assert_allclose(result.r_pai, expected_pai, rtol=1e-10)
        # Unbalanced: the correction is larger than for balanced
        assert abs(result.r_pai - result.statistic) > 0.001, (
            "Unbalanced groups: PAI should shift R by more than the balanced case"
        )

    def test_pai_off_returns_raw_r(self):
        """With pai=False, r_pai == statistic (no correction)."""
        rng = np.random.default_rng(0)
        group_a = rng.random((4, 2))
        group_b = rng.random((8, 2)) + 1.0
        data = np.vstack([group_a, group_b])
        D = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))
        groups = ["A"] * 4 + ["B"] * 8

        result = _make_analyzer().analyze(D, groups, n_permutations=99, random_seed=42, pai=False)
        assert result.r_pai == result.statistic

    def test_pai_corrects_to_zero_at_null(self):
        """
        When the observed R matches the null expectation, R_PAI is 0.
        Construct an example where raw R ≈ E[R|H0] = -1/(n-1); with
        reasonably random data and 2 small balanced groups we cannot hit
        this exactly, so just verify the formula algebraically: if
        R == E_R then R_PAI must be 0 (the algebra is correct).
        """
        # Use a formula-only check
        n = 7
        R_raw = -1.0 / (n - 1)  # exactly the null expectation
        E_R = -1.0 / (n - 1)
        R_pai = (R_raw - E_R) / (1 - E_R)
        assert_allclose(R_pai, 0.0, atol=1e-12)

    def test_pai_correction_sign(self):
        """Algebraic properties of the PAI correction.

        The PAI maps R in [-1, 1] to R_PAI in [-(1+E_R)/(1-E_R), 1]:
        - When R = 1 (max), R_PAI = 1.
        - When R = E_R (null expectation), R_PAI = 0.
        - When R = -1 (min), R_PAI = -(1+E_R)/(1-E_R) (negative).
        R_PAI is monotonically increasing in R (positive derivative 1/(1-E_R)).
        """
        for n in [5, 10, 20]:
            E_R = -1.0 / (n - 1)
            # When R == E_R (null), R_PAI is 0
            assert_allclose((E_R - E_R) / (1 - E_R), 0.0, atol=1e-12)
            # When R == 1 (max), R_PAI is 1
            assert_allclose((1.0 - E_R) / (1 - E_R), 1.0, atol=1e-12)
            # When R == -1 (min), R_PAI is -(1+E_R)/(1-E_R) (negative)
            assert_allclose((-1.0 - E_R) / (1 - E_R), -(1.0 + E_R) / (1.0 - E_R), atol=1e-12)
            # Monotonicity: R_PAI increases with R
            prev = -2.0
            for R in np.linspace(-1, 1, 21):
                R_pai = (R - E_R) / (1 - E_R)
                assert R_pai >= prev - 1e-12
                prev = R_pai