# tests/ecology/test_combo_score_scale.py
"""
Regression test for the combo-score scale mixing (缺陷 6).

The previous implementation averaged the raw C-score (which is
O(n_sites²), typically ≫ 1) with a normalised checkerboard count in
[0, 1], so the C-score dominated and the result was not interpretable
on a common scale. The fix normalises C-score by the theoretical
``n_sites²`` per-pair maximum.
"""

import numpy as np

from ecology.null_models import NullModelAnalyzer


def _perfect_checkerboard(n: int) -> np.ndarray:
    """n × n matrix with alternating presence — every species pair
    forms a perfect checkerboard."""
    return np.indices((n, n)).sum(axis=0) % 2


def _all_present(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=int)


class TestComboScoreScale:
    """Combo score must be on a [0, 1] scale for both components."""

    def test_combo_score_bounded_above_by_one(self):
        """Even with a maximally segregated matrix, the combo score
        must not exceed 1 — both components sit in [0, 1]."""
        m = _perfect_checkerboard(6)
        analyzer = NullModelAnalyzer()
        score = analyzer._compute_combo_score(m)
        assert 0.0 <= score <= 1.0 + 1e-9, f"Combo score {score} outside [0, 1] — components are on different scales"

    def test_combo_score_high_for_segregated(self):
        """A perfectly segregated matrix must give a combo score
        substantially larger than a fully co-occurring one."""
        m_segreg = _perfect_checkerboard(6)
        m_cocos = _all_present(6)
        analyzer = NullModelAnalyzer()
        s_segreg = analyzer._compute_combo_score(m_segreg)
        s_cocos = analyzer._compute_combo_score(m_cocos)
        assert s_segreg > s_cocos, (
            f"Score {s_segreg} (segregated) ≤ {s_cocos} (co-occurring) — "
            "the metric is not actually measuring segregation"
        )

    def test_combo_score_c_score_normalised(self):
        """For a maximally-segregated matrix the C-score component
        of the combo must equal the normalized C-score, i.e. must
        NOT be the raw C-score divided by 2.

        Concretely: a checkerboard 6×6 matrix has
        raw_C = mean of C_ij over all pairs.  With the fix,
        combo_score = (norm_C + norm_checkerboard) / 2.
        With the bug, combo_score = (raw_C + norm_checkerboard) / 2,
        which for a 6×6 matrix would be much larger than 1.
        """
        m = _perfect_checkerboard(6)
        analyzer = NullModelAnalyzer()
        score = analyzer._compute_combo_score(m)
        # Max possible C-score for an n_sites=6 matrix is 6×6=36.
        # Bug: combo ≈ (raw_C + 1) / 2 — for n=6 raw_C is around 15-20.
        # Fix: combo = (raw_C / 36 + 1) / 2 ≈ 0.7-0.78.
        # We just assert combo ≤ 1.0 to detect the scale bug.
        assert score <= 1.0 + 1e-9
