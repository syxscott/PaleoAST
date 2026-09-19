"""Tests for statistics/procD_lm.py (geomorph procD.lm port: Procrustes ANOVA + permutation)."""

import numpy as np
import pytest

from utils.exceptions import DataValidationError
from statistics.procD_lm import ProcDLMResult, procD_lm


@pytest.fixture
def correlated_shape_data():
    """Shapes driven by a scalar predictor plus small noise (strong signal)."""
    rng = np.random.default_rng(4)
    n, p = 25, 5
    x = np.linspace(-1.0, 1.0, n)
    pattern = rng.normal(size=(p, 2))
    noise = rng.normal(scale=0.01, size=(n, p, 2))
    shapes = x[:, None, None] * 0.3 * pattern[None] + noise
    shapes = shapes - shapes.mean(axis=1, keepdims=True)
    design = np.column_stack([np.ones(n), x])
    return shapes, design, x


class TestProcDLM:
    def test_strong_predictor_is_significant(self, correlated_shape_data):
        shapes, design, _ = correlated_shape_data
        res = procD_lm(shapes, design, n_permutations=199, seed=7)
        assert isinstance(res, ProcDLMResult)
        by_term = {t.term: t for t in res.terms}
        assert "Intercept" in by_term
        x_term = by_term["X1"]
        assert x_term.p_value is not None
        assert x_term.p_value <= 0.05
        assert x_term.f_value > 1.0
        assert by_term["Intercept"].p_value is None

    def test_residual_df_formula(self, correlated_shape_data):
        shapes, design, _ = correlated_shape_data
        res = procD_lm(shapes, design, n_permutations=0)
        n, p, _ = shapes.shape
        rank = np.linalg.matrix_rank(design)
        assert res.residual_df == n - rank - p + 1
        assert res.n_permutations == 0

    def test_no_permutations_gives_none_p(self, correlated_shape_data):
        shapes, design, _ = correlated_shape_data
        res = procD_lm(shapes, design, n_permutations=0)
        assert all(t.p_value is None for t in res.terms)

    def test_null_predictor_not_significant(self):
        rng = np.random.default_rng(8)
        n, p = 20, 4
        shapes = rng.normal(size=(n, p, 2))
        shapes = shapes - shapes.mean(axis=1, keepdims=True)
        x = rng.normal(size=n)
        design = np.column_stack([np.ones(n), x])
        res = procD_lm(shapes, design, n_permutations=199, seed=8)
        x_term = [t for t in res.terms if t.term == "X1"][0]
        assert x_term.p_value > 0.05

    def test_dict_design_names(self, correlated_shape_data):
        shapes, design, x = correlated_shape_data
        res = procD_lm(shapes, {"Intercept": np.ones(len(x)), "size": x}, n_permutations=49, seed=1)
        term_names = [t.term for t in res.terms]
        assert term_names == ["Intercept", "size"]

    def test_explicit_names(self, correlated_shape_data):
        shapes, design, _ = correlated_shape_data
        res = procD_lm(shapes, design, names=["Intercept", "pred"], n_permutations=0)
        assert [t.term for t in res.terms] == ["Intercept", "pred"]

    def test_r2_and_summary(self, correlated_shape_data):
        shapes, design, _ = correlated_shape_data
        res = procD_lm(shapes, design, n_permutations=0)
        assert 0.0 <= res.r2_by_term["X1"] <= 1.0
        text = res.summary()
        assert "X1" in text and "Residuals" in text

    def test_ss_reduces_with_more_terms(self, correlated_shape_data):
        shapes, design, _ = correlated_shape_data
        res = procD_lm(shapes, design, n_permutations=0)
        assert res.residual_ss < res.total_ss

    def test_error_labels_accepted(self, correlated_shape_data):
        shapes, design, x = correlated_shape_data
        labels = np.arange(len(x)) // 2  # pairs as permutation units
        res = procD_lm(shapes, design, error_labels=labels, n_permutations=49, seed=2)
        assert all(t.p_value is not None for t in res.terms if t.f_value is not None)

    def test_2d_input_raises(self):
        with pytest.raises(DataValidationError):
            procD_lm(np.zeros((10, 4)), np.ones((10, 1)))

    def test_overfull_design_raises(self):
        rng = np.random.default_rng(5)
        shapes = rng.normal(size=(6, 3, 2))
        with pytest.raises(DataValidationError):
            procD_lm(shapes, rng.normal(size=(6, 6)))

    def test_df_zero_raises(self):
        rng = np.random.default_rng(6)
        # n=6, rank=2, p=5 -> df = 6-2-5+1 = 0
        shapes = rng.normal(size=(6, 5, 2))
        design = np.column_stack([np.ones(6), rng.normal(size=6)])
        with pytest.raises(DataValidationError, match="residual df"):
            procD_lm(shapes, design)
