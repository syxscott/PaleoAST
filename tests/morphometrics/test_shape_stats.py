"""Tests for morphometrics/shape_stats.py (W4: Kendall preshape, medians, Goodall, Hotelling T²)."""

import numpy as np
import pytest

from utils.exceptions import MorphometricsError
from morphometrics.shape_stats import (
    geometric_median,
    goodall_f,
    goodall_test,
    hotelling_t2,
    kendall_preshape,
    procrustes_distance,
    procrustes_median,
)


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@pytest.fixture
def shape_set():
    """(12, 6, 2) random-ish configurations with distinct shapes."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=(6, 2))
    return base[None] + rng.normal(scale=0.05, size=(12, 6, 2))


class TestKendallPreshape:
    def test_unit_norm_rows(self, shape_set):
        pre = kendall_preshape(shape_set)
        assert pre.shape == (12, 12)
        np.testing.assert_allclose(np.linalg.norm(pre, axis=1), 1.0, rtol=1e-12)

    def test_centred(self, shape_set):
        pre = kendall_preshape(shape_set).reshape(12, 6, 2)
        np.testing.assert_allclose(pre.mean(axis=1), 0.0, atol=1e-12)

    def test_invariance_to_translation_and_scale(self, shape_set):
        moved = shape_set * 7.5 + np.array([123.0, -45.0])
        np.testing.assert_allclose(kendall_preshape(moved), kendall_preshape(shape_set), atol=1e-10)

    def test_zero_size_raises(self):
        X = np.zeros((3, 4, 2))
        with pytest.raises(MorphometricsError, match="zero-size"):
            kendall_preshape(X)

    def test_wrong_ndim_raises(self):
        with pytest.raises(MorphometricsError):
            kendall_preshape(np.zeros((4, 2)))


class TestProcrustesDistance:
    def test_rotation_only_is_zero(self, shape_set):
        R = _rotation_matrix_2d(0.7)
        rotated = np.stack([cfg @ R for cfg in shape_set])
        # d² = 2 - 2·ΣS suffers ~1 ulp cancellation; the sqrt leaves ~1e-8
        for a, b in zip(shape_set, rotated):
            assert procrustes_distance(a, b) == pytest.approx(0.0, abs=1e-4)

    def test_scaled_copy_is_zero(self, shape_set):
        assert procrustes_distance(shape_set[0], shape_set[0] * 3.2 + 17.0) == pytest.approx(0.0, abs=1e-4)

    def test_positive_and_symmetric(self, shape_set):
        d1 = procrustes_distance(shape_set[0], shape_set[1])
        d2 = procrustes_distance(shape_set[1], shape_set[0])
        assert d1 > 0.01
        np.testing.assert_allclose(d1, d2, rtol=1e-8)

    def test_degenerate_raises(self):
        with pytest.raises(MorphometricsError):
            procrustes_distance(np.ones((5, 2)), np.zeros((5, 2)) + 3.0)


class TestGeometricMedian:
    def test_majority_point_1d(self):
        V = np.array([[0.0], [10.0], [10.0], [10.0]])
        np.testing.assert_allclose(geometric_median(V), [10.0], atol=1e-6)

    def test_symmetric_center(self):
        V = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        np.testing.assert_allclose(geometric_median(V), [0.0, 0.0], atol=1e-6)

    def test_robust_to_outlier_vs_mean(self):
        V = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        m = geometric_median(V)
        # median lies within the middle interval, unlike a far mean pull
        assert 1.0 <= m[0] <= 2.0 + 1e-6

    def test_empty_raises(self):
        with pytest.raises(MorphometricsError):
            geometric_median(np.zeros((0, 3)))


class TestProcrustesMedian:
    def test_rotated_copies_recover_base(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(6, 2))
        base = base - base.mean(axis=0)
        base = base / np.sqrt(np.sum(base**2))
        configs = np.stack([base @ _rotation_matrix_2d(t) for t in rng.uniform(0, 2 * np.pi, 9)])
        med = procrustes_median(configs)
        # median is a unit-norm centred shape matching base up to rotation
        assert med.shape == (6, 2)
        np.testing.assert_allclose(med.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.sum(med**2)), 1.0, rtol=1e-8)
        assert procrustes_distance(med, base) < 1e-6

    def test_single_specimen(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
        med = procrustes_median(X[None])
        np.testing.assert_allclose(med, X - X.mean(axis=0), atol=1e-12)

    def test_wrong_ndim_raises(self):
        with pytest.raises(MorphometricsError):
            procrustes_median(np.zeros((3, 2)))


class TestGoodall:
    @pytest.fixture
    def two_groups(self):
        rng = np.random.default_rng(21)
        g1 = rng.normal(scale=0.02, size=(10, 5, 2))
        g2 = g1.copy()
        g2[:, 0, 0] += 0.9  # clear shape shift in one landmark
        X = np.concatenate([g1, g2])
        labels = np.array([0] * 10 + [1] * 10)
        return X, labels

    def test_f_positive_for_separated_groups(self, two_groups):
        X, labels = two_groups
        f, between, within = goodall_f(X, labels)
        assert f > 1.0
        assert between > within

    def test_p_small_for_separated_groups(self, two_groups):
        X, labels = two_groups
        res = goodall_test(X, labels, n_permutations=199, seed=5)
        assert res.p_value <= 0.05
        assert res.n_permutations == 199

    def test_p_large_under_null(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(14, 5, 2))
        labels = np.array([0] * 7 + [1] * 7)
        res = goodall_test(X, labels, n_permutations=199, seed=9)
        assert res.p_value > 0.05

    def test_single_group_raises(self):
        X = np.zeros((4, 3, 2)) + np.arange(4)[:, None, None]
        with pytest.raises(MorphometricsError, match="two groups"):
            goodall_f(X, np.ones(4, dtype=int))

    def test_zero_within_ss_raises(self):
        X = np.concatenate([np.zeros((3, 3, 2)), np.ones((3, 3, 2))])
        labels = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(MorphometricsError, match="within-group SS"):
            goodall_f(X, labels)

    def test_shape_mismatch_raises(self):
        with pytest.raises(MorphometricsError):
            goodall_test(np.zeros((5, 3, 2)), np.zeros(4))


class TestHotellingT2:
    def test_d1_matches_squared_t_test(self):
        rng = np.random.default_rng(31)
        x1 = rng.normal(loc=0.4, size=(15, 1))
        x2 = rng.normal(loc=0.0, size=(18, 1))
        res = hotelling_t2(x1, x2)
        from scipy import stats

        t = stats.ttest_ind(x1.ravel(), x2.ravel(), equal_var=True)
        t2_expected = t.statistic**2
        assert res.t2 == pytest.approx(float(t2_expected), rel=1e-8)
        assert res.p_value == pytest.approx(float(t.pvalue), rel=1e-6)

    def test_identical_samples_give_zero(self):
        rng = np.random.default_rng(32)
        x = rng.normal(size=(10, 3))
        res = hotelling_t2(x, x.copy())
        assert res.t2 == pytest.approx(0.0, abs=1e-12)
        assert res.p_value > 0.9

    def test_shift_detected(self):
        rng = np.random.default_rng(33)
        x1 = rng.normal(size=(20, 2)) + 0.8
        x2 = rng.normal(size=(20, 2))
        res = hotelling_t2(x1, x2)
        assert res.p_value < 0.001
        np.testing.assert_allclose(res.mean_difference, [0.8, 0.8], atol=0.3)

    def test_df_check(self):
        rng = np.random.default_rng(34)
        res = hotelling_t2(rng.normal(size=(8, 2)), rng.normal(size=(12, 2)))
        assert res.df1 == 2
        assert res.df2 == 8 + 12 - 2 - 1

    def test_too_few_samples_raises(self):
        with pytest.raises(MorphometricsError):
            hotelling_t2(np.zeros((1, 2)), np.zeros((1, 2)))

    def test_dim_mismatch_raises(self):
        with pytest.raises(MorphometricsError):
            hotelling_t2(np.zeros((5, 2)), np.zeros((5, 3)))
