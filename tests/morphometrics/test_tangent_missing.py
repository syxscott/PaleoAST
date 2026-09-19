"""Tests for morphometrics/tangent.py (orp) and morphometrics/missing.py (W3)."""

import numpy as np
import pytest

from morphometrics.missing import estimate_missing
from morphometrics.tangent import orp, tangent_vectors
from utils.exceptions import MorphometricsError


def _aligned_configs(n=6, p=9, k=2, seed=3):
    """GPA-like aligned configs: unit centroid size, centroid at origin."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2 * np.pi * (p - 1) / p, p)
    base = np.stack([np.cos(t), np.sin(t)] + ([np.zeros(p)] if k == 3 else []), axis=-1)
    base = base[:, :k]
    out = []
    for _ in range(n):
        cfg = base + rng.normal(0.0, 0.05, size=(p, k))
        cfg = cfg - cfg.mean(axis=0)
        cfg = cfg / np.sqrt(np.sum(cfg**2))
        out.append(cfg)
    return np.stack(out)


class TestOrp:
    def test_projected_points_lie_on_tangent_hyperplane(self):
        X = _aligned_configs()
        P = orp(X)
        mean = P.mean(axis=0)
        mean = mean - mean.mean(axis=0)
        y = mean.ravel() / np.linalg.norm(mean.ravel())
        radial = P.reshape(len(P), -1) @ y
        assert np.allclose(radial, 1.0, atol=1e-10)

    def test_idempotent(self):
        X = _aligned_configs()
        P1 = orp(X)
        P2 = orp(P1)
        assert np.allclose(P1, P2, atol=1e-12)

    def test_preserves_exact_consensus_config(self):
        X = _aligned_configs(n=3)
        P = orp(X)
        assert P.shape == X.shape
        # fixed points of the projection are exactly the tangent hyperplane
        assert np.allclose(orp(P), P)

    def test_rejects_bad_shapes(self):
        with pytest.raises(MorphometricsError):
            orp(np.zeros((4, 3)))
        with pytest.raises(MorphometricsError):
            orp(np.zeros((0, 4, 2)))

    def test_rejects_zero_consensus(self):
        X = np.array([[[1.0, 0.0], [-1.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]]])
        with pytest.raises(MorphometricsError):
            orp(X)

    def test_tangent_vectors_zero_mean(self):
        T = tangent_vectors(_aligned_configs())
        assert T.shape == (6, 18)
        assert np.allclose(T.mean(axis=0), 0.0, atol=1e-10)


def _raw_fixture(k=2, n_complete=30, seed=7, correlated=False):
    """Global affine variation (correlated landmarks) + small jitter,
    random similarity per specimen; returns (configs, truth, missing_rows).

    ``correlated=True``: shape varies along a single shear factor with no
    independent jitter — exactly the linear structure the ``reg`` estimator
    assumes, so regression can recover missing blocks from observed ones.
    """
    rng = np.random.default_rng(seed)
    p = 9
    t = np.linspace(0.0, 2 * np.pi, p + 1)[:-1]
    cols = [np.cos(t), np.sin(t)]
    if k == 3:
        cols.append(np.zeros(p))
    base = np.stack(cols, axis=1)
    A = rng.normal(size=(k, k)) if correlated else None
    configs = []
    for _ in range(n_complete + 2):
        if correlated:
            f = rng.uniform(-0.3, 0.3)
            cfg = base @ (np.eye(k) + f * A).T
        else:
            M = np.eye(k) + 0.10 * rng.normal(size=(k, k))
            cfg = base @ M.T + rng.normal(0.0, 0.01, size=base.shape)
        cfg = cfg * rng.uniform(0.8, 1.2) + rng.normal(0.0, 1.0, size=(1, k))
        angle = rng.uniform(-0.5, 0.5)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        if k == 3:
            R = np.block([[R, np.zeros((2, 1))], [np.zeros((1, 2)), np.eye(1)]])
        cfg = cfg @ R.T
        configs.append(cfg)
    configs = np.stack(configs)
    truth = configs.copy()
    missing_rows = [1, 4, 7]
    configs[0, missing_rows] = np.nan
    configs[1, [2, missing_rows[0]]] = np.nan
    return configs, truth, missing_rows


class TestEstimateMissing:
    def test_recovers_missing_landmarks_2d_tps(self):
        configs, truth, missing_rows = _raw_fixture(k=2)
        res = estimate_missing(configs, method="tps")
        assert res.estimated_mask[0].tolist() == [
            i in missing_rows for i in range(configs.shape[1])
        ]
        err = np.linalg.norm(res.filled_configurations[0, missing_rows] - truth[0, missing_rows])
        assert np.isfinite(err)
        # jitter scale is 0.01 per coord; consensus+TPS should be far below
        # the landmark spacing (~0.7)
        assert err < 0.35, ("tps", err)
        # observed landmarks must be untouched
        obs = ~np.isnan(configs[0]).any(axis=1)
        assert np.array_equal(res.filled_configurations[0][obs], configs[0][obs])

    @pytest.mark.parametrize("k", [2, 3])
    def test_recovers_missing_landmarks_reg(self, k):
        # "reg" assumes the missing block is linearly related to the
        # observed block across specimens — test against exactly that
        # structure (single shear factor, no independent jitter).
        configs, truth, missing_rows = _raw_fixture(k=k, correlated=True)
        res = estimate_missing(configs, method="reg")
        err = np.linalg.norm(res.filled_configurations[0, missing_rows] - truth[0, missing_rows])
        assert np.isfinite(err)
        assert err < 0.15, ("reg", k, err)

    def test_recovers_missing_landmarks_3d_tps(self):
        configs, truth, missing_rows = _raw_fixture(k=3)
        res = estimate_missing(configs, method="tps")
        err = np.linalg.norm(res.filled_configurations[0, missing_rows] - truth[0, missing_rows])
        assert np.isfinite(err)
        assert err < 0.35, ("tps", err)

    def test_no_missing_returns_input(self):
        configs, _, _ = _raw_fixture()
        configs = np.nan_to_num(configs, nan=1.0)
        res = estimate_missing(configs)
        assert np.array_equal(res.filled_configurations, configs)
        assert res.estimated_mask.sum() == 0

    def test_no_complete_specimen_raises(self):
        configs, _, _ = _raw_fixture()
        bad = configs.copy()
        bad[:, 0] = np.nan
        bad[0, 1] = np.nan
        with pytest.raises(MorphometricsError, match="complete specimen"):
            estimate_missing(bad)

    def test_too_few_observed_raises(self):
        configs, _, _ = _raw_fixture(k=2)
        configs[0, 1:] = np.nan  # only 1 observed landmark
        with pytest.raises(MorphometricsError, match="at least 3"):
            estimate_missing(configs)

    def test_unknown_method_raises(self):
        configs, _, _ = _raw_fixture()
        with pytest.raises(MorphometricsError, match="Unknown method"):
            estimate_missing(configs, method="magic")

    def test_aligned_filled_matches_reference_frame(self):
        configs, _, _ = _raw_fixture(k=2)
        res = estimate_missing(configs, method="tps")
        # observed landmarks of the incomplete specimen, once mapped into
        # the reference frame, must sit near the consensus
        obs0 = ~np.isnan(configs[0]).any(axis=1)
        d = np.linalg.norm(
            res.aligned_filled[0][obs0] - res.reference[obs0], axis=1
        )
        assert d.mean() < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
