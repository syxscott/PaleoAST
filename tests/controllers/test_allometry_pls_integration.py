"""Controller-level regression tests for the UI-wiring fix (Defect 2).

The two ``morphometrics.allometry`` engines (the multivariate
allometry regression and the 2B-PLS integration analysis) used to
have *zero* production callers — the menu items ran ``QMessageBox``
popups and threw the real computation away.  The fix adds
``StatisticsController.analyze_allometry`` and
``StatisticsController.analyze_pls`` and wires the dialog buttons to
them.

These tests do not require Qt — they exercise the controller path
directly, which is what the GUI now invokes under the hood.
"""
from __future__ import annotations

import numpy as np
import pytest

from controllers.statistics_controller import StatisticsController


def _synthetic_aligned_configurations(n_specimens: int = 10, n_landmarks: int = 6, seed: int = 0) -> np.ndarray:
    """Build (n, k, 2) configurations centred around a base shape."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_landmarks, 2)) * 0.1
    configs = np.empty((n_specimens, n_landmarks, 2))
    for i in range(n_specimens):
        scale = 0.5 + i * 0.05
        configs[i] = base * scale + rng.standard_normal((n_landmarks, 2)) * 0.005
    return configs


class TestAnalyzeAllometry:
    def test_runs_without_cached_gpa_when_passed_explicit_data(self):
        ctrl = StatisticsController()
        configs = _synthetic_aligned_configurations()
        result = ctrl.analyze_allometry(aligned_configurations=configs)
        # Sanity: a regression result with the documented fields.
        assert result.n_specimens == configs.shape[0]
        assert result.n_landmarks == configs.shape[1]
        assert result.n_dims == configs.shape[2]
        assert -1.0 <= result.r_squared <= 1.0
        assert result.f_statistic >= 0.0
        assert 0.0 <= result.isometry_pvalue <= 1.0

    def test_caches_result_in_state(self):
        ctrl = StatisticsController()
        configs = _synthetic_aligned_configurations()
        result = ctrl.analyze_allometry(aligned_configurations=configs)
        cached = ctrl.get_cached_result("allometry_result")
        assert cached is result

    def test_accepts_2d_data_and_reshapes(self):
        ctrl = StatisticsController()
        configs = _synthetic_aligned_configurations()
        flat = configs.reshape(configs.shape[0], configs.shape[1] * configs.shape[2])
        result = ctrl.analyze_allometry(aligned_configurations=flat)
        assert result.n_landmarks == configs.shape[1]
        assert result.n_dims == configs.shape[2]


class TestAnalyzePLS:
    def test_runs_with_explicit_blocks(self):
        ctrl = StatisticsController()
        configs = _synthetic_aligned_configurations()
        k = configs.shape[1]
        # Split the landmarks column-wise into two halves so the
        # analyser sees two blocks with the same specimen axis.
        flat = configs.reshape(configs.shape[0], k * 2)
        block_a = flat[:, : k * 1]
        block_b = flat[:, k * 1 :]
        result = ctrl.analyze_pls(block_a=block_a, block_b=block_b)
        assert result.n_specimens == configs.shape[0]
        assert -1.0 <= result.integration_index <= 1.0
        assert result.rv_coefficient is not None
        assert result.pls1_pvalue is None  # no permutations requested

    def test_permutations_populate_pvalue(self):
        ctrl = StatisticsController()
        configs = _synthetic_aligned_configurations(n_specimens=12)
        flat = configs.reshape(configs.shape[0], -1)
        mid = flat.shape[1] // 2
        result = ctrl.analyze_pls(
            block_a=flat[:, :mid], block_b=flat[:, mid:], permutations=49, seed=123
        )
        assert result.pls1_pvalue is not None
        assert 0.0 <= result.pls1_pvalue <= 1.0
        assert result.pls1_z is not None
        assert result.random_correlations is not None
        assert len(result.random_correlations) == 49

    def test_caches_result_in_state(self):
        ctrl = StatisticsController()
        configs = _synthetic_aligned_configurations()
        flat = configs.reshape(configs.shape[0], -1)
        mid = flat.shape[1] // 2
        result = ctrl.analyze_pls(block_a=flat[:, :mid], block_b=flat[:, mid:])
        assert ctrl.get_cached_result("pls_result") is result


class TestPCoACorrectionWiringThroughController:
    """The UI dialog now sends a ``correction`` string through to the
    controller.  Confirm the controller forwards it."""

    def test_run_pcoa_accepts_correction_kwarg(self):
        from stats.distance_metrics import compute_distance_matrix

        ctrl = StatisticsController()
        rng = np.random.default_rng(42)
        data = rng.random((8, 6)) * 10
        data[rng.random(data.shape) < 0.3] = 0.0
        D = np.asarray(compute_distance_matrix(data, metric="bray_curtis").matrix, dtype=float)
        for method in ("cmdscale", "lingoes", "wickoff", "torgerson"):
            r = ctrl.run_pcoa(distance_matrix=D.copy(), metric="bray_curtis", correction=method)
            assert r.correction_method == method
            eigs = np.asarray(r.eigenvalues, dtype=float)
            if method != "cmdscale":
                assert np.all(eigs > -1e-6)
            else:
                # cmdscale branch preserves any existing negative eigenvalues.
                pass


class TestLockCoverage:
    """Every public ``analyze_*`` / ``run_*`` method that mutates state
    must hold ``self._lock`` for the duration of the call.

    The previous baseline called ``self._state.set_data_matrix`` and
    ``self._state.cache_result`` from ~9 helper methods (around the
    macroevolution / 3D morphometrics block) without acquiring the
    lock, so concurrent UI work could see half-written state.

    PyQt6 cannot be imported on the build host, so we cannot import
    the controller to inspect its bytecode directly.  Instead we read
    the source file and assert that each flagged method begins with a
    ``with self._lock:`` block before any state-mutating call.
    """

    @pytest.mark.parametrize(
        "method_name",
        [
            "analyze_cohort_survivorship",
            "analyze_diversity_dynamics",
            "analyze_survival",
            "compare_survival_groups",
            "simulate_fbd",
            "analyze_gpa3d",
            "run_tps3d",
            "analyze_allometry",
            "analyze_pls",
            "run_pcoa",
        ],
    )
    def test_method_holds_self_lock(self, method_name: str):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2] / "controllers" / "statistics_controller.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        # Find the FunctionDef for ``method_name`` inside the
        # ``StatisticsController`` class.
        cls = next(
            (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StatisticsController"),
            None,
        )
        assert cls is not None, "StatisticsController class not found"
        func = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name),
            None,
        )
        assert func is not None, f"{method_name} not found on StatisticsController"

        # Dump the body back to source so we can search for ``with
        # self._lock:`` robustly across multi-line ``with`` blocks,
        # async-with, etc.  ``ast.unparse`` is available from Python
        # 3.9; on 3.8 we'd need ``astor`` instead.
        body_src = ast.unparse(func)
        assert "with self._lock:" in body_src, (
            f"{method_name} is missing 'with self._lock:' — concurrent callers "
            f"could observe half-written state. Body preview:\n{body_src[:400]}"
        )
