"""
Regression tests for defect 5: evolution-rate bootstrap was non-reproducible
and used a model-agnostic residualisation that broke the directional / stasis
bootstrap CIs.

The original implementation called ``np.random.choice`` (the global
numpy random stream) inside the bootstrap loop, so:
* the CIs differed every run with no way to reproduce them;
* CIs for the directional / stasis models were meaningless because the
  residuals were always the random-walk first-difference residuals (the
  code's own comment admitted this).

The fix:

* Adds a ``seed`` parameter and uses a local ``np.random.default_rng``.
* Computes **model-specific** residuals:
  - random_walk : first-difference residuals
    (dx - mean(dx))
  - directional : residuals around the fitted linear trend
    (dx - beta * dt)
  - stasis : residuals around the OU optimum
    (dx + alpha * (x[:-1] - theta) * dt — the same expression the
    likelihood uses).

These tests verify:

* same seed → bit-identical CIs across two runs;
* different seeds → different CIs of comparable magnitude;
* CIs are finite and well-defined for all three models.
"""

import numpy as np
import pytest

from morphometrics.evolution_rate import EvolutionRateAnalyzer


def _trend_series(seed: int = 5, n: int = 40, trend: float = 0.4):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.5, 1.5, size=n - 1))
    base = rng.normal(size=n)
    base += trend * np.arange(n)  # clear directional trend
    return base, times


def _stasis_series(seed: int = 7, n: int = 40, theta: float = 2.0):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.5, 1.5, size=n - 1))
    # OU simulation around theta
    x = np.zeros(n)
    x[0] = theta
    alpha, sigma = 0.5, 0.3
    dt = times[0]
    for i in range(1, n):
        if i - 1 < len(times):
            dt = times[i - 1]
        x[i] = x[i - 1] + alpha * (theta - x[i - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return x, times


class TestBootstrapReproducibility:
    def test_same_seed_bit_identical(self):
        trait, times = _trend_series()
        # Fit model first to know the best model
        analyzer1 = EvolutionRateAnalyzer()
        r1 = analyzer1.analyze(trait, time_intervals=times, confidence_level=0.95)
        # Now bootstrap with explicit seed
        ci_lower1, ci_upper1 = analyzer1._bootstrap_rate_ci(
            trait,
            times,
            r1.best_model,
            confidence_level=0.95,
            n_bootstrap=99,
            seed=2026,
        )

        analyzer2 = EvolutionRateAnalyzer()
        r2 = analyzer2.analyze(trait, time_intervals=times, confidence_level=0.95)
        ci_lower2, ci_upper2 = analyzer2._bootstrap_rate_ci(
            trait,
            times,
            r2.best_model,
            confidence_level=0.95,
            n_bootstrap=99,
            seed=2026,
        )
        # Bit-identical
        assert ci_lower1 == ci_lower2
        assert ci_upper1 == ci_upper2

    def test_different_seeds_different_results(self):
        trait, times = _trend_series()
        analyzer = EvolutionRateAnalyzer()
        model = analyzer.analyze(trait, time_intervals=times).best_model
        cl1, cu1 = analyzer._bootstrap_rate_ci(trait, times, model, 0.95, 99, seed=1)
        cl2, cu2 = analyzer._bootstrap_rate_ci(trait, times, model, 0.95, 99, seed=2)
        # Different seeds MUST produce different CIs.
        assert (cl1, cu1) != (cl2, cu2)
        # …but the magnitudes are of comparable order (within 10x).
        if cl1 != 0.0 and cl2 != 0.0:
            ratio = max(abs(cl1), abs(cl2)) / min(abs(cl1), abs(cl2))
            assert ratio < 10.0


class TestBootstrapAllModelsFinite:
    """The original code's CIs were undefined for the directional/stasis
    models because the residuals were random-walk-specific.  After the
    fix, all three models produce finite CIs."""

    @pytest.mark.parametrize("model", ["random_walk", "directional", "stasis"])
    def test_bootstrap_returns_finite_ci(self, model):
        if model == "stasis":
            trait, times = _stasis_series()
        else:
            trait, times = _trend_series()
        analyzer = EvolutionRateAnalyzer()
        cl, cu = analyzer._bootstrap_rate_ci(trait, times, model, 0.95, n_bootstrap=49, seed=99)
        assert cl is not None and cu is not None
        assert np.isfinite(cl) and np.isfinite(cu)
        assert cl <= cu


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
