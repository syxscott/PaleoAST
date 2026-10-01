"""
Ground truth for ``stratigraphy/arma.py``, which carried 26.9% line coverage.

ARMA fitting is the numeric core of cyclostratigraphic periodogram analysis --
a real palaeontological workflow. A wrong estimate does not crash: it produces
a plausible-looking spectrum, and the researcher has no way to tell. So the
model is pinned against series generated from KNOWN parameters.

Two defects were found this way:

1. Without statsmodels installed (it is an optional dependency), the manual
   fallback set ``ma_params = np.zeros(q)`` and logged a warning nobody
   sees. ``summary()`` then printed ``MA coefficients: [0.0]``, which reads
   as "measured, and it is zero" rather than "never estimated". A MA(1) fit
   to a series with a real moving-average term returned "no MA structure".
   The fallback now estimates theta by Hannan-Rissanen.
2. A constant series fits exactly, so the residual variance is 0 and
   ``n*ln(0)`` made AIC and BIC **-inf** -- not a score, a sign that model
   ordering is meaningless. They are now NaN, with an explanation.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratigraphy.arma import ARMAAnalyzer

SEED = 7
N = 4000


def _ar1(phi: float, n: int = N, rng=None) -> np.ndarray:
    """A series with a known AR(1) coefficient."""
    rng = rng or np.random.default_rng(SEED)
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _ma1(theta: float, n: int = N, rng=None) -> np.ndarray:
    """A series with a known MA(1) coefficient."""
    rng = rng or np.random.default_rng(SEED)
    e = rng.normal(0, 1, n + 1)
    return np.array([e[i] + theta * e[i - 1] for i in range(n)])


def _times(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


@pytest.fixture(scope="module")
def analyzer() -> ARMAAnalyzer:
    return ARMAAnalyzer()


def _first(values) -> float | None:
    arr = np.ravel(np.asarray(values, dtype=float))
    return float(arr[0]) if arr.size else None


class TestParameterRecovery:
    def test_ar1_recovers_phi(self, analyzer):
        result = analyzer.fit(_times(N), _ar1(0.7), p=1, q=0)
        phi = _first(result.ar_params)
        assert phi is not None
        assert phi == pytest.approx(0.7, abs=0.08), f"recovered phi={phi}, true 0.7"

    def test_ma1_recovers_theta(self, analyzer):
        """The regression: the fallback used to return zero MA coefficients.

        ``MA coefficients: [0.0]`` in the summary reads as a measurement; it
        was in fact an absence of one.
        """
        result = analyzer.fit(_times(N), _ma1(0.5), p=0, q=1)
        theta = _first(result.ma_params)
        assert theta is not None, "no MA coefficient returned at all"
        assert abs(abs(theta) - 0.5) < 0.15, f"recovered theta={theta}, true 0.5"

    def test_ar1_fit_does_not_invent_ma_structure(self, analyzer):
        result = analyzer.fit(_times(N), _ar1(0.7), p=1, q=1)
        theta = _first(result.ma_params)
        assert theta is None or abs(theta) < 0.15, (
            f"a pure AR(1) series produced theta={theta}"
        )

    def test_fit_is_deterministic(self, analyzer):
        series = _ar1(0.7)
        first_fit = analyzer.fit(_times(N), series, p=1, q=1)
        second_fit = analyzer.fit(_times(N), series, p=1, q=1)
        assert np.allclose(first_fit.ar_params, second_fit.ar_params)
        assert np.allclose(first_fit.ma_params, second_fit.ma_params)
        assert first_fit.aic == second_fit.aic


class TestInformationCriteria:
    def test_aic_bic_finite_on_a_real_fit(self, analyzer):
        result = analyzer.fit(_times(N), _ar1(0.7), p=1, q=0)
        assert np.isfinite(result.aic), "AIC is not finite on a genuine fit"
        assert np.isfinite(result.bic), "BIC is not finite on a genuine fit"

    def test_aic_bic_not_infinite_on_a_constant_series(self, analyzer):
        """A constant series fits exactly, so sigma^2 = 0 and ln(0) = -inf.

        -inf is not a score: it asserts that every other model is infinitely
        worse, which is a statement about the degeneracy rather than about
        the fit, and it silently breaks any ranking that consumes it. NaN
        (reported, with a log line) is the honest answer.
        """
        for label, data in (("zeros", np.zeros(200)), ("constant", np.full(200, 3.0))):
            result = analyzer.fit(_times(200), data, p=1, q=1)
            assert not np.isinf(result.aic), f"{label}: AIC = {result.aic}"
            assert not np.isinf(result.bic), f"{label}: BIC = {result.bic}"

    def test_bic_penalises_more_parameters(self, analyzer):
        """BIC's whole point is the extra penalty; if it were inverted the
        model ranking would prefer the over-fitted model."""
        series = _ar1(0.7)
        small = analyzer.fit(_times(N), series, p=1, q=0)
        large = analyzer.fit(_times(N), series, p=3, q=2)
        # Both fit the same series; the larger model's BIC must not be lower
        # just because it has more parameters.
        assert large.bic > small.bic - 1e-9 or np.isnan(large.bic)


class TestDegenerateInputs:
    @pytest.mark.parametrize(
        "label,data",
        [
            ("all zeros", np.zeros(200)),
            ("constant", np.full(200, 3.0)),
            ("two points", np.array([1.0, 2.0])),
            ("single point", np.array([1.0])),
        ],
    )
    def test_degenerate_input_never_yields_nan_coefficients(self, analyzer, label, data):
        """Either a clear error, or a result with finite coefficients.

        Not NaN: a NaN AR/MA coefficient flows straight into a periodogram
        and the user sees an empty or nonsensical spectrum.
        """
        from utils.exceptions import ComputationError, DataValidationError

        try:
            result = analyzer.fit(_times(len(data)), data, p=1, q=1)
        except (ComputationError, DataValidationError, ValueError):
            return  # refused clearly, which is fine
        for name in ("ar_params", "ma_params", "residuals", "predicted"):
            values = np.asarray(getattr(result, name), dtype=float)
            assert np.all(np.isfinite(values)), f"{label}: {name} has non-finite values"

    def test_nan_input_is_refused(self, analyzer):
        from utils.exceptions import DataValidationError

        with pytest.raises((DataValidationError, ValueError)):
            analyzer.fit(
                _times(5), np.array([1.0, np.nan, 3.0, 4.0, 5.0]), p=1, q=1
            )

    def test_too_few_observations_is_refused(self, analyzer):
        from utils.exceptions import ComputationError

        with pytest.raises((ComputationError, ValueError)):
            analyzer.fit(_times(2), np.array([1.0, 2.0]), p=1, q=1)


class TestOrderSelection:
    def test_cross_validate_returns_scores_per_order(self, analyzer):
        report = analyzer.cross_validate(_times(1000), _ar1(0.7)[:1000], max_p=2, max_q=1)
        assert set(report) >= {"best_order", "best_aic", "aic_table", "bic_table"}
        # Every order in 0..max_p x 0..max_q must be reachable: p=0 and q=0
        # used to be skipped, which forced a spurious MA term onto every
        # pure-AR series.
        expected = {(p, q) for p in range(0, 3) for q in range(0, 2)}
        reachable = set(report["aic_table"])
        assert (0, 0) in reachable, f"white noise is unreachable: {sorted(reachable)}"
        assert (1, 0) in reachable, f"a pure AR model is unreachable: {sorted(reachable)}"
        assert expected - reachable == set(), (
            f"orders never evaluated: {sorted(expected - reachable)}"
        )

    def test_cross_validate_does_not_invent_ma_structure(self, analyzer):
        """A strong AR(1) series must not come back with a moving-average term.

        The old loop started q at 1, so q=0 was unreachable and the search
        had to add a spurious MA coefficient to win.
        """
        report = analyzer.cross_validate(_times(N), _ar1(0.9), max_p=3, max_q=2)
        best = report["best_order"]
        assert best[1] == 0, (
            f"an AR(0.9) series was given a moving-average term: {best} "
            f"({report['aic_table']})"
        )
