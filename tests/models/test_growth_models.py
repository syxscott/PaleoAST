# =============================================================================
# FILE: tests/models/test_growth_models.py
# =============================================================================
"""
Tests for the growth-curve models.

The core property is **recovery of ground truth**: data are generated from
known parameters, noise is added, and the fit has to give the parameters back.
That is the only assertion here that actually tests a growth model -- an R^2
of 0.999 says the curve is shaped roughly right, not that its asymptote,
rate constant and origin age are the ones that generated the data. A module
that fitted every series with a straight line through the middle would score
well on fit-quality assertions and fail this one immediately.

The properties asserted here are chosen so that a broken implementation
fails rather than merely returning a different number:

  * each of the six models recovers its own generating parameters, over
    several noise draws so the tolerance cannot have been tuned to one of
    them;
  * every tolerance is tied to the *standard error of the fitted parameter*,
    not to whatever number the implementation happened to produce (see
    ``_tolerance``), so a start-value heuristic that stops working shows up
    as a failed recovery rather than as a loosened bound;
  * ``predict`` returns the curve that was fitted, not a fresh guess, and
    extrapolates beyond the measured range;
  * the model that generated a series wins the AICc ranking by a margin far
    larger than the ~2 units Burnham & Anderson call "substantial";
  * degenerate input raises PaleoAST's own exception types, and never a bare
    ``ValueError`` that the rest of the application does not catch;
  * the fit is deterministic: it draws no random numbers, so refitting the
    same data reproduces the first fit exactly rather than approximately.
"""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pytest

from models.growth_models import (
    GROWTH_MODELS,
    GrowthModelAnalyzer,
    gaussian_growth,
    gompertz,
    logistic,
    michaelis_menten,
    resolve_model_name,
    sinusoidal,
    von_bertalanffy,
)
from utils.exceptions import DataValidationError, ValidationError
from utils.statistics_core import make_rng

# =============================================================================
# Ground truth
# =============================================================================

# Public curve function and parameter order for each model key. Written out
# rather than imported from the module's private registry, so the tests
# generate data through the same public entry point a user would.
_CURVES: dict[str, tuple[object, tuple[str, ...]]] = {
    "von_bertalanffy": (von_bertalanffy, ("linf", "b", "t0")),
    "gompertz": (gompertz, ("linf", "b", "t0")),
    "michaelis_menten": (michaelis_menten, ("K", "Km")),
    "logistic": (logistic, ("K", "b", "t0")),
    "gaussian": (gaussian_growth, ("K", "b", "t0")),
    "sinusoidal": (sinusoidal, ("mean", "amplitude", "period", "t0")),
}

# model -> (true parameters, time axis, noise standard deviation)
#
# The time axes are chosen so that the measured series is physically sensible:
#   * von Bertalanffy starts at t = 2.0 because the curve is zero at t0 and
#     negative before it, so a series measured from t = 0 would contain
#     negative "shell lengths" -- which the module rejects, correctly;
#   * the saturating curves span 40+ points over a window twice the width of
#     their sigmoid, so the plateau is visible and the asymptote is
#     identifiable rather than an extrapolation;
#   * noise is 1 unit on a 100-unit asymptote (1 %), the ratio a measured
#     shell length series typically carries, and 0.5 units on the sinusoid
#     whose amplitude is 10.
_GROUND_TRUTH: dict[str, tuple[dict[str, float], npt.NDArray, float]] = {
    "von_bertalanffy": ({"linf": 100.0, "b": 0.15, "t0": 1.0}, np.linspace(2.0, 21.0, 41), 1.0),
    "gompertz": ({"linf": 100.0, "b": 0.18, "t0": 2.0}, np.linspace(0.0, 20.0, 41), 1.0),
    "michaelis_menten": ({"K": 100.0, "Km": 6.0}, np.linspace(0.5, 25.0, 41), 1.0),
    "logistic": ({"K": 100.0, "b": 0.25, "t0": 8.0}, np.linspace(0.0, 20.0, 41), 1.0),
    "gaussian": ({"K": 100.0, "b": 0.20, "t0": 9.0}, np.linspace(0.0, 20.0, 41), 1.0),
    "sinusoidal": (
        {"mean": 50.0, "amplitude": 10.0, "period": 8.0, "t0": 1.5},
        np.linspace(0.0, 20.0, 41),
        0.5,
    ),
}

# --- tolerances, and where each one comes from -------------------------------
#
# Each is several standard errors of the parameter it guards, measured from
# the fits this very module produces (the standard errors come back in
# ``result.fit.stderr``). Using the sampling error as the yardstick is what
# makes these numbers defensible rather than arbitrary: a tolerance far
# tighter than the standard error would fail on noise alone, and one far
# looser would not notice a wrong curve.

_SCALE_REL = 0.05
"""Asymptotes and level (``linf``, ``K``, ``mean``), relative.

The relative standard error of a fitted asymptote on 41 points at 1 % noise
is about 0.6 % (``linf`` = 100 +/- 0.6). 5 % is ~8 standard errors, and 5 % of
a shell length is far below the precision anyone reports one to.
"""

_RATE_REL = 0.15
"""Rate constants (``b``, ``Km``), relative.

The rate constant is the worst-determined parameter of the saturating curves
because it trades off against the asymptote: the whole family of nearby
(``linf``, ``b``) pairs describes almost the same early curve. On the von
Bertalanffy fit below the relative standard error of ``b`` is 2.1 %
(0.0032 on 0.15), and Michaelis-Menten's ``Km`` is similar at 2.2 %. 15 % is
~7 standard errors -- loose enough not to fail on noise, far tighter than the
distance between the true rate and the rate of a neighbouring model.
"""

_TIME_SPAN_FRACTION = 0.02
"""Age offsets (``t0``), as a fraction of the observed time span.

The standard error of ``t0`` is about 0.3 % of the span (0.056 on a 19-unit
window), so 2 % of the span is ~6 standard errors and a fifteenth of the
window: a curve placed in the wrong part of the measured range fails here,
while noise alone does not.
"""

_PERIOD_REL = 0.05
"""Sinusoidal period, relative.

Its standard error on 2.5 cycles in the window is 0.5 % (0.04 on 8.0), so 5 %
is ~10 standard errors. Period is the parameter a grid-search start value is
most likely to get wrong, so this is the one that would notice.
"""

_AMPLITUDE_REL = 0.10
"""Sinusoidal amplitude, relative.

Mean and amplitude are strongly anti-correlated whenever the window does not
hold a whole number of cycles, which inflates the standard error of the
amplitude to 2.2 % (0.22 on 10.0). 10 % is ~4.5 standard errors, and still a
tenth of the amplitude itself.
"""


def _synthetic(model: str, seed: int = 42) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate ``(times, values)`` from a known curve plus seeded noise.

    The generator is an isolated ``default_rng`` from ``make_rng``, so the
    noise is reproducible from the seed and nothing here touches the legacy
    global ``numpy.random`` stream that other tests may be drawing from.
    """
    curve, names = _CURVES[model]
    truth, times, sigma = _GROUND_TRUTH[model]
    clean = curve(times, *[truth[name] for name in names])
    rng = make_rng(seed, context="growth model test data")
    return times, clean + rng.normal(0.0, sigma, times.shape)


def _tolerance(model: str, name: str, times: npt.NDArray) -> dict[str, float]:
    """Comparison tolerance for one parameter, chosen by what it measures."""
    if name == "t0":
        return {"abs": _TIME_SPAN_FRACTION * float(np.ptp(times))}
    if name == "period":
        return {"rel": _PERIOD_REL}
    if name == "amplitude":
        return {"rel": _AMPLITUDE_REL}
    if name in ("b", "Km"):
        return {"rel": _RATE_REL}
    return {"rel": _SCALE_REL}


@pytest.fixture
def analyzer() -> GrowthModelAnalyzer:
    """A fresh analyzer."""
    return GrowthModelAnalyzer()


# =============================================================================
# Ground-truth recovery -- the check with teeth
# =============================================================================


@pytest.mark.parametrize("model", sorted(_GROUND_TRUTH))
def test_recovers_known_parameters_from_noisy_data(model: str, analyzer: GrowthModelAnalyzer) -> None:
    """Each model gives back the parameters that generated the data.

    The curve functions themselves are used to generate the series, so a
    wrong formula on both sides would still pass; what this pins down is that
    the *fitted* parameter values match the generating ones, which is the
    only route by which the fit, the parameterisation and the reported names
    are all correct at once.
    """
    times, values = _synthetic(model)
    result = analyzer.fit(model, times, values)

    assert result.success
    assert result.model == model
    assert result.param_names == _CURVES[model][1]
    truth = _GROUND_TRUTH[model][0]
    for name in result.param_names:
        assert result.param(name) == pytest.approx(truth[name], **_tolerance(model, name, times)), (
            f"{model}: {name} recovered as {result.param(name):.6g}, generated value {truth[name]:.6g}"
        )
    # A recovery this good should also leave a residual at the noise level,
    # not merely a high R-squared.
    assert result.rmse == pytest.approx(_GROUND_TRUTH[model][2], rel=0.35)
    assert result.r_squared > 0.99


@pytest.mark.parametrize("model", sorted(_GROUND_TRUTH))
@pytest.mark.parametrize("seed", [1, 7, 2024])
def test_recovery_survives_any_noise_draw(model: str, seed: int, analyzer: GrowthModelAnalyzer) -> None:
    """The tolerances are not tuned to one lucky noise realisation.

    Three further seeds per model: a start-value heuristic that only works for
    a particular draw -- the argmax of an unsmoothed derivative landing on a
    noise spike, say -- passes a single-seed test and fails this one.
    """
    times, values = _synthetic(model, seed=seed)
    result = analyzer.fit(model, times, values)
    truth = _GROUND_TRUTH[model][0]
    for name in result.param_names:
        assert result.param(name) == pytest.approx(truth[name], **_tolerance(model, name, times)), (
            f"{model} (seed {seed}): {name} recovered as {result.param(name):.6g}, generated value {truth[name]:.6g}"
        )


def test_explicit_start_values_are_used(analyzer: GrowthModelAnalyzer) -> None:
    """A caller-supplied ``p0`` of the true parameters fits with no search."""
    times, values = _synthetic("von_bertalanffy")
    result = analyzer.fit("von_bertalanffy", times, values, p0=[100.0, 0.15, 1.0])
    assert result.param("linf") == pytest.approx(100.0, rel=_SCALE_REL)
    assert result.param("b") == pytest.approx(0.15, rel=_RATE_REL)
    assert result.param("t0") == pytest.approx(1.0, abs=_TIME_SPAN_FRACTION * float(np.ptp(times)))


def test_michaelis_menten_fits_a_series_that_starts_at_zero(analyzer: GrowthModelAnalyzer) -> None:
    """A series beginning at ``t = 0`` cannot be inverted for the linear start.

    The Lineweaver-Burk starting-value search needs ``1/t`` and ``1/L``, so a
    zero at the origin removes its first point. The fallback must carry the
    fit, because "the first measurement is zero" is the normal case for this
    model, not an edge case.
    """
    times = np.linspace(0.0, 25.0, 41)
    values = michaelis_menten(times, 100.0, 6.0)
    rng = make_rng(3, context="growth model test data")
    values = values + rng.normal(0.0, 1.0, times.shape)
    result = analyzer.fit("michaelis_menten", times, values)
    assert result.param("K") == pytest.approx(100.0, rel=_SCALE_REL)
    assert result.param("Km") == pytest.approx(6.0, rel=_RATE_REL)


# =============================================================================
# Degenerate input
# =============================================================================


def test_rejects_too_few_points(analyzer: GrowthModelAnalyzer) -> None:
    """A three-parameter curve needs more than three points to be fittable.

    The requirement is ``n_params + 2``: the adjusted R-squared and the
    t-based parameter intervals both need at least one residual degree of
    freedom, and a fit whose intervals rest on nothing is worse than a
    refusal.
    """
    times = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(DataValidationError, match="at least 5 points"):
        analyzer.fit("logistic", times, 10.0 + times)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(bad_value: float, analyzer: GrowthModelAnalyzer) -> None:
    """One NaN or infinity poisons every parameter, so the series is refused.

    Silently dropping the row would be defensible for a missing measurement,
    but it is not this module's call to make: which rows survive changes the
    fit, and a user who cannot see that a point was discarded cannot check the
    fit either.
    """
    times = np.linspace(1.0, 20.0, 25)
    values = 10.0 + times
    values[7] = bad_value
    with pytest.raises(DataValidationError, match="non-finite"):
        analyzer.fit("logistic", times, values)


def test_rejects_negative_measurements(analyzer: GrowthModelAnalyzer) -> None:
    """A negative size is not a small size; it is a data error."""
    times = np.linspace(1.0, 20.0, 25)
    values = 10.0 + times
    values[3] = -0.5
    with pytest.raises(DataValidationError, match="non-negative"):
        analyzer.fit("logistic", times, values)


def test_rejects_a_constant_series(analyzer: GrowthModelAnalyzer) -> None:
    """A series with no variation has no growth to fit.

    Every one of these models can produce a nearly flat curve, so this input
    would otherwise return a plausible R-squared and a meaningless asymptote.
    """
    times = np.linspace(1.0, 20.0, 25)
    with pytest.raises(DataValidationError, match="no growth to fit"):
        analyzer.fit("logistic", times, np.full(25, 7.0))


def test_rejects_an_unknown_model(analyzer: GrowthModelAnalyzer) -> None:
    """An unrecognised name must not fall back to some default curve."""
    times = np.linspace(1.0, 20.0, 25)
    with pytest.raises(DataValidationError, match="Unknown growth model"):
        analyzer.fit("weibull", times, 10.0 + times)


@pytest.mark.parametrize("direction", ["reversed", "repeated"])
def test_rejects_a_non_monotonic_time_axis(direction: str, analyzer: GrowthModelAnalyzer) -> None:
    """Time must increase strictly: a reversed axis is a different problem.

    Reversing the time axis silently turns a growth series into a decay
    series, which several of these models can fit just as well and report
    with the sign of ``b`` flipped -- an asymptote that is still correct and a
    rate that is not.
    """
    times = np.linspace(1.0, 20.0, 25)
    values = 10.0 + times
    if direction == "reversed":
        times = times[::-1].copy()
        values = values[::-1].copy()
    else:
        times = np.concatenate([times[:10], times[9:]])
        values = np.concatenate([values[:10], values[9:]])
    with pytest.raises(DataValidationError, match="Time must increase strictly"):
        analyzer.fit("logistic", times, values)


def test_rejects_mismatched_or_wrongly_shaped_series(analyzer: GrowthModelAnalyzer) -> None:
    """Times and values are one paired series, not two independent columns."""
    times = np.linspace(1.0, 20.0, 25)
    values = 10.0 + times
    with pytest.raises(DataValidationError, match="must pair up"):
        analyzer.fit("logistic", times, values[:-1])
    with pytest.raises(DataValidationError, match="one-dimensional"):
        analyzer.fit("logistic", times[:, np.newaxis], values)


def test_rejects_start_values_of_the_wrong_length(analyzer: GrowthModelAnalyzer) -> None:
    """Two starting values cannot fit a three-parameter curve."""
    times, values = _synthetic("logistic")
    with pytest.raises(DataValidationError, match="p0 must supply 3 values"):
        analyzer.fit("logistic", times, values, p0=[100.0, 0.25])


def test_fit_all_needs_at_least_one_model(analyzer: GrowthModelAnalyzer) -> None:
    """An empty comparison is a mistake, not a ranking with no winner."""
    times, values = _synthetic("logistic")
    with pytest.raises(DataValidationError, match="at least one model"):
        analyzer.fit_all([], times, values)


def test_fit_all_rejects_an_unknown_model_before_fitting_anything(
    analyzer: GrowthModelAnalyzer,
) -> None:
    """A typo in one of several names must not return a partial ranking."""
    times, values = _synthetic("logistic")
    with pytest.raises(DataValidationError, match="Unknown growth model"):
        analyzer.fit_all(["logistic", "gompertz", "sigmoidal"], times, values)


def test_input_errors_are_paleoast_errors_not_bare_value_errors() -> None:
    """The application's one exception handler has to be able to catch these.

    ``ValidationError`` is an alias of ``DataValidationError``, which derives
    from ``PaleoASTError`` and *not* from ``ValueError``. A bare ``ValueError``
    would sail past every handler written against that contract.
    """
    analyzer = GrowthModelAnalyzer()
    with pytest.raises(DataValidationError) as info:
        analyzer.fit("logistic", np.linspace(1.0, 20.0, 25), np.full(25, 7.0))
    assert isinstance(info.value, DataValidationError)
    assert not isinstance(info.value, ValueError)
    assert ValidationError is DataValidationError


def test_unknown_parameter_lookup_raises_key_error(analyzer: GrowthModelAnalyzer) -> None:
    """Asking for a parameter the model does not have is a lookup failure."""
    times, values = _synthetic("logistic")
    result = analyzer.fit("logistic", times, values)
    with pytest.raises(KeyError, match="Km"):
        result.param("Km")


# =============================================================================
# Model comparison
# =============================================================================


def test_fit_all_returns_one_entry_per_requested_model(analyzer: GrowthModelAnalyzer) -> None:
    """Every requested model comes back, keyed by canonical name."""
    times, values = _synthetic("von_bertalanffy")
    requested = ["von_bertalanffy", "gompertz", "logistic", "michaelis_menten"]
    results, ranking = analyzer.fit_all(requested, times, values)

    assert set(results) == set(requested)
    assert set(ranking) == set(requested)
    for name, result in results.items():
        assert result.model == name
        assert result.times.shape == times.shape
        assert result.values.shape == values.shape
        assert result.success


def test_fit_all_keeps_one_entry_for_a_repeated_model(analyzer: GrowthModelAnalyzer) -> None:
    """Asking for the same model twice fits it once.

    A dict cannot hold two fits under one name, and fitting the same data
    twice would return the same answer anyway -- the caller cannot tell a
    duplicate from a distinct fit by looking at the keys.
    """
    times, values = _synthetic("logistic")
    results, ranking = analyzer.fit_all(["logistic", "logistic"], times, values)
    assert set(results) == {"logistic"}
    assert set(ranking) == {"logistic"}


def test_aicc_ranking_is_ordered_lowest_first(analyzer: GrowthModelAnalyzer) -> None:
    """The ranking is sorted by AICc, and lower really is better."""
    times, values = _synthetic("von_bertalanffy")
    results, ranking = analyzer.fit_all(list(GROWTH_MODELS), times, values)

    assert list(ranking) == sorted(ranking, key=lambda name: ranking[name])
    for name, aicc in ranking.items():
        assert aicc == pytest.approx(results[name].aicc)
    assert len(ranking) == len(GROWTH_MODELS)


def test_the_generating_model_wins_the_ranking_by_a_wide_margin(
    analyzer: GrowthModelAnalyzer,
) -> None:
    """Data generated by von Bertalanffy rank von Bertalanffy first.

    Burnham & Anderson (2002) call a gap of 2 AICc units "substantial" and 10
    "approaching conclusive". The requirement here is 20 -- ten times the
    decisive threshold -- because all six curves fit the same sigmoid-shaped
    data and several of them are close to it. A model comparison that cannot
    separate them is not wrong, but it is not evidence of anything, and this
    asserts the separation exists.
    """
    times, values = _synthetic("von_bertalanffy")
    _, ranking = analyzer.fit_all(list(GROWTH_MODELS), times, values)

    assert next(iter(ranking)) == "von_bertalanffy"
    ordered = sorted(ranking.values())
    assert ordered[1] - ordered[0] > 20.0


def test_fit_all_sets_last_result_to_the_winner(analyzer: GrowthModelAnalyzer) -> None:
    """``last_result`` after ``fit_all`` is the best model, not the last fitted."""
    times, values = _synthetic("von_bertalanffy")
    analyzer.fit_all(list(GROWTH_MODELS), times, values)
    assert analyzer.last_result is not None
    assert analyzer.last_result.model == "von_bertalanffy"
    assert analyzer.last_ranking is not None
    assert next(iter(analyzer.last_ranking)) == "von_bertalanffy"


# =============================================================================
# Prediction and extrapolation
# =============================================================================


@pytest.mark.parametrize("model", sorted(_GROUND_TRUTH))
def test_predict_reproduces_the_fitted_curve_inside_the_range(model: str, analyzer: GrowthModelAnalyzer) -> None:
    """``predict`` returns the curve that was fitted, at the fitted points.

    Without this, ``predict`` could be an independent re-implementation of
    the curve -- the same mistake as fitting twice with two code paths --
    and every extrapolation test below would still pass.
    """
    times, values = _synthetic(model)
    result = analyzer.fit(model, times, values)
    assert result.predict(times) == pytest.approx(result.fit.fitted, rel=1e-12, abs=1e-12)


_SATURATING = ("von_bertalanffy", "gompertz", "michaelis_menten", "logistic", "gaussian")


@pytest.mark.parametrize("model", _SATURATING)
def test_predict_extrapolates_past_the_last_measurement(model: str, analyzer: GrowthModelAnalyzer) -> None:
    """One further window beyond the data is finite and still rising.

    Checking an asymptote means evaluating the curve outside the measured
    range, so ``predict`` must not clamp. For these five the curve is
    monotonically increasing, so a step of one whole window past the last
    observation has to land above it and stay finite. (The sinusoid is
    excluded deliberately: it is not monotone, and "larger than the last
    value" is not a property it has.)
    """
    times, values = _synthetic(model)
    result = analyzer.fit(model, times, values)
    beyond = float(times[-1] + np.ptp(times))

    predicted = result.predict(beyond)
    # A scalar request comes back as a 0-d array, which ``float`` reads
    # without the size-1 conversion NumPy has deprecated.
    assert np.isfinite(predicted).all()
    assert float(predicted) > float(result.values[-1])
    # And the whole extrapolated window, not just its first point.
    window = result.predict(np.linspace(times[-1], beyond, 25))
    assert np.all(np.isfinite(window))
    assert np.all(np.diff(window) >= -1e-9)


def test_sinusoidal_prediction_stays_inside_its_own_band(analyzer: GrowthModelAnalyzer) -> None:
    """The sinusoidal curve never leaves ``mean +/- amplitude``, however far out.

    The obvious failure of an oscillatory model extrapolated far past the
    data is a drifting offset or a growing amplitude; both show up here as a
    value outside the band.
    """
    times, values = _synthetic("sinusoidal")
    result = analyzer.fit("sinusoidal", times, values)
    mean = result.param("mean")
    amplitude = result.param("amplitude")

    far = result.predict(np.linspace(-500.0, 500.0, 1001))
    assert np.all(np.abs(far - mean) <= amplitude + 1e-6)


def test_predict_accepts_a_scalar(analyzer: GrowthModelAnalyzer) -> None:
    """A single age is a valid request, not a shape error."""
    times, values = _synthetic("logistic")
    result = analyzer.fit("logistic", times, values)
    assert float(result.predict(float(times[0]))) == pytest.approx(float(result.fit.fitted[0]), rel=1e-12)


# =============================================================================
# Determinism
# =============================================================================


def test_refitting_the_same_data_is_exactly_identical(analyzer: GrowthModelAnalyzer) -> None:
    """Nothing in the fit draws a random number, so it must repeat exactly.

    ``approx`` would be too weak here: the fit is a deterministic optimiser
    run on the same array, so anything short of bitwise equality means
    something stochastic -- an unseeded global RNG, a randomised restart --
    has crept in, and a published fit that reproduces only approximately is
    no longer reproducible at all.
    """
    times, values = _synthetic("gompertz")
    first = analyzer.fit("gompertz", times, values)
    second = analyzer.fit("gompertz", times, values)
    assert np.array_equal(first.params, second.params)
    assert np.array_equal(first.fit.fitted, second.fit.fitted)
    assert first.aicc == second.aicc


def test_a_seeded_data_series_is_reproducible() -> None:
    """The test's own noise is seeded, so a failure here is reproducible."""
    _, first = _synthetic("logistic", seed=99)
    _, second = _synthetic("logistic", seed=99)
    _, other = _synthetic("logistic", seed=100)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_a_separate_analyzer_reproduces_the_same_fit() -> None:
    """The result does not depend on the state of the analyzer instance."""
    times, values = _synthetic("gaussian")
    here = GrowthModelAnalyzer().fit("gaussian", times, values)
    there = GrowthModelAnalyzer().fit("gaussian", times, values)
    assert np.array_equal(here.params, there.params)


# =============================================================================
# Model registry and result plumbing
# =============================================================================


def test_every_registered_model_has_a_curve_and_parameter_names() -> None:
    """The public tuple is the contract; it must match the module's own view."""
    assert len(GROWTH_MODELS) == len(set(GROWTH_MODELS)) == 6
    for model in GROWTH_MODELS:
        assert resolve_model_name(model) == model
        curve, names = _CURVES[model]
        assert callable(curve)
        assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("von Bertalanffy", "von_bertalanffy"),
        ("von_bertalanffy", "von_bertalanffy"),
        ("vonBertalanffy", "von_bertalanffy"),
        ("Michaelis-Menten", "michaelis_menten"),
        ("michaelis_menten", "michaelis_menten"),
        ("Gompertz", "gompertz"),
        ("Logistic", "logistic"),
        ("Gaussian growth curve", "gaussian"),
        ("sinusoidal", "sinusoidal"),
    ],
)
def test_model_labels_are_accepted(label: str, expected: str) -> None:
    """PAST3 calls these curves things; those names have to work."""
    assert resolve_model_name(label) == expected


def test_label_and_key_produce_the_same_fit(analyzer: GrowthModelAnalyzer) -> None:
    """An alias is a naming convenience, not a second model."""
    times, values = _synthetic("michaelis_menten")
    by_key = analyzer.fit("michaelis_menten", times, values)
    by_label = analyzer.fit("Michaelis-Menten", times, values)
    assert np.array_equal(by_key.params, by_label.params)
    assert by_label.model == "michaelis_menten"


def test_last_result_is_none_before_any_fit(analyzer: GrowthModelAnalyzer) -> None:
    """A fresh analyzer has no result to report, and says so."""
    assert analyzer.last_result is None
    assert analyzer.last_ranking is None


def test_summary_names_the_model_and_every_parameter(analyzer: GrowthModelAnalyzer) -> None:
    """The summary is what a user pastes into a results table."""
    times, values = _synthetic("gaussian")
    result = analyzer.fit("gaussian", times, values)
    text = result.summary()

    assert "Gaussian growth curve" in text
    assert "Observations: 41" in text
    for name in result.param_names:
        assert name in text
    assert "AICc" in text


def test_to_dict_is_json_serialisable_and_scalar_only(analyzer: GrowthModelAnalyzer) -> None:
    """Export has to survive ``json.dumps`` -- no arrays, no NumPy scalars.

    ``json.dumps`` raises on a NumPy scalar but silently accepts a NumPy
    *array* by way of the ``ndarray`` repr fallback in some encoders, so the
    type walk below is the part that has teeth.
    """
    times, values = _synthetic("von_bertalanffy")
    result = analyzer.fit("von_bertalanffy", times, values)
    payload = result.to_dict()

    assert json.loads(json.dumps(payload))["model"] == "von_bertalanffy"

    def _check(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                _check(item)
        elif isinstance(node, list):
            for item in node:
                _check(item)
        else:
            assert isinstance(node, (str, int, float, bool, type(None))), (
                f"non-JSON scalar {type(node).__name__} in to_dict()"
            )

    _check(payload)
    assert payload["n_obs"] == len(times)
    assert payload["params"] == [float(p) for p in result.params]
