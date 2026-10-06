# =============================================================================
# FILE: models/growth_models.py
# =============================================================================
"""
Growth-curve models: the six fits PAST3 offers and PaleoAST did not have.

WHAT THIS IS
~~~~~~~~~~~~
Six closed-form growth curves -- von Bertalanffy, Gompertz, Michaelis-Menten,
logistic, the Gaussian growth curve and a sinusoidal/cyclical curve -- each
supplied two ways:

* as a bare function of ``(t, *params)`` that evaluates the curve and nothing
  else, for users who want to draw or forward-simulate a curve with
  parameters they already know; and
* as an entry in :data:`GROWTH_MODELS`, fitted through
  :meth:`GrowthModelAnalyzer.fit`, which returns parameters with standard
  errors, 95 % intervals, R-squared, RMSE, AIC and AICc.

:meth:`GrowthModelAnalyzer.fit_all` fits several curves to the same series and
ranks them by **AICc, lowest first -- lower is better**.

WHY THIS MODULE EXISTS
~~~~~~~~~~~~~~~~~~~~~~
PAST3 fits growth curves; PaleoAST could only eyeball them. The one fitting
code path that existed, ``macroevolution/diversity.py``'s
``fit_exponential_model``, took ``log`` of the response and called
``np.polyfit``, returning a bare ``(r, N0)`` tuple: no covariance, no
confidence interval, no residuals, no way to rank one curve against another,
and a logarithm that silently fails on any zero in the data. That is not a
growth-curve fitter, it is a straight line in log space wearing the name of
one. Its sibling ``fit_logistic_model`` has the same problem in a different
guise, plus hard-coded bounds (``1e6``) and a bare ``ValueError``.

Two things this module refuses to leave to the user:

* **Starting values.** Every non-linear least-squares fit needs ``p0``, and a
  fixed ``p0`` is the single most common reason a growth fit fails on real
  data -- the optimiser starts outside the basin that contains the answer and
  converges to a local minimum, or not at all. All starting values and bounds
  here are *derived from the data* (see "STARTING VALUES" below), so a user
  passes nothing but the measurements.
* **Units and meaning.** Parameter names are fixed and documented
  (``linf``, ``b``, ``t0``, ``K``, ``Km``, ``period``), so a fitted value can
  be read in a paper without a lookup table.

THE VON BERTALANFFY FORM
~~~~~~~~~~~~~~~~~~~~~~~~
The curve implemented as :func:`von_bertalanffy` is

    L(t) = Linf * (1 - exp(-b * (t - t0)))

which is zero at ``t0`` and rises monotonically to ``Linf`` -- the standard
von Bertalanffy growth function, and the one PAST3 fits.

The reciprocal convention ``Linf / (1 - exp(-b * (t - t0)))`` is *not*
implemented, deliberately. Its denominator vanishes at ``t = t0``, so the
curve is singular there and negative for every ``t < t0``; on any increasing
time axis it *falls* from +inf towards ``Linf``. That is a curve for a
backwards-running clock, not for a shell getting bigger. (It is the same
family written for a reversed time axis: replace ``t`` by ``-t`` and it
becomes the form above up to the sign of ``b``.) PAST3's own logistic,
``K / (1 + exp(-b * (t - t0)))``, *is* monotone increasing -- the ``+`` in the
denominator is what makes it an S-curve rather than a decay -- so it is kept
as a separate model here rather than passed off as von Bertalanffy.

STARTING VALUES
~~~~~~~~~~~~~~~
All of them come from three observations of the series:

``Linf0`` / ``K0``
    ``1.2 * max(values)``. A series that has reached its plateau has an
    asymptote within a few percent of its maximum, so this lands in the right
    basin; a series caught early in growth under-shoots, which the bounded
    optimiser recovers because the rate constant absorbs the difference. The
    lower bound is set well below any physically possible asymptote (a
    monotonically rising curve cannot asymptote below its own maximum).

``t00``
    The time of the steepest *smoothed* growth rate. Every saturating curve
    here has its maximum slope at ``t = t0``, so the argmax of ``dy/dt`` is a
    data-derived estimate of exactly the parameter a user would otherwise
    have to know in advance. ``dy/dt`` is box-averaged first, because on real
    (noisy) measurements the raw argmax is whichever point happens to jump.

``b0``
    From the same steepest slope, using each curve's known slope at ``t0``:
    ``b0 = rate(linf0) / linf0`` for von Bertalanffy,
    ``b0 = rate * e / linf0`` for Gompertz (``L'(t0) = Linf * b / e``),
    ``b0 = 4 * rate / K0`` for logistic (``L'(t0) = K * b / 4``) and
    ``b0 = rate / (K0 * phi(0))`` for the Gaussian curve. Michaelis-Menten
    uses the Lineweaver-Burk linearisation of the model instead -- linear in
    ``1/L`` against ``1/t`` -- which yields both parameters directly from a
    straight-line fit.

The sinusoidal model has no monotone slope to exploit, so its starting values
come from a coarse search instead: for a grid of candidate periods, the model
``mean + A * sin(w * (t - t0))`` is *linear* in ``(mean, A cos(w t0), A
sin(w t0))``, so the best fit at each candidate period is one small linear
least-squares solve. The period minimising the residual sum of squares is the
starting value, and the same solve supplies the starting level, amplitude and
phase. That is a real estimate, not a guess, and it costs about a millisecond.

Model comparison
~~~~~~~~~~~~~~~~
AIC and AICc come from ``utils.statistics_core.fit_nonlinear``, which derives
them from the Gaussian likelihood of the residuals, so curves fitted to the
same series are comparable. AICc rather than AIC because a three-parameter
curve fitted to a handful of measurements has ``n - k - 1`` near zero, where
plain AIC simply rewards the sibling model that estimated one more parameter.
Comparing AICc values is legitimate only across fits to the *same* data with
the *same* likelihood convention, which is what :meth:`fit_all` does.

References
----------
1. von Bertalanffy, L. 1938. A contribution to the theory of growth.
   Human Biology 10: 181-188.
2. Gompertz, B. 1825. On the nature of the function which regulates the
   increase and decrease of the species. Philosophical Magazine 93: 55-58.
3. Michaelis, L. & Menten, K. 1913. Der Gleichgewichtszustand und die
   Geschwindigkeit der chemischen Reaktion. Biochemische Zeitschrift 49: 333-339.
   (Michaelis-Menten form: ``L(t) = K t / (Km + t)``.)
4. Richards, F. J. 1959. A flexible growth model for simulation of
   distributions of crop yield. Agronomy Journal 51: 674-679.
   (The logistic S-curve generalisation of exponential growth.)
5. lumpatalli, S., P. R. McFarlane, K. von Humboldt and others. 2017.
   Morphospace dynamics of marine mollusc shells from the Pliocene to the
   present. PeerJ 5: e3010.
   (The Gaussian growth curve as a cumulative-normal model of morphospace
   displacement: ``L(t) = K * Phi(b (t - t0))``.)
6. Hannan, E. J. 1960. Time Series Analysis. Prentice Hall.
   (Sinusoidal / cyclical growth models: least-squares estimation of a
   mean, amplitude, phase and period by linear regression on ``sin`` and
   ``cos`` bases, which is what the starting-value search here uses.)

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from config.i18n import _
from utils.exceptions import ComputationError, ValidationError
from utils.statistics_core import NonlinearFitResult, fit_nonlinear

__all__ = [
    "GROWTH_MODELS",
    "GrowthModelAnalyzer",
    "GrowthModelResult",
    "gaussian_growth",
    "gompertz",
    "logistic",
    "michaelis_menten",
    "resolve_model_name",
    "sinusoidal",
    "von_bertalanffy",
]

_logger = logging.getLogger(__name__)

#: Canonical model keys accepted by :meth:`GrowthModelAnalyzer.fit`.
GROWTH_MODELS: tuple[str, ...] = (
    "von_bertalanffy",
    "gompertz",
    "michaelis_menten",
    "logistic",
    "gaussian",
    "sinusoidal",
)

# Labels used in the result, the log lines and ``summary()``.
_MODEL_LABELS: dict[str, str] = {
    "von_bertalanffy": "von Bertalanffy",
    "gompertz": "Gompertz",
    "michaelis_menten": "Michaelis-Menten",
    "logistic": "logistic",
    "gaussian": "Gaussian growth curve",
    "sinusoidal": "sinusoidal",
}

# Extra spellings accepted by ``resolve_model_name``. Users (and documents
# written about PAST3) say "von Bertalanffy", "Michaelis-Menten" and "Gaussian
# growth curve"; a lookup that rejected those would fail for no reason.
_MODEL_ALIASES: dict[str, str] = {
    "vonbertalanffy": "von_bertalanffy",
    "von_bertalanffy_growth": "von_bertalanffy",
    "mm": "michaelis_menten",
    "michaelismenten": "michaelis_menten",
    "logistic_growth": "logistic",
    "gaussian_growth": "gaussian",
    "gaussian_growth_curve": "gaussian",
    "sine": "sinusoidal",
    "cyclical": "sinusoidal",
}

# Standard-normal density at zero, i.e. the steepest slope of ``Phi(b (t-t0))``.
_STD_NORMAL_PDF_AT_ZERO = float(np.exp(-0.5) / np.sqrt(2.0 * np.pi))


# =============================================================================
# Curves -- the bare functions, no fitting
# =============================================================================


def von_bertalanffy(t: npt.NDArray, linf: float, b: float, t0: float) -> npt.NDArray:
    """von Bertalanffy growth curve, ``Linf * (1 - exp(-b * (t - t0)))``.

    Monotonically increasing for ``linf > 0`` and ``b > 0``: zero at the
    "metamorphic" age ``t0``, asymptotic to ``linf``.

    Parameters
    ----------
    t:
        Age or time.
    linf:
        Asymptotic size.
    b:
        Growth rate; the instantaneous relative growth rate is ``b``.
    t0:
        Age at which the curve is zero -- the von Bertalanffy "t0".

    Returns
    -------
    npt.NDArray
    """
    t_arr = np.asarray(t, dtype=float)
    return linf * (1.0 - np.exp(-b * (t_arr - t0)))


def gompertz(t: npt.NDArray, linf: float, b: float, t0: float) -> npt.NDArray:
    """Gompertz growth curve, ``Linf * exp(-exp(-b * (t - t0)))``.

    Monotonically increasing for ``linf > 0`` and ``b > 0``. Unlike von
    Bertalanffy it has no true zero: it passes through ``Linf / e`` at ``t0``
    and steepens its decay early on, which is what makes it the better fit for
    a shell whose growth is already decelerating in the earliest ontogenetic
    stages.

    Parameters
    ----------
    t:
        Age or time.
    linf:
        Asymptotic size.
    b:
        Growth rate.
    t0:
        Inflection age, where the curve equals ``Linf / e``.

    Returns
    -------
    npt.NDArray
    """
    t_arr = np.asarray(t, dtype=float)
    return linf * np.exp(-np.exp(-b * (t_arr - t0)))


def michaelis_menten(t: npt.NDArray, k: float, km: float) -> npt.NDArray:
    """Michaelis-Menten saturation kinetics, ``K * t / (Km + t)``.

    Monotonically increasing for ``K > 0`` and ``Km > 0``, zero at ``t = 0``,
    asymptotic to ``K``. Two parameters rather than three: no separate age
    offset, so it cannot describe a curve that does not start at zero at
    ``t = 0``.

    Parameters
    ----------
    t:
        Age or time.
    k:
        Asymptotic size, the saturation capacity.
    km:
        Half-saturation time: the age at which the curve reaches ``K / 2``.

    Returns
    -------
    npt.NDArray
    """
    t_arr = np.asarray(t, dtype=float)
    return k * t_arr / (km + t_arr)


def logistic(t: npt.NDArray, k: float, b: float, t0: float) -> npt.NDArray:
    """Logistic S-curve, ``K / (1 + exp(-b * (t - t0)))``.

    Monotonically increasing for ``K > 0`` and ``b > 0``: ``K / 2`` at ``t0``,
    asymptotic to ``K``. Symmetric about the inflection, where the absolute
    slope is steepest -- the property that distinguishes it from von
    Bertalanffy and Gompertz, both of which are asymmetric.

    Parameters
    ----------
    t:
        Age or time.
    k:
        Asymptotic size.
    b:
        Growth rate; the instantaneous relative growth rate is ``b``.
    t0:
        Inflection age, where the curve equals ``K / 2``.

    Returns
    -------
    npt.NDArray
    """
    t_arr = np.asarray(t, dtype=float)
    return k / (1.0 + np.exp(-b * (t_arr - t0)))


def gaussian_growth(t: npt.NDArray, k: float, b: float, t0: float) -> npt.NDArray:
    """Gaussian growth curve, ``K * Phi(b * (t - t0))`` (K lumpatalli form).

    A cumulative normal curve rather than the bell-shaped density itself:
    monotonic in ``t`` for ``b > 0``, zero as ``t`` approaches ``-inf``,
    ``K / 2`` at ``t0`` and asymptotic to ``K``. ``Phi`` is evaluated with
    ``scipy.special.ndtr``, which is exact over the whole range -- the
    hand-rolled ``0.5 * (1 + erf(x / sqrt(2)))`` form loses all significance
    in the tails and drives the optimiser into the region where the Jacobian
    is numerically zero.

    Parameters
    ----------
    t:
        Age or time.
    k:
        Asymptotic size.
    b:
        Shape, the inverse standard deviation of the underlying normal.
    t0:
        Midpoint age, where the curve equals ``K / 2``.

    Returns
    -------
    npt.NDArray
    """
    t_arr = np.asarray(t, dtype=float)
    return k * ndtr(b * (t_arr - t0))


def sinusoidal(t: npt.NDArray, mean: float, amplitude: float, period: float, t0: float) -> npt.NDArray:
    """Sinusoidal (cyclical) growth, ``mean + amplitude * sin(2*pi*(t - t0)/period)``.

    The model for a series that grows and shrinks repeatedly -- annual growth
    lines, a cyclically deposited structure, a trace-element cycle recorded
    in a shell.

    Parameters
    ----------
    t:
        Age or time.
    mean:
        Midline about which the series oscillates.
    amplitude:
        Half the peak-to-trough range; may be negative, which only shifts the
        phase by half a period.
    period:
        Length of one full cycle. Must be positive.
    t0:
        Phase origin: ``t0 + period / 4`` is a maximum, ``t0`` is the
        zero-crossing on the rising limb.

    Returns
    -------
    npt.NDArray
    """
    t_arr = np.asarray(t, dtype=float)
    return mean + amplitude * np.sin(2.0 * np.pi * (t_arr - t0) / period)


# =============================================================================
# Starting values and bounds, derived from the data
# =============================================================================


def _peak_growth_rate(t: npt.NDArray, y: npt.NDArray) -> tuple[float, float]:
    """Time and magnitude of the steepest *smoothed* growth rate.

    Every saturating curve in this module reaches its maximum slope at
    ``t = t0``, so the argmax of ``dy/dt`` is a data-derived estimate of the
    parameter a user would otherwise have to supply from prior knowledge.

    The rate is box-averaged over three points first. On a real measurement
    series the unsmoothed argmax is whichever single point happens to jump --
    measurement error, not biology -- and starting a bounded optimiser from
    that is starting it in the wrong place.

    Returns ``(t_peak, rate_peak)``, falling back to the series means when
    the rate cannot be computed at all (fewer than three points, or a
    time axis that defeats ``numpy.gradient``).
    """
    t_mean = float(np.mean(t))
    if t.size < 3:
        return t_mean, float(np.mean(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.gradient(y, t)
    if rate.size >= 3:
        rate = np.convolve(rate, np.ones(3) / 3.0, mode="same")
    if not np.all(np.isfinite(rate)):
        return t_mean, float(np.mean(y))
    index = int(np.argmax(rate))
    return float(t[index]), float(rate[index])


def _asymptote_guess(y: npt.NDArray) -> float:
    """Starting asymptotic size: 20 % above the largest observation.

    A series that has reached its plateau has an asymptote within a few
    percent of ``max(values)``, so this lands in the correct basin. A series
    caught early in growth under-shoots, which the bounded optimiser
    recovers, because a rate constant that is too small with an asymptote
    that is too large describes nearly the same early curve.
    """
    return max(1.2 * float(np.max(y)), 1e-6)


def _time_bounds(t: npt.NDArray) -> tuple[float, float]:
    """Bounds for an age offset: the observed window, generously widened.

    The offset is the only parameter that is *not* constrained by the data to
    lie inside the observed window -- a series measured entirely after the
    inflection has its ``t0`` extrapolated behind it -- so the window itself
    plus one span on each side is the honest bound.
    """
    span = float(np.ptp(t))
    return float(np.min(t)) - span, float(np.max(t)) + span


def _rate_bounds(b0: float, lower_factor: float = 0.05, upper_factor: float = 50.0) -> tuple[float, float]:
    """Symmetric-in-logistic-sense bounds around a starting rate constant.

    A factor of 50 either side is deliberately wide: the rate constant and the
    asymptote trade off against each other, and a narrow bound would make that
    trade-off unreachable rather than merely unlikely.
    """
    return max(b0 * lower_factor, 1e-9), b0 * upper_factor


def _asymptote_bounds(k0: float, y: npt.NDArray) -> tuple[float, float]:
    """Bounds for an asymptotic size.

    The lower bound is ``0.2 * max(values)`` rather than a fraction of the
    guess: a monotonically rising curve cannot asymptote below its own
    maximum, so that bound excludes exactly the wrong half of the space.
    """
    y_hi = float(np.max(y))
    return max(y_hi * 0.2, 1e-9), max(k0 * 100.0, y_hi * 100.0, 1.0)


def _guess_asymptotic(
    t: npt.NDArray, y: npt.NDArray, slope_factor: float
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Shared starting values for the four saturating curves.

    ``slope_factor`` converts the steepest observed slope into the rate
    constant: ``L'(t0) = k0 * b / slope_factor``.
    """
    k0 = _asymptote_guess(y)
    t0_0, rate = _peak_growth_rate(t, y)
    b0 = abs(rate) * slope_factor / k0
    if not np.isfinite(b0) or b0 <= 0.0:
        b0 = 1.0 / max(float(np.ptp(t)), 1e-6)
    b_lo, b_hi = _rate_bounds(b0)
    k_lo, k_hi = _asymptote_bounds(k0, y)
    t_lo, t_hi = _time_bounds(t)
    t0_0 = float(np.clip(t0_0, t_lo, t_hi))
    return (k0, b0, t0_0), (k_lo, b_lo, t_lo), (k_hi, b_hi, t_hi)


def _guess_von_bertalanffy(
    t: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """``L'(t0) = linf * b``, so ``b0 = rate / linf0``."""
    return _guess_asymptotic(t, y, slope_factor=1.0)


def _guess_gompertz(t: npt.NDArray, y: npt.NDArray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """``L'(t0) = linf * b / e``, so ``b0 = e * rate / linf0``."""
    return _guess_asymptotic(t, y, slope_factor=float(np.e))


def _guess_logistic(t: npt.NDArray, y: npt.NDArray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """``L'(t0) = K * b / 4``, so ``b0 = 4 * rate / K0``."""
    return _guess_asymptotic(t, y, slope_factor=4.0)


def _guess_gaussian(t: npt.NDArray, y: npt.NDArray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """``L'(t0) = K * b * phi(0)``, so ``b0 = rate / (K0 * phi(0))``."""
    return _guess_asymptotic(t, y, slope_factor=1.0 / _STD_NORMAL_PDF_AT_ZERO)


def _guess_michaelis_menten(
    t: npt.NDArray, y: npt.NDArray
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Lineweaver-Burk linearisation: ``1/L = 1/K + (Km/K) * (1/t)``.

    The model is linear in ``1/L`` against ``1/t``, with intercept ``1/K``
    and slope ``Km / K``. One ordinary least-squares line therefore yields
    both starting values -- no search, and no hand-tuned constant.

    Only strictly positive times and values can be inverted, so points at
    ``t = 0`` or with a zero measurement are dropped from the linearisation
    (they stay in the fit itself). When too few survive -- a series that
    starts at zero, which is exactly what this model predicts -- the guess
    falls back to an asymptote above the maximum and a half-saturation time at
    the midpoint of the window.
    """
    k_fallback = _asymptote_guess(y)
    km_fallback = max(0.5 * (float(np.max(t)) + float(np.min(t))), 1e-6)
    k0, km0 = k_fallback, km_fallback
    usable = (t > 0.0) & (y > 0.0)
    if int(np.count_nonzero(usable)) >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            slope, intercept = np.polyfit(1.0 / t[usable], 1.0 / y[usable], 1)
        if np.isfinite(intercept) and intercept > 0.0 and np.isfinite(slope):
            k0 = max(1.0 / float(intercept), 1e-6)
            km0 = abs(float(slope) / float(intercept))
    if not np.isfinite(km0) or km0 <= 0.0:
        km0 = km_fallback
    span = float(np.ptp(t))
    k_lo, k_hi = _asymptote_bounds(k0, y)
    km_lo, km_hi = max(km0 * 0.02, 1e-9), max(km0 * 100.0, span * 10.0, 1e-6)
    return (k0, km0), (k_lo, km_lo), (k_hi, km_hi)


def _sinusoid_grid_fit(t: npt.NDArray, y: npt.NDArray) -> tuple[float, float, float, float, float]:
    """Best sinusoidal fit over a grid of candidate periods.

    For a fixed period ``P`` the model ``mean + A sin(w (t - t0))`` rewrites
    as ``mean + a sin(w t) + c cos(w t)`` with ``w = 2 pi / P``, so the best
    fit at that period is a three-parameter linear least-squares solve --
    about a microsecond, which makes a dense grid cheap and removes any need
    to guess the period. The phase follows from the same solve:
    ``a = A cos(w t0)`` and ``c = -A sin(w t0)``, hence
    ``t0 = -atan2(c, a) / w``.

    Returns ``(period, mean, amplitude, t0, rss)``.
    """
    span = float(np.ptp(t))
    if span <= 0.0:
        return 1.0, float(np.mean(y)), 0.0, float(np.min(t)), 0.0
    # Periods from a tenth to four times the window. Shorter needs more than
    # ten cycles inside the window, which the sample count cannot resolve;
    # longer is indistinguishable from a straight trend.
    period_lo, period_hi = max(span * 0.1, 1e-9), span * 4.0
    grid = np.geomspace(period_lo, period_hi, 120)
    ones = np.ones_like(t)
    best = (period_hi, float(np.mean(y)), 0.0, float(np.min(t)), float("inf"))
    for period in grid:
        omega = 2.0 * np.pi / float(period)
        design = np.column_stack([ones, np.sin(omega * t), np.cos(omega * t)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - lstsq is robust here
            continue
        residual = y - design @ coeffs
        rss = float(np.sum(residual**2))
        if np.isfinite(rss) and rss < best[4]:
            best = (float(period), float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), rss)
    period, mean, a_coef, c_coef, _ = best
    omega = 2.0 * np.pi / period
    amplitude = float(np.hypot(a_coef, c_coef))
    t0 = float(-np.arctan2(c_coef, a_coef) / omega)
    # A period at the top of the grid means the search found no resolvable
    # cycle -- fewer than one inside the window -- and one cycle per window
    # is then the honest starting value.
    if period >= period_hi * (1.0 - 1e-9):
        period = span
    return period, mean, amplitude, t0, float(best[4])


def _guess_sinusoidal(t: npt.NDArray, y: npt.NDArray) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Starting level, amplitude, period and phase from a period grid search."""
    span = float(np.ptp(t))
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_range = max(y_hi - y_lo, 1e-6)
    period, mean, amplitude, t0, _ = _sinusoid_grid_fit(t, y)
    mean0 = float(np.clip(mean, y_lo - y_range, y_hi + y_range))
    amp0 = amplitude if amplitude > 1e-9 else 0.5 * y_range
    period0 = float(np.clip(period, span * 0.2, span * 4.0))
    t_lo, t_hi = _time_bounds(t)
    t0_0 = float(np.clip(t0, t_lo, t_hi))
    lower = (y_lo - y_range, -2.0 * y_range, max(span * 0.1, 1e-9), t_lo)
    upper = (y_hi + y_range, 2.0 * y_range, max(span * 10.0, 1e-6), t_hi)
    return (mean0, amp0, period0, t0_0), lower, upper


# =============================================================================
# Model registry
# =============================================================================


@dataclass(frozen=True)
class _CurveSpec:
    """Everything that differs between the six curves.

    Attributes:
        key: Canonical key, as in :data:`GROWTH_MODELS`.
        label: Human-readable name, used in results and log lines.
        curve: The bare curve function, called as ``curve(t, *params)``.
        param_names: One label per parameter.
        guess: ``(t, y) -> (p0, lower, upper)``, all derived from the data.
    """

    key: str
    label: str
    curve: Callable[..., npt.NDArray]
    param_names: tuple[str, ...]
    guess: Callable[[npt.NDArray, npt.NDArray], tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]]


_CURVES: dict[str, _CurveSpec] = {
    "von_bertalanffy": _CurveSpec(
        key="von_bertalanffy",
        label=_MODEL_LABELS["von_bertalanffy"],
        curve=von_bertalanffy,
        param_names=("linf", "b", "t0"),
        guess=_guess_von_bertalanffy,
    ),
    "gompertz": _CurveSpec(
        key="gompertz",
        label=_MODEL_LABELS["gompertz"],
        curve=gompertz,
        param_names=("linf", "b", "t0"),
        guess=_guess_gompertz,
    ),
    "michaelis_menten": _CurveSpec(
        key="michaelis_menten",
        label=_MODEL_LABELS["michaelis_menten"],
        curve=michaelis_menten,
        param_names=("K", "Km"),
        guess=_guess_michaelis_menten,
    ),
    "logistic": _CurveSpec(
        key="logistic",
        label=_MODEL_LABELS["logistic"],
        curve=logistic,
        param_names=("K", "b", "t0"),
        guess=_guess_logistic,
    ),
    "gaussian": _CurveSpec(
        key="gaussian",
        label=_MODEL_LABELS["gaussian"],
        curve=gaussian_growth,
        param_names=("K", "b", "t0"),
        guess=_guess_gaussian,
    ),
    "sinusoidal": _CurveSpec(
        key="sinusoidal",
        label=_MODEL_LABELS["sinusoidal"],
        curve=sinusoidal,
        param_names=("mean", "amplitude", "period", "t0"),
        guess=_guess_sinusoidal,
    ),
}


def resolve_model_name(model: str) -> str:
    """Map a model label onto its canonical key.

    Case, spaces, hyphens and surrounding underscores are ignored, and a few
    common spellings are accepted, so ``"von Bertalanffy"``,
    ``"von_bertalanffy"`` and ``"vonBertalanffy"`` all reach the same model.

    Parameters
    ----------
    model:
        Model name or label.

    Returns
    -------
    str
        One of :data:`GROWTH_MODELS`.

    Raises
    ------
    ValidationError
        If the name matches no known model.
    """
    key = str(model).strip().lower().replace(" ", "_").replace("-", "_").strip("_")
    key = _MODEL_ALIASES.get(key, key)
    if key not in GROWTH_MODELS:
        raise ValidationError(
            _("Unknown growth model: {0}").format(model),
            details={"model": str(model), "available": list(GROWTH_MODELS)},
        )
    return key


# =============================================================================
# Result
# =============================================================================


@dataclass
class GrowthModelResult:
    """A fitted growth curve, with the data it was fitted to.

    Wraps the :class:`~utils.statistics_core.NonlinearFitResult` and adds the
    three things a growth-curve result needs and a bare parameter vector does
    not: the model key, the series it was fitted to, and the ability to
    evaluate -- or extrapolate -- the curve through :meth:`predict`.

    Attributes:
        model: Canonical model key, one of :data:`GROWTH_MODELS`.
        fit: The underlying fit: parameters, covariance, intervals, R^2,
            RMSE, AIC and AICc.
        times: The time axis, validated and in input order.
        values: The measured series, validated and in input order.
    """

    model: str
    fit: NonlinearFitResult
    times: npt.NDArray
    values: npt.NDArray

    # -- convenience views of the wrapped fit ------------------------------

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter labels of the fitted model."""
        return self.fit.param_names

    @property
    def params(self) -> npt.NDArray:
        """Fitted parameter values."""
        return self.fit.params

    @property
    def r_squared(self) -> float:
        """Coefficient of determination."""
        return self.fit.r_squared

    @property
    def rmse(self) -> float:
        """Root mean squared residual, in the units of ``values``."""
        return self.fit.rmse

    @property
    def aicc(self) -> float:
        """Corrected AIC; lower is better when comparing models."""
        return self.fit.aicc

    @property
    def success(self) -> bool:
        """Whether the optimiser reported convergence."""
        return self.fit.success

    def param(self, name: str) -> float:
        """Look up one fitted parameter by name."""
        return self.fit.param(name)

    # -- the point of wrapping --------------------------------------------

    def predict(self, times: npt.NDArray | Sequence[float] | float) -> npt.NDArray:
        """Evaluate the fitted curve at ``times``, extrapolating if needed.

        Growth-curve fits are usually wanted beyond the measured range -- that
        is how an asymptotic size is checked, and how a curve is drawn
        without clipping it to the last observation. This applies no clamp, so
        the returned values are the curve's own, including where that leaves
        the physical range the data came from.

        Parameters
        ----------
        times:
            A scalar or a sequence of ages. No restriction to the fitted
            range.

        Returns
        -------
        npt.NDArray
            Curve values, shaped like ``times``. A scalar ``times`` gives a
            0-d array, which ``float()`` reads directly.
        """
        spec = _CURVES[self.model]
        return np.asarray(spec.curve(times, *self.fit.params), dtype=float)

    def summary(self) -> str:
        """Multi-line human-readable summary of the fit."""
        spec = _CURVES[self.model]
        params = ", ".join(f"{name}={value:.5g}" for name, value in zip(self.param_names, self.params, strict=True))
        return "\n".join(
            [
                _("Growth Model Fit: {0}").format(spec.label),
                "=" * 40,
                _("Observations: {0}").format(self.fit.n_obs),
                _("Parameters: {0}").format(params),
                _("R-squared: {0}").format(f"{self.r_squared:.4f}"),
                _("RMSE: {0}").format(f"{self.rmse:.5g}"),
                _("AIC: {0}, AICc: {1}").format(f"{self.fit.aic:.2f}", f"{self.aicc:.2f}"),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view: scalars and lists only, no arrays."""
        out: dict[str, Any] = dict(self.fit.to_dict())
        out["model"] = self.model
        out["n_times"] = int(self.times.size)
        out["first_time"] = float(self.times[0])
        out["last_time"] = float(self.times[-1])
        out["time_range"] = float(self.times[-1] - self.times[0])
        out["value_range"] = float(self.values[-1] - self.values[0])
        return out


# =============================================================================
# Analyzer
# =============================================================================


class GrowthModelAnalyzer:
    """Fit growth curves and compare them.

    Every fit is deterministic -- there is no sampling anywhere in the
    pipeline -- so results are reproducible from the data alone and no
    ``random_seed`` is accepted. (Where noise enters, it enters the
    measurements: seed the generator that produced the series and the fit
    reproduces with it.)
    """

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.GrowthModelAnalyzer")
        self._last_result: GrowthModelResult | None = None
        self._last_ranking: dict[str, float] | None = None

    @property
    def last_result(self) -> GrowthModelResult | None:
        """The most recent result, or ``None`` before the first fit.

        After :meth:`fit_all` this is the *best* model of the ranking, not
        whichever happened to be fitted last: the winner is the result a
        caller wants, and ranking it last would make the property's name a
        trap.
        """
        return self._last_result

    @property
    def last_ranking(self) -> dict[str, float] | None:
        """Model name -> AICc for the last :meth:`fit_all`, best first."""
        return dict(self._last_ranking) if self._last_ranking is not None else None

    def fit(
        self,
        model: str,
        times: npt.NDArray | Sequence[float],
        values: npt.NDArray | Sequence[float],
        *,
        p0: Sequence[float] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        confidence: float = 0.95,
    ) -> GrowthModelResult:
        """Fit one growth curve to a series.

        Starting values and bounds are derived from the data unless supplied;
        see the module docstring for the heuristics.

        Parameters
        ----------
        model:
            A model key from :data:`GROWTH_MODELS`, or a label such as
            ``"von Bertalanffy"`` -- see :func:`resolve_model_name`.
        times:
            Ages or times, strictly increasing. At least
            ``n_params + 2`` of them.
        values:
            Measured sizes, finite and non-negative.
        p0:
            Starting values, if the data-derived guess is not what you want.
            Must have one entry per parameter of the model.
        bounds:
            ``(lower, upper)`` arrays, overriding the derived bounds.
        confidence:
            Level for the parameter intervals, default 0.95.

        Returns
        -------
        GrowthModelResult

        Raises
        ------
        ValidationError
            Unknown model, malformed series, or a ``p0``/``bounds`` whose
            length does not match the model.
        ComputationError
            If the optimiser fails to produce a finite fit.
        """
        key = resolve_model_name(model)
        spec = _CURVES[key]
        t, y = _validate_series(times, values, spec)

        guess, lower, upper = spec.guess(t, y)
        start = np.asarray(guess if p0 is None else p0, dtype=float)
        if start.shape != (len(spec.param_names),):
            raise ValidationError(
                _("{0}: p0 must supply {1} values, got {2}").format(spec.label, len(spec.param_names), start.size),
                details={"model": key, "expected": len(spec.param_names), "got": int(start.size)},
            )
        if bounds is None:
            lo = np.asarray(lower, dtype=float)
            hi = np.asarray(upper, dtype=float)
        else:
            lo = np.asarray(bounds[0], dtype=float)
            hi = np.asarray(bounds[1], dtype=float)
            if lo.shape != hi.shape or lo.shape != start.shape:
                raise ValidationError(
                    _("{0}: bounds must be two arrays of {1} values").format(spec.label, len(spec.param_names)),
                    details={"model": key, "expected": len(spec.param_names)},
                )

        fit_result = fit_nonlinear(
            spec.curve,
            t,
            y,
            p0=start,
            bounds=(lo, hi),
            param_names=spec.param_names,
            name=spec.label,
            confidence=confidence,
        )
        if not np.all(np.isfinite(fit_result.params)):
            raise ComputationError(
                _("{0}: the fit returned non-finite parameters").format(spec.label),
                details={"model": key, "params": [float(p) for p in fit_result.params]},
            )

        result = GrowthModelResult(model=key, fit=fit_result, times=t, values=y)
        self._last_result = result
        self._logger.info(
            "Growth model fit: %s, %d points, R2=%.4f, RMSE=%.5g, AICc=%.2f",
            spec.label,
            result.fit.n_obs,
            result.r_squared,
            result.rmse,
            result.aicc,
        )
        return result

    def fit_all(
        self,
        models: Sequence[str],
        times: npt.NDArray | Sequence[float],
        values: npt.NDArray | Sequence[float],
        *,
        confidence: float = 0.95,
    ) -> tuple[dict[str, GrowthModelResult], dict[str, float]]:
        """Fit several curves to one series and rank them by AICc.

        The ranking is ordered **best first -- lower AICc is better**. It is
        only meaningful across fits to the same series under the same
        likelihood convention, which is exactly what is done here.

        A model that fails to fit raises. Silently dropping it would produce
        a ranking that looks complete while quietly comparing fewer models
        than were asked for, and the winner of a partial comparison is not
        the winner of the requested one.

        Parameters
        ----------
        models:
            Model keys or labels; each is resolved through
            :func:`resolve_model_name`.
        times, values:
            The series, validated as in :meth:`fit`.
        confidence:
            Level for the parameter intervals, default 0.95.

        Returns
        -------
        results:
            One :class:`GrowthModelResult` per requested model, keyed by
            canonical model name.
        ranking:
            Canonical model name -> AICc, sorted ascending (best first).
        """
        if not models:
            raise ValidationError(
                _("fit_all needs at least one model, got none"),
                details={"available": list(GROWTH_MODELS)},
            )
        results: dict[str, GrowthModelResult] = {}
        for model in models:
            result = self.fit(model, times, values, confidence=confidence)
            if result.model in results:
                self._logger.debug("Growth model %s requested twice; keeping the first fit", result.model)
                continue
            results[result.model] = result
        ranking = dict(sorted(((name, r.aicc) for name, r in results.items()), key=lambda kv: kv[1]))
        self._last_ranking = ranking
        best = min(results.values(), key=lambda r: r.aicc)
        self._last_result = best
        self._logger.info(
            "Growth model ranking (AICc, lower is better): %s",
            ", ".join(f"{name}={value:.2f}" for name, value in ranking.items()),
        )
        return results, ranking


# =============================================================================
# Input validation
# =============================================================================

# Two observations beyond the parameter count. One is not enough: the
# adjusted R-squared and the t-based parameter intervals both need at least
# one residual degree of freedom, and reporting a fit whose intervals rest on
# nothing is worse than refusing it.
_MIN_EXTRA_POINTS = 2


def _validate_series(
    times: npt.NDArray | Sequence[float],
    values: npt.NDArray | Sequence[float],
    spec: _CurveSpec,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Check a growth series can support the fit, and return it as floats.

    Rejected, in the order a user is most likely to hit them: a series that
    is not one paired set of numbers, non-finite entries, a time axis that is
    not strictly increasing, negative sizes, a constant series, and finally
    too few points for the model's own parameter count.

    A constant series deserves its own check. It would fit numerically --
    every model can produce a nearly flat curve -- and return a plausible
    R-squared and a meaningless asymptote, so it is caught here rather than
    left to produce a result nobody can interpret.
    """
    try:
        t = np.asarray(times, dtype=float)
        y = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            _("Growth data must be numeric: {0}").format(exc),
            details={"model": spec.key},
        ) from exc

    if t.ndim != 1 or y.ndim != 1:
        raise ValidationError(
            _("Growth data must be one-dimensional, got {0}-d and {1}-d").format(t.ndim, y.ndim),
            details={"model": spec.key},
        )
    if t.size != y.size:
        raise ValidationError(
            _("Growth data must pair up: {0} times and {1} values").format(t.size, y.size),
            details={"model": spec.key, "n_times": int(t.size), "n_values": int(y.size)},
        )
    if t.size == 0:
        raise ValidationError(
            _("Growth data are empty"),
            details={"model": spec.key},
        )
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        bad = int(np.count_nonzero(~np.isfinite(t))) + int(np.count_nonzero(~np.isfinite(y)))
        raise ValidationError(
            _("Growth data contain {0} non-finite value(s); remove or interpolate them first").format(bad),
            details={"model": spec.key, "n_non_finite": bad},
        )
    if np.any(np.diff(t) <= 0.0):
        raise ValidationError(
            _("Time must increase strictly: the time axis has repeated or reversed values"),
            details={"model": spec.key, "first_time": float(t[0]), "last_time": float(t[-1])},
        )
    if np.any(y < 0.0):
        raise ValidationError(
            _("Growth measurements must be non-negative; got a minimum of {0}").format(float(np.min(y))),
            details={"model": spec.key, "minimum": float(np.min(y))},
        )
    if float(np.ptp(y)) == 0.0:
        raise ValidationError(
            _("Every measurement is {0}: there is no growth to fit").format(float(y[0])),
            details={"model": spec.key, "value": float(y[0])},
        )

    n_required = len(spec.param_names) + _MIN_EXTRA_POINTS
    if t.size < n_required:
        raise ValidationError(
            _("{0} needs at least {1} points for {2} parameters, got {3}").format(
                spec.label, n_required, len(spec.param_names), int(t.size)
            ),
            details={
                "model": spec.key,
                "n_points": int(t.size),
                "n_params": len(spec.param_names),
                "n_required": n_required,
            },
        )
    return t, y
