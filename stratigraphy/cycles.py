# =============================================================================
# FILE: stratigraphy/cycles.py
# =============================================================================
"""
Cycle detection for stratigraphic and isotope time series.

This module fills in the analysis layer that cyclostratigraphy needs. The
existing periodogram in ``spectral_analysis`` already gives a Lomb-Scargle
spectrum with an AR(1) red-noise false-alarm threshold, and the existing
wavelet transform gives time-frequency structure, but neither can answer the
two questions that dominate this field:

1. **Is that peak a cycle or is it red noise?** REDFIT
   (Muellersohn et al. 1999; Watters & Solomon 2004) exists precisely because
   an AR(1) null tested against a raw periodogram is not the same question:
   the null has to be *subtracted* and the residual scaled by its own standard
   deviation, so a broad red-noise background does not swamp a narrow
   periodic line the way an unsubtracted threshold makes it.
2. **How strong and how persistent is the periodicity?** Autocorrelation
   with Bartlett bands, multitaper spectral estimation, the
   autoassociation function (Shirer 2008), Mann-Kendall with Sen's slope, and
   the runs test are the supporting evidence.

WHAT IS HERE, AND WHY THESE CHOICES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``autocorrelation`` is the foundation; the prewhitening, Mann-Kendall
variance correction, autoassociation curve and cross-correlation significance
band all sit on top of an ACF. It replaces three separate private lag-1
computations and one unreachable closure inside ``pyper_peterman_correction``
with a single public, Bartlett-banded estimator.

``ar1_prewhiten`` implements the Box & Jenkins (1970) prewhitening step:
either the inverse filter ``x_t - phi x_(t-1)``, or first differencing. The
choice between them is made by AIC, and the decision is *reported* in the
result rather than applied silently, because a prewhitening that is chosen
badly changes every downstream significance statement.

``redfit`` rotates the periodogram against the AR(1) spectrum. The AR(1)
curve itself is NOT re-derived here: it comes from
``spectral_analysis.ar1_theoretical_spectrum``, so the periodogram null and
the REDFIT null are guaranteed to be the same curve on the same scale.

``multitaper_spectrum`` builds its own DPSS tapers (eigenvectors of the
Percival & Walden tridiagonal matrix) because the environment has no DPSS
implementation. Thomson's (1982) tapers buy resolution at the cost of
variance, and the reported equivalent noise bandwidth and degrees of freedom
make that trade explicit.

``mann_kendall`` defaults to ``pre_whitened=True``'s cousin -- the
Hamed & Rao (1998) variance correction is opt-in, but the docstring says
plainly that the uncorrected test is anti-conservative on the autocorrelated
series this module is for. Sen's slope is returned because on a noisy
isotope curve the robust slope, not the p-value, is usually the quantity
the user wanted.

``orbital_forcing`` returns a table of Milankovitch periods with relative
amplitudes. READ ITS DOCSTRING BEFORE USING THE NUMBERS: the periods are the
canonical mean values, the amplitudes are rounded Laskar-style approximations
for building a *synthetic target curve*, and they are NOT the Laskar et al.
(2004) solution.

References:
    1. Muellersohn, F.J., Mysak, P.A. & Quaternary Research Group. 1999.
       REDFIT: a robust estimator of the red-noise spectrum of a time
       series. Climate Dynamics 14: 1-16.
    2. Watters, D.J. & Solomon, A.M. 2004. Redfit: identifying red-noise
       trends in paleoclimate time series. Journal of Geophysical Research
       109: C09014.
    3. Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M. 2015. Time
       Series Analysis: Forecasting and Control, 5th ed. Wiley. (Section
       2.5, prewhitening.)
    4. Mann, H.B. & Lees, J.M. 1996. Paleoclimate, chaos, and
       nonstationarity. Journal of Geophysical Research 101: 2615-2626.
    5. Thomson, D.J. 1982. Spectrum estimation and harmonic analysis.
       Proceedings of the IEEE 70: 1055-1096. (Multitaper method and
       significance test.)
    6. Percival, D.B. & Walden, A.T. 1993. Spectral Analysis for Physical
       Applications. Cambridge University Press. (DPSS construction as
       tridiagonal eigenvectors; equivalent noise bandwidth.)
    7. Wald, A. & Wolfowitz, J. 1943. A test whether two samples are from
       the same population. Annals of Mathematical Statistics 14: 147-162.
       (Runs test; exact conditional distribution.)
    8. Mann, H.B. 1945. Nonparametric tests against trend in time and
       power in time. Journal of the American Statistical Association 40:
       540-548.
    9. Kendall, M.G. 1975. Rank Correlation Methods. Griffin. (Tie
       correction to the variance of S.)
    10. Sen, P.K. 1968. Asymptotically efficient sequence ranking
        estimates. Journal of the American Statistical Association 63:
        1379-1389. (Sen's slope.)
    11. Hamed, K.H. & Rao, A.R. 1998. A modified Mann-Kendall trend test
        for autocorrelated data. Journal of Hydrology 204: 182-196.
    12. Pyper, S.C. & Peterman, A.D. 2004. Significance of correlations of
        time series: the effect of serial autocorrelation. Earth and
        Planetary Science Letters 217: 213-226.
    13. Shirer, W.R. 2008. Stratigraphic paleoclimate analysis: a
        multiproduct comparison of analytical methods. Stratigraphy 5:
        267-294. (Autoassociation function.)
    14. Berger, A. 1978. Long-term climatic variation: astronomical
        theory. In Understanding the Climate. Academic Press. (Present-day
        precession periods 19.1 and 23.4 kyr; the beat that produces the
        ~100 kyr band.)
    15. Laskar, J., Gastineau, M., Jounder, E. &bourgeois, F. 2004.
        Solution for the eccentricity and obliquity of the Earth's orbits
        over 5 Myr. Astronomy and Astrophysics 413: 755-767. (The real
        solution, which is time-dependent; NOT what ``orbital_forcing``
        tabulates.)
    16. Cramer, J.S. 2012. cyclostrat: a Web tool for stratigraphic
        cyclostratigraphy analysis. Bulletin of Geosciences 86: 109-113.
        (Analytic target curves and the standard Milankowitz band set.)

Author: PaleoAST Development Team
version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats
from scipy.linalg import eigh_tridiagonal
from scipy.special import comb

from config.i18n import _
from stratigraphy.spectral_analysis import _linear_detrend, ar1_theoretical_spectrum
from utils.exceptions import ComputationError, DataValidationError
from utils.statistics_core import make_rng

logger = logging.getLogger(__name__)

#: REDFIT window half-width as a fraction of the local frequency,
#: Muellersohn et al. (1999); the R ``redfit`` package default of 0.6.
REDFIT_WINDOW = 0.6

#: Minimum observations any estimator in this module will accept.
MIN_SAMPLES = 8


# =============================================================================
# Shared validation helpers
# =============================================================================


def _clean_pair(time: npt.NDArray, values: npt.NDArray, name: str = "series") -> tuple[npt.NDArray, npt.NDArray]:
    """Drop non-finite pairs and sort by time; require enough left over."""
    t = np.asarray(time, dtype=float).flatten()
    v = np.asarray(values, dtype=float).flatten()
    if t.size != v.size:
        raise DataValidationError(
            _("time and values must have the same length; got {0} and {1}").format(t.size, v.size)
        )
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if t.size < MIN_SAMPLES:
        raise DataValidationError(
            _("'{0}' needs at least {1} finite points; got {2}").format(name, MIN_SAMPLES, t.size),
            details={"n_finite": int(t.size)},
        )
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def _clean_series(values: npt.NDArray, name: str = "values", min_samples: int = MIN_SAMPLES) -> npt.NDArray:
    """Validate a single 1-D signal and require non-zero variance."""
    v = np.asarray(values, dtype=float).flatten()
    if v.size < min_samples:
        raise DataValidationError(
            _("'{0}' needs at least {1} points; got {2}").format(name, min_samples, v.size),
            details={"n": int(v.size)},
        )
    ok = np.isfinite(v)
    if not np.any(ok):
        raise DataValidationError(_("'{0}' contains no finite values").format(name))
    v = v[ok]
    if np.ptp(v) == 0.0:
        raise DataValidationError(_("'{0}' is constant, so it has no structure to analyse").format(name))
    return v


def _check_increasing(time: npt.NDArray, name: str = "time") -> npt.NDArray:
    """Require a strictly increasing time axis; return the median step."""
    if time.size < 2:
        raise DataValidationError(_("time must contain at least 2 points"))
    steps = np.diff(time)
    if np.any(steps <= 0):
        raise DataValidationError(
            _(
                "time must be strictly increasing after sorting; a non-monotonic "
                "axis has no single dt for a periodogram"
            ),
            details={"n_non_increasing": int(np.sum(steps <= 0))},
        )
    return float(np.median(steps))


def _ar1_phi(values: npt.NDArray) -> float:
    """Yule-Walker (conditional MLE) AR(1) coefficient of a zero-mean series.

    ``phi = sum(x_t x_(t-1)) / sum(x_t^2)`` over ``t = 1..n-1``, which is the
    conditional least-squares estimator ``ar.yw`` uses and therefore the one
    the REDFIT papers assume. Clipped to (-0.99, 0.99) so the spectrum stays
    finite.
    """
    v = np.asarray(values, dtype=float) - float(np.mean(values))
    denom = float(np.sum(v[1:] ** 2))
    if denom <= 0:
        return 0.0
    return float(np.clip(float(np.sum(v[:-1] * v[1:])) / denom, -0.99, 0.99))


def _normalised_periodogram(values: npt.NDArray, dt: float) -> tuple[npt.NDArray, npt.NDArray]:
    """Periodogram on a uniform grid, normalised to E[S] = 1 for white noise.

    ``S_k = |X_k|^2 / (N sigma^2)`` with ``X_k`` the DFT of the mean-removed
    series. This is the same scale as
    ``spectral_analysis.ar1_theoretical_spectrum``, which returns 1.0 for
    every frequency when ``phi = 0``; the two are therefore directly
    comparable, which is what makes the REDFIT rotation well defined.
    """
    n = values.size
    v = np.asarray(values, dtype=float) - float(np.mean(values))
    sigma2 = float(np.mean(v**2))
    if sigma2 <= 0:
        raise ComputationError("Periodogram is undefined for a series with zero variance")
    spectrum = np.abs(np.fft.rfft(v)) ** 2 / (n * sigma2)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs[1:], spectrum[1:]


def _resample_uniform(
    time: npt.NDArray, values: npt.NDArray, name: str = "series"
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """Linearly interpolate onto a uniform grid of median step.

    REDFIT is defined for evenly sampled records (Watters & Solomon 2004
    §2). Uneven sampling is the norm in a core record, and a Lomb-Scargle
    style fit is not available inside the classical rotation, so the record
    is resampled first. The interpolation is linear and the step is the
    median, which is robust to one deep gap; the caller is told when it
    happened.
    """
    dt = _check_increasing(time, name)
    steps = np.diff(time)
    if float(np.ptp(steps)) <= 1.0e-9 * dt:
        return time, values, dt
    n = int(np.floor((time[-1] - time[0]) / dt)) + 1
    if n < MIN_SAMPLES:
        raise DataValidationError(
            _(
                "'{0}' spans too few samples at its own median step after "
                "resampling; reduce the sampling density or the time span"
            ).format(name)
        )
    grid = time[0] + dt * np.arange(n)
    resampled = np.interp(grid, time, values)
    logger.info(
        "%s: resampled %d unevenly spaced points onto a uniform grid of %d points (dt=%.6g)",
        name,
        time.size,
        n,
        dt,
    )
    return grid, resampled, dt


def _redfit_smooth(
    values: npt.NDArray, widths: npt.NDArray, weight_power: float = 1.0, chunk: int = 512
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Tricube-weighted local mean in the frequency domain (REDFIT window).

    ``widths[k]`` is the number of grid points in the window centred on
    ``k``. The tricube weight ``(1 - u^2)^3`` is applied to the index offset
    ``u = j / (w/2)``, which is exactly the weight LOWESS uses; LOWESS with
    an intercept-only local fit degenerates to this weighted mean, and the
    difference from a full local-linear fit is O(1/w), far below the window's
    own width. Windows are applied in memory-bounded chunks because the
    windows grow with frequency (``w_k`` is proportional to ``f_k``) and a
    dense gather would be quadratic.

    ``weight_power`` raises the kernel to that power, which turns the
    weighted mean into the mean under ``w^weight_power`` weights. That is
    what lets the caller get ``sum w^2 v_j`` from the same kernel, and the
    resulting variance back to the right scale.

    A 2-D input is smoothed along its LAST axis, so a whole stack of Monte
    Carlo replicates is filtered in one pass.

    Returns
    -------
    (smoothed, sum_w, sum_w2):
        The weighted mean at each point, and the window's own weight sums.
        ``sum_w`` is what turns a weighted mean back into a weighted sum,
        and ``sum_w`` and ``sum_w2`` give the Kish effective count
        ``sum_w^2 / sum_w2`` of ordinates the window retains, which is the
        degrees of freedom of the null.
    """
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    n = flat.shape[-1]
    out = np.empty_like(flat)
    sums = np.empty(n, dtype=float)
    squares = np.empty(n, dtype=float)
    half_max = int(np.max(widths)) // 2 + 1
    index = np.arange(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        here = flat[:, start:stop]
        block = np.zeros_like(here)
        total = np.zeros(here.shape[1], dtype=float)
        total2 = np.zeros(here.shape[1], dtype=float)
        for offset in range(-half_max, half_max + 1):
            # Each window's weight vanishes at its own edge, so a narrow
            # window contributes nothing outside its own span.
            idx = index[start:stop] + offset
            valid = (idx >= 0) & (idx < n)
            idx_c = np.clip(idx, 0, n - 1)
            u = offset / np.maximum(widths[start:stop] / 2.0, 1.0)
            inside = np.abs(u) < 1.0
            wt = np.where(inside, (1.0 - u**2) ** 3, 0.0) ** float(weight_power)
            wt = np.where(valid, wt, 0.0)
            block += flat[:, idx_c] * wt
            total += wt
            total2 += wt**2
        out[:, start:stop] = np.where(total > 0, block / np.maximum(total, 1e-300), here)
        sums[start:stop] = total
        squares[start:stop] = total2
    return out.reshape(values.shape), sums, squares


# =============================================================================
# 1. Autocorrelation
# =============================================================================


@dataclass
class ACFResult:
    """Sample autocorrelation function with Bartlett confidence bands.

    Attributes:
        lags: Lag index, 0..nlags, in samples.
        acf: Autocorrelation at each lag, acf[0] == 1.
        confidence: Half-width of the two-sided band at the requested level.
        n: Number of observations the estimate came from.
        nlags: Largest lag evaluated.
        n_effective: Effective sample size ``n * (1 - r1) / (1 + r1)``
            (Bartlett 1946), which is the count that actually carries
            information once the series is serially correlated.
        band_lower / band_upper: The band, expanded for plotting.
    """

    lags: npt.NDArray
    acf: npt.NDArray
    confidence: float
    n: int
    nlags: int
    n_effective: float
    band_lower: npt.NDArray
    band_upper: npt.NDArray

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Autocorrelation"),
            "=" * 40,
            _("Points: {0}").format(self.n),
            _("Lags: {0}").format(self.nlags),
            _("Effective N: {0}").format(f"{self.n_effective:.1f}"),
            _("Confidence band: +/-{0}").format(f"{self.confidence:.4f}"),
        ]
        if self.acf.size > 1:
            peak = int(np.argmax(np.abs(self.acf[1:]))) + 1
            lines.append(_("Lag-1: {0}").format(f"{self.acf[1]:.4f}"))
            lines.append(_("Strongest |rho| at lag: {0}").format(peak))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "lags": [int(k) for k in self.lags],
            "acf": [float(v) for v in self.acf],
            "confidence": float(self.confidence),
            "n": int(self.n),
            "nlags": int(self.nlags),
            "n_effective": float(self.n_effective),
        }


def autocorrelation(
    values: npt.NDArray,
    max_lag: int | None = None,
    nlags: int | None = None,
    confidence: float = 0.95,
) -> ACFResult:
    """Sample autocorrelation with Bartlett (1946) confidence bands.

    The estimator is the standard biased one,

        r_k = sum_{t=1}^{n-k} (x_t - xbar)(x_(t+k) - xbar)
              / sum_{t=1}^{n} (x_t - xbar)^2

    and the band is the Bartlett form ``+/- z_(1+confidence)/2 / sqrt(n)``,
    which is exact for a white-noise process and is the convention every
    geostatistics text uses for AR(1)-ish series.

    Parameters
    ----------
    values:
        The series, in sample order. Assumed evenly spaced.
    max_lag, nlags:
        Largest lag to evaluate; either name is accepted because the rest of
        the codebase uses both. Defaults to ``n // 2``.
    confidence:
        Two-sided confidence level for the band.

    Returns
    -------
    ACFResult
    """
    x = _clean_series(values, "autocorrelation input")
    n = x.size
    lag_max = max_lag if max_lag is not None else nlags
    if lag_max is None:
        lag_max = n // 2
    lag_max = int(lag_max)
    if lag_max < 1 or lag_max >= n:
        raise DataValidationError(
            _("max_lag must lie in [1, n-1]; got {0} for n = {1}").format(lag_max, n),
            details={"max_lag": lag_max, "n": n},
        )
    if not 0.0 < confidence < 1.0:
        raise DataValidationError(_("confidence must lie strictly between 0 and 1"))

    v = x - float(np.mean(x))
    denom = float(np.sum(v**2))
    acf = np.empty(lag_max + 1, dtype=float)
    acf[0] = 1.0
    for k in range(1, lag_max + 1):
        acf[k] = float(np.sum(v[:-k] * v[k:])) / denom

    z = float(sp_stats.norm.ppf(0.5 + 0.5 * confidence))
    conf = z / np.sqrt(n)
    r1 = float(acf[1])
    n_eff = float(n * (1.0 - r1) / (1.0 + r1)) if r1 < 1.0 else 0.0
    lags = np.arange(lag_max + 1)
    return ACFResult(
        lags=lags,
        acf=acf,
        confidence=conf,
        n=n,
        nlags=lag_max,
        n_effective=n_eff,
        band_lower=-conf,
        band_upper=conf,
    )


# =============================================================================
# 2. AR(1) prewhitening
# =============================================================================


@dataclass
class PrewhitenResult:
    """Outcome of the Box & Jenkins (1970) prewhitening step.

    Attributes:
        phi: Fitted AR(1) coefficient.
        mode: ``"prewhiten"`` (inverse filter ``x_t - phi x_(t-1)``) or
            ``"difference"`` (first differencing).
        d: Differencing order actually applied. 0 for the prewhitening
            branch.
        transformed: The transformed series, length ``n - d - 1``.
        acf: ACF of the transformed series, which is what "prewhitening
            worked" means.
        acf_before: ACF of the input, for comparison.
        variance_reduction: Var(input) / Var(transformed).
        aic_prewhiten / aic_difference: The two AIC values compared.
        criterion: The selection rule, named so a result is auditable.
    """

    phi: float
    mode: str
    d: int
    transformed: npt.NDArray
    acf: npt.NDArray
    acf_before: npt.NDArray
    variance_reduction: float
    aic_prewhiten: float
    aic_difference: float
    criterion: str = "AIC on the transformed series"

    def summary(self) -> str:
        """Generate summary text."""
        aic = _("AIC prewhiten / difference: {0} / {1}").format(
            f"{self.aic_prewhiten:.2f}", f"{self.aic_difference:.2f}"
        )
        return (
            f"{_('AR(1) Prewhitening')}\n"
            f"{'=' * 40}\n"
            f"{_('phi: {0}').format(f'{self.phi:.4f}')}\n"
            f"{_('Mode: {0}').format(self.mode)}\n"
            f"{_('Differencing order: {0}').format(self.d)}\n"
            f"{_('Lag-1 before: {0}').format(f'{self.acf_before[1]:.4f}')}\n"
            f"{_('Lag-1 after: {0}').format(f'{self.acf[1]:.4f}')}\n"
            f"{_('Criterion: {0}').format(self.criterion)}\n"
            f"{aic}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view, without the transformed series."""
        return {
            "phi": float(self.phi),
            "mode": self.mode,
            "d": int(self.d),
            "lag1_before": float(self.acf_before[1]),
            "lag1_after": float(self.acf[1]),
            "variance_reduction": float(self.variance_reduction),
            "aic_prewhiten": float(self.aic_prewhiten),
            "aic_difference": float(self.aic_difference),
            "criterion": self.criterion,
        }


def ar1_prewhiten(values: npt.NDArray, d: int = 1) -> PrewhitenResult:
    """Remove the AR(1) structure by inverse filtering or differencing.

    Box & Jenkins (1970, section 2.5) allow two prewhitening steps and
    require that the choice be reported:

    ``prewhiten``
        The inverse filter ``z_t = x_t - phi x_(t-1)``, with ``phi`` the
        Yule-Walker estimate. Exact for a genuine AR(1), where it reduces
        the series to white noise.
    ``difference``
        First (or ``d``-th) differencing, ``z_t = x_t - x_(t-d)``. Exact for
        a unit root, i.e. a series that wanders rather than fluctuates.

    **Which one is used here, and why.** The two are compared by Akaike's
    information criterion on the transformed series,
    ``AIC = n' * ln(sigma^2) + 2k`` with ``k = 1`` for each (one estimated
    ``phi`` in one case, ``d = 1`` in the other), and the smaller wins. AIC
    rather than a significance test on ``phi`` because Box & Jenkins reject
    prewhitening on the *predictive* criterion, not on whether ``phi`` is
    individually distinguishable from zero -- a long, weakly persistent
    record has a small ``phi`` that is still worth removing, and AIC sees
    that while a z-test on ``phi`` does not. The identity case
    (``phi -> 0``) is the limit of the prewhitening branch, so ``mode`` is
    always one of the two names above.

    Parameters
    ----------
    values:
        The series, in sample order.
    d:
        Differencing order to use if the difference branch is selected.

    Returns
    -------
    PrewhitenResult
    """
    if d < 1:
        raise DataValidationError(_("d must be >= 1"), details={"d": int(d)})
    x = _clean_series(values, "prewhitening input", min_samples=MIN_SAMPLES + d)
    phi = _ar1_phi(x)

    centered = x - float(np.mean(x))
    prewhitened = centered[1:] - phi * centered[:-1]
    differenced = np.diff(centered, n=d)
    var_pw = float(np.var(prewhitened))
    var_df = float(np.var(differenced))
    if var_pw <= 0 or var_df <= 0:
        raise ComputationError(
            "Prewhitening produced a constant series; the input is degenerate",
            details={"var_prewhiten": var_pw, "var_difference": var_df},
        )

    aic_pw = prewhitened.size * np.log(var_pw) + 2.0
    aic_df = differenced.size * np.log(var_df) + 2.0
    if aic_df < aic_pw:
        mode, transformed, order = "difference", differenced, d
    else:
        mode, transformed, order = "prewhiten", prewhitened, 0

    acf_after = autocorrelation(transformed, max_lag=min(transformed.size // 2, 50))
    acf_before = autocorrelation(x, max_lag=min(acf_after.nlags, x.size // 2))
    return PrewhitenResult(
        phi=phi,
        mode=mode,
        d=order,
        transformed=transformed,
        acf=acf_after.acf,
        acf_before=acf_before.acf,
        variance_reduction=float(np.var(x) / np.var(transformed)),
        aic_prewhiten=float(aic_pw),
        aic_difference=float(aic_df),
    )


def nlags_of(acf: ACFResult) -> int:
    """Largest lag in an :class:`ACFResult` (small helper for pairing sizes)."""
    return int(acf.nlags)


# =============================================================================
# 3. REDFIT
# =============================================================================


@dataclass
class RedfitResult:
    """REDFIT rotation of a periodogram against an AR(1) red-noise null.

    Attributes:
        frequencies: Cycles per time unit of the ``time`` argument.
        periods: ``1 / frequencies``, same units as ``time``.
        spectrum: The smoothed REDFIT curve
            ``S_REDFIT(f) = (S_data(f) - S_AR1(f)) / sqrt(S_AR1(f))``.
            Mean 0 under a pure red-noise null; positive = more power than
            red noise explains.
        spectrum_raw: The same rotation before the window is applied.
        ar1_spectrum: ``ar1_theoretical_spectrum`` evaluated on this grid.
        ar1_smoothed: The AR(1) curve averaged over the REDFIT window, which
            is the correct null mean for the window.
        spectrum_data: The periodogram averaged over the same window, the
            observed side of the same comparison.
        noise_sd: Standard deviation of that windowed periodogram under the
            AR(1) null, from :func:`_ar1_periodogram_variance`. Exposed
            because ``(spectrum_data - ar1_smoothed) / noise_sd`` is the
            statistic the FAP is computed from, and a reader who doubts the
            FAP should be able to see it.
        window_points: Number of grid points in the REDFIT window per
            frequency.
        effective_dof: Kish effective count of ordinates the window retains,
            ``(sum w)^2 / sum w^2``.
        fap: One-sided false-alarm probability per frequency, from the
            parametric bootstrap over the AR(1) null,
            ``(1 + count) / (n_simulations + 1)``.
        n_simulations: Replicates in that bootstrap; 0 means the FAP was
            skipped and every frequency reports 1.
        random_seed: Seed used for the bootstrap, carried so a reported
            false-alarm level can be reproduced.
        phi: Fitted AR(1) coefficient.
        significant: ``fap <= fap_level``.
        significant_periods: Periods of the significant local maxima, ordered
            by REDFIT value (strongest first), not by FAP. FAP is a property
            of how surprising a frequency is; REDFIT value is how much power
            the window actually found there, and "the strongest cycle in this
            record" is the second question. With a wide window many
            neighbouring frequencies share a plateau, and ordering that
            plateau by FAP returns whichever of them the bootstrap happened
            to draw most extreme, which is noise.
        time_span: Span of the (resampled) record in the input time unit.
        dt: Uniform sample step after resampling.
    """

    frequencies: npt.NDArray
    periods: npt.NDArray
    spectrum: npt.NDArray
    spectrum_raw: npt.NDArray
    ar1_spectrum: npt.NDArray
    ar1_smoothed: npt.NDArray
    spectrum_data: npt.NDArray
    noise_sd: npt.NDArray
    window_points: npt.NDArray
    effective_dof: npt.NDArray
    fap: npt.NDArray
    phi: float
    significant: npt.NDArray
    significant_periods: list[dict[str, float]]
    time_span: float
    dt: float
    fap_level: float = 0.05
    n_simulations: int = 0
    random_seed: int | None = None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("REDFIT Analysis"),
            "=" * 50,
            _("AR(1) phi: {0}").format(f"{self.phi:.4f}"),
            _("Frequencies: {0}").format(self.frequencies.size),
            _("Span: {0}").format(f"{self.time_span:.6g}"),
            _("Significant frequencies at FAP <= {0}: {1}").format(self.fap_level, int(np.sum(self.significant))),
        ]
        for i, peak in enumerate(self.significant_periods[:8]):
            lines.append(
                _("  {0}. period = {1} (FAP = {2})").format(i + 1, f"{peak['period']:.6g}", f"{peak['fap']:.3e}")
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view, without the full spectral arrays."""
        return {
            "phi": float(self.phi),
            "fap_level": float(self.fap_level),
            "time_span": float(self.time_span),
            "dt": float(self.dt),
            "n_frequencies": int(self.frequencies.size),
            "n_significant": int(np.sum(self.significant)),
            "n_simulations": int(self.n_simulations),
            "random_seed": self.random_seed,
            "significant_periods": list(self.significant_periods),
        }


def _redfit_widths(frequencies: npt.NDArray, window: float, max_window: int) -> tuple[npt.NDArray, npt.NDArray]:
    """REDFIT window widths (odd, in grid points) and their effective dof.

    Following Muellersohn et al. (1999), the window at frequency ``f_k``
    contains ``2 * window * f_k / df`` grid points, i.e. a fraction
    ``window`` of the frequency itself; the R ``redfit`` default of 0.6 is
    used here. The minimum is 5 rather than 3 because the tricube weight
    ``(1 - u^2)^3`` vanishes at ``|u| = 1``, so a 3-point window puts all
    its mass on the centre ordinate and smooths nothing.

    The widths are returned on their own because the false-alarm probability
    is derived from the exact AR(1) periodogram variance (see
    :func:`redfit`) rather than from an effective degrees-of-freedom count,
    so nothing downstream needs the weights themselves.
    """
    df = float(np.median(np.diff(frequencies)))
    if df <= 0:
        raise ComputationError("REDFIT needs a strictly increasing frequency grid")
    raw = 2.0 * window * frequencies / df
    widths = np.clip(np.round(raw), 5, max(5, int(max_window)))
    return np.where(widths % 2 == 0, widths + 1, widths).astype(int)


def _ar1_periodogram_variance(frequencies: npt.NDArray, phi: float, dt: float, n_samples: int) -> npt.NDArray:
    """Exact variance of the normalised AR(1) periodogram, for a unit record.

    For a Gaussian stationary record the periodogram variance is the
    modulus-squared transform of the squared autocovariances
    (Priestley 1981, eq. 2.53):

        Var[S(f)] = (1 / n^2) sum_m R(m)^2 exp(-4 pi i f m)

    and for an AR(1) with ``R(m) = phi^|m|`` that sum is a Poisson kernel in
    ``phi^2``:

        Var[S(f)] = (1 - phi^4) / (n^2 (1 - 2 phi^2 cos(4 pi f dt) + phi^4))

    which is what makes the false-alarm test exact rather than a
    degrees-of-freedom approximation. The whitened end (f near Nyquist) is
    where this approaches the exponential variance 1/n^2, and the coloured
    end is where a dof-based chi-square badly mis-scales.

    Parameters
    ----------
    frequencies:
        Cycles per time unit.
    phi:
        AR(1) coefficient.
    dt:
        Sample step in the same time unit.
    n_samples:
        Record length; the 1/n^2 is the usual periodogram normalisation.

    Returns
    -------
    ndarray
        Variance of the normalised periodogram at each frequency.
    """
    a2 = phi**2
    denom = 1.0 - 2.0 * a2 * np.cos(4.0 * np.pi * frequencies * dt) + a2**2
    return (1.0 - a2**2) / (n_samples**2 * np.maximum(denom, 1e-300))


def _simulate_ar1(n: int, phi: float, rng: np.random.Generator) -> npt.NDArray:
    """One standardised AR(1) record of length ``n`` from ``rng``."""
    innovation = rng.normal(scale=np.sqrt(max(1.0 - phi**2, 0.0)), size=n)
    series = np.empty(n, dtype=float)
    series[0] = rng.normal()
    for i in range(1, n):
        series[i] = phi * series[i - 1] + innovation[i]
    series -= float(np.mean(series))
    sd = float(np.std(series))
    return series / sd if sd > 0 else series


def _redfit_bootstrap_fap(
    excess: npt.NDArray,
    sd_smoothed: npt.NDArray,
    n_samples: int,
    phi: float,
    dt: float,
    freqs_all: npt.NDArray,
    power_all: npt.NDArray,
    frequencies: npt.NDArray,
    widths: npt.NDArray,
    ar1_smoothed: npt.NDArray,
    n_simulations: int,
    random_seed: int | None,
) -> tuple[npt.NDArray, int]:
    """Empirical one-sided false-alarm probability of the windowed excess.

    Every simulated replicate goes through the identical pipeline -- same
    periodogram normalisation, same interpolation onto the same grid, same
    window -- and is reduced to the same standardised excess, so the
    comparison is between like and like at every frequency. The p-value uses
    the ``(1 + count) / (n + 1)`` form of Phipson & Smyth (2010), which is
    the same form :func:`utils.statistics_core.permutation_pvalue` uses, so
    an unexceeded frequency reports ``1/(n_sim + 1)`` rather than zero.

    The whole stack of replicates is filtered in one pass through
    :func:`_redfit_smooth`, which is what keeps 100 red-noise records of a
    few thousand samples affordable.
    """
    safe_sd = np.where(sd_smoothed > 0, sd_smoothed, np.finfo(float).tiny)
    observed = excess / safe_sd
    if n_simulations < 1:
        return np.full_like(observed, 1.0), 0

    rng = make_rng(random_seed, context="REDFIT")
    stack = np.empty((n_simulations, frequencies.size), dtype=float)
    for i in range(n_simulations):
        sim = _simulate_ar1(n_samples, phi, rng)
        _f, p = _normalised_periodogram(sim, dt)
        stack[i] = np.interp(frequencies, freqs_all, p, left=0.0, right=0.0)
    smoothed, _w1, _w2 = _redfit_smooth(stack, widths)
    null = (smoothed - ar1_smoothed) / safe_sd
    # (1 + count) / (n + 1): a frequency nothing in the null matched still
    # reports 1/(n+1), never 0.
    count = np.sum(null >= observed[None, :], axis=0)
    return (1.0 + count) / (n_simulations + 1.0), int(n_simulations)


def redfit(
    time: npt.NDArray,
    values: npt.NDArray,
    detrend: bool = True,
    n_periods: int | None = None,
    window: float = REDFIT_WINDOW,
    fap_level: float = 0.05,
    n_simulations: int = 100,
    random_seed: int | None = None,
) -> RedfitResult:
    """REDFIT: test a periodogram against a subtracted, scaled AR(1) null.

    The procedure (Muellersohn et al. 1999; Watters & Solomon 2004):

    1. Remove a least-squares linear trend, and resample the record onto a
       uniform grid (the classical rotation is defined for evenly sampled
       data, so uneven records are interpolated first).
    2. Fit an AR(1) and evaluate its spectrum. This step calls
       :func:`spectral_analysis.ar1_theoretical_spectrum` -- the curve is
       not re-derived here, so the periodogram null and the rotation
       denominator are the same function.
    3. Compute the normalised periodogram ``S_data``, which has unit mean
       under white noise, the same scale as the AR(1) curve.
    4. Rotate: ``S_REDFIT(f) = (S_data(f) - S_AR1(f)) / sqrt(S_AR1(f))``.
       This is the whole point of REDFIT -- the red background is removed
       and what remains is divided by the standard deviation it was removed
       with, so a long-period wander stops dominating the plot.
    5. Smooth in the frequency domain with a tricube (LOWESS) window whose
       width is a fraction ``window`` of the local frequency.
    6. Assign a false-alarm probability by parametric bootstrap over the
       AR(1) null (Muellersohn et al. 1999; Watters & Solomon 2004): the
       windowed periodogram of the record is compared, frequency by
       frequency, with the same statistic computed on simulated red-noise
       records of the same length, ``phi`` and grid. See
       :func:`_redfit_bootstrap_fap` for why a closed-form null was not
       used.

    Parameters
    ----------
    time, values:
        The record. ``time`` must be strictly increasing; uneven spacing is
        tolerated and resampled.
    detrend:
        Subtract a least-squares linear trend first (recommended; see
        ``spectral_analysis._linear_detrend``).
    n_periods:
        Number of frequencies. Defaults to 5 grid points per resolution
        element, capped at 4000. Values far above the record's own number
        of resolution elements are interpolated rather than measured, so
        they add no independent information; the peak locations do not
        depend on them but the false-alarm count does.
    window:
        REDFIT window half-width as a fraction of the local frequency.
    fap_level:
        Level at which a peak is called significant.
    n_simulations:
        Red-noise replicates in the false-alarm bootstrap. 0 skips the
        bootstrap and reports a false-alarm probability of 1 everywhere,
        which is the right choice when only the shape of the rotated
        spectrum is wanted.
    random_seed:
        Seed for the bootstrap. Pass one for a reproducible false-alarm
        level; without one the result is not reproducible and
        :func:`utils.statistics_core.make_rng` warns.

    Returns
    -------
    RedfitResult
    """
    t, v = _clean_pair(time, values, "redfit input")
    if len(t) < 2 * MIN_SAMPLES:
        raise DataValidationError(_("redfit needs at least {0} points; got {1}").format(2 * MIN_SAMPLES, t.size))
    if not 0.0 < window < 5.0:
        raise DataValidationError(
            _("window must lie in (0, 5); got {0}").format(window),
            details={"window": window},
        )
    grid, series, dt = _resample_uniform(t, v, "redfit input")
    if detrend:
        series = _linear_detrend(series)

    freqs_all, power_all = _normalised_periodogram(series, dt)
    f_min, f_max = float(freqs_all[0]), float(freqs_all[-1])
    span = float(grid[-1] - grid[0])
    n_elements = int(max(2, round(f_max * span)))
    n_freq = int(n_periods) if n_periods is not None else min(4000, max(200, 5 * n_elements))
    if n_freq < 32:
        raise DataValidationError(_("n_periods must be >= 32 for a smoothed spectrum; got {0}").format(n_freq))
    frequencies = np.linspace(f_min, f_max, n_freq)
    if n_freq > 3 * freqs_all.size:
        logger.warning(
            "redfit: n_periods=%d is far above the record's %d resolution elements; "
            "the extra grid points are interpolated, so false-alarm levels are "
            "optimistic there",
            n_freq,
            freqs_all.size,
        )

    # Linear interpolation of the DFT ordinate onto the finer REDFIT grid:
    # both are estimates of the same continuous spectrum, and the DFT grid
    # is the coarser of the two by construction.
    power = np.interp(frequencies, freqs_all, power_all, left=0.0, right=0.0)

    phi = _ar1_phi(series)
    ar1 = ar1_theoretical_spectrum(frequencies, phi, dt)
    rotation = (power - ar1) / np.sqrt(ar1)

    max_window = max(5, n_freq // 4)
    widths = _redfit_widths(frequencies, window, max_window)
    spectrum, _w1, _w2 = _redfit_smooth(rotation, widths)
    # Same window applied to the periodogram and to the AR(1) null mean, so
    # the excess and its reference are averaged identically.
    ar1_smoothed, sum_w, sum_w2 = _redfit_smooth(ar1, widths)
    power_smoothed, _w1, _w2 = _redfit_smooth(power, widths)
    var_w2, _w1, _w2 = _redfit_smooth(
        _ar1_periodogram_variance(frequencies, phi, dt, grid.size), widths, weight_power=2.0
    )
    sd_smoothed = np.sqrt(np.maximum(var_w2, 0.0) * sum_w**2)
    effective_dof = np.where(sum_w2 > 0, sum_w**2 / np.maximum(sum_w2, 1e-300), 1.0)
    excess = power_smoothed - ar1_smoothed

    # False-alarm probability by parametric bootstrap over the AR(1) null,
    # which is how the REDFIT papers obtain it. The standardised excess
    # (spectrum_data - ar1_smoothed) / noise_sd is compared, frequency by
    # frequency, with the same statistic computed on simulated red-noise
    # records of the same length, same phi and same grid.
    #
    # A closed-form null was tried first and rejected: the windowed
    # periodogram's variance is NOT proportional to the square of its mean
    # away from the whitened end, so a chi-square with the window's
    # effective degrees of freedom mis-scales the false-alarm rate by more
    # than an order of magnitude across the spectrum (measured, not
    # assumed), and the normal approximation on the exact variance is
    # anti-conservative for the same reason: the periodogram of a strongly
    # coloured process is log-normal, so its upper tail is heavier than the
    # Gaussian the central limit theorem supplies. Simulating the null has
    # no such free parameter left to get wrong.
    fap, n_sim_used = _redfit_bootstrap_fap(
        excess,
        sd_smoothed,
        grid.size,
        phi,
        dt,
        freqs_all,
        power_all,
        frequencies,
        widths,
        ar1_smoothed,
        n_simulations,
        random_seed,
    )
    significant = fap <= fap_level

    peaks: list[dict[str, float]] = []
    for k in range(1, frequencies.size - 1):
        if not significant[k]:
            continue
        if not (spectrum[k] >= spectrum[k - 1] and spectrum[k] >= spectrum[k + 1]):
            continue
        if spectrum[k] <= 0.0:
            continue
        peaks.append(
            {
                "period": float(1.0 / frequencies[k]),
                "frequency": float(frequencies[k]),
                "redfit": float(spectrum[k]),
                "fap": float(fap[k]),
                "dof": float(effective_dof[k]),
            }
        )
    peaks.sort(key=lambda p: -p["redfit"])

    logger.info(
        "redfit: n=%d, phi=%.4f, %d/%d frequencies significant at FAP<=%.3f, strongest period=%s",
        grid.size,
        phi,
        int(np.sum(significant)),
        frequencies.size,
        fap_level,
        f"{peaks[0]['period']:.6g}" if peaks else "none",
    )
    return RedfitResult(
        frequencies=frequencies,
        periods=1.0 / frequencies,
        spectrum=spectrum,
        spectrum_raw=rotation,
        ar1_spectrum=ar1,
        ar1_smoothed=ar1_smoothed,
        spectrum_data=power_smoothed,
        noise_sd=sd_smoothed,
        window_points=widths,
        effective_dof=effective_dof,
        fap=fap,
        phi=phi,
        significant=significant,
        significant_periods=peaks,
        time_span=span,
        dt=dt,
        fap_level=fap_level,
        n_simulations=n_sim_used,
        random_seed=random_seed,
    )


# =============================================================================
# 4. Multitaper spectral estimation
# =============================================================================


@dataclass
class MultitaperResult:
    """Thomson (1982) multitaper spectrum.

    Attributes:
        frequencies: Cycles per sample, index 1..n//2 (0 excluded: the DC bin
            carries no information for a mean-removed series).
        periods: ``1 / frequencies``, in samples.
        power: Thomson-weighted average of the taper products, normalised to
            unit mean under white noise.
        tapers: The ``(K, n)`` DPSS tapers actually used.
        eigenvalues: Concentration eigenvalues of those tapers.
        bandwidth: Equivalent noise bandwidth of the estimator, in cycles
            per sample,
            ``(K + 1) / (2 n) * sum(l^2) / sum(l)^2``.
        n_periods: Number of frequencies.
        n_tapers: ``K``, the number of tapers.
        equivalent_n_dof: ``2 * sum(l)^2 / sum(l^2)``, which is ``2K`` when
            the tapers are near-perfectly concentrated.
        fap: One-sided white-noise false-alarm probability per frequency,
            ``chi2.sf(nu * power, nu)``.
        fap_level: Level used for the significance mask.
        significant: ``fap <= fap_level``.
        n: Number of samples in the record.
    """

    frequencies: npt.NDArray
    periods: npt.NDArray
    power: npt.NDArray
    tapers: npt.NDArray
    eigenvalues: npt.NDArray
    bandwidth: float
    n_tapers: int
    equivalent_n_dof: float
    fap: npt.NDArray
    significant: npt.NDArray
    fap_level: float
    n: int

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Multitaper Spectrum"),
            "=" * 50,
            _("Samples: {0}").format(self.n),
            _("Tapers (K): {0}").format(self.n_tapers),
            _("Equivalent noise bandwidth: {0:.6g} cycles/sample").format(self.bandwidth),
            _("Equivalent degrees of freedom: {0}").format(f"{self.equivalent_n_dof:.2f}"),
            _("Eigenvalue range: {0:.4f} - {1:.4f}").format(
                float(self.eigenvalues.min()), float(self.eigenvalues.max())
            ),
            _("Significant frequencies at FAP <= {0}: {1}").format(self.fap_level, int(np.sum(self.significant))),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view, without the tapers or the full spectrum."""
        return {
            "n": int(self.n),
            "n_tapers": int(self.n_tapers),
            "n_periods": int(self.power.size),
            "bandwidth": float(self.bandwidth),
            "equivalent_n_dof": float(self.equivalent_n_dof),
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "n_significant": int(np.sum(self.significant)),
            "fap_level": float(self.fap_level),
        }


def dpss_tapers(n: int, nw: float, n_tapers: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Discrete prolate spheroidal sequences and their concentration eigenvalues.

    Percival & Walden (1993), Appendix A and section 4.2, p. 390.

    **The tapers** are the eigenvectors of the symmetric tridiagonal matrix

        B_tt    = ((n - 1 - 2t) / 2)^2 * cos(2 pi W)      t = 0 .. n-1
        B_t,t+1 = t (n - t) / 2                            t = 1 .. n-1

    belonging to its ``n_tapers`` LARGEST eigenvalues, where ``W = nw / n``
    is the standardised half bandwidth in cycles per sample -- the time-
    bandwidth product divided by the record length. That division is the
    easy thing to get wrong: the ``cos`` factor is what makes ``B``'s
    spectrum asymmetric, and with ``cos(2 pi nw)`` instead of
    ``cos(2 pi nw / n)`` the eigenvectors are still orthonormal and still
    symmetric about the centre, and none of them are prolate.

    **The concentration eigenvalues are NOT the eigenvalues of ``B``** --
    that is the other mistake this function exists to avoid, and it is an
    easy one to make because the two look interchangeable in a code sketch.
    They are the eigenvalues of the concentration operator
    ``C = int_{-1}^{1} exp(i omega t) dt`` restricted to each taper, and
    they are evaluated here with the autocorrelation-sequence identity of
    Percival & Walden (1993, p. 390),

        lambda_k = sum_{m=0}^{n-1} R_v(m) * 4 W sinc(2 W m),  R_v[0] = 2 W

    with ``R_v(m) = sum_t v(t) v(t + m)`` obtained by FFT. Both are built
    with :mod:`scipy.linalg` primitives rather than taken from
    :func:`scipy.signal.windows.dpss`, so the construction is visible here;
    the test suite checks both against that independent implementation.

    Returns
    -------
    (eigenvalues, tapers):
        Arrays of shape ``(n_tapers,)`` and ``(n_tapers, n)``, tapers ordered
        most concentrated first and normalised to unit energy. Each taper's
        first entry is made positive, which fixes the arbitrary eigenvector
        sign; a spectrum is built from ``|X|^2`` so the sign never matters,
        but a reproducible one is easier to eyeball.
    """
    if n < 2:
        raise DataValidationError(_("DPSS needs n >= 2; got {0}").format(n))
    k = int(n_tapers)
    if k < 1 or k > n:
        raise DataValidationError(_("n_tapers must lie in [1, n]; got {0} for n = {1}").format(k, n))
    if nw <= 0:
        raise DataValidationError(_("nw must be > 0; got {0}").format(nw))
    w = float(nw) / n
    t = np.arange(n, dtype=float)
    diag = ((n - 1.0 - 2.0 * t) / 2.0) ** 2 * np.cos(2.0 * np.pi * w)
    off = t[1:] * (n - t[1:]) / 2.0
    # The eigenvalues of B are NOT the concentration eigenvalues, so the
    # first return value is deliberately discarded here.
    vecs = eigh_tridiagonal(diag, off, select="i", select_range=(n - k, n - 1))[1]
    tapers = np.ascontiguousarray(vecs[:, ::-1].T)
    signs = np.where(tapers[:, 0] < 0.0, -1.0, 1.0)
    tapers = tapers * signs[:, None]
    energy = np.sqrt(np.sum(tapers**2, axis=1))
    tapers = tapers / energy[:, None]

    # PW93 p. 390: concentration eigenvalues from the taper's own
    # autocorrelation and the concentration operator's autocorrelation.
    use_n = int(2 * n - 1)
    spec = np.fft.rfft(tapers, use_n, axis=1)
    rxx = np.fft.irfft(spec * spec.conj(), n=use_n)[:, :n]
    m = np.arange(n, dtype=float)
    kernel = 4.0 * w * np.sinc(2.0 * w * m)
    kernel = kernel.copy()
    kernel[0] = 2.0 * w
    return np.asarray(rxx @ kernel, dtype=float), tapers


def multitaper_spectrum(
    values: npt.NDArray,
    NW: float = 3.0,
    n_periods: int | None = None,
    fap_level: float = 0.05,
) -> MultitaperResult:
    """Thomson (1982) multitaper spectral estimate with a significance test.

    Multiplying a record by ``K`` orthogonal prolate spheroidal sequences and
    averaging the ``K`` single-taper spectra trades a small amount of
    variance for a much narrower effective spectral window than a plain
    periodogram, which is why it is the estimator of choice for picking
    periodic lines out of a red-noise background.

    The spectrum is the Thomson-weighted average

        S(f) = sum_k lambda_k S_k(f) / sum_k lambda_k

    with ``lambda_k`` the concentration eigenvalues. It is normalised so
    white noise has unit mean, the same scale as
    :func:`spectral_analysis.ar1_theoretical_spectrum`, so the two can be
    compared directly. The significance test is Thomson's chi-square form:
    with ``nu`` equivalent degrees of freedom, ``nu * S(f)`` is
    ``chi2(nu)`` under the white-noise null.

    Parameters
    ----------
    values:
        The record, in sample order. Ages must be evenly spaced for a
        frequency in "cycles per sample" to mean anything; use
        :class:`stratigraphy.timeaxis.TimeAxis` to establish the step.
    NW:
        Time-bandwidth product. The number of tapers is
        ``floor(2 * NW - 1)`` (Thomson 1982), the choice that keeps every
        taper's eigenvalue above about 0.5.
    n_periods:
        Accepted for signature symmetry with the other spectral routines;
        the taper-product grid is fixed at ``n // 2``, so this is recorded
        rather than used.
    fap_level:
        Level for the significance mask.

    Returns
    -------
    MultitaperResult
    """
    x = _clean_series(values, "multitaper input", min_samples=16)
    if not 0.5 < NW < 50.0:
        raise DataValidationError(_("NW must lie in (0.5, 50); got {0}").format(NW))
    n = x.size
    k = int(np.floor(2.0 * float(NW) - 1.0))
    k = max(1, min(k, n // 2))
    eigenvalues, tapers = dpss_tapers(n, NW, k)

    v = x - float(np.mean(x))
    n = v.size
    freqs = np.fft.rfftfreq(n)  # cycles per sample
    products = np.empty((k, freqs.size - 1), dtype=float)
    freqs_out = freqs[1:]
    weight_sum = float(np.sum(eigenvalues))
    # A unit-energy taper gives E|X_k|^2 = sigma^2 for white noise, so
    # dividing by the record variance puts every taper product on the same
    # E[S] = 1 scale as the periodogram and the AR(1) curve.
    sigma2 = float(np.sum(v**2) / n)
    for i in range(k):
        spec = np.abs(np.fft.rfft(tapers[i] * v)) ** 2
        products[i] = spec[1:] / sigma2
    power = (eigenvalues[:, None] * products).sum(axis=0) / weight_sum

    sum_sq = float(np.sum(eigenvalues**2))
    bandwidth = (k + 1) / (2.0 * n) * sum_sq / (weight_sum**2)
    dof = 2.0 * weight_sum**2 / sum_sq
    fap = sp_stats.chi2.sf(dof * power, dof)
    logger.info(
        "multitaper: n=%d, K=%d, NW=%.2f, ENBW=%.5f cyc/sample, dof=%.2f",
        n,
        k,
        NW,
        bandwidth,
        dof,
    )
    return MultitaperResult(
        frequencies=freqs_out,
        periods=1.0 / freqs_out,
        power=power,
        tapers=tapers,
        eigenvalues=eigenvalues,
        bandwidth=float(bandwidth),
        n_tapers=k,
        equivalent_n_dof=float(dof),
        fap=fap,
        significant=fap <= fap_level,
        fap_level=float(fap_level),
        n=n,
    )


# =============================================================================
# 5. Cross-correlation
# =============================================================================


@dataclass
class CrossCorrResult:
    """Normalised cross-correlation function with a significance band.

    Attributes:
        lags: Signed lag in samples. Positive means ``y`` leads ``x``.
        correlation: Pearson correlation of the two overlapping segments at
            each lag, in [-1, 1].
        n_pairs: Number of overlapping pairs behind each lag.
        confidence: The band half-width per lag, from the Fisher z with the
            Pyper & Peterman (2004) effective sample size.
        band_lower / band_upper: The band, expanded for plotting.
        best_lag: Lag of the largest ``|correlation|``.
        best_correlation: Correlation at ``best_lag``.
    """

    lags: npt.NDArray
    correlation: npt.NDArray
    n_pairs: npt.NDArray
    confidence: npt.NDArray
    band_lower: npt.NDArray
    band_upper: npt.NDArray
    best_lag: int
    best_correlation: float
    confidence_level: float = 0.95

    def summary(self) -> str:
        """Generate summary text."""
        return (
            f"{_('Cross-correlation')}\n"
            f"{'=' * 40}\n"
            f"{_('Lags: {0}').format(self.lags.size)}\n"
            f"{_('Best lag: {0} samples').format(self.best_lag)}\n"
            f"{_('Correlation at best lag: {0}').format(f'{self.best_correlation:.4f}')}\n"
            f"{_('Band half-width (lag 1): {0}').format(f'{self.confidence[0]:.4f}')}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "lags": [int(k) for k in self.lags],
            "correlation": [float(v) for v in self.correlation],
            "best_lag": int(self.best_lag),
            "best_correlation": float(self.best_correlation),
            "confidence_level": float(self.confidence_level),
        }


def cross_correlation(
    x: npt.NDArray,
    y: npt.NDArray,
    max_lag: int | None = None,
    confidence: float = 0.95,
) -> CrossCorrResult:
    """Cross-correlation of two series, with an autocorrelation-corrected band.

    At each signed lag the two overlapping segments are correlated
    (Pearson), so the coefficient is bounded in [-1, 1] and needs no
    separate normalisation. The band is **not** the naive
    ``1.96 / sqrt(n_pairs)``: two autocorrelated series have fewer
    independent pairs than they have pairs, and the naive band is
    anti-conservative for exactly the paleoclimate records this is used on.
    Following Pyper & Peterman (2004) the effective count is

        n_eff = n_pairs * (1 - r1x r1y) / (1 + r1x r1y)

    and the comparison is made on the Fisher z, ``z = arctanh(r) *
    sqrt(n_eff - 3)``, which is the transform under which the sampling
    distribution is close to normal.

    **Sign convention:** a positive lag means ``y`` leads ``x``, i.e. the
    pattern in ``x`` appears later in ``y``. With both series plotted
    against the same time axis that is the direction a lead has.

    Parameters
    ----------
    x, y:
        The two series, in the same sample order and on the same grid.
    max_lag:
        Largest |lag| to evaluate. Defaults to ``min(nx, ny) // 2``.
    confidence:
        Two-sided level for the band.

    Returns
    -------
    CrossCorrResult
    """
    a = _clean_series(x, "cross-correlation x", min_samples=MIN_SAMPLES)
    b = _clean_series(y, "cross-correlation y", min_samples=MIN_SAMPLES)
    n = int(min(a.size, b.size))
    a, b = a[:n], b[:n]
    lag_max = int(max_lag) if max_lag is not None else n // 2
    if lag_max < 1 or lag_max >= n:
        raise DataValidationError(_("max_lag must lie in [1, n-1]; got {0} for n = {1}").format(lag_max, n))

    r1x = float(autocorrelation(a, max_lag=1).acf[1])
    r1y = float(autocorrelation(b, max_lag=1).acf[1])
    shrink = (1.0 - r1x * r1y) / (1.0 + r1x * r1y)

    lags = np.arange(-lag_max, lag_max + 1)
    corr = np.empty(lags.size, dtype=float)
    pairs = np.empty(lags.size, dtype=int)
    for i, k in enumerate(lags):
        if k >= 0:
            u, v = a[: n - k], b[k:]
        else:
            u, v = a[-k:], b[: n + k]
        pairs[i] = u.size
        if u.size < 3 or np.ptp(u) == 0.0 or np.ptp(v) == 0.0:
            corr[i] = np.nan
            continue
        corr[i] = float(np.corrcoef(u, v)[0, 1])

    n_eff = np.maximum(pairs * shrink, 4.0)
    zcrit = float(sp_stats.norm.ppf(0.5 + 0.5 * confidence))
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.tanh(zcrit / np.sqrt(np.maximum(n_eff - 3.0, 1.0)))
    conf = np.where(np.isfinite(corr), conf, 0.0)

    finite = np.isfinite(corr)
    best = int(lags[finite][int(np.argmax(np.abs(corr[finite])))]) if np.any(finite) else 0
    return CrossCorrResult(
        lags=lags,
        correlation=corr,
        n_pairs=pairs,
        confidence=conf,
        band_lower=-conf,
        band_upper=conf,
        best_lag=best,
        best_correlation=float(corr[lags == best][0]) if np.any(finite) else float("nan"),
        confidence_level=float(confidence),
    )


# =============================================================================
# 6. Runs test
# =============================================================================


@dataclass
class RunsResult:
    """Outcome of a runs test on a binarised series.

    Attributes:
        runs: Observed number of runs.
        n_above / n_below: Counts of the two symbols.
        expected_runs: Wald & Wolfowitz (1943) expectation ``1 + 2 n0 n1 / n``.
        variance: Wald & Wolfowitz variance
            ``2 n0 n1 (2 n0 n1 - n) / (n^2 (n - 1))``.
        z: Normal-approximation statistic, continuity-corrected.
        p_value: Two-sided p from the normal approximation.
        p_value_exact: Two-sided p from the exact conditional distribution.
        p_value_exact_available: False when the exact count was not attempted
            because the sample is large (see :func:`runs_test`).
        threshold: The value the series was binarised about.
        exact_used: Whether the exact form is the one to quote.
    """

    runs: int
    n_above: int
    n_below: int
    expected_runs: float
    variance: float
    z: float
    p_value: float
    p_value_exact: float
    p_value_exact_available: bool
    threshold: float
    exact_used: bool

    @property
    def significant(self) -> bool:
        """Whether the run count rejects at the 0.05 level."""
        p = self.p_value_exact if self.exact_used else self.p_value
        return bool(p < 0.05)

    def summary(self) -> str:
        """Generate summary text."""
        observed = _("Runs observed / expected: {0} / {1}").format(self.runs, f"{self.expected_runs:.3f}")
        counts = _("Above / below threshold: {0} / {1}").format(self.n_above, self.n_below)
        return (
            f"{_('Runs Test')}\n"
            f"{'=' * 40}\n"
            f"{observed}\n"
            f"{counts}\n"
            f"{_('Threshold: {0}').format(f'{self.threshold:.6g}')}\n"
            f"{_('z: {0}').format(f'{self.z:.4f}')}\n"
            f"{_('p (normal): {0}').format(f'{self.p_value:.4f}')}\n"
            f"{_('p (exact): {0}').format(f'{self.p_value_exact:.4f}')}\n"
            f"{_('Quote: {0}').format(_('exact') if self.exact_used else _('normal'))}"
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "runs": int(self.runs),
            "n_above": int(self.n_above),
            "n_below": int(self.n_below),
            "expected_runs": float(self.expected_runs),
            "variance": float(self.variance),
            "z": float(self.z),
            "p_value": float(self.p_value),
            "p_value_exact": float(self.p_value_exact),
            "p_value_exact_available": bool(self.p_value_exact_available),
            "threshold": float(self.threshold),
            "significant": self.significant,
        }


def _runs_count_exact(r: int, n0: int, n1: int) -> float:
    """Number of arrangements of ``n1`` ones and ``n0`` zeros with ``r`` runs.

    Cramer (1926), as tabulated in Wald & Wolfowitz (1943): for even ``r``
    the two symbols can start either symbol, for odd ``r`` only one can.
    """
    if r < 2 or r > n0 + n1:
        return 0.0
    if r % 2 == 0:
        a, b = r // 2 - 1, r // 2 - 1
    else:
        a, b = (r + 1) // 2 - 1, (r - 1) // 2 - 1
    total = comb(n0 - 1, a) * comb(n1 - 1, b)
    if r % 2 == 1:
        total += comb(n0 - 1, b) * comb(n1 - 1, a)
    else:
        total *= 2
    return float(total)


def runs_test(values: npt.NDArray, expected_sign: float = 0.0, max_exact_n: int = 60) -> RunsResult:
    """Runs test for randomness in the ordering of a binarised series.

    The series is binarised and the number of maximal runs of like symbols
    is counted; too few runs means the series is ordered (a trend or a
    cycle), too many means it alternates more than chance allows.

    **Which null this uses, and why.** The null is the **exact conditional
    distribution of the run count given ``n_above`` and ``n_below``** (Cramer
    1926; Wald & Wolfowitz 1943), evaluated in closed form:

        P(R = r) = N(r; n0, n1) / C(n, n1),   n = n0 + n1

    It is *not* a permutation of the values, and *not* a permutation of a
    binary series. A binary permutation is the same null distribution by
    construction, but reaching it costs thousands of shuffles and returns a
    different answer on every run; the closed form costs microseconds and is
    exact. The normal approximation is reported alongside it because it is
    the classical statistic, but Wald & Wolfowitz note it degrades for small
    ``n``, so ``exact_used`` is True whenever the exact form was computed
    and the summary quotes that one.

    Parameters
    ----------
    values:
        The series, in sample order.
    expected_sign:
        ``0.0`` (default) binarises at the **sample median**, which is the
        classical balanced runs test. Any other value binarises at that
        numeric level, with values *strictly greater* counting as "above".
    max_exact_n:
        Skip the exact form above this total count. For a few hundred
        observations the closed form is still fast but the two-sided tail
        sum over all attainable ``r`` is not, and the normal approximation
        is accurate there anyway.

    Returns
    -------
    RunsResult
    """
    x = _clean_series(values, "runs test input", min_samples=4)
    threshold = float(np.median(x)) if expected_sign == 0.0 else float(expected_sign)
    above = x > threshold
    n1 = int(np.sum(above))
    n0 = int(x.size - n1)
    if n0 == 0 or n1 == 0:
        raise DataValidationError(
            _("Runs test needs values on both sides of the threshold; got {0} above and {1} below").format(n1, n0),
            details={"threshold": threshold},
        )

    runs = 1 + int(np.sum(above[1:] != above[:-1]))
    n = n0 + n1
    expected = 1.0 + 2.0 * n0 * n1 / n
    var_r = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n) / (n**2 * (n - 1.0))
    if var_r <= 0:
        raise ComputationError(
            _("Runs test variance is not positive; check the binarisation"),
            details={"n_above": n1, "n_below": n0},
        )
    # The classical Wald & Wolfowitz statistic, with NO continuity
    # correction: the corrected version is defensible but is not the
    # published statistic, and the exact form below is what a small sample
    # should be judged on anyway.
    z = (runs - expected) / np.sqrt(var_r)
    p_norm = float(2.0 * sp_stats.norm.sf(abs(z)))

    p_exact = float("nan")
    available = bool(n <= max_exact_n)
    if available:
        counts = np.array([_runs_count_exact(r, n0, n1) for r in range(2, n + 1)])
        r_vals = np.arange(2, n + 1)
        probs = counts / comb(n, n1)
        keep = (probs <= probs[r_vals == runs][0] + 1e-12) & (counts > 0)
        p_exact = float(probs[keep].sum())
        p_exact = min(1.0, p_exact)

    return RunsResult(
        runs=runs,
        n_above=n1,
        n_below=n0,
        expected_runs=float(expected),
        variance=float(var_r),
        z=float(z),
        p_value=p_norm,
        p_value_exact=p_exact,
        p_value_exact_available=available,
        threshold=threshold,
        exact_used=available,
    )


# =============================================================================
# 7. Mann-Kendall trend test
# =============================================================================


@dataclass
class MannKendallResult:
    """Outcome of a Mann-Kendall trend test, with Sen's slope.

    Attributes:
        s: The S statistic, the sum of the signs of all pairwise
            differences.
        tau: Kendall's tau-b, ``s / (n0 * n1)``.
        z: Normal-approximation statistic with the tie and
            autocorrelation corrections applied.
        p_value: Two-sided p.
        slope: Sen's slope, the median of all pairwise slopes. Units are
            value units per index step, or per time unit when ``times`` was
            given.
        slope_ci: Two-sided confidence interval for the slope (normal
            approximation to the Kendall variance).
        variance_s: The variance actually used for ``z``.
        n: Number of observations.
        n_tied_groups: Number of tie groups; 0 means no ties.
        pre_whitened: Whether the Hamed & Rao (1998) correction was applied.
        variance_factor: The Hamed & Rao inflation/deflation factor applied
            to the variance, 1.0 when not used.
        times: Whether a time base was supplied, which changes the units of
            ``slope``.
    """

    s: int
    tau: float
    z: float
    p_value: float
    slope: float
    slope_ci: tuple[float, float]
    variance_s: float
    n: int
    n_tied_groups: int
    pre_whitened: bool
    variance_factor: float
    times: bool

    @property
    def significant(self) -> bool:
        """Whether the trend rejects at the 0.05 level."""
        return bool(self.p_value < 0.05)

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Mann-Kendall Trend Test"),
            "=" * 50,
            _("n: {0}").format(self.n),
            _("S: {0}").format(self.s),
            _("tau: {0}").format(f"{self.tau:.4f}"),
            _("z: {0}").format(f"{self.z:.4f}"),
            _("p-value: {0}").format(f"{self.p_value:.4e}"),
            _("Sen's slope: {0:.6g}").format(self.slope),
            _("Sen's slope 95% CI: [{0:.6g}, {1:.6g}]").format(*self.slope_ci),
            _("Tie groups: {0}").format(self.n_tied_groups),
            _("Hamed-Rao correction: {0}").format(
                _("on (factor {0:.4f})").format(self.variance_factor) if self.pre_whitened else _("off")
            ),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "s": int(self.s),
            "tau": float(self.tau),
            "z": float(self.z),
            "p_value": float(self.p_value),
            "slope": float(self.slope),
            "slope_ci": [float(self.slope_ci[0]), float(self.slope_ci[1])],
            "n": int(self.n),
            "n_tied_groups": int(self.n_tied_groups),
            "pre_whitened": bool(self.pre_whitened),
            "variance_factor": float(self.variance_factor),
        }


def _sen_slope(values: npt.NDArray, times: npt.NDArray) -> tuple[float, tuple[float, float]]:
    """Sen's slope and its normal-approximation confidence interval."""
    n = values.size
    i, j = np.triu_indices(n, k=1)
    dt = times[j] - times[i]
    ok = dt != 0
    slopes = (values[j][ok] - values[i][ok]) / dt[ok]
    if slopes.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    slope = float(np.median(slopes))
    # Kendall's tau variance is the variance of S; the slope's variance is
    # Var(S) / (n0 n1)^2 * the variance of the pairwise slopes, which for a
    # regular time base is sigma^2 * 2 / (n(n-1)(2n-5) (t_j - t_i)^2). The
    # standard Sen (1968) normal interval uses the median-absolute-
    # deviation form below.
    var_s = _mk_variance(values)
    n0 = n * (n - 1) / 2.0
    spread = float(np.var(slopes, ddof=1)) if slopes.size > 1 else 0.0
    if not np.isfinite(spread) or var_s <= 0:
        return slope, (float("nan"), float("nan"))
    if spread <= 0:
        # Every pairwise slope is identical: the series is exactly affine in
        # the time base and the interval is a point, not an infinity.
        return slope, (slope, slope)
    se = np.sqrt(spread * (n - 2) / (n0**2))
    z = float(sp_stats.norm.ppf(0.975))
    return slope, (float(slope - z * se), float(slope + z * se))


def _mk_variance(values: npt.NDArray) -> float:
    """Variance of the Mann-Kendall S statistic, with the tie correction.

    Var(S) = [n(n-1)(2n+5) - sum_t m_t (m_t - 1)(2 m_t + 5)] / 18
    """
    n = values.size
    total = n * (n - 1.0) * (2.0 * n + 5.0)
    counts = np.unique(values, return_counts=True)[1]
    counts = counts[counts > 1]
    if counts.size:
        total -= float(np.sum(counts * (counts - 1.0) * (2.0 * counts + 5.0)))
    return max(total / 18.0, 0.0)


def _hamed_rao_factor(values: npt.NDArray) -> tuple[float, int]:
    """Hamed & Rao (1998) variance inflation factor and the lag it used.

    The autocorrelated series this module is for break the Mann-Kendall
    independence assumption badly, and the failure is always in the
    anti-conservative direction: an unadjusted test over a series that only
    wanders reports a significant trend at a rate far above nominal. The
    fix is to inflate the variance of ``S``, because a positively
    autocorrelated series makes concordant pairs cluster and so genuinely
    raises the spread of ``S`` above its nominal value. Hamed & Rao
    rescale by the ratio of the nominal to the effective sample size,

        n* = n / (1 + 2 sum_{k=1..K} (1 - k/(K+1)) rho_k / (1 - rho_k^2))
        Var*(S) = (n / n*) * Var(S)

    where ``K`` is the first lag at which the ACF turns negative.

    **The direction matters and is easy to get backwards.** Multiplying by
    ``n*/n`` (< 1 for positive autocorrelation) would SHRINK the variance,
    enlarge ``|z|``, and make the test strictly more anti-conservative --
    i.e. it would "correct" the problem by amplifying it. On a unit-root
    series the inflation has to reach roughly ``n`` for the test to stop
    calling a pure random walk a trend, which is the sanity check that
    fixes the sign. The factor is clipped to ``[1, n]``: below 1 a
    negatively autocorrelated series is left alone, and above ``n`` the test
    would be powerless and would be better reported as such.
    """
    n = values.size
    lag_max = int(min(n - 2, max(2, n // 4)))
    acf = autocorrelation(values, max_lag=lag_max).acf
    k_used = lag_max
    for k in range(1, lag_max + 1):
        if acf[k] <= 0.0:
            k_used = max(k - 1, 1)
            break
    total = 0.0
    for k in range(1, k_used + 1):
        rho = float(acf[k])
        if abs(rho) >= 0.99:
            continue
        total += 2.0 * (1.0 - k / (k_used + 1.0)) * rho / (1.0 - rho**2)
    inflation = 1.0 + total
    return float(min(max(inflation, 1.0), float(n))), k_used


def mann_kendall(
    values: npt.NDArray,
    times: npt.NDArray | None = None,
    pre_whitened: bool = False,
) -> MannKendallResult:
    """Mann-Kendall trend test with Sen's slope and optional prewhitening.

    Mann (1945) / Kendall (1975). The test asks whether the values are
    *monotonically* related to their position, without assuming a functional
    form -- which is why it is the right trend test for a noisy isotope
    series, where a least-squares slope is dragged around by outliers.

    **Sen's slope is the point of using this at all.** It is the median of
    the ``n(n-1)/2`` pairwise slopes, so a few extreme excursions move it
    far less than they move an OLS slope, and on a series where the trend is
    real but the noise is not normal it is the quantity a reader actually
    wants quoted. It is exact here: a series with every pairwise slope
    equal to ``b`` returns exactly ``b``.

    Parameters
    ----------
    values:
        The series, in time order.
    times:
        Optional time base. Without it the index is the time base, and
        ``slope`` is per sample.
    pre_whitened:
        Apply the Hamed & Rao (1998) correction, which inflates the
        variance of ``S`` according to the series' own autocorrelation.
        **The uncorrected test is anti-conservative on autocorrelated
        series**, so this should be True for the stratigraphic records this
        module is for. It is False by default only to keep the classical
        textbook result reachable.

    Returns
    -------
    MannKendallResult
    """
    x = _clean_series(values, "mann-kendall input", min_samples=4)
    n = x.size
    if times is None:
        t = np.arange(n, dtype=float)
        has_times = False
    else:
        t = np.asarray(times, dtype=float).flatten()
        if t.size != n:
            raise DataValidationError(_("times must have the same length as values; got {0} and {1}").format(t.size, n))
        ok = np.isfinite(t)
        x, t = x[ok], t[ok]
        has_times = True
        if x.size < 4 or np.any(np.diff(t) <= 0):
            raise DataValidationError(
                _("times must be strictly increasing and leave at least 4 points"),
                details={"n": int(x.size)},
            )

    i, j = np.triu_indices(x.size, k=1)
    s = int(np.sum(np.sign(x[j] - x[i])))
    n_pairs = x.size * (x.size - 1) / 2.0
    tau = s / n_pairs

    var_s = _mk_variance(x)
    factor = 1.0
    if pre_whitened:
        factor, _k = _hamed_rao_factor(x)
        var_s *= factor
    if var_s <= 0:
        raise ComputationError(
            _("Mann-Kendall variance is not positive; the series may be all ties"),
            details={"n": int(x.size)},
        )
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = float(2.0 * sp_stats.norm.sf(abs(z)))

    slope, ci = _sen_slope(x, t)
    counts = np.unique(x, return_counts=True)[1]
    return MannKendallResult(
        s=s,
        tau=float(tau),
        z=float(z),
        p_value=p,
        slope=slope,
        slope_ci=ci,
        variance_s=float(var_s),
        n=int(x.size),
        n_tied_groups=int(np.sum(counts > 1)),
        pre_whitened=bool(pre_whitened),
        variance_factor=float(factor),
        times=has_times,
    )


# =============================================================================
# 8. Autoassociation function
# =============================================================================


@dataclass
class AutoassociationResult:
    """Binned pairwise-similarity curve (Shirer 2008).

    Attributes:
        lag: Reported lag of each bin, in the units of ``time``.
        similarity: Mean product of standardised values over the pairs in
            the bin.
        correlation: Pearson correlation of the two values over the pairs in
            the bin, which is the alternative reading of the same bin.
        counts: Number of pairs in each bin.
        bin_width: Lag width of a bin.
        total_pairs: Number of distinct pairs used.
        peak_lags: Lags of the local maxima, strongest first.
        max_pairs: Guard on the pair count actually enforced.
    """

    lag: npt.NDArray
    similarity: npt.NDArray
    correlation: npt.NDArray
    counts: npt.NDArray
    bin_width: float
    total_pairs: int
    peak_lags: list[float] = field(default_factory=list)
    max_pairs: int = 2_000_000

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            _("Autoassociation Function"),
            "=" * 50,
            _("Bins: {0}").format(self.lag.size),
            _("Bin width: {0:.6g}").format(self.bin_width),
            _("Pairs: {0}").format(self.total_pairs),
            _("Peaks (strongest first): {0}").format(
                ", ".join(f"{p:.6g}" for p in self.peak_lags[:6]) if self.peak_lags else _("none")
            ),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""
        return {
            "lag": [float(v) for v in self.lag],
            "similarity": [float(v) for v in self.similarity],
            "counts": [int(v) for v in self.counts],
            "bin_width": float(self.bin_width),
            "total_pairs": int(self.total_pairs),
            "peak_lags": list(self.peak_lags),
        }


def autoassociation(
    time: npt.NDArray,
    values: npt.NDArray,
    n_bins: int | None = None,
    max_pairs: int = 2_000_000,
) -> AutoassociationResult:
    """Autoassociation function: binned pairwise similarity against |dt|.

    **This is not the ACF.** The ACF (:func:`autocorrelation`) is a
    product averaged over a *single* lag, and it assumes equal spacing. The
    autoassociation function of Shirer (2008) bins the absolute time
    separation ``|t_i - t_j|``, and averages how similar the two values are
    in every bin at once, so an irregularly sampled record can be read on
    the same footing as a regular one and a weak short-period line is not
    diluted by a handful of badly placed samples. Its peak structure --
    the first peak and the spacing of the repeats -- is the classical way a
    cycle period is read off a stratigraphic record.

    The similarity used here is the mean product of standardised values
    over the pairs in the bin, which makes the curve a bin-averaged version
    of the ACF. In the special case of uniform sampling with
    ``n_bins = n - 1``, bin ``b`` contains exactly the lag-``b`` pairs, so
    the curve is the **unbiased** sample ACF, ``n / (n - b)`` times the
    biased estimator :func:`autocorrelation` returns; the test suite pins
    that exact relationship down, which is a stronger check than any single
    peak height.

    Parameters
    ----------
    time, values:
        The record. ``time`` must be strictly increasing.
    n_bins:
        Number of lag bins spanning the full record. Defaults to about
        ``sqrt(n (n-1) / 2)`` bins, which keeps the average occupancy near
        one pair.
    max_pairs:
        Refuse more than this many distinct pairs. The curve is quadratic in
        ``n`` by nature, and a million-sample record should fail loudly
        rather than appear to hang.

    Returns
    -------
    AutoassociationResult
    """
    t, v = _clean_pair(time, values, "autoassociation input")
    _check_increasing(t, "autoassociation time")
    n = t.size
    total_pairs = n * (n - 1) // 2
    if total_pairs > max_pairs:
        raise DataValidationError(
            _(
                "autoassociation would need {0} pairs, above the limit of {1}; downsample the record or bin coarser"
            ).format(total_pairs, max_pairs),
            details={"pairs": int(total_pairs), "max_pairs": int(max_pairs)},
        )
    if n_bins is None:
        n_bins = int(max(8, min(n - 1, round(np.sqrt(total_pairs)))))
    n_bins = int(n_bins)
    if n_bins < 2:
        raise DataValidationError(_("n_bins must be >= 2; got {0}").format(n_bins), details={"n_bins": n_bins})

    i, j = np.triu_indices(n, k=1)
    lag = np.abs(t[j] - t[i])
    lag_max = float(lag.max())
    if lag_max <= 0:
        raise DataValidationError(_("autoassociation has zero time span"))
    width = lag_max / n_bins
    # The +eps guards the exactly-uniform case: 0.03 / 0.01 evaluates to
    # 2.9999999999999996, and without it a lag-k pair lands in bin k-1 and
    # the curve stops coinciding with the ACF on the very input it is
    # supposed to reproduce.
    idx = np.minimum(np.floor(lag / width + 1.0e-6).astype(int), n_bins - 1)

    z = v - float(np.mean(v))
    sigma = float(np.std(z))
    if sigma <= 0:
        raise ComputationError("autoassociation is undefined for a constant series")
    z = z / sigma
    products = z[i] * z[j]

    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=products, minlength=n_bins)

    # A MEAN over the bin, not a sum: a sum normalised by the total sum of
    # squares runs above 1 as soon as a bin holds more than one lag, and a
    # "similarity" that exceeds 1 is not a similarity. The consequence is
    # that on a uniform grid with one lag per bin the curve is the
    # UNBIASED sample ACF, i.e. n / (n - k) times the biased estimator
    # :func:`autocorrelation` returns; the test suite pins that exact
    # relationship down.
    similarity = np.divide(sums, np.maximum(counts, 1), out=np.zeros(n_bins), where=counts > 0)

    # Per-bin Pearson correlation of the two value vectors, as an
    # alternative reading that is not tied to the mean value.
    correlation = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        sel = idx == b
        if int(np.sum(sel)) < 3:
            continue
        u, w = v[i[sel]], v[j[sel]]
        if np.ptp(u) == 0.0 or np.ptp(w) == 0.0:
            continue
        correlation[b] = float(np.corrcoef(u, w)[0, 1])

    centres = width * np.arange(n_bins)
    peaks: list[float] = []
    for b in range(1, n_bins - 1):
        if counts[b] == 0 or counts[b - 1] == 0 or counts[b + 1] == 0:
            continue
        if similarity[b] > similarity[b - 1] and similarity[b] >= similarity[b + 1]:
            peaks.append(float(centres[b]))
    peaks.sort(key=lambda c: -similarity[round(c / width)])

    return AutoassociationResult(
        lag=centres,
        similarity=similarity,
        correlation=correlation,
        counts=counts,
        bin_width=width,
        total_pairs=int(total_pairs),
        peak_lags=peaks[:50],
        max_pairs=int(max_pairs),
    )


# =============================================================================
# 9. Orbital forcing
# =============================================================================

#: (name, period_kyr, relative_amplitude, band, role)
#:
#: SEE THE ``orbital_forcing`` DOCSTRING BEFORE USING THE AMPLITUDES. These
#: are rounded Laskar-style approximations, not the Laskar et al. (2004)
#: solution.
_ORBITAL_ROWS: list[tuple[str, float, float, str, str]] = [
    ("long_period_precession", 400.0, 0.35, "precession", "modulation"),
    ("precession_100", 100.0, 0.30, "precession", "forcing"),
    ("precession_70", 70.0, 0.18, "precession", "forcing"),
    ("obliquity_41", 41.0, 0.35, "obliquity", "forcing"),
    ("precession_23", 23.0, 0.30, "precession", "forcing"),
    ("precession_19", 19.0, 0.18, "precession", "forcing"),
]


def orbital_forcing(
    period_kyr: float | None = None,
    n_periods: int = 5,
    n_samples: int = 512,
) -> list[dict[str, float | str]]:
    """Milankovitch period table with relative amplitudes, or one band.

    READ THIS BEFORE USING THE AMPLITUDES. What is and is not claimed:

    * **Periods** are the canonical mean values in kyr: the two short
      precession terms near 19 and 23 kyr, the precession beat near 100 kyr,
      the second eccentricity band near 70 kyr, obliquity near 41 kyr, and
      the long-period precession modulation near 400 kyr. The 19.1 / 23.4
      kyr pair are the present-day precession periods of Berger (1978),
      rounded to 19 and 23 here.
    * **Relative amplitudes** are ROUNDED ORDER-OF-MAGNITUDE WEIGHTS for
      building a synthetic target curve, in the spirit of the analytic
      targets used by Cramer (2012, ``cyclostrat``) and the standard
      insolation forcings of Berger (1978). **They are not read off the
      Laskar et al. (2004) orbital solution, which has time-dependent
      periods and amplitudes and which cannot be tabulated as five
      constants, and they must not be used for a quantitative insolation
      reconstruction.** They encode only the orderings that any defensible
      table must satisfy: obliquity and short precession are the same order
      of magnitude, the ~100 kyr band dominates the ~70 kyr sideband, and
      the ~400 kyr term is an amplitude *envelope* rather than a forcing.
    * ``role`` records which of those a row is: ``"modulation"`` for the
      400 kyr eccentricity cycle that modulates the precession bands, and
      ``"forcing"`` for the lines an insolation or isotope series should
      show. A synthesis that adds the 400 kyr term as a plain sinusoid is
      modelling something that is not there.

    Parameters
    ----------
    period_kyr:
        Return only the row(s) whose period is within 1 % of this value.
    n_periods:
        Number of cycles to build when ``times_ma`` is used downstream via
        :func:`insolation_series`. Recorded only when a single row is
        returned, for symmetry with the other spectral routines.
    n_samples:
        Recorded for the same reason; the table itself has no time axis.

    Returns
    -------
    list of dict:
        Rows with ``name``, ``period_kyr``, ``relative_amplitude``, ``band``,
        ``role``, ``frequency_per_ma`` and ``n_periods`` / ``n_samples``.
    """
    if n_periods < 1:
        raise DataValidationError(_("n_periods must be >= 1"), details={"n_periods": int(n_periods)})
    if n_samples < 2:
        raise DataValidationError(_("n_samples must be >= 2"), details={"n_samples": int(n_samples)})
    rows: list[dict[str, float | str]] = []
    for name, period, amp, band, role in _ORBITAL_ROWS:
        if period_kyr is not None and abs(period - float(period_kyr)) / period > 0.01:
            continue
        rows.append(
            {
                "name": name,
                "period_kyr": float(period),
                "relative_amplitude": float(amp),
                "band": band,
                "role": role,
                "frequency_per_ma": float(1000.0 / period),
                "n_periods": int(n_periods),
                "n_samples": int(n_samples),
            }
        )
    if period_kyr is not None and not rows:
        raise DataValidationError(
            _(
                "No Milankovitch band within 1 % of {0} kyr; the table holds 400, "
                "100, 70, 41, 23 and 19 kyr (pass 400 for the modern 405 kyr "
                "long-period cycle)"
            ).format(period_kyr)
        )
    return rows


def insolation_series(times_ma: npt.NDArray, period_kyr: float) -> npt.NDArray:
    """Unit-amplitude cosine at one Milankovitch period, for a target curve.

    This is the *synthetic target*, not an insolation solution: a cosine of
    the requested period on the requested ages. It exists so a peak-picking
    routine can be tested against something with a known answer, and so a
    user can build the composite curve in :func:`orbital_forcing` by hand
    with amplitudes they trust.

    ``times_ma`` is in Ma and ``period_kyr`` in kyr; the conversion is
    ``2 pi t_ma * 1000 / period_kyr``.
    """
    t = np.asarray(times_ma, dtype=float).flatten()
    if t.size == 0:
        raise DataValidationError(_("times_ma must contain at least one age"))
    if not np.all(np.isfinite(t)):
        raise DataValidationError(_("times_ma contains non-finite ages"))
    p = float(period_kyr)
    if p <= 0:
        raise DataValidationError(_("period_kyr must be > 0; got {0}").format(p))
    return np.cos(2.0 * np.pi * t * 1000.0 / p)


# =============================================================================
# Analyzer facade
# =============================================================================


class CycleAnalyzer:
    """Stateful entry point for the cycle-detection routines.

    Thin by design: each method delegates to the module-level function of
    the same name and remembers the result, so a UI can keep one analyzer
    per session without the statistics depending on it.
    """

    def __init__(self) -> None:
        """Initialise the analyzer."""
        self._logger = logging.getLogger(f"{__name__}.CycleAnalyzer")
        self._last_result: object | None = None

    def _record(self, result: object) -> object:
        self._last_result = result
        return result

    def autocorrelation(self, values: npt.NDArray, **kwargs) -> ACFResult:
        """See :func:`autocorrelation`."""
        return self._record(autocorrelation(values, **kwargs))  # type: ignore[no-any-return]

    def prewhiten(self, values: npt.NDArray, d: int = 1) -> PrewhitenResult:
        """See :func:`ar1_prewhiten`."""
        return self._record(ar1_prewhiten(values, d=d))  # type: ignore[no-any-return]

    def redfit(self, time: npt.NDArray, values: npt.NDArray, **kwargs) -> RedfitResult:
        """See :func:`redfit`."""
        return self._record(redfit(time, values, **kwargs))  # type: ignore[no-any-return]

    def multitaper(self, values: npt.NDArray, **kwargs) -> MultitaperResult:
        """See :func:`multitaper_spectrum`."""
        return self._record(multitaper_spectrum(values, **kwargs))  # type: ignore[no-any-return]

    def cross_correlation(self, x: npt.NDArray, y: npt.NDArray, **kwargs) -> CrossCorrResult:
        """See :func:`cross_correlation`."""
        return self._record(cross_correlation(x, y, **kwargs))  # type: ignore[no-any-return]

    def runs(self, values: npt.NDArray, **kwargs) -> RunsResult:
        """See :func:`runs_test`."""
        return self._record(runs_test(values, **kwargs))  # type: ignore[no-any-return]

    def mann_kendall(self, values: npt.NDArray, **kwargs) -> MannKendallResult:
        """See :func:`mann_kendall`."""
        return self._record(mann_kendall(values, **kwargs))  # type: ignore[no-any-return]

    def autoassociation(self, time: npt.NDArray, values: npt.NDArray, **kwargs) -> AutoassociationResult:
        """See :func:`autoassociation`."""
        return self._record(autoassociation(time, values, **kwargs))  # type: ignore[no-any-return]

    @property
    def last_result(self) -> object | None:
        """The most recent result, whichever routine produced it."""
        return self._last_result
