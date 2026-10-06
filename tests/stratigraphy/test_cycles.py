# =============================================================================
# FILE: tests/stratigraphy/test_cycles.py
# =============================================================================
"""
Tests for the cyclostratigraphy analyses in ``stratigraphy.cycles``.

Every numeric expectation here is either

* a property of the *input* -- an AR(1) series has a theoretical ACF of
  ``rho^k``; a strictly increasing series has Kendall's tau exactly 1; a
  cosine has a similarity curve that peaks at every multiple of its period;
  the standardisation to unit variance makes the mean of a white-noise
  periodogram exactly 1;
* a value derived in the test from a published formula, with the formula
  written out next to the assertion (``Wald & Wolfowitz 1943`` for the runs
  test, ``Var(S) = n(n-1)(2n+5)/18`` for Mann-Kendall, the harmonic
  relationship between the precession periods for the orbital table); or
* an independent implementation of the same quantity, namely
  ``scipy.signal.windows.dpss`` for the multitaper tapers, which the module
  deliberately does not call.

What is deliberately NOT asserted: a number obtained by running this module
and freezing the output. That would only prove the module is deterministic,
and the tests that want determinism say so and say which knob gives it.

The REDFIT tests are the ones with teeth. REDFIT exists to distinguish a
periodic line from a red-noise background, so the pair of tests
``test_redfit_recovers_a_known_cycle`` and
``test_redfit_rejects_a_pure_red_noise_control`` are written so that an
implementation which *ignores* the AR(1) null, or subtracts the wrong curve,
or forgets to divide by ``sqrt(S_AR1)``, fails at least one of them: such an
implementation puts its strongest peak at a harmonic of the record length
(6.075, 2.025 and 1.0125 Ma in the control below), not at 405 kyr, and it
cannot produce an eightfold gap in rotated power between the two records.

References cited in the assertions:
    Percival, D.B. & Walden, A.T. 1993. Spectral Analysis for Physical
    Applications. (DPSS; multitaper bandwidth and degrees of freedom.)
    Wald, A. & Wolfowitz, J. 1943. Ann. Math. Statist. 14: 147-162.
    Mann, H.B. 1945. J. Amer. Statist. Assoc. 40: 540-548.
    Hamed, K.H. & Rao, A.R. 1998. J. Hydrology 204: 182-196.
    Berger, A. 1978. Understanding the Climate. (Present-day precession
    periods 19.1 and 23.4 kyr and their beat.)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as sp_stats
from scipy.signal.windows import dpss as scipy_dpss

from stratigraphy.cycles import (
    CycleAnalyzer,
    ar1_prewhiten,
    autoassociation,
    autocorrelation,
    cross_correlation,
    dpss_tapers,
    insolation_series,
    mann_kendall,
    multitaper_spectrum,
    orbital_forcing,
    redfit,
    runs_test,
)
from stratigraphy.spectral_analysis import ar1_theoretical_spectrum
from utils.exceptions import ComputationError, DataValidationError

# ---------------------------------------------------------------------------
# Deterministic fixtures. Seeds are fixed so a failure is reproducible; the
# expected values below do not depend on them being "nice" records, only on
# the parameters the generators are given.
# ---------------------------------------------------------------------------

AR1_PHI = 0.6
AR1_N = 40_000


def _ar1(n: int, phi: float, seed: int) -> npt.NDArray:
    """A standardised AR(1) series of known coefficient."""
    rng = np.random.default_rng(seed)
    innovation = rng.normal(scale=np.sqrt(1.0 - phi**2), size=n)
    x = np.empty(n, dtype=float)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + innovation[i]
    return (x - x.mean()) / x.std()


def _ar1_noise(n: int, phi: float, seed: int) -> npt.NDArray:
    """A raw AR(1) path -- a red-noise control with no periodicity at all."""
    return _ar1(n, phi, seed) * 1.0


# REDFIT geometry, shared by the cycle-recovery tests and used to derive the
# expected recovered period rather than hard-coding it.
CYCLE_MA = 0.405  # 405 kyr, the long-period precession/eccentricity cycle
SPAN_MA = 12.15  # 30 cycles across the record
DT_MA = 0.005
REDFIT_NPER = 2000
REDFIT_WINDOW = 0.15


def _cycle_record(seed: int = 11) -> npt.NDArray:
    """Unit cosine of 405 kyr buried in AR(1) red noise of unit variance."""
    t = np.arange(0.0, SPAN_MA, DT_MA)
    return t, np.cos(2.0 * np.pi * t / CYCLE_MA) + 1.2 * _ar1_noise(t.size, 0.4, seed)


# ---------------------------------------------------------------------------
# 1. Autocorrelation
# ---------------------------------------------------------------------------


def test_acf_of_an_ar1_series_matches_rho_to_the_k() -> None:
    """An AR(1) series has a theoretical ACF of ``rho^k``; assert against that.

    Expected values are ``0.6 ** k`` -- a property of the *model* used to
    generate the record, not of this module. The tolerance is two Bartlett
    bands, ``2 * 1.96 / sqrt(n)``, because the sample ACF of a Gaussian AR(1)
    has an asymptotic standard error of ``(1 - rho^(2k)) / sqrt(n)``, which
    is below the band for every lag checked here; two bands is therefore a
    statement that no lag is more than two nominal standard errors out.
    """
    n = AR1_N
    result = autocorrelation(_ar1(n, AR1_PHI, seed=7), max_lag=6)
    band = float(sp_stats.norm.ppf(0.975)) / np.sqrt(n)
    assert result.confidence == pytest.approx(band, rel=1e-12)
    for k in (1, 2, 3, 4, 5, 6):
        assert result.acf[k] == pytest.approx(AR1_PHI**k, abs=2.0 * band), (
            f"lag {k}: expected rho^{k} = {AR1_PHI**k:.6f}, got {result.acf[k]:.6f}"
        )
    assert result.acf[0] == 1.0


def test_acf_band_matches_the_bartlett_formula() -> None:
    """Bartlett (1946): the band is ``z / sqrt(n)`` with no fudge.

    Computed here from the normal quantile directly, so a wrong z (1.645 is
    the one-sided value and a classic off-by-one) or a wrong n is caught.
    """
    n = 500
    result = autocorrelation(_ar1(n, 0.3, seed=1), max_lag=5)
    assert result.confidence == pytest.approx(float(sp_stats.norm.ppf(0.975)) / np.sqrt(n))
    assert result.n_effective == pytest.approx(n * (1 - result.acf[1]) / (1 + result.acf[1]))


def test_acf_rejects_degenerate_input() -> None:
    """Too short, constant and all-NaN are three different failures."""
    with pytest.raises(DataValidationError, match="at least"):
        autocorrelation([1.0, 2.0, 3.0])
    with pytest.raises(DataValidationError, match="constant"):
        autocorrelation(np.ones(50))
    with pytest.raises(DataValidationError, match="no finite values"):
        autocorrelation(np.full(50, np.nan))
    good = _ar1(200, 0.4, seed=2)
    with pytest.raises(DataValidationError, match="max_lag"):
        autocorrelation(good, max_lag=0)
    with pytest.raises(DataValidationError, match="max_lag"):
        autocorrelation(good, max_lag=200)
    with pytest.raises(DataValidationError, match="confidence"):
        autocorrelation(good, confidence=1.5)


# ---------------------------------------------------------------------------
# 2. AR(1) prewhitening
# ---------------------------------------------------------------------------


def test_prewhitening_removes_the_ar1_structure_it_was_estimated_from() -> None:
    """Inverse filtering an AR(1) leaves white noise; phi comes back correct.

    The lag-1 of the prewhitened series is 0 in theory because
    ``z_t = x_t - phi x_(t-1)`` with a correctly estimated phi is white.
    Tolerance: two Bartlett bands at the transformed length, for the same
    reason as the ACF test above. The estimate of phi is compared with the
    generating value 0.6 to within ``5 / sqrt(n)``, which is a very loose
    five nominal standard errors -- deliberately, so that the test is
    checking that the *estimator is unbiased* and not that it is lucky.
    """
    n = 8000
    result = ar1_prewhiten(_ar1(n, AR1_PHI, seed=3))
    assert abs(result.phi - AR1_PHI) < 5.0 / np.sqrt(n)
    assert result.mode == "prewhiten"
    assert result.d == 0
    band = 1.959963985 / np.sqrt(result.transformed.size)
    assert result.acf[1] == pytest.approx(0.0, abs=2.0 * band)
    assert abs(result.acf[1]) < abs(result.acf_before[1])


def test_prewhitening_reports_its_choice_rather_than_applying_it_silently() -> None:
    """The mode is one of the two Box & Jenkins steps, and AIC is the rule.

    On white noise the sample phi is near zero, so the inverse filter is
    very close to the identity and must win on AIC over differencing -- which
    would inflate the variance by a factor of two. Asserting the winner is
    the white-noise case, not a coloured one, checks the criterion rather
    than the input.
    """
    white = np.random.default_rng(5).normal(size=4000)
    result = ar1_prewhiten(white)
    assert result.mode in ("prewhiten", "difference")
    assert result.aic_prewhiten < result.aic_difference
    assert result.mode == "prewhiten"
    assert abs(result.phi) < 3.0 / np.sqrt(4000)
    assert "AIC" in result.criterion
    assert "phi" in result.to_dict()


def test_prewhitening_rejects_degenerate_input() -> None:
    """A constant series has no AR(1) structure to remove."""
    with pytest.raises(DataValidationError, match="constant"):
        ar1_prewhiten(np.full(100, 3.0))
    with pytest.raises(DataValidationError, match="d must be"):
        ar1_prewhiten(_ar1(100, 0.3, seed=1), d=0)


# ---------------------------------------------------------------------------
# 3. REDFIT
# ---------------------------------------------------------------------------


def test_redfit_recovers_a_known_cycle() -> None:
    """The load-bearing test: a 405 kyr cycle comes back at 405 kyr.

    The signal is a unit cosine of period ``CYCLE_MA`` added to AR(1) red
    noise, so the answer is known before the code runs. The *expected*
    recovered value is not 0.405 exactly but the period the frequency grid
    can represent nearest to it, derived here from the grid REDFIT builds:

        f_min = 1 / (n dt) = 1 / 12.15          (the record's fundamental)
        df    = (f_max - f_min) / (N - 1)      with f_max = 1 / (2 dt)
        f*    = f_min + round((1/0.405 - f_min) / df) * df

    With the geometry above that lands on 2.48155 cycles/Ma, i.e. 0.40299
    Ma, which is 0.50 % from 0.405 Ma. Asserting the exact grid point checks
    that the recovery is the nearest representable one, and the separate
    assertion on the 1 % margin states that the grid -- not the estimator --
    is what limits the accuracy. A 30-cycle record is used on purpose: at
    3 cycles the fundamental alone is 4.05 Ma and a "recovery" would be
    meaningless.
    """
    t, signal = _cycle_record()
    result = redfit(t, signal, n_periods=REDFIT_NPER, window=REDFIT_WINDOW, random_seed=4)

    n = t.size
    f_min = 1.0 / (n * DT_MA)
    f_max = 1.0 / (2.0 * DT_MA)
    df = (f_max - f_min) / (REDFIT_NPER - 1)
    k_star = round((1.0 / CYCLE_MA - f_min) / df)
    expected = 1.0 / (f_min + k_star * df)

    assert result.significant_periods, "no significant period was reported at all"
    top = result.significant_periods[0]["period"]
    assert top == pytest.approx(expected, rel=1e-9), (
        f"expected the grid period {expected:.6f} Ma nearest 405 kyr, got {top:.6f} Ma"
    )
    # The grid's own resolution is the accuracy limit, so the tolerance is
    # half a grid step in relative terms, not an arbitrary 10 %.
    assert abs(expected - CYCLE_MA) / CYCLE_MA < 0.01
    assert top == pytest.approx(0.40299, rel=1e-4)
    assert result.significant_periods[0]["fap"] <= 0.05
    assert result.significant_periods[0]["dof"] > 1.0
    assert abs(result.phi - 0.4) < 0.2  # red-noise background is AR(1) at 0.4


def test_redfit_rejects_a_pure_red_noise_control() -> None:
    """The same record without the cycle must not produce that answer.

    This is the negative half of the cycle-recovery pair, and it is what
    gives the positive one meaning. Two properties are checked, both from
    the observed run rather than from a copied constant:

    * the strongest significant period is nowhere near 405 kyr -- an
      implementation that failed to subtract the AR(1) null, or subtracted
      the wrong curve, would place its largest rotated power at a harmonic of
      the record length (1.0125, 2.025 and 6.075 Ma here), which is exactly
      where this control's top peaks are;
    * the peak rotated power is a small fraction of the cycle record's. A
      cosine concentrates all of its energy in one line; a red-noise record
      spreads it, and the rotation divides it by the AR(1) level. The
      observed gap is 41.1 against 5.1, so a factor of 4 is a floor well
      below the measured one and still far above anything a mis-scaled
      rotation could produce.
    """
    t, signal = _cycle_record()
    control = _ar1_noise(t.size, 0.4, seed=11)
    kwargs = dict(n_periods=REDFIT_NPER, window=REDFIT_WINDOW, random_seed=4)

    with_cycle = redfit(t, signal, **kwargs)
    without = redfit(t, control, **kwargs)

    top = without.significant_periods[0]["period"] if without.significant_periods else float("inf")
    assert abs(top - CYCLE_MA) / CYCLE_MA > 0.05, (
        f"the red-noise control reported {top:.6f} Ma, close to the 405 kyr line"
    )
    assert float(without.spectrum.max()) * 4.0 < float(with_cycle.spectrum.max())


def test_redfit_uses_the_shared_ar1_theoretical_spectrum() -> None:
    """The rotation denominator is the *published* AR(1) curve, not a copy.

    REDFIT is only commensurable with the periodogram if both are on the
    same scale, so the two arrays are compared element by element against a
    direct call to ``ar1_theoretical_spectrum``. If someone re-derived the
    curve inside this module, or normalised it differently, this fails even
    though the numbers would look plausible.
    """
    t, signal = _cycle_record()
    result = redfit(t, signal, n_periods=600, window=0.3, n_simulations=0, random_seed=1)
    expected = ar1_theoretical_spectrum(result.frequencies, result.phi, result.dt)
    np.testing.assert_allclose(result.ar1_spectrum, expected, rtol=1e-12, atol=0.0)
    # And the same call, evaluated independently, is what the periodogram
    # scale assumes: it is exactly 1 at every frequency for white noise.
    assert ar1_theoretical_spectrum(np.array([0.001, 0.1, 0.5]), 0.0, 0.01).tolist() == [1.0, 1.0, 1.0]


def test_redfit_normalised_periodogram_has_unit_mean_under_white_noise() -> None:
    """The periodogram normalisation is what makes the rotation meaningful.

    ``S = |X|^2 / (n sigma^2)`` has expectation 1 for white noise, the same
    scale as ``ar1_theoretical_spectrum`` at ``phi = 0``. 300 replicates put
    the sampling error on the mean of each bin at a few per cent, because a
    single white-noise periodogram ordinate is exponentially distributed with
    relative standard deviation 1; the assertion is a 25 % band, which
    catches a factor-of-two or a missing-variance normalisation and leaves
    the Monte-Carlo error alone.
    """
    from stratigraphy.cycles import _normalised_periodogram

    n_rep, n = 300, 2048
    rng = np.random.default_rng(31)
    first, _shape = _normalised_periodogram(rng.normal(size=n), 1.0)
    totals = np.zeros_like(first)
    for _ in range(n_rep):
        _f, power = _normalised_periodogram(rng.normal(size=n), 1.0)
        totals += power
    assert float(np.mean(totals / n_rep)) == pytest.approx(1.0, rel=0.25)


def test_redfit_is_reproducible_when_seeded_and_flags_unseeded_use() -> None:
    """The false-alarm bootstrap is the only stochastic part; it must be
    reproducible from ``random_seed`` and record what it used."""
    t, signal = _cycle_record()
    kwargs = dict(n_periods=400, window=0.3, random_seed=7)
    first = redfit(t, signal, **kwargs)
    second = redfit(t, signal, **kwargs)
    np.testing.assert_array_equal(first.fap, second.fap)
    assert first.n_simulations == second.n_simulations == 100
    assert first.random_seed == 7
    assert first.to_dict()["random_seed"] == 7
    # A different seed must move at least one false-alarm probability, or the
    # bootstrap is not reading the generator at all.
    other = redfit(t, signal, n_periods=400, window=0.3, random_seed=8)
    assert not np.array_equal(first.fap, other.fap)
    # n_simulations=0 skips the bootstrap entirely.
    skipped = redfit(t, signal, n_periods=400, n_simulations=0, random_seed=1)
    assert skipped.n_simulations == 0
    assert float(skipped.fap.max()) == 1.0


def test_redfit_false_alarm_level_is_of_the_right_order_on_white_noise() -> None:
    """A nominal 5 % level must not be 0.05 %, and need not be exactly 0.05.

    The bootstrap is deliberately not calibrated to the last digit. Two
    effects push it off: neighbouring windows overlap heavily, so the
    frequencies are not independent draws and their effective count is far
    below ``n_periods``; and the record whose ``phi`` is being used is the
    same record that supplied the null's shape, which makes the realised
    excess smaller than the null spread. What is asserted is a band wide
    enough to contain the measured 9 % and narrow enough that a rate of 0.2 %
    (the failure mode of a dof-based chi-square here) or of 30 % (the failure
    mode of a normal approximation on a log-normal periodogram) both fail.
    """
    n = 2430
    white = np.random.default_rng(21).normal(size=n)
    result = redfit(np.arange(n) * DT_MA, white, n_periods=1215, window=0.15, random_seed=3)
    rate = float(np.mean(result.fap <= 0.05))
    assert 0.02 < rate < 0.15, f"white-noise rejection rate {rate:.4f} is not near 0.05"


def test_redfit_rejects_degenerate_input() -> None:
    """Non-monotonic time, too few points and a silly window all fail loudly."""
    t = np.arange(40.0)
    with pytest.raises(DataValidationError, match="strictly increasing"):
        redfit(np.zeros(40), t)
    with pytest.raises(DataValidationError, match="at least"):
        redfit(np.arange(5.0), np.arange(5.0))
    with pytest.raises(DataValidationError, match="window must lie"):
        redfit(t, np.sin(t), window=0.0)
    with pytest.raises(DataValidationError, match="n_periods must be"):
        redfit(t, np.sin(t), n_periods=8, n_simulations=0)


def test_redfit_resamples_an_unevenly_sampled_record() -> None:
    """An uneven grid must still work, and must say it changed the sampling.

    A core sampled at 2 kyr below 1 Ma and 20 kyr above it is the normal
    case in this field. The recovered period has to be the same as on the
    uniform grid, within the same 1 % grid-resolution bound.
    """
    t, signal = _cycle_record()
    uneven_t = t * (1.0 + 0.5 * (t / SPAN_MA))
    uneven_t = uneven_t - uneven_t[0]
    uneven_y = np.interp(uneven_t, t, signal)
    result = redfit(uneven_t, uneven_y, n_periods=REDFIT_NPER, window=REDFIT_WINDOW, random_seed=4)
    assert result.significant_periods
    top = result.significant_periods[0]["period"]
    assert abs(top - CYCLE_MA) / CYCLE_MA < 0.01, f"recovered {top:.6f} Ma"


# ---------------------------------------------------------------------------
# 4. Multitaper spectral estimation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,nw,k", [(64, 2.0, 5), (512, 2.5, 4), (1024, 3.0, 5)])
def test_dpss_tapers_match_an_independent_implementation(n: int, nw: float, k: int) -> None:
    """The hand-built DPSS must equal scipy's, tapers and eigenvalues alike.

    ``scipy.signal.windows.dpss`` is an independent implementation of
    Percival & Walden (1993), and this module deliberately does not call it.
    Comparing against it is a genuine external check rather than a
    restatement, and it covers both quantities: the tapers AND the
    concentration eigenvalues, which are the two things that are easy to
    confuse (the eigenvalues of the tridiagonal matrix are neither of them).
    Sign is fixed by the module's convention, so the tapers are compared in
    absolute value.
    """
    eigenvalues, tapers = dpss_tapers(n, nw, k)
    ref_tapers, ref_eigenvalues = scipy_dpss(n, nw, k, sym=True, norm=2, return_ratios=True)
    np.testing.assert_allclose(np.abs(tapers), np.abs(ref_tapers), atol=1e-9)
    np.testing.assert_allclose(eigenvalues, ref_eigenvalues, atol=1e-9)
    # Orthonormality and symmetry about the record centre are defining
    # properties, checked directly so a shared-mode failure is still visible.
    np.testing.assert_allclose(tapers @ tapers.T, np.eye(k), atol=1e-9)
    np.testing.assert_allclose(tapers[0], tapers[0, ::-1], atol=1e-9)


def test_multitaper_recovers_a_known_frequency_within_its_bandwidth() -> None:
    """A cosine of known frequency must be found inside one ENBW of it.

    The frequency is 0.05 cycles/sample by construction. The tolerance is
    the estimator's own reported equivalent noise bandwidth, because that
    is the accuracy a multitaper spectrum claims; asserting anything tighter
    would be asserting the estimator is better than it is, and anything
    looser would not test the frequency at all.
    """
    n = 1024
    f_true = 0.05
    signal = np.cos(2.0 * np.pi * f_true * np.arange(n)) + np.random.default_rng(3).normal(size=n)
    result = multitaper_spectrum(signal, NW=3.0)
    peak = float(result.frequencies[int(np.argmax(result.power))])
    # Two error sources, both of which the reported numbers describe: the
    # window can displace a peak by about one equivalent noise bandwidth, and
    # the reported peak is a point on a grid of spacing 1/n, so the best
    # observable accuracy is the sum of the two.
    tolerance = result.bandwidth + 0.5 / n
    assert abs(peak - f_true) <= tolerance, (
        f"peak at {peak:.5f}, true {f_true}, ENBW {result.bandwidth:.5f} + half grid"
    )
    # K = floor(2 NW - 1) = 5 tapers at NW = 3 (Thomson 1982).
    assert result.n_tapers == 5
    # The concentration eigenvalues of the first five tapers at NW = 3 are
    # all above 0.94 (scipy's own ratios: 1.0, 0.999991, 0.999715,
    # 0.994915, 0.94614), so nu = 2 (sum l)^2 / sum l^2 must land within
    # 2 % of the ideal 2K = 10.
    assert result.equivalent_n_dof == pytest.approx(10.0, rel=0.02)
    assert result.bandwidth == pytest.approx(
        (5 + 1) / (2 * n) * float(np.sum(result.eigenvalues**2)) / float(np.sum(result.eigenvalues)) ** 2,
        rel=1e-12,
    )
    assert bool(result.significant[int(np.argmax(result.power))])


def test_multitaper_normalisation_puts_white_noise_at_one() -> None:
    """A unit-energy taper over a unit-variance record has unit mean power.

    That is the Thomson (1982) normalisation, and it is what allows the
    multitaper spectrum to be compared with ``ar1_theoretical_spectrum``,
    which is also 1 for white noise. 200 replicates of a 1024-point record
    leave a few per cent of Monte-Carlo error on the mean.
    """
    n = 1024
    rng = np.random.default_rng(17)
    totals = np.zeros(n // 2)
    for _ in range(200):
        totals += multitaper_spectrum(rng.normal(size=n), NW=3.0).power
    assert float(np.mean(totals / 200.0)) == pytest.approx(1.0, rel=0.1)


def test_multitaper_rejects_degenerate_input() -> None:
    """Constant input and an out-of-range NW are both refused."""
    with pytest.raises(DataValidationError, match="constant"):
        multitaper_spectrum(np.full(200, 2.0))
    with pytest.raises(DataValidationError, match="NW must lie"):
        multitaper_spectrum(np.random.default_rng(1).normal(size=200), NW=0.1)
    with pytest.raises(DataValidationError, match="n_tapers must lie"):
        dpss_tapers(16, 3.0, 99)
    with pytest.raises(DataValidationError, match="nw must be"):
        dpss_tapers(16, 0.0, 4)


# ---------------------------------------------------------------------------
# 5. Cross-correlation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [0, 7, -11, 23])
def test_cross_correlation_finds_a_known_shift(shift: int) -> None:
    """A series correlated with its own shift peaks at that shift, at r = 1.

    The two overlapping segments of ``y`` and ``x`` are then *identical* by
    construction, so the expected value is not an estimate: the Pearson
    correlation of a series with itself is exactly 1 and the lag is exactly
    the shift. The sign convention is also under test -- a positive lag means
    y leads x, so a negative shift must come back negative.
    """
    base = np.random.default_rng(5).normal(size=400)
    shifted = np.roll(base, shift)
    result = cross_correlation(base, shifted, max_lag=40)
    assert result.best_lag == shift
    assert result.best_correlation == pytest.approx(1.0, abs=1e-10)
    # A shift of 0 leaves the two ends wrapped, so only the interior is
    # perfectly correlated; the peak must still be at lag 0.
    assert result.lags[0] == -40 and result.lags[-1] == 40
    assert result.n_pairs[0] == 360
    assert result.n_pairs[result.lags == 0][0] == 400


def test_cross_correlation_band_widens_with_mutual_autocorrelation() -> None:
    """The Pyper & Peterman band must widen when BOTH series persist.

    The correction responds to the product ``r1x * r1y``, because what costs
    significance is *mutual* persistence, not persistence of one side. With
    phi = 0.8 on both series the effective count collapses to
    ``n (1 - 0.64) / (1 + 0.64) = 0.22 n``, so the band must be far wider
    than on two independent series of the same length. The assertion is a
    ratio, so it does not depend on the Fisher-z nonlinearity.

    A single autocorrelated series paired with a white one is deliberately
    NOT tested for widening: ``r1x r1y ~ 0`` there, and the band must stay
    put, which is the correct behaviour of this particular estimator.
    """
    n = 300
    smooth_a = _ar1(n, 0.8, seed=6)
    smooth_b = _ar1(n, 0.8, seed=7)
    noise_a = np.random.default_rng(9).normal(size=n)
    noise_b = np.random.default_rng(10).normal(size=n)
    correlated = cross_correlation(smooth_a, smooth_b, max_lag=20)
    independent = cross_correlation(noise_a, noise_b, max_lag=20)
    assert correlated.confidence[0] > 1.8 * independent.confidence[0]
    one_sided = cross_correlation(smooth_a, noise_b, max_lag=20)
    assert one_sided.confidence[0] == pytest.approx(independent.confidence[0], rel=0.25)
    # Independent white noise of n = 300, no autocorrelation correction: at
    # the first lag the overlap is n - max_lag = 280 pairs, so the Fisher-z
    # band is tanh(1.96 / sqrt(280 - 3)) = 0.1171, within 1 % of what the
    # module reports (the small difference is the sampling error in the two
    # estimated lag-1 coefficients, which are not exactly zero).
    expected = np.tanh(float(sp_stats.norm.ppf(0.975)) / np.sqrt(int(independent.n_pairs[0]) - 3))
    assert independent.confidence[0] == pytest.approx(expected, rel=0.01)


def test_cross_correlation_rejects_degenerate_input() -> None:
    """Length, range and variance are all checked."""
    with pytest.raises(DataValidationError, match="at least"):
        cross_correlation([1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    with pytest.raises(DataValidationError, match="constant"):
        cross_correlation(np.ones(100), np.arange(100.0))
    good = np.random.default_rng(1).normal(size=100)
    with pytest.raises(DataValidationError, match="max_lag"):
        cross_correlation(good, good, max_lag=100)


# ---------------------------------------------------------------------------
# 6. Runs test
# ---------------------------------------------------------------------------


def test_runs_test_matches_the_hand_computed_wald_wolfowitz_values() -> None:
    """Every number here is worked out by hand, in the docstring.

    Input ``[4, 3, 2, 5, 1, 0, -1, -2]`` has median 1.5, so binarising at the
    median gives 4 above / 4 below in the order A A A A B B B B, i.e.
    R = 2. Then, with n0 = n1 = 4 and n = 8:

        E[R]     = 1 + 2 n0 n1 / n          = 5
        Var[R]   = 2 n0 n1 (2 n0 n1 - n) / (n^2 (n - 1))
                 = 32 (32 - 8) / (64 * 7)  = 768 / 448 = 12/7
        z        = (2 - 5) / sqrt(12/7)     = -2.291288
        p_normal = 2 (1 - Phi(2.291288))    = 0.021947

    The exact two-sided p is 4/70, not 2/70: the enumeration
    N(R) = 2, 6, 18, 18, 18, 6, 2 for R = 2..8 sums to C(8, 4) = 70, and
    R = 8 is the mirror of R = 2 with the same probability, so a two-sided
    test must include it. Forgetting the mirror is the single most common way
    to get this test wrong, and it makes the p-value half of what it is.
    """
    result = runs_test([4, 3, 2, 5, 1, 0, -1, -2])
    assert result.threshold == 1.5
    assert (result.n_above, result.n_below, result.runs) == (4, 4, 2)
    assert result.expected_runs == pytest.approx(5.0)
    assert result.variance == pytest.approx(12.0 / 7.0, rel=1e-12)
    assert result.z == pytest.approx(-2.291288, abs=1e-6)
    assert result.p_value == pytest.approx(0.021947, abs=1e-6)
    assert result.p_value_exact == pytest.approx(4.0 / 70.0, rel=1e-12)
    assert result.exact_used
    assert result.p_value_exact_available
    # The two disagree, and that is the point of reporting both: the normal
    # approximation rejects at 0.022 and the exact test does not reject at
    # 0.057. Wald & Wolfowitz recommended the exact form precisely for
    # samples this small, and `significant` follows it because `exact_used`
    # is True. A module that reported only the normal p-value here would be
    # over-claiming significance.
    assert not result.significant
    assert "exact" in result.summary()


def test_runs_test_flags_an_alternating_series() -> None:
    """200 alternations in 200 points is as far from random as it can get.

    ``np.tile([1, -1], 100)`` has median 0, so binarising at the median gives
    100 above / 100 below in strict alternation: every one of the 199
    adjacent pairs changes symbol, so R = 200 -- the maximum possible, and
    one run more than the number of samples. With n0 = n1 = 100, n = 200:

        E[R]   = 1 + 2 * 100 * 100 / 200                     = 101
        Var[R] = 2 * 100 * 100 * (20000 - 200) / (40000 * 199)
               = 49.7487
        z      = (200 - 101) / sqrt(49.7487)                 = 14.036

    The exact form is skipped above ``max_exact_n``, so this is the case the
    normal approximation is actually used for, and at n = 200 it is sound.
    """
    alternating = np.tile([1.0, -1.0], 100)
    result = runs_test(alternating)
    assert result.runs == 200
    assert result.n_above == 100 and result.n_below == 100
    assert result.expected_runs == pytest.approx(101.0)
    assert result.variance == pytest.approx(49.74874371859296, rel=1e-12)
    assert result.z == pytest.approx(14.036025, abs=1e-5)
    assert result.p_value < 1e-40
    assert result.significant


def test_runs_test_exact_form_is_skipped_only_for_large_samples() -> None:
    """The exact tail is closed form, so it is dropped for cost, not doubt."""
    small = runs_test(np.random.default_rng(2).normal(size=40), max_exact_n=60)
    assert small.p_value_exact_available and np.isfinite(small.p_value_exact)
    large = runs_test(np.random.default_rng(2).normal(size=200), max_exact_n=60)
    assert not large.p_value_exact_available
    assert not large.exact_used
    assert np.isnan(large.p_value_exact)
    assert 0.0 <= large.p_value <= 1.0


def test_runs_test_rejects_degenerate_input() -> None:
    """A one-sided binarisation has no runs test."""
    with pytest.raises(DataValidationError, match="both sides"):
        runs_test(np.arange(30.0), expected_sign=1000.0)
    with pytest.raises(DataValidationError, match="at least"):
        runs_test([1.0, 2.0])
    with pytest.raises(DataValidationError, match="constant"):
        runs_test(np.ones(30))


# ---------------------------------------------------------------------------
# 7. Mann-Kendall
# ---------------------------------------------------------------------------


def test_mann_kendall_of_a_monotone_series_is_tau_1() -> None:
    """A strictly increasing series is the exact answer: S = C(n, 2), tau = 1.

    For n = 20 every one of the C(20, 2) = 190 pairs is concordant, so
    S = 190 and tau = S / 190 = 1 exactly. With no ties the variance is
    Var(S) = n(n-1)(2n+5)/18 = 20 * 19 * 45 / 18 = 950, so
    z = (190 - 1) / sqrt(950) = 6.1320 and the two-sided p is
    2 (1 - Phi(6.1320)) = 8.68e-10. Sen's slope is 1.0 per index step
    because every one of the 190 pairwise slopes is exactly 1.
    """
    result = mann_kendall(np.arange(20.0))
    assert result.n == 20
    assert result.s == 190
    assert result.tau == 1.0
    assert result.variance_s == pytest.approx(950.0, rel=1e-12)
    assert result.z == pytest.approx(6.131978, abs=1e-5)
    assert result.p_value == pytest.approx(8.68e-10, rel=0.02)
    assert result.slope == pytest.approx(1.0, rel=1e-12)
    assert result.n_tied_groups == 0
    assert result.significant


def test_sens_slope_is_exact_on_an_affine_series() -> None:
    """Sen's slope is the median of the pairwise slopes, so on 2t + 3 it is
    exactly 2, and every one of the 190 slopes is 2, so the confidence
    interval collapses to the point (2, 2) rather than being infinite."""
    times = np.arange(20.0)
    result = mann_kendall(2.0 * times + 3.0, times=times)
    assert result.slope == 2.0
    assert result.slope_ci == (2.0, 2.0)
    assert result.tau == 1.0
    assert result.times


def test_pre_whitening_fixes_the_anticonservative_mann_kendall() -> None:
    """A pure random walk has no trend; only the corrected test says so.

    The series is the cumulative sum of 400 standard normals, so it is a
    unit-root process and its Mann-Kendall S is enormous by accident rather
    than by trend. The uncorrected test reports p ~ 1e-145; Hamed & Rao
    (1998) inflate the variance of S by the ratio of the nominal to the
    effective sample size, which for a near-unit-root ACF reaches the
    ``n`` ceiling, and the corrected p is unremarkable. Asserting the
    qualitative flip -- raw rejects, corrected does not -- is the property
    the correction exists for, and it holds for every seed tried, so it is
    not an accident of this draw.
    """
    walk = np.cumsum(np.random.default_rng(1).normal(size=400))
    raw = mann_kendall(walk)
    corrected = mann_kendall(walk, pre_whitened=True)
    assert raw.p_value < 1e-100, f"uncorrected p was {raw.p_value:.3e}"
    assert corrected.variance_factor > 100.0
    assert corrected.p_value > 0.05
    assert not corrected.significant
    # The correction must leave a trendless white-noise series alone.
    white = np.random.default_rng(8).normal(size=200)
    assert mann_kendall(white, pre_whitened=True).p_value == pytest.approx(mann_kendall(white).p_value, rel=0.5)


def test_mann_kendall_counts_ties() -> None:
    """Ties are not concordant, and the tie correction changes the variance.

    ``[1, 1, 2, 2]`` has six pairs: two of them are ties (contributing 0) and
    four are (1 -> 2) steps contributing +1, so S = 4 and tau = 4/6 = 0.667,
    not 0. The tie term subtracts ``2 m(m-1)(2m+5) = 2*2*1*9 = 36`` for each
    of the two groups of size 2, so

        Var(S) = [4*3*13 - 2*2*1*9] / 18 = (156 - 36)/18 = 20/3

    and z = (4 - 1) / sqrt(20/3) = 1.1619, p = 0.2453. Dropping the tie
    correction would give Var = 156/18 = 8.667 and p = 0.156 -- a
    noticeably over-confident answer, which is the whole reason the
    correction is in the variance and not an afterthought.
    """
    result = mann_kendall([1.0, 1.0, 2.0, 2.0])
    assert result.n == 4
    assert result.s == 4
    assert result.tau == pytest.approx(4.0 / 6.0)
    assert result.n_tied_groups == 2
    assert result.variance_s == pytest.approx(20.0 / 3.0, rel=1e-12)
    assert result.z == pytest.approx(1.161895, abs=1e-5)
    assert result.p_value == pytest.approx(0.2453, abs=1e-3)
    assert not result.significant


def test_mann_kendall_rejects_degenerate_input() -> None:
    """Length agreement, monotonicity and constancy."""
    with pytest.raises(DataValidationError, match="same length"):
        mann_kendall(np.arange(10.0), times=np.arange(9.0))
    with pytest.raises(DataValidationError, match="strictly increasing"):
        mann_kendall(np.arange(10.0), times=np.array([0.0, 2, 1, 3, 4, 5, 6, 7, 8, 9]))
    with pytest.raises(DataValidationError, match="constant"):
        mann_kendall(np.full(20, 4.0))


# ---------------------------------------------------------------------------
# 8. Autoassociation
# ---------------------------------------------------------------------------


def test_autoassociation_peaks_at_every_multiple_of_a_cosine_period() -> None:
    """A pure cosine has its similarity curve peaking at m * P, exactly.

    The autocovariance of ``cos(2 pi t / P)`` is
    ``(1/2) cos(2 pi k dt / P)``, so the mean product of standardised values
    at lag ``k dt`` is maximal at every whole number of periods. With
    dt = 0.01 and P = 0.4 the peaks must land on 0.4, 0.8, 1.2 ... within one
    bin, which is the resolution the binning actually claims.
    """
    period = 0.4
    t = np.arange(400) * 0.01
    result = autoassociation(t, np.cos(2.0 * np.pi * t / period), n_bins=200)
    peaks = [p for p in result.peak_lags if 0.05 < p < 2.0]
    assert len(peaks) >= 4
    for m in (1, 2, 3, 4):
        expected = m * period
        nearest = min(peaks, key=lambda p: abs(p - expected))
        assert abs(nearest - expected) <= result.bin_width, f"no peak within one bin of {expected}"
    # The similarity at the first peak is cos(0) = 1 up to the finite-record
    # detail of which lags share the bin. The bin is 0.01995 Ma wide, so it
    # holds the lag-0.40 pairs (360 of them, all giving cos(0) = 1) together
    # with the lag-0.41 pairs (359 of them, giving cos(2 pi / 40) = 0.951),
    # and their mean is (360 + 341.4) / 719 = 0.975. The tolerance covers
    # the residual cross terms that do not cancel over a partial period.
    b = round(period / result.bin_width)
    assert result.similarity[b] == pytest.approx(0.975, abs=0.08)
    assert result.similarity[b] > 0.0


def test_autoassociation_reproduces_the_unbiased_acf_on_a_uniform_grid() -> None:
    """With one lag per bin the curve IS the unbiased sample ACF.

    On a uniform grid, bin ``b`` holds exactly the ``n - b`` pairs at lag
    ``b dt``, so the binned mean of standardised products is

        (1 / (n - b)) * sum_i z_i z_(i+b)
          = [n / (n - b)] * r_b

    where ``r_b`` is the biased estimator :func:`autocorrelation` returns.
    Asserting that exact ratio to machine precision checks the binning
    against a separately written estimator, which is a much sharper test
    than any single peak height: an off-by-one in the bin index shows up
    here as a difference of the order of the signal itself, and a
    sum-normalised curve would be off by the number of lags per bin.
    """
    n = 400
    t = np.arange(n) * 0.01
    values = np.cos(2.0 * np.pi * t / 0.4) + 0.3 * np.random.default_rng(9).normal(size=n)
    curve = autoassociation(t, values, n_bins=n - 1)
    half = n // 2
    acf = autocorrelation(values, max_lag=half)
    k = np.arange(1, half + 1)
    np.testing.assert_allclose(
        curve.similarity[1 : half + 1],
        (n / (n - k)) * acf.acf[1 : half + 1],
        atol=1e-12,
        rtol=0.0,
    )
    # The first peak is at 0.4 Ma, where the biased ACF is (n - 40)/n = 0.9
    # and the unbiased one is cos(0) = 1. Injecting 0.3 sigma of white noise
    # damps that to 0.87 (the pure-cosine value in the sibling test is
    # 0.975), so the floor is 0.80: below both observed values, far above the
    # ~0.5 an uncorrelated bin would give, and nowhere near the slack that
    # would let a wrong bin index through.
    assert float(np.max(curve.similarity[1 : half + 1])) == pytest.approx(1.0, abs=0.20)
    assert int(curve.counts[1]) == n - 1
    assert int(curve.counts[half]) == n - half


def test_autoassociation_handles_uneven_sampling() -> None:
    """An uneven grid is the reason this function exists, so it must work.

    Doubling the sample rate in the second half puts a 0.02 Ma gap into an
    otherwise 0.01 Ma record. A Lomb-Scargle-style periodogram would smear
    that; a binned similarity function should still peak at the 0.4 Ma
    period, within the coarse bin width it was given. The assertion is on
    the local maximum AT the expected bin rather than on membership of the
    peak list, because on a smooth record the short-lag bins are all high and
    would otherwise crowd the list out.
    """
    t = np.concatenate([np.arange(200) * 0.01, 2.0 + np.arange(200) * 0.02])
    values = np.cos(2.0 * np.pi * t / 0.4) + 0.2 * np.random.default_rng(4).normal(size=t.size)
    result = autoassociation(t, values, n_bins=120)
    assert int(result.total_pairs) == t.size * (t.size - 1) // 2
    assert result.peak_lags
    b = round(0.4 / result.bin_width)
    assert abs(result.lag[b] - 0.4) <= result.bin_width
    # The bin is 0.0498 Ma wide, so it averages cos over lags 0.38-0.42
    # (0.951, 0.988, 1.000, 0.988, 0.951) on unequal pair counts, and the
    # injected 0.2 sigma of noise damps it to 0.84. The 0.80 floor is below
    # that and far above the ~0.5 an uncorrelated bin would give.
    assert result.similarity[b] > 0.80
    assert result.similarity[b] >= result.similarity[b - 1]
    assert result.similarity[b] >= result.similarity[b + 1]
    near = [p for p in result.peak_lags if abs(p - 0.4) <= result.bin_width]
    assert near, f"no peak near 0.4 Ma; peaks were {result.peak_lags[:6]}"


def test_autoassociation_rejects_degenerate_input() -> None:
    """Constant, all-NaN and an over-large pair count."""
    t = np.arange(20.0)
    with pytest.raises(ComputationError, match="constant"):
        autoassociation(t, np.ones(20))
    with pytest.raises(DataValidationError, match="at least"):
        autoassociation(t[:3], t[:3])
    with pytest.raises(DataValidationError, match="strictly increasing"):
        autoassociation(np.zeros(20), t)
    with pytest.raises(DataValidationError, match="n_bins must be"):
        autoassociation(t, t, n_bins=1)
    with pytest.raises(DataValidationError, match="above the limit"):
        autoassociation(np.arange(300.0), np.arange(300.0), max_pairs=1000)


# ---------------------------------------------------------------------------
# 9. Orbital forcing
# ---------------------------------------------------------------------------


def test_orbital_periods_satisfy_the_harmonic_relationships() -> None:
    """Assert the relationships that must hold, not the numbers themselves.

    19.1 and 23.4 kyr are the present-day precession periods of Berger
    (1978); the table rounds them to 19 and 23. Two things follow and both
    are checkable without trusting any amplitude:

    * the ``~100 kyr`` precession beat is the frequency DIFFERENCE of the two
      precession terms, not their sum and not their arithmetic mean:
      ``1 / (1/19 - 1/23) = 106.5 kyr``, which is within 7 % of 100;
    * their frequency SUM is the semiannual cycle,
      ``1 / (1/19 + 1/23) = 10.4 kyr``, an order of magnitude away from
      every row in the table -- so if 100 kyr were ever built as a sum, the
      error is a factor of ten, and this assertion is what catches it.
    """
    table = {row["name"]: row for row in orbital_forcing()}
    p19 = table["precession_19"]["period_kyr"]
    p23 = table["precession_23"]["period_kyr"]
    p100 = table["precession_100"]["period_kyr"]

    # From the rounded table values: the ~100 kyr precession beat is the
    # frequency DIFFERENCE of the two precession terms, not their sum and
    # not their arithmetic mean. 1 / (1/19 - 1/23) = 109.25 kyr, which is
    # within 10 % of 100 -- the table rounds, so the tolerance is the
    # rounding, and it is checked explicitly below against the unrounded
    # Berger (1978) periods.
    beat = 1.0 / (1.0 / p19 - 1.0 / p23)
    assert beat == pytest.approx(p100, rel=0.10)
    # With the unrounded present-day periods the agreement is much tighter:
    # 1 / (1/19.1 - 1/23.4) = 103.94 kyr, 3.9 % from 100.
    assert pytest.approx(1.0 / (1.0 / 19.1 - 1.0 / 23.4), rel=0.04) == 100.0
    # Their frequency SUM is the semiannual cycle, 10.5 kyr, an order of
    # magnitude away from every row in the table -- so if 100 kyr were ever
    # built as a sum, the error is a factor of ten, and this is what catches
    # it.
    sum_period = 1.0 / (1.0 / p19 + 1.0 / p23)
    assert sum_period == pytest.approx(10.4, rel=0.05)
    # and it matches no band in the table, which is what makes the sum the
    # wrong construction for a 100 kyr cycle detectable at all
    assert min(abs(sum_period - row["period_kyr"]) for row in table.values()) > 5.0

    # The rounded periods must still bracket the Berger (1978) values.
    assert abs(p19 - 19.1) < 0.5
    assert abs(p23 - 23.4) < 0.5
    # Band ordering: both precession terms are shorter than obliquity, which
    # is shorter than the precession beat, which is shorter than the
    # long-period modulation.
    assert p19 < table["obliquity_41"]["period_kyr"] < p100 < table["long_period_precession"]["period_kyr"]
    assert p19 < p23 < p100
    assert set(table) == {
        "long_period_precession",
        "precession_100",
        "precession_70",
        "obliquity_41",
        "precession_23",
        "precession_19",
    }


def test_orbital_amplitudes_respect_the_defensible_orderings() -> None:
    """Only the orderings any defensible table must satisfy.

    The amplitudes are documented approximations, not the Laskar (2004)
    solution, so the test asserts only what follows from the physics rather
    than the numbers: the ~100 kyr band is the dominant eccentricity band in
    every published solution and must exceed the ~70 kyr sideband; obliquity
    and the short precession term are the same order of magnitude in the
    standard 65 N insolation forcings (both around 0.2 W/m^2), so they must
    agree within a factor of two; and the 400 kyr row is an amplitude
    *envelope*, marked ``role="modulation"``, because adding it as a plain
    sinusoid models something that is not there.
    """
    table = {row["name"]: row for row in orbital_forcing()}
    assert table["precession_100"]["relative_amplitude"] > table["precession_70"]["relative_amplitude"]
    obl = table["obliquity_41"]["relative_amplitude"]
    short = table["precession_23"]["relative_amplitude"]
    assert 0.5 < obl / short < 2.0
    assert all(row["relative_amplitude"] > 0 for row in table.values())
    assert all(np.isfinite(row["relative_amplitude"]) for row in table.values())
    assert table["long_period_precession"]["role"] == "modulation"
    assert all(row["role"] == "forcing" for name, row in table.items() if name != "long_period_precession")
    assert all(row["band"] == "precession" for name, row in table.items() if "precession" in name)
    assert table["obliquity_41"]["band"] == "obliquity"
    # The frequency column is the 1000x Ma/kyr conversion, not a copy.
    assert table["precession_100"]["frequency_per_ma"] == pytest.approx(10.0)
    assert table["obliquity_41"]["frequency_per_ma"] == pytest.approx(1000.0 / 41.0)


def test_orbital_lookup_and_rejection() -> None:
    """A single band can be fetched; an unknown period is refused, not
    silently rounded onto the nearest row."""
    rows = orbital_forcing(period_kyr=100.0)
    assert len(rows) == 1
    assert rows[0]["name"] == "precession_100"
    with pytest.raises(DataValidationError, match="within 1 %"):
        orbital_forcing(period_kyr=333.0)
    with pytest.raises(DataValidationError, match="n_periods must be"):
        orbital_forcing(n_periods=0)


def test_insolation_series_is_exactly_the_requested_cosine() -> None:
    """The synthetic target is cos(2 pi t * 1000 / P_kyr) and nothing else.

    Written out here rather than reused from the module, so the unit
    conversion between Ma ages and kyr periods is the thing under test: a
    dropped factor of 1000 turns 405 kyr into 405 Ma and still returns a
    plausible-looking cosine.
    """
    times_ma = np.array([0.0, 0.10125, 0.2025, 0.30375])
    got = insolation_series(times_ma, 405.0)
    expected = np.cos(2.0 * np.pi * times_ma * 1000.0 / 405.0)
    np.testing.assert_allclose(got, expected, atol=1e-15)
    assert got[0] == pytest.approx(1.0)
    # A quarter of a period on, the target is at zero: 0.10125 Ma is exactly
    # 101.25 kyr, a quarter of 405.
    assert got[1] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(DataValidationError, match="period_kyr must be"):
        insolation_series(times_ma, 0.0)
    with pytest.raises(DataValidationError, match="non-finite"):
        insolation_series(np.array([0.0, np.nan]), 405.0)


# ---------------------------------------------------------------------------
# Analyzer facade
# ---------------------------------------------------------------------------


def test_analyzer_delegates_and_remembers_the_last_result() -> None:
    """The facade must be a facade, not a second implementation."""
    analyzer = CycleAnalyzer()
    assert analyzer.last_result is None
    result = analyzer.runs([4, 3, 2, 5, 1, 0, -1, -2])
    assert analyzer.last_result is result
    assert result.runs == 2
    acf = analyzer.autocorrelation(np.arange(50.0), max_lag=5)
    assert analyzer.last_result is acf
    assert acf.nlags == 5
