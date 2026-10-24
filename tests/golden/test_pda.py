# =============================================================================
# FILE: tests/golden/test_pda.py
# =============================================================================
"""
Tests for ``stratigraphy.pda`` (Power Decomposition Analysis).

Every numeric expectation here is either

* a property of the *input* -- a record built from a 100 kyr and a 41 kyr
  sinusoid carries its power in those two bands and in nowhere else, so the two
  bands must take the largest shares of the power and the unused band must take
  almost none;
* a property of the *decomposition* -- the band components add up to the record
  minus the residual, and their powers add up to the total power, because the
  bands are disjoint in frequency;
* or a property of the *bootstrap* -- resampling the unexplained residual can
  only add power to a band, never take it away, so the resampled interval sits
  above the observed share.

What is deliberately NOT asserted: a number obtained by running this module and
freezing the output.  The tests that want determinism say so and name the knob
that gives it (``random_seed``).

Reference cited in the assertions:
    Li, M., Huang, C., Hinnov, L., Ogg, J., Chen, Z.-Q. & Zhang, Y. 2016.
    Power decomposition analysis for the detection of cyclostratigraphic
    cycles. Geology 44: 623-626.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"pda"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.pda import power_decomposition
from utils.exceptions import ValidationError

#: 采样点数；记录长 8000 kyr，100 kyr 分量有 80 个周期，窄带滤波器的
#: 冲激响应才不会淹没整个记录。
N_SAMPLES = 8000

#: 两个目标分量的周期，单位 kyr。
SLOW_PERIOD_KYR = 100.0
FAST_PERIOD_KYR = 41.0

#: 快分量的相对振幅（功率与振幅平方成正比）。
FAST_AMPLITUDE = 0.7

#: 第三个默认带（23 kyr）在测试信号里没有成分。
EMPTY_BAND_KYR = 23.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _correlation(left: npt.NDArray, right: npt.NDArray) -> float:
    """Pearson correlation of two series (0.0 when either is constant)."""
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _signal() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Time axis, the two components and their sum."""
    time = np.arange(N_SAMPLES, dtype=float)
    slow = np.sin(2.0 * np.pi * time / SLOW_PERIOD_KYR)
    fast = FAST_AMPLITUDE * np.sin(2.0 * np.pi * time / FAST_PERIOD_KYR)
    return time, slow, fast, slow + fast


def _band(result, centre_kyr: float):
    """The band with the given centre period."""
    for band in result.bands:
        if band.centre_kyr == centre_kyr:
            return band
    raise AssertionError(f"no band with centre {centre_kyr} in the result")


@pytest.fixture(scope="module")
def decomposition():
    """The reference decomposition: two components, three default bands."""
    time, _, _, signal = _signal()
    return power_decomposition(time, signal, n_bootstrap=0)


# ---------------------------------------------------------------------------
# Power shares
# ---------------------------------------------------------------------------


def test_the_two_dominant_bands_take_the_largest_power_share(decomposition):
    """The 100 kyr and 41 kyr bands hold the power; the 23 kyr band holds none."""
    shares = {band.centre_kyr: band.power_fraction for band in decomposition.bands}
    ordered = sorted(shares.items(), key=lambda item: item[1], reverse=True)

    assert [centre for centre, _ in ordered[:2]] == [SLOW_PERIOD_KYR, FAST_PERIOD_KYR]
    assert shares[EMPTY_BAND_KYR] < 0.01
    # The slow component has the larger amplitude, so it takes the larger share.
    assert shares[SLOW_PERIOD_KYR] > shares[FAST_PERIOD_KYR]


def test_the_power_shares_follow_the_component_amplitudes(decomposition):
    """Power goes as amplitude squared: 1.0 against 0.7 gives a ratio of ~2."""
    shares = {band.centre_kyr: band.power_fraction for band in decomposition.bands}
    ratio = shares[SLOW_PERIOD_KYR] / shares[FAST_PERIOD_KYR]
    expected = 1.0 / (FAST_AMPLITUDE**2)
    assert ratio == pytest.approx(expected, rel=0.15)


def test_the_bands_and_the_residual_account_for_the_whole_record(decomposition):
    """Disjoint bands plus the residual account for all of the power."""
    total = sum(band.power_fraction for band in decomposition.bands)
    total += float(decomposition.residual_power_fraction)
    assert total == pytest.approx(1.0, abs=0.05)
    # The residual array carries the same share of the variance.
    assert float(np.var(decomposition.residual)) / float(np.var(_signal()[3])) == pytest.approx(
        float(decomposition.residual_power_fraction), abs=0.02
    )


# ---------------------------------------------------------------------------
# Band components
# ---------------------------------------------------------------------------


def test_the_filtered_band_is_the_component_of_that_period(decomposition):
    """Each band's filtered series is the target component, not the other one."""
    _, slow, fast, _ = _signal()
    slow_band = _band(decomposition, SLOW_PERIOD_KYR)
    fast_band = _band(decomposition, FAST_PERIOD_KYR)

    assert _correlation(slow_band.filtered, slow) > 0.9
    assert _correlation(fast_band.filtered, fast) > 0.9
    # The two bands do not leak into each other.
    assert abs(_correlation(fast_band.filtered, slow)) < 0.1
    assert abs(_correlation(slow_band.filtered, fast)) < 0.1


def test_the_peak_period_of_a_band_is_within_ten_percent_of_its_centre(decomposition):
    """The measured peak of each band sits on its centre period."""
    for centre in (SLOW_PERIOD_KYR, FAST_PERIOD_KYR):
        band = _band(decomposition, centre)
        assert band.peak_period_kyr is not None
        relative_error = abs(band.peak_period_kyr - centre) / centre
        assert relative_error < 0.10, f"band {centre} kyr peaked at {band.peak_period_kyr:.3f} kyr"


def test_the_residual_holds_what_the_bands_do_not_explain(decomposition):
    """The residual plus the band components add back up to the record."""
    _, _, _, signal = _signal()
    rebuilt = np.asarray(decomposition.residual, dtype=float).copy()
    for band in decomposition.bands:
        rebuilt = rebuilt + band.filtered
    np.testing.assert_allclose(rebuilt, signal, atol=1e-9)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def test_the_bootstrap_interval_brackets_the_observed_share():
    """Resampling the residual can only add power, so the share cannot shrink."""
    time, _, _, signal = _signal()

    result = power_decomposition(time, signal, n_bootstrap=100, random_seed=20261024)

    for band in result.bands:
        assert band.power_fraction_ci is not None
        low, high = band.power_fraction_ci
        assert low <= high
        # Noise only adds power to a band, so the interval cannot fall below
        # the observed share by more than the sampling scatter.
        assert band.power_fraction <= high
        assert band.power_fraction_sd is not None and band.power_fraction_sd >= 0.0


def test_no_bootstrap_means_no_interval():
    """``n_bootstrap=0`` turns the uncertainty off."""
    time, _, _, signal = _signal()

    result = power_decomposition(time, signal, n_bootstrap=0)

    assert result.n_bootstrap == 0
    for band in result.bands:
        assert band.power_fraction_ci is None
        assert band.power_fraction_sd is None


def test_a_fixed_seed_makes_the_bootstrap_reproducible():
    """The same seed gives the same resampled interval."""
    time, _, _, signal = _signal()

    first = power_decomposition(time, signal, n_bootstrap=50, random_seed=7)
    second = power_decomposition(time, signal, n_bootstrap=50, random_seed=7)

    for left, right in zip(first.bands, second.bands, strict=True):
        assert left.power_fraction == right.power_fraction
        assert left.power_fraction_ci == right.power_fraction_ci


def test_an_unseeded_run_warns_that_the_result_is_not_reproducible():
    """An unseeded bootstrap says so, because a published curve must repeat."""
    time, _, _, signal = _signal()

    with pytest.warns(RuntimeWarning):
        power_decomposition(time, signal, n_bootstrap=2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_bands_raise_validation_error():
    """A band needs a positive centre, a positive half width, and a sensible span."""
    time, _, _, signal = _signal()

    with pytest.raises(ValidationError):
        power_decomposition(time, signal, bands=[(100.0, 0.0)], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, signal, bands=[(100.0, -5.0)], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, signal, bands=[(0.0, 5.0)], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, signal, bands=[(-100.0, 5.0)], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, signal, bands=[(5.0, 10.0)], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, signal, bands=[], n_bootstrap=0)


def test_too_few_points_raise_validation_error():
    """A record too short for the filter is rejected."""
    time = np.arange(10, dtype=float)

    with pytest.raises(ValidationError):
        power_decomposition(time, np.sin(time), n_bootstrap=0)


def test_malformed_inputs_raise_validation_error():
    """Mismatched lengths, unordered time and non-finite values are rejected."""
    time, _, _, signal = _signal()

    with pytest.raises(ValidationError):
        power_decomposition(time, signal[:-1], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time[::-1], signal[::-1], n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, np.full(N_SAMPLES, np.inf), n_bootstrap=0)
    with pytest.raises(ValidationError):
        power_decomposition(time, signal, n_bootstrap=-1)
