# =============================================================================
# FILE: tests/golden/test_emd.py
# =============================================================================
"""
Tests for ``stratigraphy.emd`` (EMD and EEMD).

Every numeric expectation here is either

* a property of the *algorithm* stated by the papers the module implements --
  Huang et al. (1998) require an IMF to have as many zero crossings as extrema
  (up to one), and the sifting must stop on the von Neumann criterion
  ``SD = sum(h_prev * h) / sum(h_prev ** 2)``;
* a property of the *input* -- a sum of two sinusoids has exactly those two
  components in it, and a linear trend is the part an EMD leaves behind as the
  residual;
* or a property of the *ensemble average* -- EEMD's white-noise trials perturb
  each IMF at random, so the ensemble mean of a component is the component
  again (Wu & Huang 2009).

What is deliberately NOT asserted: a number obtained by running this module and
freezing the output.  The tests that want determinism say so and name the knob
that gives it (``random_seed``).

References cited in the assertions:
    Huang, N.E. et al. 1998. Proc. R. Soc. Lond. A 454: 903-995.
    Wu, Z. & Huang, N.E. 2009. Adv. Data Sci. Eng. 1: 1-41.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"eemd", "emd"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.emd import eemd, emd
from utils.exceptions import ValidationError

#: 采样点数。两个分量分别有 32 个和 109 个周期，足以让 EMD 把它们分开。
N_SAMPLES = 1200

#: 两个正弦分量的周期（样本数）。
SLOW_PERIOD = 37.0
FAST_PERIOD = 11.0

#: 快分量的相对振幅。
FAST_AMPLITUDE = 0.8

#: 线性趋势的斜率（每样本）。
TREND_SLOPE = 0.02


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


def _best_match(imfs: npt.NDArray, component: npt.NDArray) -> tuple[int, float]:
    """The IMF that matches ``component`` best.

    EMD orders its IMFs by frequency, but EEMD's ensemble mean is indexed by
    the IMF *slot*, and either way the slot a component lands in is not part of
    the contract.  The test therefore asks "is this component in there at all?",
    not "is it in row k?".

    Returns:
        ``(row_index, correlation)`` of the best matching IMF.
    """
    scores = [_correlation(imf, component) for imf in imfs]
    best = int(np.argmax(scores))
    return best, scores[best]


def _extrema_and_zero_crossings(values: npt.NDArray) -> tuple[int, int]:
    """Count interior extrema and zero crossings of a series.

    End points are not extrema, exactly as in the Huang et al. (1998) IMF
    definition; a sample that sits exactly on zero is skipped rather than
    counted as two crossings.
    """
    array = np.asarray(values, dtype=float)
    differences = np.diff(array)
    maxima = int(np.count_nonzero((differences[:-1] > 0.0) & (differences[1:] <= 0.0)))
    minima = int(np.count_nonzero((differences[:-1] < 0.0) & (differences[1:] >= 0.0)))
    nonzero = array[array != 0.0]
    crossings = int(np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1]))) if nonzero.size >= 2 else 0
    return maxima + minima, crossings


def _components() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The two sinusoidal components and their sum."""
    index = np.arange(N_SAMPLES, dtype=float)
    slow = np.sin(2.0 * np.pi * index / SLOW_PERIOD)
    fast = FAST_AMPLITUDE * np.sin(2.0 * np.pi * index / FAST_PERIOD)
    return slow, fast, slow + fast


# ---------------------------------------------------------------------------
# EMD
# ---------------------------------------------------------------------------


def test_emd_separates_two_sine_components():
    """The two sinusoids of the input come back as two separate IMFs."""
    slow, fast, signal = _components()

    result = emd(signal)

    assert result.imfs.shape[0] >= 2
    slow_row, slow_score = _best_match(result.imfs, slow)
    fast_row, fast_score = _best_match(result.imfs, fast)
    assert slow_score > 0.9, f"slow component not recovered (correlation {slow_score:.4f})"
    assert fast_score > 0.9, f"fast component not recovered (correlation {fast_score:.4f})"
    assert slow_row != fast_row, "both components matched the same IMF"


def test_emd_leaves_the_linear_trend_as_the_residual():
    """The residual of a trending signal is the trend."""
    _, _, signal = _components()
    index = np.arange(N_SAMPLES, dtype=float)
    trend = TREND_SLOPE * index

    result = emd(signal + trend)

    assert _correlation(result.trend, trend) > 0.99
    # The trend is a straight line, so its slope is recovered exactly.
    recovered = np.polyfit(index, result.trend, 1)[0]
    assert recovered == pytest.approx(TREND_SLOPE, rel=0.05)


def test_emd_reconstructs_the_input_from_imfs_and_residual():
    """``values == sum(imfs) + trend`` -- the decomposition loses nothing."""
    _, _, signal = _components()

    result = emd(signal)

    reconstructed = result.trend + result.imfs.sum(axis=0)
    np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


def test_every_imf_satisfies_the_huang_conditions():
    """Each IMF has as many zero crossings as extrema (up to one)."""
    _, _, signal = _components()

    result = emd(signal)

    assert result.imfs.shape[0] >= 2
    for row, imf in enumerate(result.imfs):
        n_extrema, n_crossings = _extrema_and_zero_crossings(imf)
        assert abs(n_extrema - n_crossings) <= 1, f"IMF {row}: {n_extrema} extrema vs {n_crossings} zero crossings"


def test_emd_reports_the_sifting_iterations():
    """``n_sift`` counts the sifting passes; a decomposition cannot take none."""
    _, _, signal = _components()

    result = emd(signal)

    assert result.n_sift > 0
    assert result.summary()


# ---------------------------------------------------------------------------
# EEMD
# ---------------------------------------------------------------------------


def test_eemd_recovers_the_components_of_a_noisy_record():
    """White-noise trials do not stop EEMD from finding the components."""
    slow, fast, signal = _components()
    rng = np.random.default_rng(20261024)
    noisy = signal + rng.normal(scale=0.15 * float(np.std(signal)), size=N_SAMPLES)

    result = eemd(noisy, n_trials=15, random_seed=42)

    _, slow_score = _best_match(result.imfs, slow)
    _, fast_score = _best_match(result.imfs, fast)
    assert slow_score > 0.9, f"slow component not recovered (correlation {slow_score:.4f})"
    assert fast_score > 0.9, f"fast component not recovered (correlation {fast_score:.4f})"


def test_eemd_is_reproducible_for_a_fixed_seed():
    """The same seed gives the same ensemble average, bit for bit."""
    _, _, signal = _components()
    rng = np.random.default_rng(20261024)
    noisy = signal + rng.normal(scale=0.15 * float(np.std(signal)), size=N_SAMPLES)

    first = eemd(noisy, n_trials=15, random_seed=42)
    second = eemd(noisy, n_trials=15, random_seed=42)

    np.testing.assert_array_equal(first.imfs, second.imfs)
    np.testing.assert_array_equal(first.trend, second.trend)


def test_eemd_without_a_seed_warns_that_the_result_is_not_reproducible():
    """An unseeded ensemble says so, because a published curve must repeat."""
    _, _, signal = _components()

    with pytest.warns(RuntimeWarning):
        eemd(signal, n_trials=2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_inputs_raise_validation_error():
    """Constant series and series that are too short are rejected.

    ``ValidationError`` is the backwards-compatible alias of
    ``DataValidationError`` in ``utils.exceptions``.
    """
    with pytest.raises(ValidationError):
        emd(np.ones(N_SAMPLES))
    with pytest.raises(ValidationError):
        emd(np.arange(4, dtype=float))
    with pytest.raises(ValidationError):
        eemd(np.ones(N_SAMPLES))
    with pytest.raises(ValidationError):
        eemd(np.arange(4, dtype=float))
    with pytest.raises(ValidationError):
        emd(np.full(N_SAMPLES, np.nan))
    with pytest.raises(ValidationError):
        emd(np.ones((10, 2)))
    with pytest.raises(ValidationError):
        emd(np.sin(np.arange(N_SAMPLES, dtype=float)), max_imfs=0)
    with pytest.raises(ValidationError):
        eemd(np.sin(np.arange(N_SAMPLES, dtype=float)), n_trials=0)
