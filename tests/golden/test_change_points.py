# =============================================================================
# FILE: tests/golden/test_change_points.py
# =============================================================================
"""
Tests for the Bayesian change-point detector in ``stratigraphy.change_points``.

The detector is a usable implementation of the Bayesian change-point idea of
Ruggieri (2013): a piecewise-constant model, compared position by position with
a local Bayes factor for a mean shift, and turned into a posterior probability
with a prior on the number of change points.

Every numeric expectation here is either

* a property of the *input* -- a step inserted into a constant series is at a
  known sample, and a record without a step has none to find; or
* a property of the *estimator* -- with the same seed the Monte-Carlo estimate
  is the same number twice, and without a seed the module warns that the result
  is not reproducible (the repository-wide convention).

What is deliberately NOT asserted: a number obtained by running this module and
freezing the output.  The reproducibility test says which knob gives it.

References cited in the assertions:
    Ruggieri, E. 2013. A Bayesian approach to detecting change points in
    climatic records. International Journal of Climatology 33: 520-528.
    Rubin, D.B. 1981. The Bayesian bootstrap. Annals of Statistics 9: 130-134.
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"change_points"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.change_points import detect_change_points
from utils.exceptions import DataValidationError

STEP_INDEX = 150
SEGMENT = 150


def _step_series(levels: list[float], noise: float = 0.2, seed: int = 11) -> npt.NDArray:
    """A piecewise-constant series with one level per segment."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(loc=level, scale=noise, size=SEGMENT) for level in levels])


def test_detects_a_single_step():
    """The step in the middle of a constant series must be found, within two samples."""
    values = _step_series([0.0, 5.0])

    result = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=7)

    assert result.positions.size == 1
    assert abs(int(result.positions[0]) - STEP_INDEX) <= 2
    assert result.probabilities[0] >= 0.5
    assert result.probability_curve.size == values.size
    # The curve is zero outside the candidate window. A candidate needs a full
    # block of min_segment samples on each side, so the window is
    # [min_segment, n - min_segment] with both ends included.
    window_start, window_stop = 10, values.size - 10
    assert result.probability_curve[:window_start].max() == 0.0
    assert result.probability_curve[window_stop + 1 :].max() == 0.0


def test_stationary_series_reports_no_change_point():
    """A record without a step must not produce one."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=400)

    result = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=5)

    assert result.positions.size == 0
    assert result.probability_curve.max() < 0.5


def test_multiple_steps_are_all_detected():
    """Four levels give three change points, and every one of them is found."""
    values = _step_series([0.0, 3.0, 1.0, 4.0], seed=21)

    result = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=13)

    assert result.positions.size == 3
    for expected in (STEP_INDEX, 2 * SEGMENT, 3 * SEGMENT):
        assert np.min(np.abs(result.positions - expected)) <= 2


def test_default_threshold_finds_a_step_and_ignores_a_stationary_record():
    """At the default threshold a strong step is found and a flat record is not.

    The per-position posterior is a local quantity and there are O(n) candidate
    positions, so the default threshold has to sit above the multiplicity band.
    This locks that in over several seeds rather than on a single record.
    """
    stepped = _step_series([0.0, 5.0])

    found = detect_change_points(stepped, min_segment=10, random_seed=7)

    assert found.positions.size == 1
    assert abs(int(found.positions[0]) - STEP_INDEX) <= 2

    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        stationary = rng.normal(size=400)

        result = detect_change_points(stationary, min_segment=10, random_seed=seed)

        assert result.positions.size == 0, f"seed {seed} reported {result.positions}"


def test_fixed_seed_is_reproducible():
    """The same seed gives the same numbers; a different seed does not."""
    values = _step_series([0.0, 5.0])

    first = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=99)
    second = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=99)
    other = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=100)

    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_allclose(first.probability_curve, second.probability_curve)
    assert not np.array_equal(first.probability_curve, other.probability_curve)


def test_unseeded_run_warns_about_reproducibility():
    """The repository convention: no seed means no reproducibility, and we say so."""
    values = _step_series([0.0, 5.0])

    with pytest.warns(RuntimeWarning):
        detect_change_points(values, min_segment=10)


def test_invalid_inputs_raise_validation_error():
    """Bad lengths, block sizes and thresholds raise DataValidationError."""
    with pytest.raises(DataValidationError):
        detect_change_points(np.ones(5), min_segment=10, random_seed=1)
    with pytest.raises(DataValidationError):
        detect_change_points(np.ones(100), min_segment=1, random_seed=1)
    with pytest.raises(DataValidationError):
        detect_change_points(np.ones(100), threshold=0.0, random_seed=1)
    with pytest.raises(DataValidationError):
        detect_change_points(np.ones(100), threshold=1.0, random_seed=1)
    with pytest.raises(DataValidationError):
        detect_change_points(np.ones(100), threshold=1.5, random_seed=1)


def test_constant_record_has_no_change_point():
    """A record without any variation is a legal record, not a bad input.

    Every candidate has a zero mean shift, so the posterior probability is zero
    everywhere and nothing is reported -- which is different from an error.
    """
    result = detect_change_points(np.ones(100), min_segment=10, threshold=0.5, random_seed=1)

    assert result.positions.size == 0
    assert result.probability_curve.max() == 0.0


def test_summary_reports_the_detected_positions():
    """The summary is part of the contract: it names the positions and probabilities."""
    values = _step_series([0.0, 5.0])

    result = detect_change_points(values, min_segment=10, threshold=0.5, random_seed=7)

    text = result.summary()
    assert "change-point" in text
    assert str(int(result.positions[0])) in text
