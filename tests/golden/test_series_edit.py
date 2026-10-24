# =============================================================================
# FILE: tests/golden/test_series_edit.py
# =============================================================================
"""
Tests for ``stratigraphy.series_edit`` (remove_part / remove_range).

Every numeric expectation here is either

* a property of the *definition* -- a closed interval ``[start, end]`` keeps
  or drops the samples whose **time** falls inside it, endpoints included,
  and the two returned arrays stay aligned with each other; or
* a property of the *boundary contract* -- a reversed interval, or one that
  lies outside the record's time span, is a mistake in the caller's
  coordinates and is rejected with ``ValidationError`` rather than silently
  removing nothing.

References cited in the assertions:
    Li, M. et al. 2018. Earth Planet. Sci. Lett. 501: 165-179. (The Acycle
    "series data engineering" workflow the two helpers belong to.)
"""

from __future__ import annotations

#: The analyses this file is evidence for.  The matrix reads this
#: declaration as authoritative: the file covers these and nothing
#: else.  Without it the fallback -- matching the function name AND
#: the module name in the text -- misses several of them.
ANALYSES_COVERED = {"remove_part", "remove_range"}


import numpy as np
import numpy.typing as npt
import pytest

from stratigraphy.series_edit import remove_part, remove_range
from utils.exceptions import ValidationError

# =============================================================================
# 测试常量
# =============================================================================

#: 合成记录的长度。
N_SAMPLES = 20

#: 合成记录的时间跨度。
TIME_SPAN = 19.0


# =============================================================================
# Helpers
# =============================================================================


def _series() -> tuple[npt.NDArray, npt.NDArray]:
    """A record with one sample per integer time step."""
    times = np.arange(N_SAMPLES, dtype=float)
    values = times * 2.0 + 1.0
    return times, values


# =============================================================================
# remove_part
# =============================================================================


def test_remove_part_drops_the_samples_inside_the_interval():
    """The samples inside the interval are gone, the rest are untouched."""
    times, values = _series()

    kept_times, kept_values = remove_part(times, values, (4.0, 7.0))

    assert kept_times.size == N_SAMPLES - 4
    assert kept_values.size == N_SAMPLES - 4
    np.testing.assert_array_equal(kept_times, np.delete(times, [4, 5, 6, 7]))
    np.testing.assert_array_equal(kept_values, np.delete(values, [4, 5, 6, 7]))


def test_remove_part_uses_a_closed_interval():
    """Both end points of the interval are removed."""
    times, values = _series()

    kept_times, _kept_values = remove_part(times, values, (3.0, 5.0))

    assert 3.0 not in kept_times
    assert 5.0 not in kept_times
    assert 2.0 in kept_times
    assert 6.0 in kept_times


def test_remove_part_keeps_the_original_order():
    """The remaining samples stay in time order."""
    times, values = _series()

    kept_times, kept_values = remove_part(times, values, (8.0, 11.0))

    assert np.all(np.diff(kept_times) > 0.0)
    np.testing.assert_array_equal(kept_values, 2.0 * kept_times + 1.0)


def test_remove_part_of_the_whole_record_leaves_nothing():
    """Removing everything is allowed; the result is empty, not an error."""
    times, values = _series()

    kept_times, kept_values = remove_part(times, values, (0.0, TIME_SPAN))

    assert kept_times.size == 0
    assert kept_values.size == 0


# =============================================================================
# remove_range
# =============================================================================


def test_remove_range_keeps_only_the_samples_inside_the_interval():
    """Only the samples inside the interval survive."""
    times, values = _series()

    kept_times, kept_values = remove_range(times, values, (4.0, 7.0))

    np.testing.assert_array_equal(kept_times, np.array([4.0, 5.0, 6.0, 7.0]))
    np.testing.assert_array_equal(kept_values, 2.0 * kept_times + 1.0)


def test_remove_range_uses_a_closed_interval():
    """Both end points of the interval are kept."""
    times, values = _series()

    kept_times, _kept_values = remove_range(times, values, (3.0, 5.0))

    np.testing.assert_array_equal(kept_times, np.array([3.0, 4.0, 5.0]))


def test_remove_range_and_remove_part_are_complementary():
    """What one helper drops, the other keeps."""
    times, values = _series()

    kept_times, _kept_values = remove_range(times, values, (4.0, 7.0))
    rest_times, _rest_values = remove_part(times, values, (4.0, 7.0))

    assert kept_times.size + rest_times.size == N_SAMPLES
    np.testing.assert_array_equal(np.sort(np.concatenate([kept_times, rest_times])), times)


def test_remove_range_keeps_the_two_arrays_aligned():
    """The returned value array still belongs to the returned time array."""
    times, values = _series()

    kept_times, kept_values = remove_range(times, values, (2.0, 6.0))

    assert kept_times.size == kept_values.size
    np.testing.assert_array_equal(kept_values, 2.0 * kept_times + 1.0)


# =============================================================================
# Validation
# =============================================================================


def test_a_reversed_interval_raises_validation_error():
    """An interval that starts after it ends is a mistake, not a no-op."""
    times, values = _series()

    with pytest.raises(ValidationError):
        remove_part(times, values, (7.0, 4.0))
    with pytest.raises(ValidationError):
        remove_range(times, values, (7.0, 4.0))


def test_an_interval_outside_the_record_raises_validation_error():
    """An interval outside the record's time span is rejected."""
    times, values = _series()

    with pytest.raises(ValidationError):
        remove_part(times, values, (100.0, 200.0))
    with pytest.raises(ValidationError):
        remove_range(times, values, (-50.0, -10.0))


def test_malformed_inputs_raise_validation_error():
    """Mismatched lengths and non-finite time axes are rejected.

    ``ValidationError`` is the backwards-compatible alias of
    ``DataValidationError`` in ``utils.exceptions``.
    """
    times, values = _series()

    with pytest.raises(ValidationError):
        remove_part(times, values[:-1], (1.0, 2.0))
    with pytest.raises(ValidationError):
        remove_range(times, values[:-1], (1.0, 2.0))
    with pytest.raises(ValidationError):
        remove_part(np.array([0.0, 1.0, np.nan, 3.0]), values[:4], (1.0, 2.0))
    with pytest.raises(ValidationError):
        remove_range(times, values, (1.0, np.inf))
    with pytest.raises(ValidationError):
        remove_range(np.array([]), np.array([]), (1.0, 2.0))
