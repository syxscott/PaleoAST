# =============================================================================
# FILE: tests/property/_helpers.py
# =============================================================================
"""
Shared pieces for the property-based tests.

Two things live here, and both are about not testing the wrong thing.

**Stepping over the inputs an analysis refuses.**  The application
refuses degenerate input -- a group with one observation, a sample with
no variation, a table with an empty cell -- and refusing it is correct
behaviour, not a property violation.  A property test has to step over
those inputs, because the claim being checked is about the inputs the
analysis *accepts*.  :func:`call_or_none` does that in one place, so
each test reads as the claim it is making rather than as a wall of
try/except.

**The transforms the properties are stated in.**  A positive affine
transform ``y -> a*y + b`` (``a > 0``) and a strictly increasing
transform are the two transformations under which a statistical test
must keep its conclusions.  Keeping them in one place means every file
uses the same ones, and the tolerances that go with them.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from utils.exceptions import PaleoASTError

#: A sample of finite floats.
values_strategy = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=40,
)

#: A sample of INTEGERS, for the tests that apply an affine transform.
#:
#: Integers, and not floats, because an affine transform of floats can
#: collapse two distinct values onto the same double -- ``x * 0.25 + 100``
#: loses the low bits of a large ``x`` -- and a collapsed pair is a tie
#: that the original sample did not have.  Two distinct integers are at
#: least 1 apart, and a scale of at least a quarter keeps them that far
#: apart after the transform, so the transform stays injective.
integer_strategy = st.lists(
    st.integers(min_value=-100, max_value=100),
    min_size=4,
    max_size=40,
)

#: A sample with enough observations for a two-group design.
two_group_strategy = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=4,
    max_size=40,
)

#: A sample with an odd number of observations, so the median is a single
#: observation and a strictly increasing transform carries it along.
odd_strategy = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=39,
).filter(lambda v: len(v) % 2 == 1)

#: A strictly positive scale, and any shift.
#:
#: Both are chosen so that the affine transform is EXACT in floating
#: point: the scale is a power of two, so multiplying by it only moves
#: the exponent, and the shift is an integer, so adding it to an
#: integer leaves the low bits alone.  That matters more than it
#: sounds: the transform of two tied values must leave them tied, and an
#: inexact transform can separate them by an ulp -- which is a new tie
#: structure, and therefore new ranks, and therefore a different
#: statistic for a reason that has nothing to do with the test.
scale_strategy = st.sampled_from([0.25, 0.5, 1.0, 2.0, 4.0])
shift_strategy = st.integers(min_value=-100, max_value=100)


def call_or_none(function, *args, **kwargs):
    """Run an analysis, or return ``None`` if the input is outside its domain.

    ``None`` means "this input is not one the analysis answers", and the
    caller steps over it.  It does NOT mean the analysis failed: a
    failure that is not a refusal of the input is a defect, and it is
    raised, not swallowed.
    """
    try:
        return function(*args, **kwargs)
    except PaleoASTError:
        return None
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return None


def as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def split_in_two(values) -> tuple[np.ndarray, list[int]]:
    """Split a sample into two groups of at least two observations.

    Returns ``(values, labels)``.  The split is in the middle, so the
    two groups have nearly the same size, which keeps every test in the
    group well populated.
    """
    array = as_array(values)
    half = array.size // 2
    return array, [0] * half + [1] * (array.size - half)


def two_samples(values) -> tuple[np.ndarray, np.ndarray]:
    """Two samples of the SAME length, as the two columns of one table.

    The application takes the two samples of a two-sample test as the
    two columns of one matrix, so the samples have to be the same length
    for the test to be the one the interface can express.  An odd
    observation is dropped rather than kept in one of the two.
    """
    array = as_array(values)
    half = array.size // 2
    return array[:half], array[half : 2 * half]


def affine(values: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """``y -> a*y + b`` with ``a > 0``: a mean shift and a rescaling."""
    return values * scale + shift


def monotone(values: np.ndarray, normaliser: float | None = None) -> np.ndarray:
    """A strictly increasing transform of the data.

    The transform is built from the RANKS: the sample is replaced by the
    normal scores of its ranks, which is strictly increasing in the rank
    and gives tied observations the same score.  That is the strongest
    form of the claim a rank-based test has to satisfy -- the
    transformed sample has the same order and the same ties as the
    original and nothing else -- and it is exact in floating point.

    A transform applied to the *values* (``tanh``, say) cannot be used
    here: it saturates, and two distinct values at the far end of the
    sample become the same double.  That is a tie the original sample
    did not have, and it changes the ranks, so it would fail a test
    about rank invariance for a reason that has nothing to do with the
    test.

    The ``normaliser`` parameter is kept for the one place a scalar has
    to be transformed with the same map as the sample it came from.
    """
    from scipy import stats

    ranks = stats.rankdata(values)
    normal_scores = stats.norm.ppf((ranks - 0.5) / ranks.size)
    if normaliser is not None:
        # The caller wants the transform of one more value with the
        # sample's own scale; the rank of that value is not defined, so
        # the caller is expected to pass the transformed value instead.
        raise ValueError("the rank-based transform has no scalar form")
    return normal_scores


def monotone_normaliser(values: np.ndarray) -> float:
    """Kept for callers that only need the scale of the sample."""
    return max(1.0, float(np.max(np.abs(values))))


def has_variation(values: np.ndarray) -> bool:
    """Whether the sample carries any variation at all."""
    return values.size >= 2 and not np.allclose(values.std(ddof=1), 0.0)
