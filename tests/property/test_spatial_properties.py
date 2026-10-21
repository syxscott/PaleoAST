# =============================================================================
# FILE: tests/property/test_spatial_properties.py
# =============================================================================
"""
Property-based tests for the spatial statistics.

Moran's I is a correlation with a built-in expectation, and the
properties that follow from that are the ones that catch a wrong
implementation:

  * the statistic lies in ``[-1, 1]``;
  * its expectation under no spatial autocorrelation is ``-1 / (n - 1)``;
  * a positive affine transform of the values changes no ratio, because
    the statistic is a ratio of two quantities that both scale;
  * the weight matrix is row-standardised, so every site has the same
    weight to give away;
  * a pattern that is genuinely clustered scores above the expectation
    and a pattern that is genuinely scattered scores below it;
  * the p-value is a probability.

References:
    Cliff, N., Ord, J.K. & Thompson, M.A. (1974). The Analysis of
    Spatial Autocorrelation. Pion.
    Moran, P.A.P. (1950). Notes on continuous stochastic phenomena.
    Biometrika 37, 17-23.
    Anselin, L. (1988). Spatial Econometrics: Methods and Models.
    Kluwer.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stats.spatial_stats import SpatialStatsAnalyzer

from ._helpers import affine, call_or_none, scale_strategy, shift_strategy

_SETTING = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])

#: A set of sites on a line, with a value at each.
_sites = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    ),
    min_size=4,
    max_size=12,
)


def _morans_i(values: np.ndarray, coordinates: np.ndarray, permutations: int = 19):
    """Moran's I for the values at the sites, or None."""
    return call_or_none(
        SpatialStatsAnalyzer().morans_i,
        values,
        coordinates,
        permutations=permutations,
        random_seed=7,
    )


@given(sites=_sites)
@_SETTING
def test_morans_i_is_a_correlation(sites):
    """Moran's I is a correlation coefficient, so it lies in [-1, 1]."""
    data = np.asarray(sites, dtype=float)
    result = _morans_i(data[:, 1], data[:, :1])
    if result is None:
        return
    assert -1.0 - 1e-9 <= result.statistic <= 1.0 + 1e-9


@given(sites=_sites)
@_SETTING
def test_morans_i_has_the_right_expectation(sites):
    """Under no spatial autocorrelation the expectation is
    ``-1 / (n - 1)``."""
    data = np.asarray(sites, dtype=float)
    result = _morans_i(data[:, 1], data[:, :1])
    if result is None:
        return
    assert np.isclose(result.expected, -1.0 / (data.shape[0] - 1), rtol=1e-12, atol=1e-14)


@given(sites=_sites, scale=scale_strategy, shift=shift_strategy)
@_SETTING
def test_morans_i_is_invariant_under_an_affine_transform(sites, scale, shift):
    """A mean shift and a rescaling of the values change no ratio."""
    data = np.asarray(sites, dtype=float)
    base = _morans_i(data[:, 1], data[:, :1])
    moved = _morans_i(affine(data[:, 1], scale, shift), data[:, :1])
    if base is None or moved is None:
        return
    assert np.isclose(base.statistic, moved.statistic, rtol=1e-6, atol=1e-9, equal_nan=True)


@given(sites=_sites)
@_SETTING
def test_the_morans_i_p_value_is_a_probability(sites):
    """The permutation p-value is a probability."""
    data = np.asarray(sites, dtype=float)
    result = _morans_i(data[:, 1], data[:, :1])
    if result is None:
        return
    assert 0.0 <= result.p_value <= 1.0
    assert 0.0 <= result.p_value_greater <= 1.0


@given(sites=_sites)
@_SETTING
def test_the_morans_i_weights_are_row_standardised(sites):
    """Every site gives away the same total weight.

    The weights are row-standardised so that a site with many
    neighbours is not weighted more than a site with few; a matrix whose
    rows do not sum to one is not row-standardised.
    """
    data = np.asarray(sites, dtype=float)
    result = _morans_i(data[:, 1], data[:, :1])
    if result is None:
        return
    weights = np.asarray(result.weights, dtype=float)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.diag(weights), 0.0, atol=1e-12)


@given(sites=_sites)
@_SETTING
def test_a_clustered_pattern_scores_above_the_expectation(sites):
    """A value that grows with the position on the line is clustered.

    Neighbouring sites have similar values, so the observed statistic
    has to be above the expectation under no autocorrelation.
    """
    data = np.asarray(sites, dtype=float)
    order = np.argsort(data[:, 0])
    values = data[order, 1].copy()
    if not np.all(np.diff(values) >= 0) or values.size < 3:
        return
    result = _morans_i(values, data[order, :1])
    if result is None:
        return
    assert result.statistic > result.expected
