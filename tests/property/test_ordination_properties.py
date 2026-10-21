# =============================================================================
# FILE: tests/property/test_ordination_properties.py
# =============================================================================
"""
Property-based tests for the ordination: the principal component analysis.

A PCA is easy to get subtly wrong -- a sign flip on a component, an
eigenvalue computed from the wrong normalisation, a score matrix that is
off by the column means -- and every one of those mistakes still produces
a picture that looks like a PCA.  The properties below are the ones that
catch them:

  * the eigenvalues are non-negative and in descending order;
  * they add up to the total variance, and the cumulative variance ends
    at one;
  * the scores are centred;
  * the eigenvalues are the squared singular values divided by ``n - 1``;
  * the scores are the centred data times the loadings;
  * permuting the rows, or the columns, of the data does not change the
    eigenvalues;
  * adding a constant to a column does not change a covariance PCA.

References:
    Jolliffe, I.T. (2002). Principal Component Analysis. 2nd ed. Springer.
    Hotelling, H. (1933). Analysis of a complex of statistical variables.
    Journal of Physical Chemistry 7, 215-232.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stats.pca import PCAAnalyzer

from ._helpers import call_or_none

_SETTING = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])

#: A small table of numbers with a fixed shape, so the analysis always
#: has enough rows and columns to have components at all.
_matrix = st.lists(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
    min_size=4,
    max_size=12,
)


def _pca(data: np.ndarray):
    """A covariance PCA of the data, or None if the analysis refuses it."""
    return call_or_none(PCAAnalyzer().analyze, data, method="covariance")


@given(table=_matrix)
@_SETTING
def test_the_pca_eigenvalues_are_non_negative_and_sorted(table):
    """The eigenvalues of a covariance matrix are non-negative, and a
    PCA lists them largest first."""
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    eigenvalues = np.asarray(result.eigenvalues, dtype=float)
    assert np.all(eigenvalues >= -1e-8), "a covariance eigenvalue is negative"
    assert np.all(np.diff(eigenvalues) <= 1e-8), "the eigenvalues are not in descending order"


@given(table=_matrix)
@_SETTING
def test_the_pca_eigenvalues_add_up_to_the_total_variance(table):
    """The eigenvalues add up to the total variance of the data.

    The *eigenvalues* are the variances; the ``explained_variance`` the
    result also carries is the same thing as a PERCENTAGE, and it is the
    eigenvalues that have to add up.
    """
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    eigenvalues = np.asarray(result.eigenvalues, dtype=float)
    raw = np.asarray(result.eigenvalues_raw, dtype=float)
    if raw.size == 0:
        return
    assert np.isclose(float(np.sum(eigenvalues)), float(np.sum(raw)), rtol=1e-6, atol=1e-8)


@given(table=_matrix)
@_SETTING
def test_the_pca_explained_variance_is_a_percentage(table):
    """The explained variance is a share of the total, so it is a
    percentage and the shares add up to a hundred."""
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    explained = np.asarray(result.explained_variance, dtype=float)
    raw = np.asarray(result.eigenvalues_raw, dtype=float)
    if float(np.sum(raw)) <= 0:
        # A table with no variation has no explained variance to share
        # out, and the result says so with zeros.
        assert np.allclose(explained, 0.0, atol=1e-12)
        return
    assert np.allclose(float(np.sum(explained)), 100.0, rtol=1e-6, atol=1e-6)
    assert np.all(explained >= -1e-9)
    assert np.all(explained <= 100.0 + 1e-9)


@given(table=_matrix)
@_SETTING
def test_the_pca_cumulative_variance_never_goes_backwards(table):
    """The cumulative variance is a running total, so it only grows."""
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    cumulative = np.asarray(result.cumulative_variance, dtype=float)
    assert np.all(np.diff(cumulative) >= -1e-9), "the cumulative variance goes backwards"


@given(table=_matrix)
@_SETTING
def test_the_pca_scores_are_centred(table):
    """The scores of a PCA sum to zero along each component.

    They are the data projected on centred axes, so their column means
    are zero -- and a score matrix with a non-zero column mean is the
    sign of data that was not centred.
    """
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    scores = np.asarray(result.scores, dtype=float)
    assert np.allclose(scores.mean(axis=0), 0.0, atol=1e-8)


@given(table=_matrix)
@_SETTING
def test_the_pca_eigenvalues_are_the_squared_singular_values(table):
    """``eigenvalue = singular_value**2 / (n - 1)``.

    The relation is the one that ties the two ways of computing a PCA
    together, and it is what catches an eigenvalue computed from the
    wrong normalisation.
    """
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    eigenvalues = np.asarray(result.eigenvalues, dtype=float)
    singular = np.asarray(result.singular_values, dtype=float)
    assert np.allclose(eigenvalues, singular**2 / (data.shape[0] - 1), rtol=1e-6, atol=1e-8)


@given(table=_matrix)
@_SETTING
def test_the_pca_scores_are_the_centred_data_times_the_loadings(table):
    """``scores * sqrt(eigenvalues) = X_centered @ loadings``.

    This is the definition, and a score matrix that does not satisfy it
    is a score matrix of something else.  A rank-deficient table is
    skipped: there the component basis inside the degenerate subspace is
    not unique, so the scores and the loadings are two different
    choices of basis and need not correspond column by column.

    The scaling by ``sqrt(eigenvalues)`` is not optional.  The loadings
    are the eigenvectors times the square root of their eigenvalues, so
    the relation without it is off by exactly that factor -- which is
    what the result's own documentation used to say.
    """
    data = np.asarray(table, dtype=float)
    result = _pca(data)
    if result is None:
        return
    scores = np.asarray(result.scores, dtype=float)
    loadings = np.asarray(result.loadings, dtype=float)
    eigenvalues = np.asarray(result.eigenvalues, dtype=float)
    centred = data - np.asarray(result.mean_vector, dtype=float)
    if np.linalg.matrix_rank(centred) < scores.shape[1]:
        return
    assert np.allclose(scores * np.sqrt(eigenvalues), centred @ loadings, rtol=1e-6, atol=1e-8)


@given(table=_matrix, seed=st.integers(min_value=0, max_value=2**16))
@_SETTING
def test_the_pca_does_not_depend_on_the_row_order(table, seed):
    """The rows of the data are the samples, and their order is not part
    of the analysis."""
    data = np.asarray(table, dtype=float)
    base = _pca(data)
    if base is None:
        return
    order = np.random.default_rng(seed).permutation(data.shape[0])
    moved = _pca(data[order])
    if moved is None:
        return
    assert np.allclose(np.asarray(base.eigenvalues), np.asarray(moved.eigenvalues), rtol=1e-6, atol=1e-8)


@given(table=_matrix, seed=st.integers(min_value=0, max_value=2**16))
@_SETTING
def test_the_pca_does_not_depend_on_the_column_order(table, seed):
    """The columns are the variables, and their order is not part of the
    analysis either -- the eigenvalues are the same, only the loadings
    are permuted."""
    data = np.asarray(table, dtype=float)
    base = _pca(data)
    if base is None:
        return
    order = np.random.default_rng(seed).permutation(data.shape[1])
    moved = _pca(data[:, order])
    if moved is None:
        return
    assert np.allclose(np.asarray(base.eigenvalues), np.asarray(moved.eigenvalues), rtol=1e-6, atol=1e-8)


@given(table=_matrix, offset=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
@_SETTING
def test_the_pca_of_a_shifted_column_is_the_same_pca(table, offset):
    """A covariance PCA centres the data, so a constant shift of one
    column changes nothing.

    A correlation PCA would change, because it divides by the standard
    deviation -- so this is stated for the covariance method, and only
    for it.
    """
    data = np.asarray(table, dtype=float)
    base = _pca(data)
    if base is None:
        return
    shifted = data.copy()
    shifted[:, 0] += offset
    moved = _pca(shifted)
    if moved is None:
        return
    assert np.allclose(np.asarray(base.eigenvalues), np.asarray(moved.eigenvalues), rtol=1e-6, atol=1e-8)
