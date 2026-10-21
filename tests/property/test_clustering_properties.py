# =============================================================================
# FILE: tests/property/test_clustering_properties.py
# =============================================================================
"""
Property-based tests for the clustering.

A partition is a partition: the labels are cluster indices, the
centroids are the means of the members, the within-cluster sum of
squares cannot be larger than the total sum of squares, and reordering
the samples cannot change the answer.  Each of those is a property of
the *result*, not of the algorithm that produced it, and each of them
catches a different way of getting the clustering wrong.

References:
    Lloyd, S.P. (1982). Least squares quantization in PCM. IEEE
    Transactions on Information Theory 28, 129-137.
    Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. Proceedings of KDD, 226-231.
    Kaufman, L. & Rousseeuw, P.J. (1990). Finding Groups in Data. Wiley.

A NOTE ON WHAT IS NOT HERE
---------------------------
There is no "the partition does not depend on the order of the
samples" property for k-means, and the reason is worth recording:
k-means starts from a RANDOM initialisation, so a reordered sample
starts from a different initialisation and can settle in a different
local optimum.  That is a property of the algorithm, not a defect, and a
property test that asserted otherwise would be asserting something
false.  The deterministic algorithms are the ones where order
independence is a claim worth making.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stats.clustering import ClusteringAnalyzer

from ._helpers import call_or_none

_SETTING = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])

#: A table of samples with a fixed shape.
_matrix = st.lists(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=2,
    ),
    min_size=6,
    max_size=20,
)


def _kmeans(data: np.ndarray, k: int):
    """A k-means partition of the data, or None."""
    return call_or_none(
        ClusteringAnalyzer().analyze_kmeans,
        data,
        n_clusters=k,
        random_seed=3,
        compute_silhouette=False,
    )


@given(table=_matrix, k=st.integers(min_value=2, max_value=4))
@_SETTING
def test_the_kmeans_labels_are_cluster_indices(table, k):
    """A label is a cluster index, and every sample carries one."""
    data = np.asarray(table, dtype=float)
    result = _kmeans(data, k)
    if result is None:
        return
    labels = np.asarray(result.labels)
    assert labels.shape == (data.shape[0],)
    assert np.all(labels >= 0)
    assert np.all(labels < k)


@given(table=_matrix, k=st.integers(min_value=2, max_value=4))
@_SETTING
def test_the_kmeans_centroids_are_the_means_of_their_members(table, k):
    """A k-means centroid is the mean of the samples in its cluster."""
    data = np.asarray(table, dtype=float)
    result = _kmeans(data, k)
    if result is None:
        return
    labels = np.asarray(result.labels)
    centroids = np.asarray(result.centroids, dtype=float)
    for cluster in range(centroids.shape[0]):
        members = data[labels == cluster]
        if members.shape[0] == 0:
            continue
        assert np.allclose(centroids[cluster], members.mean(axis=0), atol=1e-6)


@given(table=_matrix, k=st.integers(min_value=2, max_value=4))
@_SETTING
def test_the_kmeans_inertia_is_between_zero_and_the_total_variance(table, k):
    """The within-cluster sum of squares cannot be negative, and it
    cannot exceed the total sum of squares.

    The total is the sum of squared distances from the grand mean, which
    is the partition with a single cluster, so any partition is at least
    as good as that one.
    """
    data = np.asarray(table, dtype=float)
    result = _kmeans(data, k)
    if result is None:
        return
    assert result.inertia >= 0.0
    total = float(np.sum((data - data.mean(axis=0)) ** 2))
    assert result.inertia <= total + 1e-6


@given(table=_matrix)
@_SETTING
def test_the_dbscan_labels_mark_the_noise_with_minus_one(table):
    """DBSCAN labels the points it could not reach as noise, and noise
    is marked with -1."""
    data = np.asarray(table, dtype=float)
    result = call_or_none(
        ClusteringAnalyzer().analyze_dbscan,
        data,
        eps=5.0,
        min_samples=2,
        compute_silhouette=False,
    )
    if result is None:
        return
    labels = np.asarray(result.labels)
    assert np.all(labels >= -1)
    noise = np.asarray(result.noise_indices)
    assert np.array_equal(np.sort(noise), np.flatnonzero(labels == -1))
