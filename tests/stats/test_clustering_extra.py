# =============================================================================
# FILE: tests/stats/test_clustering_extra.py
# =============================================================================
"""
Tests for the PAST4 clustering methods added to stats/clustering.py:
k-medoids (PAM), DBSCAN, and k-nearest neighbours classification.

tests/stats/test_clustering.py covers the hierarchical and k-means paths.
This file covers the three methods that were missing.

The assertions here are built where possible on a known right answer
rather than on self-consistency, because "the output looks plausible" is
what a wrong implementation also produces:

* DBSCAN is checked against the textbook outcome for two separated
  blobs plus one isolated point -- exact labels, with the isolated point
  coming back as noise (-1) rather than being absorbed into a cluster.
  The silhouette is checked against sklearn's own score over the
  clustered samples, so noise cannot quietly be counted as a class.
* k-medoids is checked on the property that *defines* a medoid: the
  returned centres are rows of the input, not means. The PAM fit is
  checked against the two things the algorithm guarantees -- that no
  single medoid exchange improves the result, and that SWAP did better
  than the BUILD step alone -- rather than against a global optimum,
  which PAM does not promise to find.
* k-nearest neighbours is checked for accuracy 1.0 on a trivially
  separable set, and the neighbour-vote counts are checked against the
  votes actually cast, so a method that merely reshuffles probabilities
  cannot pass.

Edge cases are tested per method because they fail differently: a
degenerate partition, a degenerate neighbourhood, and a degenerate
classifier are three different problems.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from stats.clustering import ClusteringAnalyzer, _pam
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError


@pytest.fixture
def analyzer() -> ClusteringAnalyzer:
    """A fresh analyzer."""
    return ClusteringAnalyzer()


def two_blobs_with_isolated_point() -> tuple[npt.NDArray, npt.NDArray]:
    """Two tight blobs 10 apart, plus one point far from both.

    Returns the data and the expected labels. The blobs are tight rather
    than Gaussian so that no sample sits near the eps boundary: the point
    of the fixture is an unambiguous two-cluster-plus-noise answer, not a
    stochastic one that might differ under a different scikit-learn
    version.
    """
    data = np.vstack([np.zeros((6, 2)), np.ones((6, 2)) * 10.0, [[5.0, 5.0]]])
    expected = np.array([0] * 6 + [1] * 6 + [-1])
    return data, expected


def _blobs_with_two_noise_points() -> npt.NDArray:
    """Two loose blobs plus two points far away from both.

    Unlike the fixture above, this one has spread: the blobs are
    Gaussian rather than tight, and the noise points are not isolated
    from everything but form a pair of their own. That matters for the
    silhouette tests: a partition whose points all sit at the same
    distance from the two clusters scores 1.0 whichever way the
    distances are supplied, so it cannot tell a correct score from a
    wrong one.
    """
    rng = np.random.default_rng(21)
    return np.vstack(
        [
            rng.normal(0.0, 0.6, (10, 2)),
            rng.normal(5.0, 0.6, (8, 2)),
            np.array([[20.0, 20.0], [20.5, 20.0]]),
        ]
    )


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


def test_dbscan_labels_two_blobs_and_flags_the_isolated_point_as_noise(
    analyzer: ClusteringAnalyzer,
) -> None:
    """Two separated blobs are two clusters; the lone point is noise."""
    data, expected = two_blobs_with_isolated_point()
    result = analyzer.analyze_dbscan(data, eps=1.0, min_samples=3)

    # Exact labels, not merely "two clusters found": a run that merged
    # the blobs, or absorbed the isolated point into one of them, fails
    # here even though n_clusters would still be 2 in the merge case.
    assert np.array_equal(result.labels, expected)
    assert result.n_clusters == 2
    assert result.n_noise == 1
    assert list(result.noise_indices) == [12]
    # Noise is excluded from the cluster tally, not counted as a class.
    assert result.cluster_sizes() == {0: 6, 1: 6}


def test_dbscan_core_indices_exclude_noise(analyzer: ClusteringAnalyzer) -> None:
    """A point that is not in a dense neighbourhood is not a core point.

    The 12 blob members each have 5 neighbours within eps, meeting
    min_samples=3. The isolated point has only itself, which is 1, so it
    cannot be core. Fails if core_sample_indices_ is used without the
    min_samples check applied.
    """
    data, _ = two_blobs_with_isolated_point()
    result = analyzer.analyze_dbscan(data, eps=1.0, min_samples=3)
    assert 12 not in set(result.core_indices.tolist())
    assert len(result.core_indices) == 12


def test_dbscan_min_samples_counts_the_point_itself(analyzer: ClusteringAnalyzer) -> None:
    """min_samples includes the point, so min_samples=1 makes it core.

    sklearn's documented semantics: a point is core when the number of
    points within eps of it, *counting itself*, is at least
    min_samples. With duplicated rows every point has 10 within eps of
    it, so min_samples=11 turns the whole set into noise while 10 keeps
    one cluster. A test that used a non-boundary value could not tell
    the off-by-one apart from the correct behaviour.
    """
    data = np.zeros((10, 2))  # every point identical, 10 within eps
    kept = analyzer.analyze_dbscan(data, eps=1.0, min_samples=10)
    dropped = analyzer.analyze_dbscan(data, eps=1.0, min_samples=11)
    assert kept.n_clusters == 1 and kept.n_noise == 0
    assert dropped.n_clusters == 0
    assert dropped.n_noise == 10


def test_dbscan_precomputed_matrix_matches_raw_data(analyzer: ClusteringAnalyzer) -> None:
    """A precomputed distance matrix must give the same answer.

    PAST accepts a distance matrix wherever it accepts observations, so
    the two paths are wired to the same result. Feeding sklearn the
    condensed rather than the square form would silently change the
    neighbourhoods; this fails if it regresses.
    """
    data, expected = two_blobs_with_isolated_point()
    raw = analyzer.analyze_dbscan(data, eps=1.0, min_samples=3)
    precomputed = analyzer.analyze_dbscan(squareform(pdist(data)), eps=1.0, min_samples=3, precomputed=True)

    assert precomputed.metric == "precomputed"
    assert np.array_equal(precomputed.labels, raw.labels)
    assert np.array_equal(precomputed.labels, expected)


def test_dbscan_precomputed_silhouette_matches_raw_data(analyzer: ClusteringAnalyzer) -> None:
    """How the distances were supplied must not change the silhouette.

    The same trap as for k-medoids: sklearn's ``silhouette_score``
    defaults to metric='euclidean', so a precomputed distance matrix
    handed over without saying metric='precomputed' is read as n
    samples with n features. Worse, the score is taken over the
    clustered samples only, so the matrix has to be cut down to its
    clustered square sub-matrix; a row selection is not a distance
    matrix at all.
    """
    data = _blobs_with_two_noise_points()
    dm = squareform(pdist(data))

    raw = analyzer.analyze_dbscan(data, eps=1.0, min_samples=3)
    precomputed = analyzer.analyze_dbscan(dm, eps=1.0, min_samples=3, precomputed=True)

    assert np.isclose(precomputed.silhouette, raw.silhouette)


def test_dbscan_silhouette_excludes_noise(analyzer: ClusteringAnalyzer) -> None:
    """Noise is excluded from the silhouette, not scored as a class.

    The score is over the clustered samples only, so it is the score of
    the partition DBSCAN actually found. A run that counted the noise
    points -- as a class of their own, or as members of the nearest
    cluster -- would report a different number, because the noise
    points sit away from every cluster.
    """
    data = _blobs_with_two_noise_points()
    dm = squareform(pdist(data))

    result = analyzer.analyze_dbscan(data, eps=1.0, min_samples=3)
    clustered = result.labels != -1

    assert result.n_noise == 2
    expected = silhouette_score(data[clustered], result.labels[clustered])
    assert np.isclose(result.silhouette, expected)

    # The precomputed path has to score the same partition, over the
    # square sub-matrix of the clustered samples.
    precomputed = analyzer.analyze_dbscan(dm, eps=1.0, min_samples=3, precomputed=True)
    sub_dm = dm[np.ix_(clustered, clustered)]
    assert np.isclose(
        precomputed.silhouette,
        silhouette_score(sub_dm, result.labels[clustered], metric="precomputed"),
    )


def test_dbscan_rejects_eps_of_zero_or_less(analyzer: ClusteringAnalyzer) -> None:
    """A non-positive radius cannot define a neighbourhood."""
    data, _ = two_blobs_with_isolated_point()
    with pytest.raises(ValidationError, match="eps"):
        analyzer.analyze_dbscan(data, eps=0.0)


def test_dbscan_rejects_min_samples_below_one(analyzer: ClusteringAnalyzer) -> None:
    """min_samples counts the point itself, so 0 is not a valid request."""
    data, _ = two_blobs_with_isolated_point()
    with pytest.raises(ValidationError, match="min_samples"):
        analyzer.analyze_dbscan(data, eps=1.0, min_samples=0)


def test_dbscan_rejects_a_single_sample(analyzer: ClusteringAnalyzer) -> None:
    """One sample can never grow a cluster, so the request is refused.

    Without the guard this returns a result with 0 clusters and 1 noise
    point -- technically true and entirely uninformative.
    """
    with pytest.raises(ValidationError, match="at least 2 samples"):
        analyzer.analyze_dbscan(np.array([[1.0, 2.0]]))


def test_dbscan_rejects_a_non_square_precomputed_matrix(analyzer: ClusteringAnalyzer) -> None:
    """A distance matrix must be square."""
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze_dbscan(np.zeros((5, 4)), eps=1.0, precomputed=True)


def test_dbscan_rejects_an_asymmetric_precomputed_matrix(analyzer: ClusteringAnalyzer) -> None:
    """A distance matrix must be symmetric with a zero diagonal."""
    asymmetric = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze_dbscan(asymmetric, eps=1.0, precomputed=True)


def test_dbscan_summary_and_dict_expose_the_noise_count(analyzer: ClusteringAnalyzer) -> None:
    """The noise count is the point of DBSCAN, so both views must carry it."""
    data, _ = two_blobs_with_isolated_point()
    result = analyzer.analyze_dbscan(data, eps=1.0, min_samples=3)

    payload = result.to_dict()
    assert payload["n_noise"] == 1
    assert payload["noise_indices"] == [12]
    assert payload["labels"] == list(result.labels)
    assert "Noise points" in result.summary()
    # Noise must not appear as a cluster of its own.
    assert -1 not in payload["cluster_sizes"]


# ---------------------------------------------------------------------------
# K-medoids (PAM)
# ---------------------------------------------------------------------------


def test_kmedoids_returns_medoids_that_are_actual_input_rows(analyzer: ClusteringAnalyzer) -> None:
    """The defining property of a medoid: it is an observed sample.

    A k-means-style implementation that replaced a centre with a mean
    would pass a "there are k of them" check and fail this one, because
    a mean of several points is not any of the points.
    """
    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(0.0, 0.3, (15, 3)), rng.normal(6.0, 0.3, (15, 3))])
    result = analyzer.analyze_kmedoids(data, n_clusters=2)

    assert result.n_clusters == 2
    assert len(result.medoid_indices) == 2
    for medoid in result.medoid_coordinates:
        assert np.any(np.all(np.isclose(data, medoid), axis=1)), "medoid is not a row of the input data"
    # medoid_coordinates must be the input rows the indices name.
    assert np.allclose(result.medoid_coordinates, data[result.medoid_indices])


def test_kmedoids_recovers_two_separated_blobs(analyzer: ClusteringAnalyzer) -> None:
    """Two tight, far-apart blobs must come back as two clusters."""
    rng = np.random.default_rng(1)
    data = np.vstack([rng.normal(0.0, 0.3, (15, 3)), rng.normal(8.0, 0.3, (15, 3))])
    result = analyzer.analyze_kmedoids(data, n_clusters=2)

    assert sorted(result.cluster_sizes().values()) == [15, 15]
    # Each blob tight around its medoid, so the silhouette is near its max.
    assert result.silhouette > 0.9
    # One medoid per blob: neither medoid may sit in the other cluster.
    for medoid in result.medoid_coordinates:
        assert np.min(np.linalg.norm(data - medoid, axis=1)) < 1e-9


def test_pam_ends_at_a_local_optimum_under_swaps(analyzer: ClusteringAnalyzer) -> None:
    """After BUILD and SWAP, no single medoid/non-medoid exchange may improve.

    This is the property PAM actually guarantees, and the one worth
    asserting. It is deliberately *not* "reaches the global optimum": PAM is a
    greedy BUILD followed by a SWAP hill-climb, so it can stop short of the
    best set. On this fixture (`np.random.default_rng(7)` over 40 points in
    3-D) it happens to reach the global optimum -- cost 36.48204710565008,
    the same as an exhaustive search over all C(40,3) = 9880 medoid sets --
    but that is a property of this data, not a promise the algorithm makes,
    so nothing below depends on it.

    An earlier version of this test asserted the global optimum and passed,
    but only because its fixture happened to be one of the instances where PAM
    gets there. Asserting a guarantee the method does not make means the test
    would fail for a legitimate reason the next time the fixture moved, and
    says more than it checks.
    """
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 3))
    result = analyzer.analyze_kmedoids(data, n_clusters=3)
    distances = squareform(pdist(data))
    meds = list(result.medoid_indices)
    cost = float(distances[:, meds].min(axis=1).sum())

    assert len(set(meds)) == 3, "SWAP must not leave duplicate medoids"
    for out_med in meds:
        for in_med in range(40):
            if in_med in meds:
                continue
            swapped = [in_med if m == out_med else m for m in meds]
            assert float(distances[:, swapped].min(axis=1).sum()) >= cost - 1e-9, (
                f"swapping medoid {out_med} out for {in_med} improves the cost, so SWAP terminated early"
            )


def _greedy_build_cost(distances: npt.NDArray, n_clusters: int) -> float:
    """The cost of the greedy BUILD step, recomputed independently.

    BUILD is the first half of PAM: it takes the point closest to
    everything in total as its first medoid, then repeatedly adds
    whichever remaining point lowers the cost most, and stops when none
    does. Recomputing it here, rather than asking the implementation for
    it, is what makes the comparison in
    ``test_pam_swaps_when_build_alone_would_stop_short`` a test of the
    algorithm rather than a test of itself.

    Parameters:
        distances: The square distance matrix.
        n_clusters: The number of medoids BUILD should choose.

    Returns:
        float: The total distance to the medoids BUILD would pick.
    """
    chosen = [int(np.argmin(distances.sum(axis=1)))]
    current = float(distances[:, chosen].min(axis=1).sum())
    while len(chosen) < n_clusters:
        best_cost = current
        best_candidate: int | None = None
        for candidate in range(distances.shape[0]):
            if candidate in chosen:
                continue
            trial_cost = float(distances[:, [*chosen, candidate]].min(axis=1).sum())
            if trial_cost < best_cost:
                best_cost = trial_cost
                best_candidate = candidate
        if best_candidate is None:
            # No remaining point lowers the cost: the rest are duplicates
            # of the medoids already chosen.
            break
        chosen.append(best_candidate)
        current = best_cost
    return current


def test_pam_swaps_when_build_alone_would_stop_short(analyzer: ClusteringAnalyzer) -> None:
    """SWAP has to be doing something, or BUILD is the whole algorithm.

    The search above proves the result is a local optimum but not that any
    exchange was ever tried. Here a fixture is chosen where BUILD's first
    choice is not the answer: the reported cost must be *strictly* better
    than what a no-swap implementation would produce on the same data.

    The earlier version of this test compared the cost against a BUILD
    recomputed from the first medoid of the *returned* result, and only
    asserted ``cost <= build_only``. That passed with SWAP deleted: with
    SWAP gone the result is a pure BUILD result, so the recomputed build
    equals it and the inequality holds trivially. The comparison is now
    against BUILD recomputed the way BUILD actually works -- starting
    from the point closest to everything in total -- and it has to be a
    strict improvement, so a missing SWAP fails the test.
    """
    rng = np.random.default_rng(11)
    data = rng.normal(size=(35, 4))
    result = analyzer.analyze_kmedoids(data, n_clusters=4)
    distances = squareform(pdist(data))
    meds = list(result.medoid_indices)
    cost = float(distances[:, meds].min(axis=1).sum())

    build_only = _greedy_build_cost(distances, 4)

    # A fit that never exchanged anything did not exchange anything.
    assert result.n_iter > 0, "no medoid exchange was performed, so SWAP did not run"

    # And the exchanges it did make have to have been worth something:
    # PAM has to beat the BUILD step it started from.
    assert cost < build_only - 1e-9, (
        f"PAM cost {cost} is not better than the BUILD-only cost {build_only}, so SWAP improved nothing"
    )


def test_pam_build_starts_from_the_point_closest_to_everything(analyzer: ClusteringAnalyzer) -> None:
    """BUILD's first medoid is the point closest to everything in total.

    The `_pam` docstring makes that claim ("the point closest to
    everything in total, which is the PAM initialisation rather than an
    arbitrary one") and nothing pinned it down: replacing BUILD with
    "take the first k indices" left every test green.

    BUILD is checked on its own here, by running PAM with no swap
    iterations allowed, so the medoids returned are exactly the ones
    BUILD chose. The fixture is one where that first medoid is neither
    the first index of the data nor the medoid the full fit ends up
    with -- SWAP replaces it -- so a BUILD that started anywhere else
    cannot pass by accident.
    """
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 3))
    distances = squareform(pdist(data))

    build_medoids, _build_cost, build_iters = _pam(distances, 3, max_iter=0)

    assert build_iters == 0
    assert build_medoids[0] == int(np.argmin(distances.sum(axis=1)))
    # The fixture has to make the claim meaningful: the full fit ends up
    # with a different first medoid, because SWAP moved it.
    full = analyzer.analyze_kmedoids(data, n_clusters=3)
    assert int(full.medoid_indices[0]) != build_medoids[0]


def test_kmedoids_cost_is_distance_to_medoid_not_squared_distance_to_mean(
    analyzer: ClusteringAnalyzer,
) -> None:
    """PAM minimises L1-style total *distance*; k-means minimises squares.

    Two metrics that agree on clean blobs and disagree in general. This
    asserts the cost equals the summed distance to the assigned medoid,
    and that it is *not* the k-means inertia, so a silently swapped
    objective cannot pass.
    """
    rng = np.random.default_rng(3)
    data = rng.normal(size=(30, 2))
    result = analyzer.analyze_kmedoids(data, n_clusters=3)
    distances = squareform(pdist(data))

    by_labels = sum(float(distances[i, result.medoid_indices[result.labels[i]]]) for i in range(len(data)))
    assert np.isclose(by_labels, result.cost)
    # The medoid of each cluster is its own lowest-total-distance member.
    for label in range(result.n_clusters):
        members = np.flatnonzero(result.labels == label)
        within = distances[np.ix_(members, members)].sum(axis=1)
        assert int(members[int(np.argmin(within))]) == int(result.medoid_indices[label])


def test_kmedoids_is_deterministic(analyzer: ClusteringAnalyzer) -> None:
    """Two runs on the same data agree, because PAM takes no seed.

    k-means needs a random seed and a restart count; PAM breaks ties by
    index and needs neither, which is worth pinning down before someone
    adds a seed parameter on the assumption that the fit is stochastic.
    """
    rng = np.random.default_rng(5)
    data = rng.normal(size=(25, 3))
    first = analyzer.analyze_kmedoids(data, n_clusters=3)
    second = analyzer.analyze_kmedoids(data, n_clusters=3)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.medoid_indices, second.medoid_indices)
    assert first.cost == second.cost


def test_kmedoids_duplicate_rows_still_give_distinct_medoids(analyzer: ClusteringAnalyzer) -> None:
    """Duplicated samples must not collapse the medoids into one.

    With fewer distinct points than requested clusters, BUILD cannot
    reduce the cost any further: the medoids it picks are duplicates of
    each other. The medoids that are reported are still distinct
    indices, because two medoids at the same place are not two
    medoids.

    A cluster that ends up owning nothing is a different matter: it is
    dropped rather than reported, which is what
    ``test_kmedoids_duplicate_rows_report_only_the_clusters_that_exist``
    covers.
    """
    data = np.repeat(np.array([[0.0, 0.0], [5.0, 5.0]]), 5, axis=0)
    result = analyzer.analyze_kmedoids(data, n_clusters=3)

    assert len(set(result.medoid_indices.tolist())) == len(result.medoid_indices)
    assert len(result.labels) == len(data)
    # The reported n_clusters and the cluster sizes describe the same
    # partition.
    assert result.n_clusters == len(result.cluster_sizes())


def test_kmedoids_duplicate_rows_report_only_the_clusters_that_exist(analyzer: ClusteringAnalyzer) -> None:
    """An empty cluster is not a cluster, so the labels are renumbered.

    With fewer distinct samples than requested clusters, two medoids can
    be duplicates of each other. Every sample then goes to the first of
    the pair and the second owns nobody. Reporting three clusters while
    the labels carry only two describes a partition that does not exist,
    and the reported ``n_clusters`` then disagrees with the cluster
    sizes.

    The empty cluster is dropped and the labels renumbered from zero, so
    the label values, the reported ``n_clusters`` and
    ``cluster_sizes()`` all describe the same partition -- and the
    medoids are renumbered with them, so medoid_indices[j] is still the
    medoid of cluster j.
    """
    data = np.repeat(np.array([[0.0, 0.0], [5.0, 5.0]]), 5, axis=0)
    result = analyzer.analyze_kmedoids(data, n_clusters=3)

    assert result.n_clusters == 2
    assert result.cluster_sizes() == {0: 5, 1: 5}
    assert sorted(set(result.labels.tolist())) == [0, 1]
    # One medoid per cluster that exists, and it is the medoid of that
    # cluster.
    assert len(result.medoid_indices) == result.n_clusters
    assert len(result.medoid_coordinates) == result.n_clusters
    distances = squareform(pdist(data))
    for label in range(result.n_clusters):
        members = np.flatnonzero(result.labels == label)
        medoid = int(result.medoid_indices[label])
        assert np.allclose(result.medoid_coordinates[label], data[medoid])
        # Every member is closer to its own medoid than to any other.
        for member in members:
            for other_label in range(result.n_clusters):
                if other_label == label:
                    continue
                other = int(result.medoid_indices[other_label])
                assert distances[member, medoid] <= distances[member, other] + 1e-12
    payload = result.to_dict()
    assert payload["n_clusters"] == len(payload["cluster_sizes"])


def test_kmedoids_rejects_n_clusters_at_or_above_the_sample_count(analyzer: ClusteringAnalyzer) -> None:
    """k >= n means every sample is its own medoid: cost 0, no information."""
    data = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ValidationError, match="smaller than the number of samples"):
        analyzer.analyze_kmedoids(data, n_clusters=6)


def test_kmedoids_rejects_n_clusters_below_one(analyzer: ClusteringAnalyzer) -> None:
    """Zero clusters is not a partition."""
    data = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ValidationError, match="n_clusters"):
        analyzer.analyze_kmedoids(data, n_clusters=0)


def test_kmedoids_rejects_data_with_no_variation(analyzer: ClusteringAnalyzer) -> None:
    """Identical samples make every medoid choice cost zero.

    Returning an arbitrary partition here would present index order as
    though it were structure, so this is refused instead.
    """
    with pytest.raises(ComputationError, match="no variation"):
        analyzer.analyze_kmedoids(np.ones((8, 3)))


def test_kmedoids_precomputed_matrix_matches_raw_data(analyzer: ClusteringAnalyzer) -> None:
    """A precomputed distance matrix gives the same medoids."""
    rng = np.random.default_rng(2)
    data = np.vstack([rng.normal(0.0, 0.3, (12, 2)), rng.normal(7.0, 0.3, (12, 2))])
    raw = analyzer.analyze_kmedoids(data, n_clusters=2)
    precomputed = analyzer.analyze_kmedoids(squareform(pdist(data)), n_clusters=2, precomputed=True)

    assert precomputed.metric == "precomputed"
    assert np.array_equal(precomputed.labels, raw.labels)
    assert np.array_equal(precomputed.medoid_indices, raw.medoid_indices)
    assert np.isclose(precomputed.cost, raw.cost)


def test_kmedoids_precomputed_silhouette_matches_raw_data(analyzer: ClusteringAnalyzer) -> None:
    """How the distances were supplied must not change the silhouette.

    A precomputed distance matrix is a matrix of distances, not a table
    of features. sklearn's ``silhouette_score`` defaults to
    metric='euclidean', so handing it the matrix without saying
    metric='precomputed' makes it read n samples with n features and
    score a partition of the distance profiles -- a different number,
    silently, for the same data. The two paths have to agree.
    """
    rng = np.random.default_rng(2)
    data = np.vstack([rng.normal(0.0, 0.3, (12, 2)), rng.normal(7.0, 0.3, (12, 2))])
    dm = squareform(pdist(data))

    raw = analyzer.analyze_kmedoids(data, n_clusters=2)
    precomputed = analyzer.analyze_kmedoids(dm, n_clusters=2, precomputed=True)

    assert np.isclose(precomputed.silhouette, raw.silhouette)
    # ... and both are the score of the partition that was returned.
    assert np.isclose(
        precomputed.silhouette,
        silhouette_score(dm, precomputed.labels, metric="precomputed"),
    )


def test_kmedoids_summary_and_dict_expose_the_medoid_indices(analyzer: ClusteringAnalyzer) -> None:
    """The medoids are the result, so both views must carry them."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(20, 2))
    result = analyzer.analyze_kmedoids(data, n_clusters=2)

    payload = result.to_dict()
    assert payload["medoid_indices"] == [int(i) for i in result.medoid_indices]
    assert len(payload["medoid_coordinates"]) == 2
    assert "K-Medoids" in result.summary()


# ---------------------------------------------------------------------------
# K-nearest neighbours classification
# ---------------------------------------------------------------------------


def test_knn_is_perfectly_accurate_on_a_separable_set(analyzer: ClusteringAnalyzer) -> None:
    """Accuracy must be exactly 1.0 when the classes are far apart."""
    train = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10.0])
    labels = np.array([0] * 10 + [1] * 10)
    queries = np.array([[0.1, 0.1], [9.9, 9.9]])
    query_labels = np.array([0, 1])

    result = analyzer.classify_knn(train, labels, queries, n_neighbors=3, query_labels=query_labels)

    assert result.accuracy == 1.0
    assert result.predictions == [0, 1]
    assert result.n_queries == 2
    assert result.n_train == 20


def test_knn_scores_are_none_when_no_query_labels_are_supplied(analyzer: ClusteringAnalyzer) -> None:
    """Accuracy is NaN rather than a score invented from the training fit.

    Classifying the training rows is in-sample -- a sample is always its
    own nearest neighbour -- so a default of 1.0 here would be an
    artefact, not a measurement.
    """
    train = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10.0])
    labels = np.array([0] * 10 + [1] * 10)

    result = analyzer.classify_knn(train, labels, np.array([[0.1, 0.1]]), n_neighbors=3)

    assert np.isnan(result.accuracy)
    assert result.confusion.size == 0
    assert "not scored" in result.summary()


def test_knn_in_sample_accuracy_is_not_one_by_construction(analyzer: ClusteringAnalyzer) -> None:
    """Scoring the training rows is in-sample, and it is not perfect.

    A point is always its own nearest neighbour, but its own vote is
    only one of the k votes, so with k > 1 its own class can be
    outvoted by the neighbours. The point at the origin here sits
    inside a ring of class 1 points: with k = 5 its own vote is one
    against four, so it is classified as class 1 and the in-sample
    accuracy is 8/9 rather than 1.0.

    This pins down the behaviour the docstring describes, after the
    docstring wrongly promised an accuracy of 1.0 "by construction".
    """
    train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.2, 5.2],
        ]
    )
    labels = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2])

    result = analyzer.classify_knn(train, labels, None, n_neighbors=5, query_labels=labels)

    assert result.predictions[0] == 1, "the class 0 point should have been outvoted by its neighbours"
    assert result.accuracy == pytest.approx(8 / 9)
    assert result.accuracy < 1.0


def test_knn_confusion_matrix_counts_true_against_predicted(analyzer: ClusteringAnalyzer) -> None:
    """A query placed next to the wrong class must land off-diagonal.

    With the confusion matrix unchecked, a method that returned the
    training labels unchanged would still score well here. One query
    deliberately sits inside class 1's territory but is labelled 0, so
    the matrix has an off-diagonal entry and accuracy drops below 1.0.
    """
    train = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10.0])
    labels = np.array([0] * 10 + [1] * 10)
    queries = np.array([[0.1, 0.1], [8.0, 8.0]])  # second query really is class 1
    query_labels = np.array([0, 0])  # ...but is claimed to be class 0

    result = analyzer.classify_knn(train, labels, queries, n_neighbors=3, query_labels=query_labels)

    assert result.predictions == [0, 1]
    assert result.accuracy == 0.5
    # classes are [0, 1]; rows are the TRUE class and columns the
    # PREDICTED one. Both queries were claimed to be class 0: the first
    # was predicted 0 and the second 1, so row 0 holds one hit and one
    # miss and row 1 is empty.
    assert result.confusion.tolist() == [[1, 1], [0, 0]]


def test_knn_neighbor_counts_are_actual_votes(analyzer: ClusteringAnalyzer) -> None:
    """The per-class counts must be real vote counts, not probabilities.

    Recovering them as predict_proba * k looks equivalent and is not.
    Under weights='distance' sklearn returns distance-weighted
    probabilities, so multiplying by k yields weights, not counts. The
    uneven-neighbourhood case below is the one that separates them: its
    five neighbours are three of one class and two of the other at
    unequal distances, so the weighted probabilities round to [1, 4]
    while the votes actually cast are [2, 3].
    """
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, (60, 2))
    labels = np.array([0] * 30 + [1] * 30)
    query = np.array([[-1.6814, -1.0121]])

    result = analyzer.classify_knn(train, labels, query, n_neighbors=5, weights="distance")

    # Counts the neighbours themselves determine, computed independently
    # of the implementation: 2 vs 3, summing to k.
    assert result.neighbor_counts[0].tolist() == [2, 3]
    assert result.neighbor_counts.sum() == 5


def test_knn_neighbor_counts_sum_to_k_for_uniform_weights(analyzer: ClusteringAnalyzer) -> None:
    """Every row of the vote table must total k, in either weighting."""
    train = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10.0])
    labels = np.array([0] * 10 + [1] * 10)
    queries = np.array([[0.1, 0.1], [9.9, 9.9]])

    for weights in ("uniform", "distance"):
        result = analyzer.classify_knn(train, labels, queries, n_neighbors=3, weights=weights)
        assert result.neighbor_counts.shape == (2, 2)
        assert result.neighbor_counts.sum(axis=1).tolist() == [3, 3]
        # Each query sits inside its own class, so all 3 votes agree.
        assert result.neighbor_counts[0, 0] == 3
        assert result.neighbor_counts[1, 1] == 3


def test_knn_neighbor_counts_have_one_row_per_query(analyzer: ClusteringAnalyzer) -> None:
    """The vote table is (n_queries, n_classes), not the shape of the confusion matrix.

    The two are easy to confuse. The confusion matrix is
    (n_classes, n_classes) and counts queries; the vote table is one
    row per query and counts neighbours. A fixture with three queries
    and two classes is used here precisely because the two shapes then
    differ, so a test that asserted (2, 2) for a two-query fixture
    would not have told them apart.
    """
    train = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10.0])
    labels = np.array([0] * 10 + [1] * 10)
    queries = np.array([[0.1, 0.1], [9.9, 9.9], [0.2, 0.2]])

    result = analyzer.classify_knn(train, labels, queries, n_neighbors=3, query_labels=np.array([0, 1, 0]))

    assert result.confusion.shape == (2, 2)
    assert result.neighbor_counts.shape == (3, 2)
    assert result.neighbor_counts.sum(axis=1).tolist() == [3, 3, 3]


def test_knn_handles_string_class_labels(analyzer: ClusteringAnalyzer) -> None:
    """Species names are strings, and must survive as strings.

    A numeric-only implementation returns numpy string scalars or an
    index instead of the label, which would be useless for naming a
    taxon. Compared against Python str so the type is checked, not just
    the value.
    """
    train = np.vstack([np.zeros((8, 2)), np.ones((8, 2)) * 10.0])
    labels = ["T. rex"] * 8 + ["T. triceratops"] * 8

    result = analyzer.classify_knn(train, labels, np.array([[0.1, 0.1]]), n_neighbors=3)

    assert result.predictions == ["T. rex"]
    assert isinstance(result.predictions[0], str)
    assert result.classes == ["T. rex", "T. triceratops"]


def test_knn_predictions_are_plain_python_values(analyzer: ClusteringAnalyzer) -> None:
    """The predictions are plain Python values, not numpy scalars.

    A numpy integer is not JSON-serialisable and a numpy string scalar
    is not a ``str``, so a caller that names a taxon or exports the
    result gets the wrong type unless the values are converted. The
    docstring promises plain Python values, so the type is checked here
    and not just the value.
    """
    train = np.vstack([np.zeros((8, 2)), np.ones((8, 2)) * 10.0])

    numeric = analyzer.classify_knn(train, np.array([0] * 8 + [1] * 8), np.array([[0.1, 0.1]]), n_neighbors=3)
    assert numeric.predictions == [0]
    assert type(numeric.predictions[0]) is int

    named = analyzer.classify_knn(
        train,
        ["T. rex"] * 8 + ["T. triceratops"] * 8,
        np.array([[0.1, 0.1]]),
        n_neighbors=3,
    )
    assert named.predictions == ["T. rex"]
    assert type(named.predictions[0]) is str


def test_knn_rejects_k_larger_than_the_training_set(analyzer: ClusteringAnalyzer) -> None:
    """k cannot exceed the number of stored samples."""
    train = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValidationError, match="n_neighbors"):
        analyzer.classify_knn(train, np.arange(10) % 2, n_neighbors=11)


def test_knn_rejects_k_below_one(analyzer: ClusteringAnalyzer) -> None:
    """Zero neighbours is not a classifier."""
    train = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValidationError, match="n_neighbors"):
        analyzer.classify_knn(train, np.arange(10) % 2, n_neighbors=0)


def test_knn_rejects_an_unknown_weighting(analyzer: ClusteringAnalyzer) -> None:
    """Only 'uniform' and 'distance' mean anything here."""
    train = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValidationError, match="weights"):
        analyzer.classify_knn(train, np.arange(10) % 2, weights="nearest")


def test_knn_rejects_mismatched_training_label_count(analyzer: ClusteringAnalyzer) -> None:
    """One label per training sample, no more and no fewer."""
    train = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValidationError, match="train_labels"):
        analyzer.classify_knn(train, np.arange(4) % 2)


def test_knn_rejects_a_query_with_a_different_variable_count(analyzer: ClusteringAnalyzer) -> None:
    """Queries must have the same variables as the training data."""
    train = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(MatrixDimensionError):
        analyzer.classify_knn(train, np.arange(10) % 2, query_data=np.zeros((4, 5)))


def test_knn_rejects_a_single_training_sample(analyzer: ClusteringAnalyzer) -> None:
    """One reference sample cannot form a neighbourhood."""
    with pytest.raises(ValidationError, match="at least 2 training samples"):
        analyzer.classify_knn(np.zeros((1, 2)), [0])


def test_knn_rejects_mismatched_query_label_count(analyzer: ClusteringAnalyzer) -> None:
    """query_labels must line up with the rows actually classified."""
    train = np.vstack([np.zeros((6, 2)), np.ones((6, 2)) * 10.0])
    with pytest.raises(ValidationError, match="query_labels"):
        analyzer.classify_knn(
            train,
            np.array([0] * 6 + [1] * 6),
            np.array([[0.1, 0.1]]),
            n_neighbors=3,
            query_labels=np.array([0, 1]),
        )


# ---------------------------------------------------------------------------
# Parameter handling shared by all three
# ---------------------------------------------------------------------------


def test_an_unknown_metric_is_rejected_as_a_validation_error(analyzer: ClusteringAnalyzer) -> None:
    """A metric no backend understands must come back as ValidationError.

    The three methods used to let the name through to scipy or
    scikit-learn, which raised their own exceptions: a bare ValueError
    ("Unknown Distance Metric") for k-medoids and an
    InvalidParameterError for the other two. Neither is one of this
    project's exceptions, so a caller catching PaleoASTError did not
    catch them and had no idea which method had refused.
    """
    data = np.random.default_rng(0).normal(size=(10, 2))

    with pytest.raises(ValidationError, match="metric"):
        analyzer.analyze_dbscan(data, eps=1.0, metric="not_a_metric")
    with pytest.raises(ValidationError, match="metric"):
        analyzer.analyze_kmedoids(data, n_clusters=2, metric="not_a_metric")
    with pytest.raises(ValidationError, match="metric"):
        analyzer.classify_knn(data, np.arange(10) % 2, data[:2], n_neighbors=3, metric="not_a_metric")


def test_precomputed_metric_is_rejected_where_it_does_not_apply(analyzer: ClusteringAnalyzer) -> None:
    """'precomputed' is a metric name, not a way of saying the data is a matrix.

    DBSCAN and k-nearest neighbours both know the name, but for them it
    means "the data you hand me is already a distance matrix". DBSCAN
    has a flag for that; k-nearest neighbours has no way to accept a
    distance matrix at all, because it has to fit the classifier on
    observations. Passing the name instead of the flag used to reach
    scikit-learn and fail there with a shape error.
    """
    data = np.random.default_rng(0).normal(size=(10, 2))

    with pytest.raises(ValidationError, match="precomputed"):
        analyzer.analyze_dbscan(data, eps=1.0, metric="precomputed")
    with pytest.raises(ValidationError, match="precomputed"):
        analyzer.classify_knn(data, np.arange(10) % 2, data[:2], n_neighbors=3, metric="precomputed")


# ---------------------------------------------------------------------------
# Result plumbing shared by all three
# ---------------------------------------------------------------------------


def test_last_result_properties_track_the_most_recent_run(analyzer: ClusteringAnalyzer) -> None:
    """Each method records its result and hands it back on the property.

    Asserts the value is the same object, not merely an equal one, so a
    copy returned from somewhere else cannot pass.
    """
    assert analyzer.last_kmedoids_result is None
    assert analyzer.last_dbscan_result is None
    assert analyzer.last_knn_result is None

    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(0.0, 0.3, (10, 2)), rng.normal(8.0, 0.3, (10, 2))])

    kmedoids = analyzer.analyze_kmedoids(data, n_clusters=2)
    assert analyzer.last_kmedoids_result is kmedoids

    dbscan = analyzer.analyze_dbscan(data, eps=1.0, min_samples=2)
    assert analyzer.last_dbscan_result is dbscan

    knn = analyzer.classify_knn(data, np.arange(20) % 2, np.array([[0.1, 0.1]]), n_neighbors=3)
    assert analyzer.last_knn_result is knn

    # The earlier hierarchical/k-means slots are untouched by the new ones.
    assert analyzer.last_result is None
    assert analyzer.last_kmeans_result is None


def test_to_dict_output_is_json_serialisable(analyzer: ClusteringAnalyzer) -> None:
    """to_dict feeds the JSON export path, so numpy scalars must not leak.

    numpy ints are not JSON-serialisable; a leaked np.int64 raises from
    the json module rather than from this code, so the round trip is what
    actually proves the contract.
    """
    import json

    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(0.0, 0.3, (10, 2)), rng.normal(8.0, 0.3, (10, 2))])

    for payload in (
        analyzer.analyze_kmedoids(data, n_clusters=2).to_dict(),
        analyzer.analyze_dbscan(data, eps=1.0, min_samples=2).to_dict(),
        analyzer.classify_knn(data, np.arange(20) % 2, np.array([[0.1, 0.1]]), n_neighbors=3).to_dict(),
    ):
        assert isinstance(json.dumps(payload), str)


def test_new_methods_do_not_break_import_without_scikit_learn() -> None:
    """stats.clustering must still import on a base install.

    scikit-learn is in the ``full`` extra, not the base dependencies, so
    a module-level import of DBSCAN or KNeighborsClassifier would break
    every package that reaches this module -- ``controllers``, and
    therefore ``views`` -- on a base install. K-medoids must keep
    working there, because it is implemented on numpy and scipy alone,
    and the other two must refuse with the project's own error naming
    the package.
    """
    script = (
        "import builtins\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name == 'sklearn' or name.startswith('sklearn.'):\n"
        "        raise ImportError(name)\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import numpy as np\n"
        "from utils.exceptions import ComputationError\n"
        "import stats.clustering as C\n"
        "rng = np.random.default_rng(0)\n"
        "data = np.vstack([rng.normal(0, .3, (10, 2)), rng.normal(8, .3, (10, 2))])\n"
        "a = C.ClusteringAnalyzer()\n"
        "r = a.analyze_kmedoids(data, n_clusters=2, compute_silhouette=False)\n"
        "assert r.n_clusters == 2, 'k-medoids must not need scikit-learn'\n"
        "for name, call in (\n"
        "    ('DBSCAN', lambda: a.analyze_dbscan(data, eps=1.0)),\n"
        "    ('kNN', lambda: a.classify_knn(data, np.arange(20) % 2, data[:2], n_neighbors=3)),\n"
        "):\n"
        "    try:\n"
        "        call()\n"
        "    except ComputationError as exc:\n"
        "        assert 'scikit-learn' in str(exc), (name, str(exc))\n"
        "    else:\n"
        "        raise AssertionError(name + ' should have refused')\n"
        "print('OK')\n"
    )
    root = Path(__file__).resolve().parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout
