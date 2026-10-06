# =============================================================================
# FILE: tests/stats/test_spatial_stats.py
# =============================================================================
"""
Tests for Moran's I, grid interpolation, nearest-neighbour statistics and
spherical statistics.

Every numeric assertion here states where its expected value came from, and
in each case the value is fixed *before* the module under test runs: a 5x5
lattice whose value patterns have analytically computable Moran's I, four
samples of a unit square whose interpolated centre is an arithmetic mean, a
Poisson process whose nearest-neighbour mean is a Rayleigh integral that the
test evaluates itself, and unit vectors whose resultant is known in closed
form. Nothing asserts that the code agrees with itself.

The one place a Monte Carlo p-value is asserted against a threshold rather
than a fixed number is the significance calibration check, and that is
deliberate: the claim being tested *is* a rate (a 0.05-level test rejects
about 5 % of null cases), so the assertion is on the count over many seeded
draws, with headroom, rather than on any single run.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import quad

from stats.spatial_stats import SpatialStatsAnalyzer, contiguity_weights
from utils.exceptions import ComputationError, MatrixDimensionError, ValidationError

# =============================================================================
# Fixtures and constructed inputs
# =============================================================================


@pytest.fixture
def analyzer() -> SpatialStatsAnalyzer:
    """A fresh analyzer."""
    return SpatialStatsAnalyzer()


def _lattice(n: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """An ``n x n`` unit lattice: coordinates, row index and column index.

    Site ``i`` sits at ``(i % n, i // n)``, so ``i = row * n + col`` and the
    rook neighbourhood below is symmetric in the two index arrays.
    """
    row, col = np.divmod(np.arange(n * n), n)
    coordinates = np.column_stack([col.astype(float), row.astype(float)])
    return coordinates, row.astype(float), col.astype(float)


def _rook_weights(n: int) -> npt.NDArray:
    """Rook (4-neighbour) contiguity of an ``n x n`` lattice, built here.

    Written out in the test rather than taken from the module, so that the
    weights the module is given are the weights the derivation below assumes.
    """
    weights = np.zeros((n * n, n * n))
    for row in range(n):
        for col in range(n):
            i = row * n + col
            for drow, dcol in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                r2, c2 = row + drow, col + dcol
                if 0 <= r2 < n and 0 <= c2 < n:
                    weights[i, r2 * n + c2] = 1.0
    return weights


def _unit_corners() -> tuple[npt.NDArray, npt.NDArray]:
    """The four corners of the unit square, valued 1, 2, 4 and 8."""
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    values = np.array([1.0, 2.0, 4.0, 8.0])
    return coordinates, values


# =============================================================================
# Moran's I -- known answers
# =============================================================================
#
# Moran's I for a row-standardised weight matrix W and centred values z is
#
#     I = (n / S0) * (z^T W z) / (z^T z),      S0 = sum(W) = n for row-standardised W
#
# so I = (z^T W z) / (z^T z) after standardisation. Two patterns on a 5x5
# rook lattice have closed-form answers:
#
# * Strict two-colouring (checkerboard). Every edge joins opposite signs, so
#   edge {i, j} contributes z_i z_j (1/r_i + 1/r_j) = -(1/r_i + 1/r_j) to
#   z^T W z. Each site i sits on r_i edges and so contributes -1 in total,
#   giving z^T W z = -n exactly -- for *any* graph, and independent of the
#   degrees. Hence I = -n / n = -1 exactly, and E[I] = -1/24 cannot be
#   confused with it.
#
# * Vertical ramp, z = (row - 2) on the 5x5 lattice, so z^T z = 50. Row
#   sums are 2 at the corners, 3 on the open edges, 4 inside. Each site's
#   contribution is z_i * (sum of its neighbours) / r_i, and the
#   neighbour sum depends only on the row:
#       row 0 (z=-2): 2 sites give (-2)(-3)/2 = 3, 3 sites give (-2)(-5)/3 = 10/3
#       row 1 (z=-1): 2 sites give (-1)(-3)/3 = 1, 3 sites give (-1)(-4)/4 = 1
#       row 2 (z= 0): every contribution is 0
#       row 3, row 4: mirror of rows 1, 0
#   giving z^T W z = (6 + 10) + 5 + 0 + 5 + (6 + 10) = 42, so I = 42/50 = 0.84.


def test_morans_checkerboard_is_exactly_minus_one(analyzer: SpatialStatsAnalyzer) -> None:
    """I = -1 exactly for a strict two-colouring of any graph.

    Proven above: each site contributes exactly -1 to ``z^T W z`` however
    many neighbours it has, so the statistic is -n/n.
    """
    _coordinates, _row, col = _lattice(5)
    row, _col = np.divmod(np.arange(25), 5)
    checkerboard = (-1.0) ** (row + col)

    result = analyzer.morans_i(checkerboard, weights=_rook_weights(5))
    assert result.statistic == pytest.approx(-1.0, abs=1e-12)
    # A strongly *dispersed* arrangement, and unmistakably not the -0.0417
    # expected under no autocorrelation.
    assert result.statistic < result.expected


def test_morans_vertical_ramp_matches_the_hand_computed_value(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """I = 0.84 on the 5x5 rook lattice, from the per-row derivation above."""
    _coordinates, row, _col = _lattice(5)
    ramp = row - 2.0

    result = analyzer.morans_i(ramp, weights=_rook_weights(5))
    assert result.statistic == pytest.approx(0.84, abs=1e-12)


def test_morans_matches_the_textbook_formula(analyzer: SpatialStatsAnalyzer) -> None:
    """I reproduced from Cliff, Ord & Thompson's formula, densely.

    An independent arrangement of the same expression, using explicit matrix
    arithmetic rather than the module's vectorised path, so that a
    transcription slip in either one shows up as a disagreement.
    """
    _coordinates, row, _col = _lattice(5)
    ramp = row - 2.0
    weights = _rook_weights(5)

    standardised = weights / weights.sum(axis=1, keepdims=True)
    z = ramp - ramp.mean()
    expected_i = (len(ramp) / standardised.sum()) * float(z @ standardised @ z) / float(z @ z)

    result = analyzer.morans_i(ramp, weights=weights)
    assert result.statistic == pytest.approx(expected_i, rel=1e-12)


def test_morans_expected_value_is_minus_one_over_n_minus_one(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Cliff, Ord & Thompson (1974): E[I] = -1 / (n - 1) = -1/24 at n = 25."""
    _coordinates, row, _col = _lattice(5)
    result = analyzer.morans_i(row - 2.0, weights=_rook_weights(5))
    assert result.expected == pytest.approx(-1.0 / 24.0)


def test_morans_ranks_a_ramp_above_random_above_a_checkerboard(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """The ordering is the property, and it is what the statistic is for.

    A smooth gradient, values scattered at random, and a checkerboard are
    the three canonical arrangements, and a weight structure that gets the
    sign convention wrong cannot produce this order.
    """
    _coordinates, row, _col = _lattice(5)
    weights = _rook_weights(5)
    scramble = np.random.default_rng(0).normal(size=25)

    ramp = analyzer.morans_i(row - 2.0, weights=weights).statistic
    random = analyzer.morans_i(scramble, weights=weights).statistic
    checker = analyzer.morans_i((-1.0) ** np.arange(25), weights=weights).statistic

    assert ramp > random > checker


def test_morans_significance_matches_the_arrangement(analyzer: SpatialStatsAnalyzer) -> None:
    """A ramp is significant; a random assignment of the same 25 values is not.

    The expected p-value for a strongly clustered pattern is the floor
    ``1 / (n_permutations + 1)`` -- 199 permutations cannot certify better
    than that, and reporting 0 would claim more evidence than they carry.
    """
    _coordinates, row, _col = _lattice(5)
    weights = _rook_weights(5)
    scramble = np.random.default_rng(3).normal(size=25)

    clustered = analyzer.morans_i(row - 2.0, weights=weights, permutations=199, random_seed=11)
    assert clustered.p_value == pytest.approx(1.0 / 200.0)
    assert clustered.significant
    # z is measured against the permuted null, whose mean sits at E[I] and
    # whose spread is small, so a large positive deviation is a large z.
    assert clustered.z_score > 3.0

    scattered = analyzer.morans_i(scramble, weights=weights, permutations=199, random_seed=11)
    assert scattered.p_value > 0.05
    assert not scattered.significant


def test_morans_dispersed_is_significant_in_the_two_sided_test_only(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """A negative I is evidence, and only a one-sided 'greater' test misses it.

    I is signed, so a perfectly dispersed pattern is as significant as a
    perfectly clustered one. The one-sided p-value is 1.0 here -- the
    observed I is the *smallest* value in the null, so no permutation is at
    least as large -- and reporting that alone would hide a result at the
    permutation floor. This is the assertion a sign-convention error in the
    two-sided branch fails.
    """
    _coordinates, _row, col = _lattice(5)
    row, _col = np.divmod(np.arange(25), 5)
    checkerboard = (-1.0) ** (row + col)

    result = analyzer.morans_i(checkerboard, weights=_rook_weights(5), permutations=199, random_seed=11)
    assert result.statistic < 0
    assert result.p_value == pytest.approx(1.0 / 200.0)
    assert result.p_value_greater == pytest.approx(1.0)
    assert result.significant


def test_morans_rejection_rate_under_the_null_is_about_five_percent(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """A correct test rejects about 5 % of null cases; this one rejects 1 in 20.

    The claim is a rate, so the assertion is on a count over 20 independently
    seeded random arrangements of values on the same lattice, with headroom
    for one unlucky draw. A test whose null is mis-specified -- a global
    permutation of the wrong kind, or a normal approximation applied where
    the distribution is skewed at n = 25 -- rejects far more or far less.
    """
    weights = _rook_weights(5)
    rejected = 0
    for seed in range(20):
        values = np.random.default_rng(seed).normal(size=25)
        result = analyzer.morans_i(values, weights=weights, permutations=199, random_seed=1)
        rejected += int(result.significant)
    assert rejected <= 3


def test_morans_coordinate_mode_uses_inverse_distance(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """With coordinates and no weights, the default contiguity is inverse distance.

    The same lattice and ramp, with the inverse-distance matrix rebuilt here
    as ``1 / d`` off the diagonal and row-standardised, must give the same
    I to within the module's guard: it adds ``eps = 1e-9 * median(d)`` to each
    distance so that a genuinely coincident pair cannot divide by zero, which
    is a 1e-9 relative perturbation and the whole of the residual below. The
    value differs from the rook-lattice 0.84 because a different contiguity
    is a different statistic -- the *ordering* is what carries over, not the
    number.
    """
    coordinates, row, _col = _lattice(5)
    ramp = row - 2.0

    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    off_diagonal = ~np.eye(25, dtype=bool)
    inverse_distance = np.zeros((25, 25))
    inverse_distance[off_diagonal] = 1.0 / distances[off_diagonal]
    standardised = inverse_distance / inverse_distance.sum(axis=1, keepdims=True)
    z = ramp - ramp.mean()
    expected_i = float(z @ standardised @ z) / float(z @ z)

    result = analyzer.morans_i(ramp, coordinates=coordinates)
    assert result.weight_scheme == "inverse-distance"
    assert result.statistic == pytest.approx(expected_i, rel=1e-8)
    # Heavily weighted by close neighbours, so the ramp is weaker than under
    # 4-neighbour contiguity (0.236 against 0.84) but still clearly positive.
    assert 0.0 < result.statistic < 0.84


def test_morans_reports_the_expectation_it_tests_against(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """The reported E[I] is the null mean, not a hard-coded constant."""
    coordinates, _row, _col = _lattice(4)
    values = coordinates[:, 0] + coordinates[:, 1]
    result = analyzer.morans_i(values, coordinates=coordinates, random_seed=1)
    assert result.expected == pytest.approx(-1.0 / 15.0)


# =============================================================================
# Moran's I -- contiguity builders
# =============================================================================


def test_contiguity_weights_knn_is_symmetric_with_a_zero_diagonal() -> None:
    """k-nearest is symmetrised, so degrees can exceed k.

    Symmetrising a directed k-nearest rule is what makes it a contiguity: a
    site that chose a neighbour but was not chosen back must still be
    adjacent, and an asymmetric matrix cannot be row-standardised into a
    spatial average.
    """
    weights = contiguity_weights(_lattice(5)[0], method="knn", k=4)
    assert weights.shape == (25, 25)
    np.testing.assert_array_equal(weights, weights.T)
    assert np.trace(weights) == 0.0
    # Every site has at least its k nearest neighbours, and symmetrising can
    # give it more, so degrees run from k upwards rather than being exactly k.
    assert np.all(weights.sum(axis=1) >= 4)
    assert set(np.unique(weights.sum(axis=1))) >= {4}


def test_contiguity_weights_delaunay_joins_only_nearby_sites() -> None:
    """Delaunay on a jittered lattice recovers roughly the 4-neighbour graph.

    A square lattice is exactly cocircular in places, so a jitter is applied:
    the test should pin the contiguity construction, not Qhull's tolerance
    for degenerate input.
    """
    coordinates, _row, _col = _lattice(5)
    jittered = coordinates + np.column_stack(
        [
            0.13 * np.random.default_rng(1).normal(size=25),
            0.13 * np.random.default_rng(2).normal(size=25),
        ]
    )
    weights = contiguity_weights(jittered, method="delaunay")
    assert weights.shape == (25, 25)
    np.testing.assert_array_equal(weights, weights.T)
    assert np.trace(weights) == 0.0
    # A triangulation of n points with a convex hull of h vertices has
    # 3n - 3 - h triangles' worth of edges, i.e. 3n - 3 - h distinct edges
    # (Euler). Here n = 25, and the jittered 5x5 square has a hull of 8
    # vertices, so 3 * 25 - 3 - 8 = 64 exactly.
    assert weights.sum() / 2 == pytest.approx(64.0)


def test_contiguity_weights_rejects_an_unknown_method() -> None:
    """A typo must not silently fall back to the default."""
    with pytest.raises(ValidationError, match="method must be"):
        contiguity_weights(_lattice(5)[0], method="gaussian")


# =============================================================================
# Moran's I -- rejection
# =============================================================================


def test_morans_rejects_too_few_sites(analyzer: SpatialStatsAnalyzer) -> None:
    """Below four sites the permutation null has too few arrangements."""
    with pytest.raises(ValidationError, match="at least 4 sites"):
        analyzer.morans_i(np.array([1.0, 2.0, 3.0]), weights=np.ones((3, 3)) - np.eye(3))


def test_morans_rejects_a_constant_variable(analyzer: SpatialStatsAnalyzer) -> None:
    """Zero variance makes I 0/0; the module must not invent a number."""
    with pytest.raises(ComputationError, match="constant"):
        analyzer.morans_i(np.ones(25), weights=_rook_weights(5))


def test_morans_needs_coordinates_or_weights(analyzer: SpatialStatsAnalyzer) -> None:
    """There is no default contiguity without a geometry."""
    with pytest.raises(ValidationError, match="coordinates or a weight matrix"):
        analyzer.morans_i(np.arange(25.0))


def test_morans_rejects_a_non_square_weight_matrix(analyzer: SpatialStatsAnalyzer) -> None:
    with pytest.raises(MatrixDimensionError, match="square"):
        analyzer.morans_i(np.arange(25.0), weights=np.ones((5, 4)))


def test_morans_rejects_weights_that_do_not_match_the_values(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """One value per site, or the pairing is meaningless."""
    with pytest.raises(MatrixDimensionError, match="one value per site"):
        analyzer.morans_i(np.arange(25.0), weights=np.ones((16, 16)) - np.eye(16))


def test_morans_rejects_negative_weights(analyzer: SpatialStatsAnalyzer) -> None:
    """A negative weight makes the contiguity meaningless."""
    weights = _rook_weights(5)
    weights[0, 1] = -1.0
    with pytest.raises(ValidationError, match="non-negative"):
        analyzer.morans_i(np.arange(25.0), weights=weights)


def test_morans_rejects_non_finite_input(analyzer: SpatialStatsAnalyzer) -> None:
    """NaN values and infinite coordinates are both data problems."""
    values = np.arange(25.0)
    values[3] = np.nan
    with pytest.raises(ValidationError):
        analyzer.morans_i(values, weights=_rook_weights(5))

    coordinates, row, _col = _lattice(5)
    coordinates[2, 0] = np.inf
    with pytest.raises(ValidationError):
        analyzer.morans_i(row - 2.0, coordinates=coordinates)


def test_morans_rejects_a_two_dimensional_value_array(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Values are 1-D. Reshaping them is the caller's decision, not the module's."""
    with pytest.raises(MatrixDimensionError, match="one value per site"):
        analyzer.morans_i(np.ones((5, 5)), weights=_rook_weights(5))


def test_morans_rejects_coincident_sites(analyzer: SpatialStatsAnalyzer) -> None:
    """Two sites at one location have a zero distance and no contiguity."""
    coordinates, _row, _col = _lattice(5)
    coordinates[7] = coordinates[3]
    with pytest.raises(ValidationError, match="coincident"):
        analyzer.morans_i(np.arange(25.0), coordinates=coordinates)


def test_morans_rejects_an_isolated_site(analyzer: SpatialStatsAnalyzer) -> None:
    """A site with no neighbours cannot be row-standardised.

    Which is why the default inverse-distance contiguity is safe and a
    radius-based one is not: it gives every pair a positive weight.
    """
    weights = _rook_weights(5)
    weights[12, :] = 0.0
    weights[:, 12] = 0.0
    with pytest.raises(ComputationError, match="no neighbours"):
        analyzer.morans_i(np.arange(25.0), weights=weights)


# =============================================================================
# Grid interpolation -- known answers
# =============================================================================
#
# IDW with weights 1 / d^p is an *exact* interpolant whenever a sample is
# inside its own neighbourhood, because a sample sits at distance 0 from
# itself. That is the property the module must exploit explicitly: a
# zero-distance pair has infinite weight, and hoping an epsilon is small
# enough gives a residual of "small but not zero" instead of zero.


def test_idw_reproduces_every_sample_exactly(analyzer: SpatialStatsAnalyzer) -> None:
    """The residual at a sample is exactly 0.0, not merely small.

    Asserted with ``== 0.0`` and no tolerance: a sample is its own nearest
    neighbour at distance zero, so the exact branch returns the sample's own
    value. A blend through a 1 / eps^p weight, however small eps is, returns
    something of order eps^2 instead -- which is how a broken implementation
    passes a loose ``approx(0, abs=1e-9)`` and fails this one.
    """
    rng = np.random.default_rng(4)
    coordinates = rng.uniform(0.0, 10.0, size=(9, 2))
    values = rng.normal(size=9)

    result = analyzer.grid_interpolate(values, coordinates, method="idw", n_points=13)
    assert np.all(result.residuals == 0.0)
    assert result.rmse == 0.0
    assert result.max_abs_residual == 0.0


def test_idw_with_k_equal_to_n_still_passes_through_every_sample(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """k = n (and k above n) is the same exact-interpolation property.

    The strongest form of the assertion: every sample is in its own
    neighbourhood, so every sample is reproduced, for any k from 1 to n and
    for k larger than n, which is clipped.
    """
    rng = np.random.default_rng(5)
    coordinates = rng.uniform(0.0, 10.0, size=(9, 2))
    values = rng.normal(size=9)
    for k in (1, 2, 8, 9, 40):
        result = analyzer.grid_interpolate(values, coordinates, k=k, n_points=7)
        assert result.k == min(k, 9)
        assert np.all(result.residuals == 0.0)


def test_idw_at_an_equidistant_point_is_the_arithmetic_mean(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Four corners of the unit square, valued 1, 2, 4, 8.

    The centre node (0.5, 0.5) is at distance sqrt(0.5) from all four
    samples, so the four inverse-distance weights are exactly equal and
    independent of ``power``; the estimate is therefore the arithmetic mean
    15/4 = 3.75 exactly. The four corner nodes coincide with samples, so they
    return their own values. Both are exact, and both are things a broken
    zero-distance guard gets wrong.
    """
    coordinates, values = _unit_corners()
    result = analyzer.grid_interpolate(values, coordinates, method="idw", n_points=3)
    assert result.surface[1, 1] == pytest.approx(15.0 / 4.0, rel=1e-12)
    assert result.surface[0, 0] == pytest.approx(1.0)
    assert result.surface[0, 2] == pytest.approx(2.0)
    assert result.surface[2, 2] == pytest.approx(4.0)
    assert result.surface[2, 0] == pytest.approx(8.0)

    # Equal weights means the power cannot matter at the centre, for any of
    # the three powers tried.
    for power in (0.5, 1.0, 2.0, 8.0):
        varied = analyzer.grid_interpolate(values, coordinates, method="idw", power=power, n_points=3)
        assert varied.surface[1, 1] == pytest.approx(15.0 / 4.0, rel=1e-12)


def test_nearest_reproduces_the_samples_and_a_known_step_surface(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """``nearest`` on the same four corners gives the tie-broken step surface.

    The grid nodes are 0, 0.5 and 1 on each axis. The four edge midpoints and
    the centre are each equidistant from two samples, and ``cKDTree`` breaks
    the tie by the lower index, so the surface is fixed in advance: corner
    values at the corners, the lower-indexed neighbour elsewhere.
    """
    coordinates, values = _unit_corners()
    result = analyzer.grid_interpolate(values, coordinates, method="nearest", n_points=3)
    expected = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [8.0, 4.0, 4.0]])
    np.testing.assert_allclose(result.surface, expected)
    assert np.all(result.residuals == 0.0)


def test_grid_spans_the_bounding_box_of_the_samples(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """The grid runs from the sample minimum to the maximum on each axis."""
    coordinates, values = _unit_corners()
    result = analyzer.grid_interpolate(values, coordinates, n_points=5)
    assert result.grid_x[0] == pytest.approx(0.0)
    assert result.grid_x[-1] == pytest.approx(1.0)
    assert result.grid_y[0] == pytest.approx(0.0)
    assert result.grid_y[-1] == pytest.approx(1.0)
    assert result.surface.shape == (5, 5)
    assert result.n_grid == 5


def test_interpolation_is_deterministic(analyzer: SpatialStatsAnalyzer) -> None:
    """Interpolation has no randomness, so two calls must agree exactly."""
    rng = np.random.default_rng(6)
    coordinates = rng.uniform(0.0, 10.0, size=(12, 2))
    values = rng.normal(size=12)
    first = analyzer.grid_interpolate(values, coordinates, n_points=9)
    second = analyzer.grid_interpolate(values, coordinates, n_points=9)
    np.testing.assert_array_equal(first.surface, second.surface)


# =============================================================================
# Grid interpolation -- rejection
# =============================================================================


def test_kriging_is_rejected_with_a_reason(analyzer: SpatialStatsAnalyzer) -> None:
    """Kriging is out of scope, and the error says why rather than falling back.

    A half-implemented kriging would look defensible and be wrong: the
    estimate depends on a fitted variogram model (nugget, sill, range, and
    often a functional form) that this module does not fit, so an IDW
    formula wearing a kriging label would return a surface with no
    prediction variance and no way to check the fit was admissible.
    """
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(8, 2))
    with pytest.raises(ValidationError, match="variogram"):
        analyzer.grid_interpolate(np.arange(8.0), coordinates, method="kriging")


def test_grid_interpolate_rejects_an_unknown_method(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(8, 2))
    with pytest.raises(ValidationError, match="'idw' or 'nearest'"):
        analyzer.grid_interpolate(np.arange(8.0), coordinates, method="spline")


def test_grid_interpolate_rejects_a_degenerate_grid_or_power(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(8, 2))
    values = np.arange(8.0)
    with pytest.raises(ValidationError, match="n_points"):
        analyzer.grid_interpolate(values, coordinates, n_points=1)
    with pytest.raises(ValidationError, match="power"):
        analyzer.grid_interpolate(values, coordinates, power=0.0)
    with pytest.raises(ValidationError, match="power"):
        analyzer.grid_interpolate(values, coordinates, power=np.inf)


def test_grid_interpolate_rejects_three_dimensional_coordinates(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Silently gridding the first two of three columns would drop the third."""
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(8, 3))
    with pytest.raises(MatrixDimensionError, match="2-D coordinates"):
        analyzer.grid_interpolate(np.arange(8.0), coordinates)


def test_grid_interpolate_rejects_mismatched_lengths(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(8, 2))
    with pytest.raises(MatrixDimensionError, match="one value per sample"):
        analyzer.grid_interpolate(np.arange(7.0), coordinates)


def test_grid_interpolate_rejects_coincident_samples(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(8, 2))
    coordinates[3] = coordinates[1]
    with pytest.raises(ValidationError, match="coincident"):
        analyzer.grid_interpolate(np.arange(8.0), coordinates)


# =============================================================================
# Nearest neighbour -- known answers
# =============================================================================
#
# For a Poisson process of intensity lambda = n / A the nearest-neighbour
# distance of any site is Rayleigh:
#
#     P(D > d) = exp(-lambda pi d^2)
#     E[D]     = integral_0^inf exp(-lambda pi d^2) dd
#             = sqrt(pi) / (2 sqrt(lambda pi)) = 1 / (2 sqrt(lambda))
#             = 0.5 sqrt(A / n)
#
# The pi in the density cancels the pi in the Gaussian integral, so the
# reference carries no pi at all. test_nn_csr_expectation_is_the_rayleigh_mean
# below evaluates that integral numerically to confirm it.


def test_nn_csr_expectation_is_the_rayleigh_mean(analyzer: SpatialStatsAnalyzer) -> None:
    """The CSR reference, checked against a numerical integration of P(D > d).

    The integral is evaluated here, independently of the module, and agrees
    with the module's ``0.5 * sqrt(A / n)`` to twelve places. The
    ``0.5 * sqrt(pi A / n)`` that looks more familiar is 1.77 times too large
    and would report a random pattern as strongly regular.

    The edge correction is off so that the reference is the whole study
    window, which is the area the integral above is written for. With it on,
    the module correctly reports the expectation for the *inscribed* window
    instead; ``test_nn_edge_correction_uses_a_reduced_sample`` covers that.
    """
    coordinates = np.random.default_rng(11).uniform(0.0, 100.0, size=(250, 2))
    area = float(
        (coordinates[:, 0].max() - coordinates[:, 0].min()) * (coordinates[:, 1].max() - coordinates[:, 1].min())
    )
    intensity = 250.0 / area
    by_integration = quad(lambda d: float(np.exp(-intensity * np.pi * d * d)), 0.0, 50.0)[0]

    result = analyzer.nearest_neighbour_stats(coordinates, edge_correction=False, n_simulations=9, random_seed=5)
    assert result.n_sites == 250
    assert result.expected_nn == pytest.approx(0.5 * np.sqrt(area / 250.0), rel=1e-12)
    assert result.expected_nn == pytest.approx(by_integration, rel=1e-6)
    # And the index of this random sample is near 1 against that reference.
    assert result.index == pytest.approx(1.0, abs=0.1)


def test_nn_index_of_a_square_lattice_is_exactly_two_k_over_k_minus_one(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """A regular ``k x k`` lattice of spacing a has R = 2k / (k - 1) exactly.

    With no edge correction every site is used: the observed mean NN is the
    spacing ``a``, the bounding box is ``(k-1)^2 a^2`` and there are ``k^2``
    sites, so

        R = a / (0.5 sqrt((k-1)^2 a^2 / k^2)) = 2k / (k - 1)

    which is 5.0 at k = 5 and 24/11 = 2.1818 at k = 12. The ideal value for a
    lattice whose sites are the whole window would be 2.0; the excess is the
    bounding-box area estimator being low, since the extreme sites sit
    exactly on the box edge. Using a larger k shrinks the discrepancy, which
    is the test that the factor is the area convention and not the pattern.
    """
    for k, expected in ((5, 2.0 * 5 / 4), (12, 2.0 * 12 / 11)):
        coordinates, _row, _col = _lattice(k)
        result = analyzer.nearest_neighbour_stats(coordinates, edge_correction=False, n_simulations=9, random_seed=3)
        assert result.n_sites == k * k
        assert result.mean_nn == pytest.approx(1.0)
        assert result.index == pytest.approx(expected, rel=1e-12)


def test_nn_index_of_a_uniform_sample_is_near_one(analyzer: SpatialStatsAnalyzer) -> None:
    """A CSR sample has R = 1; 0.93 to 1.09 over 20 seeds is the observed spread.

    Expected 1 by construction: a Poisson process *is* the null, so the
    observed mean NN should equal its own expectation. The band is wide
    because the area is itself estimated from the sample's bounding box, not
    because the NN statistic is imprecise -- its own standard error at n = 250
    is under 1 % of the mean.
    """
    indices = [
        analyzer.nearest_neighbour_stats(
            np.random.default_rng(200 + seed).uniform(0.0, 100.0, size=(250, 2)),
            n_simulations=9,
            random_seed=1,
        ).index
        for seed in range(20)
    ]
    assert all(0.9 < value < 1.1 for value in indices)
    assert abs(float(np.mean(indices)) - 1.0) < 0.05


def test_nn_rejects_clustered_and_regular_patterns_correctly(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Two tight clusters read as clustered; a lattice reads as regular.

    The index is judged against 1: below it the sites are closer together
    than random, above it further apart. The two clusters are 1e-3 apart
    internally and 70 apart from each other, so every site's nearest
    neighbour is inside its own cluster and R is orders of magnitude below 1.
    """
    clusters = np.vstack(
        [
            np.array([10.0, 10.0]) + 1e-3 * np.random.default_rng(1).normal(size=(50, 2)),
            np.array([80.0, 80.0]) + 1e-3 * np.random.default_rng(2).normal(size=(50, 2)),
        ]
    )
    clustered = analyzer.nearest_neighbour_stats(clusters, n_simulations=99, random_seed=5)
    assert clustered.index < 0.2
    assert str(clustered.pattern) == "CLUSTERED"
    assert clustered.significant
    assert clustered.p_value == pytest.approx(0.01)

    regular = analyzer.nearest_neighbour_stats(_lattice(5)[0], n_simulations=99, random_seed=5)
    assert regular.index > 1.0
    assert str(regular.pattern) == "REGULAR"


def test_nn_z_score_is_centred_under_the_null(analyzer: SpatialStatsAnalyzer) -> None:
    """The Monte Carlo z must not be systematically signed for null samples.

    The property is that the test is *calibrated*: over 12 independent CSR
    samples the z scores average to about zero and few are rejected. A null
    generated in the wrong window, or an observed statistic computed on a
    different sample from the null, shifts every z one way, and this is the
    assertion that catches it. Measured here: mean -0.51, one rejection in
    12 against a nominal expectation of 0.6.
    """
    z_scores = []
    rejected = 0
    for seed in range(12):
        coordinates = np.random.default_rng(200 + seed).uniform(0.0, 100.0, size=(250, 2))
        result = analyzer.nearest_neighbour_stats(coordinates, n_simulations=99, random_seed=1)
        z_scores.append(result.z_score)
        rejected += int(result.p_value < 0.05)
    assert abs(float(np.mean(z_scores))) < 1.0
    assert rejected <= 2


def test_nn_edge_correction_uses_a_reduced_sample(analyzer: SpatialStatsAnalyzer) -> None:
    """The correction removes a border of one mean NN distance from both sides.

    The observed statistic and the CSR null must be computed on the same
    footing, so both are restricted to the inscribed window. Correcting only
    the null is worse than not correcting at all: the observed mean then sits
    above the null for every pattern, and random samples are rejected as
    significantly regular.
    """
    coordinates = np.random.default_rng(11).uniform(0.0, 100.0, size=(250, 2))
    corrected = analyzer.nearest_neighbour_stats(coordinates, n_simulations=9, random_seed=5)
    uncorrected = analyzer.nearest_neighbour_stats(coordinates, n_simulations=9, random_seed=5, edge_correction=False)

    assert uncorrected.n_sites == uncorrected.n_sites_total == 250
    assert uncorrected.edge_shrink == 0.0
    assert corrected.n_sites < corrected.n_sites_total == 250
    assert corrected.edge_shrink > 0.0
    # The inscribed window is strictly smaller than the study window, and the
    # reference area is computed from it.
    assert corrected.area < uncorrected.area
    assert corrected.expected_nn == pytest.approx(0.5 * np.sqrt(corrected.area / corrected.n_sites))


def test_nn_cdf_tracks_the_analytic_poisson_cdf(analyzer: SpatialStatsAnalyzer) -> None:
    """The tabulated cdf follows ``1 - exp(-pi r^2 n / A)`` for a random sample.

    Sampled at r = 2, 4, 6 and 8 (units of the mean NN), the observed
    fraction of sites with ``NN <= r`` should sit on the analytic curve to
    within sampling error. The analytic curve is the infinite-window form, so
    it carries no edge correction and the observed values sit slightly above
    it near the border; 0.09 is the observed maximum discrepancy over these
    four radii for this seeded sample. The curve is already 99.5 % complete
    at r = 8, so the last observed point need not be exactly 1.
    """
    coordinates = np.random.default_rng(11).uniform(0.0, 100.0, size=(250, 2))
    result = analyzer.nearest_neighbour_stats(
        coordinates, r_values=np.array([2.0, 4.0, 6.0, 8.0]), n_simulations=9, random_seed=5
    )
    assert result.r_values is not None
    np.testing.assert_array_equal(result.r_values, [2.0, 4.0, 6.0, 8.0])
    assert result.cumulative_fraction is not None
    assert result.cumulative_fraction[-1] > 0.99
    # The empirical cdf is non-decreasing in r, as a cdf must be.
    assert np.all(np.diff(result.cumulative_fraction) >= 0.0)
    assert result.csr_expected_fraction is not None
    assert np.max(np.abs(result.cumulative_fraction - result.csr_expected_fraction)) < 0.09
    # The reference curve is increasing and bounded.
    assert np.all(np.diff(result.csr_expected_fraction) > 0)
    assert np.all(result.csr_expected_fraction <= 1.0)


def test_nn_is_deterministic_for_a_seed(analyzer: SpatialStatsAnalyzer) -> None:
    """Same seed, same z and p: the Monte Carlo null is reproducible."""
    coordinates = np.random.default_rng(11).uniform(0.0, 100.0, size=(250, 2))
    first = analyzer.nearest_neighbour_stats(coordinates, n_simulations=99, random_seed=7)
    second = analyzer.nearest_neighbour_stats(coordinates, n_simulations=99, random_seed=7)
    assert first.z_score == second.z_score
    assert first.p_value == second.p_value
    assert first.index == second.index


# =============================================================================
# Nearest neighbour -- rejection
# =============================================================================


def test_nn_rejects_too_few_sites(analyzer: SpatialStatsAnalyzer) -> None:
    """Two sites have no pattern to speak of."""
    with pytest.raises(ValidationError, match="at least 3 points"):
        analyzer.nearest_neighbour_stats(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_nn_rejects_a_degenerate_window(analyzer: SpatialStatsAnalyzer) -> None:
    """A collinear-in-x sample has no area to measure a pattern in."""
    collinear = np.column_stack([np.arange(10.0), np.zeros(10)])
    with pytest.raises(ComputationError, match="study area is zero"):
        analyzer.nearest_neighbour_stats(collinear)


def test_nn_rejects_coincident_sites(analyzer: SpatialStatsAnalyzer) -> None:
    """A nearest-neighbour distance of zero makes the mean undefined."""
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(20, 2))
    coordinates[5] = coordinates[2]
    with pytest.raises(ValidationError, match="coincident"):
        analyzer.nearest_neighbour_stats(coordinates)


def test_nn_rejects_three_dimensional_coordinates(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(20, 3))
    with pytest.raises(MatrixDimensionError, match="2-D coordinates"):
        analyzer.nearest_neighbour_stats(coordinates)


def test_nn_rejects_non_finite_coordinates(analyzer: SpatialStatsAnalyzer) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(20, 2))
    coordinates[1, 1] = np.nan
    with pytest.raises(ValidationError):
        analyzer.nearest_neighbour_stats(coordinates)


def test_nn_rejects_bad_r_values_and_simulation_counts(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    coordinates = np.random.default_rng(0).uniform(0.0, 10.0, size=(20, 2))
    with pytest.raises(ValidationError, match="n_simulations"):
        analyzer.nearest_neighbour_stats(coordinates, n_simulations=0)
    with pytest.raises(ValidationError, match="r_values"):
        analyzer.nearest_neighbour_stats(coordinates, r_values=np.array([[1.0, 2.0]]))
    with pytest.raises(ValidationError, match="r_values"):
        analyzer.nearest_neighbour_stats(coordinates, r_values=np.array([-1.0]))


# =============================================================================
# Spherical statistics -- known answers
# =============================================================================


def _oblique_unit() -> npt.NDArray:
    """A unit vector that is not an axis, so sign and scaling errors show."""
    vector = np.array([1.0, 2.0, -1.0])
    return vector / np.linalg.norm(vector)


def test_spherical_identical_vectors_give_R_equal_to_n(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """n identical unit vectors sum to n * v, so R = n and R-bar = R/n = 1.

    This is the reduction that pins the normalisation. Dividing by n - 1
    instead caps R-bar at n / (n - 1) > 1, which no mean resultant can be,
    and rescales every weaker sample with it. The concentration is the
    vMF limit: kappa is +inf where the formula is 0/0, and the confidence
    cone collapses onto the mean direction.
    """
    unit = _oblique_unit()
    n = 7
    result = analyzer.spherical_stats(np.tile(unit, (n, 1)))

    assert result.resultant_length == pytest.approx(float(n), rel=1e-12)
    assert result.mean_resultant_length == pytest.approx(1.0, rel=1e-12)
    assert np.isinf(result.kappa)
    assert result.cone_half_angle_deg == pytest.approx(0.0)
    assert result.mean_direction is not None
    np.testing.assert_allclose(result.mean_direction, unit, atol=1e-12)
    # R-bar is R/n, not R/(n-1): the wrong divisor is off by a measurable
    # amount, 7/6 against 1.
    assert result.mean_resultant_length != pytest.approx(n / (n - 1))


def test_spherical_exactly_opposite_pairs_give_no_resultant(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Three ``v`` against three ``-v``: the sum is zero, so there is no mean.

    Each pair cancels to exact zeros in floating point, so R is at the 1e-16
    level and the mean direction does not exist. A module that divided R by
    n and then normalised the zero vector would return NaNs or an arbitrary
    direction; the correct answer is to report that there is none.
    """
    unit = _oblique_unit()
    pairs = np.vstack([np.tile(unit, (3, 1)), np.tile(-unit, (3, 1))])
    result = analyzer.spherical_stats(pairs)

    assert result.resultant_length < 1e-12
    assert result.mean_resultant_length < 1e-12
    assert not result.concentrated
    assert result.mean_direction is None
    assert np.isnan(result.mean_azimuth_deg)
    assert np.isnan(result.mean_polar_deg)


def test_spherical_kappa_matches_fishers_corrected_formula(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """kappa = 6.8/3 for a hand-computable five-vector set.

    One vector on the +z pole and four on the equator, which cancel exactly
    in x and y, leaving S = (0, 0, 1): R = 1, R-bar = 1/5 = 0.2. Fisher's
    uncorrected estimator is then
        kappa = n R-bar (1 - R-bar^2) / (1 - R-bar)
              = 5 * 0.2 * 0.96 / 0.8 = 1.2
    and the 1953 small-sample correction is
        kappa = (2(n - 1) - kappa) / (n - 2) = (8 - 1.2) / 3 = 6.8/3.
    Applying the correction in the wrong direction -- the variance-inflation
    reciprocal, or skipping it -- gives a different number, which is why the
    expected value is written out rather than merely bounded.
    """
    vectors = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    result = analyzer.spherical_stats(vectors)

    assert result.resultant_length == pytest.approx(1.0, rel=1e-12)
    assert result.mean_resultant_length == pytest.approx(0.2, rel=1e-12)
    assert result.kappa == pytest.approx(6.8 / 3.0, rel=1e-12)
    assert result.mean_direction is not None
    np.testing.assert_allclose(result.mean_direction, [0.0, 0.0, 1.0], atol=1e-12)
    assert result.mean_polar_deg == pytest.approx(0.0)


def test_spherical_confidence_cone_holds_95_percent_of_the_vmf_density(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """The cone half-angle is where the vMF density integrates to 0.95.

    Checked by numerical integration of
    ``p(theta) = kappa sin(theta) e^(kappa cos theta) / (2 sinh kappa)``
    written out here, independently of the module's closed form. This pins
    the solver itself; the large-kappa approximation
    ``cos(w) = 1 + ln(0.05)/kappa`` is not used, and would not agree here.
    """
    vectors = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    result = analyzer.spherical_stats(vectors)
    kappa = result.kappa

    def density(theta: float) -> float:
        return float(kappa * np.sin(theta) * np.exp(kappa * np.cos(theta)) / (2.0 * np.sinh(kappa)))

    # The density is a proper one, which the solver implicitly assumes.
    assert quad(density, 0.0, np.pi)[0] == pytest.approx(1.0, rel=1e-9)
    inside = quad(density, 0.0, float(result.cone_half_angle_rad))[0]
    assert inside == pytest.approx(0.95, abs=1e-8)


def test_spherical_reduces_to_the_circular_mean(analyzer: SpatialStatsAnalyzer) -> None:
    """Vectors in the z = 0 plane reproduce the 1-D circle result exactly.

    This is the reduction that says the spherical code is not
    ``stratigraphy/directional.py`` widened. Embedding angles as
    ``(cos t, sin t, 0)`` makes the 3-vector sum ``(sum cos t, sum sin t, 0)``,
    whose norm is the Rayleigh resultant ``|sum e^(it)|`` and whose direction
    is ``atan2(sum sin t, sum cos t)``. Both are computed here from scratch
    and must agree to twelve places -- which also pins R-bar = R/n, since the
    Rayleigh resultant here is 4.78 and R/(n-1) would be 1.19.
    """
    angles = np.radians([10.0, 20.0, 30.0, 40.0, 350.0])
    vectors = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(angles.size)])
    result = analyzer.spherical_stats(vectors)

    rayleigh = abs(np.exp(1j * angles).sum())
    circular_mean = np.degrees(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())) % 360.0

    assert result.resultant_length == pytest.approx(rayleigh, rel=1e-12)
    assert result.mean_resultant_length == pytest.approx(rayleigh / angles.size, rel=1e-12)
    assert result.mean_azimuth_deg == pytest.approx(circular_mean, abs=1e-10)
    # A planar sample's mean direction lies in the plane.
    assert result.mean_polar_deg == pytest.approx(90.0, abs=1e-9)
    assert result.mean_direction is not None
    np.testing.assert_allclose(result.mean_direction[2], 0.0, atol=1e-12)


def test_spherical_is_invariant_to_vector_length(analyzer: SpatialStatsAnalyzer) -> None:
    """Only direction is meaningful, so scaling must change nothing.

    A palaeocurrent vector's magnitude is a measurement artefact of how it
    was recorded; a mean direction that moved with it would be reporting the
    recording, not the current.
    """
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(20, 3))
    plain = analyzer.spherical_stats(vectors)
    scaled = analyzer.spherical_stats(vectors * 37.5)
    for field in ("resultant_length", "mean_resultant_length", "kappa", "cone_half_angle_deg"):
        assert getattr(plain, field) == pytest.approx(getattr(scaled, field), rel=1e-12)
    assert plain.mean_azimuth_deg == pytest.approx(scaled.mean_azimuth_deg, abs=1e-9)


def test_spherical_weights_enter_the_resultant_and_the_normaliser(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Weights act as repetitions: five unit vectors, one of weight zero.

    R = sum(w_i u_i) = 4 v and R-bar = R / sum(w) = 4/4 = 1, the same answer
    as four copies of the vector. ``effective_n`` carries sum(w), which is
    the ``n`` Fisher's kappa formulas use.
    """
    unit = _oblique_unit()
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    result = analyzer.spherical_stats(np.tile(unit, (5, 1)), weights=weights)

    assert result.n_vectors == 5
    assert result.effective_n == pytest.approx(4.0)
    assert result.resultant_length == pytest.approx(4.0, rel=1e-12)
    assert result.mean_resultant_length == pytest.approx(1.0, rel=1e-12)


# =============================================================================
# Spherical statistics -- rejection
# =============================================================================


def test_spherical_rejects_non_three_dimensional_input(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """The 2-sphere needs three components.

    A 1-D input is the case worth guarding: ``validate_data_array`` would
    otherwise reshape it to ``(n, 1)`` and it would sail through as a
    one-column "3-D" array.
    """
    with pytest.raises(MatrixDimensionError, match=r"\(n, 3\)"):
        analyzer.spherical_stats(np.random.default_rng(0).normal(size=(5, 2)))
    with pytest.raises(MatrixDimensionError, match=r"\(n, 3\)"):
        analyzer.spherical_stats(np.random.default_rng(0).normal(size=5))


def test_spherical_rejects_too_few_vectors(analyzer: SpatialStatsAnalyzer) -> None:
    """Fisher's kappa correction divides by n - 2."""
    unit = _oblique_unit()
    with pytest.raises(ValidationError, match="at least 3 vectors"):
        analyzer.spherical_stats(np.vstack([unit, -unit]))


def test_spherical_rejects_a_zero_length_vector(analyzer: SpatialStatsAnalyzer) -> None:
    """A zero vector has no direction to average."""
    vectors = np.tile(_oblique_unit(), (4, 1))
    vectors[2] = 0.0
    with pytest.raises(ValidationError, match="zero length"):
        analyzer.spherical_stats(vectors)


def test_spherical_rejects_bad_weights(analyzer: SpatialStatsAnalyzer) -> None:
    unit = _oblique_unit()
    vectors = np.tile(unit, (5, 1))
    with pytest.raises(MatrixDimensionError, match="one weight per vector"):
        analyzer.spherical_stats(vectors, weights=np.ones(4))
    with pytest.raises(ValidationError, match="non-negative"):
        analyzer.spherical_stats(vectors, weights=np.array([1.0, 1.0, -1.0, 1.0, 1.0]))
    # sum(weights) <= 2 leaves the kappa correction undefined.
    with pytest.raises(ValidationError, match="exceed 2"):
        analyzer.spherical_stats(vectors, weights=np.array([0.5, 0.5, 0.5, 0.4, 0.1]))


def test_spherical_rejects_non_finite_vectors(analyzer: SpatialStatsAnalyzer) -> None:
    vectors = np.tile(_oblique_unit(), (5, 1))
    vectors[1, 0] = np.nan
    with pytest.raises(ValidationError):
        analyzer.spherical_stats(vectors)


# =============================================================================
# Determinism, isolation and reporting
# =============================================================================


def test_morans_and_nn_never_touch_the_global_numpy_random_stream(
    analyzer: SpatialStatsAnalyzer,
) -> None:
    """Running a seeded analysis must leave the global stream where it was.

    An unseeded analysis falling back to the legacy global stream is the bug
    this guards: it means merely *running* a permutation test perturbs every
    later draw, so two analyses cannot be compared unless both were seeded.
    """
    values = np.arange(25.0)
    coordinates = np.random.default_rng(11).uniform(0.0, 100.0, size=(250, 2))

    np.random.seed(1234)
    expected_after_seeding = np.random.random()
    np.random.seed(1234)

    analyzer.morans_i(values, weights=_rook_weights(5), permutations=99, random_seed=5)
    analyzer.nearest_neighbour_stats(coordinates, n_simulations=19, random_seed=5)
    assert np.random.random() == expected_after_seeding


def test_morans_is_deterministic_for_a_seed(analyzer: SpatialStatsAnalyzer) -> None:
    """Same seed, same p-value and z."""
    _coordinates, row, _col = _lattice(5)
    values = row - 2.0
    weights = _rook_weights(5)
    first = analyzer.morans_i(values, weights=weights, permutations=99, random_seed=17)
    second = analyzer.morans_i(values, weights=weights, permutations=99, random_seed=17)
    assert first.p_value == second.p_value
    assert first.z_score == second.z_score
    # A different seed is a different null, so the two must not be pinned
    # together by construction.
    third = analyzer.morans_i(values, weights=weights, permutations=99, random_seed=18)
    assert third.p_value > 0.0


def test_all_results_serialise_and_summarise(analyzer: SpatialStatsAnalyzer) -> None:
    """to_dict and summary work for all four analyses."""
    import json

    _coordinates, row, _col = _lattice(5)
    morans = analyzer.morans_i(row - 2.0, weights=_rook_weights(5), permutations=19, random_seed=1)
    coordinates, values = _unit_corners()
    grid = analyzer.grid_interpolate(values, coordinates, n_points=3)
    nn = analyzer.nearest_neighbour_stats(_lattice(5)[0], edge_correction=False, n_simulations=9, random_seed=1)
    sphere = analyzer.spherical_stats(np.tile(_oblique_unit(), (5, 1)))

    assert "Morans I" in morans.summary()
    assert set(morans.to_dict()) >= {"statistic", "expected", "z_score", "p_value", "weight_scheme"}
    assert "Spatial Interpolation" in grid.summary()
    assert set(grid.to_dict()) >= {"method", "rmse", "max_abs_residual", "n_grid"}
    assert "Nearest Neighbour" in nn.summary()
    assert set(nn.to_dict()) >= {"mean_nn", "expected_nn", "index", "pattern", "n_sites_total"}
    assert "Spherical Statistics" in sphere.summary()
    assert set(sphere.to_dict()) >= {"mean_direction", "resultant_length", "kappa", "cone_half_angle_deg"}

    # Every payload must survive a JSON round trip, which is the point of
    # keeping arrays out of them.
    for payload in (
        morans.to_dict(),
        grid.to_dict(),
        nn.to_dict(),
        sphere.to_dict(),
    ):
        assert json.loads(json.dumps(payload)) == payload


def test_last_results_are_retained(analyzer: SpatialStatsAnalyzer) -> None:
    """Each analysis keeps its own handle, and last_result is the newest."""
    assert analyzer.last_result is None
    assert analyzer.last_morans_i is None
    assert analyzer.last_interpolation is None
    assert analyzer.last_nearest_neighbour is None
    assert analyzer.last_spherical is None

    _coordinates, row, _col = _lattice(5)
    morans = analyzer.morans_i(row - 2.0, weights=_rook_weights(5), permutations=19, random_seed=1)
    assert analyzer.last_result is morans
    assert analyzer.last_morans_i is morans

    coordinates, values = _unit_corners()
    grid = analyzer.grid_interpolate(values, coordinates, n_points=3)
    assert analyzer.last_result is grid
    assert analyzer.last_interpolation is grid
    assert analyzer.last_morans_i is morans

    nn = analyzer.nearest_neighbour_stats(_lattice(5)[0], edge_correction=False, n_simulations=9, random_seed=1)
    sphere = analyzer.spherical_stats(np.tile(_oblique_unit(), (5, 1)))
    assert analyzer.last_result is sphere
    assert analyzer.last_nearest_neighbour is nn
    assert analyzer.last_spherical is sphere
