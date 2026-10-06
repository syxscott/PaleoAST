# =============================================================================
# FILE: tests/stats/test_mantel.py
# =============================================================================
"""
Tests for the Mantel and partial Mantel tests.

The properties asserted here are chosen so that a broken implementation
fails rather than merely returning a different number:

  * identical inputs must give r = 1 exactly, not "close to" 1;
  * the residual-based partial correlation must equal the textbook
    partial-correlation formula to machine precision -- that identity is
    what makes the residual form (Guillot & Rousset 2013) a correct
    implementation of a textbook quantity, and it is checkable without
    trusting this module at all;
  * the p-values under the null must be uniform. A permutation loop that
    shuffles the wrong array still returns a plausible-looking number, but
    its p-values are not uniform, so the calibration check is the one that
    actually has teeth.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as sp_stats
from scipy.spatial import distance

from stats.mantel import MantelAnalyzer
from utils.exceptions import (
    ComputationError,
    DataValidationError,
    MatrixDimensionError,
    ValidationError,
)


@pytest.fixture
def analyzer() -> MantelAnalyzer:
    """A fresh analyzer."""
    return MantelAnalyzer()


def _triangle(matrix: npt.NDArray) -> npt.NDArray:
    """Distinct pairs from a square matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


# ---------------------------------------------------------------------------
# Observed statistic
# ---------------------------------------------------------------------------


def test_identical_inputs_give_r_exactly_one(analyzer: MantelAnalyzer) -> None:
    """The same data under the same metric is perfectly self-correlated."""
    data = np.random.default_rng(0).normal(size=(25, 4))
    result = analyzer.analyze(data, data, n_permutations=199, random_seed=1)
    assert result.statistic == pytest.approx(1.0, abs=1e-12)
    assert result.n_objects == 25
    assert result.n_pairs == 25 * 24 // 2
    assert result.p_value <= 1.0 / 200 + 1e-12


def test_different_metrics_on_same_data_are_not_one(analyzer: MantelAnalyzer) -> None:
    """Euclidean and Manhattan on the same points are related, not equal."""
    data = np.random.default_rng(0).normal(size=(25, 4))
    result = analyzer.analyze(
        data,
        data,
        metric_a="euclidean",
        metric_b="manhattan",
        n_permutations=99,
        random_seed=1,
    )
    assert 0.5 < result.statistic < 1.0
    assert result.metric_a == "euclidean"
    assert result.metric_b == "manhattan"


def test_spearman_option_differs_from_pearson(analyzer: MantelAnalyzer) -> None:
    """A non-linear relationship separates the two coefficients."""
    rng = np.random.default_rng(2)
    a = rng.uniform(size=(30, 1))
    b = a**3 + rng.normal(0, 0.01, (30, 1))
    pearson = analyzer.analyze(a, b, n_permutations=99, random_seed=1)
    spearman = analyzer.analyze(a, b, correlation="spearman", n_permutations=99, random_seed=1)
    assert pearson.statistic < spearman.statistic


def test_monte_carlo_z_is_reported(analyzer: MantelAnalyzer) -> None:
    """MC-Z comes from Legendre, Fortin & Borcard (2015)."""
    data = np.random.default_rng(0).normal(size=(25, 4))
    result = analyzer.analyze(data, data, n_permutations=199, random_seed=1)
    assert np.isfinite(result.mc_z)
    assert result.mc_z > 5


# ---------------------------------------------------------------------------
# Calibration -- the check with teeth
# ---------------------------------------------------------------------------


def test_p_values_are_uniform_under_the_null(analyzer: MantelAnalyzer) -> None:
    """Independent data must not produce significant results more than
    alpha of the time, and the p-values must look uniform.

    A permutation loop that reshuffles the wrong array, or that compares
    against the wrong tail, still returns a number in [0, 1]. It fails
    here.
    """
    alpha = 0.05
    reps = 40
    p_values = []
    for seed in range(reps):
        a = np.random.default_rng(1000 + seed).normal(size=(18, 3))
        b = np.random.default_rng(5000 + seed).normal(size=(18, 3))
        p_values.append(analyzer.analyze(a, b, n_permutations=99, random_seed=seed).p_value)
    arr = np.asarray(p_values)
    false_positive_rate = float((arr < alpha).mean())
    # Binomial 95% interval around alpha is wide at 40 reps, so allow slack
    # while still catching a p-value that is systematically too small.
    assert false_positive_rate <= 0.20, f"false positive rate {false_positive_rate} is far above alpha={alpha}"
    assert sp_stats.kstest(arr, "uniform").pvalue > 0.001
    assert arr.min() >= 1.0 / 100 - 1e-12


# ---------------------------------------------------------------------------
# Partial Mantel
# ---------------------------------------------------------------------------


def test_partial_mantel_equals_textbook_formula(analyzer: MantelAnalyzer) -> None:
    """The residual form must reproduce (r_ab - r_ac r_bc) / sqrt(...).

    Computed here with scipy directly, so this asserts the identity
    rather than agreeing with itself.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(size=(25, 4))
    b = rng.normal(size=(25, 3)) * 0.5 + a[:, :3]
    control = rng.normal(size=(25, 2))

    result = analyzer.analyze_partial(a, b, control, n_permutations=99, random_seed=2)

    da = distance.squareform(distance.pdist(a))
    db = distance.squareform(distance.pdist(b))
    dc = distance.squareform(distance.pdist(control))
    va, vb, vc = _triangle(da), _triangle(db), _triangle(dc)
    r_ab = sp_stats.pearsonr(va, vb).statistic
    r_ac = sp_stats.pearsonr(va, vc).statistic
    r_bc = sp_stats.pearsonr(vb, vc).statistic
    textbook = (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac**2) * (1 - r_bc**2))

    assert result.statistic == pytest.approx(textbook, abs=1e-10)
    assert result.r_a_control == pytest.approx(r_ac, abs=1e-10)
    assert result.r_b_control == pytest.approx(r_bc, abs=1e-10)
    assert result.scheme == "residual"


def test_partial_mantel_of_identical_matrices_is_one(analyzer: MantelAnalyzer) -> None:
    """Controlling cannot remove variance that both responses share."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(25, 4))
    control = rng.normal(size=(25, 2))
    result = analyzer.analyze_partial(a, a, control, n_permutations=99, random_seed=5)
    assert result.statistic == pytest.approx(1.0, abs=1e-10)


def test_partial_mantel_rejects_a_control_that_explains_everything(
    analyzer: MantelAnalyzer,
) -> None:
    """A fully explanatory control leaves floating-point noise, not a number.

    Correlating that noise would return a finite value -- 0.0075 in the
    case that prompted this check -- which reads as a real weak
    association.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(size=(25, 4))
    b = rng.normal(size=(25, 3)) * 0.5 + a[:, :3]
    with pytest.raises(ComputationError, match="explains"):
        analyzer.analyze_partial(a, b, a, n_permutations=99, random_seed=6)


def test_partial_mantel_allows_a_partially_explanatory_control(
    analyzer: MantelAnalyzer,
) -> None:
    """Explaining most but not all of a response leaves it defined."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(25, 4))
    b = rng.normal(size=(25, 3)) * 0.5 + a[:, :3]
    control = a[:, :1] * 0.9 + np.random.default_rng(11).normal(size=(25, 1))
    result = analyzer.analyze_partial(a, b, control, n_permutations=99, random_seed=12)
    assert np.isfinite(result.statistic)


# ---------------------------------------------------------------------------
# Entry points and reproducibility
# ---------------------------------------------------------------------------


def test_analyze_from_matrices_matches_analyze(analyzer: MantelAnalyzer) -> None:
    """The matrix entry point must agree with the raw-data one."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(22, 4))
    b = rng.normal(size=(22, 3))
    from_raw = analyzer.analyze(a, b, n_permutations=99, random_seed=4)
    da = distance.squareform(distance.pdist(a))
    db = distance.squareform(distance.pdist(b))
    from_matrix = analyzer.analyze_from_matrices(da, db, n_permutations=99, random_seed=4)
    assert from_matrix.statistic == pytest.approx(from_raw.statistic, abs=1e-12)


def test_seeded_runs_are_reproducible(analyzer: MantelAnalyzer) -> None:
    """Same seed, same p-value."""
    rng = np.random.default_rng(9)
    a = rng.normal(size=(20, 4))
    b = rng.normal(size=(20, 3))
    first = analyzer.analyze(a, b, n_permutations=99, random_seed=7)
    second = analyzer.analyze(a, b, n_permutations=99, random_seed=7)
    assert first.p_value == second.p_value
    assert first.statistic == second.statistic


def test_unseeded_run_does_not_touch_the_global_stream(analyzer: MantelAnalyzer) -> None:
    """An unseeded Mantel must not advance the caller's random state."""
    rng = np.random.default_rng(9)
    a = rng.normal(size=(18, 3))
    b = rng.normal(size=(18, 3))
    np.random.seed(4242)
    control = np.random.rand(4)
    np.random.seed(4242)
    with pytest.warns(RuntimeWarning):
        analyzer.analyze(a, b, n_permutations=49)
    np.testing.assert_allclose(np.random.rand(4), control)


def test_last_result_is_retained(analyzer: MantelAnalyzer) -> None:
    """The analyzer remembers its most recent run."""
    assert analyzer.last_result is None
    data = np.random.default_rng(0).normal(size=(20, 3))
    result = analyzer.analyze(data, data, n_permutations=49, random_seed=1)
    assert analyzer.last_result is result


# ---------------------------------------------------------------------------
# Rejection of input that cannot produce an answer
# ---------------------------------------------------------------------------


def test_constant_matrix_is_rejected(analyzer: MantelAnalyzer) -> None:
    """Zero variance makes the correlation undefined; say so, do not invent."""
    with pytest.raises(ComputationError):
        analyzer.analyze(np.zeros((10, 3)), np.ones((10, 3)), n_permutations=19, random_seed=1)


def test_too_few_objects_is_rejected(analyzer: MantelAnalyzer) -> None:
    """Fewer than 4 objects gives fewer than 6 distinct pairs."""
    with pytest.raises((DataValidationError, MatrixDimensionError, ValidationError)):
        analyzer.analyze(np.ones((3, 2)), np.ones((3, 2)), n_permutations=19)


def test_mismatched_object_counts_are_rejected(analyzer: MantelAnalyzer) -> None:
    """The two matrices must describe the same objects."""
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze(
            np.random.default_rng(0).normal(size=(20, 3)),
            np.random.default_rng(1).normal(size=(25, 3)),
            n_permutations=19,
        )


def test_invalid_correlation_is_rejected(analyzer: MantelAnalyzer) -> None:
    """Only the two implemented coefficients are accepted."""
    data = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValidationError):
        analyzer.analyze(data, data, correlation="kendall", n_permutations=19)


def test_jitter_scheme_needs_raw_coordinates(analyzer: MantelAnalyzer) -> None:
    """A finished distance matrix no longer carries the jitter."""
    data = np.random.default_rng(0).normal(size=(20, 3))
    matrix = distance.squareform(distance.pdist(data))
    with pytest.raises((MatrixDimensionError, ValidationError)):
        analyzer.analyze_from_matrices(matrix, matrix, scheme="jm")


def test_partial_mantel_rejects_mismatched_object_counts(
    analyzer: MantelAnalyzer,
) -> None:
    """All three inputs must share the object count."""
    rng = np.random.default_rng(0)
    with pytest.raises(MatrixDimensionError):
        analyzer.analyze_partial(
            rng.normal(size=(20, 3)),
            rng.normal(size=(20, 3)),
            rng.normal(size=(25, 1)),
            n_permutations=19,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_result_serialises(analyzer: MantelAnalyzer) -> None:
    """to_dict and summary must both work and stay JSON-friendly."""
    data = np.random.default_rng(0).normal(size=(20, 3))
    result = analyzer.analyze(data, data, n_permutations=49, random_seed=1)
    payload = result.to_dict()
    assert set(payload) >= {"statistic", "p_value", "n_objects", "significant"}
    assert isinstance(payload["p_value"], float)
    assert "Mantel" in result.summary()


def test_partial_result_serialises(analyzer: MantelAnalyzer) -> None:
    """The partial form adds its own fields."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(20, 4))
    control = rng.normal(size=(20, 2))
    result = analyzer.analyze_partial(a, a, control, n_permutations=49, random_seed=5)
    payload = result.to_dict()
    assert payload["statistic"] == pytest.approx(1.0, abs=1e-10)
    assert "Partial Mantel" in result.summary()
