# tests/ecology/test_null_models.py
"""Regression tests for ecology/null_models.py randomization engines.

The 2026-09 audit found that:

* an unknown ``algorithm`` silently fell back to a ``shuffle`` (only the
  grand total fixed) instead of the constrained null that was asked for;
* "rrs" / "rcs", which the GUI offers as *Swap (Gotelli) / RRS / RCS*,
  were never implemented and were therefore identical to "shuffle";
* the multiprocessing path dropped the remainder of ``n_permutations``
  that did not fit an integer chunk, and computed the p-value against the
  requested rather than the performed number of replicates;
* the worker copy of the swap used ``int(n_species * n_sites * 0.1)``
  swap attempts, so small matrices were never randomized at all;
* ``SES`` was reported as 0.0 whenever the null had zero variance;
* ``analyze`` reseeded the global ``numpy.random`` state.
"""

import numpy as np
import pytest

from ecology.null_models import (
    NullModelAnalyzer,
    _permute_with_algorithm,
    _swap_matrix_worker,
    _worker_permute,
)
from utils.exceptions import ValidationError

MATRIX = np.array(
    [
        [1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
    ],
    dtype=np.int32,
)


def test_unknown_algorithm_is_rejected():
    analyzer = NullModelAnalyzer()
    with pytest.raises(ValidationError):
        analyzer.analyze(MATRIX, algorithm="frobnicate", n_permutations=10)
    with pytest.raises(ValidationError):
        analyzer._permute_matrix(MATRIX, "frobnicate")
    with pytest.raises(ValidationError):
        _permute_with_algorithm(MATRIX, "quick-swap")


def test_invalid_permutation_count_is_rejected():
    with pytest.raises(ValidationError):
        NullModelAnalyzer().analyze(MATRIX, n_permutations=0)


@pytest.mark.parametrize(
    "algorithm, rows_fixed, cols_fixed",
    [("swap", True, True), ("rrs", True, False), ("rcs", False, True), ("shuffle", False, False)],
)
def test_algorithms_preserve_the_documented_margins(algorithm, rows_fixed, cols_fixed):
    analyzer = NullModelAnalyzer()
    rng = np.random.default_rng(2026)
    for _ in range(25):
        permuted = analyzer._permute_matrix(MATRIX, algorithm, rng)
        same_rows = np.array_equal(permuted.sum(axis=1), MATRIX.sum(axis=1))
        same_cols = np.array_equal(permuted.sum(axis=0), MATRIX.sum(axis=0))
        assert same_rows or not rows_fixed
        assert same_cols or not cols_fixed
        assert np.array_equal(np.sort(permuted.ravel()), np.sort(MATRIX.ravel()))


def test_each_algorithm_is_a_distinct_null_model():
    """'rrs' and 'rcs' used to be silent aliases of 'shuffle'."""
    analyzer = NullModelAnalyzer()
    means = {
        algorithm: analyzer.analyze(
            MATRIX, algorithm=algorithm, n_permutations=400, random_seed=3
        ).mean_simulated
        for algorithm in ("swap", "rrs", "rcs", "shuffle")
    }
    assert len({round(v, 6) for v in means.values()}) == len(means)


def test_small_matrices_are_actually_randomized():
    """The worker swap used to perform zero exchanges on a 2x2/3x3 matrix."""
    matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int32)
    draws = {_swap_matrix_worker(matrix, np.random.default_rng(seed)).tobytes() for seed in range(20)}
    assert len(draws) > 1
    for raw in draws:
        swapped = np.frombuffer(raw, dtype=matrix.dtype).reshape(matrix.shape)
        np.testing.assert_array_equal(swapped.sum(axis=1), matrix.sum(axis=1))
        np.testing.assert_array_equal(swapped.sum(axis=0), matrix.sum(axis=0))


def test_worker_and_object_swap_share_implementation():
    matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)
    seed = 99
    worker = _swap_matrix_worker(matrix, np.random.default_rng(seed))
    via_analyzer = NullModelAnalyzer()._swap_matrix(matrix, np.random.default_rng(seed))
    np.testing.assert_array_equal(worker, via_analyzer)


def test_parallel_runs_every_requested_replicate():
    analyzer = NullModelAnalyzer()
    # n_permutations = 997 does not divide by any worker count; the old
    # floor-division chunking silently discarded the tail.
    for n_workers in (2, 3, 4, 8):
        result = analyzer.analyze(
            MATRIX, n_permutations=997, n_workers=n_workers, algorithm="swap", random_seed=11
        )
        assert result.n_permutations == 997
        assert result.simulated_scores.size == 997


def test_parallel_matches_sequential_distribution():
    analyzer = NullModelAnalyzer()
    parallel = analyzer.analyze(MATRIX, n_permutations=200, n_workers=4, random_seed=5)
    sequential = analyzer.analyze(MATRIX, n_permutations=200, n_workers=1, random_seed=5)
    assert parallel.observed_score == sequential.observed_score
    assert abs(parallel.mean_simulated - sequential.mean_simulated) < 0.2
    assert abs(parallel.std_simulated - sequential.std_simulated) < 0.2


def test_p_value_uses_performed_replicates():
    result = NullModelAnalyzer().analyze(MATRIX, n_permutations=101, random_seed=2)
    expected = (1 + np.sum(result.simulated_scores >= result.observed_score)) / (
        result.n_permutations + 1
    )
    assert result.p_value == pytest.approx(expected)
    assert 0.0 < result.p_value <= 1.0


def test_seeded_runs_are_reproducible_without_touching_global_state():
    analyzer = NullModelAnalyzer()
    np.random.seed(4242)
    control = np.random.rand(4)

    np.random.seed(4242)
    first = analyzer.analyze(MATRIX, n_permutations=25, random_seed=8)
    second = analyzer.analyze(MATRIX, n_permutations=25, random_seed=8)
    third = analyzer.analyze(MATRIX, n_permutations=25, random_seed=9)
    np.testing.assert_array_equal(first.simulated_scores, second.simulated_scores)
    assert not np.array_equal(first.simulated_scores, third.simulated_scores)
    # analyze() must not draw from, or reseed, the legacy global stream.
    np.testing.assert_allclose(np.random.rand(4), control)


def test_zero_variance_null_reports_undefined_ses():
    matrix = np.ones((3, 4), dtype=np.int32)
    analyzer = NullModelAnalyzer()
    result = analyzer.analyze(matrix, n_permutations=20, random_seed=1)
    assert result.std_simulated == 0.0
    assert np.isnan(result.standardized_effect_size)
    assert "n/a" in result.summary()


def test_worker_seed_produces_independent_streams():
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in np.random.SeedSequence(3).spawn(4)]
    scores = [_worker_permute(MATRIX, 5, "swap", "c_score", seed) for seed in seeds]
    assert len({tuple(s) for s in scores}) == len(scores)
