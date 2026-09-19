# tests/stratigraphy/test_markov.py
"""Regression tests for the facies-transition chi-squared test.

The 2026-09 audit found that ``MarkovAnalyzer.analyze`` mixed the
bed-by-bed independence convention (observed counts including the
diagonal, ``df = (s-1)**2``) with the embedded-chain expectation
``E_ij = r_i c_j / (N - r_i)``, which inflated chi-squared until every
sequence - even white noise - was declared Markovian.  The tests below
pin the calibration of both supported conventions.
"""

import numpy as np
import pytest

from stratigraphy.markov import MarkovAnalyzer


def test_iid_sequences_are_not_declared_markovian():
    """Type-I error of the independence test must sit near alpha = 0.05."""
    analyzer = MarkovAnalyzer()
    rng = np.random.default_rng(20260906)
    chi2s = []
    rejections = 0
    reps = 200
    for _ in range(reps):
        seq = rng.integers(0, 4, 300)
        result = analyzer.analyze(seq)
        assert result.df == 9
        chi2s.append(result.chi_squared)
        rejections += result.p_value < 0.05
    rejection_rate = rejections / reps
    assert rejection_rate < 0.15, f"type-I error {rejection_rate:.3f} is inflated"
    # E[chi2] ~ df under the null; the buggy build measured ~5 x df.
    assert np.mean(chi2s) < 2.0 * 9


def test_true_markov_chain_is_detected():
    analyzer = MarkovAnalyzer()
    rng = np.random.default_rng(7)
    transition = np.array(
        [[0.0, 0.7, 0.2, 0.1],
         [0.1, 0.0, 0.8, 0.1],
         [0.5, 0.1, 0.0, 0.4],
         [0.2, 0.6, 0.2, 0.0]]
    )
    hits = 0
    for _ in range(20):
        state = int(rng.integers(4))
        seq = [state]
        for _ in range(399):
            state = int(rng.choice(4, p=transition[state]))
            seq.append(state)
        hits += analyzer.analyze(np.array(seq)).is_markovian
    assert hits >= 18


def test_expected_counts_use_full_table():
    """E_ij = row_i * col_j / N over every cell, diagonal included."""
    seq = [0, 0, 1, 1, 0, 1]
    # pairs: (0,0) (0,1) (1,1) (1,0) (0,1) -> T = [[1, 2], [1, 1]]
    observed = np.array([[1.0, 2.0], [1.0, 1.0]])
    expected = np.array([[1.2, 1.8], [0.8, 1.2]])
    result = MarkovAnalyzer().analyze(seq)
    np.testing.assert_allclose(result.transition_matrix, observed)
    np.testing.assert_allclose(result.expected_matrix, expected)
    np.testing.assert_allclose(result.difference_matrix, observed - expected)
    assert result.n_transitions == 5
    assert result.df == 1
    np.testing.assert_allclose(
        result.chi_squared,
        float(np.sum((observed - expected) ** 2 / expected)),
    )


def test_embedded_chain_uses_quasi_independence():
    """A sequence without repeated facies has a structural-zero diagonal."""
    seq = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    result = MarkovAnalyzer().analyze(seq)
    assert result.df == (3 - 1) * (3 - 2)
    assert np.all(np.diag(result.expected_matrix) == 0.0)
    assert np.all(np.diag(result.difference_matrix) == 0.0)
    # Off-diagonal margins drive the expectations.
    T = result.transition_matrix
    Td = T.copy()
    np.fill_diagonal(Td, 0.0)
    manual = np.outer(Td.sum(axis=1), Td.sum(axis=0)) / Td.sum()
    np.fill_diagonal(manual, 0.0)
    np.testing.assert_allclose(result.expected_matrix, manual)


def test_accepts_column_shaped_sequence():
    """(n, 1) arrays used to break the facies-code lookup."""
    column = np.array([[0], [1], [2], [0], [1], [2], [0], [1], [0]])
    result = MarkovAnalyzer().analyze(column)
    assert result.transition_matrix.shape == (3, 3)


def test_non_contiguous_codes_are_remapped():
    result = MarkovAnalyzer().analyze([0, 2, 5, 2, 0, 5],
                                      facies_names=["F0", "F2", "F5"])
    assert result.transition_matrix.shape == (3, 3)
    assert result.facies_names == ["F0", "F2", "F5"]
    assert result.n_transitions == 5


def test_short_and_degenerate_sequences_raise():
    analyzer = MarkovAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze([1])
    with pytest.raises(ValueError):
        analyzer.analyze([2, 2, 2, 2])          # a single facies
    with pytest.raises(ValueError):
        analyzer.analyze([0, 1, 0, 1])          # embedded chain, s = 2
    with pytest.raises(ValueError):
        analyzer.analyze([0, 1, 2, 0], facies_names=["only-two"])


def test_sparse_table_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        MarkovAnalyzer().analyze([0, 0, 0, 1, 2])
    assert any("below 1" in record.message for record in caplog.records)
