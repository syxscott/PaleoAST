"""
Ground truth for ``ecology/rarefaction.py`` (55% covered).

Both rarefaction estimators are checked against exact combinatorics
(``math.comb``) rather than against a re-implementation, so the oracle cannot
share a bug with the code under test.

The two estimators use DIFFERENT denominators and it is easy to check them
against each other's formula by accident:

* **individual-based** (``compute_rarefaction``, Hurlbert 1971) draws ``k``
  *individuals*; the denominator ``C(N, k)`` uses ``N`` = total individuals.
* **sample-based** (``compute_sample_based_rarefaction``) draws ``k``
  *sites*; the denominator uses ``N`` = total samples and ``n_i`` = the
  number of sites in which species ``i`` occurs.

The sample-based path had no independent test at all before this file.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecology.rarefaction import (  # noqa: E402
    compute_rarefaction,
    compute_sample_based_rarefaction,
)

SEED = 9


def _hurlbert(occurrences: np.ndarray, n_total: int, k: int) -> float:
    """E[S_k] = sum_i [ 1 - C(N - n_i, k) / C(N, k) ]  (Hurlbert 1971)."""
    return sum(
        1.0 - comb(n_total - int(a), k) / comb(n_total, k) for a in occurrences
    )


class TestIndividualBased:
    """``compute_rarefaction`` draws k individuals."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_whole_curve_matches_exact_combinatorics(self, seed):
        rng = np.random.default_rng(seed)
        abundances = rng.integers(1, 30, size=9)
        total = int(abundances.sum())
        max_n = min(10, total - 1)

        result = compute_rarefaction(
            abundances.astype(float), max_n=max_n, n_points=max_n
        )
        sizes = np.ravel(result.sample_sizes)
        values = np.ravel(result.expected_taxa)
        expected = np.array(
            [_hurlbert(abundances, total, int(k)) for k in sizes]
        )
        assert values == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_curve_is_monotone_and_bounded(self):
        rng = np.random.default_rng(5)
        abundances = rng.integers(1, 30, size=10)
        result = compute_rarefaction(abundances.astype(float), max_n=8, n_points=8)
        values = np.ravel(result.expected_taxa)
        assert np.all(np.diff(values) >= -1e-9)
        assert np.all(values <= len(abundances) + 1e-9)

    def test_full_sample_yields_total_richness(self):
        """E[S_N] = S exactly: taking every individual shows every species."""
        rng = np.random.default_rng(6)
        abundances = rng.integers(1, 12, size=7)
        total = int(abundances.sum())
        result = compute_rarefaction(
            abundances.astype(float), max_n=total, n_points=total
        )
        assert np.ravel(result.expected_taxa)[-1] == pytest.approx(
            len(abundances), rel=1e-6
        )

    def test_exact_beyond_the_stirling_threshold(self):
        """N >= 60 used to switch to a Stirling approximation.

        That cost ~1e-4 of relative accuracy, and at k = 1 the expectation is
        exactly 1.0 -- so a rarefaction curve started at 1.0001, an artefact
        a reader would reasonably take for a real excess richness.
        """
        rng = np.random.default_rng(31)
        abundances = rng.integers(1, 25, size=12)
        total = int(abundances.sum())
        assert total > 60, "fixture must exceed the Stirling threshold"

        at_one = compute_rarefaction(
            abundances.astype(float), max_n=1, n_points=1
        )
        assert np.ravel(at_one.expected_taxa)[0] == pytest.approx(1.0, abs=1e-12)

        at_max = compute_rarefaction(
            abundances.astype(float), max_n=min(20, total - 1), n_points=20
        )
        sizes = np.ravel(at_max.sample_sizes)
        values = np.ravel(at_max.expected_taxa)
        expected = np.array([_hurlbert(abundances, total, int(k)) for k in sizes])
        assert values == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_single_individual_expects_exactly_one_species(self):
        rng = np.random.default_rng(7)
        abundances = rng.integers(1, 12, size=7)
        result = compute_rarefaction(abundances.astype(float), max_n=1, n_points=1)
        assert np.ravel(result.expected_taxa)[0] == pytest.approx(1.0, abs=1e-9)


class TestSampleBased:
    """``compute_sample_based_rarefaction`` draws k SITES.

    The denominator is the number of samples, NOT the number of occurrences --
    using the wrong one is the easiest mistake here and produces a curve
    that is smooth and plausible but wrong everywhere.
    """

    @pytest.fixture(scope="class")
    def occurrence_matrix(self) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        return (rng.random((24, 9)) < 0.45).astype(float)

    def test_whole_curve_matches_exact_combinatorics(self, occurrence_matrix):
        occurrences = occurrence_matrix.sum(axis=0)
        occurrences = occurrences[occurrences > 0]
        n_samples = occurrence_matrix.shape[0]
        requested = np.array([1, 2, 3, 5, 8, 12, 18], dtype=float)

        results = compute_sample_based_rarefaction(occurrence_matrix, requested)
        assert len(results) == 1, "the API returns one result carrying the whole curve"
        result = results[0]

        sizes = np.ravel(result.sample_sizes)
        values = np.ravel(result.expected_taxa)
        assert list(sizes) == list(requested.astype(int))

        expected = np.array(
            [_hurlbert(occurrences, n_samples, int(k)) for k in sizes]
        )
        assert values == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_curve_is_monotone_and_bounded(self, occurrence_matrix):
        result = compute_sample_based_rarefaction(
            occurrence_matrix, np.array([1, 3, 6, 10], dtype=float)
        )[0]
        values = np.ravel(result.expected_taxa)
        richness = int(occurrence_matrix.sum(axis=0).astype(bool).sum())
        assert np.all(np.diff(values) >= -1e-9)
        assert np.all(values <= richness + 1e-9)

    def test_one_site_expects_that_site_s_mean_richness(self, occurrence_matrix):
        """Drawing a single site: the expectation is the mean richness."""
        per_site = occurrence_matrix.sum(axis=1)
        result = compute_sample_based_rarefaction(
            occurrence_matrix, np.array([1.0])
        )[0]
        assert np.ravel(result.expected_taxa)[0] == pytest.approx(
            per_site.mean(), rel=1e-9
        )

    def test_sizes_beyond_the_sample_count_are_clipped(self, occurrence_matrix):
        """k > N is undefined; the curve must stop at N, not blow up."""
        n_samples = occurrence_matrix.shape[0]
        result = compute_sample_based_rarefaction(
            occurrence_matrix, np.array([1.0, float(n_samples), float(n_samples * 4)])
        )[0]
        sizes = np.ravel(result.sample_sizes)
        assert np.all(sizes <= n_samples)
        assert np.all(np.isfinite(np.ravel(result.expected_taxa)))

    def test_default_grid_covers_every_size(self, occurrence_matrix):
        result = compute_sample_based_rarefaction(occurrence_matrix)[0]
        sizes = np.ravel(result.sample_sizes)
        assert sizes[0] == 1
        assert sizes[-1] == occurrence_matrix.shape[0]


class TestDegenerateInput:
    def test_empty_matrix_returns_nothing(self):
        assert compute_sample_based_rarefaction(np.zeros((0, 5))) == []

    def test_all_zero_columns_returns_nothing(self):
        assert compute_sample_based_rarefaction(np.zeros((6, 4))) == []
