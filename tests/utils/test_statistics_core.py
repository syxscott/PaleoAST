# =============================================================================
# Test: utils.statistics_core -- the shared permutation machinery
# =============================================================================
"""
The permutation helper every stochastic test in the project goes through.

Two things about it are easy to get wrong and hard to notice:

* the seed it reports has to be the seed it *used*. A result that says
  "the seed used" but cannot reproduce its own p-value is worse than one
  that says nothing;
* a permutation whose statistic is undefined is not evidence against the
  null. One NaN in the null used to make the two-sided p-value collapse
  to 1/(n+1) -- a p-value of 0.001 for a test that found nothing.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from utils.statistics_core import permutation_pvalue


def test_the_seed_used_is_reported():
    """A seeded run reports the seed it was given."""
    result = permutation_pvalue(1.0, lambda rng: 0.0, n_permutations=10, random_seed=7)
    assert result.random_seed == 7


def test_an_unseeded_run_still_reports_the_seed_it_used():
    """The result must name the seed that produced it."""
    with pytest.warns(RuntimeWarning):
        result = permutation_pvalue(1.0, lambda rng: 0.0, n_permutations=10, random_seed=None)
    assert isinstance(result.random_seed, int)


def test_the_reported_seed_reproduces_the_p_value():
    """The seed in the result has to reproduce the result."""

    def stat(rng):
        return float(rng.normal())

    with pytest.warns(RuntimeWarning):
        first = permutation_pvalue(1.0, stat, n_permutations=50, random_seed=None)

    second = permutation_pvalue(1.0, stat, n_permutations=50, random_seed=first.random_seed)
    assert second.p_value == first.p_value
    assert second.null_distribution.tolist() == first.null_distribution.tolist()


def test_a_permutation_with_an_undefined_statistic_is_not_counted_as_extreme():
    """A NaN null must not collapse the two-sided p-value."""
    values = [0.0] * 5 + [float("nan")] * 5
    source = itertools.cycle(values)

    def stat(rng):
        return next(source)

    result = permutation_pvalue(1.0, stat, n_permutations=10, random_seed=1, alternative="two_sided")
    # Five of the ten permutations produced a usable statistic, so the
    # denominator is six, not eleven.
    assert result.p_value_two_sided == pytest.approx(1.0 / 6.0)
    assert np.isfinite(result.p_value_two_sided)
