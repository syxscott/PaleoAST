# =============================================================================
# FILE: tests/property/test_diversity_properties.py
# =============================================================================
"""
Property-based tests for the diversity indices.

A diversity index is a number that a reader will quote, so the
properties are about the two things that make it the number it claims to
be:

  * it does not depend on the order of the taxa;
  * a taxon with zero individuals is not a taxon -- adding one must not
    move the index;
  * the indices stay inside the range they are defined on: Shannon on
    ``[0, ln(S)]``, Simpson and Pielou's evenness on ``[0, 1]``;
  * the richness is the number of taxa that are actually there, and the
    number of individuals is the number of individuals.

References:
    Pielou, E.C. (1966). The measurement of diversity in biological
    collections. American Zoologist 6, 211-218.
    Shannon, C.E. (1948). A mathematical theory of communication. Bell
    System Technical Journal 27, 379-423.
    Simpson, E.H. (1949). The interpretation of interaction diversity.
    Ecology 30, 165-168.
    Chao, A. (1984). Non-parametric estimation of the number of classes
    in a population. Scandinavian Journal of Statistics 11, 265-270.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ecology.diversity import DiversityAnalyzer, chao1_confidence_interval

from ._helpers import call_or_none

_SETTING = settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])

#: A sample of abundances: a non-negative count per taxon.
_abundances = st.lists(
    st.integers(min_value=0, max_value=500),
    min_size=2,
    max_size=20,
)


def _indices(abundances: list[int]):
    """The diversity indices of one sample, or None if there is no sample."""
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return None
    return call_or_none(DiversityAnalyzer().analyze_sample, array)


@given(abundances=_abundances)
@_SETTING
def test_the_diversity_indices_do_not_depend_on_the_order_of_the_taxa(abundances):
    """A sample is a set of taxa with counts, and the order they are
    listed in is not part of it."""
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    base = _indices(abundances)
    shuffled = array.copy()
    np.random.default_rng(11).shuffle(shuffled)
    moved = _indices(list(shuffled))
    if base is None or moved is None:
        return
    assert base.indices.keys() == moved.indices.keys()
    for name in base.indices:
        assert np.isclose(
            base.indices[name].value,
            moved.indices[name].value,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ), f"the {name} index moved when the taxa were reordered"


@given(abundances=_abundances)
@_SETTING
def test_a_taxon_with_no_individuals_is_not_a_taxon(abundances):
    """Adding a taxon with zero individuals must not move the index.

    This is the sharpest of the properties here: a formula that divides
    by the number of taxa rather than by the number of individuals with
    them changes the answer, and a formula that takes a log of the
    counts fails on the zero.
    """
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    base = _indices(abundances)
    moved = _indices([*list(array), 0])
    if base is None or moved is None:
        return
    for name in ("shannon", "simpson", "pielou"):
        if name not in base.indices or name not in moved.indices:
            continue
        assert np.isclose(
            base.indices[name].value,
            moved.indices[name].value,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ), f"the {name} index moved when an empty taxon was added"


@given(abundances=_abundances)
@_SETTING
def test_the_shannon_index_is_inside_its_range(abundances):
    """Shannon's H' lies between zero and the log of the richness.

    Zero is a single taxon, and the upper end is the even community.
    """
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    result = _indices(abundances)
    if result is None or "shannon" not in result.indices:
        return
    value = result.indices["shannon"].value
    richness = int(np.sum(array > 0))
    if richness < 2:
        return
    assert -1e-9 <= value <= np.log(richness) + 1e-9


@given(abundances=_abundances)
@_SETTING
def test_the_simpson_index_is_a_proportion(abundances):
    """Simpson's 1 - D lies between zero and one."""
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    result = _indices(abundances)
    if result is None or "simpson" not in result.indices:
        return
    value = result.indices["simpson"].value
    assert -1e-9 <= value <= 1.0 + 1e-9


@given(abundances=_abundances)
@_SETTING
def test_pielous_evenness_is_a_proportion(abundances):
    """Pielou's J is H' divided by the log of the richness, so it lies
    between zero and one."""
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    result = _indices(abundances)
    if result is None or "pielou" not in result.indices:
        return
    value = result.indices["pielou"].value
    assert -1e-9 <= value <= 1.0 + 1e-9


@given(abundances=_abundances)
@_SETTING
def test_the_richness_and_the_individual_count_are_what_they_say(abundances):
    """The richness is the number of taxa that are there, and the number
    of individuals is the number of individuals."""
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    result = _indices(abundances)
    if result is None:
        return
    assert result.taxa_count == int(np.sum(array > 0))
    assert result.individuals == int(np.sum(array))


@given(abundances=_abundances)
@_SETTING
def test_the_chao1_interval_contains_its_estimate(abundances):
    """The Chao-1 confidence interval brackets the Chao-1 estimate."""
    array = np.asarray(abundances, dtype=float)
    if array.sum() <= 0:
        return
    estimate = _indices(abundances)
    if estimate is None or "chao1" not in estimate.indices:
        return
    interval = call_or_none(chao1_confidence_interval, array)
    if interval is None:
        return
    # The function returns (estimate, lower, upper).
    estimate_from_interval, lower, upper = interval
    assert lower <= estimate_from_interval <= upper
    assert np.isclose(estimate.indices["chao1"].value, estimate_from_interval, rtol=1e-9, atol=1e-12)
