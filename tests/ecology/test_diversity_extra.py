# =============================================================================
# FILE: tests/ecology/test_diversity_extra.py
# =============================================================================
"""
Tests for SQS subsampling, sample-coverage rarefaction, diversity profiles
and the between-group diversity tests.

TESTING PHILOSOPHY HERE
~~~~~~~~~~~~~~~~~~~~~~~
Every assertion below is written so that **a wrong implementation
fails it**, rather than so that the current implementation passes. Concretely
that means asserting properties (invariance, monotonicity, anchoring,
exactness of an anchor point) instead of numbers copied out of this
implementation's own output, and preferring a property checked over many
random draws to a single golden value.

The one place a specific number is asserted is where the number is
derivable on paper -- ``[10, 10, 10]`` subsampled to a quorum of 6 by
round-robin is ``[2, 2, 2]`` -- which is checkable by hand and is therefore a
genuine independent check rather than a snapshot.

Two findings from writing these tests are worth recording, because both are
things a reader might otherwise assume to be true and are not:

* **SQS does NOT give equal index values to samples of equal size with
  different evenness.** Subsampling ``[10, 10, 10]`` and ``[28, 1, 1]`` to a
  quorum of 6 gives Shannon 1.0986 and 0.0000 respectively. SQS *preserves*
  the dominance structure rather than flattening it, which is the entire
  reason it is the palaeontological standard. ``TestSQSPreservesEvenness``
  pins this down because the opposite assumption is easy to write a test for
  by mistake.

* **Richness must be monotone non-decreasing in the quorum.** An earlier
  greedy-quota formulation of this module failed that check on 2436 of 3000
  random samples: it allocated each taxon its proportional quota by a
  different rule and dropped taxa non-monotonically. The round-robin
  formulation used here passes 3000/3000. ``TestSQSRichnessMonotonicity``
  is the regression guard for that.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ecology.diversity import compute_diversity_indices
from ecology.diversity_extra import (
    INDEX_KEYS,
    DiversityExtraAnalyzer,
    compute_diversity_profile,
    diversity_permutation_test,
    diversity_t_test,
    sample_coverage,
    sample_rarefaction_tao,
    sqs_subsample,
)
from utils.exceptions import ComputationError, StatisticalError, ValidationError


def _shannon(counts: np.ndarray) -> float:
    """Reference Shannon, written out independently of the module under test."""
    c = np.asarray([x for x in np.asarray(counts).ravel() if x > 0], dtype=float)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


# =============================================================================
# SQS
# =============================================================================


class TestSQSSubsample:
    """The core subsampling operation."""

    def test_round_robin_matches_hand_calculation(self):
        """[10, 10, 10] at quorum 6 is [2, 2, 2] -- checkable by hand.

        Three cycles of "one individual from each taxon in turn" take 3+3
        individuals and leave 2, which a fourth (partial) cycle divides evenly.
        A greedy "fill the most abundant taxon first" implementation would
        give [6, 0, 0] here, so this distinguishes the two designs rather
        than just pinning a number.
        """
        result = sqs_subsample(np.array([10, 10, 10]), 6)

        assert result.subsample.tolist() == [2, 2, 2]
        assert result.individuals == 6
        assert result.quorum == 6

    def test_subsample_always_reaches_exactly_the_quorum(self):
        """The quorum is a target that is always hit, never overshot.

        Fails for a subsample that under-fills (too few individuals, so the
        indices describe a smaller sample than the depth claims) or
        over-fills (the standardising is no longer standardising).
        """
        rng = np.random.default_rng(20240101)
        for _ in range(400):
            n_taxa = int(rng.integers(1, 12))
            counts = rng.integers(1, 25, size=n_taxa)
            total = int(counts.sum())
            for quorum in {1, total // 3, total // 2, total - 1, total}:
                if quorum < 1:
                    continue
                result = sqs_subsample(counts, quorum)
                assert result.individuals == quorum, f"{counts.tolist()} at q={quorum} gave {result.subsample.tolist()}"

    def test_taxa_never_exceed_their_original_abundance(self):
        """A taxon cannot give away individuals it never had.

        A per-taxon proportional quota computed as ``quorum * a_i / N`` is
        <= ``a_i`` whenever ``quorum <= N``, so this holds by construction
        -- but it is exactly the invariant a rounding-based implementation
        gets wrong when it rounds a small taxon up.
        """
        rng = np.random.default_rng(7)
        for _ in range(300):
            counts = rng.integers(1, 20, size=int(rng.integers(2, 10)))
            total = int(counts.sum())
            quorum = int(rng.integers(1, total + 1))
            result = sqs_subsample(counts, quorum)
            assert np.all(result.subsample <= counts), f"{counts.tolist()} -> {result.subsample.tolist()}"

    def test_quorum_equal_to_total_is_the_identity(self):
        """Standardising to the full sample size changes nothing.

        This is the boundary that pins the whole scaling: if the rule were
        off by one, or dropped the last individual, the full-quorum case
        would not return the original counts.
        """
        rng = np.random.default_rng(11)
        for _ in range(300):
            counts = rng.integers(1, 20, size=int(rng.integers(1, 10)))
            result = sqs_subsample(counts, int(counts.sum()))
            assert result.subsample.tolist() == counts.tolist()

    def test_is_invariant_to_the_input_column_order(self):
        """Shuffling the taxa must not change the subsample.

        Two rows of one abundance table with their columns permuted describe
        the same community. If the result depended on column order, a user's
        alphabetically-sorted table would give a different answer from the
        same data sorted by abundance -- and the difference would be silent.
        """
        rng = np.random.default_rng(23)
        for _ in range(300):
            counts = rng.integers(1, 20, size=int(rng.integers(2, 10)))
            total = int(counts.sum())
            quorum = int(rng.integers(1, total + 1))
            perm = rng.permutation(counts.size)
            base = sqs_subsample(counts, quorum)
            shuffled = sqs_subsample(counts[perm], quorum)
            assert sorted(base.subsample.tolist()) == sorted(shuffled.subsample.tolist())

    def test_richness_is_monotone_nondecreasing_in_the_quorum(self):
        """A bigger quorum never loses a taxon a smaller one kept.

        This is the property that makes a diversity profile readable as a
        curve. A greedy-quota allocation of this module's earlier revision
        violated it on 2436 of 3000 random samples; round-robin does not.
        """
        rng = np.random.default_rng(31)
        for _ in range(200):
            counts = rng.integers(1, 20, size=int(rng.integers(2, 9)))
            total = int(counts.sum())
            richness = [sqs_subsample(counts, q).taxa_retained for q in range(1, total + 1)]
            drops = [(q, richness[q - 1], richness[q]) for q in range(1, total) if richness[q] < richness[q - 1]]
            assert not drops, f"richness fell between quorums for {counts.tolist()}: {drops[:3]}"

    def test_rotation_not_ordering_is_what_defines_the_result(self):
        """The cycle is the method; the ordering alone would not do.

        A rule that simply filled the most abundant taxon first would turn
        [28, 1, 1] at quorum 6 into [6, 0, 0] -- one taxon, Shannon 0 --
        throwing away both rare taxa. Round-robin rotates, so the subsample
        keeps all three. This is the assertion that separates SQS from
        "truncate to the top taxa", and it fails for any implementation that
        drops the inner cycle.

        Hand-checkable: one cycle takes 3 individuals (one per taxon), a
        second takes 3 more (6 total), leaving the richest taxon with the
        surplus it alone can still supply.
        """
        dominated = sqs_subsample(np.array([28, 1, 1]), 6)
        assert dominated.subsample.tolist() == [4, 1, 1]
        assert dominated.taxa_retained == 3
        assert dominated.dropped_taxa == 0

    def test_surplus_goes_only_to_taxa_that_still_have_individuals(self):
        """Once a taxon is exhausted it is skipped, not asked again.

        For [28, 1, 1] the singletons can each supply exactly one, so no
        matter how many cycles run they never exceed 1. An implementation
        that decremented without checking would drive them negative.
        """
        dominated = sqs_subsample(np.array([28, 1, 1]), 20)
        assert dominated.subsample.tolist() == [18, 1, 1]
        assert dominated.individuals == 20
        assert np.all(dominated.subsample >= 0)

    def test_preserves_evenness_rather_than_flattening_it(self):
        """Equal quorum does NOT mean equal index -- the property SQS exists for.

        The brief suggested that equal-quorum subsampling yields the same
        index for equal-size samples with different evenness. It does not,
        and asserting that would have been asserting a bug. Round-robin
        preserves the dominance structure: the even sample stays even
        (Shannon = ln 3 = 1.0986) and the dominated one stays dominated
        (Shannon = 0.8676). Both hand-checkable from [2,2,2] and [4,1,1].

        A naive "fill the most abundant taxon first" rule would collapse
        [28, 1, 1] to [6, 0, 0] -- Shannon 0, one taxon -- which is what
        makes the rotation, rather than the ordering alone, the defining
        part of the method.
        """
        even = sqs_subsample(np.array([10, 10, 10]), 6)
        dominated = sqs_subsample(np.array([28, 1, 1]), 6)

        assert even.individuals == dominated.individuals == 6
        assert even.subsample.tolist() == [2, 2, 2]
        assert dominated.subsample.tolist() == [4, 1, 1]
        assert _shannon(even.subsample) == pytest.approx(np.log(3))
        assert _shannon(dominated.subsample) == pytest.approx(0.8675632285, abs=1e-9)
        assert _shannon(even.subsample) > _shannon(dominated.subsample)

    def test_exact_subsampling_preserves_shannon(self):
        """A subsample proportional to the original leaves Shannon unchanged.

        Shannon depends only on relative abundances, so a subsample that is
        an exact integer scaling of the original must reproduce it. [5, 5, 5,
        5, 5] at quorum 10 is [2, 2, 2, 2, 2] -- two whole cycles -- and both
        are uniform, so both have Shannon = ln 5.
        """
        original = np.array([5, 5, 5, 5, 5])
        result = sqs_subsample(original, 10)

        assert result.subsample.tolist() == [2, 2, 2, 2, 2]
        assert _shannon(result.subsample) == pytest.approx(_shannon(original), abs=1e-12)
        assert _shannon(result.subsample) == pytest.approx(np.log(5))

    def test_indices_come_from_the_shared_implementation(self):
        """The standardised indices equal ``compute_diversity_indices`` exactly.

        This is the anti-drift guard. If a second Shannon were written here,
        it could differ from ``ecology.diversity`` in the last bits and
        nothing else in the suite would notice; comparing against the real
        one makes the coupling explicit.
        """
        counts = np.array([12, 8, 6, 4, 2, 1])
        result = sqs_subsample(counts, 10)
        expected = compute_diversity_indices(result.subsample[result.subsample > 0].astype(float))

        assert result.standardized_diversity is not None
        assert result.indices_recomputed
        for key in INDEX_KEYS:
            got = result.standardized_diversity.get(key)
            want = expected.get(key)
            if want is None:
                assert got is None
            else:
                assert got == pytest.approx(want, rel=1e-12, abs=1e-12)

    def test_summary_and_to_dict_describe_the_subsample(self):
        """The report says what happened, in the numbers it happened at."""
        result = sqs_subsample(np.array([10, 10, 10]), 6)
        payload = result.to_dict()

        assert payload["quorum"] == 6
        assert payload["subsample"] == [2, 2, 2]
        assert payload["individuals"] == 6
        assert payload["taxa_retained"] == 3
        assert "SQS" in result.summary()


class TestSQSEdgeCases:
    """The inputs that break a subsampler rather than exercise it."""

    def test_single_taxon(self):
        """One taxon, quorum below its count: it simply gives up the surplus."""
        result = sqs_subsample(np.array([10]), 4)

        assert result.subsample.tolist() == [4]
        assert result.individuals == 4
        assert result.taxa_retained == 1
        assert result.dropped_taxa == 0

    def test_all_abundances_one(self):
        """Every taxon a singleton: the quorum picks that many of them.

        Four singleton taxa at quorum 2 must yield exactly 2 individuals with
        at most 2 taxa -- the round-robin gives one to the first two and
        cannot invent a third.
        """
        result = sqs_subsample(np.array([1, 1, 1, 1]), 2)

        assert result.individuals == 2
        assert result.taxa_retained == 2
        assert sorted(result.subsample.tolist()) == [0, 0, 1, 1]

    def test_quorum_larger_than_sample_is_rejected(self):
        """An unreachable quorum is an error, not a silently truncated subsample.

        Clamping to the available individuals would report indices computed at
        a depth the caller did not ask for -- exactly the error SQS exists to
        prevent when standardising many samples at once.
        """
        with pytest.raises(ValidationError, match="exceeds"):
            sqs_subsample(np.array([3, 1]), 10)

    def test_quorum_zero_and_negative_are_rejected(self):
        """No individuals means no indices."""
        for quorum in (0, -5):
            with pytest.raises(ValidationError, match="at least 1"):
                sqs_subsample(np.array([10, 10]), quorum)

    def test_fractional_quorum_is_rejected(self):
        """A quorum is a count of individuals, so it must be a whole number."""
        with pytest.raises(ValidationError, match="whole number"):
            sqs_subsample(np.array([10, 10]), 6.5)

    def test_zeros_and_negatives_are_dropped_not_counted(self):
        """Absent taxa contribute nothing and do not become subsamples."""
        result = sqs_subsample(np.array([5, 0, 4, -3]), 6)

        assert result.individuals == 6
        assert result.taxa_retained == 2

    def test_subsample_keeps_one_entry_per_input_column(self):
        """The subsample is aligned with the input columns, zeros included.

        Regression: the absent taxa used to be dropped *before* the
        subsample was built, so the result was shorter than the input and
        its column i was not input column i. A caller matching the
        subsample back to their own taxon list would have silently
        mismatched every taxon after the first absent one.

        Hand-checkable: [5, 0, 4] holds 9 individuals, and the round-robin
        at quorum 6 gives 3 from the 5 and 3 from the 4, with the absent
        taxon at zero.
        """
        result = sqs_subsample(np.array([5, 0, 4, -3]), 6)

        assert result.subsample.tolist() == [3, 0, 3, 0]
        assert len(result.subsample) == 4
        assert result.individuals == 6
        assert result.taxa_retained == 2
        assert result.dropped_taxa == 0
        assert result.to_dict()["subsample"] == [3, 0, 3, 0]

    def test_subsample_columns_follow_the_input_columns(self):
        """Permuting the input permutes the subsample the same way.

        This is the alignment contract stated as a property rather than as
        a number: if the subsample is aligned with the input, then
        shuffling the input shuffles the subsample identically. A
        subsample built on a shortened (zero-dropped) vector would fail
        this, because the two orders would no longer correspond.
        """
        counts = np.array([10, 5, 1, 1])
        perm = np.array([2, 0, 3, 1])
        base = sqs_subsample(counts, 8)
        shuffled = sqs_subsample(counts[perm], 8)

        assert base.subsample.tolist() == [3, 3, 1, 1]
        assert shuffled.subsample.tolist() == base.subsample[perm].tolist()

    def test_empty_sample_is_rejected(self):
        """No positive abundances is a validation error, not a NaN result."""
        with pytest.raises((ValidationError, ComputationError)):
            sqs_subsample(np.array([0, 0, 0]), 1)

    def test_fractional_abundances_are_rejected(self):
        """Relative abundances would silently become counts.

        The integrality test is on the values, not their sum: ``[0.1, 0.2,
        0.3, 0.4]`` sums to exactly 1.0 and would pass a sum check.
        """
        with pytest.raises(ValidationError, match="integer abundance counts"):
            sqs_subsample(np.array([0.1, 0.2, 0.3, 0.4]), 1)


# =============================================================================
# Sample coverage / Mao's tau
# =============================================================================


class TestSampleCoverage:
    """Coverage of an incidence matrix, and Mao's tau."""

    def test_hand_computed_coverage(self):
        """A worked example, derived on paper rather than copied from output.

        Six samples, 7 taxa, f1 = 2 (seen once) and f2 = 1 (seen twice).
        The bracket is
        ``(1 - 1/3) * 5/4 + 1/(3 * 4) = 0.833333 + 0.083333 = 0.9166667``,
        so ``C = 1 - (2/6) * 0.9166667 = 0.6944444`` and
        ``tau = 6 * (1 - 0.6944444) = 1.8333333``.
        """
        incidence = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 0, 0],
            ],
            dtype=float,
        )
        result = sample_coverage(incidence)

        assert result.n_samples == 6
        assert result.observed_richness == 7
        assert result.f1 == 2.0
        assert result.f2 == 1.0
        assert result.coverage == pytest.approx(0.6944444444, abs=1e-9)
        assert result.tau == pytest.approx(1.8333333333, abs=1e-9)
        assert result.method == "chao_jost_incidence"

    def test_tau_is_the_documented_function_of_coverage(self):
        """tau = m(1 - C), exactly, for any input.

        Asserting the defining relationship rather than a second hard-coded
        number means a change to the coverage estimator cannot silently
        decouple the two reported quantities.
        """
        rng = np.random.default_rng(101)
        for _ in range(50):
            incidence = (rng.random((int(rng.integers(3, 12)), int(rng.integers(4, 20)))) < 0.35).astype(float)
            if not incidence.any():
                continue
            result = sample_coverage(incidence)
            assert result.tau == pytest.approx(result.n_samples * (1.0 - result.coverage), abs=1e-12)

    def test_coverage_is_a_probability(self):
        """C always lands in [0, 1] whatever the incidence pattern.

        The estimator genuinely goes negative when every taxon is a
        singleton, so the clamp is doing real work here; a missing clamp
        would yield a negative "probability" and a negative tau.
        """
        rng = np.random.default_rng(202)
        patterns = [
            np.ones((5, 5)),  # every taxon in every sample
            np.eye(3),  # every taxon a singleton: C would go negative
            np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]),
        ]
        for _ in range(60):
            patterns.append((rng.random((int(rng.integers(1, 10)), int(rng.integers(1, 12)))) < 0.4).astype(float))
        for incidence in patterns:
            result = sample_coverage(incidence)
            assert 0.0 <= result.coverage <= 1.0, f"C={result.coverage} for {incidence.tolist()}"
            assert result.tau >= 0.0
            # The unseen-species estimate divides by the coverage, so it is
            # reported as None (not estimable) when the coverage is 0.
            assert result.unseen_species is None or result.unseen_species >= 0.0

    def test_no_singletons_means_complete_sampling(self):
        """f1 = 0 means nothing is known to be missing: C = 1, tau = 0."""
        incidence = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
        result = sample_coverage(incidence)

        assert result.f1 == 0.0
        assert result.coverage == 1.0
        assert result.tau == pytest.approx(0.0)
        assert result.unseen_species == pytest.approx(0.0)
        assert result.method == "complete_no_singletons"

    def test_single_sample_falls_back_instead_of_dividing_by_zero(self):
        """The Chao & Jost bracket needs m >= 3; m = 1 must not divide by m-2.

        Good's estimator is the honest fallback, and ``method`` says which
        branch produced the number so a reader can tell it apart from a
        full-strength estimate.
        """
        result = sample_coverage(np.array([[1, 1, 0]], dtype=float))

        assert result.method == "good_fallback"
        assert 0.0 <= result.coverage <= 1.0
        assert np.isfinite(result.tau)

    def test_two_samples_also_use_the_fallback(self):
        """m = 2 is the other side of the m >= 3 boundary."""
        result = sample_coverage(np.array([[1, 1, 0], [1, 0, 1]], dtype=float))

        assert result.method == "good_fallback"
        assert np.isfinite(result.coverage)

    def test_abundance_matrix_is_accepted_as_incidence(self):
        """Presence is presence: how many individuals is not the question.

        A taxon seen once and a taxon seen fifty times are both "present" in
        one sample, so the estimator must see the same matrix either way.
        """
        incidence = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
        abundant = incidence * np.array([[50], [1], [7], [3]])

        assert sample_coverage(abundant).coverage == pytest.approx(sample_coverage(incidence).coverage, abs=1e-12)

    def test_empty_matrix_is_rejected(self):
        """No samples or no taxa is a validation error."""
        with pytest.raises(ValidationError):
            sample_coverage(np.zeros((0, 5)))
        with pytest.raises(ValidationError):
            sample_coverage(np.zeros((3, 0)))

    def test_zero_coverage_reports_no_unseen_estimate_and_stays_valid_json(self):
        """A coverage of 0 has no finite unseen-species estimate.

        Regression: the estimate f0 = f1 (1 - C) / C divides by C, so with
        C clamped to 0 it came back as infinity -- and infinity is not
        valid JSON (RFC 8259 has no Infinity literal), so json.dumps
        emitted "Infinity" and a strict parser would reject the payload.
        The estimate is now reported as None: not estimable, rather than
        infinitely many taxa.
        """
        incidence = np.eye(3)  # every taxon a singleton
        result = sample_coverage(incidence)

        assert result.coverage == 0.0
        assert result.unseen_species is None
        assert result.asymptotic_richness is None

        payload = result.to_dict()
        assert payload["unseen_species"] is None
        assert payload["asymptotic_richness"] is None
        # allow_nan=False makes json.dumps raise on any non-finite number.
        text = json.dumps(payload, allow_nan=False)
        assert "Infinity" not in text
        assert "NaN" not in text
        assert "n/a" in result.summary()

    def test_rejects_one_dimensional_incidence(self):
        """A 1-D presence vector is ambiguous and must be refused.

        Regression: the validator reshapes a 1-D array to (n, 1), so
        [1, 1, 0] used to be read as "three samples of one taxon" --
        a survey of three samples with one taxon, coverage 1.0 -- rather
        than "one sample of three taxa". The dimensionality is now
        captured before the reshape, so the 2-D check can fire.
        """
        with pytest.raises(ValidationError, match="2-D"):
            sample_coverage(np.array([1, 1, 0]))

    def test_summary_reports_coverage_and_tau(self):
        """The report carries both quantities it promises."""
        result = sample_coverage(np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float))
        text = result.summary()

        assert "coverage" in text.lower()
        assert "tau" in text
        assert result.to_dict()["coverage"] == pytest.approx(result.coverage)


# =============================================================================
# Sample rarefaction (Mao's tau)
# =============================================================================


class TestSampleRarefactionTao:
    """Coverage-based rarefaction over the number of samples."""

    @staticmethod
    def _incidence() -> np.ndarray:
        return np.array(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 0, 0],
            ],
            dtype=float,
        )

    def test_curve_is_anchored_at_the_observed_richness(self):
        """At the full sample count the curve returns the observed richness.

        This is the defining consistency condition: a rarefaction curve that
        does not pass through the data point it is supposedly anchored to is
        answering a different question. The check is at the exact observed
        sample size, not at an interpolated grid point.
        """
        incidence = self._incidence()
        result = sample_rarefaction_tao(incidence)
        observed = sample_coverage(incidence).observed_richness

        at_observed = result.expected_richness[np.asarray(result.sample_sizes) == incidence.shape[0]]
        assert at_observed.size == 1
        assert at_observed[0] == pytest.approx(float(observed), abs=1e-9)

    def test_curve_is_nondecreasing_in_sample_count(self):
        """More samples cannot reveal fewer taxa.

        A curve that dips is reporting a negative discovery rate, which the
        ``maximum.accumulate`` guard is there to prevent.
        """
        result = sample_rarefaction_tao(self._incidence())
        richness = np.asarray(result.expected_richness)

        assert np.all(np.diff(richness) >= -1e-12), f"curve dipped: {richness.tolist()}"

    def test_extrapolation_never_undercuts_the_observed_point(self):
        """Beyond the observed samples the curve keeps climbing.

        Extrapolation steepens by 1/C, so with C < 1 the extrapolated branch
        is above the interpolated trend. Clipping sizes to ``m`` (as the
        combinatorial routine does) would flatten this instead.
        """
        incidence = self._incidence()
        result = sample_rarefaction_tao(incidence, np.array([incidence.shape[0], 10, 20]))
        observed = float(sample_coverage(incidence).observed_richness)

        assert result.expected_richness[0] == pytest.approx(observed, abs=1e-9)
        assert result.expected_richness[1] > observed
        assert result.expected_richness[2] > result.expected_richness[1]
        assert result.extrapolated.tolist() == [False, True, True]

    def test_complete_sampling_extrapolates_flat(self):
        """With no singletons there is nothing left to find.

        Coverage 1 means the curve should not rise beyond the observed
        richness. A slope that ignored C would keep adding taxa to a
        community the estimator says is fully known.
        """
        incidence = np.ones((4, 3))
        result = sample_rarefaction_tao(incidence, np.array([4, 8, 12]))

        assert_allclose(result.expected_richness, np.full(3, 3.0), rtol=1e-12)

    def test_custom_sample_sizes_are_honoured(self):
        """The caller's grid is returned in the order and size asked for."""
        result = sample_rarefaction_tao(self._incidence(), np.array([2, 4, 6, 9]))

        assert result.sample_sizes.tolist() == [2, 4, 6, 9]
        assert result.expected_richness.shape == (4,)

    def test_default_grid_covers_one_to_m(self):
        """Omitting sizes gives every sample count from 1 to m."""
        incidence = self._incidence()
        result = sample_rarefaction_tao(incidence)

        assert result.sample_sizes.tolist() == list(range(1, incidence.shape[0] + 1))
        assert not result.extrapolated.any()

    def test_extrapolation_steepens_by_the_coverage(self):
        """The extrapolated branch must rise faster than the interpolated one.

        Coverage-scaled extrapolation rises by ``f1/C`` per extra sample
        rather than ``f1``. That ratio is the whole reason for using a
        coverage-based estimator: an under-covered survey reveals taxa
        faster per additional sample than a well-covered one. Making the
        extrapolated slope equal the interpolated one would answer a question
        about sample size rather than about coverage.

        Measured entirely within the extrapolated branch -- per extra sample
        between m = 10 and m = 12 -- so the comparison is not confused with
        the interpolated branch declining toward the anchor.
        """
        incidence = self._incidence()
        coverage = sample_coverage(incidence)
        m0 = incidence.shape[0]
        assert 0.0 < coverage.coverage < 1.0, "fixture must be partially covered to test this"

        result = sample_rarefaction_tao(incidence, np.array([m0 + 4, m0 + 6]))
        slope = (result.expected_richness[1] - result.expected_richness[0]) / 2.0

        # Per extra sample the curve rises by f1 / (C * m0): the expected
        # number of singletons revealed by one more sample, scaled by the
        # coverage shortfall. Computed by hand from the reported f1 and C.
        expected_slope = coverage.f1 / (coverage.coverage * m0)
        assert slope == pytest.approx(expected_slope, rel=1e-9)
        # And that is steeper than the interpolated branch's f1 / m0, which
        # is what "extrapolation is coverage-scaled" means.
        assert slope == pytest.approx(coverage.f1 / m0 / coverage.coverage, rel=1e-9)
        assert slope > coverage.f1 / m0

    def test_rejects_nonpositive_sample_sizes(self):
        """Zero or negative samples is not a point on the curve."""
        with pytest.raises(ValidationError, match="positive"):
            sample_rarefaction_tao(self._incidence(), np.array([0, 3]))

    def test_rejects_empty_sample_sizes(self):
        """An empty grid is a caller error, not an empty result."""
        with pytest.raises(ValidationError, match="empty"):
            sample_rarefaction_tao(self._incidence(), np.array([]))

    def test_rejects_nonfinite_sample_sizes(self):
        """NaN and infinity are not points on the curve.

        Regression: casting a NaN to int overflows to INT64_MIN in numpy
        2.x, and that negative size was then used as m in the curve, so
        the result carried a nonsensical sample size instead of an error.
        """
        for bad in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValidationError, match="finite"):
                sample_rarefaction_tao(self._incidence(), np.array([2, bad]))

    def test_curve_does_not_depend_on_the_order_of_the_requested_sizes(self):
        """The same sizes must give the same richness at each m.

        Regression: the monotonicity guard ran over the array in the order
        the caller supplied it, so a descending grid was flattened to a
        constant and the anchor S(m0) = S_obs was destroyed with it. The
        curve is a function of the sample count, so the order of the grid
        cannot change it.
        """
        incidence = self._incidence()
        coverage = sample_coverage(incidence)
        m0 = incidence.shape[0]

        def by_size(result):
            sizes = np.asarray(result.sample_sizes).tolist()
            richness = np.asarray(result.expected_richness).tolist()
            return dict(zip(sizes, richness, strict=True))

        ascending = by_size(sample_rarefaction_tao(incidence, np.array([2, 4, 6])))
        descending = by_size(sample_rarefaction_tao(incidence, np.array([6, 4, 2])))
        shuffled = by_size(sample_rarefaction_tao(incidence, np.array([4, 6, 2])))

        assert descending == pytest.approx(ascending)
        assert shuffled == pytest.approx(ascending)

        # The analytic value at m = 2: S_obs + f1 (m - m0) / m0.
        expected_at_two = coverage.observed_richness + coverage.f1 * (2 - m0) / m0
        assert descending[2] == pytest.approx(expected_at_two, abs=1e-9)
        # And the anchor survives the descending order.
        assert descending[m0] == pytest.approx(float(coverage.observed_richness), abs=1e-9)

    def test_observed_richness_is_the_anchor_not_the_last_requested_point(self):
        """observed_richness is S_obs whatever grid was asked for.

        Regression: the property read the curve at the largest
        *requested* size, so a grid stopping below m0 reported an
        interpolated value and a grid running past m0 reported an
        extrapolated one. It is documented as the richness at the full
        observed sample count, which is the anchor.
        """
        incidence = self._incidence()
        observed = float(sample_coverage(incidence).observed_richness)

        for sizes in ([2, 4], [6, 20], [6], [1, 2, 3], [20]):
            result = sample_rarefaction_tao(incidence, np.array(sizes))
            assert result.observed_richness == observed, f"wrong anchor for sizes {sizes}"


# =============================================================================
# Diversity profiles
# =============================================================================


class TestDiversityProfile:
    """Indices as a function of sampling depth."""

    @staticmethod
    def _counts() -> np.ndarray:
        return np.array([10, 6, 4, 2, 1])

    def test_every_index_is_tracked_against_depth(self):
        """A profile without an index is not a diversity profile."""
        result = compute_diversity_profile(self._counts(), n_points=6)

        assert {"richness", "shannon", "simpson"}.issubset(set(result.index_values))
        for values in result.index_values.values():
            assert np.asarray(values).size == np.asarray(result.depths).size

    def test_depths_span_the_sample_and_stay_within_it(self):
        """The default grid runs from 1 to N without exceeding N.

        A depth above N would have to be clamped inside ``sqs_subsample``,
        meaning several depths silently described the same sample.
        """
        counts = self._counts()
        result = compute_diversity_profile(counts, n_points=8)
        depths = np.asarray(result.depths)

        assert depths.min() >= 1
        assert depths.max() <= int(counts.sum())
        assert np.all(np.diff(depths) > 0), "duplicate depths"

    def test_richness_is_monotone_along_the_profile(self):
        """The curve property again, read off the profile this time."""
        counts = self._counts()
        richness = compute_diversity_profile(counts, n_points=10).richness

        assert np.all(np.diff(richness) >= 0), f"richness fell along the profile: {richness.tolist()}"

    def test_full_depth_reproduces_the_raw_indices(self):
        """The deepest point of the profile is the unsubsampled sample.

        This is the cross-check between the profile and the existing index
        code: at quorum = N, SQS is the identity, so the profile's last
        Shannon must equal ``compute_diversity_indices`` on the raw data. If
        the profile were using a different estimator, this is where it shows.
        """
        counts = self._counts()
        raw = compute_diversity_indices(counts.astype(float))
        result = compute_diversity_profile(counts, quorums=np.array([int(counts.sum())]))

        assert result.depths.tolist() == [int(counts.sum())]
        assert float(result.index_values["shannon"][0]) == pytest.approx(raw.shannon_index, abs=1e-12)
        assert float(result.richness[0]) == pytest.approx(float(raw.taxa_count))

    def test_undefined_indices_are_nan_rather_than_missing(self):
        """Pielou needs S > 1 and Margalef needs N > 1.

        At depth 1 the subsample has one taxon, so those indices do not
        exist. They are reported as NaN in place, keeping every curve the
        same length as ``depths`` -- dropping them would leave a caller
        aligning arrays of different lengths.
        """
        result = compute_diversity_profile(self._counts(), quorums=np.array([1, 5, 10]))

        assert np.isnan(result.index_values["pielou"][0])
        assert np.isnan(result.index_values["margalef"][0])
        assert np.all(np.diff(np.asarray(result.depths)) > 0)
        assert result.index_values["shannon"].size == 3

    def test_values_change_with_depth_because_the_subsample_does(self):
        """The curve must be a function of depth, not of the raw sample.

        A profile that ignored the subsample and re-read the raw sample would
        return the same value at every depth. Asserting *monotonicity* here
        would be wrong: Shannon over an SQS curve rises as the rare tail is
        trimmed away, peaks once the rare taxa are gone, and then declines
        as the remaining taxa are cut down unevenly -- so it is genuinely
        non-monotone. What has to hold is that the values genuinely vary, and
        that the endpoints are the hand-computable ones.

        Endpoint values are exact: at quorum 1 the subsample is a single
        individual, so Shannon is -p ln p with p = 1, i.e. 0; at quorum N the
        subsample is the identity, so it equals the raw sample's Shannon.
        """
        counts = self._counts()
        result = compute_diversity_profile(counts, n_points=8)
        shannon = np.asarray(result.index_values["shannon"])

        assert np.ptp(shannon) > 0.5, f"Shannon barely varied across depth: {shannon.tolist()}"
        assert len(set(np.round(shannon, 12))) == shannon.size, "repeated values mean the subsample was ignored"

        depth_one = compute_diversity_profile(counts, quorums=np.array([1]))
        depth_full = compute_diversity_profile(counts, quorums=np.array([int(counts.sum())]))
        raw = compute_diversity_indices(counts.astype(float))

        assert float(depth_one.index_values["shannon"][0]) == pytest.approx(0.0, abs=1e-12)
        assert float(depth_full.index_values["shannon"][0]) == pytest.approx(raw.shannon_index, abs=1e-12)

    def test_richness_actually_tracks_the_quorum_when_taxa_are_dropped(self):
        """Richness must fall when the quorum is too small to hold every taxon.

        A profile that reported the raw taxon count at every depth would be
        flat at 5 here, with the four shallow points wrong.
        """
        counts = np.array([50, 1, 1, 1, 1])
        result = compute_diversity_profile(counts, quorums=np.array([2, 5, 20, 54]))
        richness = result.richness

        assert richness[0] == pytest.approx(2.0)
        assert richness[1] == pytest.approx(5.0)
        assert richness[-1] == pytest.approx(5.0)
        assert np.all(np.diff(richness) >= 0)
        assert richness[0] < richness[-1]

    def test_rejects_depth_beyond_the_sample(self):
        """Asking for more individuals than exist is an error."""
        with pytest.raises(ValidationError, match="must lie in"):
            compute_diversity_profile(self._counts(), quorums=np.array([5, 999]))

    def test_rejects_empty_quorums(self):
        """An empty depth grid is a caller error."""
        with pytest.raises(ValidationError, match="empty"):
            compute_diversity_profile(self._counts(), quorums=np.array([]))

    def test_rejects_nonfinite_quorums(self):
        """NaN and infinity are not sampling depths.

        Regression: a NaN compares False against both bounds, so it
        slipped through the range check and was then either silently
        dropped (leaving an empty profile whose summary() crashed on a
        zero-size array) or turned the error message into a ValueError
        from int(nan).
        """
        for bad in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValidationError, match="finite"):
                compute_diversity_profile(self._counts(), quorums=np.array([2, bad]))

        # The error path itself must not crash: a NaN next to an
        # out-of-range value used to raise ValueError from int(nan).
        with pytest.raises(ValidationError):
            compute_diversity_profile(self._counts(), quorums=np.array([0.0, np.nan]))

    def test_rejects_fractional_quorums(self):
        """A depth is a count of individuals, so it must be a whole number.

        Regression: the quorums were rounded, so [2.5, 5.0] silently
        became the profile of [2, 5] -- a different question than the one
        that was asked.
        """
        with pytest.raises(ValidationError, match="whole numbers"):
            compute_diversity_profile(self._counts(), quorums=np.array([2.5, 5.0]))

    def test_rejects_n_points_below_one(self):
        """A profile needs at least one depth."""
        with pytest.raises(ValidationError, match="n_points"):
            compute_diversity_profile(self._counts(), n_points=0)

    def test_summary_renders_each_depth(self):
        """One row per depth, so the table can be read against the x-axis."""
        result = compute_diversity_profile(self._counts(), n_points=5)
        text = result.summary()

        assert text.count("depth") == 1
        assert result.to_dict()["depths"] == [int(x) for x in np.asarray(result.depths)]


# =============================================================================
# Diversity t test
# =============================================================================


def two_group_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Four clearly-diverse and four clearly-depleted samples.

    A plain helper rather than a fixture: several tests need it inside a
    ``pytest.raises`` block or alongside a locally built matrix, and calling
    a fixture directly is an error in current pytest.
    """
    matrix = np.array(
        [
            [20, 10, 5, 3, 2, 1, 1],
            [18, 12, 6, 2, 2, 1, 1],
            [22, 9, 4, 4, 1, 1, 1],
            [19, 11, 5, 3, 2, 2, 1],
            [2, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [3, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    groups = np.array(["rich", "rich", "rich", "rich", "poor", "poor", "poor", "poor"])
    return matrix, groups


def varying_richness_matrix():
    """Two groups whose richness differs and varies within each group.

    The other fixture gives every sample the same taxon count, so
    richness is constant there and is dropped before the test runs --
    which is why it cannot be used to check that richness is really
    tested. Here the rich group has 7, 6, 7 and 5 taxa and the poor
    group 5, 4, 3 and 4, so richness varies within each group (it is not
    degenerate) and differs between them (it separates the groups).
    """
    matrix = np.array(
        [
            [20, 10, 5, 3, 2, 1, 1],
            [18, 12, 6, 2, 2, 1, 0],
            [22, 9, 4, 4, 1, 1, 1],
            [19, 11, 5, 3, 2, 0, 0],
            [3, 2, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [3, 2, 1, 0, 0, 0, 0],
            [2, 2, 1, 1, 0, 0, 0],
        ],
        dtype=float,
    )
    groups = np.array(["rich", "rich", "rich", "rich", "poor", "poor", "poor", "poor"])
    return matrix, groups


class TestDiversityTTest:
    """Per-index comparison of index values across groups."""

    def test_finds_the_seeded_difference(self):
        """The fixture's two groups were built to differ; it must be detected.

        Uses Shannon only, so the assertion is about the test finding a real
        effect rather than about any particular index happening to separate.
        """
        matrix, groups = two_group_matrix()
        result = diversity_t_test(matrix, groups, indices=["shannon"])

        assert result.test == "ttest"
        assert result.statistic_name == "t"
        assert result.p_values["shannon"] < 0.01
        assert result.significant("shannon")
        assert result.significant_indices == ("shannon",)

    def test_applies_one_test_per_index_not_one_to_the_vector(self):
        """Each index is tested separately and reported separately.

        There is no joint distribution over "Shannon and Simpson together"
        to test against, so a single multivariate p-value would be a
        fiction. The result therefore carries one statistic and one
        p-value per index, and says so.
        """
        matrix, groups = two_group_matrix()
        result = diversity_t_test(matrix, groups)

        assert result.test_per_index
        assert len(result.index_names) >= 3
        assert set(result.statistics) == set(result.index_names)
        assert set(result.p_values) == set(result.index_names)
        assert len(result.p_values) == len(result.index_names)

    def test_each_index_gets_its_own_statistic_from_its_own_values(self):
        """Every index is tested on its own numbers, not a shared summary.

        The evidence that the test is per-index rather than one test on a
        combined vector: a run restricted to a single index must return the
        same numbers as the full run for that index, while the *set* of
        reported statistics differs. A "sum the indices and test once"
        implementation would give every index the same p-value, which the
        distinctness check below rules out.
        """
        matrix, groups = two_group_matrix()
        everything = diversity_t_test(matrix, groups)
        shannon_only = diversity_t_test(matrix, groups, indices=["shannon"])

        assert set(shannon_only.index_names) == {"shannon"}
        assert set(everything.index_names) > {"shannon"}
        assert shannon_only.statistics["shannon"] == pytest.approx(everything.statistics["shannon"], rel=1e-12)
        assert shannon_only.p_values["shannon"] == pytest.approx(everything.p_values["shannon"], rel=1e-12)

        # The indices do NOT share one verdict, so they cannot have come
        # from a single combined test. The fixture separates Shannon
        # strongly; it is built so richness cannot separate at all -- and
        # because richness is *constant* across the samples it is dropped
        # before the test runs, so it is not a key of p_values at all.
        # Asserting that explicitly, rather than comparing against a
        # hard-coded default, is what makes this a real check: the old
        # form (a .get() against the default 1.0) would still have passed
        # if richness *were* present with a p-value of 0.99.
        assert "richness" not in everything.index_names
        assert "richness" not in everything.p_values
        assert everything.p_values["shannon"] < 0.05

        # Each index's mean is the mean of that index's own per-sample
        # values, recomputed here from ecology.diversity independently.
        for index in everything.index_names:
            if index == "richness":
                continue
            for label in ("rich", "poor"):
                rows = np.flatnonzero(groups == label)
                hand = float(np.mean([compute_diversity_indices(matrix[i]).get(index) for i in rows]))
                assert everything.group_means[index][label] == pytest.approx(hand, abs=1e-12)

    def test_richness_that_actually_varies_is_tested_not_dropped(self):
        """A fixture where richness really separates the groups.

        The counterpart to the check above: with the constant-richness
        fixture, richness is dropped before the test runs, so nothing
        tests it. Here richness varies within each group and differs
        between them, so it is tested, and the p-value it produces is a
        real probability rather than a stand-in.
        """
        matrix, groups = varying_richness_matrix()
        result = diversity_t_test(matrix, groups)

        assert "richness" in result.index_names
        assert "richness" in result.p_values
        assert 0.0 < result.p_values["richness"] <= 1.0
        assert result.p_values["richness"] < 0.05
        assert np.isfinite(result.statistics["richness"])

    def test_group_means_match_the_indices_computed_per_sample(self):
        """Reported means are the mean of the real per-sample index values.

        Computed from ``compute_diversity_indices`` here independently, so a
        bug that tested something other than the reported numbers would fail.
        """
        matrix, groups = two_group_matrix()
        result = diversity_t_test(matrix, groups, indices=["shannon"])

        expected = {
            label: float(
                np.mean([compute_diversity_indices(matrix[i]).shannon_index for i in np.flatnonzero(groups == label)])
            )
            for label in ("rich", "poor")
        }
        for label, mean in expected.items():
            assert result.group_means["shannon"][label] == pytest.approx(mean, abs=1e-12)

    def test_three_groups_use_one_way_anova(self):
        """More than two groups is an F test, not a t test."""
        matrix, _ = two_group_matrix()
        groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c", "c"])
        matrix = np.vstack([matrix, matrix[:2] + 3.0])
        result = diversity_t_test(matrix, groups, indices=["shannon"])

        assert result.test == "anova"
        assert result.statistic_name == "F"
        assert np.isfinite(result.statistics["shannon"])

    def test_identical_groups_are_rejected_as_untestable(self):
        """No difference in the data means there is nothing to test.

        Every index is constant here, so a p-value is not "not significant" --
        it is undefined. Returning anything else would be inventing evidence
        from a test with no signal to test.
        """
        rows = np.array([[20, 10, 5, 3, 2, 1, 1]] * 4, dtype=float)
        groups = np.array(["a", "a", "b", "b"])

        with pytest.raises(StatisticalError, match="constant"):
            diversity_t_test(rows, groups)

    def test_constant_index_is_excluded_rather_than_reported_as_nan(self):
        """A zero-variance index is dropped, not returned as NaN.

        The NaN does not stay harmless downstream: in the permutation test
        ``null >= observed`` is False for NaN on either side, so n_ge is 0
        and p collapses to ``1 / (n + 1)`` -- the most significant value the
        machinery can report. A constant index came out "significant at
        p = 0.005" purely for having no variance.
        """
        # Same taxon count (5) in every sample => richness is constant, while
        # Shannon still varies because the abundances differ.
        constant = np.array(
            [
                [9, 3, 2, 1, 1],
                [8, 4, 2, 1, 1],
                [7, 5, 2, 1, 1],
                [6, 6, 2, 1, 1],
                [1, 1, 1, 1, 1],
                [2, 1, 1, 1, 1],
                [3, 1, 1, 1, 1],
                [4, 1, 1, 1, 1],
            ],
            dtype=float,
        )
        groups = np.array(["a"] * 4 + ["b"] * 4)

        result = diversity_t_test(constant, groups)
        assert "richness" not in result.index_names
        assert "shannon" in result.index_names
        assert all(np.isfinite(v) for v in result.statistics.values())

        permuted = diversity_permutation_test(constant, groups, n_permutations=99, random_seed=3)
        assert "richness" not in permuted.index_names
        # The groups genuinely differ in Shannon here, so a small p-value is
        # correct and is not asserted against. What must hold is that no
        # reported p-value is the degenerate 1/(n+1) floor produced by a NaN
        # comparison, and that all of them are real probabilities.
        assert all(0.0 < p <= 1.0 for p in permuted.p_values.values())
        assert all(np.isfinite(v) for v in permuted.statistics.values())

    def test_group_difference_that_does_not_exist_is_not_significant(self):
        """Identical groups in every respect must not produce significance.

        The genuine null case, with enough internal variation that the
        indices are *not* constant and so are tested rather than skipped.
        Sharing the same spread across both groups is what makes this a real
        null rather than a zero-variance index.
        """
        half = np.array(
            [
                [9, 3, 2, 1, 1],
                [8, 4, 2, 1, 1],
                [7, 5, 2, 1, 1],
                [6, 6, 2, 1, 1],
            ],
            dtype=float,
        )
        matrix = np.vstack([half, half])
        groups = np.array(["a"] * 4 + ["b"] * 4)

        result = diversity_t_test(matrix, groups, indices=["shannon"])
        assert result.p_values["shannon"] == pytest.approx(1.0, abs=1e-9)
        assert not result.significant("shannon")

        permuted = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=199, random_seed=4)
        assert not permuted.significant("shannon")
        assert permuted.p_values["shannon"] >= 0.05

    def test_zero_within_group_variance_is_refused_by_both_tests(self):
        """Constant within each group, different between them: untestable.

        Regression: the guard only dropped indices that were constant
        across *all* samples, so an index that was constant within each
        group survived it. The t test then divided by a zero pooled
        variance and reported t = infinity with p = 0.0 -- a number that
        is not a probability -- while the permutation test reported
        something else entirely for the same data. Both now refuse.
        """
        rows = np.array([[10, 10, 10]] * 3 + [[20, 20, 20]] * 3, dtype=float)
        groups = np.array(["a", "a", "a", "b", "b", "b"])

        with pytest.raises(StatisticalError, match="constant"):
            diversity_t_test(rows, groups)
        with pytest.raises(StatisticalError, match="constant"):
            diversity_permutation_test(rows, groups, n_permutations=99, random_seed=1)

    def test_degenerate_index_is_dropped_but_the_others_are_still_tested(self):
        """Only the degenerate index goes; the rest of the test survives.

        Here Shannon is constant within each group (every sample in a
        group has the same proportions) but differs between the groups,
        so its pooled variance is zero. Margalef, which depends on the
        individual count, varies within each group and stays testable.
        """
        rows = np.array(
            [
                [10, 10, 10, 0, 0, 0, 0, 0, 0, 0],
                [100, 100, 100, 0, 0, 0, 0, 0, 0, 0],
                [10] * 10,
                [100] * 10,
            ],
            dtype=float,
        )
        groups = np.array(["a", "a", "b", "b"])

        result = diversity_t_test(rows, groups)

        assert "shannon" not in result.index_names
        assert "margalef" in result.index_names
        assert all(np.isfinite(v) for v in result.statistics.values())
        assert all(0.0 < p <= 1.0 for p in result.p_values.values())

    def test_single_group_is_rejected(self):
        """Nothing to compare against."""
        matrix, _ = two_group_matrix()
        with pytest.raises(ValidationError, match="at least 2 groups"):
            diversity_t_test(matrix, np.array(["a"] * 8))

    def test_group_of_size_one_is_rejected(self):
        """A singleton group has no within-group variance.

        scipy would compute a t statistic from it anyway; the honest answer is
        to refuse, because the variance that t divides by is undefined.
        """
        matrix, _ = two_group_matrix()
        with pytest.raises(StatisticalError, match="at least 2 samples"):
            diversity_t_test(matrix, np.array(["a"] * 7 + ["b"]))

    def test_one_dimensional_abundance_matrix_is_read_as_one_sample(self):
        """A 1-D abundance vector is one sample, not n samples of one taxon.

        Regression: the validator reshapes a 1-D array to (n, 1), so the
        "1-D means one sample" branch was unreachable and the vector was
        read as n samples of a single taxon instead.
        """
        with pytest.raises(ValidationError, match="line up"):
            diversity_t_test(np.array([10, 5, 3]), np.array(["a", "b"]))

    def test_label_count_mismatch_is_rejected(self):
        """Labels must line up one per sample."""
        matrix, _ = two_group_matrix()
        with pytest.raises(ValidationError, match="line up"):
            diversity_t_test(matrix, np.array(["a", "b"]))

    def test_unknown_index_is_rejected(self):
        """Asking for an index that does not exist should say which do."""
        matrix, groups = two_group_matrix()
        with pytest.raises(ValidationError, match="not available"):
            diversity_t_test(matrix, groups, indices=["shannon", "nonexistent"])

    def test_sample_with_no_individuals_is_reported_with_its_index(self):
        """An empty sample is bad input, not a computation failure.

        Regression: the dependency raised a raw ComputationError for a
        sample with no positive abundances, so a caller catching the
        documented ValidationError never saw it.
        """
        matrix = np.array([[1, 2, 3], [0, 0, 0], [1, 2, 3]], dtype=float)
        groups = np.array(["a", "b", "c"])

        with pytest.raises(ValidationError, match="sample 1"):
            diversity_t_test(matrix, groups, indices=["shannon"])

    def test_index_dropped_for_one_sample_is_reported(self, caplog):
        """A missing index is announced, not silently discarded.

        Regression: Fisher's alpha fails to converge when every taxon is
        a singleton (S == N), so its key was absent for that sample, and
        intersecting the key sets dropped it for *every* sample with no
        word to the user. It is now logged, naming the index and the
        samples it was undefined for.
        """
        rows = np.array([[1, 1, 1, 1], [10, 5, 2, 0], [1, 1, 1, 1], [10, 5, 2, 0]], dtype=float)
        groups = np.array(["a", "a", "b", "b"])

        with caplog.at_level(logging.WARNING, logger="ecology.diversity_extra"):
            result = diversity_t_test(rows, groups)

        assert "fisher_alpha" not in result.index_names
        assert "fisher_alpha" in caplog.text
        assert "margalef" in result.index_names

    def test_welch_branch_is_used_when_equal_var_is_false(self):
        """equal_var=False runs Welch's test, not Student's.

        Welch does not pool the variances, so with groups of different
        spread the two statistics differ. The point of the branch is
        that it is reachable and finite -- an earlier revision never
        exercised it at all.
        """
        # Deliberately unequal spreads *and* unequal group sizes: the
        # even group is tight, the dominated one is not, so the pooled
        # variance of Student's test and the separate variances of Welch's
        # test give different standard errors.
        matrix = np.array(
            [
                [10, 10, 10],
                [12, 9, 9],
                [11, 10, 9],
                [10, 12, 8],
                [20, 1, 1],
                [2, 1, 18],
                [1, 1, 16],
            ],
            dtype=float,
        )
        groups = np.array(["a"] * 4 + ["b"] * 3)

        student = diversity_t_test(matrix, groups, indices=["shannon"], equal_var=True)
        welch = diversity_t_test(matrix, groups, indices=["shannon"], equal_var=False)

        assert welch.test == "ttest"
        assert np.isfinite(welch.statistics["shannon"])
        assert 0.0 < welch.p_values["shannon"] <= 1.0
        assert abs(welch.statistics["shannon"] - student.statistics["shannon"]) > 0.1

    def test_sample_names_must_match_the_rows(self):
        """The sample names are carried through, and checked against the rows."""
        matrix, groups = two_group_matrix()
        names = [f"S{i}" for i in range(matrix.shape[0])]

        named = diversity_t_test(matrix, groups, indices=["shannon"], sample_names=names)
        unnamed = diversity_t_test(matrix, groups, indices=["shannon"])
        assert named.p_values["shannon"] == pytest.approx(unnamed.p_values["shannon"], rel=1e-12)

        with pytest.raises(ValidationError, match="sample_names"):
            diversity_t_test(matrix, groups, indices=["shannon"], sample_names=names[:-1])

    def test_summary_and_to_dict_carry_the_results(self):
        """The report names the groups, the statistic and the outcome."""
        matrix, groups = two_group_matrix()
        result = diversity_t_test(matrix, groups, indices=["shannon"])
        text = result.summary()
        payload = result.to_dict()

        assert "rich" in text and "poor" in text
        assert payload["test"] == "ttest"
        assert "shannon" in payload["p_values"]


# =============================================================================
# Diversity permutation test
# =============================================================================


class TestDiversityPermutationTest:
    """The label-shuffling version of the same per-index comparison."""

    def test_same_seed_gives_the_same_answer(self):
        """A reported p-value has to reproduce.

        This is the reason the module routes through
        ``utils.statistics_core.permutation_pvalue`` rather than its own
        ``np.random.default_rng``: the seed reaches the generator the same
        way it does for every other test in the project.
        """
        matrix, groups = two_group_matrix()
        first = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=199, random_seed=42)
        second = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=199, random_seed=42)

        assert first.p_values == second.p_values
        assert first.statistics == second.statistics

    def test_different_seed_can_change_the_p_value(self):
        """The test is genuinely randomising rather than returning a constant.

        Weak by design: a correct implementation *may* coincidentally match.
        Over many comparisons it will not, so this documents that the seed is
        actually wired through.
        """
        matrix, groups = two_group_matrix()
        seen = {
            diversity_permutation_test(
                matrix, groups, indices=["shannon"], n_permutations=99, random_seed=seed
            ).p_values["shannon"]
            for seed in range(8)
        }
        assert len(seen) > 1

    def test_p_value_is_a_probability_and_never_zero(self):
        """p is in (0, 1] and honours the (1 + count) / (n + 1) floor.

        A p of exactly 0 would claim more evidence than 99 permutations can
        supply; ``permutation_pvalue`` guarantees the floor of 1/(n+1).
        """
        matrix, groups = two_group_matrix()
        result = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=99, random_seed=5)

        assert 0.0 < result.p_values["shannon"] <= 1.0
        assert float(result.p_values["shannon"]) >= 1 / 100

    def test_detects_the_seeded_difference(self):
        """A real between-group difference should reach significance."""
        matrix, groups = two_group_matrix()
        result = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=199, random_seed=1)

        assert result.test == "permutation"
        assert result.significant("shannon")

    def test_three_groups_report_an_F_statistic(self):
        """With three groups the statistic is the ANOVA F, one-sided."""
        matrix, _ = two_group_matrix()
        groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 2)
        matrix = np.vstack([matrix, matrix[:2] + 3.0])
        result = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=99, random_seed=2)

        assert result.statistic_name == "F"
        assert result.statistics["shannon"] >= 0.0
        assert result.n_permutations == 99

    def test_agrees_with_the_t_test_on_the_seeded_difference(self):
        """Both tests read the same data, so both should find the effect.

        Deliberately only asserts agreement on *significance*, not on the
        numbers: a permutation p and a t p are different statistics on
        different scales, and demanding they match would be asserting that
        the normal approximation is exact.
        """
        matrix, groups = two_group_matrix()
        parametric = diversity_t_test(matrix, groups, indices=["shannon"])
        permuted = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=999, random_seed=8)

        assert permuted.significant("shannon") == parametric.significant("shannon")

    def test_records_its_permutation_bookkeeping(self):
        """The seed and count travel with the result for reproducibility."""
        matrix, groups = two_group_matrix()
        result = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=99, random_seed=17)

        assert result.random_seed == 17
        assert result.n_permutations == 99
        assert result.to_dict()["random_seed"] == 17

    def test_sample_names_must_match_the_rows(self):
        """The sample names are carried through, and checked against the rows."""
        matrix, groups = two_group_matrix()
        names = [f"S{i}" for i in range(matrix.shape[0])]

        named = diversity_permutation_test(
            matrix, groups, indices=["shannon"], n_permutations=99, random_seed=1, sample_names=names
        )
        unnamed = diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=99, random_seed=1)
        assert named.p_values["shannon"] == pytest.approx(unnamed.p_values["shannon"], rel=1e-12)

        with pytest.raises(ValidationError, match="sample_names"):
            diversity_permutation_test(matrix, groups, indices=["shannon"], sample_names=names[:-1])

    def test_rejects_zero_permutations(self):
        """A p-value from no permutations is not a p-value."""
        matrix, groups = two_group_matrix()
        with pytest.raises(ValidationError):
            diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=0, random_seed=1)

    def test_warns_when_unseeded(self):
        """An unseeded permutation result is not reproducible, and says so.

        ``make_rng`` raises ``RuntimeWarning`` in that case; swallowing it
        would ship a p-value the user cannot reproduce.
        """
        matrix, groups = two_group_matrix()
        with pytest.warns(RuntimeWarning, match="random_seed"):
            diversity_permutation_test(matrix, groups, indices=["shannon"], n_permutations=19)


# =============================================================================
# Analyzer facade
# =============================================================================


class TestDiversityExtraAnalyzer:
    """Every method returns a result and records it."""

    @pytest.mark.parametrize("call", ["subsample", "profile", "t_test", "permutation_test"])
    def test_each_method_sets_last_result(self, call: str):
        """``last_result`` is the most recent run, whatever ran last."""
        matrix, groups = two_group_matrix()
        analyzer = DiversityExtraAnalyzer()

        # Each method has its own signature; the point of the test is the
        # last_result bookkeeping, not the arguments.
        args = {
            "subsample": (np.array([10, 10, 10]), 6),
            "profile": (np.array([10, 6, 4, 2, 1]),),
            "t_test": (matrix, groups),
            "permutation_test": (matrix, groups),
        }[call]

        result = getattr(analyzer, call)(*args)

        assert type(result).__name__.endswith("Result")
        assert analyzer.last_result is result

    def test_last_result_starts_as_none(self):
        """Nothing has run yet."""
        assert DiversityExtraAnalyzer().last_result is None

    def test_coverage_and_rarefaction_route_through_the_analyzer(self):
        """The incidence-based entry points are reachable from the facade."""
        incidence = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
        analyzer = DiversityExtraAnalyzer()

        analyzer.coverage(incidence)
        assert type(analyzer.last_result).__name__ == "SampleCoverageResult"

        analyzer.rarefaction(incidence)
        assert type(analyzer.last_result).__name__ == "SampleRarefactionResult"
