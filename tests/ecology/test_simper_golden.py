# =============================================================================
# FILE: tests/ecology/test_simper_golden.py
# =============================================================================
"""
Unit tests for SIMPER implementation with golden values.

Golden values are derived from manual calculations verified against
the Clarke 1993 SIMPER methodology.

Reference: Clarke (1993) Non-parametric multivariate analyses of
changes in community structure. Australian Journal of Ecology, 18, 117-143.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statistics.simper import SimperAnalyzer


class TestSimperGoldenValues:
    """Test SIMPER against known golden values."""

    def test_simper_single_pair(self):
        """Single pair case: [3,2,1] vs [1,2,3].

        Bray-Curtis:
        - |3-1| + |2-2| + |1-3| = 4
        - (3+1) + (2+2) + (1+3) = 12
        - BC = 4/12 = 0.333

        Species contributions (Clarke 1993: |diff| / sum):
        - Sp0: |3-1|/12 = 2/12
        - Sp1: |2-2|/12 = 0
        - Sp2: |1-3|/12 = 2/12
        - Total: 4/12 = BC (contributions sum to the dissimilarity)
        """
        data = np.array([
            [3.0, 2.0, 1.0],  # Group A
            [1.0, 2.0, 3.0],  # Group B
        ])
        groups = [0, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups, variable_names=["Sp0", "Sp1", "Sp2"])

        # Overall dissimilarity should be Bray-Curtis
        assert_allclose(result.overall_dissimilarity, 4.0/12.0, atol=1e-6)

        # Species contributions sum to the overall dissimilarity
        total_contrib = sum(c.average for c in result.contributions)
        assert_allclose(total_contrib, 4.0/12.0, atol=1e-6)

        contribs = {c.name: c.average for c in result.contributions}
        assert_allclose(contribs["Sp0"], 2.0/12.0, atol=1e-6)
        assert_allclose(contribs["Sp1"], 0.0, atol=1e-6)
        assert_allclose(contribs["Sp2"], 2.0/12.0, atol=1e-6)

    def test_simper_identical_samples(self):
        """Identical samples should have zero dissimilarity."""
        data = np.array([
            [5.0, 3.0, 2.0],
            [5.0, 3.0, 2.0],
        ])
        groups = [0, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        # Identical samples have BC = 0
        assert_allclose(result.overall_dissimilarity, 0.0, atol=1e-6)

    def test_simper_no_overlap(self):
        """No shared species: [1,0] vs [0,1].

        Bray-Curtis = 1.0 (no overlap)
        """
        data = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        groups = [0, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        # No overlap means BC = 1.0
        assert_allclose(result.overall_dissimilarity, 1.0, atol=1e-6)

    def test_simper_multiple_pairs(self):
        """Multiple pairs: verify mean calculation.

        Group A: [5,1,0], [4,2,0]
        Group B: [1,5,0], [0,4,2]

        Pairwise Bray-Curtis:
        (0,2): (|5-1| + |1-5| + |0-0|) / (5+1 + 1+5 + 0+0) = 8/12 = 0.667
        (0,3): (|5-0| + |1-4| + |0-2|) / (5+0 + 1+4 + 0+2) = 10/12 = 0.833
        (1,2): (|4-1| + |2-5| + |0-0|) / (4+1 + 2+5 + 0+0) = 6/12 = 0.500
        (1,3): (|4-0| + |2-4| + |0-2|) / (4+0 + 2+4 + 0+2) = 8/12 = 0.667

        Overall = mean(0.667, 0.833, 0.500, 0.667) = 2.667/4 = 0.667
        """
        data = np.array([
            [5.0, 1.0, 0.0],  # Group A
            [4.0, 2.0, 0.0],  # Group A
            [1.0, 5.0, 0.0],  # Group B
            [0.0, 4.0, 2.0],  # Group B
        ])
        groups = [0, 0, 1, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        expected_overall = (8/12 + 10/12 + 6/12 + 8/12) / 4
        assert_allclose(result.overall_dissimilarity, expected_overall, atol=1e-4)


class TestSimperSpeciesContributionFormula:
    """Verify the species contribution formula."""

    def test_min_vs_absolute_difference(self):
        """Species contribution uses |diff| (Clarke 1993), not 2*min.

        For [3,0] vs [0,3]:
        - BC = (|3-0| + |0-3|) / (3+0 + 0+3) = 6/6 = 1.0
        - Sp0: |3-0|/6 = 0.5, Sp1: |0-3|/6 = 0.5
        - Total: 1.0 = BC — the two species jointly account for all
          of the dissimilarity (a 2*min implementation would wrongly
          report 0 contribution from each).
        """
        data = np.array([
            [3.0, 0.0],
            [0.0, 3.0],
        ])
        groups = [0, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups, variable_names=["Sp0", "Sp1"])

        # BC = 1.0, contributions = 0.5 each
        assert_allclose(result.overall_dissimilarity, 1.0, atol=1e-6)

        contribs = {c.name: c.average for c in result.contributions}
        assert_allclose(contribs["Sp0"], 0.5, atol=1e-6)
        assert_allclose(contribs["Sp1"], 0.5, atol=1e-6)

    def test_full_overlap(self):
        """Full overlap: identical samples [2,3] vs [2,3].

        BC = 0 and every species contribution = 0.
        """
        data = np.array([
            [2.0, 3.0],
            [2.0, 3.0],
        ])
        groups = [0, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        assert_allclose(result.overall_dissimilarity, 0.0, atol=1e-6)

        total_contrib = sum(c.average for c in result.contributions)
        assert_allclose(total_contrib, 0.0, atol=1e-6)


class TestSimperCumulativeContributions:
    """Test cumulative contribution calculations."""

    def test_cumulative_sums_to_one(self):
        """Cumulative contributions should sum to 1.0 (100%)."""
        data = np.array([
            [5.0, 3.0, 2.0, 1.0],
            [4.0, 2.0, 3.0, 1.0],
            [1.0, 5.0, 2.0, 3.0],
            [2.0, 4.0, 1.0, 3.0],
        ])
        groups = [0, 0, 1, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        # The last contribution should have cumulative = 1.0
        last_cumulative = result.contributions[-1].cumulative
        assert_allclose(last_cumulative, 1.0, atol=1e-6)

    def test_contributions_sorted_by_average(self):
        """Contributions should be sorted by average descending."""
        data = np.array([
            [5.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [1.0, 5.0, 2.0],
            [2.0, 3.0, 3.0],
        ])
        groups = [0, 0, 1, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        # Check sorted order
        for i in range(len(result.contributions) - 1):
            assert result.contributions[i].average >= result.contributions[i+1].average


class TestSimperEdgeCases:
    """Test SIMPER with edge cases."""

    def test_single_sample_per_group(self):
        """Test with single sample per group."""
        data = np.array([
            [5.0, 2.0],
            [2.0, 5.0],
        ])
        groups = [0, 1]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        assert result.n_groups == 2
        assert result.n_variables == 2
        assert len(result.contributions) == 2

    def test_three_groups(self):
        """Test with three groups (three pairs)."""
        data = np.array([
            [5.0, 2.0],
            [4.0, 3.0],
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [3.0, 3.0],
        ])
        groups = [0, 0, 1, 1, 2, 2]

        analyzer = SimperAnalyzer()
        result = analyzer.analyze(data, groups)

        # Should have 3 pairs: (0,1), (0,2), (1,2)
        assert len(result.group_pairs) == 3
        assert result.n_groups == 3
