# tests/ecology/test_diversity_standardization.py
"""
Regression tests for cross-sample diversity standardization (缺陷 4).

Raw Shannon/Simpson are biased by sample size — a sample with N=10 and
one with N=1000 give different Shannon even when drawn from the same
underlying community. ``analyze_multiple`` previously returned only the
raw indices. The fix adds coverage-based standardized Hill numbers
(Chao & Jost 2012; Chao et al. 2014) in ``DiversityResult.metadata``,
while keeping the raw path intact for backward compatibility.
"""

import logging

import numpy as np
import pytest

from ecology.diversity import DiversityAnalyzer


class TestRawPathBackwardCompatible:
    """The raw-indices path must keep its existing shape."""

    def test_raw_path_returns_list(self):
        abundance = np.array(
            [
                [10, 5, 0, 0],
                [8, 0, 3, 0],
                [0, 2, 0, 7],
            ]
        )
        results = DiversityAnalyzer().analyze_multiple(
            abundance, compute_standardized=False
        )
        assert len(results) == 3
        for r in results:
            assert "shannon" in r.indices
            assert "simpson" in r.indices


class TestStandardizedHill:
    """Coverage-based Hill numbers must appear in metadata."""

    def test_metadata_has_q0_q1_q2(self):
        abundance = np.array(
            [
                [10, 5, 2, 1, 0, 0],
                [100, 50, 20, 10, 0, 0],  # 10× N — should NOT have 10× Shannon
                [50, 30, 15, 5, 5, 1],
            ]
        )
        results = DiversityAnalyzer().analyze_multiple(
            abundance,
            compute_standardized=True,
            coverage_levels=(0.50, 0.90),
        )
        for r in results:
            assert "standardized_hill" in r.metadata
            sh = r.metadata["standardized_hill"]
            assert "q0" in sh
            assert "q1" in sh
            assert "q2" in sh
            assert "C=0.50" in sh["q0"]
            assert "C=0.90" in sh["q0"]
            assert "asymptote" in sh
            assert "q0" in sh["asymptote"]
            assert "q1" in sh["asymptote"]
            assert "q2" in sh["asymptote"]

    def test_unequal_sample_sizes_trigger_note(self, caplog):
        """Sample-size inequality must be flagged — comparing raw indices
        across unequal-N samples is the whole reason the standardized
        path exists."""
        abundance = np.array(
            [
                [10, 5, 2],
                [200, 100, 50],  # 20× N
            ]
        )
        with caplog.at_level(logging.WARNING):
            results = DiversityAnalyzer().analyze_multiple(abundance)
        notes = [
            r.metadata.get("standardized_hill", {}).get("note")
            for r in results
        ]
        assert any(n is not None and "differ" in n for n in notes)

    def test_equal_sample_sizes_no_note(self):
        abundance = np.array(
            [
                [10, 5, 2, 1],  # N = 18
                [8, 6, 3, 1],   # N = 18
            ]
        )
        results = DiversityAnalyzer().analyze_multiple(abundance)
        for r in results:
            meta = r.metadata.get("standardized_hill", {})
            # Note is only set when sizes differ
            assert "note" not in meta

    def test_standardized_q0_is_finite(self):
        abundance = np.array(
            [
                [10, 5, 2, 1, 0, 0],
                [10, 4, 3, 2, 1, 0],
                [12, 8, 5, 3, 1, 1],
            ]
        )
        results = DiversityAnalyzer().analyze_multiple(
            abundance, coverage_levels=(0.50,)
        )
        for r in results:
            v = r.metadata["standardized_hill"]["q0"]["C=0.50"]
            assert np.isfinite(v)
            assert v > 0

    def test_asymptote_greater_than_or_equal_to_observed(self):
        """The asymptotic Hill number should be at least as large as the
        sample's observed richness (the asymptote bounds the curve from
        above)."""
        abundance = np.array(
            [
                [10, 5, 2, 1, 0, 0, 0, 0],
                [10, 4, 3, 2, 1, 0, 0, 0],
            ]
        )
        results = DiversityAnalyzer().analyze_multiple(abundance)
        for r in results:
            asymp = r.metadata["standardized_hill"]["asymptote"]["q0"]
            assert asymp >= r.taxa_count - 1e-9
