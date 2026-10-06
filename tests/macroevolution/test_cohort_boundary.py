"""
================================================================================
Tests for cohort boundary handling (bug 5)
================================================================================

Bug 5: ``CohortSurvivorshipAnalysis.analyze`` uses ``started_before = o > t_end``
(strict ``>``) for the boundary classification. A taxon whose first-appearance
datum (FAD) lands *exactly* on the oldest bin's ``t_end`` becomes invisible to
every bin: ``o > t_end`` is false (it equals), ``t_start <= o < t_end`` is also
false (it equals the right endpoint), and ``o < t_start`` is false (it's
strictly older than the bin's young edge). The same is true for ``L == t_end``
when ``L`` equals the old boundary of any other bin. The module docstring
acknowledges this convention but ``analyze``'s own docstring promised a
non-trivial example on the very data that triggers the boundary.

The fix is NOT to switch to ``>=`` (that double-counts overlapping
intervals); it is to detect the boundary silently-dropped case and emit a
``UserWarning`` so the caller can choose to widen their bin grid.
"""

from __future__ import annotations

import warnings

from macroevolution.cohort import CohortSurvivorshipAnalysis


class TestCohortBoundaryTaxaWarning:
    """analyze should warn when taxa fall on a boundary outside the bin grid."""

    def test_fad_at_oldest_bin_t_end_warns(self):
        """A taxon with FAD exactly equal to the oldest bin's t_end would be
        invisible to every bin under the strict-``>`` convention. The
        analyzer must warn."""
        # Intervals (5, 10] -> t_start=5, t_end=10 (oldest bin: 5 to 10).
        # Taxon: FAD=10 (exactly equal to t_end), LAD=0 (extant).
        records = [(10.0, 0.0)]
        intervals = [(5.0, 10.0)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CohortSurvivorshipAnalysis().analyze(records, intervals)

        warnings_text = [str(w.message) for w in caught]
        # At least one warning should mention the boundary issue.
        assert any(
            "FAD" in msg or "boundary" in msg.lower() or "t_end" in msg
            for msg in warnings_text
        ), f"expected boundary warning, got: {warnings_text}"

    def test_lad_at_oldest_bin_t_start_warns(self):
        """A taxon with LAD exactly equal to the oldest bin's t_start would
        similarly be invisible. The analyzer must warn."""
        # Bin (5, 10]: t_start=5, t_end=10. Taxon: FAD=15 (way older),
        # LAD=5 (extinct at exactly the young edge).
        records = [(15.0, 5.0)]
        intervals = [(5.0, 10.0)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CohortSurvivorshipAnalysis().analyze(records, intervals)

        warnings_text = [str(w.message) for w in caught]
        assert any(
            "LAD" in msg or "boundary" in msg.lower() or "t_start" in msg or "t_end" in msg
            for msg in warnings_text
        ), f"expected boundary warning, got: {warnings_text}"

    def test_no_warning_when_all_taxa_visible(self):
        """When every taxon falls strictly inside a bin, no warning fires."""
        records = [(7.5, 0.0), (6.0, 0.0)]
        intervals = [(5.0, 10.0)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CohortSurvivorshipAnalysis().analyze(records, intervals)

        # No FAD/LAD boundary warning expected.
        relevant = [
            w for w in caught
            if "FAD" in str(w.message) or "LAD" in str(w.message)
            or "boundary" in str(w.message).lower()
        ]
        assert len(relevant) == 0, f"unexpected warnings: {[str(w.message) for w in relevant]}"


class TestCohortBoundaryTaxaDoNotDoubleCount:
    """Switching to ``>=`` would double-count on overlapping bins; the warning
    fix must not introduce that regression."""

    def test_overlapping_intervals_do_not_double_count(self):
        # Two bins sharing an edge: (0,5] and (5,10]. A taxon originating at
        # exactly FAD=5 must be visible in bin (5,10] only, not in both.
        records = [(5.0, 0.0)]
        intervals = [(0.0, 5.0), (5.0, 10.0)]

        result = CohortSurvivorshipAnalysis().analyze(records, intervals)
        # n_total of the (5,10] bin should be 1 (the taxon originates there).
        # n_total of the (0,5] bin should be 0 (it falls on the old edge,
        # not strictly inside).
        assert result.intervals[0].n_total == 0
        assert result.intervals[1].n_total == 1
