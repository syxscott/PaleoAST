"""
================================================================================
Tests for the normalized Robinson-Foulds distance
================================================================================

Bug 4: the raw unweighted Robinson-Foulds distance is the symmetric
difference of split sets, but it is NOT comparable across trees of different
size.  For two unrooted binary trees over ``n`` leaves the maximum possible
RF is ``2 * (n - 3)``; for non-binary trees it is smaller (every internal
node past the binary ones collapses splits).  This module adds a normalized
metric ``normalized_robinson_foulds_distance`` that divides by the maximum
and so returns a value in [0, 1].
"""

from __future__ import annotations

import pytest

from phylogenetics.tree import PhyloTree
from phylogenetics.tree_distance import (
    normalized_robinson_foulds_distance,
    robinson_foulds_distance,
)


def tree(newick: str) -> PhyloTree:
    return PhyloTree.from_newick(newick)


class TestNormalizedRobinsonFouldsIdentical:
    def test_identical_binary_trees_zero(self):
        t = tree("((A,B),(C,D));")
        assert normalized_robinson_foulds_distance(t, t) == 0.0

    def test_identical_polytomy_zero(self):
        t = tree("(A,B,C,D);")
        assert normalized_robinson_foulds_distance(t, t) == 0.0

    def test_rotation_invariance_binary(self):
        # Rotation is a relabelling of branches -- same splits.
        first = tree("((A,B),(C,D));")
        second = tree("((D,C),(B,A));")
        assert normalized_robinson_foulds_distance(first, second) == 0.0


class TestNormalizedRobinsonFouldsMaximum:
    def test_maximum_for_binary_trees_is_one(self):
        # Two maximally-different binary trees over 4 leaves:
        # ((A,B),(C,D)) vs ((A,D),(B,C)). All splits disagree -> max.
        first = tree("((A,B),(C,D));")
        second = tree("((A,D),(B,C));")
        # For n=4: max_RF = 2*(n-3) = 2.
        assert robinson_foulds_distance(first, second) == 2
        assert normalized_robinson_foulds_distance(first, second) == 1.0

    def test_maximum_for_five_taxa_binary(self):
        # For n=5: max_RF = 2*(5-3) = 4. Pick a cross comparison.
        first = tree("((A,B),((C,D),E));")
        second = tree("((D,E),((A,B),C));")
        rf = robinson_foulds_distance(first, second)
        assert rf > 0
        norm = normalized_robinson_foulds_distance(first, second)
        # Normalized should be <= 1.
        assert 0.0 <= norm <= 1.0


class TestNormalizedRobinsonFouldsBoundary:
    def test_three_taxa_returns_zero(self):
        # n=3 has no non-trivial splits for any tree -> RF always 0.
        first = tree("(A,B,C);")
        second = tree("(A,B,C);")
        assert normalized_robinson_foulds_distance(first, second) == 0.0

    def test_single_leaf_returns_zero(self):
        # No non-trivial splits for a single-leaf tree.
        first = tree("A:0.1;")
        second = tree("A:0.1;")
        assert normalized_robinson_foulds_distance(first, second) == 0.0

    def test_two_leaves_returns_zero(self):
        # No non-trivial splits for a two-leaf tree.
        first = tree("(A:0.1,B:0.2);")
        second = tree("(A:0.1,B:0.2);")
        assert normalized_robinson_foulds_distance(first, second) == 0.0


class TestNormalizedRobinsonFouldsInBounds:
    @pytest.mark.parametrize(
        "newick1,newick2",
        [
            ("((A,B),(C,D));", "((A,C),(B,D));"),
            ("((A,B),(C,D));", "((A,B,C),D);"),
            ("(((A,B),C),D);", "(A,B,(C,D));"),
            ("((A,B),(C,(D,E)));", "((A,C),(B,(D,E)));"),
        ],
    )
    def test_result_in_unit_interval(self, newick1, newick2):
        first = tree(newick1)
        second = tree(newick2)
        result = normalized_robinson_foulds_distance(first, second)
        assert 0.0 <= result <= 1.0


class TestRawRobinsonFouldsBackwardCompatible:
    """Bug 4 requirement: raw RF must remain bit-identical."""

    def test_raw_rf_unchanged(self):
        first = tree("((A,B),(C,D));")
        second = tree("((A,C),(B,D));")
        assert robinson_foulds_distance(first, second) == 2

    def test_polytomy_raw_rf_unchanged(self):
        first = tree("((A,B),(C,D),(E,F));")
        second = tree("((A,B),(C,E),(D,F));")
        assert robinson_foulds_distance(first, second) == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))