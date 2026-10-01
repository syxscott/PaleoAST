"""
================================================================================
Tests for Fitch parsimony down-pass correctness on multifurcating trees
================================================================================

Bug 1 regression: the previous down-pass folded children left-to-right, so a
union with the *previous* ``combined`` was repeatedly intersected with the
*next* child. Once ``combined`` became a union like ``{A, T}`` and the next
child was ``{T}``, the intersection ``{T}`` discarded the previously absorbed
``A``.  Per Fitch (1971) the operation is on the *whole* child set: the
parent state set is the intersection of *all* children's state sets; if that
intersection is empty, the parent state set is the union of *all* children's
state sets.

The bug surfaces most clearly in the *internal* state sets of the down-pass
(``FitchResult.character_states[site][node]``). The parsimony score happens to
be hidden from the bug for many inputs because ``_pick_state`` uses leaf
support counts, but the recorded state sets — which feed ancestral-state
reconstruction and CI/RI bookkeeping — are wrong whenever the leaf ordering
out-of-order lets the left-fold drop a state that was earlier absorbed by a
union.  These tests assert the state sets directly, plus the binary-tree
parsimony scores that must stay bit-identical.

Reference: Fitch, W. M. (1971). Toward defining the course of evolution:
minimum change for a specific tree topology. Systematic Zoology 20:406-416.
"""

from __future__ import annotations

import pytest

from phylogenetics.fitch import FitchAlgorithm
from phylogenetics.tree import PhyloTree


def _compute(newick: str, sequences: dict[str, str], **kwargs):
    """Return the FitchResult for the given tree and data."""
    tree = PhyloTree.from_newick(newick)
    return FitchAlgorithm().compute(tree, sequences, **kwargs)


def _root_states(result, site: int = 0) -> set:
    return set(result.character_states[site][result.character_states[site].__iter__().__next__() and
                                          next(n for n, s in result.character_states[site].items()
                                               if s) or result.character_states[site].__iter__().__next__()])


def _root_states_simple(result, site: int = 0):
    """Return the state set at the root of the tree for the given site."""
    tree = result.character_states[site]
    # The root is the unique node with no parent — find it by traversing.
    for node in tree:
        if node.parent is None:
            return tree[node]
    raise RuntimeError("No root found")


class TestFitchBinaryUnchanged:
    """Binary trees must yield bit-identical results to the old behaviour."""

    def test_identical_sequences_zero_length(self):
        result = _compute("((A,B),(C,D));", {"A": "AATT", "B": "AATT", "C": "AATT", "D": "AATT"})
        assert result.tree_length == 0

    def test_one_difference_one_step(self):
        # Binary tree ((A,B),(C,D)) with one change (C vs A at position 0).
        result = _compute("((A,B),(C,D));", {"A": "AATT", "B": "AATT", "C": "CATT", "D": "CATT"})
        assert result.tree_length == 1

    def test_caterpillar_four_steps(self):
        # Binary caterpillar (((A,B),C),D): 4 sites, each requires 1 step.
        result = _compute("(((A,B),C),D);", {"A": "AAAA", "B": "AAAA", "C": "AAAA", "D": "TTTT"})
        assert result.tree_length == 4

    def test_caterpillar_one_step(self):
        # 1 site only.
        result = _compute("(((A,B),C),D);", {"A": "A", "B": "A", "C": "A", "D": "T"})
        assert result.tree_length == 1


class TestFitchMultifurcationStateSet:
    """On multifurcating trees, the down-pass must record the correct state set.

    The buggy implementation left-folds the children: ``combined = {first}``,
    then ``combined = combined & child_i`` when non-empty, else
    ``combined | child_i``. This loses any state absorbed by an earlier
    union as soon as a later child intersects it.

    The parsimony score is often masked by ``_pick_state``'s leaf-support
    logic, but the recorded ``character_states`` (which drive ancestral-state
    reconstruction and downstream CI/RI bookkeeping) are directly wrong.
    """

    def test_four_way_polytomy_two_states(self):
        # (A,B,C,D); one site: A, A, T, T.
        # Correct Fitch: intersect of all four = {}, parent = {A, T}.
        # Buggy: combined starts {A}. & {A} -> {A}. & {T} empty -> {A,T}.
        # & {T} -> {T}. Parent = {T}, dropping A.
        result = _compute("(A,B,C,D);", {"A": "X", "B": "X", "C": "Y", "D": "Y"})
        assert _root_states_simple(result, 0) == {"X", "Y"}

    def test_four_way_polytomy_state_loss_pattern(self):
        # The classic bug-trigger pattern: first leaf is X, the rest Y.
        # Buggy left-fold: {X} & {Y} empty -> {X,Y}, then & {Y} -> {Y}.
        # Correct: {X,Y} (intersection of all is empty, union of all is {X,Y}).
        result = _compute("(A,B,C,D);", {"A": "X", "B": "Y", "C": "Y", "D": "Y"})
        assert _root_states_simple(result, 0) == {"X", "Y"}

    def test_five_way_polytomy_three_states(self):
        # 5-way polytomy: A has X, B has Y, C has Z, D and E have Y.
        # Buggy: {X} & {Y} -> {X,Y}, & {Z} -> {X,Y,Z}, & {Y} -> {Y}, & {Y} -> {Y}.
        # Correct: intersection of all = {}, union of all = {X,Y,Z}.
        result = _compute("(A,B,C,D,E);",
                          {"A": "X", "B": "Y", "C": "Z", "D": "Y", "E": "Y"})
        assert _root_states_simple(result, 0) == {"X", "Y", "Z"}

    def test_polytomy_full_union(self):
        # All leaves identical -> intersection equals union.
        result = _compute("(A,B,C,D);", {"A": "A", "B": "A", "C": "A", "D": "A"})
        assert _root_states_simple(result, 0) == {"A"}


class TestFitchGapAndMissing:
    """The fix must not regress existing behaviour for ``?`` and ``-``."""

    def test_missing_char_is_skipped(self):
        # Missing characters are empty sets, so the root's state set is
        # determined by the remaining leaves alone.
        result = _compute("(A,B,C,D);", {"A": "A", "B": "A", "C": "A", "D": "?"})
        assert _root_states_simple(result, 0) == {"A"}

    def test_gap_as_missing_default(self):
        # gap_as_missing=True (default): '-' is treated like '?' -> empty set.
        result = _compute("(A,B,C,D);", {"A": "A", "B": "A", "C": "A", "D": "-"})
        assert _root_states_simple(result, 0) == {"A"}

    def test_gap_as_state_real_state(self):
        """When gap_as_missing=False, '-' is a fourth state, recorded in the
        union of the root's state set."""
        tree = PhyloTree.from_newick("(A,B,C,D);")
        result = FitchAlgorithm().compute(
            tree,
            {"A": "A", "B": "A", "C": "A", "D": "-"},
            gap_as_missing=False,
        )
        assert _root_states_simple(result, 0) == {"A", "-"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))