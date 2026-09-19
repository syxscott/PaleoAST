"""
================================================================================
Tests for bipartition bitmasks, Robinson-Foulds distances and the MCCT
================================================================================

覆盖 phylogenetics/tree_distance.py:
- taxon / clade bitmask 编码 (lsb0, sorted label order)
- normalize / trivial / compatibility 的位运算事实
- split_bitmasks 与无权 Robinson-Foulds 距离的黄金值
- split_lengths 与加权 Robinson-Foulds 距离的黄金值 (手工推导)
- BEAST 风格 MCCT (最大 split 支持乘积树) 的胜负与平票决胜

黄金值均为手工推导，例如 4 分类单元 (A=1, B=2, C=4, D=8, all=15):
    ((A,B),(C,D)) 的唯一非平凡 split = {C,D} = 0b1100 = 12
    ((A,C),(B,D)) 的唯一非平凡 split = {B,D} = 0b1010 = 10
"""

from __future__ import annotations

import pytest

from phylogenetics.tree import PhyloNode, PhyloTree
from phylogenetics.tree_distance import (
    clade_bitmask,
    is_compatible_bitmask_pair,
    is_trivial_bitmask,
    maximum_product_of_split_support_tree,
    mcct,
    normalize_bitmask,
    robinson_foulds_distance,
    split_bitmasks,
    split_lengths,
    split_support_frequencies,
    taxon_bitmask_map,
    weighted_robinson_foulds_distance,
)
from utils.exceptions import ValidationError

# 4-taxon shorthand used throughout the hand-computed expectations.
ALL4 = 0b1111  # taxa A=1, B=2, C=4, D=8 in sorted-label order


def tree(newick: str) -> PhyloTree:
    """Parse a Newick string into a PhyloTree."""
    return PhyloTree.from_newick(newick)


class TestTaxonBitmaskMap:
    """lsb0 taxon indexing over the sorted leaf-label order."""

    def test_bits_follow_sorted_labels(self):
        result = taxon_bitmask_map(tree("((D,C),(B,A));"))
        assert result == {"A": 0b0001, "B": 0b0010, "C": 0b0100, "D": 0b1000}

    def test_index_is_independent_of_tree_shape(self):
        first = taxon_bitmask_map(tree("((A,B),(C,D));"))
        second = taxon_bitmask_map(tree("(((C,D),B),A);"))
        assert first == second

    def test_five_taxa_bit_positions(self):
        result = taxon_bitmask_map(tree("((A,B),(C,(D,E)));"))
        assert result == {"A": 1, "B": 2, "C": 4, "D": 8, "E": 16}

    def test_single_leaf(self):
        assert taxon_bitmask_map(tree("A:0.5;")) == {"A": 1}

    def test_rootless_tree_gives_empty_index(self):
        assert taxon_bitmask_map(PhyloTree()) == {}

    def test_duplicate_leaf_label_raises(self):
        with pytest.raises(ValidationError, match="unique"):
            taxon_bitmask_map(tree("((A,B),(A,D));"))

    def test_unnamed_leaf_raises(self):
        # The Newick parser rejects empty labels, so build the node by hand.
        root = PhyloNode(name="root")
        root.add_child(PhyloNode(name="A"))
        root.add_child(PhyloNode(name=""))
        with pytest.raises(ValidationError, match="no label"):
            taxon_bitmask_map(PhyloTree(root=root))


class TestCladeBitmask:
    """Leafset bitmasks of individual nodes."""

    def test_leaf_and_internal_nodes(self):
        reference = tree("((A,B),(C,D));")
        index = taxon_bitmask_map(reference)
        left, right = reference.root.children

        assert clade_bitmask(node=reference.root.children[0], index=index) == 0b0011
        assert clade_bitmask(node=right, index=index) == 0b1100
        assert clade_bitmask(node=reference.root, index=index) == ALL4
        assert clade_bitmask(node=left.children[1], index=index) == 0b0010

    def test_polytomy_clade(self):
        reference = tree("((A,B,C),D);")
        index = taxon_bitmask_map(reference)
        assert clade_bitmask(node=reference.root.children[0], index=index) == 0b0111

    def test_leaf_outside_index_raises(self):
        index = taxon_bitmask_map(tree("((A,B),(C,D));"))
        with pytest.raises(ValidationError, match="Zebra"):
            clade_bitmask(node=PhyloNode(name="Zebra"), index=index)


class TestNormalizeBitmask:
    """DendroPy lsb0 normalization: keep the side without the first taxon."""

    @pytest.mark.parametrize(
        ("bitmask", "expected"),
        [
            (0b0011, 0b1100),  # {A,B} -> {C,D}
            (0b1100, 0b1100),  # already canonical
            (0b0101, 0b1010),  # {A,C} -> {B,D}
            (0b1001, 0b0110),  # {A,D} -> {B,C}
            (0b0000, 0b0000),
            (ALL4, 0b0000),  # the full leafset normalizes away
        ],
    )
    def test_four_taxa(self, bitmask, expected):
        assert normalize_bitmask(bitmask, ALL4) == expected

    def test_five_taxa(self):
        assert normalize_bitmask(0b00111, 0b11111) == 0b11000

    def test_bits_outside_fill_mask_are_dropped(self):
        assert normalize_bitmask(0b1_0011, ALL4) == 0b1100

    def test_empty_fill_mask(self):
        assert normalize_bitmask(0b1010, 0) == 0


class TestIsTrivialBitmask:
    """A split is trivial when one side holds at most one taxon."""

    @pytest.mark.parametrize(
        ("bitmask", "expected"),
        [
            (0b0000, True),  # empty side
            (ALL4, True),  # empty complement
            (0b0001, True),  # single taxon
            (0b1110, True),  # single taxon on the other side
            (0b0111, True),  # complement is {D}
            (0b0011, False),  # 2 | 2
            (0b0101, False),  # 2 | 2
        ],
    )
    def test_four_taxa(self, bitmask, expected):
        assert is_trivial_bitmask(bitmask, ALL4) is expected

    def test_normalization_does_not_change_triviality(self):
        for bitmask in range(1 << 4):
            assert is_trivial_bitmask(bitmask, ALL4) is is_trivial_bitmask(normalize_bitmask(bitmask, ALL4), ALL4)


class TestSplitBitmasks:
    """Normalized, non-trivial split sets."""

    def test_paired_split_of_four_taxa(self):
        assert split_bitmasks(tree("((A,B),(C,D));")) == frozenset({0b1100})

    def test_caterpillar_shares_the_same_split(self):
        assert split_bitmasks(tree("(((A,B),C),D);")) == frozenset({0b1100})

    def test_rotation_invariance(self):
        assert split_bitmasks(tree("((A,B),(C,D));")) == split_bitmasks(tree("((D,C),(B,A));"))

    def test_star_tree_has_no_splits(self):
        assert split_bitmasks(tree("(A,B,C,D);")) == frozenset()

    def test_three_taxon_tree_has_only_trivial_splits(self):
        assert split_bitmasks(tree("((A,B),C);")) == frozenset()

    def test_six_taxon_three_cherries(self):
        assert split_bitmasks(tree("((A,B),(C,D),(E,F));")) == frozenset({0b111100, 0b001100, 0b110000})

    def test_rootless_tree(self):
        assert split_bitmasks(PhyloTree()) == frozenset()


class TestSplitCompatibility:
    """Four-way intersection test on bitmask pairs."""

    def test_nested_splits_are_compatible(self):
        # {A,B}|{C,D} against {A,B,C}|{D}
        assert is_compatible_bitmask_pair(0b0011, 0b0111, ALL4) is True

    def test_disjoint_splits_are_compatible(self):
        assert is_compatible_bitmask_pair(0b0011, 0b1100, ALL4) is True

    def test_crossing_splits_are_incompatible(self):
        # {A,B}|{C,D} against {A,C}|{B,D}: all four intersections non-empty
        assert is_compatible_bitmask_pair(0b0011, 0b0101, ALL4) is False

    def test_incompatibility_survives_normalization(self):
        first = normalize_bitmask(0b0011, ALL4)
        second = normalize_bitmask(0b0101, ALL4)
        assert is_compatible_bitmask_pair(first, second, ALL4) is False

    def test_every_pair_of_four_taxon_splits_matches_set_logic(self):
        """All 4-taxon split pairs: compatible iff one side of each nests."""

        def members(mask: int) -> frozenset[int]:
            return frozenset(i for i in range(4) if mask & (1 << i))

        splits = [m for m in range(1 << 4) if not is_trivial_bitmask(m, ALL4)]
        for first in splits:
            for second in splits:
                sides1 = (members(first), members(ALL4 ^ first))
                sides2 = (members(second), members(ALL4 ^ second))
                expected = any(not (s1 & s2) for s1 in sides1 for s2 in sides2)
                assert is_compatible_bitmask_pair(first, second, ALL4) is expected

    def test_unnormalized_masks_need_the_fourth_check(self):
        # {A,B,C}|{D,E} vs {C,D,E}|{A,B}: only the (complement ∩ complement)
        # quadrant is empty, which DendroPy's three-check shortcut misses.
        assert is_compatible_bitmask_pair(0b00111, 0b11100, 0b11111) is True


class TestRobinsonFouldsDistance:
    """Unweighted RF distance."""

    def test_golden_value_two(self):
        first = tree("((A,B),(C,D));")
        second = tree("((A,C),(B,D));")
        assert robinson_foulds_distance(first, second) == 2

    def test_identical_topology_is_zero(self):
        first = tree("((A,B),(C,D));")
        assert robinson_foulds_distance(first, tree("((A,B),(C,D));")) == 0

    def test_rotation_is_zero(self):
        assert robinson_foulds_distance(tree("((A,B),(C,D));"), tree("((D,C),(B,A));")) == 0

    def test_same_object(self):
        reference = tree("(((A,B),C),(D,E));")
        assert robinson_foulds_distance(reference, reference) == 0

    def test_branch_lengths_are_ignored(self):
        first = tree("((A:1,B:2):3,(C:4,D:5):6);")
        second = tree("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);")
        assert robinson_foulds_distance(first, second) == 0

    def test_six_taxon_golden_value(self):
        first = tree("((A,B),(C,D),(E,F));")
        second = tree("((A,B),(C,E),(D,F));")
        assert robinson_foulds_distance(first, second) == 4

    def test_three_taxon_trees_are_always_equidistant(self):
        assert robinson_foulds_distance(tree("((A,B),C);"), tree("(A,B,C);")) == 0

    def test_taxon_mismatch_raises(self):
        with pytest.raises(ValidationError, match="identical taxon set"):
            robinson_foulds_distance(tree("((A,B),(C,D));"), tree("((A,B),(C,E));"))

    def test_duplicate_labels_raise(self):
        with pytest.raises(ValidationError, match="unique"):
            robinson_foulds_distance(tree("((A,B),(A,D));"), tree("((A,B),(C,D));"))

    def test_distance_is_symmetric(self):
        first = tree("(((A,B),(C,D)),E);")
        second = tree("(((A,C),(B,D)),E);")
        assert robinson_foulds_distance(first, second) == robinson_foulds_distance(second, first)


class TestSplitLengths:
    """Split-length profiles keyed by normalized bitmask."""

    def test_lengths_sum_over_shared_keys(self):
        reference = tree("((A:0.1,B:0.2)l:1.0,(C:0.3,D:0.4)m:0.5)root:9.0;")
        assert split_lengths(reference) == {
            0b1110: 0.1,  # {B,C,D} -> terminal edge of A
            0b0010: 0.2,  # {B}
            0b0100: 0.3,  # {C}
            0b1000: 0.4,  # {D}
            0b1100: 1.5,  # the root edge: both children carry the same split
        }

    def test_root_branch_length_excluded(self):
        reference = tree("((A:0.1,B:0.2)l:1.0,(C:0.3,D:0.4)m:0.5)root:9.0;")
        assert sum(split_lengths(reference).values()) == pytest.approx(2.5)

    def test_none_branch_length_counts_as_zero(self):
        lengths = split_lengths(tree("((A,B),(C,D));"))
        assert set(lengths) == {0b1110, 0b0010, 0b0100, 0b1000, 0b1100}
        assert all(value == 0.0 for value in lengths.values())

    def test_single_leaf_tree(self):
        assert split_lengths(tree("A:0.5;")) == {}


class TestWeightedRobinsonFouldsDistance:
    """Split-length weighted RF distance (Goyal et al. 2012)."""

    def test_identical_trees_are_zero(self):
        first = tree("((A:0.1,B:0.2):1.0,(C:0.3,D:0.4):0.5);")
        assert weighted_robinson_foulds_distance(first, first) == pytest.approx(0.0)

    def test_rotation_invariance(self):
        first = tree("(A:0.1,(B:0.2,C:0.3)i:0.4);")
        second = tree("((C:0.3,B:0.2)i:0.4,A:0.1);")
        assert weighted_robinson_foulds_distance(first, second) == pytest.approx(0.0)

    def test_internal_and_terminal_difference(self):
        # Hand-computed: internal split key 0b1100 carries 1.0 + 0.5 vs
        # 2.0 + 0.5 (difference 1.0) and terminal A differs by 0.1.
        first = tree("((A:0.1,B:0.2)i:1.0,(C:0.3,D:0.4)j:0.5);")
        second = tree("((A:0.2,B:0.2)i:2.0,(C:0.3,D:0.4)j:0.5);")
        assert weighted_robinson_foulds_distance(first, second) == pytest.approx(1.1)

    def test_topological_difference_with_unit_lengths(self):
        first = tree("((A:1,B:1):1,(C:1,D:1):1);")
        second = tree("((A:1,C:1):1,(B:1,D:1):1);")
        assert weighted_robinson_foulds_distance(first, second) == pytest.approx(4.0)

    def test_min_dist_substitutes_for_missing_splits(self):
        first = tree("((A:1,B:1):1,(C:1,D:1):1);")
        second = tree("((A:1,C:1):1,(B:1,D:1):1);")
        assert weighted_robinson_foulds_distance(first, second, min_dist=0.5) == pytest.approx(3.0)

    def test_trivial_splits_capture_terminal_lengths(self):
        # Unweighted RF is blind to terminal edges; the weighted version is not.
        first = tree("(A:0.1,B:0.2);")
        second = tree("(A:0.1,B:0.5);")
        assert robinson_foulds_distance(first, second) == 0
        assert weighted_robinson_foulds_distance(first, second) == pytest.approx(0.3)

    def test_missing_branch_length_counts_as_zero(self):
        first = tree("((A,B)i:0.5,(C,D)j:0.5);")
        second = tree("((A:0.5,B:0.5)i:0.5,(C:0.5,D:0.5)j:0.5);")
        assert weighted_robinson_foulds_distance(first, second) == pytest.approx(2.0)

    def test_taxon_mismatch_raises(self):
        with pytest.raises(ValidationError, match="identical taxon set"):
            weighted_robinson_foulds_distance(tree("(A:1,B:1);"), tree("(A:1,C:1);"))


class TestSplitSupportFrequencies:
    """Frequencies of normalized splits over a collection."""

    def test_golden_frequencies(self):
        trees = [
            tree("((A,E),(B,(C,D)));"),
            tree("(((A,B),C),(D,E));"),
            tree("((A,C),(B,(D,E)));"),
            tree("((A,B),C,(D,E));"),
        ]
        frequencies = split_support_frequencies(trees)
        assert len(frequencies) == 5
        assert frequencies[0b11100] == pytest.approx(2 / 4)
        assert frequencies[0b11000] == pytest.approx(3 / 4)
        assert frequencies[0b11010] == pytest.approx(1 / 4)
        assert frequencies[0b01110] == pytest.approx(1 / 4)
        assert frequencies[0b01100] == pytest.approx(1 / 4)

    def test_repeated_tree_counts_repeatedly(self):
        paired = tree("((A,B),(C,D));")
        crossed = tree("((A,C),(B,D));")
        frequencies = split_support_frequencies([paired, paired, crossed])
        assert frequencies[0b1100] == pytest.approx(2 / 3)
        assert frequencies[0b1010] == pytest.approx(1 / 3)

    def test_single_tree_has_full_support(self):
        reference = tree("(((A,B),C),(D,E));")
        assert set(split_support_frequencies([reference]).values()) == {1.0}

    def test_empty_collection_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            split_support_frequencies([])

    def test_taxon_mismatch_raises(self):
        with pytest.raises(ValidationError, match="identical taxon set"):
            split_support_frequencies([tree("((A,B),C);"), tree("((A,B),D);")])


class TestMaximumProductOfSplitSupportTree:
    """BEAST-style MCCT: the maximum product of split supports."""

    def test_popular_splits_beat_a_unique_split(self):
        # Splits: {A,E} tree contributes 0b01110 + 0b01100 (freq 1/4 each),
        # trees[1] contributes 0b11100 (2/4) and 0b11000 (3/4) and therefore
        # wins; trees[3] ties with trees[1] and loses the tie-break.
        trees = [
            tree("((A,E),(B,(C,D)));"),
            tree("(((A,B),C),(D,E));"),
            tree("((A,C),(B,(D,E)));"),
            tree("((A,B),C,(D,E));"),
        ]
        result = maximum_product_of_split_support_tree(trees)
        assert result is trees[1]
        assert robinson_foulds_distance(result, tree("(((A,B),C),(D,E));")) == 0

    def test_tie_breaks_to_the_first_tree(self):
        first = tree("((A,B),(C,D));")
        second = tree("(((A,B),C),D);")  # same non-trivial split 0b1100
        assert maximum_product_of_split_support_tree([first, second]) is first
        assert maximum_product_of_split_support_tree([second, first]) is second

    def test_duplicated_tree_beats_rarer_topology(self):
        # The duplicated topology carries split frequency 2/3 against 1/3, so it
        # wins whichever position it occupies in the collection.
        paired = tree("((A,B),(C,D));")
        crossed = tree("((A,C),(B,D));")
        assert maximum_product_of_split_support_tree([crossed, paired, paired]) is paired
        assert maximum_product_of_split_support_tree([paired, paired, crossed]) is paired

    def test_single_tree_is_returned(self):
        reference = tree("(((A,B),C),(D,E));")
        assert maximum_product_of_split_support_tree([reference]) is reference

    def test_star_and_single_leaf_collections_score_zero(self):
        star = tree("(A,B,C,D);")
        assert maximum_product_of_split_support_tree([star, star]) is star
        single = tree("A:0.5;")
        assert maximum_product_of_split_support_tree([single, single]) is single

    def test_alias_matches_implementation(self):
        trees = [tree("((A,B),(C,D));"), tree("((A,C),(B,D));")]
        assert mcct(trees) is maximum_product_of_split_support_tree(list(trees))

    def test_empty_collection_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            maximum_product_of_split_support_tree([])

    def test_taxon_mismatch_raises(self):
        with pytest.raises(ValidationError, match="identical taxon set"):
            maximum_product_of_split_support_tree([tree("((A,B),C);"), tree("((A,B),E);")])
