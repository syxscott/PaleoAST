# =============================================================================
# FILE: tests/statistics/test_pcm_contrasts.py
# =============================================================================
"""
Regression tests for statistics/pcm.py - independent-contrast bookkeeping.

Covered bugs
------------
1. ``phylogenetic_anova()`` used to map a contrast back onto the tree by
   *name*.  Polytomy contrasts carry synthetic labels (``ABC_c0``, ``ABC_c1``,
   ...) which never resolve to a node, so every one of them fell through to
   ``groups[0]``.  All contrasts then shared a single factor level, giving
   ``SS_between = 0`` and hence the meaningless ``F = 0, p = 1`` for any tree
   with a hard polytomy.  Contrasts now carry the node object itself, and a
   contrast that cannot be classified raises ComputationError instead of being
   silently assigned to the first group.
2. Zero / missing branch lengths made the contrast standardisation divide by
   ``sqrt(v1 + v2) == 0``, escaping as a bare ``ZeroDivisionError``.  The tree
   is now validated up front (ValidationError) and the recursion itself is
   defensive (ComputationError).
"""

import os

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

from phylogenetics.tree import PhyloTree  # noqa: E402
from statistics.pcm import (  # noqa: E402
    PCMAnalyzer,
    _check_positive_branch_lengths,
    _compute_contrasts_recursive,
)
from utils.exceptions import ComputationError, ValidationError  # noqa: E402

# A tree whose two focal clades are *trichotomies* (hard polytomies).
POLYTOMOUS_NEWICK = "((A:1,B:1,C:1)ABC:1,(D:1,E:1,F:1)DEF:1)R;"
TRAITS = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 10.0, "E": 11.0, "F": 12.0}
GROUPS = {"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2", "F": "g2"}


@pytest.fixture
def analyzer() -> PCMAnalyzer:
    return PCMAnalyzer(n_randomizations=99)


class TestContrastNodeIdentity:
    def test_contrast_entries_carry_their_node(self):
        tree = PhyloTree.from_newick(POLYTOMOUS_NEWICK)
        _, _, contrasts, _ = _compute_contrasts_recursive(tree.root, TRAITS)

        assert contrasts, "expected at least one contrast"
        for contrast, se, name, node in contrasts:
            assert node is not None, f"contrast {name} lost its node reference"
            assert not node.is_leaf
            assert se > 0
            assert contrast == pytest.approx(contrast * se / se)

    def test_named_internal_nodes_resolve(self):
        tree = PhyloTree.from_newick(POLYTOMOUS_NEWICK)
        _, _, contrasts, _ = _compute_contrasts_recursive(tree.root, TRAITS)
        names = {node.name for _c, _se, _n, node in contrasts}
        # Both polytomy nodes and the root contribute contrasts.
        assert names == {"ABC", "DEF", "R"}

    def test_unnamed_nodes_get_unique_labels(self):
        tree = PhyloTree.from_newick("(((A:1,B:1,C:1):1,(D:1,E:1):1):1,F:1,G:1);")
        traits = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0}
        _, _, contrasts, _ = _compute_contrasts_recursive(tree.root, traits)
        labels = [name for _c, _se, name, _node in contrasts]
        # Synthetic labels must not collide, otherwise a downstream name lookup
        # could not tell the contrasts apart even in principle.
        assert len(set(labels)) == len(labels)


class TestPhylogeneticAnovaPolytomy:
    def test_polytomy_anova_is_not_degenerate(self, analyzer):
        """F must not collapse to 0 / p to 1 just because the tree has polytomies."""
        tree = PhyloTree.from_newick(POLYTOMOUS_NEWICK)
        result = analyzer.phylogenetic_anova(tree, TRAITS, GROUPS, random_seed=42)

        assert result.f_statistic > 0.0, (
            "Every contrast was assigned to a single group: the polytomy "
            "contrasts were dropped into the default group again"
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_perfect_separation_gives_larger_f_than_noise(self, analyzer):
        separated = dict(TRAITS)
        tree = PhyloTree.from_newick(POLYTOMOUS_NEWICK)
        far = analyzer.phylogenetic_anova(
            tree, separated, GROUPS, n_permutations=99, random_seed=7
        )
        # Same grouping, but the trait no longer follows the groups.
        scrambled = {"A": 1.0, "B": 5.0, "C": 9.0, "D": 2.0, "E": 6.0, "F": 10.0}
        noisy = analyzer.phylogenetic_anova(
            tree, scrambled, GROUPS, n_permutations=99, random_seed=7
        )
        assert far.f_statistic > noisy.f_statistic


class TestBranchLengthValidation:
    def test_zero_branch_length_raises_validation_error(self, analyzer):
        tree = PhyloTree.from_newick("((A:1,B:0,C:1)ABC:1,(D:1,E:1,F:1)DEF:1)R;")
        with pytest.raises(ValidationError):
            analyzer.compute_contrasts(tree, TRAITS)

    def test_zero_branch_length_raises_in_anova(self, analyzer):
        tree = PhyloTree.from_newick("((A:1,B:0,C:1)ABC:1,(D:1,E:1,F:1)DEF:1)R;")
        with pytest.raises(ValidationError):
            analyzer.phylogenetic_anova(tree, TRAITS, GROUPS)

    def test_helper_reports_offending_edge(self):
        tree = PhyloTree.from_newick("((A:1,B:1)AB:1,(C:0,D:1)CD:1)R;")
        with pytest.raises(ValidationError) as excinfo:
            _check_positive_branch_lengths(tree.root, context="PIC")
        assert "CD" in str(excinfo.value)

    def test_negative_branch_length_rejected(self):
        tree = PhyloTree.from_newick("((A:1,B:1)AB:1,(C:1,D:1)CD:-1)R;")
        with pytest.raises(ValidationError):
            _check_positive_branch_lengths(tree.root, context="PIC")

    def test_all_zero_siblings_do_not_raise_zerodivision(self):
        """The recursion itself must never leak a ZeroDivisionError."""
        tree = PhyloTree.from_newick("((A:0,B:0)AB:1,(C:1,D:1)CD:1)R;")
        with pytest.raises((ValidationError, ComputationError)):
            _compute_contrasts_recursive(tree.root, TRAITS)
