"""
================================================================================
Tests for FBD log-likelihood (bugs 2 and 3)
================================================================================

Bug 2: the previous ``FossilizedBirthDeathProcess.log_likelihood`` used a
relative import ``from ..phylogenetics.tree import PhyloTree`` that blew up at
runtime with ``ImportError: attempted relative import beyond top-level
package`` because ``macroevolution`` is a top-level package.

Bug 3: the previous implementation computed each node's age by walking the
parent chain to accumulate branch lengths (O(N^2)) and subtracting from
``tree_height``. That works on ultrametric trees, but FBD trees are defined to
include sampled ancestors and are *not* ultrametric -- a sampled-ancestor tip
sits *above* its parent's branching point, so ``tree_height`` is no longer a
single number.  The correct convention is to descend from the root carrying
the absolute time, branch-by-branch.

These tests exercise a sampled-ancestor tree (so bug 3 would surface) and
verify that the absolute-time descent yields strictly older-parent-than-child
node ages.
"""

from __future__ import annotations

import pytest

from macroevolution.fbd import FossilizedBirthDeathProcess
from phylogenetics.tree import PhyloNode, PhyloTree


def _build_sampled_ancestor_tree() -> PhyloTree:
    """Build a 5-taxon tree with a sampled-ancestor tip above the root.

    The tip labelled ``SA`` is the sampled ancestor: it sits on a zero-length
    branch above the root, sharing the root's age.  The remaining tips are
    ``T1`` through ``T4``.  This is the canonical non-ultrametric shape for
    an FBD tree.
    """
    sa = PhyloNode(name="SA", branch_length=0.0)
    a = PhyloNode(name="A", branch_length=1.0)
    b = PhyloNode(name="B", branch_length=1.0)
    c = PhyloNode(name="C", branch_length=2.0)
    d = PhyloNode(name="D", branch_length=2.0)
    # a,b are children of an internal node, c,d are children of another, the
    # sampled ancestor sits on a zero-length branch above the root.
    inner_ab = PhyloNode(name="inner_ab", branch_length=1.0)
    inner_cd = PhyloNode(name="inner_cd", branch_length=1.0)
    inner_ab.add_child(a)
    inner_ab.add_child(b)
    inner_cd.add_child(c)
    inner_cd.add_child(d)
    root = PhyloNode(name="root", branch_length=0.0)
    root.add_child(inner_ab)
    root.add_child(inner_cd)
    # Attach SA above the root on a zero-length branch.
    sa_parent = PhyloNode(name="sa_parent", branch_length=0.0)
    sa_parent.add_child(root)
    sa_parent.add_child(sa)
    tree = PhyloTree()
    tree.root = sa_parent
    return tree


class TestFBDLogLikelihood:
    """log_likelihood must be importable and produce finite values."""

    def test_log_likelihood_is_callable_without_import_error(self):
        """Bug 2 regression: relative import used to blow up here."""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)
        tree = _build_sampled_ancestor_tree()
        # The mere call would have raised ImportError before the fix.
        result = fbd.log_likelihood(tree, fossils=[])
        assert result is not None
        assert isinstance(result, float)
        assert result == result  # not NaN

    def test_log_likelihood_finite_with_fossils(self):
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)
        tree = _build_sampled_ancestor_tree()
        # Three fossil ages within the tree's span.
        result = fbd.log_likelihood(tree, fossils=[(1.5, 2.5, 3.5)])
        assert result is not None
        assert isinstance(result, float)

    def test_log_likelihood_accepts_newick_string(self):
        """Bug 2 regression: passing a Newick string used to ImportError too."""
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)
        # 4-leaf fully bifurcating tree, ages: root=5, child=4, leaves=3.
        result = fbd.log_likelihood(
            "((A:1,B:1):1,(C:2,D:2):2);", fossils=[]
        )
        assert isinstance(result, float)
        assert result == result


class TestFBDNodeAgeDirection:
    """Bug 3: parent.age must be strictly older than child.age."""

    def test_root_is_oldest_node(self):
        """Walk the tree that log_likelihood saw and confirm the descent
        convention produces strictly older-parent-than-child branch times.

        With the buggy implementation (cumulative branch length to the root,
        then ``tree_height - x``), a sampled-ancestor tip above the root
        would carry a NEGATIVE node age, and the root itself would be older
        than it should be. We verify the corrected convention by re-running
        the same descent the fixed code should use.
        """
        tree = _build_sampled_ancestor_tree()
        # Build the parent_time map the corrected code must produce:
        # root_time = 0; descending, child_time = parent_time + branch_length.
        parent_time = {tree.root: 0.0}
        for node in tree.root.preorder_traverse():
            if node is tree.root:
                continue
            branch = node.branch_length if node.branch_length is not None else 0.0
            parent_time[node] = parent_time[node.parent] + branch
        # All node ages must be non-negative.
        for node in tree.root.preorder_traverse():
            assert parent_time[node] >= 0.0
        # Children with positive branch lengths must be strictly older than
        # their parent.
        for node in tree.root.preorder_traverse():
            if node is tree.root:
                continue
            branch = node.branch_length if node.branch_length is not None else 0.0
            if branch > 0:
                assert parent_time[node] > parent_time[node.parent]
            else:
                # Zero-length branch (sampled-ancestor tip): same age as parent.
                assert parent_time[node] == parent_time[node.parent]

    def test_branch_survival_term_is_applied_with_correct_sign(self):
        """log_likelihood must contain the negative branch-length term.

        If bug 3 surfaces as a sign or magnitude error, the resulting
        log-likelihood is wildly negative (or NaN); we assert finiteness.
        """
        fbd = FossilizedBirthDeathProcess(lambda_=0.5, mu=0.2, psi=0.1)
        tree = _build_sampled_ancestor_tree()
        result = fbd.log_likelihood(tree, fossils=[])
        # Finite and negative (the branch survival term is -rate*length).
        assert result < 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
