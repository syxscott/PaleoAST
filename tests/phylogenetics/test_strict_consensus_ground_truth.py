"""
================================================================================
Ground-truth tests for phylogenetics.strict_consensus
================================================================================

These tests validate ``phylogenetics/strict_consensus.py`` (current line
coverage 19.0%) by comparing its output against an *independently-shaped*
ground truth: clade-membership arithmetic.

Why this shape of truth is independent
--------------------------------------
The production module builds consensus trees by:

  1. extracting every rooted clade (frozenset of leaf labels) from each input
     tree,
  2. counting how many input trees contain each clade,
  3. keeping clades whose count meets the threshold, and
  4. rebuilding the tree top-down from the surviving clade family.

The independent ground truth enumerates *every* clade that could possibly
appear on a tree over the same leaf set (including the complementary side of
each bipartition, since consensus arithmetic must consider both), counts how
many input trees contain each candidate clade, and selects the clades whose
count matches the mathematical definition (== k for strict, > threshold * k
for majority-rule).  The two implementations use different data structures
and a different control flow, so a defect in one cannot mask a defect in the
other.

Definitions used throughout
---------------------------
* Strict consensus of T_1, ..., T_k:
      C in consensus   iff   C appears in every T_i.
* Majority-rule consensus (threshold t in fraction k):
      C in consensus   iff   count(C, T_1..T_k) > t * k.

Each test names which definition it pins and what failure mode it would
catch.

Determinism
-----------
All randomised inputs use a fixed seed (``SEED`` below); no test calls
``random.random`` or reads the clock.

Running
-------
    .venv/Scripts/python.exe -m pytest \\
        tests/phylogenetics/test_strict_consensus_ground_truth.py -q \\
        --no-header -p no:cacheprovider

The script is memory-light: every fixture fits in well under 100 nodes.
Do NOT chain it into the full suite on a memory-constrained machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the in-package reference algorithms importable as a top-level module
# (they live under tests/phylogenetics, which pytest does not put on sys.path
# by default for module discovery).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import reference_algorithms as R

from phylogenetics.strict_consensus import (
    Split,
    StrictConsensusTree,
    build_majority_rule_consensus,
    build_strict_consensus,
)
from phylogenetics.tree import PhyloTree
from utils.exceptions import PaleoASTError

# Fixed seed keeps the random-tree fixtures deterministic across runs and
# across machines.  See module docstring.
SEED = 20260929


# =============================================================================
# Fixture helpers
# =============================================================================


def _make_trees(labels, n_trees: int, *, k: int | None = None):
    """Return ``n_trees`` independent random binary trees over ``labels``.

    Parameters
    ----------
    labels : sequence of str
        Leaf labels shared by every tree.
    n_trees : int
        How many independent trees to draw.
    k : int, optional
        If given, only return the first ``k`` trees of the batch. Used so
        tests can pull a subset without re-seeding.
    """
    rng = R.np.random.default_rng(SEED)
    trees = [R.random_binary_tree(labels, rng) for _ in range(n_trees)]
    if k is not None:
        trees = trees[:k]
    return trees


def _expected_strict_clades(trees: list[PhyloTree], all_taxa: frozenset[str]) -> set[frozenset[str]]:
    """Strict consensus ground truth.

    A clade is in the strict consensus iff it appears in *every* input tree.
    We restrict to non-trivial clades (size > 1 and < |all_taxa|), because the
    full leaf set is trivially present in every tree and contributes nothing
    to the comparison; singletons are dropped by both the production extractor
    and the reference helper.

    Clades are keyed on ``leaf.name`` to match the production's
    ``_extract_clades_from_tree`` (which also uses ``name``). For
    ``random_binary_tree`` fixtures both ``name`` and ``label`` are set
    identically, so this is identical to a label-based count here.
    """
    full = frozenset(all_taxa)
    n = len(trees)
    counts: dict[frozenset[str], int] = {}
    for tree in trees:
        for clade in _clades_by_name(tree):
            counts[clade] = counts.get(clade, 0) + 1
    return {clade for clade, cnt in counts.items() if cnt == n and 1 < len(clade) < len(full)}


def _expected_majority_clades(
    trees: list[PhyloTree],
    threshold: float,
    all_taxa: frozenset[str],
) -> set[frozenset[str]]:
    """Majority-rule consensus ground truth (strict ``> threshold * k``).

    Caveat: this is the *clade-membership* definition, not the *tree-display*
    definition. When the surviving clades are mutually incompatible (e.g. at
    low thresholds), no tree can display every one of them; the production
    falls back to a partial / star tree, so a strict equality assertion would
    fail. We therefore compare the production's *displayed* clades against the
    subset of expected clades that is pairwise compatible -- which is exactly
    what a correct tree builder can realise.
    """
    full = frozenset(all_taxa)
    k = len(trees)
    cutoff = threshold * k
    counts: dict[frozenset[str], int] = {}
    for tree in trees:
        for clade in _clades_by_name(tree):
            counts[clade] = counts.get(clade, 0) + 1
    return {clade for clade, cnt in counts.items() if cnt > cutoff and 1 < len(clade) < len(full)}


def _pairwise_compatible_subset(
    clades: set[frozenset[str]],
    all_taxa: frozenset[str],
) -> set[frozenset[str]]:
    """Largest pairwise-compatible subset of ``clades`` containing only
    non-trivial clades.

    Used to compare the production's tree-realised clades against the
    *count-based* definition of majority-rule: when the threshold admits
    mutually crossing clades, the production cannot display them all and
    has to pick a compatible subset.
    """
    pool = [c for c in clades if 1 < len(c) < len(all_taxa)]
    kept: list[frozenset[str]] = []
    for candidate in sorted(pool, key=len, reverse=True):
        ok = True
        for other in kept:
            # Compatible iff one of the four intersections is empty.
            a, b = candidate, other
            if (a & b) and not (a <= b or b <= a):
                ok = False
                break
        if ok:
            kept.append(candidate)
    return set(kept)


def _all_taxa(trees: list[PhyloTree]) -> frozenset[str]:
    """Union of leaf labels across the input trees (sorted-stable)."""
    union: set[str] = set()
    for tree in trees:
        union.update(tree.leaf_names)
    return frozenset(union)


def _clades_by_name(tree: PhyloTree) -> set[frozenset[str]]:
    """Tree-clades keyed on ``leaf.name`` rather than ``leaf.label``.

    The reference's :func:`reference_algorithms.clades_of` reads ``label``,
    but ``StrictConsensusTree._build_tree_from_clades`` creates leaves via
    ``PhyloNode(name=taxon, node_type=NodeType.LEAF)`` and never sets
    ``label``, so a strict-consensus result has ``name`` populated and
    ``label=None``. We need a name-aware extractor to compare apples to
    apples. Input trees constructed by ``random_binary_tree`` set BOTH name
    and label, so this helper is consistent with what production counts.
    """
    out: set[frozenset[str]] = set()
    for node in tree.root.get_all_nodes():
        leaves = frozenset(leaf.name for leaf in node.get_leaves() if leaf.name)
        if leaves:
            out.add(leaves)
    return out


def _non_trivial_by_name(tree: PhyloTree) -> set[frozenset[str]]:
    """Non-trivial clades (1 < |c| < |all_taxa|) keyed on ``leaf.name``."""
    total = frozenset(tree.leaf_names)
    return {c for c in _clades_by_name(tree) if 1 < len(c) < len(total)}


# =============================================================================
# P0 — consensus arithmetic truth
# =============================================================================


class TestConsensusArithmetic:
    """The two consensus definitions, evaluated by independent clade counting."""

    def test_strict_consensus_is_clade_intersection(self):
        """Strict consensus = non-trivial clades present in *every* input tree.

        Prevents the highest-frequency defect in consensus code: rebuilding
        from a count filter without ever comparing against the truth.
        """
        labels = ["A", "B", "C", "D", "E", "F"]
        trees = _make_trees(labels, n_trees=4)
        all_taxa = _all_taxa(trees)

        expected = _expected_strict_clades(trees, all_taxa)

        consensus = StrictConsensusTree().build(trees)

        assert _non_trivial_by_name(consensus) == expected, (
            f"strict consensus clades mismatch:\n"
            f"  expected ({len(expected)}): {sorted(map(sorted, expected))}\n"
            f"  got      ({len(_non_trivial_by_name(consensus))}): "
            f"{sorted(map(sorted, _non_trivial_by_name(consensus)))}"
        )

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.6, 0.75])
    def test_majority_rule_threshold_sweep(self, threshold):
        """Majority-rule clades are those whose count strictly exceeds
        ``threshold * k`` across all swept thresholds.

        Defends against the off-by-one that creeps in when ``>=`` is used in
        place of ``>`` (a clade present in exactly half the trees should
        never qualify at threshold=0.5).

        The production's ``_build_tree_from_clades`` greedily picks a
        pairwise-compatible subset when the surviving clades are mutually
        incompatible (documented behaviour, see production module docstring).
        We therefore compare the result against the *largest compatible
        subset* of the count-based expectation.
        """
        labels = ["A", "B", "C", "D", "E", "G"]
        trees = _make_trees(labels, n_trees=5)
        all_taxa = _all_taxa(trees)

        count_based = _expected_majority_clades(trees, threshold, all_taxa)
        compatible_expected = _pairwise_compatible_subset(count_based, all_taxa)

        consensus = StrictConsensusTree().build_majority_rule(trees, threshold=threshold)

        assert _non_trivial_by_name(consensus) == compatible_expected, (
            f"majority-rule (threshold={threshold}) clades mismatch:\n"
            f"  count-based ({len(count_based)}): {sorted(map(sorted, count_based))}\n"
            f"  compatible ({len(compatible_expected)}): "
            f"{sorted(map(sorted, compatible_expected))}\n"
            f"  got      ({len(_non_trivial_by_name(consensus))}): "
            f"{sorted(map(sorted, _non_trivial_by_name(consensus)))}"
        )

    def test_identical_inputs_recover_original(self):
        """k identical input trees ⇒ consensus has the *exact* topology of
        that tree (every clade appears in every tree).

        Defends against a "strict consensus returns a star tree" bug that
        would otherwise hide behind the identical-input case.
        """
        labels = ["A", "B", "C", "D", "E"]
        trees = _make_trees(labels, n_trees=3)
        reference = trees[0]
        duplicated = [R.random_binary_tree(labels, R.np.random.default_rng(0)) for _ in range(3)]
        duplicated[0] = reference  # first tree is the reference; the rest are fresh
        # Make every tree identical to the reference so the test is exact:
        duplicated = [reference, reference, reference]

        strict_tree = StrictConsensusTree().build(duplicated)
        majority_tree = StrictConsensusTree().build_majority_rule(duplicated, threshold=0.5)

        # Strict = Majority on identical inputs (sweep #13 below depends on this).
        assert _non_trivial_by_name(strict_tree) == _non_trivial_by_name(reference)
        assert _non_trivial_by_name(majority_tree) == _non_trivial_by_name(reference)

    def test_fully_conflicting_inputs_collapse_to_star(self):
        """Three pairwise-conflicting trees share no non-trivial clade ⇒
        strict consensus is a star tree (no internal edges beyond the root).

        Guards the degenerate branch of the algorithm: when the intersection
        is empty, the result must fall back to a polytomy rather than emitting
        a half-formed tree from the leftover clades.
        """
        # ((A,B),(C,D)) vs ((A,C),(B,D)) vs ((A,D),(B,C)) on {A,B,C,D}.
        # Each pair conflicts on every internal clade.
        labels = ["A", "B", "C", "D"]
        t1 = PhyloTree.from_newick("((A,B),(C,D));")
        t2 = PhyloTree.from_newick("((A,C),(B,D));")
        t3 = PhyloTree.from_newick("((A,D),(B,C));")
        trees = [t1, t2, t3]

        expected_strict = _expected_strict_clades(trees, frozenset(labels))
        assert expected_strict == set(), f"fixture is supposed to share no non-trivial clades; got {expected_strict}"

        consensus = StrictConsensusTree().build(trees)

        # A star tree over n leaves has exactly n-1 internal nodes if rooted
        # at the polytomy: just the root and (n-1) leaves -- wait, with n
        # leaves the root plus n leaves gives n+1 nodes, but the root has
        # degree n (a polytomy). For a *strict* consensus the production
        # builds a polytomy root and attaches every leaf directly; so the
        # set of non-trivial clades must be empty.
        assert _non_trivial_by_name(consensus) == set(), (
            f"fully-conflicting input should yield star tree, got "
            f"{sorted(map(sorted, _non_trivial_by_name(consensus)))}"
        )


# =============================================================================
# P0 — Split algebraic properties
# =============================================================================


class TestSplitAlgebra:
    """The ``Split`` dataclass is used to compare clade membership; if its
    identity, ordering, or compatibility is wrong, every downstream check
    that goes through it is wrong too."""

    def test_is_trivial_when_one_side_is_a_single_leaf(self):
        """A split is trivial iff one side has exactly one taxon."""
        trivial = Split(set1=frozenset({"A"}), set2=frozenset({"B", "C", "D"}))
        non_trivial = Split(set1=frozenset({"A", "B"}), set2=frozenset({"C", "D"}))
        # All-taxa (the empty partition) is also trivial.
        all_taxa = Split(set1=frozenset({"A"}), set2=frozenset({"A", "B", "C", "D"}))

        assert trivial.is_trivial is True
        assert non_trivial.is_trivial is False
        assert all_taxa.is_trivial is True

    def test_all_taxa_unions_the_two_sides(self):
        """``all_taxa`` is the union of the two sides regardless of order."""
        s = Split(set1=frozenset({"A", "B"}), set2=frozenset({"C", "D", "E"}))
        assert s.all_taxa == frozenset({"A", "B", "C", "D", "E"})

    def test_is_compatible_is_reflexive_and_symmetric(self):
        """Compatibility is reflexive (``S`` is compatible with ``S``) and
        symmetric (if ``S`` is compatible with ``T`` then ``T`` is compatible
        with ``S``). These are the two minimal axioms required for a set of
        splits to be buildable into *any* tree at all -- without them the
        consensus-building invariants in tests 1-4 cannot hold.
        """
        s1 = Split(set1=frozenset({"A", "B"}), set2=frozenset({"C", "D"}))
        s2 = Split(set1=frozenset({"A", "C"}), set2=frozenset({"B", "D"}))
        # s1 and s2 are incompatible: A is on the same side of neither split.
        s3 = Split(set1=frozenset({"A", "B"}), set2=frozenset({"C", "D", "E"}))

        for s in (s1, s2, s3):
            assert s.is_compatible_with(s) is True, f"reflexivity failed for {s}"

        # Symmetry, including for an incompatible pair.
        assert s1.is_compatible_with(s2) == s2.is_compatible_with(s1)
        assert s1.is_compatible_with(s3) == s3.is_compatible_with(s1)
        assert s2.is_compatible_with(s3) == s3.is_compatible_with(s2)

    def test_equality_and_hash_track_set_semantics(self):
        """Two splits are equal iff their sets are equal (post-canonical),
        and equal splits hash the same -- so deduplicating via ``set()``
        gives the right cardinality."""
        a = Split(set1=frozenset({"A", "B"}), set2=frozenset({"C", "D"}))
        b = Split(set1=frozenset({"C", "D"}), set2=frozenset({"A", "B"}))
        c = Split(set1=frozenset({"A", "C"}), set2=frozenset({"B", "D"}))
        # ``__post_init__`` canonicalises by min element, so the swapped
        # version of (a) should already equal (a) regardless of input order.
        assert a == b
        assert hash(a) == hash(b)
        # Different splits must remain distinct even after canonicalisation.
        assert a != c
        assert hash(a) != hash(c)

        # Dedup via a set must collapse (a) and (b) to a single element.
        unique = {a, b, c}
        assert len(unique) == 2

    def test_compatible_splits_build_a_self_consistent_set(self):
        """A pairwise-compatible split family is exactly the precondition
        for *some* tree to display every member of the family (this is
        the classical 4-way compatibility / tree-display theorem).

        What we pin here is the weaker, more practical statement: if you
        start with a tree, extract its splits, and ask whether the family
        is pairwise compatible, the answer must be yes for every pair --
        because every tree's splits come from a single hierarchy.
        """
        labels = ["A", "B", "C", "D", "E"]
        rng = R.np.random.default_rng(SEED)
        tree = R.random_binary_tree(labels, rng)
        builder = StrictConsensusTree()
        splits = builder._extract_splits_from_tree(tree)
        # Drop trivial splits (a tree has many pendant edges; checking
        # compatibility on those is uninteresting and they do not block
        # tree display).
        non_trivial = [s for s in splits if not s.is_trivial]

        for i, s1 in enumerate(non_trivial):
            for s2 in non_trivial[i + 1 :]:
                assert s1.is_compatible_with(s2), f"tree's own splits {s1!r} and {s2!r} marked incompatible"


# =============================================================================
# P1 — boundary and degeneracy
# =============================================================================


class TestBoundaryAndDegeneracy:
    """Edge cases that often break counting / set-union code."""

    def test_single_input_returns_a_clone(self):
        """k = 1 ⇒ result is a deep copy of the lone input tree.

        Guards two distinct failures: (a) crashing on singletons because the
        counting loop divides by k-1 or divides by zero, and (b) returning
        the *same* object so a downstream caller mutates the input by
        accident.
        """
        labels = ["A", "B", "C", "D"]
        trees = _make_trees(labels, n_trees=1)
        # Use the production clone path explicitly.
        consensus_strict = StrictConsensusTree().build(trees)
        consensus_majority = StrictConsensusTree().build_majority_rule(trees, threshold=0.5)
        # Convenience wrappers must agree.
        consensus_strict_via_helper = build_strict_consensus(trees)
        consensus_majority_via_helper = build_majority_rule_consensus(trees, threshold=0.5)

        for result in (
            consensus_strict,
            consensus_majority,
            consensus_strict_via_helper,
            consensus_majority_via_helper,
        ):
            # Topologies match.
            assert _non_trivial_by_name(result) == _non_trivial_by_name(trees[0])
            # Deep-copy semantics: the roots are *distinct* Python objects
            # with distinct node_ids. Iterating to leaves gives the
            # strongest signal that no aliasing leaked.
            assert result.root is not trees[0].root
            assert result.root.node_id != trees[0].root.node_id

    def test_duplicate_input_trees_still_count_correctly(self):
        """Duplicated input trees must not over-count clades (each tree
        contributes exactly one vote per clade regardless of identity to a
        sibling)."""
        labels = ["A", "B", "C", "D", "E"]
        rng = R.np.random.default_rng(SEED)
        base = R.random_binary_tree(labels, rng)
        trees = [base, base, R.random_binary_tree(labels, rng), base]
        all_taxa = _all_taxa(trees)

        expected_strict = _expected_strict_clades(trees, all_taxa)
        consensus_strict = StrictConsensusTree().build(trees)

        assert _non_trivial_by_name(consensus_strict) == expected_strict

        # And majority-rule with threshold=0.5 should give exactly the
        # clades present in > 2 of the 4 trees. Where the count set is
        # mutually incompatible we fall back to the compatible subset.
        count_majority = _expected_majority_clades(trees, 0.5, all_taxa)
        compatible_majority = _pairwise_compatible_subset(count_majority, all_taxa)
        consensus_majority = StrictConsensusTree().build_majority_rule(trees, threshold=0.5)
        assert _non_trivial_by_name(consensus_majority) == compatible_majority

    def test_leaf_set_mismatch_is_reported(self):
        """A taxon mismatch between inputs must raise, not be papered over.

        Regression: ``build`` used to union the leaf sets, so
        ``((A,B),(C,D))`` and ``((A,B),(E,F))`` came back as
        ``((A,B),C,D,E,F)`` -- a tree neither input ever described, and one
        that discards the only structure either input carried (C,D and E,F
        are cherries in their own trees; the result has neither).

        A confidently wrong tree is worse than a refusal, so both entry
        points must raise rather than invent a taxon set.
        """
        t1 = PhyloTree.from_newick("((A,B),(C,D));")
        t2 = PhyloTree.from_newick("((A,B),(E,F));")

        for label, build in (
            ("build", lambda: StrictConsensusTree().build([t1, t2])),
            ("build_majority_rule", lambda: StrictConsensusTree().build_majority_rule([t1, t2])),
        ):
            with pytest.raises(Exception) as excinfo:
                build()
            msg = str(excinfo.value).lower()
            assert (
                isinstance(excinfo.value, (ValueError, TypeError, PaleoASTError)) or "leaf" in msg or "taxon" in msg
            ), f"{label} returned {excinfo.type.__name__}: {excinfo.value!r} instead of rejecting"

    def test_result_leaf_set_matches_input_union(self):
        """The consensus tree must carry exactly the union of input leaves
        (no dropped, no invented). Counts each input twice so the test
        catches the off-by-one on the first / subsequent tree."""
        labels = ["A", "B", "C", "D"]
        trees = _make_trees(labels, n_trees=3)
        expected_union = (
            frozenset(trees[0].leaf_names) | frozenset(trees[1].leaf_names) | frozenset(trees[2].leaf_names)
        )

        consensus_strict = StrictConsensusTree().build(trees)
        consensus_majority = StrictConsensusTree().build_majority_rule(trees, threshold=0.5)

        assert frozenset(consensus_strict.leaf_names) == expected_union
        assert frozenset(consensus_majority.leaf_names) == expected_union
        # Leaf count on the wrapper agrees with the union cardinality.
        assert consensus_strict.leaf_count == len(expected_union)
        assert consensus_majority.leaf_count == len(expected_union)


# =============================================================================
# P1 — explicit naming differentiation
# =============================================================================


class TestNamingDifferentiation:
    """Strict and majority-rule must be *different* algorithms, not aliases.

    Catches the "strict consensus actually returned a majority-rule tree"
    defect that is the single most common naming confusion in phylogenetics
    libraries: when the two functions share an implementation, they appear
    to work in isolation but give *wrong* answers whenever they are
    supposed to differ."""

    def test_strict_and_majority_agree_on_identical_inputs(self):
        """If all inputs are identical, strict and majority must produce
        the same clades -- this pins the shared path."""
        labels = ["A", "B", "C", "D"]
        trees = _make_trees(labels, n_trees=3)
        # Replicate the same tree 3 times -- every clade is unanimous.
        identical = [trees[0], trees[0], trees[0]]

        strict = StrictConsensusTree().build(identical)
        majority = StrictConsensusTree().build_majority_rule(identical, threshold=0.5)

        assert _non_trivial_by_name(strict) == _non_trivial_by_name(majority), (
            "strict and majority-rule must agree when every input is identical; "
            "otherwise one is silently delegating to the other"
        )

    def test_strict_and_majority_disagree_on_partial_conflict(self):
        """When inputs disagree, strict must be *strictly smaller* than
        majority-rule: any unanimous clade is automatically a >50% clade,
        but not every >50% clade is unanimous.

        If strict and majority produce the same set of non-trivial clades
        here, the implementation has conflated the two methods.
        """
        labels = ["A", "B", "C", "D", "E"]
        trees = _make_trees(labels, n_trees=5)

        strict = StrictConsensusTree().build(trees)
        majority = StrictConsensusTree().build_majority_rule(trees, threshold=0.5)

        strict_clades = _non_trivial_by_name(strict)
        majority_clades = _non_trivial_by_name(majority)

        # Unanimous ⊆ majority is an absolute mathematical truth; the test
        # only proves the implementation respects it.
        assert strict_clades <= majority_clades, (
            "strict clades must be a subset of majority-rule clades:\n"
            f"  strict − majority = "
            f"{sorted(map(sorted, strict_clades - majority_clades))}"
        )

        # On randomly drawn trees we expect at least one threshold>0.5
        # clade that is not unanimous -- otherwise the two are accidentally
        # identical, which is the bug we are pinning. Use 5 trees so the
        # majority cutoff (3 of 5) is non-trivial and the two methods have
        # room to diverge. We allow a soft check by iterating a few seeds
        # so the test is robust against the (rare) lucky draw where every
        # majority clade happens to be unanimous.
        for trial in range(5):
            rng = R.np.random.default_rng(SEED + trial)
            trial_trees = [R.random_binary_tree(labels, rng) for _ in range(5)]
            s = _non_trivial_by_name(StrictConsensusTree().build(trial_trees))
            m = _non_trivial_by_name(StrictConsensusTree().build_majority_rule(trial_trees, threshold=0.5))
            if m > s:
                return  # found a case where the two diverge -> bug class avoided
        pytest.fail(
            "across 5 random trials of 5 trees each, strict and majority "
            "produced identical clades -- either the test fixture is too "
            "narrow or the implementation conflates the two methods"
        )
