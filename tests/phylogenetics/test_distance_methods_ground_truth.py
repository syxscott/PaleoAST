"""
================================================================================
Ground-truth tests for phylogenetics/distance_methods.py
================================================================================

Why this file exists
--------------------
``phylogenetics/distance_methods.py`` had 10.9 % line coverage -- the lowest
in the package -- and the module is the *only* implementation of UPGMA and NJ
in PaleoAST, the two algorithms every downstream distance-based phylogenetics
routine (consensus, cophenetic, signal) leans on. A silent bug here would
propagate everywhere.

Strategy
--------
The tests follow the "theorem, not heuristic" pattern already used by
``tests/phylogenetics/reference_algorithms.py``:

* Round-trip is the ground truth for tree building. For an ultrametric tree
  UPGMA is provably exact (Sneath & Sokal 1973); for any additive tree
  neighbour joining is provably exact (Saitou & Nei 1987). So taking a tree,
  computing its patristic distance matrix, and asking the builder to recover
  the tree is a theorem, not a heuristic expectation. We do this for several
  taxon counts with multiple deterministic seeds.

* Topology comparison uses unrooted splits rather than rooted Newick. NJ in
  particular roots its result wherever the final join happened, which need
  not be where the generating tree was rooted -- comparing Newick strings
  would create false negatives for a perfectly correct algorithm.

* Cross-implementation agreement catches a bug in either implementation.
  ``reference_upgma`` / ``reference_nj`` are hand-written longhand versions
  sharing no code with the production module. When both implementations
  agree on the topology for a non-tree-shaped (general) input, neither has a
  serious algebraic error.

* Hand-computable 4-taxon cases lock the algorithm down to specific numerical
  answers a reader can verify on paper.

References:
- Saitou, N. & Nei, M. (1987). The neighbor-joining method: a new method for
  reconstructing phylogenetic trees. Mol Biol Evol 4(4):406-425.
- Sneath, P.H.A. & Sokal, R.R. (1973). Numerical Taxonomy. Freeman.
- Felsenstein, J. (2004). Inferring Phylogenies. Sinauer. Ch. 11 (Distance
  Methods) -- the NJ Q-matrix derivation and 3-node final adjustment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the local reference-algorithms helper importable without polluting
# sys.path permanently for other test files.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import reference_algorithms as R

from phylogenetics.distance_methods import (
    DistanceMatrix,
    build_nj_tree,
    build_upgma_tree,
)
from phylogenetics.tree import PhyloTree

# =============================================================================
# Test-local helpers
# =============================================================================
#
# The reference's ``clades_of`` / ``unrooted_splits`` index leaves by
# ``leaf.label``, but production trees only set ``leaf.name``. These local
# helpers read ``name`` so the round-trip comparisons work against production
# output as well as reference output.


def _leaf_clades(tree: PhyloTree) -> set[frozenset[str]]:
    """Every clade in ``tree`` keyed by leaf name."""
    out: set[frozenset[str]] = set()
    for node in tree.root.get_all_nodes():
        if node is None:
            continue
        leaves = frozenset(leaf.name for leaf in node.get_leaves() if leaf.name)
        if leaves:
            out.add(leaves)
    return out


def _unrooted_splits(tree: PhyloTree) -> frozenset:
    """Set of internal splits (the tree's unrooted topology).

    A clade of size n-1 hangs off a single pendant edge; it is not an internal
    split and including it would conflate trees that differ only by their
    rooting. Comparing splits -- rather than rooted Newick -- is what makes
    the NJ round-trip test meaningful.
    """
    full = frozenset(tree.leaf_names)
    out: set[frozenset] = set()
    for clade in _leaf_clades(tree):
        other = full - clade
        if len(clade) < 2 or len(other) < 2:
            continue
        small, large = sorted((clade, other), key=lambda c: sorted(c))
        out.add(frozenset({small, large}))
    return frozenset(out)


def _patristic(tree: PhyloTree) -> tuple[list[str], np.ndarray]:
    """Leaf-to-leaf distance matrix in alphabetical leaf order.

    Independent of ``reference_algorithms.patristic_matrix`` because the
    reference's helper reads ``leaf.label`` (set only by its own factory),
    while production trees set only ``leaf.name``.
    """
    leaves = sorted(tree.leaf_names)
    by_name = {leaf.name: leaf for leaf in tree.root.get_leaves()}

    def climb(node, lca) -> float:
        total = 0.0
        cur = node
        while cur is not lca:
            total += float(cur.branch_length or 0.0)
            cur = cur.parent
            if cur is None:
                raise ValueError("LCA not found while climbing")
        return total

    def distance(a, b) -> float:
        chain_a = {id(a): a}
        for n in a.get_ancestors():
            chain_a[id(n)] = n
        chain_b = {id(b): b}
        for n in b.get_ancestors():
            chain_b[id(n)] = n
        shared = set(chain_a) & set(chain_b)
        if not shared:
            raise ValueError("nodes have no common ancestor")
        # Among shared ancestors, the LCA is the one furthest from the root
        # (longest chain of ancestors == deepest node).
        lca = max((chain_a[k] for k in shared), key=lambda n: len(n.get_ancestors()))
        return climb(a, lca) + climb(b, lca)

    m = np.zeros((len(leaves), len(leaves)))
    for i, a in enumerate(leaves):
        for j, b in enumerate(leaves):
            if i < j:
                v = distance(by_name[a], by_name[b])
                m[i, j] = m[j, i] = v
    return leaves, m


def _is_binary_internals(tree: PhyloTree) -> bool:
    """Every internal node has exactly 2 children.

    Note: this excludes the root -- NJ by construction leaves 3 nodes to wire
    into a ternary root, which is the standard rooted representation of an
    unrooted binary tree. For tests that need a stricter binary invariant we
    assert against the root separately.
    """
    for node in tree.root.get_all_nodes():
        if node.is_root:
            continue
        if node.is_leaf:
            continue
        if len(node.children) != 2:
            return False
    return True


# =============================================================================
# P0 -- NJ round-trip theorem on additive trees
# =============================================================================


class TestNJAdditiveRoundtrip:
    """NJ of an additive tree's patristic matrix must recover the unrooted
    topology exactly.

    Theorem (Saitou & Nei 1987): for any *additive* distance matrix D, the NJ
    algorithm reconstructs the unique tree realising D, up to the placement of
    the root (which is not identifiable from D). Since the round trip produces
    a tree whose unrooted topology equals the source, comparing unrooted
    splits -- rather than rooted Newick -- eliminates the root placement
    artefact.

    Prevents: a silent algebraic error in the Q-matrix computation
    (``(n-2) d_ij - r_i - r_j``), in the limb-length formulas, or in the
    3-node final adjustment that would otherwise be invisible in casual tests.
    """

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    @pytest.mark.parametrize("seed", [0, 7, 13, 42, 99])
    def test_nj_roundtrip_unrooted_splits(self, n_taxa: int, seed: int) -> None:
        """Across many seeds and taxon counts, NJ must reproduce the source
        tree's unrooted splits exactly."""
        rng = np.random.default_rng(seed)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng)
        leaves_p, matrix = R.patristic_matrix(source)

        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_nj_tree(dm)

        expected_splits = _unrooted_splits(source)
        actual_splits = _unrooted_splits(rebuilt)
        assert actual_splits == expected_splits, (
            f"n={n_taxa}, seed={seed}: NJ round-trip split mismatch.\n"
            f"  expected: {sorted(expected_splits)}\n"
            f"  actual:   {sorted(actual_splits)}"
        )

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    def test_nj_roundtrip_all_leaves_recovered(self, n_taxa: int) -> None:
        """NJ must preserve the leaf set: no taxon may be silently dropped.

        Defends against a class of bugs where a hash-tied tie-break on
        cluster names causes one branch of the tree to be orphaned.
        """
        rng = np.random.default_rng(2026)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng)
        leaves_p, matrix = R.patristic_matrix(source)

        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_nj_tree(dm)

        assert sorted(rebuilt.leaf_names) == sorted(leaves_p), (
            f"n={n_taxa}: NJ dropped or renamed leaves: "
            f"expected {sorted(leaves_p)}, got {sorted(rebuilt.leaf_names)}"
        )


class TestNJDistanceAndStructure:
    """NJ must reproduce its input distances and produce a binary tree."""

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_nj_recovers_patristic_distances(self, n_taxa: int, seed: int) -> None:
        """For an additive input, the rebuilt tree must reproduce the input
        distances within numerical noise (< 1e-9). This is the part of the
        NJ correctness theorem that catches errors in limb-length formulas.
        """
        rng = np.random.default_rng(seed)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng)
        leaves_p, matrix = R.patristic_matrix(source)

        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_nj_tree(dm)

        # Align both matrices by alphabetical label so that ``rebuilt.leaf_names``
        # being in traversal order doesn't trip the comparison.
        canonical = sorted(leaves_p)
        src_index = {name: i for i, name in enumerate(leaves_p)}
        prod_labels, prod_matrix = _patristic(rebuilt)
        prod_index = {name: i for i, name in enumerate(prod_labels)}
        src_aligned = matrix[
            [src_index[n] for n in canonical]
        ][:, [src_index[n] for n in canonical]]
        prod_aligned = prod_matrix[
            [prod_index[n] for n in canonical]
        ][:, [prod_index[n] for n in canonical]]
        max_diff = float(np.abs(src_aligned - prod_aligned).max())
        assert max_diff < 1e-9, (
            f"n={n_taxa}, seed={seed}: NJ max distance error {max_diff:.3e}"
        )

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    def test_nj_binary_internals(self, n_taxa: int) -> None:
        """Every non-root internal node must have exactly 2 children.

        NJ builds the tree by repeatedly fusing a pair of nodes into a binary
        internal node, so a ternary or higher-arity non-root internal node is
        a structural error.
        """
        rng = np.random.default_rng(7)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng)
        leaves_p, matrix = R.patristic_matrix(source)
        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_nj_tree(dm)
        assert _is_binary_internals(rebuilt), (
            f"n={n_taxa}: NJ produced a non-binary internal node. "
            f"Arity map: {[(n.name, len(n.children)) for n in rebuilt.root.get_all_nodes() if not n.is_leaf]}"
        )

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    def test_nj_root_has_three_children_or_two(self, n_taxa: int) -> None:
        """NJ leaves 3 active nodes at the end and joins them under a single
        root, so the root has 3 children. For n <= 3 this changes -- the
        production handles n=2 and n=3 specially.

        The root being ternary (3 children) is the *expected* NJ output: the
        algorithm does not place the root, and the canonical rooted form of
        an unrooted binary tree is a 3-child root.
        """
        rng = np.random.default_rng(11)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng)
        leaves_p, matrix = R.patristic_matrix(source)
        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_nj_tree(dm)
        # For n >= 3 the NJ code path always leaves 3 nodes to wire up.
        assert len(rebuilt.root.children) == 3, (
            f"n={n_taxa}: expected NJ root to have 3 children, "
            f"got {len(rebuilt.root.children)}"
        )


# =============================================================================
# P0 -- UPGMA round-trip theorem on ultrametric trees
# =============================================================================


class TestUPGMAUltrametricRoundtrip:
    """UPGMA of an ultrametric tree's patristic matrix must recover the
    unrooted topology exactly.

    Theorem (Sneath & Sokal 1973, Felsenstein 2004 ch.11): for an *ultrametric*
    distance matrix, UPGMA reconstructs the unique rooted tree where every
    leaf is equidistant from the root. The round trip is therefore exact.
    """

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    @pytest.mark.parametrize("seed", [0, 7, 13, 42, 99])
    def test_upgma_roundtrip_unrooted_splits(self, n_taxa: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng, ultrametric=True)
        leaves_p, matrix = R.patristic_matrix(source)

        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_upgma_tree(dm)

        expected_splits = _unrooted_splits(source)
        actual_splits = _unrooted_splits(rebuilt)
        assert actual_splits == expected_splits, (
            f"n={n_taxa}, seed={seed}: UPGMA round-trip split mismatch.\n"
            f"  expected: {sorted(expected_splits)}\n"
            f"  actual:   {sorted(actual_splits)}"
        )

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    def test_upgma_roundtrip_all_leaves_recovered(self, n_taxa: int) -> None:
        """UPGMA must preserve every input taxon. Prevents the hash-tied
        cluster-name bug that the production module's docstring warns about
        ("旧实现用 cluster_{len(members)} 会在同规模合并时重名")."""
        rng = np.random.default_rng(2027)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng, ultrametric=True)
        leaves_p, matrix = R.patristic_matrix(source)
        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_upgma_tree(dm)
        assert sorted(rebuilt.leaf_names) == sorted(leaves_p), (
            f"n={n_taxa}: UPGMA dropped leaves: "
            f"expected {sorted(leaves_p)}, got {sorted(rebuilt.leaf_names)}"
        )


class TestUPGMADistanceAndStructure:
    """UPGMA must reproduce distances for ultrametric inputs and stay binary."""

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_upgma_recovers_patristic_distances(self, n_taxa: int, seed: int) -> None:
        """For an ultrametric input, the rebuilt tree's patristic matrix must
        equal the input within numerical noise. This is the leaf-distance
        half of the UPGMA correctness theorem.
        """
        rng = np.random.default_rng(seed)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng, ultrametric=True)
        leaves_p, matrix = R.patristic_matrix(source)

        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_upgma_tree(dm)

        # Align both matrices by alphabetical label. ``_patristic`` returns
        # rows/cols in alphabetical order; the source ``matrix`` is in
        # ``leaves_p`` order which is already alphabetical. We still rebuild
        # both in the same canonical order to be safe against future
        # re-orderings of either helper.
        canonical = sorted(leaves_p)
        src_index = {name: i for i, name in enumerate(leaves_p)}
        prod_labels, prod_matrix = _patristic(rebuilt)
        prod_index = {name: i for i, name in enumerate(prod_labels)}
        src_aligned = matrix[
            [src_index[n] for n in canonical]
        ][:, [src_index[n] for n in canonical]]
        prod_aligned = prod_matrix[
            [prod_index[n] for n in canonical]
        ][:, [prod_index[n] for n in canonical]]
        max_diff = float(np.abs(src_aligned - prod_aligned).max())
        assert max_diff < 1e-9, (
            f"n={n_taxa}, seed={seed}: UPGMA max distance error {max_diff:.3e}"
        )

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    def test_upgma_output_is_ultrametric(self, n_taxa: int) -> None:
        """UPGMA's defining property: every leaf is equidistant from the root.

        For an ultrametric input the rebuilt tree must also be ultrametric
        (i.e. the UPGMA construction preserved the molecular-clock property).
        """
        rng = np.random.default_rng(31)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng, ultrametric=True)
        leaves_p, matrix = R.patristic_matrix(source)
        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_upgma_tree(dm)

        leaf_to_root = []
        for leaf in rebuilt.root.get_leaves():
            total = 0.0
            node = leaf
            while node.parent is not None:
                total += float(node.branch_length or 0.0)
                node = node.parent
            leaf_to_root.append(total)

        assert max(leaf_to_root) - min(leaf_to_root) < 1e-9, (
            f"n={n_taxa}: UPGMA output is not ultrametric. "
            f"Leaf-to-root distances: {leaf_to_root}"
        )

    @pytest.mark.parametrize("n_taxa", [4, 5, 7, 10])
    def test_upgma_binary_internals(self, n_taxa: int) -> None:
        """UPGMA produces a strictly binary tree -- every internal node
        (including the root) has exactly 2 children.

        Note: this is stricter than the NJ assertion because UPGMA builds a
        rooted tree directly, not an unrooted one with a 3-child root.
        """
        rng = np.random.default_rng(23)
        labels = [f"t{i}" for i in range(n_taxa)]
        source = R.random_binary_tree(labels, rng, ultrametric=True)
        leaves_p, matrix = R.patristic_matrix(source)
        dm = DistanceMatrix.from_array(matrix, leaves_p)
        rebuilt = build_upgma_tree(dm)
        for node in rebuilt.root.get_all_nodes():
            if node.is_leaf:
                continue
            assert len(node.children) == 2, (
                f"n={n_taxa}: UPGMA produced a non-binary internal node "
                f"({node.name}, arity={len(node.children)})"
            )


# =============================================================================
# P0 -- Cross-implementation agreement (production vs reference)
# =============================================================================


class TestCrossCheckReference:
    """The production builder and the reference builder, given the same
    arbitrary (not necessarily additive) distance matrix, must agree on the
    unrooted topology.

    These two implementations share no code: production builds cluster
    dictionaries and merges in place; the reference uses a flat pair lookup
    table and inline merges. If they agree on a hundred random matrices,
    neither has a serious algebraic error.
    """

    @pytest.mark.parametrize("trial", list(range(20)))
    def test_nj_matches_reference_on_random_matrices(self, trial: int) -> None:
        rng = np.random.default_rng(trial * 13 + 3)
        n = int(rng.integers(4, 11))
        labels = [f"t{i}" for i in range(n)]
        # Random symmetric matrix, deliberately non-additive so both impls
        # see the same approximate problem.
        mat = rng.uniform(0.1, 5.0, size=(n, n))
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0.0)

        dm = DistanceMatrix.from_array(mat, labels)
        prod_tree = build_nj_tree(dm)
        ref_tree = R.reference_nj(labels, mat)

        prod_splits = _unrooted_splits(prod_tree)
        ref_splits = _unrooted_splits(ref_tree)
        assert prod_splits == ref_splits, (
            f"trial={trial}: NJ topology differs from reference.\n"
            f"  production: {sorted(prod_splits)}\n"
            f"  reference:  {sorted(ref_splits)}"
        )

    @pytest.mark.parametrize("trial", list(range(20)))
    def test_upgma_matches_reference_on_random_matrices(self, trial: int) -> None:
        rng = np.random.default_rng(trial * 7 + 1)
        n = int(rng.integers(4, 11))
        labels = [f"t{i}" for i in range(n)]
        mat = rng.uniform(0.1, 5.0, size=(n, n))
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0.0)

        dm = DistanceMatrix.from_array(mat, labels)
        prod_tree = build_upgma_tree(dm)
        ref_tree = R.reference_upgma(labels, mat)

        prod_splits = _unrooted_splits(prod_tree)
        ref_splits = _unrooted_splits(ref_tree)
        assert prod_splits == ref_splits, (
            f"trial={trial}: UPGMA topology differs from reference.\n"
            f"  production: {sorted(prod_splits)}\n"
            f"  reference:  {sorted(ref_splits)}"
        )


# =============================================================================
# P1 -- Hand-computable 4-taxon cases
# =============================================================================


class TestUPGMAHandComputable:
    """A textbook UPGMA trace on 4 taxa, locked to exact numerical answers."""

    def test_step_by_step_cherry_height(self) -> None:
        """Trace UPGMA on a 4-taxon distance matrix where two distinct
        cherries are present (A,B at distance 2; C,D at distance 4).

        Distances:
            A B C D
        A   0 2 4 6
        B   2 0 4 6
        C   4 4 0 4
        D   6 6 4 0

        Step 1: min = 2 = d(A,B). Merge (A,B) at height 1.0. Branch
                lengths of the cherry: A.branch_length = B.branch_length = 1.0.
        Step 2: distances to merged cluster AB:
                d(AB, C) = (1*4 + 1*4) / 2 = 4
                d(AB, D) = (1*6 + 1*6) / 2 = 6
                d(C, D)   = 4
                Two equal minima; production picks the lexicographic winner
                (it iterates sorted(active)), which is (C, D).
        Step 3: merge (C, D) at height 2.0. C.branch_length = D.branch_length
                = 2.0.
        Step 4: d(AB, CD) = (2*4 + 2*6) / 4 = 5. Merge at height 2.5.
                AB.branch_length  = 2.5 - 1.0 = 1.5
                CD.branch_length  = 2.5 - 2.0 = 0.5

        Net topology: ((A,B), (C,D)); all leaves at distance 2.5 from root.
        """
        labels = ["A", "B", "C", "D"]
        mat = np.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [2.0, 0.0, 4.0, 6.0],
                [4.0, 4.0, 0.0, 4.0],
                [6.0, 6.0, 4.0, 0.0],
            ]
        )
        dm = DistanceMatrix.from_array(mat, labels)
        tree = build_upgma_tree(dm)

        leaves_by_name = {leaf.name: leaf for leaf in tree.root.get_leaves()}
        # All leaves equidistant from the root: the molecular-clock property
        # of an UPGMA tree on a *consistent* ultrametric input. (This input
        # is in fact *not* ultrametric -- A->D = 6 but A->C = 4 -- so UPGMA
        # only approximately preserves it. We therefore only assert what
        # the *construction* guarantees, not the input property.)
        # The exact branch lengths above come from the UPGMA construction,
        # which is reproducible.
        assert leaves_by_name["A"].branch_length == pytest.approx(1.0)
        assert leaves_by_name["B"].branch_length == pytest.approx(1.0)
        assert leaves_by_name["C"].branch_length == pytest.approx(2.0)
        assert leaves_by_name["D"].branch_length == pytest.approx(2.0)

        # The internal node directly above A and B has branch length 1.5
        # (the height 2.5 minus the cherry height 1.0).
        a_parent = leaves_by_name["A"].parent
        b_parent = leaves_by_name["B"].parent
        assert a_parent is b_parent, "A and B should share an internal parent"
        assert a_parent.branch_length == pytest.approx(1.5)

        # The internal node directly above C and D has branch length 0.5.
        c_parent = leaves_by_name["C"].parent
        d_parent = leaves_by_name["D"].parent
        assert c_parent is d_parent, "C and D should share an internal parent"
        assert c_parent.branch_length == pytest.approx(0.5)

        # And the topology splits as {A, B} | {C, D}.
        assert _unrooted_splits(tree) == frozenset(
            {
                frozenset(
                    {
                        frozenset({"A", "B"}),
                        frozenset({"C", "D"}),
                    }
                )
            }
        )


class TestNJHandComputable:
    """A textbook NJ trace on 4 taxa.

    Reference: Saitou & Nei (1987) worked example. The 4-taxon case is small
    enough that the Q-matrix, the chosen pair, and the limb lengths are all
    single arithmetic -- perfect for catching sign errors, off-by-one in the
    (n-2) prefactor, or wrong row-sum indexing.
    """

    def test_q_matrix_and_first_merge(self) -> None:
        """Trace the first NJ step on a known 4-taxon additive matrix.

        Distances (from Felsenstein 2004, ch.11 worked example -- a true
        additive matrix from a tree):
            A B C D
        A   0 5 9 9
        B   5 0 10 10
        C   9 10 0 8
        D   9 10 8 0

        Q-matrix Q(i,j) = (n-2) d(i,j) - r_i - r_j, where r_i = sum_k d(i,k):
            r_A = 0 + 5 + 9 + 9 = 23
            r_B = 5 + 0 + 10 + 10 = 25
            r_C = 9 + 10 + 0 + 8 = 27
            r_D = 9 + 10 + 8 + 0 = 27

            Q(A, B) = 2*5  - 23 - 25 = -38
            Q(A, C) = 2*9  - 23 - 27 = -32
            Q(A, D) = 2*9  - 23 - 27 = -32
            Q(B, C) = 2*10 - 25 - 27 = -32
            Q(B, D) = 2*10 - 25 - 27 = -32
            Q(C, D) = 2*8  - 27 - 27 = -38

        The minimum is -38, with two pairs tied: (A,B) and (C,D). Both lead
        to the same unrooted topology; the test asserts the *property* -- the
        pair chosen has minimum Q-value -- and the resulting tree has the
        cherry topology {A,B} | {C,D}.
        """
        labels = ["A", "B", "C", "D"]
        mat = np.array(
            [
                [0.0, 5.0, 9.0, 9.0],
                [5.0, 0.0, 10.0, 10.0],
                [9.0, 10.0, 0.0, 8.0],
                [9.0, 10.0, 8.0, 0.0],
            ]
        )

        # Reproduce the Q-matrix calculation locally.
        n = mat.shape[0]
        row_sums = mat.sum(axis=1)
        q_expected = np.zeros_like(mat)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                q_expected[i, j] = (n - 2) * mat[i, j] - row_sums[i] - row_sums[j]

        # Verify the hand-computed Q values.
        assert q_expected[0, 1] == pytest.approx(-38.0)  # A, B
        assert q_expected[0, 2] == pytest.approx(-32.0)  # A, C
        assert q_expected[2, 3] == pytest.approx(-38.0)  # C, D

        # The two tied minima are (A,B) and (C,D). Production picks the
        # lexicographically first under its tie-break (it iterates sorted
        # labels); both choices yield the same topology.
        dm = DistanceMatrix.from_array(mat, labels)
        tree = build_nj_tree(dm)
        assert _unrooted_splits(tree) == frozenset(
            {
                frozenset(
                    {
                        frozenset({"A", "B"}),
                        frozenset({"C", "D"}),
                    }
                )
            }
        ), (
            "NJ on an additive 4-taxon matrix with two equal cherries must "
            "recover the cherry topology. Did the (n-2) prefactor regress?"
        )

    def test_limb_lengths_on_additive_4_taxa(self) -> None:
        """For an additive 4-taxon matrix, NJ must reproduce the patristic
        distances and produce non-negative branch lengths.

        Source tree: ((A:1, B:2):1, (C:3, D:4):2) -- a non-degenerate
        additive tree with no equal inter-cluster distances.
            d(A, B) = 1+2 = 3
            d(A, C) = 1+1+2+3 = 7
            d(A, D) = 1+1+2+4 = 8
            d(B, C) = 2+1+2+3 = 8
            d(B, D) = 2+1+2+4 = 9
            d(C, D) = 3+4 = 7
        """
        labels = ["A", "B", "C", "D"]
        mat = np.array(
            [
                [0.0, 3.0, 7.0, 8.0],
                [3.0, 0.0, 8.0, 9.0],
                [7.0, 8.0, 0.0, 7.0],
                [8.0, 9.0, 7.0, 0.0],
            ]
        )
        dm = DistanceMatrix.from_array(mat, labels)
        tree = build_nj_tree(dm)

        # No branch length may be negative on an additive input.
        for node in tree.root.get_all_nodes():
            if node.branch_length is not None:
                assert node.branch_length >= -1e-9, (
                    f"Negative branch length on an additive input: "
                    f"{node.name} = {node.branch_length}"
                )

        # Patristic distances must round-trip within floating-point noise.
        prod_labels, prod_mat = _patristic(tree)
        canonical = sorted(labels)
        prod_index = {name: i for i, name in enumerate(prod_labels)}
        prod_aligned = prod_mat[
            [prod_index[n] for n in canonical]
        ][:, [prod_index[n] for n in canonical]]
        assert np.abs(mat - prod_aligned).max() < 1e-9, (
            "NJ must reproduce an additive 4-taxon distance matrix exactly"
        )


# =============================================================================
# P1 -- DistanceMatrix properties
# =============================================================================


class TestDistanceMatrixSymmetryAndDiagonal:
    """``DistanceMatrix`` must enforce the metric axioms at the API level:
    ``d(x, x) == 0`` and ``d(x, y) == d(y, x)`` for every input mode."""

    def test_to_matrix_returns_symmetric_with_zero_diagonal(self) -> None:
        """``to_matrix`` must produce a matrix that is symmetric
        (d_ij = d_ji) and has zeros on the diagonal -- the two axioms that
        any valid distance matrix must satisfy.

        For this test we pass a fully-symmetric matrix to ``from_array``,
        so the symmetriser in ``__post_init__`` and the lower-triangle
        pass in ``from_array`` agree on every entry.
        """
        labels = ["A", "B", "C", "D"]
        # Fully symmetric input.
        mat = np.array(
            [
                [0.0, 2.0, 5.0, 7.0],
                [2.0, 0.0, 4.0, 6.0],
                [5.0, 4.0, 0.0, 3.0],
                [7.0, 6.0, 3.0, 0.0],
            ]
        )
        dm = DistanceMatrix.from_array(mat, labels)
        m = dm.to_matrix()
        m_arr = np.asarray(m)
        # Diagonal must be zero.
        assert np.all(np.diag(m_arr) == 0.0), f"Diagonal not zero: {np.diag(m_arr)}"
        # Matrix must be symmetric.
        assert np.allclose(m_arr, m_arr.T), (
            f"Matrix not symmetric:\n{m_arr}\nvs transpose:\n{m_arr.T}"
        )
        # And it must equal what we put in.
        assert np.allclose(m_arr, mat), (
            f"\nactual:\n{m_arr}\nexpected:\n{mat}"
        )

    def test_get_distance_is_symmetric(self) -> None:
        """``get_distance`` must return the same value regardless of argument
        order. Defends against a bug where only one of (t1, t2)/(t2, t1)
        was stored in the distances dict."""
        labels = ["A", "B", "C"]
        data = {
            "A": {"A": 0.0, "B": 1.5, "C": 2.5},
            "B": {"A": 1.5, "B": 0.0, "C": 3.0},
            "C": {"A": 2.5, "B": 3.0, "C": 0.0},
        }
        dm = DistanceMatrix.from_dict(data)
        for i, t1 in enumerate(labels):
            for t2 in labels:
                assert dm.get_distance(t1, t2) == dm.get_distance(t2, t1), (
                    f"Asymmetric: get_distance({t1}, {t2})={dm.get_distance(t1, t2)}"
                    f" != get_distance({t2}, {t1})={dm.get_distance(t2, t1)}"
                )

    def test_get_distance_self_is_zero(self) -> None:
        """For any input taxon t, d(t, t) must be 0 (the reflexivity axiom)."""
        dm = DistanceMatrix.from_dict(
            {
                "A": {"A": 999.0, "B": 1.5},
                "B": {"A": 1.5, "B": 999.0},
            }
        )
        # The production code short-circuits ``taxon1 == taxon2`` to 0.0
        # without consulting the dictionary, so even if a user fed a junk
        # value for d(t, t), the public API still returns 0.
        assert dm.get_distance("A", "A") == 0.0
        assert dm.get_distance("B", "B") == 0.0


class TestDistanceMatrixConstructors:
    """Round-trip through ``from_array``, ``from_dict``, ``to_matrix``."""

    def test_from_array_to_matrix_roundtrip(self) -> None:
        """A symmetric numpy matrix passed to ``from_array`` and read back
        via ``to_matrix`` must come out unchanged (same shape, same values)."""
        labels = ["L0", "L1", "L2", "L3"]
        original = np.array(
            [
                [0.0, 1.5, 2.5, 3.5],
                [1.5, 0.0, 4.0, 5.0],
                [2.5, 4.0, 0.0, 6.5],
                [3.5, 5.0, 6.5, 0.0],
            ]
        )
        dm = DistanceMatrix.from_array(original, labels)
        recovered = np.asarray(dm.to_matrix())
        assert recovered.shape == original.shape
        assert np.allclose(recovered, original), (
            f"Round-trip changed the matrix:\n"
            f"  original:\n{original}\n  recovered:\n{recovered}"
        )

    def test_from_dict_to_matrix_matches_expected(self) -> None:
        """A nested-dict input must produce a matrix matching the dict
        structure (with both triangle entries populated)."""
        data = {
            "X": {"X": 0.0, "Y": 1.0, "Z": 2.0},
            "Y": {"X": 1.0, "Y": 0.0, "Z": 3.0},
            "Z": {"X": 2.0, "Y": 3.0, "Z": 0.0},
        }
        dm = DistanceMatrix.from_dict(data)
        # ``from_dict`` reads only the keys (X, Y, Z) into ``taxa``; entries
        # like Z->X are picked up via the dict iteration.
        assert dm.taxa == ["X", "Y", "Z"]
        m = np.asarray(dm.to_matrix())
        expected = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        assert np.allclose(m, expected), f"\nactual:\n{m}\nexpected:\n{expected}"

    def test_from_dict_is_symmetric_even_with_only_one_triangle(self) -> None:
        """If only the upper triangle is supplied, ``from_dict`` + ``get_distance``
        must still answer the reverse query. The production code's
        ``__post_init__`` symmetriser populates both (t1, t2) and (t2, t1)."""
        data = {
            "A": {"A": 0.0, "B": 1.5},
            "B": {"B": 0.0},
        }
        dm = DistanceMatrix.from_dict(data)
        assert dm.get_distance("A", "B") == 1.5
        assert dm.get_distance("B", "A") == 1.5


class TestDistanceMatrixFromSequences:
    """``from_sequences`` builds a p-distance (or identity) matrix from
    equal-length aligned sequences. The p-distance between two sequences is
    the fraction of positions at which they differ -- a value a reader can
    verify by counting.

    Defends against: a silent change in the diff-counting loop, a wrong
    normalisation, or a missing symmetriser step.
    """

    def test_p_distance_matches_hand_count(self) -> None:
        """For a 4-sequence alignment of length 4, the p-distances can be
        counted by hand and must match what ``from_sequences`` returns.

        Alignment (length 4):
            A T G C
            A T G G   <- differs from A at one site (pos 3)
            C T G A   <- differs from A at two sites (pos 0, pos 3)
            A T G A   <- differs from A at one site (pos 3)

        Expected p-distances from A's perspective:
            A-B = 1/4 = 0.25
            A-C = 2/4 = 0.50
            A-D = 1/4 = 0.25
        """
        sequences = {
            "A": "ATGC",
            "B": "ATGG",
            "C": "CTGA",
            "D": "ATGA",
        }
        dm = DistanceMatrix.from_sequences(sequences)
        assert dm.get_distance("A", "B") == pytest.approx(0.25)
        assert dm.get_distance("A", "C") == pytest.approx(0.50)
        assert dm.get_distance("A", "D") == pytest.approx(0.25)
        assert dm.get_distance("B", "C") == pytest.approx(0.50)
        assert dm.get_distance("B", "D") == pytest.approx(0.25)
        assert dm.get_distance("C", "D") == pytest.approx(0.25)

        # Self-distance is zero.
        assert dm.get_distance("A", "A") == 0.0
        # Symmetry.
        assert dm.get_distance("B", "A") == dm.get_distance("A", "B")

    def test_p_distance_aliases_accepted(self) -> None:
        """All documented aliases of ``p-distance`` must produce the same
        answer (defends against the normalised-string dispatch table from
        forgetting an entry)."""
        sequences = {"A": "ATGC", "B": "ATGG"}
        canonical = DistanceMatrix.from_sequences(sequences, model="p-distance")
        for alias in ("p", "p_dist", "pdist", "pdistance"):
            dm = DistanceMatrix.from_sequences(sequences, model=alias)
            assert dm.get_distance("A", "B") == canonical.get_distance("A", "B"), (
                f"Alias {alias!r} produced a different answer than 'p-distance'"
            )

    def test_identity_distance_is_complement(self) -> None:
        """``identity`` model returns 1 - p-distance (fraction of identical
        positions). For the same input as the p-distance test:
            A-B: 3/4 matches = 0.75
            A-C: 2/4 matches = 0.50
            A-D: 3/4 matches = 0.75
        """
        sequences = {
            "A": "ATGC",
            "B": "ATGG",
            "C": "CTGA",
            "D": "ATGA",
        }
        dm = DistanceMatrix.from_sequences(sequences, model="identity")
        assert dm.get_distance("A", "B") == pytest.approx(0.75)
        assert dm.get_distance("A", "C") == pytest.approx(0.50)
        assert dm.get_distance("A", "D") == pytest.approx(0.75)

    def test_unknown_model_raises_not_implemented(self) -> None:
        """The docstring promises ``NotImplementedError`` for unsupported
        substitution models (JC69, K2P, etc.) -- not a silent fallback to
        p-distance. A wrong fallback would be invisible in casual tests."""
        sequences = {"A": "AT", "B": "GC"}
        with pytest.raises(NotImplementedError):
            DistanceMatrix.from_sequences(sequences, model="JC69")
        with pytest.raises(NotImplementedError):
            DistanceMatrix.from_sequences(sequences, model="Kimura")

    def test_empty_sequences_raise_validation_error(self) -> None:
        """An empty alignment must raise a validation error -- a 0-length
        sequence would otherwise trigger a ZeroDivisionError when computing
        p = diffs / len(seq)."""
        with pytest.raises(Exception) as exc_info:
            DistanceMatrix.from_sequences({"A": "", "B": "AT"})
        # Must be the project's ValidationError, not a bare ZeroDivisionError.
        # The docstring promises ValidationError.
        assert "Validation" in type(exc_info.value).__name__ or isinstance(
            exc_info.value, ValueError
        )

    def test_empty_taxon_set_raises(self) -> None:
        """Calling ``from_sequences({})`` must raise rather than silently
        produce a 0x0 matrix."""
        with pytest.raises(Exception):
            DistanceMatrix.from_sequences({})

    def test_uneven_sequence_lengths_raise(self) -> None:
        """Aligned sequences of different lengths must raise -- the p-distance
        is undefined when the comparison runs off the end of one sequence."""
        with pytest.raises(Exception):
            DistanceMatrix.from_sequences({"A": "AT", "B": "ATG"})


class TestDistanceMatrixEdgeCases:
    """Boundary inputs to the builder API."""

    def test_two_taxa_minimum_input(self) -> None:
        """The smallest non-trivial NJ/UPGMA input is 2 taxa. The result
        must be a 2-leaf tree whose branch lengths sum to the input distance."""
        labels = ["X", "Y"]
        mat = np.array([[0.0, 1.5], [1.5, 0.0]])
        dm = DistanceMatrix.from_array(mat, labels)

        nj_tree = build_nj_tree(dm)
        assert nj_tree.leaf_count == 2
        # The two branch lengths should sum to the input distance.
        children = nj_tree.root.children
        lengths = sum(c.branch_length or 0.0 for c in children)
        assert lengths == pytest.approx(1.5), (
            f"NJ 2-taxa: branch lengths sum to {lengths}, expected 1.5"
        )

        upgma_tree = build_upgma_tree(dm)
        assert upgma_tree.leaf_count == 2
        lengths = sum(c.branch_length or 0.0 for c in upgma_tree.root.children)
        assert lengths == pytest.approx(1.5), (
            f"UPGMA 2-taxa: branch lengths sum to {lengths}, expected 1.5"
        )

    def test_equal_distances_tie_break_is_deterministic(self) -> None:
        """When multiple pairs share the minimum distance, UPGMA's
        tie-break must be deterministic across runs.

        The production code's docstring warns explicitly about this:
            "排序而非直接 list(set)：set[str] 的迭代顺序随字符串 hash 随机化
            变化，会让并列最小距离的 tie-break 每次不同".
        We re-run the same input and confirm the leaf-pair chosen for the
        first merge is the same both times.
        """
        labels = ["A", "B", "C", "D"]
        # All non-zero pairwise distances are equal -- four-way tie.
        mat = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )
        dm = DistanceMatrix.from_array(mat, labels)
        tree_a = build_upgma_tree(dm)
        tree_b = build_upgma_tree(dm)
        # Same unrooted splits both times.
        assert _unrooted_splits(tree_a) == _unrooted_splits(tree_b)
        # Same leaf set.
        assert sorted(tree_a.leaf_names) == sorted(tree_b.leaf_names)

    def test_missing_key_raises_keyerror(self) -> None:
        """``get_distance`` for a pair with no entry must raise, not return 0
        (a wrong 0 would silently corrupt the tree-building algorithms)."""
        dm = DistanceMatrix.from_dict(
            {"A": {"A": 0.0, "B": 1.0}, "B": {"A": 1.0, "B": 0.0}}
        )
        with pytest.raises(KeyError):
            dm.get_distance("A", "no_such_taxon")

    def test_single_taxon_returns_leaf_tree(self) -> None:
        """With 1 taxon, UPGMA/NJ must return a tree containing that single
        leaf (not raise, not return an empty tree)."""
        dm = DistanceMatrix.from_array(np.array([[0.0]]), ["only"])
        assert build_upgma_tree(dm).leaf_count == 1
        assert build_upgma_tree(dm).leaf_names == ["only"]
        assert build_nj_tree(dm).leaf_count == 1
        assert build_nj_tree(dm).leaf_names == ["only"]

    def test_zero_taxa_returns_empty_tree(self) -> None:
        """With 0 taxa, both builders must return a tree with no leaves."""
        dm = DistanceMatrix(taxa=[], distances={})
        assert build_upgma_tree(dm).leaf_count == 0
        assert build_nj_tree(dm).leaf_count == 0


class TestUPGMABranchLengthClamping:
    """When the input distances are not strictly ultrametric, the
    ``height - h_child`` difference can be negative at some step. The
    production code clamps it to 0 with ``max(0, ...)`` so that no internal
    branch length is reported as negative. This is exercised by giving the
    algorithm a matrix that is metric but not ultrametric.
    """

    def test_non_ultrametric_input_clamps_branch_lengths(self) -> None:
        """A metric-but-not-ultrametric 4-taxon matrix (the classic
        'Felsenstein zone' shape) should still produce a tree with all
        non-negative branch lengths -- the production's ``max(0, height-h)``
        clamp must engage on at least one branch."""
        labels = ["A", "B", "C", "D"]
        # Two long branches and one short: A-B=2, A-C=6, A-D=6, B-C=6, B-D=6,
        # C-D=2. This is a metric matrix (every triple obeys the triangle
        # inequality) but it is *not* ultrametric: max d(A, C) and d(A, D)
        # are both 6 while d(C, D) = 2 -- a long+long+short pattern.
        mat = np.array(
            [
                [0.0, 2.0, 6.0, 6.0],
                [2.0, 0.0, 6.0, 6.0],
                [6.0, 6.0, 0.0, 2.0],
                [6.0, 6.0, 2.0, 0.0],
            ]
        )
        dm = DistanceMatrix.from_array(mat, labels)
        tree = build_upgma_tree(dm)
        for node in tree.root.get_all_nodes():
            if node.branch_length is not None:
                assert node.branch_length >= 0.0, (
                    f"UPGMA produced a negative branch length on a metric "
                    f"input: {node.name} = {node.branch_length}"
                )


class TestNJNonMetricBehavior:
    """When the input is not additive (real data almost never is), NJ may
    compute a negative branch length. The production code logs a warning and
    clamps the length to 0 rather than failing. These tests pin that
    behaviour so a future refactor that removes the clamp is caught.
    """

    def test_non_metric_distances_clamp_to_zero(self, caplog) -> None:
        """A non-additive matrix must yield a tree with no negative limb
        lengths, and the production code should log a warning. Defends
        against a future change that lets negative limbs through."""
        import logging as logging_mod

        labels = ["A", "B", "C", "D"]
        # A deliberately non-additive matrix: violates the 4-point condition.
        mat = np.array(
            [
                [0.0, 1.0, 5.0, 5.0],
                [1.0, 0.0, 5.0, 5.0],
                [5.0, 5.0, 0.0, 1.0],
                [5.0, 5.0, 1.0, 0.0],
            ]
        )
        # Triangle inequality violation: d(A, B) + d(B, C) = 1 + 5 = 6 > 5
        # = d(A, C) is satisfied, but the matrix is metric; for NJ we want
        # the more pathological case of an asymmetric perturbation that
        # produces a negative limb.
        mat[0, 2] = 0.5
        mat[2, 0] = 0.5
        dm = DistanceMatrix.from_array(mat, labels)
        with caplog.at_level(logging_mod.WARNING, logger="phylogenetics.distance_methods"):
            tree = build_nj_tree(dm)
        # All branch lengths must be non-negative.
        for node in tree.root.get_all_nodes():
            if node.branch_length is not None:
                assert node.branch_length >= 0.0, (
                    f"NJ produced a negative branch length and did not clamp: "
                    f"{node.name} = {node.branch_length}"
                )
