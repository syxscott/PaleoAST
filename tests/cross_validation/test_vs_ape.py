# =============================================================================
# FILE: tests/cross_validation/test_vs_ape.py
# =============================================================================
"""
Cross-validation against R's ``ape`` package (Modern Phylogenetics and
Evolutionary Analyses).

ape is the reference implementation for tree building and for Felsenstein's
independence criterion, both of which are easy to get subtly wrong and hard to
notice: a wrong branch length or a missing variance rescaling still produces a
plausible-looking tree or a plausible-looking p-value.

Comparisons are live. Tree shape is compared through the **cophenetic matrix**
-- the matrix of tip-to-tip path lengths -- rather than by Newick string, so
that the many ways of writing the same unrooted tree (rotation, which node is
called "root", node labelling) do not register as differences.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._rbridge import as_array, r, r_matrix, r_string_vector, r_vector, require

pytestmark = pytest.mark.cross_validation

R_APE = require("ape")
R_STATS = require("stats")


def _three_point_holds(matrix, i: int, j: int, k: int, tol: float = 1e-8) -> bool:
    """True if the triple (i, j, k) satisfies the three-point condition.

    A tree is ultrametric exactly when, for every triple, the two largest of the
    three pairwise distances are equal.
    """
    distances = sorted((matrix[i, j], matrix[i, k], matrix[j, k]))
    return abs(distances[1] - distances[2]) <= tol


DISTANCES = np.array(
    [
        [0.0, 5.0, 9.0, 9.0, 8.0, 9.0],
        [5.0, 0.0, 10.0, 10.0, 9.0, 10.0],
        [9.0, 10.0, 0.0, 8.0, 7.0, 4.0],
        [9.0, 10.0, 8.0, 0.0, 6.0, 5.0],
        [8.0, 9.0, 7.0, 6.0, 0.0, 5.0],
        [9.0, 10.0, 4.0, 5.0, 5.0, 0.0],
    ]
)
TAXA = ["A", "B", "C", "D", "E", "F"]


def _cophenetic(tree) -> tuple[np.ndarray, list[str]]:
    """PaleoAST tree -> (square cophenetic matrix, tip order).

    ``PhyloTree.get_distance_matrix()`` returns a symmetric dict keyed by tip
    name pairs, so the square matrix has to be assembled here.
    """
    names = list(tree.leaf_names)
    pairs = tree.get_distance_matrix()
    out = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a == b:
                continue
            key = (a, b) if (a, b) in pairs else (b, a)
            out[i, j] = pairs[key]
    return out, names


def _r_cophenetic(r_tree) -> tuple[np.ndarray, list[str]]:
    """ape tree -> (square cophenetic matrix, tip order as returned by ape)."""
    r_mat = R_APE.cophenetic_phylo(r_tree)
    names = [str(n) for n in as_array(R_APE.getTL(r_tree))]
    n_row = len(names)
    out = np.array(
        [[float(r_mat[i + 1, j + 1]) for j in range(n_row)] for i in range(n_row)],
        dtype=float,
    )
    return out, names


def _aligned(paleo: tuple[np.ndarray, list[str]], reference: tuple[np.ndarray, list[str]]):
    """Reorder a PaleoAST cophenetic matrix to the reference's tip order."""
    paleo_mat, paleo_names = paleo
    ref_mat, ref_names = reference
    missing = [n for n in ref_names if n not in paleo_names]
    if missing:
        raise AssertionError(f"ape returned tips absent from the PaleoAST tree: {missing}")
    index = {name: i for i, name in enumerate(paleo_names)}
    rows = [index[name] for name in ref_names]
    return paleo_mat[np.ix_(rows, rows)], ref_mat


class TestDistanceMethodsVsApe:
    """Verify tree building against ape."""

    def test_neighbor_joining_cophenetic_matches(self):
        """NJ tip-to-tip path lengths match ``ape::nj``."""
        from phylogenetics.distance_methods import DistanceMatrix, NeighborJoining

        paleo_tree = NeighborJoining().build(DistanceMatrix.from_array(DISTANCES, TAXA))

        # `as.dist` is stats::as.dist, not ape -- ape has no such function.
        r_dist = R_STATS.as_dist(r_matrix(DISTANCES))
        r_tree = R_APE.nj(r_dist)
        r_tree = R_APE.setRoot(r_tree, outgroup=TAXA[0])

        paleo, reference = _aligned(_cophenetic(paleo_tree), _r_cophenetic(r_tree))
        assert_allclose(
            paleo,
            reference,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Neighbor-Joining cophenetic matrix disagrees with ape::nj",
        )

    def test_upgma_cophenetic_matches(self):
        """UPGMA tip-to-tip path lengths match ``hclust(method='average')``.

        ape has no UPGMA function, so the comparison goes through
        ``stats::hclust`` -- the same function the R community uses for this --
        and then converts the dendrogram with ``ape::as.phylo``.
        """
        from phylogenetics.distance_methods import UPGMA, DistanceMatrix

        paleo_tree = UPGMA().build(DistanceMatrix.from_array(DISTANCES, TAXA))

        r_hclust = R_STATS.hclust(R_STATS.as_dist(r_matrix(DISTANCES)), method="average")
        r_tree = R_APE.as_phylo(r_hclust)

        paleo, reference = _aligned(_cophenetic(paleo_tree), _r_cophenetic(r_tree))
        assert_allclose(
            paleo,
            reference,
            rtol=1e-6,
            atol=1e-8,
            err_msg="UPGMA cophenetic matrix disagrees with stats::hclust(average)",
        )

    def test_ultrametric_property_holds_for_upgma(self):
        """UPGMA output is ultrametric; NJ output is not.

        Checked with the **three-point condition** -- a tree is ultrametric iff,
        for every triple of tips, the two largest of the three pairwise
        distances are equal.

        An earlier version of this test asserted that ``sum(d(i, .)) / 2`` is the
        same for every tip. That is not an identity for ultrametric trees: the
        quantity depends on the heights of the internal nodes, so two trees that
        are both ultrametric by construction can disagree on it. It failed here
        against a correct UPGMA, and the three-point condition below passes it.
        """
        from phylogenetics.distance_methods import UPGMA, DistanceMatrix, NeighborJoining

        paleo_mat, names = _cophenetic(UPGMA().build(DistanceMatrix.from_array(DISTANCES, TAXA)))
        assert len(set(names)) == len(TAXA)

        violations = [
            (i, j, k)
            for i, j, k in itertools.combinations(range(len(names)), 3)
            if not _three_point_holds(paleo_mat, i, j, k)
        ]
        assert not violations, (
            f"UPGMA output is not ultrametric; {len(violations)} of "
            f"{len(names) * (len(names) - 1) * (len(names) - 2) // 6} triples "
            f"violate the three-point condition, first: {violations[0]}"
        )

        # The same check must FAIL for NJ, otherwise it is not actually
        # discriminating -- a guard that passes everything catches nothing.
        nj_mat, _ = _cophenetic(NeighborJoining().build(DistanceMatrix.from_array(DISTANCES, TAXA)))
        assert any(
            not _three_point_holds(nj_mat, i, j, k) for i, j, k in itertools.combinations(range(len(names)), 3)
        ), "NJ came out ultrametric; the check cannot tell UPGMA from NJ"


class TestPICVsApe:
    """Verify independent contrasts against ``ape::pic``."""

    def test_pic_contrasts_match(self):
        """PIC residual contrasts match ``ape::pic``.

        ``ape::pic`` returns one residual per internal node: n-1 of them for an
        unrooted tree of n tips, which is exactly what ``compute_pic`` returns.
        The order is internal-node order, which is determined by the Newick
        string on both sides, so the two are comparable elementwise.
        """
        from phylogenetics import PhyloTree, compute_pic

        newick = "((A:0.1,B:0.2):0.05,(C:0.3,D:0.25):0.12);"
        traits = {"A": 1.0, "B": 3.0, "C": 2.0, "D": 5.0}

        paleo_contrasts, _pairs = compute_pic(PhyloTree.from_newick(newick), traits)

        r_tree = R_APE.read_tree(r_string_vector([newick]))
        r_x = r("c")(r("setNames")(r_vector(list(traits.values())), r_vector(list(traits))))
        r_pic = R_APE.pic(r_tree, x=r_x)

        r_contrasts = np.array([float(r_pic.rx2("pic")[i]) for i in range(len(paleo_contrasts))])

        assert_allclose(
            np.asarray(paleo_contrasts, dtype=float),
            r_contrasts,
            rtol=1e-6,
            atol=1e-8,
            err_msg="PIC contrasts disagree with ape::pic",
        )
        # n tips -> n-1 contrasts. A count mismatch would otherwise show up only
        # as a confusing shape error inside assert_allclose.
        assert len(paleo_contrasts) == len(traits) - 1
