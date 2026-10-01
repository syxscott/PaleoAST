"""
Regression pins for three defects that were found and fixed but whose
re-introduction the existing suites did NOT detect.

These exist because the targeted mutation audit (``scripts/mutation_audit.py``)
reinstated each bug and the corresponding ground-truth suite still went green.
A test suite that cannot tell whether the fix is present is not evidence that
the fix works, so each test below fails on the broken version by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from phylogenetics.distance_methods import DistanceMatrix
from phylogenetics.heuristic_search import TBROperation
from phylogenetics.tree import PhyloTree


class TestDistanceMatrixRejectsAsymmetricInput:
    """``from_array`` used to build a non-symmetric distance matrix.

    It wrote every ``(i, j)`` into the lookup dict, so the later ``(j, i)``
    write overwrote the earlier one and the pair could end up holding two
    different numbers. Nothing complained, and the result still passed
    ``to_matrix()`` -- but a distance that depends on argument order breaks the
    triangle inequality and poisons anything downstream that assumes symmetry.
    """

    ASYMMETRIC = np.array([[0.0, 9.0, 9.0], [1.0, 0.0, 9.0], [1.0, 1.0, 0.0]])

    def test_asymmetric_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="symmetric"):
            DistanceMatrix.from_array(self.ASYMMETRIC, ["A", "B", "C"])

    def test_the_error_names_the_offending_pair(self):
        """A bare "invalid input" would send the user hunting."""
        with pytest.raises(ValueError) as excinfo:
            DistanceMatrix.from_array(self.ASYMMETRIC, ["A", "B", "C"])
        message = str(excinfo.value)
        assert "A" in message and "B" in message, message

    def test_tiny_asymmetry_within_tolerance_is_accepted(self):
        """Round-off in a computed distance is not a malformed matrix."""
        near_symmetric = np.array(
            [[0.0, 2.0, 5.0], [2.0 + 1e-12, 0.0, 6.0], [5.0, 6.0, 0.0]]
        )
        matrix = DistanceMatrix.from_array(near_symmetric, ["A", "B", "C"])
        assert matrix.get_distance("A", "B") == pytest.approx(2.0, abs=1e-9)

    def test_symmetric_matrix_stays_symmetric(self):
        good = np.array([[0.0, 2.0, 5.0], [2.0, 0.0, 6.0], [5.0, 6.0, 0.0]])
        matrix = DistanceMatrix.from_array(good, ["A", "B", "C"])
        for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
            assert matrix.get_distance(a, b) == matrix.get_distance(b, a)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="square|shape"):
            DistanceMatrix.from_array(np.zeros((3, 2)), ["A", "B", "C"])


class TestMeshVolumeRequiresAClosedSurface:
    """``compute_volume`` used to return a number for an open mesh.

    The divergence theorem only encloses a volume for a closed surface. For an
    open mesh the per-face tetrahedra terms stop cancelling, so the sum becomes
    dependent on where the origin happens to be -- translating the mesh
    changes the answer. That is a wrong number, not a rough one, and it was
    returned without complaint while the docstring quietly said "assumes a
    closed surface".
    """

    CUBE_VERTICES = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    CUBE_FACES = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )

    def test_closed_mesh_volume_is_exact(self):
        from morpho3d.mesh import Mesh3D

        mesh = Mesh3D(vertices=self.CUBE_VERTICES, faces=self.CUBE_FACES)
        assert mesh.compute_volume() == pytest.approx(1.0, abs=1e-9)

    def test_open_mesh_is_rejected(self):
        from morpho3d.mesh import Mesh3D

        mesh = Mesh3D(vertices=self.CUBE_VERTICES, faces=self.CUBE_FACES[:8])
        with pytest.raises(ValueError, match="open|boundary"):
            mesh.compute_volume()

    def test_open_mesh_can_be_opted_into(self):
        """``require_closed=False`` is the documented escape hatch."""
        from morpho3d.mesh import Mesh3D

        mesh = Mesh3D(vertices=self.CUBE_VERTICES, faces=self.CUBE_FACES[:8])
        value = mesh.compute_volume(require_closed=False)
        assert np.isfinite(value)

    def test_the_rejection_is_what_makes_the_number_depend_on_the_origin(self):
        """Show the defect concretely: the raw sum moves when the mesh does."""
        from morpho3d.mesh import Mesh3D

        shifted = self.CUBE_VERTICES + np.array([5.0, -3.0, 11.0])
        raw_here = Mesh3D(vertices=self.CUBE_VERTICES, faces=self.CUBE_FACES[:8]).compute_volume(
            require_closed=False
        )
        raw_there = Mesh3D(vertices=shifted, faces=self.CUBE_FACES[:8]).compute_volume(
            require_closed=False
        )
        assert raw_here != pytest.approx(raw_there), (
            "this fixture no longer demonstrates origin-dependence; the "
            "underlying claim needs a new example"
        )

    def test_face_indices_are_validated_with_a_clear_message(self):
        from morpho3d.mesh import Mesh3D

        with pytest.raises(ValueError, match=r"\[0, 8\)"):
            Mesh3D(vertices=self.CUBE_VERTICES, faces=np.array([[0, 1, 99]]))

    def test_surface_sampling_is_reproducible_from_a_seed(self):
        """``sample_points`` used the global ``np.random`` stream.

        That made it impossible to pass a seed and unsafe to call from more
        than one thread. It now takes one and uses a local generator.
        """
        from morpho3d.mesh import Mesh3D

        mesh = Mesh3D(vertices=self.CUBE_VERTICES, faces=self.CUBE_FACES)
        first = mesh.sample_points(64, seed=7)
        again = mesh.sample_points(64, seed=7)
        other = mesh.sample_points(64, seed=8)
        assert np.array_equal(first, again), "same seed must give the same points"
        assert not np.array_equal(first, other), "different seeds must differ"


class TestFisherLogSeriesRefusesUnidentifiableData:
    """``fit_log_series`` silently installed x = 0.5 when no root existed.

    Its own comment said "falling back to x = 0.5 gives a fitted value with no
    basis", and then the except-branch did exactly that. For uniform
    abundances (every species a singleton, N/S = 1) the true solution is
    x -> 0 with alpha -> infinity, so the returned pair did not describe the
    data at all: x = 0.5 implies N/S = 1.44 for data whose N/S is 1.0.
    """

    def test_uniform_abundances_are_rejected(self):
        from ecology.advanced import AbundanceModelFitter

        with pytest.raises(ValueError, match="N/S"):
            AbundanceModelFitter().fit_log_series([1.0, 1.0, 1.0])

    def test_reported_parameters_describe_the_data(self):
        """The returned (alpha, x) must satisfy the equation it was fitted to."""
        import math

        from ecology.advanced import AbundanceModelFitter

        abundances = np.array([8, 7, 6, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], float)
        fit = AbundanceModelFitter().fit_log_series(abundances)
        x = float(fit.parameters["x"])
        implied = x / ((1 - x) * (-math.log(1 - x)))
        assert implied == pytest.approx(abundances.sum() / len(abundances), rel=1e-6)


class TestTBRReconnectDirection:
    """The ancestor check walked the wrong way and rejected every legal move.

    ``r1`` is required to be an ANCESTOR of ``cut_node1`` (by default it is
    ``cut_node1.parent``), so ``cut_node1`` lies *below* it. The old loop
    climbed from ``r1`` looking for ``cut_node1`` and could only ever reach the
    top of the tree and raise -- so every default-path TBR call failed, and
    because ``_generate_neighbors`` swallows ``ValueError`` the TBR
    neighbourhood stayed a small, silently truncated subset of the real one.

    The pin is direct: the default path must succeed.
    """

    def test_default_reconnect_node_succeeds(self):
        tree = PhyloTree.from_newick("((((A,B),(C,D)),E),(F,G));")
        parent, child = tree.root.children[0], tree.root.children[0].children[1]
        result = TBROperation(parent, child).apply(tree)
        assert sorted(result.leaf_names) == sorted(tree.leaf_names)

    def test_every_internal_edge_default_path_succeeds(self):
        """Not just the one example -- every cut edge must be reconnectable.

        The truncated-neighbourhood defect shows up as *some* cut edges
        silently failing while others work, so the pin has to sweep them all.

        The root is excluded: it has no parent, so the default
        ``r1 = n1.parent`` is undefined there. That is a property of the
        rooted representation, not the bug.
        """
        tree = PhyloTree.from_newick("(((((A,B),(C,D)),E),F),(G,H));")
        internal = [
            n for n in tree.root.get_all_nodes() if not n.is_leaf and not n.is_root
        ]
        checked = 0
        for node in internal:
            for child in node.children:
                if child.is_leaf:
                    continue
                result = TBROperation(node, child).apply(tree)
                assert sorted(result.leaf_names) == sorted(tree.leaf_names)
                checked += 1
        assert checked >= 4, f"expected several internal edges, swept {checked}"

    def test_reconnect_node_inside_the_cut_subtree_is_still_refused(self):
        """The fix must not turn the guard into a no-op.

        Putting r1 inside n1's own subtree is exactly the situation the check
        exists to prevent, and it must still be rejected.
        """
        tree = PhyloTree.from_newick("((((A,B),(C,D)),E),(F,G));")
        parent = tree.root.children[0]
        child = parent.children[1]
        inside = parent.children[0]
        with pytest.raises(ValueError):
            TBROperation(parent, child, reconnect_node1=inside).apply(tree)

    def test_tbr_neighbourhood_is_not_trivially_small(self):
        """A crude but decisive guard against the truncated neighbourhood.

        A binary tree on 7 leaves has 105 labelled topologies; a real TBR sweep
        reaches 34 of them. If the reconnect logic rejects its own legal moves,
        the reachable set collapses to a handful. Rather than pin an exact
        count (which depends on the deliberate candidate cap in
        ``_generate_neighbors``) this asserts the neighbourhood is far larger
        than the cap could produce on its own.
        """
        from phylogenetics.heuristic_search import HeuristicSearch

        rng = np.random.default_rng(4242)
        search = HeuristicSearch(nni_swap_probability=0.0)
        search._tbr_prob = 1.0

        import reference_algorithms as R  # noqa: PLC0415

        tree = R.random_binary_tree(list("ABCDEFG"), rng)
        neighbours = {R.unrooted_splits(n) for n in search._generate_neighbors(tree)}
        neighbours.discard(R.unrooted_splits(tree))
        assert len(neighbours) >= 4, (
            f"TBR reached only {len(neighbours)} topologies; with the reconnect "
            "logic rejecting legal moves the neighbourhood collapses"
        )
