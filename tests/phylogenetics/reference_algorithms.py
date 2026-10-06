"""Independent reference implementations for the phylogenetics ground-truth tests.

Why this file exists
--------------------
``phylogenetics/distance_methods.py`` (10.9% covered), ``heuristic_search.py``
(16.1%) and ``strict_consensus.py`` (19.0%) are the three least-verified
modules in the package, and all three are algorithm-heavy: tree building,
neighbourhood exchange over combinatorial topologies, and clade-set algebra.
That is exactly the shape of code where a silent bug hides, because a wrong
answer still *looks* like a tree.

There is no R available here, so the usual "compare against ape/phangorn"
route is closed. Instead everything below is derived from first principles:

* **Round-trip is the ground truth for tree building.** Take a tree with known
  topology, compute its patristic distance matrix, and ask the builder to
  recover the tree. For an *ultrametric* tree UPGMA is provably exact; for any
  *additive* tree neighbour joining is provably exact (Saitou & Nei 1987).
  So the round trip is a theorem, not a heuristic expectation -- and it needs
  no external reference to check against.

* **Reference UPGMA and NJ are written out longhand** from the textbook
  definitions, deliberately sharing no code with the production versions. A
  test comparing the two implementations catches a bug in either.

* **Neighbourhood properties are closed-form and exact.** A binary tree with
  n leaves has n-3 internal edges and each yields exactly 2 NNI alternatives,
  so NNI must produce exactly 2(n-3) distinct neighbours. TBR additionally
  satisfies a symmetry property (if B is a TBR neighbour of A, then A is a TBR
  neighbour of B) and a containment property (every NNI move is reachable by
  TBR). All three are decidable without enumerating the topology space.

* **Consensus is clade membership arithmetic.** A clade is in the strict
  consensus iff it is present in every input tree, and in the majority-rule
  consensus iff it is present in more than half. That is one line over an
  enumerated clade list -- a different shape of code from the set algebra in
  the production module.

``PhyloTree`` / ``PhyloNode`` are used only as data containers here. The thing
under test is the algorithm, not the node structure.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from phylogenetics.tree import PhyloNode, PhyloTree

__all__ = [
    "all_possible_clades",
    "canonical_newick",
    "clades_of",
    "leaf_key",
    "non_trivial_clades",
    "patristic_matrix",
    "random_binary_tree",
    "reference_nj",
    "reference_upgma",
    "topology_of",
    "unrooted_splits",
]


# =============================================================================
# Canonical tree forms (for equality comparisons)
# =============================================================================


def leaf_key(node) -> str:
    """The identity of a leaf.

    ``PhyloNode`` carries ``name`` and ``label`` as separate fields, and the
    two are NOT always both populated: ``PhyloTree.from_newick`` sets ``name``
    and leaves ``label`` as None, while a hand-built node usually sets both.
    Production code (e.g. ``strict_consensus._extract_clades_from_tree``) keys
    on ``name``, so the reference does too, with ``label`` as a fallback for
    nodes built with only a label.
    """
    value = node.name if node.name is not None else node.label
    return str(value) if value is not None else ""


def clades_of(tree: PhyloTree) -> set[frozenset[str]]:
    """Every clade in ``tree`` as a frozenset of leaf labels.

    Includes the trivial clades (single leaves) and the full leaf set, which
    is what a strict consensus has to consider.
    """
    out: set[frozenset[str]] = set()
    for node in tree.root.get_all_nodes():
        if node is None:
            continue
        leaves = frozenset(leaf_key(leaf) for leaf in node.get_leaves())
        if leaves:
            out.add(leaves)
    return out


def non_trivial_clades(tree: PhyloTree) -> set[frozenset[str]]:
    """Clades with at least two taxa and fewer than all taxa."""
    total = frozenset(tree.leaf_names)
    return {c for c in clades_of(tree) if 1 < len(c) < len(total)}


def all_possible_clades(labels: Sequence[str]) -> set[frozenset[str]]:
    """Every clade that could exist on any tree over ``labels``.

    The complement of a clade is a clade too, so consensus arithmetic must
    consider both sides or it will miss half the answer.
    """
    names = [str(name) for name in labels]
    full = frozenset(names)
    out: set[frozenset[str]] = set()
    for size in range(2, len(names)):
        for combo in itertools.combinations(names, size):
            clade = frozenset(combo)
            out.add(clade)
            out.add(full - clade)
    return out


def topology_of(tree: PhyloTree) -> str:
    """Newick with branch lengths stripped, rooted at the root.

    Two trees with the same rooted topology produce the same string, which is
    the right notion of "same tree" when branch lengths are not the question.
    """
    def render(node) -> str:
        if node.is_leaf:
            return leaf_key(node)
        return "(" + ",".join(render(c) for c in node.children) + ")"

    return render(tree.root)


def unrooted_splits(tree: PhyloTree) -> frozenset[frozenset[frozenset[str]]]:
    """The set of internal splits, i.e. the tree's *unrooted* topology.

    A root is an artefact of the algorithm: neighbour joining in particular
    roots its result wherever the final join happened, which need not be where
    the generating tree was rooted. A split {A,B} | rest is stored as the
    canonical pair ``{frozenset({'A','B'}), frozenset(rest)}`` with the
    smaller side first, so {A,B}|{C,D,E} and {C,D,E}|{A,B} are one split.

    Only clades whose *complement* also holds at least two taxa count. A clade
    of size n-1 hangs off a single terminal edge; it is a pendant edge, not an
    internal split, and including it would make two different trees look alike
    (or different) depending only on where the root happened to fall.

    Comparing this -- rather than the rooted Newick -- is what makes a
    round-trip test meaningful for NJ.
    """
    full = frozenset(tree.leaf_names)
    out: set[frozenset[frozenset[str]]] = set()
    for clade in clades_of(tree):
        other = full - clade
        if len(clade) < 2 or len(other) < 2:
            continue
        small, large = sorted((clade, other), key=lambda c: sorted(c))
        out.add(frozenset({small, large}))
    return frozenset(out)


def canonical_newick(tree: PhyloTree, places: int = 9) -> str:
    """Newick including branch lengths, for exact numeric comparison."""
    def render(node) -> str:
        length = float(node.branch_length or 0.0)
        if node.is_leaf:
            return f"{leaf_key(node)}:{length:.{places}f}"
        inner = ",".join(render(c) for c in node.children)
        return f"({inner}):{length:.{places}f}"

    return render(tree.root)


# =============================================================================
# Tree construction (reference)
# =============================================================================


def _leaf(name: str) -> PhyloNode:
    node = PhyloNode(name=str(name), label=str(name))
    node.branch_length = 0.0
    return node


def random_binary_tree(
    labels: Sequence[str],
    rng: np.random.Generator,
    *,
    ultrametric: bool = False,
) -> PhyloTree:
    """Build a random binary tree over ``labels``.

    ``ultrametric=True`` makes every leaf equidistant from the root, which is
    exactly the condition under which UPGMA inverts a tree without error.
    """
    nodes: list[PhyloNode] = [_leaf(name) for name in labels]

    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), size=2, replace=False).tolist())
        parent = PhyloNode(name="internal")
        parent.add_child(nodes[i])
        parent.add_child(nodes[j])
        parent.branch_length = 0.0
        nodes = [n for k, n in enumerate(nodes) if k not in (i, j)]
        nodes.append(parent)

    root = nodes[0]
    if ultrametric:
        # _make_ultrametric sets the leaf branch lengths too; zeroing them
        # afterwards would flatten the tree out of ultrametry.
        _make_ultrametric(root, rng)
    else:
        # Every node, leaves included, needs a POSITIVE length. Leaving the
        # leaves at zero makes each cherry a zero-distance pair, which makes
        # the metric degenerate and the topology unrecoverable -- an artefact
        # of the fixture, not a fact about neighbour joining.
        for node in root.get_all_nodes():
            if node.is_root:
                node.branch_length = 0.0
            else:
                node.branch_length = float(rng.uniform(0.05, 1.0))
    return PhyloTree(root=root)


def _make_ultrametric(root: PhyloNode, rng: np.random.Generator) -> None:
    """Make every leaf equidistant from the root, keeping the topology.

    Two leaves under different internal nodes traverse different ancestor
    chains, so simply giving each internal node a positive branch length
    does NOT produce an ultrametric tree -- the path sums come out unequal.
    Assigning internal *depths* first does:

      * internal branch length = depth(node) - depth(parent), with depths
        strictly increasing top-down and all below 1.0;
      * leaf branch length     = 1.0 - depth(parent).

    Every root-to-leaf path therefore sums to exactly 1.0.
    """
    order: list[PhyloNode] = []
    stack: list[PhyloNode] = [root]
    while stack:
        node = stack.pop()
        if not node.is_leaf:
            order.append(node)
            stack.extend(node.children)

    depths: dict[int, float] = {id(root): 0.0}
    for node in order:
        parent_depth = depths[id(node.parent)] if node.parent is not None else 0.0
        span = 1.0 - parent_depth
        depths[id(node)] = parent_depth + float(rng.uniform(0.1, 0.9)) * span

    for node in order:
        parent_depth = depths[id(node.parent)] if node.parent is not None else 0.0
        node.branch_length = depths[id(node)] - parent_depth
        for child in node.children:
            child.branch_length = 1.0 - depths[id(node)] if child.is_leaf else 0.0


# =============================================================================
# Patristic distances (reference)
# =============================================================================


def patristic_matrix(tree: PhyloTree) -> tuple[list[str], np.ndarray]:
    """Leaf-to-leaf path-length matrix, by explicit ancestor walking.

    Deliberately not ``PhyloNode.get_distance`` or ``get_distance_matrix``:
    the test must not lean on the same helper the production code uses.
    """
    leaves = sorted(str(name) for name in tree.leaf_names)
    by_label = {leaf_key(leaf): leaf for leaf in tree.root.get_leaves()}

    def distance(a: PhyloNode, b: PhyloNode) -> float:
        chain_a: dict[int, PhyloNode] = {id(a): a}
        for node in a.get_ancestors():
            chain_a[id(node)] = node
        chain_b: dict[int, PhyloNode] = {id(b): b}
        for node in b.get_ancestors():
            chain_b[id(node)] = node
        shared = set(chain_a) & set(chain_b)
        if not shared:
            raise ValueError("nodes are on different trees")
        # The shared ancestors are the LCA *and every ancestor above it*, so
        # picking an arbitrary one (e.g. via next(iter(set))) gives the wrong
        # split point. The LCA is the shared node furthest from the root.
        lca = max((chain_a[key] for key in shared), key=lambda n: len(n.get_ancestors()))

        def climb(node: PhyloNode) -> float:
            total = 0.0
            while node is not lca:
                total += float(node.branch_length or 0.0)
                node = node.parent
                if node is None:
                    raise ValueError("no common ancestor found")
            return total

        return climb(a) + climb(b)

    matrix = np.zeros((len(leaves), len(leaves)))
    for i, a in enumerate(leaves):
        for j, b in enumerate(leaves):
            if i < j:
                value = distance(by_label[a], by_label[b])
                matrix[i, j] = matrix[j, i] = value
    return leaves, matrix


# =============================================================================
# UPGMA (reference; Sokal & Michener 1958, Sneath & Sokal 1973)
# =============================================================================


def reference_upgma(labels: Sequence[str], distances: np.ndarray) -> PhyloTree:
    """Unweighted pair group method with arithmetic averages, longhand.

    Repeatedly fuse the two clusters with the smallest inter-cluster
    distance. The fused cluster's distance to any other is the size-weighted
    mean of the two it replaced.

    Branch lengths come from the molecular-clock construction: the new node
    sits at height ``d(a, b) / 2``, and each child hangs by
    ``new_height - child_height``. Using ``d/2`` for BOTH arms is only right
    for the first merge, when both children are leaves at height 0; for a
    later merge involving an already-built subtree it over-lengthens the arms
    and the result stops being ultrametric (which is UPGMA's defining
    property).
    """
    table: dict[tuple[int, int], float] = {}

    def get(i: int, j: int) -> float:
        if i == j:
            return 0.0
        return table[(min(i, j), max(i, j))]

    def put(i: int, j: int, value: float) -> None:
        table[(min(i, j), max(i, j))] = value

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            put(i, j, float(distances[i, j]))

    clusters: dict[int, PhyloNode] = {i: _leaf(labels[i]) for i in range(n)}
    sizes: dict[int, int] = {i: 1 for i in range(n)}
    heights: dict[int, float] = {i: 0.0 for i in range(n)}
    active = set(range(n))
    next_id = n

    def attach(parent: PhyloNode, child_id: int, node_height: float) -> None:
        child = clusters[child_id]
        child.branch_length = node_height - heights[child_id]
        parent.add_child(child)

    while len(active) > 2:
        a, b = min(
            ((a, b) for a, b in itertools.combinations(sorted(active), 2)),
            key=lambda pair: (get(*pair), pair),
        )
        merged = PhyloNode(name="internal")
        node_height = get(a, b) / 2.0
        attach(merged, a, node_height)
        attach(merged, b, node_height)

        for other in list(active):
            if other in (a, b):
                continue
            put(next_id, other, (
                sizes[a] * get(a, other) + sizes[b] * get(b, other)
            ) / (sizes[a] + sizes[b]))
        clusters[next_id] = merged
        sizes[next_id] = sizes[a] + sizes[b]
        heights[next_id] = node_height
        active.discard(a)
        active.discard(b)
        active.add(next_id)
        next_id += 1

    if len(active) == 1:
        return PhyloTree(root=clusters[next(iter(active))])
    a, b = sorted(active)
    node_height = get(a, b) / 2.0
    root = PhyloNode(name="root")
    attach(root, a, node_height)
    attach(root, b, node_height)
    return PhyloTree(root=root)


# =============================================================================
# Neighbour joining (reference; Saitou & Nei 1987)
# =============================================================================


def reference_nj(labels: Sequence[str], distances: np.ndarray) -> PhyloTree:
    """Neighbour joining, longhand from the Q-matrix definition.

    At each step minimise
        Q(i, j) = (n - 2) d(i, j) - r_i - r_j,    r_i = sum_k d(i, k)
    and reduce. Limb lengths come from the same Q matrix, so the
    reconstruction is exact whenever the input distances are additive.
    """
    n = len(labels)
    table: dict[tuple[int, int], float] = {}

    def get(i: int, j: int) -> float:
        if i == j:
            return 0.0
        return table[(min(i, j), max(i, j))]

    def put(i: int, j: int, value: float) -> None:
        table[(min(i, j), max(i, j))] = value

    for i in range(n):
        for j in range(i + 1, n):
            put(i, j, float(distances[i, j]))

    clusters: dict[int, PhyloNode] = {i: _leaf(labels[i]) for i in range(n)}
    active = list(range(n))
    next_id = n

    while len(active) > 2:
        m = len(active)
        row_sums = {i: sum(get(i, j) for j in active if j != i) for i in active}
        q = {
            (i, j): (m - 2) * get(i, j) - row_sums[i] - row_sums[j]
            for i, j in itertools.combinations(active, 2)
        }
        a, b = min(q, key=lambda pair: (q[pair], pair))

        limb_a = 0.5 * get(a, b) + (row_sums[a] - row_sums[b]) / (2.0 * (m - 2))
        clusters[a].branch_length = limb_a
        clusters[b].branch_length = get(a, b) - limb_a

        merged = PhyloNode(name="internal")
        merged.add_child(clusters[a])
        merged.add_child(clusters[b])

        for k in active:
            if k in (a, b):
                continue
            put(next_id, k, 0.5 * (get(a, k) + get(b, k) - get(a, b)))
        clusters[next_id] = merged
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        next_id += 1

    a, b = sorted(active)
    half = get(a, b) / 2.0
    clusters[a].branch_length = half
    clusters[b].branch_length = half
    root = PhyloNode(name="root")
    root.add_child(clusters[a])
    root.add_child(clusters[b])
    return PhyloTree(root=root)
