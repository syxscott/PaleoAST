"""
================================================================================
PaleoAST Phylogenetics - Bipartition Bitmasks, Robinson-Foulds, MCCT
================================================================================

Bitmask encoding of tree bipartitions ("splits") plus the metrics built on top
of them: the unweighted Robinson-Foulds distance, the split-length weighted
Robinson-Foulds distance, split compatibility, and a BEAST-style maximum clade
credibility tree (MCCT).

This is a *port with simplification* of DendroPy's bit-processing machinery
(``dendropy.datamodel.treemodel._bipartition``,
``dendropy.calculate.treecompare``,
``dendropy.datamodel.treecollectionmodel.TreeCollection.maximum_product_of_split_support_tree``):
plain Python ``int`` bitmasks instead of cached ``Bipartition`` objects, string
labels instead of a shared ``TaxonNamespace``, and a rooted-tree encoding in
which every non-root node ``v`` contributes the split ``leafset(v) | rest``.

Encoding
--------
lsb0 (least-significant bit first): the taxon at index ``i`` of the *sorted*
leaf-label order of a tree gets bit ``1 << i``.  Sorting -- rather than the
first-seen order a shared DendroPy ``TaxonNamespace`` would give -- makes the
index a pure function of the leaf-label set, so two independently parsed trees
over the same taxa produce directly comparable bitmasks.

Splits are stored *normalized*: always the side that does NOT contain the
lowest-indexed taxon, so rotations and re-rootings of one topology yield
identical bitmasks.

Deliberate simplifications relative to DendroPy
-----------------------------------------------
* A bifurcating root makes its two children span the *same* unrooted
  bipartition; split-length maps sum the two edges under one key instead of
  dropping one of them.  The root's own branch length is excluded either way
  (it subtends no bipartition), matching DendroPy's unrooted-edge treatment.
* Split supports are unweighted: every tree counts once, so BEAST tree weights
  are ignored.
* :func:`is_compatible_bitmask_pair` runs the full four-way intersection test,
  which stays correct for *unnormalized* masks; DendroPy's three-check shortcut
  is only complete because normalization has already emptied the fourth
  quadrant.
* No incremental re-encoding: every call recomputes bitmasks in O(nodes).

References:
    - Robinson, D. F. & Foulds, L. R. (1979). Comparison of phylogenetic trees.
      Mathematical Biosciences 17(1-2):131-147.
    - Goyal, S., Richards, S. & Sagot, M.-F. (2012). Statistical comparison of
      phylogenetic trees through the edges. IEEE/ACM Transactions on
      Computational Biology and Bioinformatics 9(6):1704-1716.
    - Heled, A. & Bouckaert, R. R. (2014). Maximum clade credibility trees for
      BEAST phylogenetic dating. Methods in Ecology and Evolution 5(1):76-86.
    - Sukumaran, J. & Holder, M. T. DendroPy: a Python library for phylogenetic
      computing. Bioinformatics 26(12):1569-1571.

作者: PaleoAST Development Team
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping

from utils.exceptions import ValidationError

from .tree import PhyloNode, PhyloTree

__all__ = [
    "clade_bitmask",
    "is_compatible_bitmask_pair",
    "is_trivial_bitmask",
    "maximum_product_of_split_support_tree",
    "mcct",
    "normalize_bitmask",
    "robinson_foulds_distance",
    "split_bitmasks",
    "split_lengths",
    "split_support_frequencies",
    "taxon_bitmask_map",
    "weighted_robinson_foulds_distance",
]


# ---------------------------------------------------------------------------
# Taxon indexing
# ---------------------------------------------------------------------------
def _leaf_labels(tree: PhyloTree | None) -> list[str]:
    """Leaf labels of ``tree`` in traversal order (empty for a rootless tree)."""
    if tree is None or tree.root is None:
        return []
    return [leaf.name for leaf in tree.root.get_leaves()]


def _all_taxa_bitmask(index: Mapping[str, int]) -> int:
    """Bitmask with every taxon of ``index`` set."""
    mask = 0
    for bit in index.values():
        mask |= bit
    return mask


def taxon_bitmask_map(tree: PhyloTree) -> dict[str, int]:
    """
    Return ``{leaf label: bitmask}`` with one bit per leaf label.

    Bits follow the lsb0 convention over the sorted leaf labels: the ``i``-th
    label (in ``sorted()`` order) gets ``1 << i``.

    Parameters:
        tree: Tree whose leaves define the taxon set.

    Returns:
        Mapping of label -> single-bit mask (empty for a rootless tree).

    Raises:
        ValidationError: If a leaf has no label or a label is used more
            than once (leaves must be unique to be encodable).
    """
    labels = _leaf_labels(tree)
    unique: set[str] = set()
    repeated: set[str] = set()
    unlabelled = 0
    for label in labels:
        if not label:
            unlabelled += 1
            continue
        if label in unique:
            repeated.add(label)
        unique.add(label)

    if unlabelled:
        raise ValidationError(
            f"Cannot encode bipartitions: {unlabelled} leaf node(s) have no label; every leaf needs a unique label"
        )
    if repeated:
        raise ValidationError(
            f"Cannot encode bipartitions: leaf labels must be unique, got duplicates {sorted(repeated)}"
        )

    return {label: 1 << position for position, label in enumerate(sorted(unique))}


# ---------------------------------------------------------------------------
# Leafset (clade) bitmasks
# ---------------------------------------------------------------------------
def clade_bitmask(node: PhyloNode, index: Mapping[str, int]) -> int:
    """
    Bitmask of the leaves descending from ``node``.

    One postorder pass over the subtree, so a full-tree traversal visits each
    node exactly once.

    Parameters:
        node: Node (leaf or internal) to encode.
        index: ``{label: bit}`` map, normally from :func:`taxon_bitmask_map`.

    Returns:
        Union of the descendant leaf bits.

    Raises:
        ValidationError: If a leaf label is absent from ``index``.
    """
    if node.is_leaf:
        bit = index.get(node.name)
        if bit is None:
            raise ValidationError(f"Leaf '{node.name}' is not part of the provided taxon index")
        return bit

    mask = 0
    for child in node.children:
        mask |= clade_bitmask(child, index)
    return mask


def _node_bitmasks(tree: PhyloTree, index: Mapping[str, int]) -> dict[PhyloNode, int]:
    """Leafset bitmask of every node, computed in one postorder pass."""
    masks: dict[PhyloNode, int] = {}
    if tree.root is None:
        return masks

    for node in tree.root.postorder_traverse():
        if node.is_leaf:
            masks[node] = clade_bitmask(node, index)
            continue
        mask = 0
        for child in node.children:
            mask |= masks[child]
        masks[node] = mask
    return masks


def _bipartitions(
    tree: PhyloTree,
    index: Mapping[str, int],
    all_taxa: int,
) -> Iterator[tuple[int, float | None]]:
    """Yield ``(normalized split, branch length)`` for every non-root node."""
    if tree.root is None:
        return
    masks = _node_bitmasks(tree, index)
    for node in tree.root.preorder_traverse():
        if node.is_root:
            continue
        yield normalize_bitmask(masks[node], all_taxa), node.branch_length


# ---------------------------------------------------------------------------
# Bitmask algebra
# ---------------------------------------------------------------------------
def normalize_bitmask(bitmask: int, all_taxa_bitmask: int) -> int:
    """
    Return the canonical side of a split: the one *without* the first taxon.

    DendroPy's lsb0 convention -- the bit of the lowest-indexed taxon is forced
    to 0 by complementing within ``all_taxa_bitmask`` -- so that the same
    unrooted bipartition gets the same integer no matter how the tree is
    rotated or where it is rooted.

    Parameters:
        bitmask: Split (or leafset) to normalize.
        all_taxa_bitmask: Union of all taxon bits of the reference tree.

    Returns:
        Normalized bitmask; bits outside ``all_taxa_bitmask`` are dropped.
    """
    masked = bitmask & all_taxa_bitmask
    lowest = all_taxa_bitmask & -all_taxa_bitmask
    if lowest and masked & lowest:
        return masked ^ all_taxa_bitmask
    return masked


def is_trivial_bitmask(bitmask: int, all_taxa_bitmask: int) -> bool:
    """
    True if at most one taxon sits on either side of the split.

    Trivial splits are carried by every tree over the same taxon set, so they
    never contribute to the unweighted Robinson-Foulds distance -- but they do
    carry the terminal edges, which is why the weighted distance keeps them.
    """
    side = bitmask & all_taxa_bitmask
    other = all_taxa_bitmask ^ side
    return side.bit_count() <= 1 or other.bit_count() <= 1


def is_compatible_bitmask_pair(bitmask1: int, bitmask2: int, all_taxa_bitmask: int) -> bool:
    """
    True if two splits can coexist on one tree.

    Splits ``A|B`` and ``C|D`` are compatible iff at least one of the four
    intersections ``A∩C``, ``A∩D``, ``B∩C``, ``B∩D`` is empty.  Normalized
    masks all place the first taxon on the ``B``/``D`` side, so ``B∩D`` there
    is empty only in a degenerate case; the fourth check is kept because this
    function also accepts raw (unnormalized) leafsets, where DendroPy's
    three-check shortcut would answer falsely.
    """
    a = bitmask1 & all_taxa_bitmask
    c = bitmask2 & all_taxa_bitmask
    b = all_taxa_bitmask ^ a
    d = all_taxa_bitmask ^ c
    return not (a & c) or not (a & d) or not (b & c) or not (b & d)


# ---------------------------------------------------------------------------
# Split sets
# ---------------------------------------------------------------------------
def split_bitmasks(tree: PhyloTree) -> frozenset[int]:
    """
    Normalized, non-trivial splits spanned by the edges of ``tree``.

    Each non-root node contributes ``leafset(v) | rest``; the root itself
    contributes nothing (its "split" is the trivial all-taxa side).

    Parameters:
        tree: Tree to encode.

    Returns:
        Frozen set of normalized split bitmasks.
    """
    return _split_bitmasks(tree, taxon_bitmask_map(tree))


def _split_bitmasks(tree: PhyloTree, index: Mapping[str, int]) -> frozenset[int]:
    """Internal worker for :func:`split_bitmasks` with a pre-computed index."""
    if not index:
        return frozenset()
    all_taxa = _all_taxa_bitmask(index)
    return frozenset(
        split for split, _ in _bipartitions(tree, index, all_taxa) if not is_trivial_bitmask(split, all_taxa)
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _require_identical_taxa(
    index1: Mapping[str, int],
    index2: Mapping[str, int],
    operation: str,
) -> None:
    """Raise unless both mappings cover exactly the same labels."""
    only_first = sorted(set(index1) - set(index2))
    only_second = sorted(set(index2) - set(index1))
    if only_first or only_second:
        raise ValidationError(
            f"{operation} requires both trees to span the identical taxon set: "
            f"only in tree 1: {only_first}; only in tree 2: {only_second}"
        )


# ---------------------------------------------------------------------------
# Unweighted Robinson-Foulds
# ---------------------------------------------------------------------------
def robinson_foulds_distance(tree1: PhyloTree, tree2: PhyloTree) -> int:
    """
    Unweighted Robinson-Foulds distance: the split symmetric difference.

    Branch lengths are ignored; only the normalized, non-trivial split sets are
    compared, so re-rootings, rotations and length changes give 0.

    Parameters:
        tree1: First tree.
        tree2: Second tree.

    Returns:
        ``|splits(tree1) Δ splits(tree2)|``.

    Raises:
        ValidationError: If the trees do not span exactly the same taxon set,
            or if either tree has duplicate/absent leaf labels.
    """
    index1 = taxon_bitmask_map(tree1)
    index2 = taxon_bitmask_map(tree2)
    _require_identical_taxa(index1, index2, "The Robinson-Foulds distance")
    return len(_split_bitmasks(tree1, index1) ^ _split_bitmasks(tree2, index2))


# ---------------------------------------------------------------------------
# Split lengths and weighted Robinson-Foulds
# ---------------------------------------------------------------------------
def _all_bipartition_lengths(tree: PhyloTree, index: Mapping[str, int]) -> dict[int, float]:
    """
    ``{normalized split: summed branch length}`` over all non-root nodes.

    Trivial splits are kept on purpose: they are exactly the terminal edges,
    without which the weighted distance could not see leaf-length differences.
    The two children of a bifurcating root span the same bipartition, so their
    lengths add under one key.
    """
    if not index:
        return {}
    all_taxa = _all_taxa_bitmask(index)
    lengths: dict[int, float] = {}
    for split, length in _bipartitions(tree, index, all_taxa):
        lengths[split] = lengths.get(split, 0.0) + (0.0 if length is None else float(length))
    return lengths


def split_lengths(tree: PhyloTree) -> dict[int, float]:
    """
    Split-length profile of ``tree``, keyed by normalized split bitmask.

    Keys are encoded like :func:`split_bitmasks` but *without* dropping the
    trivial splits; a ``None`` branch length counts as ``0.0`` and the root's
    own branch length is excluded (it subtends no bipartition).

    Parameters:
        tree: Tree to profile.

    Returns:
        ``{split bitmask: total length}``.
    """
    return _all_bipartition_lengths(tree, taxon_bitmask_map(tree))


def weighted_robinson_foulds_distance(
    tree1: PhyloTree,
    tree2: PhyloTree,
    min_dist: float = 0.0,
) -> float:
    """
    Split-length weighted Robinson-Foulds distance (Goyal et al. 2012).

    Sums ``|length1 - length2|`` over the union of the normalized bipartitions
    of both trees; a bipartition missing from one tree contributes
    ``min_dist`` for that side (``0.0``, the default, reproduces DendroPy's
    semantics, where a split absent from a tree has length 0).

    Parameters:
        tree1: First tree.
        tree2: Second tree.
        min_dist: Length ascribed to a bipartition absent from one tree.

    Returns:
        The weighted distance.

    Raises:
        ValidationError: If the trees do not span exactly the same taxon set,
            or if either tree has duplicate/absent leaf labels.
    """
    index1 = taxon_bitmask_map(tree1)
    index2 = taxon_bitmask_map(tree2)
    _require_identical_taxa(index1, index2, "The weighted Robinson-Foulds distance")

    lengths1 = _all_bipartition_lengths(tree1, index1)
    lengths2 = _all_bipartition_lengths(tree2, index2)

    total = 0.0
    for split in set(lengths1) | set(lengths2):
        total += abs(lengths1.get(split, min_dist) - lengths2.get(split, min_dist))
    return total


# ---------------------------------------------------------------------------
# Split support and the MCCT
# ---------------------------------------------------------------------------
def _collection_split_sets(trees: list[PhyloTree]) -> list[frozenset[int]]:
    """Non-trivial normalized splits per tree, after collection-wide checks."""
    if not trees:
        raise ValidationError("Cannot compute split support: the tree collection is empty")

    reference = taxon_bitmask_map(trees[0])
    split_sets = [_split_bitmasks(trees[0], reference)]
    for tree in trees[1:]:
        index = taxon_bitmask_map(tree)
        _require_identical_taxa(reference, index, "Split support over a tree collection")
        split_sets.append(_split_bitmasks(tree, index))
    return split_sets


def _split_counts(split_sets: list[frozenset[int]]) -> Counter[int]:
    """Number of trees containing each split (a split counts once per tree)."""
    counts: Counter[int] = Counter()
    for splits in split_sets:
        counts.update(splits)
    return counts


def split_support_frequencies(trees: list[PhyloTree]) -> dict[int, float]:
    """
    ``{normalized split: fraction of trees carrying it}`` for a collection.

    Identical trees repeated ``k`` times contribute ``k`` observations.

    Parameters:
        trees: Tree collection (non-empty, one shared taxon set).

    Returns:
        Mapping of split bitmask -> frequency in ``[0, 1]``.

    Raises:
        ValidationError: If the collection is empty, trees disagree on their
            taxon set, or a tree has duplicate/absent leaf labels.
    """
    split_sets = _collection_split_sets(trees)
    counts = _split_counts(split_sets)
    total = len(split_sets)
    return {split: count / total for split, count in counts.items()}


def maximum_product_of_split_support_tree(trees: list[PhyloTree]) -> PhyloTree:
    """
    Return the maximum clade credibility tree of a collection (BEAST-style).

    Each candidate is scored by ``sum(log(freq(split)))`` over its *distinct*
    non-trivial splits, i.e. the log of the product of its split supports; the
    highest score wins and ties resolve to the earliest tree in the list.  A
    tree with no non-trivial splits (e.g. a single leaf or a star tree) scores
    ``0.0``.  The object returned is one of the input trees, not a copy.

    Parameters:
        trees: Tree collection (non-empty, one shared taxon set).

    Returns:
        The best-scoring tree from ``trees``.

    Raises:
        ValidationError: If the collection is empty, trees disagree on their
            taxon set, or a tree has duplicate/absent leaf labels.
    """
    split_sets = _collection_split_sets(trees)
    counts = _split_counts(split_sets)
    total = len(split_sets)

    best_tree = trees[0]
    best_score: float | None = None
    for tree, splits in zip(trees, split_sets, strict=True):
        score = 0.0
        for split in sorted(splits):  # fixed order: float sum stays reproducible
            score += math.log(counts[split] / total)
        if best_score is None or score > best_score:
            best_tree, best_score = tree, score
    return best_tree


# Public alias for :func:`maximum_product_of_split_support_tree`.
mcct = maximum_product_of_split_support_tree
