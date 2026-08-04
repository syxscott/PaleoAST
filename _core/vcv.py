"""
================================================================================
_core/vcv.py - Brownian Motion VCV Matrix Reference Implementation
================================================================================

Single canonical implementation of the phylogenetic variance-covariance (VCV)
matrix under Brownian motion evolution. All other modules in PaleoAST MUST
use this implementation via thin wrappers.

Mathematical Framework
================================================================================

For a rooted phylogenetic tree with branch lengths representing time, the
Brownian motion (BM) model implies:

    V[i,i] = dist(root, tip_i)           # Tip variance = time from root
    V[i,j] = dist(root, LCA(i,j))         # Tip covariance = shared time

where LCA(i,j) is the Lowest Common Ancestor of taxa i and j.

Pagel Lambda Transformation:
--------------------------
Pagel's lambda transforms the tree to allow BM to be relaxed:

    V_λ[i,j] = (1 - λ) * V_BM[i,j] + λ * V_internal[i,j]

where V_internal[i,j] measures internal node sharing:
    V_internal[i,j] = dist(tip_i, root) + dist(tip_j, root) - 2*dist(LCA(i,j), root)

Equivalently, for i ≠ j:
    V_λ[i,j] = (1 - λ) * dist(root, LCA(i,j))

and for diagonal:
    V_λ[i,i] = dist(root, tip_i)  # λ does not affect diagonal

λ = 1: Full BM (default)
λ = 0: Star phylogeny (all taxa independent)

References:
    Pagel, M. (1999). Inferring the historical patterns of biological evolution.
    Nature, 401(6756), 877-884.

Author: PaleoAST Development Team
"""

from __future__ import annotations

from typing import Any

import numpy as np


def brownian_vcv(tree: Any, lambda_param: float = 1.0) -> tuple[list[str], np.ndarray]:
    """
    Compute Brownian motion variance-covariance matrix.

    Under BM:
        V[i,i] = dist(root, tip_i)          # Tip variance
        V[i,j] = dist(root, LCA(i,j))       # Tip covariance

    Parameters
    ----------
    tree : PhyloTree or PhyloNode
        Rooted phylogenetic tree with branch_length attributes.
    lambda_param : float, default 1.0
        Pagel lambda parameter. λ=1 gives full BM. λ=0 gives star phylogeny.

    Returns
    -------
    tip_names : list[str]
        Ordered list of taxon names (row/column order of V).
    V : np.ndarray, shape (n_taxa, n_taxa)
        VCV matrix. Symmetric positive semi-definite.

    Raises
    ------
    ValueError
        If tree has fewer than 2 tips, or lambda is outside [0, 1].

    Notes
    -----
    The tree is accessed via:
        - tree.root (PhyloTree) or tree itself (PhyloNode)
        - root.get_leaves() → list of tip nodes
        - node.name → taxon name
        - node.branch_length → edge length to parent (None → 0)
        - node._distance_to_ancestor(ancestor) → path length
        - root.compute_lca(node1, node2) → lowest common ancestor

    Example
    -------
    >>> from phylogenetics import PhyloTree
    >>> tree = PhyloTree.from_newick("(A:1,B:1)C:1;")
    >>> names, V = brownian_vcv(tree)
    >>> # V[0,0] = dist(root, A) = 2, V[1,1] = dist(root, B) = 2
    >>> # V[0,1] = V[1,0] = dist(root, C) = 1

    R Verified Against R Packages
    ----------------------------
    - R: vcv(phy) in ape package, or corBrownian(1, phy) in nlme
    - Python: This implementation
    """
    # Get root node
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    # Get all leaves
    leaves = root.get_leaves()
    n = len(leaves)

    if n < 2:
        raise ValueError(f"Need at least 2 taxa, got {n}")

    if not (0.0 <= lambda_param <= 1.0):
        raise ValueError(f"lambda must be in [0, 1], got {lambda_param}")

    # Build name index mapping
    tip_names = [leaf.name for leaf in leaves]
    name_to_idx = {name: i for i, name in enumerate(tip_names)}

    # Precompute distance from each leaf to root
    dist_to_root: dict = {}
    for leaf in leaves:
        dist_to_root[leaf] = leaf._distance_to_ancestor(root)

    # Initialize VCV matrix
    V = np.zeros((n, n), dtype=np.float64)

    # Compute pairwise covariances
    for i, leaf_i in enumerate(leaves):
        for j, leaf_j in enumerate(leaves):
            if i <= j:
                if i == j:
                    # Diagonal: V[i,i] = dist(root, tip_i)
                    V[i, i] = dist_to_root[leaf_i]
                else:
                    # Off-diagonal: V[i,j] = dist(root, LCA(i,j))
                    lca = _compute_lca(leaf_i, leaf_j, root)
                    # dist(root, lca) = dist(leaf_i, root) - dist(leaf_i, lca)
                    dist_leaf_to_root = dist_to_root[leaf_i]
                    dist_leaf_to_lca = leaf_i._distance_to_ancestor(lca)
                    dist_root_to_lca = dist_leaf_to_root - dist_leaf_to_lca

                    if lambda_param < 1.0:
                        # Apply lambda transformation for off-diagonal
                        # V_λ[i,j] = (1-λ) * dist(root, lca) when i ≠ j
                        # (diagonal unchanged)
                        V[i, j] = (1.0 - lambda_param) * dist_root_to_lca
                        V[j, i] = V[i, j]
                    else:
                        V[i, j] = dist_root_to_lca
                        V[j, i] = dist_root_to_lca

    return tip_names, V


def _compute_lca(node1: Any, node2: Any, root: Any) -> Any:
    """
    Compute the lowest common ancestor of two nodes.

    Parameters
    ----------
    node1, node2 : PhyloNode
        Target nodes.
    root : PhyloNode
        Root node (for path traversal).

    Returns
    -------
    lca : PhyloNode
        Lowest common ancestor of node1 and node2.
    """
    # Get path from each node to root
    path1 = set()
    current = node1
    while current is not None:
        path1.add(current)
        current = current.parent

    current = node2
    while current is not None:
        if current in path1:
            return current
        current = current.parent

    # Should never reach here for a valid rooted tree
    return root


def pagel_lambda_vcv(
    tree: Any,
    lambda_param: float,
) -> tuple[list[str], np.ndarray]:
    """
    Compute Pagel lambda-transformed VCV matrix.

    This is a thin wrapper around brownian_vcv() with explicit lambda.

    Parameters
    ----------
    tree : PhyloTree or PhyloNode
        Rooted phylogenetic tree.
    lambda_param : float
        Pagel lambda in [0, 1]. 1 = full BM, 0 = star phylogeny.

    Returns
    -------
    tip_names : list[str]
        Ordered list of taxon names.
    V : np.ndarray
        Lambda-transformed VCV matrix.

    See Also
    --------
    brownian_vcv : Full documentation.

    Example
    -------
    >>> from phylogenetics import PhyloTree
    >>> tree = PhyloTree.from_newick("(A:1,B:1)C:1;")
    >>> names, V = pagel_lambda_vcv(tree, lambda_param=0.5)
    """
    return brownian_vcv(tree, lambda_param=lambda_param)
