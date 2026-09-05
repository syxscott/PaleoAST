# =============================================================================
# FILE: statistics/pcm.py
# =============================================================================
"""
Phylogenetic Comparative Methods (PCM) for PaleoAST

Implements standard PCM techniques for analyzing trait evolution in a
phylogenetic context:

1. Phylogenetic Independent Contrasts (PIC)
   Felsenstein, J. (1985). Phylogenies and the Comparative Method.
   American Naturalist, 125(1), 1-15.

2. Ancestral State Reconstruction (ASR)
   Ancestral states computed via weighted squared-change parsimony
   or maximum likelihood under Brownian motion.

3. Phylogenetic Signal: Blomberg's K
   Blomberg et al. (2002). Testing for phylogenetic signal in
   comparative data. Evolution, 56(4), 717-745.

4. Phylogenetic ANOVA
   Tests for trait differences between a priori groups while
   accounting for phylogenetic non-independence.

Mathematical Framework:
==============================================================================

Phylogenetic Variance-Covariance Matrix (V):
    V_ij = t_ij = time since common ancestor of taxa i and j

For a continuous trait x evolving under Brownian Motion:
    E[x_i] = 0, Var[x_i] = σ² * t_i
    Cov[x_i, x_j] = σ² * t_ij

Phylogenetic Independent Contrast at node k:
    IC_k = (x_A - x_B) / sqrt(v_A + v_B)
    where v_A = sum of branch lengths from k to tip A
          v_B = sum of branch lengths from k to tip B

Blomberg's K:
    K = (Observed_MSE) / (Expected_MSE_under_BM)
    K ≈ 1 under BM; K > 1 strong phylogenetic signal

Author: PaleoAST Development Team
version: 1.0.1
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from config.i18n import _
from phylogenetics.signal import _blomberg_k_from_vcv
from phylogenetics.tree import NodeType, PhyloNode, PhyloTree
from utils.exceptions import ComputationError, ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class ContrastResult:
    """
    Container for Phylogenetic Independent Contrasts results.

    Attributes:
        contrasts: List of independent contrast values (one per internal node)
        se: Standard errors for each contrast
        node_names: Names of the internal nodes
        tip_values: Original trait values at leaf nodes
        branch_lengths: Sum of branch lengths from node to each descendant tip
        tree_height: Total tree height (root to deepest tip)
        n_contrasts: Number of contrasts computed
    """

    contrasts: NDArray[np.float64]
    se: NDArray[np.float64]
    node_names: list[str]
    tip_values: dict[str, float]
    branch_lengths: dict[str, float]
    tree_height: float
    n_contrasts: int

    def summary(self) -> str:
        """Generate summary text."""
        sig_contrasts = int(np.sum(np.abs(self.contrasts) > 1.96 * self.se))
        return (
            f"{_('Phylogenetic Independent Contrasts')}\n"
            f"{'=' * 50}\n"
            f"{_('Number of contrasts: {0}').format(self.n_contrasts)}\n"
            f"{_('Tree height: {0}').format(f'{self.tree_height:.4f}')}\n"
            f"{_('Significant contrasts (|z| > 1.96): {0}').format(sig_contrasts)}\n"
            f"{_('Mean absolute contrast: {0}').format(f'{np.mean(np.abs(self.contrasts)):.4f}')}"
        )


@dataclass
class AncestralStateResult:
    """
    Container for Ancestral State Reconstruction results.

    Attributes:
        node_states: Reconstructed trait values at each internal node
        node_names: Names of internal nodes (including root)
        tip_values: Original/fitted trait values at leaf nodes
        model: Evolution model used ('bm' = Brownian motion, 'ou' = Ornstein-Uhlenbeck)
        ou_alpha: OU alpha parameter (if applicable)
        ou_theta: OU theta parameter (if applicable)
    """

    node_states: dict[str, float]
    node_names: list[str]
    tip_values: dict[str, float]
    model: str = "bm"
    ou_alpha: float | None = None
    ou_theta: float | None = None

    def summary(self) -> str:
        """Generate summary text."""
        n_internal = len([v for v in self.node_states.values() if v is not None])
        tip_vals = list(self.tip_values.values())
        min_tip = min(tip_vals)
        max_tip = max(tip_vals)
        return (
            f"{_('Ancestral State Reconstruction')}\n"
            f"{'=' * 50}\n"
            f"{_('Model: {0}').format(self.model.upper())}\n"
            f"{_('Internal nodes reconstructed: {0}').format(n_internal)}\n"
            f"{_('Trait range (tips): {0:.3f} to {1:.3f}').format(min_tip, max_tip)}"
        )


@dataclass
class PhylogeneticSignalResult:
    """
    Container for phylogenetic signal measurement results.

    Attributes:
        k: Blomberg's K statistic
        z: Z-score for K (standardized by randomizations)
        p_value: P-value from permutation test
        n_randomizations: Number of permutations used
        lambda_: Pagel's lambda (ML-estimated). The trailing underscore
                 is required because ``lambda`` is a Python keyword and
                 cannot be used as a dataclass field name; access the
                 value via ``result.lambda_``.
        lambda_se: Standard error of Pagel's lambda
        vcv_matrix: Phylogenetic variance-covariance matrix
        tip_names: Names of taxa (rows/cols of VCV matrix)
    """

    k: float
    z: float
    p_value: float
    n_randomizations: int
    lambda_: float | None = None
    lambda_se: float | None = None
    vcv_matrix: NDArray[np.float64] | None = None
    tip_names: list[str] | None = None

    def summary(self) -> str:
        """Generate summary text."""
        if self.k < 0.5:
            interpretation = _("Low phylogenetic signal (convergent evolution)")
        elif self.k < 1.5:
            interpretation = _("Moderate phylogenetic signal (BM-like)")
        else:
            interpretation = _("High phylogenetic signal (conserved trait)")
        signal_label = _("Phylogenetic Signal (Blomberg K)")
        return (
            f"{signal_label}\n"
            f"{'=' * 50}\n"
            f"{_('K statistic: {0}').format(f'{self.k:.4f}')}\n"
            f"{_('Z-score: {0}').format(f'{self.z:.4f}')}\n"
            f"{_('P-value: {0}').format(f'{self.p_value:.4f}')}\n"
            f"{_('Interpretation: {0}').format(interpretation)}"
        )


@dataclass
class PhyloANOVAResult:
    """
    Container for Phylogenetic ANOVA results.

    Attributes:
        f_statistic: F-statistic (between-group / within-group variance)
        p_value: P-value from permutation test
        n_permutations: Number of permutations used
        groups: List of group names
        n_groups: Number of groups
        n_tips: Total number of taxa
        ss_between: Sum of squares between groups
        ss_within: Sum of squares within groups
        ms_between: Mean square between groups
        ms_within: Mean square within groups
        contrast_values: List of phylogenetic contrast values used
        group_labels: Group assignment for each tip
    """

    f_statistic: float
    p_value: float
    n_permutations: int
    groups: list[str]
    n_groups: int
    n_tips: int
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    contrast_values: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    group_labels: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary text."""
        sig = "**" if self.p_value < 0.01 else ("*" if self.p_value < 0.05 else "")
        return (
            f"{_('Phylogenetic ANOVA')}\n"
            f"{'=' * 50}\n"
            f"{_('F-statistic: {0}').format(f'{self.f_statistic:.4f}')}\n"
            f"{_('P-value: {0}').format(f'{self.p_value:.4f} {sig}')}\n"
            f"{_('Permutations: {0}').format(self.n_permutations)}\n"
            f"{_('Groups: {0}').format(', '.join(self.groups))}\n"
            f"{_('SS between: {0:.4f}, SS within: {1:.4f}').format(self.ss_between, self.ss_within)}"
        )


# =============================================================================
# Core PCM Algorithms
# =============================================================================


def _compute_vcv_matrix(root: PhyloNode) -> tuple[dict[str, int], NDArray[np.float64]]:
    """
    Compute the phylogenetic variance-covariance matrix under Brownian
    motion (Felsenstein 1985 / ape convention):

        V_ii = total branch length from root to tip i
        V_ij = shared path length from root to LCA(i, j)   (i ≠ j)

    The previous implementation returned patristic DISTANCES (zero
    diagonal), which is singular and unusable for GLS/PGLS; it also
    inverted the semantics relative to phylogenetics/signal.py (2026-09
    review C25).

    Parameters:
        root: Root node of the tree

    Returns:
        (name_to_idx, VCV matrix)
    """
    leaves = root.get_leaves()
    n = len(leaves)
    names = [leaf.name for leaf in leaves]
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Compute dist from root for each node
    dist_to_root: dict[PhyloNode, float] = {}

    def compute_distances(node: PhyloNode, dist: float) -> None:
        dist_to_root[node] = dist
        for child in node.children:
            child_dist = dist + (child.branch_length or 0.0)
            compute_distances(child, child_dist)

    compute_distances(root, 0.0)

    # Build VCV matrix
    VCV = np.zeros((n, n), dtype=np.float64)
    for i, leaf_i in enumerate(leaves):
        VCV[i, i] = dist_to_root[leaf_i]
        for j in range(i + 1, n):
            leaf_j = leaves[j]
            lca = root.compute_lca(leaf_i, leaf_j)
            shared = dist_to_root[lca] if lca is not None else 0.0
            VCV[i, j] = shared
            VCV[j, i] = shared

    return name_to_idx, VCV


def _compute_contrasts_recursive(
    node: PhyloNode,
    trait_values: dict[str, float],
) -> tuple[float | None, float, list[tuple[float, float, str]], list[str]]:
    """
    Recursively compute independent contrasts (Felsenstein 1985).

    Post-order traversal: process children first, then compute contrast at
    parent. Returns ``(reconstructed_value, cum_variance, contrasts_list,
    node_names)`` where:

    - ``reconstructed_value`` is the inverse-variance weighted estimate of
      the trait at ``node`` using only its descendants.
    - ``cum_variance`` is the variance of that reconstruction *evaluated at
      this node* — i.e. the variance accumulated along the subtree BELOW
      the node, NOT including the node's own branch_length. The caller
      (parent) is responsible for adding ``child.branch_length`` when it
      uses this value in its own contrast denominator. This matches the
      Felsenstein (1985) convention: for a binary parent P with children
      A and B,

          Var(recon_A - recon_B | P) = (V_A + branch_A) + (V_B + branch_B)

      where ``V_A = cum_variance(A)``. The previous implementation used
      ``v_i = cum_variance(child_i)`` and made leaves return
      ``branch_length`` as their ``cum_variance``; that double-counted
      leaf branch lengths while leaving *internal* child branch lengths
      uncounted, so IC standardization was systematically biased —
      especially on deep trees. The fix below keeps the convention
      uniform: leaves return ``V = 0`` and every parent adds the
      child's branch_length explicitly.
    - ``contrasts_list`` entries are ``(standardized_contrast, se, node_name)``.
    """
    if node.is_leaf:
        leaf_name = node.name
        val = trait_values.get(leaf_name)
        # Leaf has no descendants -> cum_variance below the leaf is 0.
        return val, 0.0, [], [] if val is None else [leaf_name]

    # Process all children first
    child_results: list[tuple[PhyloNode, float, float]] = []
    all_contrasts: list[tuple[float, float, str]] = []
    all_names: list[str] = []

    for child in node.children:
        c_val, c_cum_var, c_ic, c_names = _compute_contrasts_recursive(child, trait_values)
        if c_val is not None:
            child_results.append((child, c_val, c_cum_var))
            all_contrasts.extend(c_ic)
            all_names.extend(c_names)

    if len(child_results) < 2:
        return None, 0.0, all_contrasts, all_names

    if len(child_results) == 2:
        # Binary node: single contrast (Felsenstein 1985)
        child1, val1, var1 = child_results[0]
        child2, val2, var2 = child_results[1]
        # Variance of each child's reconstruction *at this node* includes
        # the child's own branch_length on top of the descendant variance.
        v1 = var1 + (child1.branch_length or 0.0)
        v2 = var2 + (child2.branch_length or 0.0)
        # Standardized contrast IC = (x_A - x_B) / sqrt(v_A + v_B)
        contrast = (val1 - val2) / math.sqrt(v1 + v2)
        se = math.sqrt(v1 + v2)
        node_name = node.name or f"node_{id(node)}"
        all_contrasts.append((contrast, se, node_name))
        all_names.append(node_name)
        # Variance of this node's reconstruction (at this node, excluding
        # this node's branch_length): for the inverse-variance weighted
        # mean of the two child reconstructions this is v1*v2/(v1+v2)
        # (Felsenstein 1985). The previous v1+v2 is the variance of the
        # CONTRAST, not of the reconstruction, and inflated the SEs of
        # every contrast above the first divergence.
        cum_var = (v1 * v2) / (v1 + v2) if (v1 + v2) > 0 else 0.0
        # Reconstructed value: inverse-variance weighted mean
        if v1 > 0 and v2 > 0:
            recon = (val1 / v1 + val2 / v2) / (1.0 / v1 + 1.0 / v2)
        else:
            recon = (val1 + val2) / 2.0
        return recon, cum_var, all_contrasts, all_names

    # Multi-way node: reduce iteratively (Felsenstein 1985)
    # Iteratively contrast adjacent children, combining one each step.
    # Build list of (value, descendant_var, branch_length, child) — the
    # branch_length is the edge from the child up to *this* node and must
    # be added to the descendant variance before computing the contrast.
    active: list[tuple[float, float, float, PhyloNode]] = [
        (val, cvar, (child.branch_length or 0.0), child)
        for child, val, cvar in child_results
    ]

    while len(active) > 1:
        val1, var1, bl1, _ch1 = active[0]
        val2, var2, bl2, _ch2 = active[1]
        # Variance at this node of each child's reconstruction.
        v1 = var1 + bl1
        v2 = var2 + bl2
        # Standardized contrast
        contrast = (val1 - val2) / math.sqrt(v1 + v2)
        se = math.sqrt(v1 + v2)
        node_name = f"{node.name}_c{len(active) - 2}" if node.name else f"node_{id(node)}_c{len(active) - 2}"
        all_contrasts.append((contrast, se, node_name))
        all_names.append(node_name)
        # Combined subtree: weighted mean; combined descendant variance
        # v1*v2/(v1+v2) (variance of the inverse-variance weighted mean).
        combined_cvar = (v1 * v2) / (v1 + v2) if (v1 + v2) > 0 else 0.0
        if v1 > 0 and v2 > 0:
            combined_val = (val1 / v1 + val2 / v2) / (1.0 / v1 + 1.0 / v2)
        else:
            combined_val = (val1 + val2) / 2.0
        combined_child = PhyloNode(name="combined", node_type=NodeType.INTERNAL, branch_length=0.0)
        # The combined pseudo-child lives at this node, so its edge to
        # this node has length 0; any future propagation up the tree will
        # pick up this node's branch_length via the outer caller.
        active = [(combined_val, combined_cvar, 0.0, combined_child), *active[2:]]

    # Reconstructed value at this node = inverse-variance weighted mean of
    # immediate child reconstructions (each child's variance at this node
    # includes its branch_length).
    total_w = 0.0
    weighted_sum = 0.0
    for child, val, cvar in child_results:
        v = cvar + (child.branch_length or 0.0)
        w = 1.0 / max(v, 0.0001)
        weighted_sum += w * val
        total_w += w
    recon = weighted_sum / total_w if total_w > 0 else (active[0][0] if active else 0.0)
    # Variance at this node = variance of the inverse-variance weighted
    # mean of the child reconstructions = 1/Σ(1/v_i) where
    # v_i = descendant_var + branch_length. Excludes this node's own
    # branch_length (the parent adds that when this node is a child).
    inv_sum = sum(1.0 / v for v in (cvar + (child.branch_length or 0.0) for child, _val, cvar in child_results) if v > 0)
    cum_var = 1.0 / inv_sum if inv_sum > 0 else 0.0
    return recon, cum_var, all_contrasts, all_names


# =============================================================================
# Main Analyzer Class
# =============================================================================


class PCMAnalyzer:
    """
    Phylogenetic Comparative Methods Analyzer.

    Provides a unified interface for PIC, ancestral state reconstruction,
    phylogenetic signal measurement, and phylogenetic ANOVA.

    Example:
        >>> analyzer = PCMAnalyzer()
        >>> tree = PhyloTree.from_newick("(A:0.1,B:0.2)C:0.3;")
        >>> traits = {"A": 2.0, "B": 4.0}
        >>> result = analyzer.compute_contrasts(tree, traits)
        >>> print(result.summary())
    """

    def __init__(self, n_randomizations: int = 999) -> None:
        """
        Initialize PCM analyzer.

        Parameters:
            n_randomizations: Number of permutations for significance testing
        """
        self._n_randomizations = n_randomizations
        self._logger = logging.getLogger(f"{__name__}.PCMAnalyzer")
        self._logger.info(f"PCMAnalyzer initialized (n_randomizations={n_randomizations})")

    # =========================================================================
    # Phylogenetic Independent Contrasts
    # =========================================================================

    def compute_contrasts(
        self,
        tree: PhyloTree,
        trait_values: dict[str, float],
    ) -> ContrastResult:
        """
        Compute phylogenetic independent contrasts.

        For each internal node with two descendant tips:
            IC = (x_A - x_B) / sqrt(v_A + v_B)
        where v_A, v_B = sum of branch lengths from node to each tip.

        Parameters:
            tree: Phylogenetic tree
            trait_values: {taxon_name: trait_value} dictionary

        Returns:
            ContrastResult with contrasts, standard errors, and summary statistics

        Raises:
            ValidationError: If tree or trait data is invalid
        """
        self._logger.info(f"Computing PIC for {len(trait_values)} taxa")

        if tree.root is None:
            raise ValidationError(_("Tree has no root node"))

        leaves = tree.root.get_leaves()
        leaf_names = set(l.name for l in leaves)
        trait_names = set(trait_values.keys())
        missing = leaf_names - trait_names
        if missing:
            raise ValidationError(_("Trait data missing for taxa: {0}").format(", ".join(sorted(missing))))

        # Filter tree to taxa with trait data
        # Prune taxa not in trait_values
        working_tree = self._prune_tree(tree, set(trait_values.keys()))
        if working_tree.root is None:
            raise ValidationError(_("No matching taxa after pruning"))

        # Compute tree height
        tree_height = working_tree.root.compute_total_length()

        # Compute contrasts via recursive traversal
        _recon, _cumvar, contrasts_data, _unused = _compute_contrasts_recursive(working_tree.root, trait_values)

        if not contrasts_data:
            raise ComputationError(_("No valid contrasts computed"))

        contrasts_arr = np.array([c[0] for c in contrasts_data], dtype=np.float64)
        se_arr = np.array([c[1] for c in contrasts_data], dtype=np.float64)
        node_names = [c[2] for c in contrasts_data]

        result = ContrastResult(
            contrasts=contrasts_arr,
            se=se_arr,
            node_names=node_names,
            tip_values=trait_values,
            branch_lengths={},
            tree_height=tree_height,
            n_contrasts=len(contrasts_arr),
        )
        self._logger.info(f"PIC computed: {result.n_contrasts} contrasts")
        return result

    # =========================================================================
    # Ancestral State Reconstruction
    # =========================================================================

    def reconstruct_ancestral_states(
        self,
        tree: PhyloTree,
        trait_values: dict[str, float],
        model: str = "bm",
    ) -> AncestralStateResult:
        """
        Reconstruct ancestral states via weighted squared-change parsimony.

        Under Brownian Motion model, the maximum likelihood estimate of the
        ancestral state at node k is the weighted average of descendant
        tip values, weighted by the inverse of their phylogenetic variances.

        Parameters:
            tree: Phylogenetic tree
            trait_values: {taxon_name: trait_value} dictionary
            model: Evolution model ('bm' = Brownian motion, 'ou' = Ornstein-Uhlenbeck)

        Returns:
            AncestralStateResult with reconstructed states at each node
        """
        self._logger.info(f"Reconstructing ancestral states (model={model})")

        if tree.root is None:
            raise ValidationError(_("Tree has no root node"))

        working_tree = self._prune_tree(tree, set(trait_values.keys()))
        if working_tree.root is None:
            raise ValidationError(_("No matching taxa after pruning"))

        node_states: dict[str, float] = {}
        internal_names: list[str] = []

        def assign_states(node: PhyloNode) -> tuple[float, float] | None:
            """后序遍历: 返回 (子树重建值, 子树累积方差, 不含自身枝长)。"""
            if node.is_leaf:
                val = trait_values.get(node.name)
                return None if val is None else (val, 0.0)

            child_res = []
            for child in node.children:
                res = assign_states(child)
                if res is not None:
                    child_res.append((child, res[0], res[1]))

            if not child_res:
                return None

            if len(child_res) == 1:
                child, val, cvar = child_res[0]
                return val, cvar + (child.branch_length or 0.0)

            # ML/BM 权重: 逆方差 1/(子树累积方差 + 子枝长)。
            # 旧实现 w = 1/枝长 忽略子树方差 (等价于简化 SQU 且深节点
            # 被过度加权), 且 `or 0.001` 把真实的 0 枝长当成缺失。
            total_w = 0.0
            weighted_sum = 0.0
            for child, val, cvar in child_res:
                v = cvar + (child.branch_length or 0.0)
                if v <= 0:
                    v = 1e-10
                total_w += 1.0 / v
                weighted_sum += val / v

            recon = weighted_sum / total_w if total_w > 0 else np.mean([v for _, v, _ in child_res])
            pooled = 1.0 / total_w if total_w > 0 else 0.0

            node_name = node.name or f"node_{id(node)}"
            node_states[node_name] = recon
            internal_names.append(node_name)
            return recon, pooled

        assign_states(working_tree.root)

        result = AncestralStateResult(
            node_states=node_states,
            node_names=internal_names,
            tip_values=trait_values,
            model=model,
        )
        self._logger.info(f"ASR completed: {len(node_states)} internal nodes reconstructed")
        return result

    # =========================================================================
    # Phylogenetic Signal (Blomberg's K)
    # =========================================================================

    def compute_phylogenetic_signal(
        self,
        tree: PhyloTree,
        trait_values: dict[str, float],
        n_randomizations: int | None = None,
        random_seed: int | None = None,
    ) -> PhylogeneticSignalResult:
        """
        Measure phylogenetic signal using Blomberg's K.

        K = Var(IC) / E_BM[Var(IC)]
             = (sum IC_i² / n) / (sum v_i / n)
        where IC_i = independent contrast at node i
              v_i = sum of branch lengths from node i to tips

        Interpretation:
            K < 1: trait evolves faster than expected under BM (convergence)
            K ≈ 1: trait evolves as expected under BM
            K > 1: trait is more conserved than expected (phylogenetic niche conservatism)

        Parameters:
            tree: Phylogenetic tree
            trait_values: {taxon_name: trait_value} dictionary
            n_randomizations: Number of permutations (defaults to constructor value)
            random_seed: Optional seed for the permutation RNG so the
                p-value and Z-score are reproducible.

        Returns:
            PhylogeneticSignalResult with K, Z-score, and p-value
        """
        n_r = n_randomizations or self._n_randomizations
        self._logger.info(
            f"Computing Blomberg's K (n_perm={n_r}, random_seed={random_seed})"
        )

        if tree.root is None:
            raise ValidationError(_("Tree has no root node"))

        working_tree = self._prune_tree(tree, set(trait_values.keys()))
        if working_tree.root is None:
            raise ValidationError(_("No matching taxa after pruning"))

        leaves = working_tree.root.get_leaves()
        n_tips = len(leaves)

        if n_tips < 3:
            raise ValidationError(_("Need at least 3 taxa with trait data"))

        _name_to_idx, VCV = _compute_vcv_matrix(working_tree.root)
        tip_names = [l.name for l in leaves]

        # Canonical Blomberg et al. (2003) K (scale-invariant):
        #     K = s²_ord / (σ̂²_GLS · tr(V)/n)
        # computed from the ape-convention VCV via the single shared
        # reference implementation in phylogenetics.signal. The previous
        # ratio Σ IC²/Σv is the BM rate estimate σ̂² — it scales with
        # trait units² and is not a signal statistic.
        tip_array = np.array([trait_values[l.name] for l in leaves], dtype=np.float64)
        K = _blomberg_k_from_vcv(tip_array, VCV)

        # Compute Z-score and p-value via permutations. Use a dedicated
        # Generator when a seed is supplied so the test is reproducible.
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random

        perm_Ks: list[float] = []
        for _ in range(n_r):
            perm_y = rng.permutation(tip_array)
            perm_Ks.append(_blomberg_k_from_vcv(perm_y, VCV))

        perm_Ks_arr = np.array(perm_Ks)
        z = (K - np.mean(perm_Ks_arr)) / np.std(perm_Ks_arr) if np.std(perm_Ks_arr) > 0 else 0.0
        # add-one corrected permutation p-value
        p_value = float((np.sum(perm_Ks_arr >= K) + 1.0) / (len(perm_Ks_arr) + 1.0))

        result = PhylogeneticSignalResult(
            k=K,
            z=z,
            p_value=p_value,
            n_randomizations=n_r,
            vcv_matrix=VCV,
            tip_names=tip_names,
        )
        self._logger.info(f"Blomberg's K = {K:.4f}, p = {p_value:.4f}")
        return result

    # =========================================================================
    # Phylogenetic ANOVA
    # =========================================================================

    def phylogenetic_anova(
        self,
        tree: PhyloTree,
        trait_values: dict[str, float],
        group_labels: dict[str, str],
        n_permutations: int | None = None,
        random_seed: int | None = None,
    ) -> PhyloANOVAResult:
        """
        Phylogenetic ANOVA: test for trait differences between groups.

        Uses phylogenetic independent contrasts to account for phylogenetic
        non-independence. Contrasts are assigned to groups based on which
        group's tips they contrast.

        F = MS_between / MS_within

        where:
            SS_between = sum_j n_j * (mean_j - grand_mean)²
            SS_within = sum_ij (x_ij - mean_j)²
            using phylogenetically independent contrasts

        Parameters:
            tree: Phylogenetic tree
            trait_values: {taxon_name: trait_value} dictionary
            group_labels: {taxon_name: group_name} dictionary
            n_permutations: Number of permutations (defaults to constructor value)
            random_seed: Optional seed for the permutation RNG so the
                p-value is reproducible.

        Returns:
            PhyloANOVAResult with F-statistic, p-value, and ANOVA table
        """
        n_p = n_permutations or self._n_randomizations
        self._logger.info(
            f"Phylogenetic ANOVA (n_perm={n_p}, random_seed={random_seed})"
        )

        if tree.root is None:
            raise ValidationError(_("Tree has no root node"))

        working_tree = self._prune_tree(tree, set(trait_values.keys()))
        if working_tree.root is None:
            raise ValidationError(_("No matching taxa after pruning"))

        # Validate groups
        tips_with_groups = {k: v for k, v in group_labels.items() if k in trait_values}
        if not tips_with_groups:
            raise ValidationError(_("No valid group assignments"))

        groups = sorted(set(tips_with_groups.values()))
        n_groups = len(groups)

        # Compute contrasts
        _recon, _cumvar, contrasts_data, _unused = _compute_contrasts_recursive(working_tree.root, trait_values)
        ic_arr = np.array([c[0] for c in contrasts_data], dtype=np.float64)

        if len(ic_arr) < 2 or n_groups < 2:
            raise ValidationError(_("Need at least 2 groups and 2 contrasts"))

        # Assign each contrast to "between-group" or "within-group" using the
        # standard Garland (1993) / Garland et al. phylogenetic-ANOVA
        # convention: a contrast at node P is *between-group* when the two
        # direct children of P have *different* dominant tip groups; it is
        # *within-group* when both children share the same dominant group.
        # Only between-group contrasts carry information about group
        # differences; within-group contrasts estimate the residual
        # (phylogenetically corrected) variance. The previous
        # implementation labelled each contrast with the single dominant
        # group of the entire contrast subtree, which silently absorbed
        # between-group signal into the within-group term and vice versa.
        node_to_tips: dict[int, set[str]] = {}

        def get_tip_names(node: PhyloNode) -> set[str]:
            node_id = id(node)
            if node_id in node_to_tips:
                return node_to_tips[node_id]
            if node.is_leaf:
                result = {node.name} if node.name else set()
            else:
                result = set()
                for child in node.children:
                    result.update(get_tip_names(child))
            node_to_tips[node_id] = result
            return result

        def find_node_by_name(start: PhyloNode, name: str) -> PhyloNode | None:
            if start.name == name:
                return start
            for child in start.children:
                result = find_node_by_name(child, name)
                if result:
                    return result
            return None

        def dominant_group_of(node: PhyloNode) -> str | None:
            """Return the most common group label among ``node``'s
            descendant tips, or ``None`` if no labelled tip is present."""
            tips = get_tip_names(node)
            counts: dict[str, int] = {}
            for tip_name in tips:
                grp = tips_with_groups.get(tip_name)
                if grp:
                    counts[grp] = counts.get(grp, 0) + 1
            if not counts:
                return None
            return max(counts, key=counts.get)

        get_tip_names(working_tree.root)

        # 单一统计量: 观测与置换必须用同一规则分类、同一公式计算 F,
        # 否则 p 值无效 (此前观测 F 用"两子树主导群不同"分类 +
        # 对比平方和, 置换 F 用"整子树主导群"标签 + 单因素 ANOVA F)。

        def _contrast_node(node_name: str) -> PhyloNode | None:
            target_node: PhyloNode | None = None
            if node_name.startswith("node_"):
                node_id_str = node_name.split("_")[-1]
                try:
                    node_id = int(node_id_str)

                    def search_by_id(n: PhyloNode, tid: int) -> PhyloNode | None:
                        if id(n) == tid:
                            return n
                        for ch in n.children:
                            r = search_by_id(ch, tid)
                            if r:
                                return r
                        return None

                    target_node = search_by_id(working_tree.root, node_id)
                except ValueError:
                    pass
            if target_node is None:
                target_node = find_node_by_name(working_tree.root, node_name)
            return target_node

        def _contrast_labels(ic_data: list[tuple[float, float, str]]) -> list[str]:
            """每条对比 → 其对应子树的主导群标签 (不可归类时归入 groups[0])。"""
            labels: list[str] = []
            for c_data in ic_data:
                target_node = _contrast_node(c_data[2])
                grp = dominant_group_of(target_node) if target_node is not None else None
                labels.append(grp if grp is not None else groups[0])
            return labels

        def _one_way_f(ic_values: list[float], labels: list[str]) -> tuple[float, float, float, float, float]:
            """经典单因素 ANOVA F。返回 (F, ss_b, ss_w, ms_b, ms_w)。"""
            ic = np.asarray(ic_values, dtype=np.float64)
            n_ic = len(ic)
            df_b = max(n_groups - 1, 1)
            df_w = max(n_ic - n_groups, 1)
            if n_ic == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            grand_mean = float(np.mean(ic))
            ss_b = 0.0
            ss_w = 0.0
            for grp in groups:
                vals = ic[np.asarray(labels) == grp]
                if len(vals) == 0:
                    continue
                gm_grp = float(np.mean(vals))
                ss_b += len(vals) * (gm_grp - grand_mean) ** 2
                ss_w += float(np.sum((vals - gm_grp) ** 2))
            ms_b = ss_b / df_b
            ms_w = ss_w / df_w
            F = ms_b / ms_w if ms_w > 0 else 0.0
            return F, ss_b, ss_w, ms_b, ms_w

        observed_labels = _contrast_labels(contrasts_data)
        F, ss_between, ss_within, ms_between, ms_within = _one_way_f(ic_arr.tolist(), observed_labels)

        # Permutation test: shuffle trait values across tips, keep groups fixed
        # This tests whether observed F is larger than chance given phylogeny and group structure.
        # Use a dedicated Generator when a seed is supplied so the
        # p-value is reproducible.
        perm_Fs: list[float] = []
        tip_names_list = [l.name for l in working_tree.root.get_leaves()]
        tip_array = np.array([trait_values[t] for t in tip_names_list], dtype=np.float64)
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random

        for _ in range(n_p):
            # Shuffle ONLY trait values, keep groups fixed
            perm_trait = rng.permutation(tip_array)
            perm_trait_dict = {tip_names_list[i]: perm_trait[i] for i in range(len(tip_names_list))}

            _, _, perm_ic_data, _ = _compute_contrasts_recursive(working_tree.root, perm_trait_dict)
            if len(perm_ic_data) < 2:
                continue

            # 与观测 F 完全相同的分类规则与 F 公式
            perm_labels = _contrast_labels(perm_ic_data)
            perm_ic = [c[0] for c in perm_ic_data]
            perm_F, _, _, _, _ = _one_way_f(perm_ic, perm_labels)
            perm_Fs.append(perm_F)

        p_value = float(np.mean(np.array(perm_Fs) >= F)) if perm_Fs else 1.0

        result = PhyloANOVAResult(
            f_statistic=F,
            p_value=p_value,
            n_permutations=n_p,
            groups=groups,
            n_groups=n_groups,
            n_tips=len(tip_names_list),
            ss_between=ss_between,
            ss_within=ss_within,
            ms_between=ms_between,
            ms_within=ms_within,
            contrast_values=ic_arr,
            group_labels=tips_with_groups,
        )
        self._logger.info(f"Phylo-ANOVA: F={F:.4f}, p={p_value:.4f}")
        return result

    # =========================================================================
    # Tree Pruning Utility
    # =========================================================================

    def _prune_tree(self, tree: PhyloTree, valid_names: set[str]) -> PhyloTree:
        """
        Prune tree to only include taxa in valid_names.

        Removes tips not in valid_names and their ancestors
        (if they become unary: have only one child).
        """
        if tree.root is None:
            return tree

        def should_keep(node: PhyloNode) -> bool:
            """Keep node if it's a leaf in valid_names or has keep-able descendants."""
            if node.is_leaf:
                return node.name in valid_names
            return any(should_keep(child) for child in node.children)

        def copy_subtree(node: PhyloNode) -> PhyloNode | None:
            """Recursively copy subtree, pruning unwanted branches."""
            if node.is_leaf:
                if node.name in valid_names:
                    return PhyloNode(
                        name=node.name,
                        node_type=NodeType.LEAF,
                        branch_length=node.branch_length,
                        data=node.data,
                    )
                return None

            kept_children: list[PhyloNode] = []
            for child in node.children:
                copied = copy_subtree(child)
                if copied is not None:
                    kept_children.append(copied)

            if not kept_children:
                return None

            new_node = PhyloNode(
                name=node.name,
                node_type=NodeType.INTERNAL,
                branch_length=node.branch_length,
                support=node.support,
                data=node.data,
            )
            for child in kept_children:
                new_node.add_child(child)

            # Collapse unary nodes
            if len(new_node.children) == 1 and not node.is_root:
                return new_node.children[0]

            return new_node

        new_root = copy_subtree(tree.root)
        if new_root is not None and new_root.is_leaf:
            new_root.node_type = NodeType.LEAF

        return PhyloTree(root=new_root, name=tree.name)


# NodeType is imported at the top
