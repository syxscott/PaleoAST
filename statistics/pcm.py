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
    Compute the phylogenetic variance-covariance matrix.

    V_ij = t_ij = total branch length from root to common ancestor of i and j
          = dist(root, i) + dist(root, j) - 2 * dist(root, LCA(i,j))

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
        for j, leaf_j in enumerate(leaves):
            lca = root.compute_lca(leaf_i, leaf_j)
            t_ij = dist_to_root[leaf_i] + dist_to_root[leaf_j] - 2 * dist_to_root[lca]
            VCV[i, j] = t_ij

    return name_to_idx, VCV


def _compute_contrasts_recursive(
    node: PhyloNode,
    trait_values: dict[str, float],
) -> tuple[float | None, float, list[tuple[float, float, str]], list[str]]:
    """
    Recursively compute independent contrasts.

    Post-order traversal: process children first, then compute contrast at parent.

    Returns:
        (reconstructed_value, cum_variance, contrasts_list, node_names)

    Where cum_variance = sum of branch lengths from this node to ALL descendant tips.
    Contrasts list entries are (raw_contrast, se, node_name).
    """
    if node.is_leaf:
        leaf_name = node.name
        val = trait_values.get(leaf_name)
        cum_var = node.branch_length or 0.0
        return val, cum_var, [], [] if val is None else [leaf_name]

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
        _child1, val1, var1 = child_results[0]
        _child2, val2, var2 = child_results[1]
        # v1 = cumulative var from child subtree = var1 (already includes branch to this node)
        # No need to add child1.branch_length - var1 already represents the cumulative
        # variance from child1's subtree to all descendant tips
        v1 = var1
        v2 = var2
        # Standardized contrast IC = (x_A - x_B) / sqrt(v_A + v_B)
        contrast = (val1 - val2) / math.sqrt(v1 + v2)
        se = math.sqrt(v1 + v2)
        node_name = node.name or f"node_{id(node)}"
        all_contrasts.append((contrast, se, node_name))
        all_names.append(node_name)
        # cum_variance from this node = sum of v1 and v2 (all tips in both subtrees)
        cum_var = v1 + v2
        # Reconstructed value: inverse-variance weighted mean
        if v1 > 0 and v2 > 0:
            recon = (val1 / v1 + val2 / v2) / (1.0 / v1 + 1.0 / v2)
        else:
            recon = (val1 + val2) / 2.0
        return recon, cum_var, all_contrasts, all_names

    # Multi-way node: reduce iteratively (Felsenstein 1985)
    # Iteratively contrast adjacent children, combining one each step
    # Build list of (value, cum_var, child) for active nodes
    active: list[tuple[float, float, PhyloNode]] = [(val, cvar, child) for child, val, cvar in child_results]

    while len(active) > 1:
        # Contrast first two (Felsenstein 1985)
        val1, var1, _ch1 = active[0]
        val2, var2, _ch2 = active[1]
        # var1/var2 already include branches to current node, no need to add again
        v1 = var1
        v2 = var2
        # Standardized contrast
        contrast = (val1 - val2) / math.sqrt(v1 + v2)
        se = math.sqrt(v1 + v2)
        node_name = f"{node.name}_c{len(active) - 2}" if node.name else f"node_{id(node)}_c{len(active) - 2}"
        all_contrasts.append((contrast, se, node_name))
        all_names.append(node_name)
        # Combined subtree: weighted mean, combined cumulative variance
        combined_cvar = v1 + v2
        if v1 > 0 and v2 > 0:
            combined_val = (val1 / v1 + val2 / v2) / (1.0 / v1 + 1.0 / v2)
        else:
            combined_val = (val1 + val2) / 2.0
        # Replace first two with combined
        combined_child = PhyloNode(name="combined", node_type=NodeType.INTERNAL, branch_length=0.0)
        active = [(combined_val, combined_cvar, combined_child), *active[2:]]

    # Reconstructed value at this node = inverse-variance weighted mean of immediate child values
    total_w = 0.0
    weighted_sum = 0.0
    for _, val, cvar in child_results:
        w = 1.0 / max(cvar, 0.0001)
        weighted_sum += w * val
        total_w += w
    recon = weighted_sum / total_w if total_w > 0 else (active[0][0] if active else 0.0)
    # cum_variance from this node = sum of all child cum_vars
    cum_var = sum(cvar for _, cvar, _ in child_results)
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

        def assign_states(node: PhyloNode) -> float | None:
            """Post-order: assign leaf values first, compute internal."""
            if node.is_leaf:
                return trait_values.get(node.name)

            child_vals: list[tuple[PhyloNode, float]] = []
            for child in node.children:
                val = assign_states(child)
                if val is not None:
                    child_vals.append((child, val))

            if not child_vals:
                return None

            if len(child_vals) == 1:
                return child_vals[0][1]

            # Weighted average (inverse variance weighting)
            total_w = 0.0
            weighted_sum = 0.0
            for child, val in child_vals:
                w = 1.0 / max(child.branch_length or 0.001, 0.0001)
                weighted_sum += w * val
                total_w += w

            recon = weighted_sum / total_w if total_w > 0 else np.mean([v for _, v in child_vals])

            node_name = node.name or f"node_{id(node)}"
            node_states[node_name] = recon
            internal_names.append(node_name)
            return recon

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

        Returns:
            PhylogeneticSignalResult with K, Z-score, and p-value
        """
        n_r = n_randomizations or self._n_randomizations
        self._logger.info(f"Computing Blomberg's K (n_perm={n_r})")

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

        # VCV inverse under BM gives weights for ancestral reconstruction
        # For K: compute contrasts and their expected variances
        _recon, _cumvar, contrasts_data, _unused = _compute_contrasts_recursive(working_tree.root, trait_values)

        ic_vals = np.array([c[0] for c in contrasts_data], dtype=np.float64)
        branch_sums = np.array([c[1] for c in contrasts_data], dtype=np.float64)

        # Blomberg's K: K = mean((IC / sqrt(v))^2)
        # = (sum IC_i^2 / n) / (sum v_i / n)
        # Standardized contrasts: IC / sqrt(v)
        std_contrasts = ic_vals / np.sqrt(branch_sums)
        K = np.mean(std_contrasts**2)

        # Compute Z-score via permutations
        tip_array = np.array([trait_values[l.name] for l in leaves], dtype=np.float64)
        perm_Ks: list[float] = []

        for _ in range(n_r):
            perm_trait = np.random.permutation(tip_array)
            perm_dict = {name: perm_trait[i] for i, name in enumerate(tip_names)}
            _, _, perm_ic_data, _ = _compute_contrasts_recursive(working_tree.root, perm_dict)
            if perm_ic_data:
                perm_ic = np.array([c[0] for c in perm_ic_data], dtype=np.float64)
                perm_branch = np.array([c[1] for c in perm_ic_data], dtype=np.float64)
                perm_std = perm_ic / np.sqrt(perm_branch)
                perm_Ks.append(np.mean(perm_std**2))

        perm_Ks_arr = np.array(perm_Ks)
        z = (K - np.mean(perm_Ks_arr)) / np.std(perm_Ks_arr) if np.std(perm_Ks_arr) > 0 else 0.0
        p_value = float(np.mean(perm_Ks_arr >= K))

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

        Returns:
            PhyloANOVAResult with F-statistic, p-value, and ANOVA table
        """
        n_p = n_permutations or self._n_randomizations
        self._logger.info(f"Phylogenetic ANOVA (n_perm={n_p})")

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

        # Assign contrasts to groups based on which group's tips dominate the node's subtree
        # Precompute tip sets for each internal node
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

        get_tip_names(working_tree.root)

        contrast_groups: list[str] = []
        for c_data in contrasts_data:
            node_name = c_data[2]
            # Find the node in the tree
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
            if target_node is None:
                contrast_groups.append(groups[0])
                continue
            # Count tips per group in this node's subtree
            all_desc_tips = get_tip_names(target_node)
            group_counts: dict[str, int] = {}
            for tip_name in all_desc_tips:
                grp = tips_with_groups.get(tip_name)
                if grp:
                    group_counts[grp] = group_counts.get(grp, 0) + 1
            if not group_counts:
                contrast_groups.append(groups[0])
                continue
            dominant_group = max(group_counts, key=group_counts.get)
            contrast_groups.append(dominant_group)

        # Compute F statistic using contrasts
        group_ic_means: dict[str, float] = {}
        for grp in groups:
            grp_ics = [ic_arr[i] for i, g in enumerate(contrast_groups) if g == grp]
            group_ic_means[grp] = np.mean(grp_ics) if grp_ics else 0.0

        grand_mean = np.mean(ic_arr)
        ss_between = 0.0
        for grp in groups:
            grp_ics = [ic_arr[i] for i, g in enumerate(contrast_groups) if g == grp]
            n_grp = len(grp_ics)
            ss_between += n_grp * (group_ic_means[grp] - grand_mean) ** 2

        ss_within = sum((ic_arr[i] - group_ic_means[g]) ** 2 for i, g in enumerate(contrast_groups))
        df_between = n_groups - 1
        df_within = len(ic_arr) - n_groups
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        ms_within = ss_within / df_within if df_within > 0 else 0.0
        F = ms_between / ms_within if ms_within > 0 else 0.0

        # Permutation test: shuffle trait values across tips, keep groups fixed
        # This tests whether observed F is larger than chance given phylogeny and group structure
        perm_Fs: list[float] = []
        tip_names_list = [l.name for l in working_tree.root.get_leaves()]
        tip_array = np.array([trait_values[t] for t in tip_names_list], dtype=np.float64)

        for _ in range(n_p):
            # Shuffle ONLY trait values, keep groups fixed
            perm_trait = np.random.permutation(tip_array)
            perm_trait_dict = {tip_names_list[i]: perm_trait[i] for i in range(len(tip_names_list))}

            _, _, perm_ic_data, _ = _compute_contrasts_recursive(working_tree.root, perm_trait_dict)
            if len(perm_ic_data) < 2:
                continue

            perm_ic = np.array([c[0] for c in perm_ic_data], dtype=np.float64)

            # Assign contrasts to groups using the same tree-based rule (groups are fixed)
            perm_contrast_groups: list[str] = []
            for c_data in perm_ic_data:
                node_name = c_data[2]
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
                if target_node is None:
                    perm_contrast_groups.append(groups[0])
                    continue
                all_desc_tips = get_tip_names(target_node)
                group_counts: dict[str, int] = {}
                for tip_name in all_desc_tips:
                    grp = tips_with_groups.get(tip_name)
                    if grp:
                        group_counts[grp] = group_counts.get(grp, 0) + 1
                if not group_counts:
                    perm_contrast_groups.append(groups[0])
                    continue
                dominant_group = max(group_counts, key=group_counts.get)
                perm_contrast_groups.append(dominant_group)

            # Compute F for permuted data
            perm_grp_means: dict[str, float] = {}
            for grp in groups:
                grp_ics = [perm_ic[i] for i, g in enumerate(perm_contrast_groups) if g == grp]
                perm_grp_means[grp] = np.mean(grp_ics) if grp_ics else 0.0

            perm_gm = np.mean(perm_ic)
            perm_ss_between = sum(
                sum(1 for i, g in enumerate(perm_contrast_groups) if g == grp) * (perm_grp_means[grp] - perm_gm) ** 2
                for grp in groups
            )
            perm_ss_within = sum((perm_ic[i] - perm_grp_means[g]) ** 2 for i, g in enumerate(perm_contrast_groups))
            perm_ms_between = perm_ss_between / df_between if df_between > 0 else 0.0
            perm_ms_within = perm_ss_within / df_within if df_within > 0 else 0.0
            perm_Fs.append(perm_ms_between / perm_ms_within if perm_ms_within > 0 else 0.0)

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
