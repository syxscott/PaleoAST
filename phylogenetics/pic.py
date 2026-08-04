"""
================================================================================
PaleoAST Phylogenetics - Phylogenetic Independent Contrasts (PIC)
================================================================================

本模块实现 Felsenstein (1985) 的系统发育独立对比法 (Phylogenetic Independent
Contrasts, PIC)，是比较形态学和系统发育比较方法的核心算法。

算法原理 (Felsenstein 1985, American Naturalist):
================================================================================

对于二叉树，设节点 i 和 j 为姐妹节点 (sister nodes)，它们的性状值分别为 x_i 和
x_j，累积方差 (accumulated variance from root) 分别为 v_i 和 v_j。

独立对比 (Independent Contrast) 定义为:

    contrast_{i,j} = (x_i - x_j) / sqrt(v_i + v_j)

其中:
    - v_i = Σ (从根到节点 i 的所有枝长)
    - v_j = Σ (从根到节点 j 的所有枝长)

此对比值在 Brown运动 (Brownian Motion) 进化模型下是统计独立的。

多分支节点 (Polytomy) 处理 (Pagel 1992, Felsenstein 2008):
================================================================================

对于具有 k 个子节点的内部节点，产生 k-1 个独立对比。

迭代组合法 (Iterative Combination):
    1. 组合 child0 和 child1:
       contrast_01 = (x_0 - x_1) / sqrt(v_0 + v_1)
       v_01 = v_0 + v_1 + branch_length_node

    2. 组合结果与 child2:
       contrast_02 = (contrast_01 - x_2) / sqrt(v_01 + v_2)
       v_02 = v_01 + v_2 + branch_length_node

    3. 重复直到所有子节点组合，产生 k-1 个独立对比

单子节点退化 (Unary Node):
    对于只有一个子节点的退化情况，使用:
       contrast = child_contrast / sqrt(2)

参考文献:
----------
- Felsenstein, J. (1985). Phylogenies and the comparative method.
  American Naturalist, 125(1), 1-15.
- Pagel, M. (1992). A method for the analysis of comparative data.
  Journal of Theoretical Biology, 156(4), 431-442.
- Felsenstein, J. (2008). Inferring phylogenies (2nd ed.). Sinauer Associates.

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PICNodeData:
    """
    PIC 计算过程中存储在节点上的临时数据

    属性:
        variance: 从根到本节点的累积方差
        contrast: 本节点的独立对比值
        trait: 性状值 (用于叶节点或未计算的节点)
    """
    variance: float = 0.0
    contrast: float = 0.0
    trait: float = 0.0


def compute_pic(
    tree,
    traits: dict[str, float],
    root_variance: float = 0.0,
) -> tuple[list[float], list[tuple[str, str]]]:
    """
    计算系统发育独立对比 (Phylogenetic Independent Contrasts)

    参数:
        tree: PhyloTree 或 PhyloNode 对象，表示系统发育树
        traits: {tip_name: trait_value} 字典，性状数据
        root_variance: 根节点的累积方差 (默认为 0)

    返回:
        (contrasts, contrast_pairs):
            - contrasts: 独立对比值列表
            - contrast_pairs: 每个对比对应的节点对列表 [(node_name1, node_name2), ...]

    算法 (Felsenstein 1985):
        1. 后序遍历计算每个节点的累积方差 v_i
        2. 对每个内部节点计算独立对比:
           - 二叉节点: contrast = (x_i - x_j) / sqrt(v_i + v_j)
           - polytomy: 使用迭代组合法产生 k-1 个独立对比
           - unary 退化: contrast = child_contrast / sqrt(2)

    示例:
        >>> from phylogenetics import PhyloTree
        >>> tree = PhyloTree.from_newick("(A:1,B:1)C:1;")
        >>> traits = {"A": 2.0, "B": 4.0}
        >>> contrasts, pairs = compute_pic(tree, traits)
        >>> print(f"Contrast: {contrasts[0]:.4f}")
    """
    # 获取根节点
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    # 初始化: 为所有节点计算累积方差
    _compute_variances(root, root_variance)

    # 为叶节点设置性状值
    for node in root.preorder_traverse():
        if node.is_leaf:
            if node.name not in traits:
                logger.warning(f"Trait value not found for tip '{node.name}', using 0.0")
                node.data = PICNodeData(trait=0.0, variance=node.metadata.get('_variance', 0.0))
            else:
                node.data = PICNodeData(
                    trait=traits[node.name],
                    variance=node.metadata.get('_variance', 0.0)
                )

    # 后序遍历计算对比
    contrasts = []
    contrast_pairs = []

    def compute_node_contrast(node):
        """
        递归计算节点的对比值

        返回:
            (contrast_value, variance_at_node)
        """
        if node.is_leaf:
            pic_data = node.data
            return pic_data.trait, pic_data.variance

        # 递归计算所有子节点
        child_results = []
        for child in node.children:
            child_contrast, child_var = compute_node_contrast(child)
            branch_len = child.branch_length if child.branch_length is not None else 0.0
            child_results.append((child_contrast, child_var, branch_len, child.name))

        # 根据子节点数量计算对比
        k = len(child_results)

        if k == 1:
            # Unary 退化情况
            child_contrast, child_var, branch_len, child_name = child_results[0]
            node_var = child_var + branch_len
            # contrast = child_contrast / sqrt(2)
            contrast = child_contrast / np.sqrt(2)
            node.data = PICNodeData(variance=node_var, contrast=contrast)
            return contrast, node_var

        elif k == 2:
            # 标准二叉情况 (Felsenstein 1985)
            c0, v0, bl0, name0 = child_results[0]
            c1, v1, bl1, name1 = child_results[1]
            var_sum = v0 + v1
            if var_sum <= 0:
                var_sum = 1e-10  # 避免除零
            contrast = (c0 - c1) / np.sqrt(var_sum)
            # 节点方差 = 两个子节点方差之和 (v0/v1 已包含 bl0/bl1)
            # Felsenstein 1985: Var(recon_A - recon_B | P) = v_A + v_B
            node_var = v0 + v1
            node.data = PICNodeData(variance=node_var, contrast=contrast)
            contrasts.append(contrast)
            contrast_pairs.append((name0, name1))
            return contrast, node_var

        else:
            # Polytomy: k > 2，产生 k-1 个独立对比
            # 使用迭代组合法
            node_var = sum(v for _, v, _, _ in child_results)
            branch_sum = sum(bl for _, _, bl, _ in child_results)
            node_var += branch_sum

            # 第一个对比: child0 - child1
            c0, v0, bl0, name0 = child_results[0]
            c1, v1, bl1, name1 = child_results[1]
            var01 = v0 + v1
            if var01 <= 0:
                var01 = 1e-10
            contrast01 = (c0 - c1) / np.sqrt(var01)
            running_contrast = contrast01
            # v0/v1 已包含 bl0/bl1, 不需要重复加
            running_var = v0 + v1  # Felsenstein 1985

            first_pair = (name0, name1)
            contrasts.append(contrast01)
            contrast_pairs.append(first_pair)

            # 后续对比: 累积结果与下一个子节点
            for idx in range(2, k):
                c_i, v_i, bl_i, name_i = child_results[idx]
                new_var = running_var + v_i + bl_i
                if new_var <= 0:
                    new_var = 1e-10
                new_contrast = (running_contrast - c_i) / np.sqrt(new_var)
                contrasts.append(new_contrast)
                contrast_pairs.append((f"_combined_{idx-1}", name_i))
                running_contrast = new_contrast
                running_var = new_var  # 不再加 bl_i, 因为 new_var 已包含

            node.data = PICNodeData(variance=running_var, contrast=running_contrast)
            return running_contrast, running_var

    # 从根开始计算
    root_contrast, root_var = compute_node_contrast(root)

    logger.info(f"PIC computation complete: {len(contrasts)} contrasts computed")

    return contrasts, contrast_pairs


def _compute_variances(node, parent_variance: float) -> None:
    """
    后序遍历计算每个节点从根到该节点的累积方差

    参数:
        node: 当前节点
        parent_variance: 父节点的累积方差
    """
    # 后序遍历: 先处理子节点
    for child in node.children:
        _compute_variances(child, parent_variance)

    # 计算当前节点的方差
    branch_len = node.branch_length if node.branch_length is not None else 0.0
    node_variance = parent_variance + branch_len

    # 存储在 metadata 中供后续使用
    node.metadata['_variance'] = node_variance
    node.metadata['_parent_variance'] = parent_variance


def compute_pic_with_ancestral_states(
    tree,
    traits: dict[str, float],
    root_variance: float = 0.0,
) -> tuple[list[float], list[tuple[str, str]], dict[str, float]]:
    """
    计算 PIC 并返回所有内部节点的祖先状态估计

    参数:
        tree: PhyloTree 或 PhyloNode 对象
        traits: {tip_name: trait_value} 字典
        root_variance: 根节点的累积方差

    返回:
        (contrasts, contrast_pairs, ancestral_states):
            - contrasts: 独立对比值列表
            - contrast_pairs: 每个对比对应的节点对
            - ancestral_states: {node_name: estimated_trait_value} 字典

    注意:
        祖先状态是基于对比计算的节点性状估计值。
        对于二叉节点，祖先状态 = (child0_trait + child1_trait) / 2
        对于 polytomy，使用加权平均。
    """
    contrasts, pairs = compute_pic(tree, traits, root_variance)

    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    ancestral_states = {}

    def compute_ancestral(node):
        """递归计算祖先状态"""
        if node.is_leaf:
            return node.data.trait if node.data else traits.get(node.name, 0.0)

        child_traits = [compute_ancestral(child) for child in node.children]
        ancestral = np.mean(child_traits)  # 简单平均
        ancestral_states[node.name if node.name else '_internal_'] = ancestral
        return ancestral

    compute_ancestral(root)
    return contrasts, pairs, ancestral_states


def validate_pic_assumptions(tree) -> dict[str, Any]:
    """
    检验 PIC 所需假设是否满足

    参数:
        tree: PhyloTree 或 PhyloNode 对象

    返回:
        validation_results: {
            'is_rooted': bool,
            'has_branch_lengths': bool,
            'polytomy_count': int,
            'leaf_count': int,
            'warnings': list[str]
        }

    PIC 假设:
        1. 树是有根的 (rooted tree)
        2. 所有枝长非负
        3. 进化遵循 Brown 运动模型
    """
    if hasattr(tree, 'root'):
        root = tree.root
    else:
        root = tree

    warnings = []
    polytomy_count = 0
    leaf_count = 0
    has_branch_lengths = True

    for node in root.preorder_traverse():
        if node.is_leaf:
            leaf_count += 1
        else:
            k = len(node.children)
            if k > 2:
                polytomy_count += 1
            if k == 1:
                warnings.append(f"Unary node detected: '{node.name}'")

        if node.branch_length is None:
            has_branch_lengths = False
        elif node.branch_length < 0:
            warnings.append(f"Negative branch length at node: '{node.name}'")

    is_rooted = root.parent is None

    results = {
        'is_rooted': is_rooted,
        'has_branch_lengths': has_branch_lengths,
        'polytomy_count': polytomy_count,
        'leaf_count': leaf_count,
        'warnings': warnings,
        'assumptions_satisfied': is_rooted and has_branch_lengths and polytomy_count == 0
    }

    if polytomy_count > 0:
        # Polytomy violates standard PIC binary-tree assumptions
        results['assumptions_satisfied'] = False
        warnings.append(
            f"Tree has {polytomy_count} polytomy(ies). "
            "Using Pagel (1992) iterative combination method."
        )

    return results
