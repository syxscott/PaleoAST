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

    # 初始化
    contrasts = []
    contrast_pairs = []

    def recurse(node):
        """
        递归计算 (重建值, 子树累积方差)。

        约定与 statistics/pcm.py 的单一参考实现一致 (Felsenstein 1985):
        - 叶节点返回 (trait, 0.0); 父节点把子节点枝长加到子树方差上;
        - 对比 IC = (x_A - x_B) / sqrt(v_A + v_B);
        - 向上传递的是逆方差加权重建值 (而非标准化对比——旧实现把
          对比值当性状值传递, 使上层对比与原始量纲混合, 结果错误);
        - 节点向上传递的方差 = v_A·v_B/(v_A+v_B) (加权重建值的方差);
        - polytomy: 迭代组合产生 k-1 个对比; unary: 透传不产生对比。
        """
        if node.is_leaf:
            val = traits.get(node.name)
            if val is None:
                logger.warning(f"Trait value not found for tip '{node.name}', using 0.0")
                val = 0.0
            val = float(val)
            node.data = PICNodeData(trait=val, variance=0.0)
            return val, 0.0

        child_results = []
        for child in node.children:
            val, cvar = recurse(child)
            child_results.append((val, cvar + (child.branch_length or 0.0), child.name))

        if len(child_results) == 1:
            # Unary 退化: 透传, 不产生对比 (旧实现 contrast/sqrt(2) 无依据)
            val, var, _ = child_results[0]
            node.data = PICNodeData(variance=var, trait=val)
            return val, var

        def _combine(res0, res1):
            val0, var0, name0 = res0
            val1, var1, name1 = res1
            var_sum = var0 + var1
            if var_sum <= 0:
                var_sum = 1e-10
            contrast = (val0 - val1) / np.sqrt(var_sum)
            if var0 > 0 and var1 > 0:
                recon = (val0 / var0 + val1 / var1) / (1.0 / var0 + 1.0 / var1)
            else:
                recon = (val0 + val1) / 2.0
            pooled = var0 * var1 / var_sum
            contrasts.append(float(contrast))
            contrast_pairs.append((name0, name1))
            return recon, pooled

        if len(child_results) == 2:
            recon, pooled = _combine(child_results[0], child_results[1])
            node.data = PICNodeData(variance=pooled, contrast=contrasts[-1], trait=recon)
            return recon, pooled

        # Polytomy (k > 2): 迭代组合, 产生 k-1 个独立对比
        active = list(child_results)
        combined_idx = 0
        while len(active) > 1:
            recon, pooled = _combine(active[0], active[1])
            combined_idx += 1
            active = [(recon, pooled, f"_combined_{combined_idx}"), *active[2:]]

        node.data = PICNodeData(
            variance=active[0][1],
            contrast=contrasts[-1] if contrasts else 0.0,
            trait=active[0][0],
        )
        return active[0][0], active[0][1]

    recurse(root)

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
