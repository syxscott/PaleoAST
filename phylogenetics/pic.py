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
        root_variance: 根节点的累积方差。仅为 API 兼容保留：Felsenstein (1985)
            的标准化对比 IC=(x_i-x_j)/sqrt(v_i+v_j) 中 v 为"到根"的累积枝长和，
            给根附加常数方差会在分子分母同步缩放，不改变任何对比值，因此本实现
            不使用它。传负值报错。

    返回:
        (contrasts, contrast_pairs):
            - contrasts: 独立对比值列表 (退化对为 NaN)
            - contrast_pairs: 每个对比对应的节点对列表 [(node_name1, node_name2), ...]

    缺失与退化处理:
        - 树中存在 traits 未覆盖的终端分类单元时，这些叶被从本次分析的**工作
          拷贝**中剔除 (不再静默按 0.0 代入)，并记录 warning；剔除的叶名同时
          写入 ``tree.metadata['pic_missing_tips']`` (若 tree 为 PhyloTree)。
        - 方差和为 0 (退化，如两侧枝长全为 0) 时对比值无定义：写入 NaN 并把
          该对记录到 ``tree.metadata['pic_degenerate_pairs']``，而不是用 1e-10
          把噪声放大约 1e5 倍。

    算法 (Felsenstein 1985):
        1. 后序遍历计算每个节点的累积方差 v_i
        2. 对每个内部节点计算独立对比:
           - 二叉节点: contrast = (x_i - x_j) / sqrt(v_i + v_j)
           - polytomy: 使用迭代组合法产生 k-1 个独立对比
           - unary 退化: 透传重建值，不产生对比

    示例:
        >>> from phylogenetics import PhyloTree
        >>> tree = PhyloTree.from_newick("(A:1,B:1)C:1;")
        >>> traits = {"A": 2.0, "B": 4.0}
        >>> contrasts, pairs = compute_pic(tree, traits)
        >>> print(f"Contrast: {contrasts[0]:.4f}")
    """
    root, dropped = _resolve_pic_root(tree, traits, root_variance)

    if root is None:
        logger.warning("PIC: no tips with trait data remain after excluding missing values")
        contrasts, pairs = [], []
    else:
        contrasts, pairs, _ = _pic_core(root, traits)

    _record_pic_diagnostics(tree, root, dropped, pairs, contrasts)
    return contrasts, pairs


def _resolve_pic_root(tree, traits: dict[str, float], root_variance: float = 0.0):
    """校验输入并把缺性状的叶剔除，返回 (工作根节点, 被剔除的叶名)。"""
    if hasattr(tree, "root"):
        root = tree.root
    else:
        root = tree

    if root is None:
        raise ValueError("Cannot compute PIC: tree has no root node (empty tree)")

    if not traits:
        raise ValueError("Cannot compute PIC: no trait values supplied")

    if root_variance is not None and root_variance < 0:
        raise ValueError(f"root_variance must be non-negative, got {root_variance}")

    leaves = root.get_leaves()
    if not leaves:
        raise ValueError("Cannot compute PIC: tree has no terminal taxa")

    missing = sorted(
        {leaf.name for leaf in leaves if traits.get(leaf.name) is None},
        key=str,
    )
    if missing:
        logger.warning(
            "Trait value(s) not found for %d tip(s): %s. These tips are excluded from the PIC "
            "analysis instead of being imputed with 0.0.",
            len(missing),
            ", ".join(str(name) for name in missing[:10]),
        )
        working = root._copy_subtree()
        _drop_missing_tips(working, set(missing))
        if not working.get_leaves():
            return None, missing
        return working, missing

    return root, []


def _drop_missing_tips(node, missing: set[str]) -> None:
    """就地从 (拷贝的) 树中移除性状缺失的叶。"""
    for child in list(node.children):
        if child.is_leaf:
            if child.name in missing:
                node.remove_child(child)
        else:
            _drop_missing_tips(child, missing)


def _record_pic_diagnostics(tree, root, dropped, pairs, contrasts) -> None:
    """把剔除/退化信息记录到日志与 tree.metadata (若可用)。"""
    degenerate = [pair for pair, value in zip(pairs, contrasts, strict=False) if not np.isfinite(value)]
    if degenerate:
        logger.warning(
            "PIC: %d contrast pair(s) have zero accumulated variance and are reported as NaN: %s",
            len(degenerate),
            degenerate[:5],
        )
    if not hasattr(tree, "metadata"):
        return
    if dropped:
        tree.metadata["pic_missing_tips"] = list(dropped)
    if degenerate:
        tree.metadata["pic_degenerate_pairs"] = list(degenerate)


def _pic_core(root, traits: dict[str, float]) -> tuple[list[float], list[tuple[str, str]], dict[str, Any]]:
    """
    PIC 核心递归。

    Returns:
        (contrasts, contrast_pairs, info); info 含 ``degenerate_pairs``。
    """
    contrasts: list[float] = []
    contrast_pairs: list[tuple[str, str]] = []
    degenerate_pairs: list[tuple[str, str]] = []
    # 多分叉迭代组合的伪节点名必须全局唯一，否则不同节点产生的
    # "_combined_1" 在 contrast_pairs 中互相歧义。
    combined_counter = 0

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
        nonlocal combined_counter

        if node.is_leaf:
            val = traits.get(node.name)
            if val is None:
                # 缺失叶已在入口处剔除；这里只可能是调用方直接传入的孤立节点。
                raise ValueError(
                    f"Trait value not found for tip '{node.name}'; supply trait data for every tip "
                    "or prune the tip before calling compute_pic()."
                )
            val = float(val)
            node.data = PICNodeData(trait=val, variance=0.0)
            return val, 0.0

        child_results = []
        for child in node.children:
            val, cvar = recurse(child)
            child_results.append((val, cvar + (child.branch_length or 0.0), child.name))

        if len(child_results) == 0:
            node.data = PICNodeData(variance=0.0, trait=0.0)
            return 0.0, 0.0

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
                # 零方差对: 对比无定义。旧实现把 var_sum 夹到 1e-10，等价于
                # 把 (x0 - x1) 放大 1e5 倍，制造出巨大的伪对比。此处记为 NaN
                # 并由调用方/下游统计流程跳过。
                contrast = float("nan")
                degenerate_pairs.append((name0, name1))
            else:
                contrast = float((val0 - val1) / np.sqrt(var_sum))
            if var0 > 0 and var1 > 0:
                recon = (val0 / var0 + val1 / var1) / (1.0 / var0 + 1.0 / var1)
                pooled = var0 * var1 / var_sum
            else:
                recon = (val0 + val1) / 2.0
                pooled = max(var0, var1)
            contrasts.append(contrast)
            contrast_pairs.append((name0, name1))
            return recon, pooled

        if len(child_results) == 2:
            recon, pooled = _combine(child_results[0], child_results[1])
            node.data = PICNodeData(variance=pooled, contrast=contrasts[-1], trait=recon)
            return recon, pooled

        # Polytomy (k > 2): 迭代组合, 产生 k-1 个独立对比
        active = list(child_results)
        while len(active) > 1:
            recon, pooled = _combine(active[0], active[1])
            combined_counter += 1
            active = [(recon, pooled, f"_combined_{combined_counter}"), *active[2:]]

        node.data = PICNodeData(
            variance=active[0][1],
            contrast=contrasts[-1] if contrasts else 0.0,
            trait=active[0][0],
        )
        return active[0][0], active[0][1]

    recurse(root)

    logger.info(f"PIC computation complete: {len(contrasts)} contrasts computed")

    return contrasts, contrast_pairs, {"degenerate_pairs": degenerate_pairs}


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
        root_variance: 根节点的累积方差 (仅为 API 兼容保留，见 compute_pic)

    返回:
        (contrasts, contrast_pairs, ancestral_states):
            - contrasts: 独立对比值列表 (退化对为 NaN)
            - contrast_pairs: 每个对比对应的节点对
            - ancestral_states: {node_key: estimated_trait_value} 字典

    注意:
        - 祖先状态取自与 contrasts **同一次** `_pic_core` 递归写入的
          ``node.data.trait``，即 Brown 运动下的逆方差加权 (极大似然) 重建值；
          两支方差相等时退化为简单平均。旧实现在此独立地做了一次
          ``np.mean(child_traits)`` 简单平均，既与对比计算脱节，又依赖
          ``node.data`` 是否残留。
        - **无名内部节点不再共用 '_internal_' 键** (旧实现会让多个无名节点互相
          覆盖，只剩最后一个)；改用 ``_internal_<序号>`` 的唯一键。
        - 缺失性状的叶按 compute_pic 的规则剔除，因此这些叶所在的祖先节点
          基于剩余数据估计；被剔除的叶不会出现在结果里。
    """
    root, dropped = _resolve_pic_root(tree, traits, root_variance)

    if root is None:
        logger.warning("PIC: no tips with trait data remain after excluding missing values")
        contrasts, pairs, ancestral_states = [], [], {}
    else:
        contrasts, pairs, _ = _pic_core(root, traits)
        ancestral_states = {}
        internal_idx = 0
        for node in root.preorder_traverse():
            if node.is_leaf:
                continue
            if node.name:
                key = node.name
            else:
                internal_idx += 1
                key = f"_internal_{internal_idx}"
            trait_value = node.data.trait if isinstance(node.data, PICNodeData) else float("nan")
            ancestral_states[key] = float(trait_value)

    _record_pic_diagnostics(tree, root, dropped, pairs, contrasts)
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
