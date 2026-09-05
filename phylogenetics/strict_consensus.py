"""
================================================================================
PaleoAST Phylogenetics - Strict Consensus Tree
================================================================================

本模块实现严格一致性树算法。

数学理论:
==============================================================================

1. 一致性树定义
--------------------
给定一组树 T = {T1, T2, ..., Tk}，
一致性树保留所有树中一致的拓扑结构。

2. 严格一致性
--------------------
严格一致性树只包含在所有输入树中都出现的分支。

数学定义:
    对于分支 b，设 S(b) = {Ti | b ∈ Ti}
    则 b 在严格一致性树中 iff |S(b)| = k (所有树)

3. 多数规则一致性
--------------------
多数规则一致性树包含在 >50% 树中出现的分支。

4. Adams一致性
--------------------
Adams一致性对严格一致性进行后处理，
将不兼容的分支合并到最近的公共祖先。

5. 分割兼容性
--------------------
两个分割 A|B 和 C|D 兼容 iff:
    A ∩ C = ∅ OR A ∩ D = ∅ OR B ∩ C = ∅ OR B ∩ D = ∅

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from .tree import NodeType, PhyloNode, PhyloTree

logger = logging.getLogger(__name__)


@dataclass
class Split:
    """
    树的分割

    分割将叶节点集合划分为两部分。

    数学表示:
        分支 b 定义分割 A|B
        其中 A 是分支一侧的叶节点集合
              B 是另一侧的叶节点集合

    属性:
        set1: 第一组叶节点 (frozenset)
        set2: 第二组叶节点 (frozenset)
        is_trivial: 是否为平凡分割 (单叶)
        frequency: 在输入树中出现的频率
    """

    set1: frozenset[str]
    set2: frozenset[str]
    frequency: float = 1.0

    def __post_init__(self):
        """确保set1 < set2 (保持唯一性)"""
        # Use min element comparison for canonical ordering
        # (frozenset > means superset, not ordering)
        if self.set1 and self.set2 and min(self.set1) > min(self.set2):
            self.set1, self.set2 = self.set2, self.set1

    @property
    def is_trivial(self) -> bool:
        """检查是否为平凡分割 (单元素)"""
        return len(self.set1) == 1 or len(self.set2) == 1

    @property
    def all_taxa(self) -> frozenset[str]:
        """获取所有分类单元"""
        return self.set1 | self.set2

    def is_compatible_with(self, other: Split) -> bool:
        """
        检查与另一个分割是否兼容

        两个分割 A1|B1 和 A2|B2 兼容 iff:
            A1 ∩ A2 = ∅ OR
            A1 ∩ B2 = ∅ OR
            B1 ∩ A2 = ∅ OR
            B1 ∩ B2 = ∅

        Parameters:
            other: 另一个分割

        Returns:
            是否兼容
        """
        # 检查所有四种交集
        intersections = [self.set1 & other.set1, self.set1 & other.set2, self.set2 & other.set1, self.set2 & other.set2]

        # 如果有交集为空，则兼容
        return any(not inter for inter in intersections)

    def __hash__(self) -> int:
        return hash((self.set1, self.set2))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Split):
            return NotImplemented
        return self.set1 == other.set1 and self.set2 == other.set2

    def __repr__(self) -> str:
        return f"Split({self.set1} | {self.set2}, freq={self.frequency})"


class StrictConsensusTree:
    """
    严格一致性树构建器

    从多棵等长最优树构建严格一致性树。

    算法步骤:
        1. 从每棵输入树提取所有分割
        2. 统计每个分割的频率
        3. 选择频率=1.0的分割 (严格一致性)
        4. 构建一致性树

    示例:
        >>> consensus = StrictConsensusTree()
        >>> tree1 = PhyloTree.from_newick("((A,B),C,D);")
        >>> tree2 = PhyloTree.from_newick("((A,C),B,D);")
        >>> tree3 = PhyloTree.from_newick("((A,D),B,C);")
        >>> consensus_tree = consensus.build([tree1, tree2, tree3])
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.StrictConsensus")

    def build(self, trees: list[PhyloTree]) -> PhyloTree:
        """
        构建严格一致性树

        Parameters:
            trees: 输入树列表

        Returns:
            一致性树
        """
        if not trees:
            raise ValueError("No input trees provided")

        if len(trees) == 1:
            return trees[0]

        # 获取所有分类单元
        all_taxa = set(trees[0].leaf_names)
        for tree in trees[1:]:
            all_taxa |= set(tree.leaf_names)

        # 统计频率: 以"包含该分割的树数 / 总树数"计。
        # 旧实现用出现总次数 / 树数——同一棵树内多条边可归一化为同一
        # 分割 (如根的子节点 (A,B) 与 (C,D) 各给出一条 AB|CD), 出现
        # 次数超过树数 (3/2=1.5), 严格分割因此被整体丢弃。
        tree_counts: Counter = Counter()
        for tree in trees:
            if tree.root is None:
                continue
            for s in set(self._extract_all_splits([tree])):
                tree_counts[s] += 1
        n_trees = len(trees)

        splits_with_freq: list[Split] = []
        for split_set, count in tree_counts.items():
            freq = count / n_trees
            splits_with_freq.append(Split(set1=split_set.set1, set2=split_set.set2, frequency=freq))

        # 筛选严格一致性分割 (频率=1.0)
        strict_splits = [s for s in splits_with_freq if abs(s.frequency - 1.0) < 1e-10]

        self._logger.info(f"Extracted {len(splits_with_freq)} unique splits, {len(strict_splits)} are strict consensus")

        # 构建一致性树
        consensus_tree = self._build_tree_from_splits(strict_splits, all_taxa)

        return consensus_tree

    def build_majority_rule(self, trees: list[PhyloTree], threshold: float = 0.5) -> PhyloTree:
        """
        构建多数规则一致性树

        Parameters:
            trees: 输入树列表
            threshold: 支持率阈值 (默认0.5 = 50%)

        Returns:
            一致性树

        注意:
            当threshold < 1.0且输入树有冲突拓扑时，
            多数规则一致性可能产生不兼容的分割组合。
            此时会发出警告，且结果树可能退化为星形树。
        """
        if not trees:
            raise ValueError("No input trees provided")

        if len(trees) == 1:
            return trees[0]

        all_taxa = set(trees[0].leaf_names)
        for tree in trees[1:]:
            all_taxa |= set(tree.leaf_names)

        # 统计频率: 以"包含该分割的树数 / 总树数"计 (与 build() 相同,
        # 出现总次数会在同一树内被重复计数)
        tree_counts: Counter = Counter()
        for tree in trees:
            if tree.root is None:
                continue
            for s in set(self._extract_all_splits([tree])):
                tree_counts[s] += 1
        n_trees = len(trees)

        # 筛选超过阈值的分割
        majority_splits = []
        for split_set, count in tree_counts.items():
            freq = count / n_trees
            if freq >= threshold:
                split_obj = Split(set1=split_set.set1, set2=split_set.set2, frequency=freq)
                majority_splits.append(split_obj)

        # 检查分割兼容性
        if len(majority_splits) > 1:
            conflicts = []
            for i, split1 in enumerate(majority_splits):
                for split2 in majority_splits[i + 1 :]:
                    if not split1.is_compatible_with(split2):
                        conflicts.append((split1, split2))

            if conflicts:
                self._logger.warning(
                    f"Majority rule consensus: found {len(conflicts)} incompatible split pairs. "
                    f"The resulting tree may be unresolved (star tree) or invalid. "
                    f"Consider using a higher threshold or checking input tree compatibility."
                )

        return self._build_tree_from_splits(majority_splits, all_taxa)

    def _extract_all_splits(self, trees: list[PhyloTree]) -> list[Split]:
        """
        从所有树中提取分割

        Parameters:
            trees: 树列表

        Returns:
            分割列表
        """
        all_splits = []

        for tree in trees:
            if tree.root is None:
                continue

            splits = self._extract_splits_from_tree(tree)
            all_splits.extend(splits)

        return all_splits

    def _extract_splits_from_tree(self, tree: PhyloTree) -> list[Split]:
        """
        从单棵树提取所有分割

        对于每个内部边，其两侧的叶节点构成一个分割。
        对于多叉节点，每个子节点与其余叶节点都构成一个有效分割。

        Parameters:
            tree: 输入树

        Returns:
            分割列表
        """
        splits = []
        leaves = set(tree.leaf_names)

        if tree.root is None:
            return splits

        # 对每个内部节点
        for node in tree.root.preorder_traverse():
            if node.is_leaf or len(node.children) < 2:
                continue

            # 对于每个子节点，计算由该子节点定义的分割
            # (该子节点的叶节点 vs 所有其他叶节点)
            for child_idx, child in enumerate(node.children):
                child_leaves = set(c.name for c in child.get_leaves())
                set1 = frozenset(child_leaves)

                # 其他所有叶节点作为set2
                remaining_leaves = leaves - set1
                set2 = frozenset(remaining_leaves)

                if not set1 or not set2:
                    continue

                # 确保set1 < set2 (保持唯一性)
                if set1 > set2:
                    set1, set2 = set2, set1

                splits.append(Split(set1=set1, set2=set2))

        return splits

    def _build_tree_from_splits(self, splits: list[Split], all_taxa: set[str]) -> PhyloTree:
        """
        从分割列表构建树

        使用迭代方式:
            1. 选择最小分割 (最小集合)
            2. 创建一个内部节点
            3. 递归处理剩余分类单元

        Parameters:
            splits: 分割列表
            all_taxa: 所有分类单元

        Returns:
            构建的树
        """
        if not splits:
            # 没有分割，返回星形树
            return self._build_star_tree(all_taxa)

        if len(all_taxa) <= 2:
            # 基础情况
            taxa_list = list(all_taxa)
            if len(taxa_list) == 1:
                root = PhyloNode(name=taxa_list[0], node_type=NodeType.LEAF)
            else:
                root = PhyloNode(name="", node_type=NodeType.INTERNAL)
                root.add_child(PhyloNode(name=taxa_list[0], node_type=NodeType.LEAF))
                root.add_child(PhyloNode(name=taxa_list[1], node_type=NodeType.LEAF))
            return PhyloTree(root=root)

        # 找最小非平凡分割
        non_trivial = [s for s in splits if not s.is_trivial]

        if not non_trivial:
            return self._build_star_tree(all_taxa)

        return self._build_recursive(splits, all_taxa)

    def _build_recursive(self, splits: list[Split], taxa: set[str]) -> PhyloTree:
        """
        递归构建树

        适用性规则: 分割在当前 clade 内生效当且仅当 set1∩taxa 与
        set2∩taxa 均非空 (全局分割的另一侧是补集, 旧实现要求双侧都
        ⊆ taxa, 使嵌套分割永远找不到; 而对无分割的 2-taxa 子集构建
        "星形树"会生成输入中不存在的二叉 clade, 例如
        ((A,B),C,D) 与自身的严格一致树被错误输出为 ((A,B),(C,D)))。

        Parameters:
            splits: 分割列表
            taxa: 当前分类单元集合

        Returns:
            树
        """
        if len(taxa) <= 1:
            if not taxa:
                return PhyloTree()
            node = PhyloNode(name=next(iter(taxa)), node_type=NodeType.LEAF)
            return PhyloTree(root=node)

        applicable = []
        for split in splits:
            if split.is_trivial:
                continue
            side1 = split.set1 & taxa
            side2 = split.set2 & taxa
            if side1 and side2:
                applicable.append((side1, side2))

        if not applicable:
            # 无分割在 clade 内部生效: 按多分叉展开 (不制造分辨节点)
            return self._build_star_tree(taxa)

        # 选择在 taxa 内部分割得最均衡的分割, 以较小侧作为子 clade
        side1, side2 = min(applicable, key=lambda p: abs(len(p[0]) - len(p[1])))
        if len(side1) > len(side2):
            side1, side2 = side2, side1

        subtree = self._build_recursive(splits, side1)

        # 另一侧: 若仍有分割在其内部生效则递归, 否则作为叶直接挂载
        rest_partitioned = any(
            (s.set1 & side2) and (s.set2 & side2) for s in splits if not s.is_trivial
        )

        root = PhyloNode(name="", node_type=NodeType.INTERNAL)
        if subtree.root:
            root.add_child(subtree.root)
        if rest_partitioned:
            rest_tree = self._build_recursive(splits, side2)
            if rest_tree.root:
                root.add_child(rest_tree.root)
        else:
            for taxon in sorted(side2):
                root.add_child(PhyloNode(name=taxon, node_type=NodeType.LEAF))

        return PhyloTree(root=root)

    def _build_star_tree(self, taxa: set[str]) -> PhyloTree:
        """
        构建星形树

        所有叶节点直接连接到一个内部根节点。

        Parameters:
            taxa: 分类单元集合

        Returns:
            星形树
        """
        if not taxa:
            return PhyloTree()

        if len(taxa) == 1:
            root = PhyloNode(name=next(iter(taxa)), node_type=NodeType.LEAF)
            return PhyloTree(root=root)

        root = PhyloNode(name="consensus", node_type=NodeType.INTERNAL)

        for taxon in sorted(taxa):
            leaf = PhyloNode(name=taxon, node_type=NodeType.LEAF)
            root.add_child(leaf)

        return PhyloTree(root=root)


def build_strict_consensus(trees: list[PhyloTree]) -> PhyloTree:
    """
    构建严格一致性树的便捷函数

    Parameters:
        trees: 树列表

    Returns:
        一致性树
    """
    consensus = StrictConsensusTree()
    return consensus.build(trees)


def build_majority_rule_consensus(trees: list[PhyloTree], threshold: float = 0.5) -> PhyloTree:
    """
    构建多数规则一致性树的便捷函数

    Parameters:
        trees: 树列表
        threshold: 阈值

    Returns:
        一致性树
    """
    consensus = StrictConsensusTree()
    return consensus.build_majority_rule(trees, threshold)
