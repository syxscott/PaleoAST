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
from typing import (
    Dict, List, Optional, Set, Tuple, Any, FrozenSet
)
from dataclasses import dataclass, field
import logging
from collections import Counter

from .tree import PhyloTree, PhyloNode, NodeType

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
    set1: FrozenSet[str]
    set2: FrozenSet[str]
    frequency: float = 1.0
    
    def __post_init__(self):
        """确保set1 < set2 (保持唯一性)"""
        # Use min element comparison for canonical ordering
        # (frozenset > means superset, not ordering)
        if self.set1 and self.set2:
            if min(self.set1) > min(self.set2):
                self.set1, self.set2 = self.set2, self.set1
    
    @property
    def is_trivial(self) -> bool:
        """检查是否为平凡分割 (单元素)"""
        return len(self.set1) == 1 or len(self.set2) == 1
    
    @property
    def all_taxa(self) -> FrozenSet[str]:
        """获取所有分类单元"""
        return self.set1 | self.set2
    
    def is_compatible_with(self, other: 'Split') -> bool:
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
        intersections = [
            self.set1 & other.set1,
            self.set1 & other.set2,
            self.set2 & other.set1,
            self.set2 & other.set2
        ]
        
        # 如果有交集为空，则兼容
        for inter in intersections:
            if not inter:
                return True
        
        return False
    
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
    
    def build(self, trees: List[PhyloTree]) -> PhyloTree:
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
        
        # 提取所有分割
        all_splits = self._extract_all_splits(trees)
        
        # 统计频率
        split_counts = Counter(all_splits)
        n_trees = len(trees)
        
        # 构建分割集合及其频率
        splits_with_freq: List[Split] = []
        for split_set in all_splits:
            freq = split_counts[split_set] / n_trees
            split_obj = Split(
                set1=split_set.set1,
                set2=split_set.set2,
                frequency=freq
            )
            splits_with_freq.append(split_obj)
        
        # 去重
        unique_splits = []
        seen = set()
        for s in splits_with_freq:
            key = (s.set1, s.set2)
            if key not in seen:
                seen.add(key)
                unique_splits.append(s)
        
        # 筛选严格一致性分割 (频率=1.0)
        strict_splits = [s for s in unique_splits if abs(s.frequency - 1.0) < 1e-10]
        
        self._logger.info(
            f"Extracted {len(unique_splits)} unique splits, "
            f"{len(strict_splits)} are strict consensus"
        )
        
        # 构建一致性树
        consensus_tree = self._build_tree_from_splits(strict_splits, all_taxa)
        
        return consensus_tree
    
    def build_majority_rule(
        self,
        trees: List[PhyloTree],
        threshold: float = 0.5
    ) -> PhyloTree:
        """
        构建多数规则一致性树
        
        Parameters:
            trees: 输入树列表
            threshold: 支持率阈值 (默认0.5 = 50%)
        
        Returns:
            一致性树
        """
        if not trees:
            raise ValueError("No input trees provided")
        
        if len(trees) == 1:
            return trees[0]
        
        all_taxa = set(trees[0].leaf_names)
        for tree in trees[1:]:
            all_taxa |= set(tree.leaf_names)
        
        all_splits = self._extract_all_splits(trees)
        split_counts = Counter(all_splits)
        n_trees = len(trees)
        
        # 筛选超过阈值的分割
        majority_splits = []
        for split_set, count in split_counts.items():
            freq = count / n_trees
            if freq >= threshold:
                split_obj = Split(
                    set1=split_set.set1,
                    set2=split_set.set2,
                    frequency=freq
                )
                majority_splits.append(split_obj)
        
        return self._build_tree_from_splits(majority_splits, all_taxa)
    
    def _extract_all_splits(
        self,
        trees: List[PhyloTree]
    ) -> List[Split]:
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
    
    def _extract_splits_from_tree(self, tree: PhyloTree) -> List[Split]:
        """
        从单棵树提取所有分割
        
        对于每个内部边，其两侧的叶节点构成一个分割。
        
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
            
            # 获取该节点一侧的所有叶节点
            # 使用第一个子节点的子树作为set1
            child1_leaves = set(c.name for c in node.children[0].get_leaves())
            set1 = frozenset(child1_leaves)
            
            # 其他所有叶节点作为set2
            remaining_leaves = leaves - set1
            set2 = frozenset(remaining_leaves)
            
            if not set1 or not set2:
                continue
            
            # 确保set1 < set2
            if set1 > set2:
                set1, set2 = set2, set1
            
            splits.append(Split(set1=set1, set2=set2))
        
        return splits
    
    def _build_tree_from_splits(
        self,
        splits: List[Split],
        all_taxa: Set[str]
    ) -> PhyloTree:
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
        
        # 选择最小的
        smallest = min(non_trivial, key=lambda s: len(s.set1))
        
        # 递归构建
        return self._build_recursive(splits, all_taxa)
    
    def _build_recursive(
        self,
        splits: List[Split],
        taxa: Set[str]
    ) -> PhyloTree:
        """
        递归构建树
        
        Parameters:
            splits: 分割列表
            taxa: 当前分类单元集合
        
        Returns:
            树
        """
        if len(taxa) <= 1:
            if not taxa:
                return PhyloTree()
            node = PhyloNode(name=list(taxa)[0], node_type=NodeType.LEAF)
            return PhyloTree(root=node)
        
        # 找覆盖taxa的分割
        compatible = []
        for split in splits:
            if split.set1.issubset(taxa) and split.set2.issubset(taxa):
                compatible.append(split)
        
        if not compatible:
            # 没有兼容分割，构建星形树
            return self._build_star_tree(taxa)
        
        # 选择最佳分割
        best = min(compatible, key=lambda s: abs(len(s.set1) - len(taxa) / 2))
        
        set1 = best.set1 & taxa
        set2 = best.set2 & taxa
        
        # 递归构建子树
        tree1 = self._build_recursive(splits, set1)
        tree2 = self._build_recursive(splits, set2)
        
        # 合并
        root = PhyloNode(name="", node_type=NodeType.INTERNAL)
        
        if tree1.root:
            root.add_child(tree1.root)
        if tree2.root:
            root.add_child(tree2.root)
        
        return PhyloTree(root=root)
    
    def _build_star_tree(self, taxa: Set[str]) -> PhyloTree:
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
            root = PhyloNode(name=list(taxa)[0], node_type=NodeType.LEAF)
            return PhyloTree(root=root)
        
        root = PhyloNode(name="consensus", node_type=NodeType.INTERNAL)
        
        for taxon in sorted(taxa):
            leaf = PhyloNode(name=taxon, node_type=NodeType.LEAF)
            root.add_child(leaf)
        
        return PhyloTree(root=root)


def build_strict_consensus(trees: List[PhyloTree]) -> PhyloTree:
    """
    构建严格一致性树的便捷函数
    
    Parameters:
        trees: 树列表
    
    Returns:
        一致性树
    """
    consensus = StrictConsensusTree()
    return consensus.build(trees)


def build_majority_rule_consensus(
    trees: List[PhyloTree],
    threshold: float = 0.5
) -> PhyloTree:
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
