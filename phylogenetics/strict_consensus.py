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

        实现说明:
            采用 **有根 clade (分支集)** 表示而非无根分割表示。同一批输入树
            中共同出现的 clade 天然构成层级 (laminar) 族，因此可以按 clade
            大小降序自顶向下对称地重建拓扑；旧的"分割 + 单侧递归"实现在两侧
            递归不对称时会丢失 clade（连 build([T, T]) 都无法还原 T）。

        Parameters:
            trees: 输入树列表

        Returns:
            一致性树 (单棵树输入时返回其深拷贝，不与入参共享节点)

        Raises:
            ValueError: 输入树列表为空
        """
        if not trees:
            raise ValueError("No input trees provided")

        if len(trees) == 1:
            # 返回深拷贝: 直接把入参别名返回会让调用方修改一致性树时改写原树
            return self._clone_tree(trees[0])

        # 获取所有分类单元
        all_taxa = set(trees[0].leaf_names)
        for tree in trees[1:]:
            all_taxa |= set(tree.leaf_names)

        clade_counts = self._count_clades(trees)
        n_trees = len(trees)

        frequencies = {clade: count / n_trees for clade, count in clade_counts.items()}
        strict_clades = [clade for clade, freq in frequencies.items() if abs(freq - 1.0) < 1e-10]

        self._logger.info(
            f"Extracted {len(frequencies)} unique clades, {len(strict_clades)} are strict consensus"
        )

        consensus_tree = self._build_tree_from_clades(strict_clades, all_taxa)
        consensus_tree.metadata["consensus_clades"] = len(strict_clades)
        consensus_tree.metadata["input_trees"] = n_trees
        return consensus_tree

    def build_majority_rule(self, trees: list[PhyloTree], threshold: float = 0.5) -> PhyloTree:
        """
        构建多数规则一致性树

        Parameters:
            trees: 输入树列表
            threshold: 支持率阈值 (默认0.5 = 50%)

        Returns:
            一致性树。结果树的 ``metadata["conflicting_clades"]`` 记录彼此不兼容
            (无法同时容纳在同一棵树中) 的 clade 对数。

        注意:
            当threshold < 1.0且输入树有冲突拓扑时，
            多数规则一致性可能产生不兼容的分割组合。
            此时会发出警告，且结果树可能退化为星形树。
        """
        if not trees:
            raise ValueError("No input trees provided")

        if len(trees) == 1:
            return self._clone_tree(trees[0])

        all_taxa = set(trees[0].leaf_names)
        for tree in trees[1:]:
            all_taxa |= set(tree.leaf_names)

        clade_counts = self._count_clades(trees)
        n_trees = len(trees)

        frequencies = {clade: count / n_trees for clade, count in clade_counts.items()}
        majority_clades = [clade for clade, freq in frequencies.items() if freq >= threshold - 1e-12]

        # 统计互不兼容的 clade 对 (两侧交集非空且互不包含)
        conflicts = self._count_conflicts(majority_clades)
        if conflicts:
            self._logger.warning(
                f"Majority rule consensus: found {conflicts} incompatible clade pairs. "
                f"The resulting tree may be unresolved (star tree) or invalid. "
                f"Consider using a higher threshold or checking input tree compatibility."
            )

        consensus_tree = self._build_tree_from_clades(majority_clades, all_taxa)
        consensus_tree.metadata["consensus_clades"] = len(majority_clades)
        consensus_tree.metadata["conflicting_clades"] = conflicts
        consensus_tree.metadata["input_trees"] = n_trees
        consensus_tree.metadata["threshold"] = threshold
        return consensus_tree

    # ------------------------------------------------------------------
    # clade 统计与树重建
    # ------------------------------------------------------------------
    def _count_clades(self, trees: list[PhyloTree]) -> Counter:
        """统计每个有根 clade (叶名集合) 出现在多少棵树中。"""
        clade_counts: Counter = Counter()
        for tree in trees:
            if tree.root is None:
                continue
            for clade in set(self._extract_clades_from_tree(tree)):
                clade_counts[clade] += 1
        return clade_counts

    def _extract_clades_from_tree(self, tree: PhyloTree) -> list[frozenset[str]]:
        """
        提取单棵树的全部有根 clade (含全分类单元根 clade)。

        Returns:
            frozenset(叶名) 列表
        """
        clades: list[frozenset[str]] = []
        for node in tree.root.preorder_traverse():
            if node.is_leaf:
                continue
            leaves = frozenset(leaf.name for leaf in node.get_leaves())
            if len(leaves) >= 2:
                clades.append(leaves)
        return clades

    @staticmethod
    def _count_conflicts(clades: list[frozenset[str]]) -> int:
        """统计互不兼容 (相交且互不包含) 的 clade 对数。"""
        unique = list(set(clades))
        conflicts = 0
        for i, c1 in enumerate(unique):
            for c2 in unique[i + 1 :]:
                if c1 & c2 and not (c1 <= c2 or c2 <= c1):
                    conflicts += 1
        return conflicts

    def _build_tree_from_clades(self, clades: list[frozenset[str]], all_taxa: set[str]) -> PhyloTree:
        """
        从 clade 族自顶向下对称建树。

        算法:
            1. 当前 clade 的直接子 clade = 其中极大真子集 (按大小降序贪心选取，
               已被子 clade 覆盖的分类单元不再重复挂载，从而对不兼容 clade 免疫)
            2. 未被任何子 clade 覆盖的分类单元作为叶节点直接挂到当前节点
            3. 对每个子 clade 递归

        Parameters:
            clades: clade 族 (叶名集合)
            all_taxa: 全部分类单元

        Returns:
            一致性树
        """
        taxa = set(all_taxa)
        if not taxa:
            return PhyloTree()
        if len(taxa) == 1:
            return PhyloTree(root=PhyloNode(name=next(iter(taxa)), node_type=NodeType.LEAF))

        # 仅保留真子集 clade，按大小降序 (自顶向下)
        usable = sorted({c for c in clades if c and c < taxa}, key=len, reverse=True)

        def build_group(group: set[str]) -> PhyloNode:
            remaining = set(group)
            children: list[tuple[str, PhyloNode]] = []
            for clade in usable:
                if clade >= group:
                    # 必须是当前组的真子集
                    continue
                if not clade <= remaining:
                    # 与已选取的子 clade 重叠 (不兼容) 或不属于本组
                    continue
                if len(clade) == 1:
                    child: PhyloNode = PhyloNode(name=next(iter(clade)), node_type=NodeType.LEAF)
                else:
                    child = build_group(set(clade))
                children.append((min(clade), child))
                remaining -= clade
            for taxon in sorted(remaining):
                children.append((taxon, PhyloNode(name=taxon, node_type=NodeType.LEAF)))

            node = PhyloNode(name="", node_type=NodeType.INTERNAL)
            # 按最小叶名排序，保证输出稳定且与常规 Newick 阅读顺序一致
            for _, child in sorted(children, key=lambda item: item[0]):
                node.add_child(child)
            return node

        root = build_group(taxa)
        return PhyloTree(root=root)

    def _clone_tree(self, tree: PhyloTree) -> PhyloTree:
        """深拷贝一棵树 (不共享节点对象)。"""
        if tree.root is None:
            return PhyloTree(name=tree.name, metadata=dict(tree.metadata))
        new_root = tree.root._copy_subtree()
        return PhyloTree(root=new_root, name=tree.name, metadata=dict(tree.metadata))

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
            for child in node.children:
                child_leaves = set(c.name for c in child.get_leaves())
                set1 = frozenset(child_leaves)

                # 其他所有叶节点作为set2
                remaining_leaves = leaves - set1
                set2 = frozenset(remaining_leaves)

                if not set1 or not set2:
                    continue

                # 规范化由 Split.__post_init__ 完成 (按最小元素定序)。
                # 此处不再有 "if set1 > set2: swap" 的必要——frozenset 的 ">"
                # 是超集关系而非大小关系，该分支在两侧互不包含时永远不成立。
                splits.append(Split(set1=set1, set2=set2))

        return splits

    def _build_tree_from_splits(self, splits: list[Split], all_taxa: set[str]) -> PhyloTree:
        """
        从分割列表构建树 (兼容旧接口)

        历史实现采用"最小分割 + 单侧递归"的方式，只在分割的一侧递归、另一侧
        按叶展开，因此两棵相同的树求共识也会丢 clade（如 ((A,B),(C,D)) 与自身
        的共识退化为 ((A,B),C,D)）。现改为把分割转成有根 clade 族后交给
        :meth:`_build_tree_from_clades` 对称重建。

        Parameters:
            splits: 分割列表
            all_taxa: 所有分类单元

        Returns:
            构建的树
        """
        if not splits:
            # 没有分割，返回星形树
            return self._build_star_tree(all_taxa)

        taxa = frozenset(all_taxa)
        clades: list[frozenset[str]] = []
        for split in splits:
            for side in (split.set1, split.set2):
                side = frozenset(side)
                # 只保留作为"子 clade"有意义的侧: 真子集且至少 2 个分类单元
                if len(side) >= 2 and side < taxa:
                    clades.append(side)

        return self._build_tree_from_clades(clades, set(all_taxa))

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
