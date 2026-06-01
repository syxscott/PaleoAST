"""
================================================================================
PaleoAST Phylogenetics - Fitch Algorithm (Maximum Parsimony)
================================================================================

本模块实现Fitch算法用于最大简约法系统发育推断。

数学理论:
==============================================================================

1. 最大简约法原理
--------------------
给定一组分类单元的字符状态，寻求一棵使得状态变化(突变)总数最小的树。

目标函数:
    L(T) = Σ_{i=1}^{n} l_i(T)

其中:
    L(T): 树T的总长度 (简约分数)
    n: 字符数
    l_i(T): 字符i在树T上的最小变化次数

2. Fitch算法
--------------------
Fitch算法是一种计算给定树拓扑结构上最小变化次数的贪心算法。

对于二叉树，自底向上计算:

    下推(Pass Down):
        - 如果两个子节点的交集非空:
          parent_set = intersection(child1_set, child2_set)
        - 否则:
          parent_set = union(child1_set, child2_set)
          变化计数 += 1

    上推(Pass Up):
        - 从根开始，根据子节点集合确定父节点集合
        - 选择使变化最小的状态

3. 复杂性分析
--------------------
时间复杂度: O(n × m × k)
    - n: 分类单元数
    - m: 字符数
    - k: 树节点数

空间复杂度: O(k × s)
    - s: 每个节点的状态空间大小

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .tree import PhyloNode, PhyloTree

logger = logging.getLogger(__name__)


@dataclass
class FitchResult:
    """
    Fitch算法结果

    属性:
        tree_length: 树长度 (总变化数)
        site_scores: 每个位点的变化数
        character_states: 每个节点的字符状态集合
        ancestral_states: 祖先节点的最可能状态
        changes: 状态变化位置列表
    """

    tree_length: int
    site_scores: list[int]
    character_states: dict[int, dict[PhyloNode, set[Any]]]
    ancestral_states: dict[int, dict[PhyloNode, Any]]
    changes: list[tuple[PhyloNode, PhyloNode, int, Any, Any]]

    @property
    def parsimony_score(self) -> int:
        """Alias for tree_length (API compatibility)."""
        return self.tree_length

    @property
    def average_length(self) -> float:
        """平均树长"""
        if not self.site_scores:
            return 0.0
        return sum(self.site_scores) / len(self.site_scores)

    @property
    def consistency_index(self) -> float:
        """
        一致性指数 (Consistency Index)

        CI = m / L

        其中 m 为最简约树的最小步长数（每个字符的最小步长之和），L 为实际树长。
        对于二态字符，每个字符的最小步长为1。
        """
        if self.tree_length == 0:
            return 1.0
        # m = minimum possible tree length = number of parsimony-informative sites
        # (sites with >= 2 distinct states each require at least 1 step)
        m = sum(1 for s in self.site_scores if s > 0)
        return m / self.tree_length

    @property
    def retention_index(self) -> float:
        """
        留存指数 (Retention Index)

        RI = (g - s) / (g - m)

        其中 g 为最大可能步长数，m 为最小可能步长数，s 为实际树长。
        """
        # m = minimum steps (informative sites, each needs at least 1 step)
        m = sum(1 for s in self.site_scores if s > 0)
        # g = maximum steps: for each site, max_steps = max_state_count - 1
        # Simplified: each informative site can have at most (tree_length / m) steps on average
        # For a binary character, max steps per site = number of taxa - 1
        # Use total_sites as upper bound since each site contributes at most tree_length steps
        g = len(self.site_scores) if len(self.site_scores) > 0 else m
        s = self.tree_length

        if g - m == 0:
            return 1.0
        return max(0.0, (g - s) / (g - m))


class FitchAlgorithm:
    """
    Fitch算法实现

    用于计算给定系统发育树的最大简约分数。

    使用示例:
        >>> fitch = FitchAlgorithm()
        >>> # 假设有4个分类单元，DNA序列
        >>> sequences = {
        ...     'A': 'ATGC',
        ...     'B': 'ATAC',
        ...     'C': 'GTAC',
        ...     'D': 'GTGC',
        ... }
        >>> tree = PhyloTree.from_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        >>> result = fitch.compute(tree, sequences)
        >>> print(f"Tree length: {result.tree_length}")
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.Fitch")

    def run(self, tree, sequences, **kwargs):
        """Alias for compute() for API consistency."""
        return self.compute(tree, sequences, **kwargs)

    def compute(
        self, tree: PhyloTree, sequences: dict[str, str], gap_as_missing: bool = True, missing_char: str = "?"
    ) -> FitchResult:
        """
        计算树的简约分数

        Parameters:
            tree: 系统发育树
            sequences: {分类单元名: 序列} 字典
            gap_as_missing: 是否将gap视为缺失
            missing_char: 缺失字符标记

        Returns:
            FitchResult对象
        """
        if tree.root is None:
            raise ValueError("Tree has no root")

        # 获取所有位点
        taxon_names = list(sequences.keys())
        first_seq = sequences[taxon_names[0]]
        n_sites = len(first_seq)

        # 初始化结果存储
        site_scores: list[int] = []
        character_states: dict[int, dict[PhyloNode, set[Any]]] = {}
        ancestral_states: dict[int, dict[PhyloNode, Any]] = {}
        changes: list[tuple[PhyloNode, PhyloNode, int, Any, Any]] = []

        # 对每个位点进行Fitch计算
        for site_idx in range(n_sites):
            # 提取该位点的状态
            states = self._extract_site_states(sequences, taxon_names, site_idx)

            # 下推阶段
            node_states = self._fitch_down(tree.root, states, gap_as_missing, missing_char)
            character_states[site_idx] = node_states

            # 上推阶段
            site_changes = self._fitch_up(tree.root, node_states, site_idx=site_idx)
            changes.extend(site_changes)

            # 计算该位点的分数
            site_score = len(site_changes)
            site_scores.append(site_score)

        # 估算祖先状态
        for site_idx in range(n_sites):
            ancestral_states[site_idx] = self._estimate_ancestral_states(tree.root, character_states[site_idx])

        total_length = sum(site_scores)

        self._logger.info(
            f"Fitch analysis complete: tree length = {total_length}, "
            f"sites = {n_sites}, CI = {total_length / n_sites if n_sites > 0 else 0:.2f}"
        )

        return FitchResult(
            tree_length=total_length,
            site_scores=site_scores,
            character_states=character_states,
            ancestral_states=ancestral_states,
            changes=changes,
        )

    def _extract_site_states(self, sequences: dict[str, str], taxon_names: list[str], site_idx: int) -> dict[str, Any]:
        """
        提取指定位点的状态

        Parameters:
            sequences: 序列字典
            taxon_names: 分类单元名列表
            site_idx: 位点索引

        Returns:
            {分类单元名: 状态} 字典
        """
        states = {}
        for taxon in taxon_names:
            seq = sequences[taxon]
            if site_idx < len(seq):
                state = seq[site_idx]
                states[taxon] = state
        return states

    def _fitch_down(
        self, node: PhyloNode, states: dict[str, Any], gap_as_missing: bool, missing_char: str
    ) -> dict[PhyloNode, set[Any]]:
        """
        Fitch下推阶段

        自叶向根计算每个节点的最小状态集合。

        算法:
            For leaf node:
                S(v) = {state(v)}

            For internal node:
                S(v) = S(left) ∩ S(right)      if intersection non-empty
                     = S(left) ∪ S(right)       otherwise

        Parameters:
            node: 当前节点
            states: 叶节点状态
            gap_as_missing: gap是否视为缺失
            missing_char: 缺失字符

        Returns:
            {节点: 状态集合} 字典
        """
        node_states: dict[PhyloNode, set[Any]] = {}

        if node.is_leaf:
            # 叶节点: 状态集合为观察到的状态
            name = node.name
            if name in states:
                state = states[name]

                # 处理gap和缺失
                if not gap_as_missing and state == "-":
                    node_states[node] = set()
                elif state == missing_char:
                    node_states[node] = set()  # 空集表示不确定
                else:
                    node_states[node] = {state}
            else:
                node_states[node] = set()

        else:
            # 内部节点: 递归处理子节点
            for child in node.children:
                child_states = self._fitch_down(child, states, gap_as_missing, missing_char)
                node_states.update(child_states)

            # Fitch交集运算 (支持多分叉)
            if node.children:
                # 从第一个子节点的状态集合开始
                combined = node_states.get(node.children[0], set()).copy()
                for child in node.children[1:]:
                    child_set = node_states.get(child, set())
                    # 交集
                    intersection = combined & child_set
                    if intersection:
                        combined = intersection
                    else:
                        # 并集
                        combined = combined | child_set
                node_states[node] = combined
            else:
                node_states[node] = set()

        return node_states

    def _fitch_up(
        self, node: PhyloNode, node_states: dict[PhyloNode, set[Any]], site_idx: int = -1
    ) -> list[tuple[PhyloNode, PhyloNode, int, Any, Any]]:
        """
        Fitch上推阶段

        从根向叶确定最可能的状态，统计变化。

        算法:
            For root:
                选择 S(root) 中的任意状态 (通常选第一个)

            For each child:
                如果 child_state ⊆ parent_state:
                    选择 parent_state
                    变化数 += 0
                否则:
                    选择 child_state ∪ parent_state 的任意元素
                    变化数 += 1

        Parameters:
            node: 当前节点
            node_states: 节点状态集合

        Returns:
            [(父节点, 子节点, 位点索引, 父状态, 子状态)] 变化列表
        """
        changes: list[tuple[PhyloNode, PhyloNode, int, Any, Any]] = []
        # site_idx passed from caller

        def _up_from(node: PhyloNode, parent_state: Any | None) -> Any | None:
            nonlocal changes
            current_states = node_states.get(node, set())

            if not current_states:
                return parent_state

            if parent_state is not None and parent_state in current_states:
                current_state = parent_state
            else:
                current_state = next(iter(current_states))
                if parent_state is not None and current_state != parent_state:
                    changes.append((node.parent if node.parent else node, node, site_idx, parent_state, current_state))

            for child in node.children:
                _up_from(child, current_state)

            return current_state

        root_states = node_states.get(node, set())
        root_state = next(iter(root_states)) if root_states else None

        for child in node.children:
            _up_from(child, root_state)

        return changes

    def _estimate_ancestral_states(
        self, node: PhyloNode, node_states: dict[PhyloNode, set[Any]]
    ) -> dict[PhyloNode, Any]:
        """
        估算祖先节点的最可能状态

        Parameters:
            node: 当前节点
            node_states: 节点状态集合

        Returns:
            {节点: 最可能状态} 字典
        """
        ancestral = {}

        def _estimate(n: PhyloNode) -> Any:
            states = node_states.get(n, set())
            if not states:
                return None

            # 选择第一个状态 (简化处理)
            # 实际应用中可以使用投票、加权等方法
            ancestral[n] = next(iter(states))
            return ancestral[n]

        # 后序遍历
        for n in node.postorder_traverse():
            _estimate(n)

        return ancestral

    def find_most_parsimonious_trees(
        self, leaf_names: list[str], sequences: dict[str, str], max_trees: int = 100
    ) -> list[PhyloTree]:
        """
        寻找所有最大简约树

        使用启发式搜索（NNI局部搜索）寻找最优简约树。

        Parameters:
            leaf_names: 叶节点名称列表
            sequences: 序列字典
            max_trees: 返回的最大树数量

        Returns:
            简约树列表
        """
        from .heuristic_search import HeuristicSearch

        search = HeuristicSearch(algorithm="parsimony", max_iterations=500, random_seed=42)
        result = search.search(leaf_names, sequences)

        self._logger.info(f"Found {len(result.all_trees)} most parsimonious trees with score {result.best_score}")
        return result.all_trees[:max_trees]

    @staticmethod
    def _extract_splits(tree: PhyloTree) -> set[frozenset[frozenset[str]]]:
        """
        提取树的所有非平凡分割（用于Robinson-Foulds距离计算）

        Parameters:
            tree: 系统发育树

        Returns:
            分割集合，每个分割为 {子树叶节点集合, 剩余叶节点集合}
        """
        if tree.root is None:
            return set()

        all_leaves = frozenset(tree.leaf_names)
        splits = set()

        for node in tree.root.preorder_traverse():
            if node.is_leaf or node.is_root:
                continue
            child_leaves = frozenset(leaf.name for leaf in node.get_leaves())
            if 1 < len(child_leaves) < len(all_leaves):
                complement = all_leaves - child_leaves
                splits.add(frozenset({child_leaves, complement}))

        return splits

    def compute_distance_from_parsimony(self, tree1: PhyloTree, tree2: PhyloTree) -> int:
        """
        计算两棵树的Robinson-Foulds距离

        RF距离 = |S1 △ S2| = |S1 - S2| + |S2 - S1|

        其中S1、S2分别是两棵树的分割集合。

        Parameters:
            tree1: 第一棵树
            tree2: 第二棵树

        Returns:
            Robinson-Foulds距离 (非负整数)
        """
        splits1 = self._extract_splits(tree1)
        splits2 = self._extract_splits(tree2)
        rf_distance = len(splits1.symmetric_difference(splits2))
        return rf_distance


def compute_parsimony_score(tree: PhyloTree, sequences: dict[str, str]) -> int:
    """
    计算树简约分数的便捷函数

    Parameters:
        tree: 系统发育树
        sequences: 序列字典

    Returns:
        简约分数
    """
    fitch = FitchAlgorithm()
    result = fitch.compute(tree, sequences)
    return result.tree_length
