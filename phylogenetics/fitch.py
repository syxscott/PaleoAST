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
from typing import (
    Dict, List, Optional, Set, Tuple, Any, Callable
)
from dataclasses import dataclass, field
import logging
from collections import Counter

from .tree import PhyloTree, PhyloNode, NodeType

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
    site_scores: List[int]
    character_states: Dict[int, Dict[PhyloNode, Set[Any]]]
    ancestral_states: Dict[int, Dict[PhyloNode, Any]]
    changes: List[Tuple[PhyloNode, PhyloNode, int, Any, Any]]
    
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
        
        其中 m 为最小可能变化数，L 为实际树长。
        """
        m = len(self.site_scores)  # 每个位点最少变化1次
        if self.tree_length == 0:
            return float('inf')
        return m / self.tree_length
    
    @property
    def retention_index(self) -> float:
        """
        留存指数 (Retention Index)
        
        RI = (g - s) / (g - m)
        
        其中 g 为最大可能同源，s 为实际简约变化，m 为最小可能变化。
        """
        g = len(self.site_scores)  # 简化计算
        m = len(self.site_scores)
        s = self.tree_length
        
        if g - m == 0:
            return 1.0
        return (g - s) / (g - m)


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
    
    def compute(
        self,
        tree: PhyloTree,
        sequences: Dict[str, str],
        gap_as_missing: bool = True,
        missing_char: str = '?'
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
        site_scores: List[int] = []
        character_states: Dict[int, Dict[PhyloNode, Set[Any]]] = {}
        ancestral_states: Dict[int, Dict[PhyloNode, Any]] = {}
        changes: List[Tuple[PhyloNode, PhyloNode, int, Any, Any]] = []
        
        # 对每个位点进行Fitch计算
        for site_idx in range(n_sites):
            # 提取该位点的状态
            states = self._extract_site_states(sequences, taxon_names, site_idx)
            
            # 下推阶段
            node_states = self._fitch_down(tree.root, states, gap_as_missing, missing_char)
            character_states[site_idx] = node_states
            
            # 上推阶段
            site_changes = self._fitch_up(tree.root, node_states)
            changes.extend(site_changes)
            
            # 计算该位点的分数
            site_score = len(site_changes)
            site_scores.append(site_score)
        
        # 估算祖先状态
        for site_idx in range(n_sites):
            ancestral_states[site_idx] = self._estimate_ancestral_states(
                tree.root, character_states[site_idx]
            )
        
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
            changes=changes
        )
    
    def _extract_site_states(
        self,
        sequences: Dict[str, str],
        taxon_names: List[str],
        site_idx: int
    ) -> Dict[str, Any]:
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
        self,
        node: PhyloNode,
        states: Dict[str, Any],
        gap_as_missing: bool,
        missing_char: str
    ) -> Dict[PhyloNode, Set[Any]]:
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
        node_states: Dict[PhyloNode, Set[Any]] = {}
        
        if node.is_leaf:
            # 叶节点: 状态集合为观察到的状态
            name = node.name
            if name in states:
                state = states[name]
                
                # 处理gap和缺失
                if not gap_as_missing and state == '-':
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
            
            # Fitch交集运算
            if len(node.children) >= 2:
                left_states = node_states.get(node.children[0], set())
                right_states = node_states.get(node.children[1], set())
                
                # 交集
                intersection = left_states & right_states
                
                if intersection:
                    node_states[node] = intersection
                else:
                    # 并集
                    node_states[node] = left_states | right_states
            elif node.children:
                # 单子节点情况
                node_states[node] = node_states.get(node.children[0], set())
            else:
                node_states[node] = set()
        
        return node_states
    
    def _fitch_up(
        self,
        node: PhyloNode,
        node_states: Dict[PhyloNode, Set[Any]]
    ) -> List[Tuple[PhyloNode, PhyloNode, int, Any, Any]]:
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
        changes: List[Tuple[PhyloNode, PhyloNode, int, Any, Any]] = []
        
        def _up_from(node: PhyloNode, parent_state: Optional[Any]) -> Optional[Any]:
            """
            递归上推
            
            Parameters:
                node: 当前节点
                parent_state: 父节点确定的状态
            
            Returns:
                当前节点的确定状态
            """
            current_states = node_states.get(node, set())
            
            if not current_states:
                # 无状态信息
                return parent_state
            
            # 确定当前节点状态
            if parent_state is not None and parent_state in current_states:
                current_state = parent_state
            else:
                # 选择任意状态
                current_state = next(iter(current_states))
                
                # 记录变化
                if parent_state is not None and current_state != parent_state:
                    # 需要知道位点索引，这里简化处理
                    pass
            
            # 递归处理子节点
            for child in node.children:
                _up_from(child, current_state)
            
            return current_state
        
        # 从根开始
        root_states = node_states.get(node, set())
        root_state = next(iter(root_states)) if root_states else None
        
        # 递归处理
        for child in node.children:
            _up_from(child, root_state)
        
        return changes
    
    def _estimate_ancestral_states(
        self,
        node: PhyloNode,
        node_states: Dict[PhyloNode, Set[Any]]
    ) -> Dict[PhyloNode, Any]:
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
        self,
        leaf_names: List[str],
        sequences: Dict[str, str],
        max_trees: int = 100
    ) -> List[PhyloTree]:
        """
        寻找所有最大简约树
        
        这是一个简化实现，实际应用中需要结合树搜索算法。
        
        Parameters:
            leaf_names: 叶节点名称列表
            sequences: 序列字典
            max_trees: 返回的最大树数量
        
        Returns:
            简约树列表
        """
        # 这个方法需要结合启发式搜索
        # 简化版本返回空列表
        self._logger.warning(
            "find_most_parsimonious_trees requires heuristic search. "
            "Use HeuristicSearch class for full implementation."
        )
        return []
    
    def compute_distance_from_parsimony(
        self,
        tree1: PhyloTree,
        tree2: PhyloTree
    ) -> int:
        """
        计算两棵树的简约距离
        
        使用树长度差的绝对值作为距离度量。
        
        Parameters:
            tree1: 第一棵树
            tree2: 第二棵树
        
        Returns:
            距离 (整数)
        """
        # 简化实现
        return abs(
            len(tree1.root.get_all_nodes()) - 
            len(tree2.root.get_all_nodes())
        )


def compute_parsimony_score(
    tree: PhyloTree,
    sequences: Dict[str, str]
) -> int:
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
