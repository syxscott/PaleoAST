"""
================================================================================
PaleoAST Phylogenetics - Tree Data Structure
================================================================================

本模块提供系统发育树的核心数据结构实现。

数学定义:
==============================================================================

树是一种无环连通图。对于系统发育树:

    T = (V, E, L, w)
    
其中:
    V: 节点集合 {v1, v2, ..., vn}
    E: 边集合 {(vi, vj), ...}
    L ⊆ V: 叶节点集合 (分类单元)
    w: E → ℝ≥0: 枝长函数

树的性质:
1. |E| = |V| - 1 (对于有根树)
2. 每个内部节点至少有两个子节点 (二叉树为正好两个)
3. 树无环: 不存在路径从节点回到自身

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import (
    Dict, List, Optional, Set, Tuple, Any, Iterator,
    Callable, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import uuid
from collections import deque

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


class NodeType(Enum):
    """节点类型枚举"""
    ROOT = auto()
    INTERNAL = auto()
    LEAF = auto()


@dataclass
class PhyloNode:
    """
    系统发育树节点
    
    属性:
        node_id: 唯一标识符
        name: 节点名称 (叶节点为分类单元名)
        node_type: 节点类型
        parent: 父节点引用
        children: 子节点列表
        branch_length: 枝长
        support: 支持率 (如Bootstrap)
        data: 节点数据 (如序列、特征)
        label: 用户自定义标签
        metadata: 元数据字典
    
    数学表示:
        节点 v 具有属性:
        - degree(v): 度 = |children(v)| + 1 (父节点)
        - depth(v): 深度 = 从根到本节点的边数
        - height(v): 高度 = 从本节点到最深叶的距离
    
    示例:
        >>> leaf = PhyloNode(name="Homo_sapiens", node_type=NodeType.LEAF)
        >>> internal = PhyloNode(name="Primates", node_type=NodeType.INTERNAL)
        >>> internal.add_child(leaf)
    """
    name: str
    node_type: NodeType = NodeType.INTERNAL
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent: Optional['PhyloNode'] = None
    children: List['PhyloNode'] = field(default_factory=list)
    branch_length: Optional[float] = None
    support: Optional[float] = None
    data: Any = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        if self.branch_length is not None and self.branch_length < 0:
            raise ValueError(
                f"Branch length must be non-negative, got {self.branch_length}"
            )
        
        # 自动设置节点类型
        if not self.children and self.node_type == NodeType.INTERNAL:
            self.node_type = NodeType.LEAF
    
    # =========================================================================
    # 节点属性
    # =========================================================================
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶节点"""
        return self.node_type == NodeType.LEAF or len(self.children) == 0
    
    @property
    def is_root(self) -> bool:
        """是否为根节点"""
        return self.parent is None
    
    @property
    def is_binary(self) -> bool:
        """是否为二叉节点 (度=3或1)"""
        return len(self.children) in (0, 2)
    
    @property
    def degree(self) -> int:
        """节点的度"""
        return len(self.children) + (0 if self.is_root else 1)
    
    @property
    def arity(self) -> int:
        """节点的分支数 (子节点数)"""
        return len(self.children)
    
    # =========================================================================
    # 树遍历
    # =========================================================================
    
    def preorder_traverse(self) -> Iterator['PhyloNode']:
        """
        前序遍历 (先访问节点，再访问子树)
        
        遍历顺序: 根 → 左子树 → 右子树
        
        Yields:
            遍历到的节点
        """
        yield self
        for child in self.children:
            yield from child.preorder_traverse()
    
    def postorder_traverse(self) -> Iterator['PhyloNode']:
        """
        后序遍历 (先访问子树，再访问节点)
        
        遍历顺序: 左子树 → 右子树 → 根
        
        Yields:
            遍历到的节点
        """
        for child in self.children:
            yield from child.postorder_traverse()
        yield self
    
    def level_order_traverse(self) -> Iterator['PhyloNode']:
        """
        层序遍历 (BFS)
        
        按层级从根到叶遍历。
        
        Yields:
            遍历到的节点
        """
        queue = deque([self])
        
        while queue:
            node = queue.popleft()
            yield node
            
            for child in node.children:
                queue.append(child)
    
    def get_leaves(self) -> List['PhyloNode']:
        """
        获取所有叶节点
        
        Returns:
            叶节点列表
        """
        if self.is_leaf:
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves
    
    def get_all_nodes(self) -> List['PhyloNode']:
        """
        获取所有节点
        
        Returns:
            节点列表 (前序遍历顺序)
        """
        return list(self.preorder_traverse())
    
    def get_path_to_root(self) -> List['PhyloNode']:
        """
        获取从本节点到根节点的路径
        
        Returns:
            节点列表 (从本节点到根)
        """
        path = [self]
        node = self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return path
    
    def get_ancestors(self) -> List['PhyloNode']:
        """
        获取所有祖先节点 (不包括本节点)
        
        Returns:
            祖先节点列表 (从父到根)
        """
        ancestors = []
        node = self.parent
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors
    
    def get_siblings(self) -> List['PhyloNode']:
        """
        获取所有兄弟节点
        
        Returns:
            兄弟节点列表
        """
        if self.is_root:
            return []
        
        siblings = []
        for sibling in self.parent.children:
            if sibling is not self:
                siblings.append(sibling)
        return siblings
    
    def get_subtree_nodes(self) -> List['PhyloNode']:
        """
        获取以本节点为根的子树中的所有节点
        
        Returns:
            子树节点列表
        """
        return list(self.preorder_traverse())
    
    # =========================================================================
    # 节点操作
    # =========================================================================
    
    def add_child(self, child: 'PhyloNode') -> 'PhyloNode':
        """
        添加子节点
        
        Parameters:
            child: 要添加的子节点
        
        Returns:
            添加的子节点
        """
        if child.parent is not None:
            # 从原父节点移除
            child.parent.children.remove(child)
        
        child.parent = self
        self.children.append(child)
        
        # 更新节点类型
        if self.node_type == NodeType.LEAF:
            self.node_type = NodeType.INTERNAL
        
        return child
    
    def remove_child(self, child: 'PhyloNode') -> bool:
        """
        移除子节点
        
        Parameters:
            child: 要移除的子节点
        
        Returns:
            如果成功移除返回True
        """
        if child not in self.children:
            return False
        
        self.children.remove(child)
        child.parent = None
        
        # 更新节点类型
        if not self.children and not self.is_root:
            self.node_type = NodeType.LEAF
        
        return True
    
    def prune(self) -> Optional['PhyloNode']:
        """
        剪除本节点，将其子树接到父节点
        
        Returns:
            被剪除的节点
        """
        if self.is_root:
            return None
        
        parent = self.parent
        siblings = self.get_siblings()
        
        # 从父节点移除
        parent.remove_child(self)
        
        # 将本节点的子节点添加到父节点
        for child in self.children:
            parent.add_child(child)
        
        return self
    
    def detach(self) -> 'PhyloNode':
        """
        从树中分离本节点
        
        Returns:
            分离后的节点 (作为新的根)
        """
        if self.is_root:
            return self
        
        # 递归复制子树
        new_root = self._copy_subtree()
        return new_root
    
    def _copy_subtree(self, parent: Optional['PhyloNode'] = None) -> 'PhyloNode':
        """
        递归复制子树
        
        Parameters:
            parent: 新树的父节点
        
        Returns:
            复制后的根节点
        """
        new_node = PhyloNode(
            name=self.name,
            node_type=self.node_type,
            branch_length=self.branch_length,
            support=self.support,
            data=self.data,
            label=self.label,
        )
        new_node.parent = parent
        
        for child in self.children:
            new_child = child._copy_subtree(new_node)
            new_node.children.append(new_child)
        
        return new_node
    
    # =========================================================================
    # 树操作
    # =========================================================================
    
    def compute_depths(self) -> Dict['PhyloNode', int]:
        """
        计算所有节点的深度
        
        Returns:
            {节点: 深度} 字典
        """
        depths = {}
        
        def dfs(node: PhyloNode, depth: int) -> None:
            depths[node] = depth
            for child in node.children:
                dfs(child, depth + 1)
        
        dfs(self, 0)
        return depths
    
    def compute_heights(self) -> Dict['PhyloNode', int]:
        """
        计算所有节点的高度
        
        节点高度 = 到最远叶节点的距离
        
        Returns:
            {节点: 高度} 字典
        """
        heights = {}
        
        def dfs(node: PhyloNode) -> int:
            if node.is_leaf:
                heights[node] = 0
                return 0
            
            max_child_height = 0
            for child in node.children:
                h = dfs(child)
                max_child_height = max(max_child_height, h)
            
            heights[node] = max_child_height + 1
            return heights[node]
        
        dfs(self)
        return heights
    
    def compute_total_length(self) -> float:
        """
        计算树的根到所有叶的总枝长
        
        Returns:
            总枝长
        """
        total = 0.0
        
        def dfs(node: PhyloNode) -> float:
            length = node.branch_length or 0.0
            if node.is_leaf:
                return length
            
            subtree_length = 0.0
            for child in node.children:
                subtree_length += dfs(child)
            
            return length + subtree_length
        
        return dfs(self)
    
    def compute_lca(self, node1: 'PhyloNode', node2: 'PhyloNode') -> Optional['PhyloNode']:
        """
        计算两个节点的最近公共祖先 (LCA)
        
        数学公式:
            LCA(v1, v2) = arg max_{v ∈ Path(v1, root) ∩ Path(v2, root)} depth(v)
        
        Parameters:
            node1: 第一个节点
            node2: 第二个节点
        
        Returns:
            LCA节点或None
        """
        # 获取路径
        path1 = set(node1.get_path_to_root())
        path2 = node2.get_path_to_root()
        
        # 找交集中深度最大的
        common = path1 & set(path2)
        if not common:
            return None
        
        return max(common, key=lambda n: len(n.get_path_to_root()))
    
    def get_distance(self, other: 'PhyloNode') -> float:
        """
        计算两个节点之间的距离
        
        距离 = 两节点到LCA的枝长之和
        
        Parameters:
            other: 另一个节点
        
        Returns:
            距离
        """
        lca = self.compute_lca(self, other)
        if lca is None:
            return float('inf')
        
        dist1 = self._distance_to_ancestor(lca)
        dist2 = other._distance_to_ancestor(lca)
        
        return dist1 + dist2
    
    def _distance_to_ancestor(self, ancestor: 'PhyloNode') -> float:
        """
        计算到祖先节点的距离
        
        Parameters:
            ancestor: 祖先节点
        
        Returns:
            距离
        """
        node = self
        distance = 0.0
        
        while node is not None and node is not ancestor:
            distance += node.branch_length or 0.0
            node = node.parent
        
        return distance
    
    # =========================================================================
    # Newick格式
    # =========================================================================
    
    def to_newick(self, include_lengths: bool = True, include_support: bool = False) -> str:
        """
        转换为Newick格式字符串
        
        递归文法:
            Subtree → Name:Length
                    | (Subtree{,Subtree})Name:Length
        
        Parameters:
            include_lengths: 是否包含枝长
            include_support: 是否包含支持率
        
        Returns:
            Newick格式字符串
        """
        if self.is_leaf:
            name = self.name if self.name else ""
            if include_lengths and self.branch_length is not None:
                return f"{name}:{self.branch_length}"
            return name
        
        # 内部节点
        child_newicks = []
        for child in self.children:
            child_newicks.append(child.to_newick(include_lengths, include_support))
        
        subtree = f"({','.join(child_newicks)})"
        
        # 添加支持率
        if include_support and self.support is not None:
            subtree += f"{self.support}"
        elif self.name:
            subtree += self.name
        
        # 添加枝长
        if include_lengths and self.branch_length is not None:
            subtree += f":{self.branch_length}"
        
        return subtree
    
    @classmethod
    def from_newick(cls, newick: str) -> 'PhyloNode':
        """
        从Newick字符串构建树
        
        Parameters:
            newick: Newick格式字符串
        
        Returns:
            根节点
        """
        parser = _NewickParser()
        return parser.parse(newick)
    
    def __repr__(self) -> str:
        return f"PhyloNode('{self.name}', type={self.node_type.name})"
    
    def __str__(self) -> str:
        return self.to_newick()


class _NewickParser:
    """
    Newick格式解析器 (内部使用)
    """
    
    def __init__(self):
        self._input = ""
        self._pos = 0
        self._length = 0
    
    def parse(self, newick: str) -> PhyloNode:
        """
        解析Newick字符串
        
        Parameters:
            newick: Newick格式字符串
        
        Returns:
            根节点
        """
        self._input = newick.strip().rstrip(';')
        self._pos = 0
        self._length = len(self._input)
        
        return self._parse_subtree()
    
    def _parse_subtree(self) -> PhyloNode:
        """解析子树"""
        self._skip_whitespace()
        
        # 检查是否开始子节点列表
        if self._current() == '(':
            return self._parse_internal_node()
        else:
            return self._parse_leaf_node()
    
    def _parse_internal_node(self) -> PhyloNode:
        """解析内部节点"""
        self._consume('(')
        
        children = []
        while True:
            children.append(self._parse_subtree())
            
            self._skip_whitespace()
            
            if self._current() == ',':
                self._consume(',')
            elif self._current() == ')':
                self._consume(')')
                break
            else:
                raise ValueError(f"Expected ',' or ')', got '{self._current()}'")
        
        # 解析名称
        name = self._parse_name()
        
        # 解析枝长
        branch_length = None
        if self._current() == ':':
            self._consume(':')
            branch_length = self._parse_number()
        
        # 创建节点
        node = PhyloNode(
            name=name or "",
            node_type=NodeType.INTERNAL,
            branch_length=branch_length
        )
        
        for child in children:
            node.add_child(child)
        
        return node
    
    def _parse_leaf_node(self) -> PhyloNode:
        """解析叶节点"""
        name = self._parse_name()
        
        branch_length = None
        if self._current() == ':':
            self._consume(':')
            branch_length = self._parse_number()
        
        return PhyloNode(
            name=name or "",
            node_type=NodeType.LEAF,
            branch_length=branch_length
        )
    
    def _parse_name(self) -> str:
        """解析节点名称"""
        self._skip_whitespace()
        
        name_chars = []
        while self._pos < self._length:
            char = self._current()
            if char in '(),:;[]':
                break
            name_chars.append(char)
            self._pos += 1
        
        return ''.join(name_chars)
    
    def _parse_number(self) -> float:
        """解析数字"""
        self._skip_whitespace()
        
        num_chars = []
        while self._pos < self._length:
            char = self._current()
            if char in '(),:;[] \t\n':
                break
            num_chars.append(char)
            self._pos += 1
        
        return float(''.join(num_chars))
    
    def _skip_whitespace(self) -> None:
        """跳过空白"""
        while self._pos < self._length and self._input[self._pos] in ' \t\n\r':
            self._pos += 1
    
    def _current(self) -> str:
        """获取当前字符"""
        if self._pos < self._length:
            return self._input[self._pos]
        return ''
    
    def _consume(self, char: str) -> None:
        """消耗指定字符"""
        self._skip_whitespace()
        if self._current() != char:
            raise ValueError(f"Expected '{char}', got '{self._current()}'")
        self._pos += 1


@dataclass
class PhyloTree:
    """
    系统发育树容器
    
    提供树级别操作的封装。
    
    属性:
        root: 根节点
        name: 树名称
        metadata: 元数据
    
    示例:
        >>> tree = PhyloTree()
        >>> tree.root = PhyloNode.from_newick("(A:0.1,B:0.2)C:0.3;")
        >>> print(tree.leaf_count)
    """
    root: Optional[PhyloNode] = None
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def leaf_count(self) -> int:
        """获取叶节点数量"""
        if self.root is None:
            return 0
        return len(self.root.get_leaves())
    
    @property
    def leaf_names(self) -> List[str]:
        """获取所有叶节点名称"""
        if self.root is None:
            return []
        return [leaf.name for leaf in self.root.get_leaves()]
    
    @property
    def node_count(self) -> int:
        """获取节点总数"""
        if self.root is None:
            return 0
        return len(self.root.get_all_nodes())
    
    def to_newick(self, include_lengths: bool = True) -> str:
        """
        转换为Newick格式
        
        Parameters:
            include_lengths: 是否包含枝长
        
        Returns:
            Newick格式字符串
        """
        if self.root is None:
            return ";"
        return self.root.to_newick(include_lengths) + ";"
    
    @classmethod
    def from_newick(cls, newick: str, name: str = "") -> 'PhyloTree':
        """
        从Newick字符串创建树
        
        Parameters:
            newick: Newick格式字符串
            name: 树名称
        
        Returns:
            PhyloTree对象
        """
        root = PhyloNode.from_newick(newick)
        return cls(root=root, name=name)
    
    def get_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        计算所有叶节点对之间的距离矩阵
        
        Returns:
            {(名称1, 名称2): 距离} 字典
        """
        leaves = self.root.get_leaves()
        distances = {}
        
        for i, leaf1 in enumerate(leaves):
            for leaf2 in leaves[i+1:]:
                dist = leaf1.get_distance(leaf2)
                distances[(leaf1.name, leaf2.name)] = dist
                distances[(leaf2.name, leaf1.name)] = dist
        
        return distances
    
    def __repr__(self) -> str:
        return f"PhyloTree(leaves={self.leaf_count}, nodes={self.node_count})"
