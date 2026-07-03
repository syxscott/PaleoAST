"""
================================================================================
PaleoAST Phylogenetics - Heuristic Tree Search
================================================================================

本模块实现启发式树搜索算法用于系统发育推断。

数学理论:
==============================================================================

1. 问题定义
--------------------
系统发育推断是一个组合优化问题:

    优化目标: 最小化树长度 (最大简约) 或 最大化似然值

    搜索空间: 所有可能的树拓扑结构
    |TreeSpace| = (2n - 5)!! = (2n-5)! / [2^{n-2} × (n-2)!]

    其中 n 为分类单元数。

2. 启发式搜索策略
--------------------
由于穷举搜索对大多数情况不可行，采用启发式方法:

a) 添加分类单元的顺序
   - 随机添加 (随机性保证)
   - 基于距离添加 (逐步添加)

b) 局部搜索操作符
   - NNI (Nearest Neighbor Interchange)
   - TBR (Tree Bisection and Reconnection)
   - SPR (Subtree Pruning and Regrafting)

3. NNI变换
--------------------
最近邻居互换，交换一条边两侧的子树。

    原始:  (A,B)-(C,D)
    NNI1:  (A,C)-(B,D)
    NNI2:  (A,D)-(B,C)

4. TBR变换
--------------------
树二分与重连:

    1. 移除一条边，将树分为两部分
    2. 在每部分中选择一个节点
    3. 通过新边连接两点

5. 搜索算法
--------------------
    HeuristicSearch():
        1. 构建初始树 (NJ或随机)
        2. 评估初始树分数
        3. 重复直到收敛:
           a) 对每条内部边应用NNI/TBR
           b) 评估新树分数
           c) 如果改善，接受新树
           d) 否则，以概率p接受 (模拟退火)
        4. 返回最佳树

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np

from .fitch import FitchAlgorithm
from .tree import NodeType, PhyloNode, PhyloTree

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    树搜索结果

    属性:
        best_tree: 最佳树
        best_score: 最佳分数
        all_trees: 所有找到的等长树
        iterations: 迭代次数
        time_elapsed: 耗时 (秒)
        neighbors_evaluated: 评估的邻居数
    """

    best_tree: PhyloTree
    best_score: float
    all_trees: list[PhyloTree] = field(default_factory=list)
    iterations: int = 0
    time_elapsed: float = 0.0
    neighbors_evaluated: int = 0

    @property
    def consensus_tree(self) -> PhyloTree | None:
        """如果有多棵等长树，返回严格一致性树"""
        if len(self.all_trees) <= 1:
            return None
        from .strict_consensus import StrictConsensusTree

        consensus_builder = StrictConsensusTree()
        return consensus_builder.build(self.all_trees)


class TreeOperation:
    """
    树变换操作的基类
    """

    def __init__(self, description: str = ""):
        self.description = description

    def apply(self, tree: PhyloTree) -> PhyloTree:
        """应用变换"""
        raise NotImplementedError

    def get_description(self) -> str:
        return self.description


@dataclass
class NNIOperation(TreeOperation):
    r"""
    NNI (最近邻居互换) 变换

    变换示意图:

        边 (X, Y) 两侧的子树互换:

        原始:      NNI结果1:     NNI结果2:
          X           X             X
         / \         / \           / \
        A   Y       A   C         A   D
           / \         / \           / \
          B   C       B   Y         B   Y
             / \         / \           / \
            D   E       D   E         C   D
                                       / \
                                      D   E
    """

    edge_node1: PhyloNode
    edge_node2: PhyloNode
    swap_option: int = 1  # 1 或 2

    def __init__(self, edge_node1: PhyloNode, edge_node2: PhyloNode, swap_option: int = 1):
        super().__init__(f"NNI: swap option {swap_option} on edge ({edge_node1.name}, {edge_node2.name})")
        self.edge_node1 = edge_node1
        self.edge_node2 = edge_node2
        self.swap_option = swap_option

    def apply(self, tree: PhyloTree) -> PhyloTree:
        """
        应用NNI变换

        NNI operates on an internal edge (node1, node2) where node1 is the
        parent of node2. We swap one child of node1 with one child of node2.

        For edge (X, Y) where X has children {A1, A2} and Y has children {B1, B2}:
            Option 1: X gets {A1, B1}, Y gets {A2, B2}  (swap A2 <-> B1)
            Option 2: X gets {A1, B2}, Y gets {A2, B1}  (swap A1 <-> B2)
        """
        if tree.root is None:
            raise ValueError("Tree has no root")

        # 深拷贝树
        new_root = self._deep_copy_tree(tree.root)

        # 找到新树中对应的节点
        node_map = self._build_node_map(tree.root, new_root)

        node1 = node_map.get(self.edge_node1)
        node2 = node_map.get(self.edge_node2)

        if node1 is None or node2 is None:
            raise ValueError("Node mapping failed")

        # NNI requires both endpoints to be internal nodes
        if node1.is_leaf or node2.is_leaf:
            raise ValueError("NNI requires both edge endpoints to be internal nodes")

        # Get children of both nodes
        a_children = list(node1.children)
        b_children = list(node2.children)

        if len(a_children) < 2 or len(b_children) < 2:
            raise ValueError("NNI requires nodes with at least 2 children")

        a1, a2 = a_children[0], a_children[1]
        b1, b2 = b_children[0], b_children[1]

        # Perform the NNI swap on node1 (parent) and node2 (child)
        if self.swap_option == 1:
            # Swap a2 with b1: node1 gets {a1, b1}, node2 gets {a2, b2}
            node1.children = [a1, b1]
            node2.children = [a2, b2]
            # Update parent references for swapped children
            b1.parent = node1
            a2.parent = node2
        else:
            # Swap a1 with b2: node1 gets {a1, b2}, node2 gets {a2, b1}
            node1.children = [a1, b2]
            node2.children = [a2, b1]
            # Update parent references for swapped children
            b2.parent = node1
            a1.parent = node2

        return PhyloTree(new_root)

    def _deep_copy_tree(self, root: PhyloNode) -> PhyloNode:
        """深度拷贝树"""

        def copy_node(node: PhyloNode, parent: PhyloNode | None) -> PhyloNode:
            new_node = PhyloNode(
                name=node.name,
                node_type=node.node_type,
                branch_length=node.branch_length,
                support=node.support,
                parent=parent,
            )
            for child in node.children:
                new_child = copy_node(child, new_node)
                new_node.children.append(new_child)
            return new_node

        return copy_node(root, None)

    def _build_node_map(self, old_root: PhyloNode, new_root: PhyloNode) -> dict[PhyloNode, PhyloNode]:
        """构建新旧节点映射"""
        mapping = {}

        for old_node in old_root.preorder_traverse():
            # 尝试通过名称和位置匹配
            new_node = self._find_matching_node(new_root, old_node)
            if new_node:
                mapping[old_node] = new_node

        return mapping

    def _find_matching_node(self, root: PhyloNode, target: PhyloNode) -> PhyloNode | None:
        """查找匹配的节点"""
        if root.name == target.name and len(root.children) == len(target.children):
            return root

        for child in root.children:
            result = self._find_matching_node(child, target)
            if result:
                return result

        return None

    def _find_common_parent(self, node1: PhyloNode, node2: PhyloNode) -> PhyloNode | None:
        """查找公共父节点"""
        ancestors1 = set(node1.get_path_to_root())
        node = node2

        while node is not None:
            if node in ancestors1:
                return node
            node = node.parent

        return None


@dataclass
class TBROperation(TreeOperation):
    """
    TBR (树二分与重连) 变换

    步骤:
        1. 选择一条内部边，将其移除，将树分为两部分
        2. 在每部分中选择一个节点作为重连点
        3. 创建新边连接两个重连点

    TBR比NNI产生更多的邻居，搜索更彻底。
    """

    cut_node1: PhyloNode
    cut_node2: PhyloNode
    reconnect_node1: PhyloNode | None = None
    reconnect_node2: PhyloNode | None = None

    def __init__(
        self,
        cut_node1: PhyloNode,
        cut_node2: PhyloNode,
        reconnect_node1: PhyloNode | None = None,
        reconnect_node2: PhyloNode | None = None,
    ):
        desc = f"TBR: cut ({cut_node1.name}, {cut_node2.name})"
        super().__init__(desc)
        self.cut_node1 = cut_node1
        self.cut_node2 = cut_node2
        self.reconnect_node1 = reconnect_node1
        self.reconnect_node2 = reconnect_node2

    def apply(self, tree: PhyloTree) -> PhyloTree:
        """
        应用TBR变换

        1. 深拷贝树
        2. 移除cut_node1和cut_node2之间的边，将树分为两棵子树
        3. 在子树1中选择reconnect_node1（默认为cut_node1的父节点）
        4. 在子树2中选择reconnect_node2（默认为cut_node2）
        5. 将子树2挂接到reconnect_node1上
        """
        if tree.root is None:
            raise ValueError("Tree has no root")

        # 深拷贝
        new_root = self._deep_copy(tree.root)
        node_map = self._build_map(tree.root, new_root)

        n1 = node_map.get(self.cut_node1)
        n2 = node_map.get(self.cut_node2)
        if n1 is None or n2 is None:
            raise ValueError("Node mapping failed")

        # 确保n2是n1的子节点
        if n2.parent is not n1:
            if n1.parent is n2:
                n1, n2 = n2, n1
            else:
                raise ValueError("cut_node1 and cut_node2 must be adjacent")

        # 选择重连点
        r1 = node_map.get(self.reconnect_node1) if self.reconnect_node1 else n1.parent
        r2 = node_map.get(self.reconnect_node2) if self.reconnect_node2 else n2

        if r1 is None:
            raise ValueError("reconnect_node1 not found in tree")

        # 如果r1就是n1（重连点就是切割点的父节点一侧），且r2就是n2，需要选择不同的重连点
        # 此时选择n1的另一个子节点作为重连点
        if r1 is n1 and r2 is n2:
            # 选择n1的其他子节点
            other_children = [c for c in n1.children if c is not n2]
            if other_children:
                r1 = other_children[0]

        # 执行TBR: 将n2子树从n1断开，挂接到r1上
        n1.children.remove(n2)
        n2.parent = None

        # 处理n1变为叶节点的情况
        if not n1.children and not n1.is_root:
            n1.node_type = NodeType.LEAF

        # 将n2子树挂接到r1
        r1.add_child(n2)

        return PhyloTree(new_root)

    def _deep_copy(self, root: PhyloNode) -> PhyloNode:
        """深拷贝树"""

        def copy_node(node: PhyloNode, parent: PhyloNode | None) -> PhyloNode:
            new_node = PhyloNode(
                name=node.name,
                node_type=node.node_type,
                branch_length=node.branch_length,
                support=node.support,
                parent=parent,
            )
            for child in node.children:
                new_node.children.append(copy_node(child, new_node))
            return new_node

        return copy_node(root, None)

    def _build_map(self, old: PhyloNode, new: PhyloNode) -> dict[PhyloNode, PhyloNode]:
        """构建新旧节点映射"""
        mapping = {}

        def walk(o: PhyloNode, n: PhyloNode):
            mapping[o] = n
            for oc, nc in zip(o.children, n.children, strict=False):
                walk(oc, nc)

        walk(old, new)
        return mapping


class HeuristicSearch:
    """
    启发式树搜索

    结合多种策略寻找最优或近似最优的系统发育树。

    搜索策略:
        1. 初始树构建 (NJ或随机)
        2. 局部搜索 (NNI/TBR)
        3. 重启策略 (多轮搜索)
        4. 模拟退火 (避免局部最优)
    """

    def __init__(
        self,
        algorithm: str = "parsimony",
        max_iterations: int = 1000,
        random_seed: int | None = None,
        nni_swap_probability: float = 0.7,
        acceptance_probability: float = 0.1,
        temperature: float = 1.0,
        cooling_rate: float = 0.95,
    ):
        """
        初始化启发式搜索

        Parameters:
            algorithm: 优化算法 ("parsimony" 或 "likelihood")
            max_iterations: 最大迭代次数
            random_seed: 随机种子
            nni_swap_probability: NNI交换概率 (vs TBR)
            acceptance_probability: 接受次优解的概率
            temperature: 初始温度 (模拟退火)
            cooling_rate: 冷却率
        """
        self._algorithm = algorithm
        self._max_iterations = max_iterations
        self._nni_prob = nni_swap_probability
        self._acceptance_prob = acceptance_probability
        self._temperature = temperature
        self._cooling_rate = cooling_rate

        if random_seed is not None:
            random.seed(random_seed)

        self._logger = logging.getLogger(f"{__name__}.HeuristicSearch")
        self._fitch = FitchAlgorithm()

        # 搜索状态
        self._current_tree: PhyloTree | None = None
        self._current_score: float = float("inf")
        self._best_tree: PhyloTree | None = None
        self._best_score: float = float("inf")
        self._iterations: int = 0
        self._neighbors_evaluated: int = 0

        # 找到的等长树
        self._optimal_trees: list[PhyloTree] = []
        self._optimal_score: float = float("inf")

    def search(
        self, leaf_names: list[str], sequences: dict[str, str], initial_tree: PhyloTree | None = None
    ) -> SearchResult:
        """
        执行启发式搜索

        Parameters:
            leaf_names: 叶节点名称列表
            sequences: 序列字典
            initial_tree: 初始树 (可选)

        Returns:
            SearchResult对象
        """
        import time

        start_time = time.time()

        # 初始化
        if initial_tree is None:
            self._current_tree = self._build_random_tree(leaf_names)
        else:
            self._current_tree = initial_tree

        self._current_score = self._evaluate_tree(self._current_tree, sequences)

        # 初始化最佳树
        self._best_tree = self._deep_copy_tree(self._current_tree)
        self._best_score = self._current_score
        self._optimal_score = self._current_score
        self._optimal_trees = [self._deep_copy_tree(self._current_tree)]

        self._logger.info(f"Starting heuristic search with initial score: {self._current_score}")

        # 主搜索循环
        for iteration in range(self._max_iterations):
            self._iterations = iteration + 1

            # 生成邻居
            neighbors = self._generate_neighbors(self._current_tree)

            # 评估邻居
            best_neighbor = None
            best_neighbor_score = float("inf")

            for neighbor in neighbors:
                self._neighbors_evaluated += 1
                score = self._evaluate_tree(neighbor, sequences)

                if score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = score

            # 决定是否接受邻居
            if self._should_accept(best_neighbor_score):
                self._current_tree = best_neighbor
                self._current_score = best_neighbor_score

                # 更新最佳
                if best_neighbor_score < self._best_score:
                    self._best_tree = self._deep_copy_tree(best_neighbor)
                    self._best_score = best_neighbor_score
                    self._optimal_score = best_neighbor_score
                    self._optimal_trees = [self._deep_copy_tree(best_neighbor)]
                    self._logger.info(f"Iteration {iteration}: New best score = {best_neighbor_score}")

                # 检查是否等长
                elif abs(best_neighbor_score - self._optimal_score) < 1e-10:
                    self._optimal_trees.append(self._deep_copy_tree(best_neighbor))

            # 冷却
            self._temperature *= self._cooling_rate

            # 检查收敛
            if self._temperature < 0.001:
                break

        elapsed = time.time() - start_time

        self._logger.info(
            f"Search complete: {self._iterations} iterations, "
            f"{self._neighbors_evaluated} neighbors evaluated, "
            f"best score = {self._best_score}, "
            f"elapsed = {elapsed:.2f}s"
        )

        return SearchResult(
            best_tree=self._best_tree,
            best_score=self._best_score,
            all_trees=self._optimal_trees,
            iterations=self._iterations,
            time_elapsed=elapsed,
            neighbors_evaluated=self._neighbors_evaluated,
        )

    def _evaluate_tree(self, tree: PhyloTree, sequences: dict[str, str]) -> float:
        """
        评估树分数

        Parameters:
            tree: 要评估的树
            sequences: 序列字典

        Returns:
            分数 (越小越好)
        """
        if self._algorithm == "parsimony":
            result = self._fitch.compute(tree, sequences)
            return float(result.tree_length)
        else:
            raise ValueError(f"Unknown algorithm: {self._algorithm}")

    def _generate_neighbors(self, tree: PhyloTree) -> list[PhyloTree]:
        """
        生成邻居树（NNI + TBR）

        Parameters:
            tree: 当前树

        Returns:
            邻居树列表
        """
        neighbors = []

        if tree.root is None:
            return neighbors

        # 收集所有内部边
        internal_edges = self._collect_internal_edges(tree.root)

        for edge in internal_edges:
            node1, node2 = edge

            # NNI邻居
            for option in [1, 2]:
                try:
                    nni = NNIOperation(node1, node2, option)
                    neighbor = nni.apply(tree)
                    neighbors.append(neighbor)
                except (ValueError, AttributeError):
                    pass

            # TBR邻居（以概率决定是否生成，避免搜索空间过大）
            if random.random() < self._nni_prob:
                continue

            # 收集node2子树中的节点作为重连点候选
            subtree_nodes = [n for n in node2.get_all_nodes() if not n.is_leaf]
            parent_candidates = [n for n in node1.get_all_nodes() if not n.is_leaf and n is not node2]

            for r2 in subtree_nodes[:3]:  # 限制候选数
                for r1 in parent_candidates[:3]:
                    try:
                        tbr = TBROperation(node1, node2, r1, r2)
                        neighbor = tbr.apply(tree)
                        neighbors.append(neighbor)
                    except (ValueError, AttributeError):
                        pass

        return neighbors

    def _collect_internal_edges(self, node: PhyloNode) -> list[tuple[PhyloNode, PhyloNode]]:
        """
        收集所有内部边

        Returns:
            [(节点1, 节点2), ...] 边列表
        """
        edges = []

        def dfs(n: PhyloNode) -> None:
            if n.is_leaf:
                return

            # 收集与子节点的边
            for child in n.children:
                edges.append((n, child))
                dfs(child)

        dfs(node)
        return edges

    def _should_accept(self, new_score: float) -> bool:
        """
        决定是否接受新解

        使用模拟退火策略:
            P(accept) = exp(-ΔE/T) if ΔE > 0
                      = 1 otherwise

        Parameters:
            new_score: 新解的分数

        Returns:
            是否接受
        """
        delta = new_score - self._current_score

        if delta <= 0:
            return True

        # 模拟退火: P(accept) = exp(-delta / T)
        if self._temperature > 1e-10:
            probability = min(1.0, np.exp(-delta / self._temperature))
        else:
            probability = 0.0

        return random.random() < probability

    def _build_random_tree(self, leaf_names: list[str]) -> PhyloTree:
        """
        构建随机初始树

        Parameters:
            leaf_names: 叶节点名称

        Returns:
            随机树
        """
        # 创建星形树，然后随机合并
        nodes = [PhyloNode(name=name, node_type=NodeType.LEAF) for name in leaf_names]

        while len(nodes) > 1:
            # 随机选择两个节点合并
            i, j = random.sample(range(len(nodes)), 2)
            node1, node2 = nodes[i], nodes[j]

            # 创建新内部节点
            new_node = PhyloNode(name=f"internal_{len(nodes)}", node_type=NodeType.INTERNAL)

            # 添加子节点
            new_node.add_child(node1)
            new_node.add_child(node2)

            # 更新节点列表
            nodes = [n for k, n in enumerate(nodes) if k not in (i, j)]
            nodes.append(new_node)

        tree = PhyloTree(root=nodes[0])
        return tree

    def _deep_copy_tree(self, tree: PhyloTree) -> PhyloTree:
        """深拷贝树"""
        if tree.root is None:
            return PhyloTree()

        def copy_node(node: PhyloNode, parent: PhyloNode | None) -> PhyloNode:
            new_node = PhyloNode(
                name=node.name,
                node_type=node.node_type,
                branch_length=node.branch_length,
                support=node.support,
                parent=parent,
            )
            for child in node.children:
                new_child = copy_node(child, new_node)
                new_node.children.append(new_child)
            return new_node

        new_root = copy_node(tree.root, None)
        return PhyloTree(root=new_root, name=tree.name)


def run_heuristic_search(
    leaf_names: list[str], sequences: dict[str, str], algorithm: str = "parsimony", max_iterations: int = 1000
) -> SearchResult:
    """
    运行启发式搜索的便捷函数

    Parameters:
        leaf_names: 叶节点名称
        sequences: 序列字典
        algorithm: 算法类型
        max_iterations: 最大迭代

    Returns:
        SearchResult对象
    """
    search = HeuristicSearch(algorithm=algorithm, max_iterations=max_iterations, random_seed=42)
    return search.search(leaf_names, sequences)
