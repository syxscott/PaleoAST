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

        NNI operates on an internal edge (node1, node2) where node2 is a child
        of node1.  Let ``sib`` be the child of node1 that is *not* node2 (the
        other side of the edge) and ``b1, b2`` two children of node2:

            Option 1: exchange ``sib`` with ``b1``
            Option 2: exchange ``sib`` with ``b2``

        旧实现在 swap_option == 2 时写了 ``a1.parent = node2`` (a1 仍是 node1
        的子节点)，使父子图成环；并且它假定 node2 恰好位于 ``node1.children[0]``，
        否则会把 node2 从 node1 的子节点列表中整体丢掉。现在按身份定位 node2，
        只对参与交换的两个子树做原位替换。
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

        # node2 必须是 node1 的子节点 (边存在于父子之间)
        if node2 not in node1.children:
            raise ValueError("NNI requires edge_node2 to be a child of edge_node1")

        # ``sib``: node1 中除 node2 之外的另一侧子树
        sib_candidates = [c for c in node1.children if c is not node2]
        b_children = [c for c in node2.children if c is not node1]

        if not sib_candidates or len(b_children) < 2:
            raise ValueError("NNI requires nodes with at least 2 children")

        sib = sib_candidates[0]
        swap_index = self.swap_option - 1
        if swap_index not in (0, 1):
            raise ValueError(f"NNI swap_option must be 1 or 2, got {self.swap_option}")
        moved = b_children[swap_index]

        # 原位交换: node1 失去 sib 得到 moved; node2 失去 moved 得到 sib
        i = node1.children.index(sib)
        node1.children[i] = moved
        moved.parent = node1

        j = node2.children.index(moved)
        node2.children[j] = sib
        sib.parent = node2

        return PhyloTree(new_root)

    def _deep_copy_tree(self, root: PhyloNode) -> PhyloNode:
        """深度拷贝树 (保留 data / label / metadata)"""

        def copy_node(node: PhyloNode, parent: PhyloNode | None) -> PhyloNode:
            new_node = PhyloNode(
                name=node.name,
                node_type=node.node_type,
                branch_length=node.branch_length,
                support=node.support,
                data=node.data,
                label=node.label,
                metadata=dict(node.metadata),
                parent=parent,
            )
            for child in node.children:
                new_child = copy_node(child, new_node)
                new_node.children.append(new_child)
            return new_node

        return copy_node(root, None)

    def _build_node_map(self, old_root: PhyloNode, new_root: PhyloNode) -> dict[PhyloNode, PhyloNode]:
        """构建新旧节点映射 (按先序位置一一对应)。

        旧实现按 (name, 子节点数) 匹配: Newick 树的内部节点通常全部
        空名, 所有内部节点都映射到第一个命中者 (根), NNI 变换因此
        会输出重复子节点的损坏树 (如 ((C,D),(C,D)))。
        新旧树来自同一棵树的同一变换, 先序位置一一对应。"""
        mapping = {}
        old_nodes = list(old_root.preorder_traverse())
        new_nodes = list(new_root.preorder_traverse())
        if len(old_nodes) != len(new_nodes):
            return mapping
        for old_node, new_node in zip(old_nodes, new_nodes):
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
        应用 TBR 变换。

        真正的 TBR 需要**两条切边、两条接边**：

        1. 切掉 ``cut_node1 -> cut_node2``，树分为 A（含 cut_node1）与 B（cut_node2 子树）
        2. 在 A 中再切一条边 ``r1 -> c1``（c1 不得是通往 cut_node1 的那条）
        3. 交叉重接：``cut_node1`` 接管 ``c1``，``cut_node2`` 接管 ``r1``

        旧实现只做了一次摘除加一次挂接（``n1.children.remove(n2)`` 后
        ``r1.add_child(n2)``），那是 **SPR**，而且从不把 n1 那一侧接回去：
        n1 若原本是二叉节点会剩一个子节点变成一元节点，r1 则变成三叉。
        实测 12/12 个生成的邻居都非二叉，而且 ``n1.node_type = NodeType.LEAF``
        会把仍有子树的内部节点标成叶端，使 ``get_leaves()`` 丢掉真正的类元，
        Fitch 随后给这些假端点空状态集，整棵树被记为 0 步。

        最后统一走 :meth:`_normalise_binary`，把一元节点收缩、多叉节点合并，
        因此返回的树保证是二叉且叶集合与输入完全一致。

        Parameters
        ----------
        reconnect_node1
            A 侧的重接点（r1）。默认取 ``cut_node1`` 的父节点。
        reconnect_node2
            保留以兼容旧调用。真正的 TBR 里 B 侧的接收点恒为 cut_node2 本身，
            该参数不再参与重接（它以前只在一个守卫分支里被读一次）。
        """
        if tree.root is None:
            raise ValueError("Tree has no root")

        new_root = self._deep_copy(tree.root)
        node_map = self._build_map(tree.root, new_root)

        n1 = node_map.get(self.cut_node1)
        n2 = node_map.get(self.cut_node2)
        if n1 is None or n2 is None:
            raise ValueError("Node mapping failed")

        if n2.parent is not n1:
            if n1.parent is n2:
                n1, n2 = n2, n1
            else:
                raise ValueError("cut_node1 and cut_node2 must be adjacent")

        r1 = node_map.get(self.reconnect_node1) if self.reconnect_node1 else n1.parent
        if r1 is None:
            raise ValueError("reconnect_node1 not found in tree")

        # 第二次切边必须在 A 侧，且切掉的那个子节点不能是通往 n1 的那条，
        # 否则 A 会被切成两半而 n1 不在任何一半里。
        if r1 is n1:
            raise ValueError("TBR needs a reconnect node distinct from the cut parent")

        on_path_to_n1 = None
        walk = r1
        while walk is not n1:
            walk = walk.parent
            if walk is None:
                raise ValueError("reconnect_node1 is not an ancestor of cut_node1")
            on_path_to_n1 = walk

        c1 = next((c for c in r1.children if c is not on_path_to_n1), None)
        if c1 is None:
            raise ValueError("reconnect_node1 has no second child to cut")

        if r1 in set(n2.get_subtree_nodes()):
            raise ValueError(
                f"TBR reconnect node '{r1.name}' lies inside the subtree rooted at "
                f"'{n2.name}'; regrafting there would create a cycle"
            )

        # --- 两条切边 ---
        n1_len = n2.branch_length
        n1.children.remove(n2)
        n2.parent = None

        r1_len = c1.branch_length
        r1.children.remove(c1)
        c1.parent = None

        # --- 两条接边：交叉重接被切开的两条边 ---
        # 切掉的是 (n1, n2) 与 (r1, c1)，必须交叉接回 (n1, c1) 与 (r1, n2)。
        # 早期版本写成 ``n2.add_child(r1)``，方向反了：n2 子树被整体丢弃
        # （6 个邻居的叶集合全部从 6 个 taxon 掉到 3-4 个）。
        # 二叉输入下这四个节点的 arity 全部守恒，归一化通常无事可做。
        n1.add_child(c1)
        c1.branch_length = r1_len

        r1.add_child(n2)
        n2.branch_length = n1_len

        new_root = self._normalise_binary(new_root)
        return PhyloTree(new_root)

    @classmethod
    def _normalise_binary(cls, root: PhyloNode) -> PhyloNode:
        """Return ``root``'s subtree as a strictly binary tree.

        A TBR on a binary input leaves a unary node where a pendant edge was
        reattached and a trifurcation where the receiving node already had two
        children. Both are removed here so the neighbour honours the invariant
        every downstream consumer assumes:

        * a node with a single child is suppressed -- the child takes its place
          and inherits the suppressed node's branch length. If the suppressed
          node is the root, the child BECOMES the root and is returned.
        * a node with more than two children is resolved by merging the first
          two, which preserves the subtree and the total path length.

        Leaf sets are never altered, so the caller can rely on the neighbour
        containing exactly the input taxa.
        """
        while True:
            unarity = None
            excess = None
            for node in root.preorder_traverse():
                k = len(node.children)
                if k == 1:
                    unarity = node
                    break
                if k > 2:
                    excess = node
                    break

            if excess is not None:
                # Resolve the polytomy by grouping two children under a NEW
                # internal node. Grafted onto an existing child instead, the
                # extra child just moves the polytomy one level down and the
                # loop never terminates; and grafting onto a LEAF gives that
                # leaf a child, which the next pass "suppresses" as a unary
                # node -- deleting the taxon outright (observed: D vanishing
                # from a 6-taxon tree).
                kids = list(excess.children)
                m = PhyloNode(name="", node_type=NodeType.INTERNAL, branch_length=0.0)
                a, b = kids[0], kids[1]
                excess.remove_child(a)
                excess.remove_child(b)
                a.parent = None
                b.parent = None
                m.add_child(a)
                m.add_child(b)
                excess.add_child(m)
                continue

            if unarity is None:
                return root

            only = unarity.children[0]
            parent = unarity.parent
            if parent is None:
                # The suppressed node is the root: its only child must become
                # the new root. Re-attaching the child to the same node (as an
                # earlier attempt did) changes nothing while claiming
                # progress, so the loop never terminated and the tree was
                # returned still holding a unary root.
                unarity.remove_child(only)
                only.parent = None
                only.branch_length = unarity.branch_length
                return only

            parent.remove_child(unarity)
            unarity.remove_child(only)
            only.parent = None
            only.branch_length = unarity.branch_length
            parent.add_child(only)

    def _deep_copy(self, root: PhyloNode) -> PhyloNode:
        """深拷贝树 (保留 data / label / metadata)"""

        def copy_node(node: PhyloNode, parent: PhyloNode | None) -> PhyloNode:
            new_node = PhyloNode(
                name=node.name,
                node_type=node.node_type,
                branch_length=node.branch_length,
                support=node.support,
                data=node.data,
                label=node.label,
                metadata=dict(node.metadata),
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
            algorithm: 优化算法；当前仅实现 ``"parsimony"`` (Fitch)。
                似然法尚未实现，传入其他值时评估阶段会抛 ValueError。
            max_iterations: 最大迭代次数
            random_seed: 随机种子 (仅作用于本实例自己的 Random 发生器，
                不再污染全局 ``random`` 状态)
            nni_swap_probability: 保留原参数名以兼容既有调用。其语义是
                "跳过 TBR 候选生成" 的概率补：TBR 生成概率
                ``tbr_probability = 1 - nni_swap_probability``。NNI 邻居
                (交换选项 1 与 2) 总是生成，不受该参数影响。
            acceptance_probability: 接受次优解的概率 (当前未被使用，接受
                概率由模拟退火温度决定)
            temperature: 初始温度 (模拟退火)
            cooling_rate: 冷却率
        """
        self._algorithm = algorithm
        self._max_iterations = max_iterations
        self._initial_temperature = temperature
        self._nni_prob = nni_swap_probability
        self._tbr_prob = 1.0 - nni_swap_probability
        self._acceptance_prob = acceptance_probability
        self._temperature = temperature
        self._cooling_rate = cooling_rate

        # 实例级随机源：random.seed(...) 会改动全局状态并影响进程内其他
        # 使用者，因此改为持有自己的 random.Random(seed) 实例。
        self._rng = random.Random(random_seed)

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

        if initial_tree is None and len(leaf_names) < 2:
            raise ValueError(
                f"Tree search needs at least 2 taxa to build a starting tree, got {len(leaf_names)}"
            )

        # 每次 search() 都从初始状态开始：温度若不在开头重置，第二次调用会
        # 直接以已冷却的温度运行（迭代在第 1 步就被 < 0.001 的收敛判据中断），
        # 计数器也会跨调用累加而使 SearchResult 报告失真。
        self._temperature = self._initial_temperature
        self._iterations = 0
        self._neighbors_evaluated = 0

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
            raise ValueError(
                f"Unknown algorithm: {self._algorithm} (only 'parsimony' is implemented; "
                "maximum likelihood tree search is not available in this build)"
            )

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

        # 全树节点（用于 TBR 重连点候选）
        all_nodes = tree.root.get_all_nodes()

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

            # TBR 邻居（以 _tbr_prob 决定是否生成，避免搜索空间过大）
            if self._rng.random() >= self._tbr_prob:
                continue

            # 挂载点候选必须是"全树节点 - node2 子树(含后代)"，否则会把
            # node2 挂回自己内部形成环。
            #
            # 旧实现还对 node2 子树内的非叶节点取 r2 候选并逐个尝试，但 TBR
            # 的第二条切边由 r1 自身决定，r2 从未参与重接——同一 (r1) 被重复
            # 生成 3 次，12 个邻居里只有 2-3 个不同拓扑。现在只沿 r1 变化。
            moved = set(node2.get_subtree_nodes())
            parent_candidates = [n for n in all_nodes if n not in moved and not n.is_leaf]

            for r1 in parent_candidates[:3]:  # 限制候选数
                if r1 is node1:
                    continue
                try:
                    tbr = TBROperation(node1, node2, r1)
                    neighbor = tbr.apply(tree)
                    neighbors.append(neighbor)
                except (ValueError, AttributeError):
                    pass

        # 去重：TBR 的 (r1, c1) 选择可能收敛到同一拓扑（例如 r1 与 c1 在不同
        # 轮次被选为同一对），保留首次出现的那棵。
        seen: set[str] = set()
        unique: list[PhyloTree] = []
        for t in neighbors:
            try:
                key = t.to_newick()
            except Exception:
                unique.append(t)
                continue
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

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

        return self._rng.random() < probability

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

        merge_step = 0
        while len(nodes) > 1:
            # 随机选择两个节点合并
            i, j = self._rng.sample(range(len(nodes)), 2)
            node1, node2 = nodes[i], nodes[j]

            # 创建新内部节点 (单调计数, 名称唯一)
            merge_step += 1
            new_node = PhyloNode(name=f"_internal_{merge_step}", node_type=NodeType.INTERNAL)

            # 添加子节点
            new_node.add_child(node1)
            new_node.add_child(node2)

            # 更新节点列表
            nodes = [n for k, n in enumerate(nodes) if k not in (i, j)]
            nodes.append(new_node)

        tree = PhyloTree(root=nodes[0])
        return tree

    def _deep_copy_tree(self, tree: PhyloTree) -> PhyloTree:
        """深拷贝树 (保留 data / label / metadata)"""
        if tree.root is None:
            return PhyloTree()

        def copy_node(node: PhyloNode, parent: PhyloNode | None) -> PhyloNode:
            new_node = PhyloNode(
                name=node.name,
                node_type=node.node_type,
                branch_length=node.branch_length,
                support=node.support,
                data=node.data,
                label=node.label,
                metadata=dict(node.metadata),
                parent=parent,
            )
            for child in node.children:
                new_child = copy_node(child, new_node)
                new_node.children.append(new_child)
            return new_node

        new_root = copy_node(tree.root, None)
        return PhyloTree(root=new_root, name=tree.name, metadata=dict(tree.metadata))


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
