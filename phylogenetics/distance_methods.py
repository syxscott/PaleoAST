"""
================================================================================
PaleoAST Phylogenetics - Distance Methods
================================================================================

本模块实现距离法系统发育推断算法。

数学理论:
==============================================================================

1. 距离矩阵
--------------------
给定 n 个分类单元，距离矩阵 D = [d_ij] 满足:
- d_ii = 0
- d_ij = d_ji (对称性)
- d_ij ≥ 0 (非负性)
- d_ij ≤ d_ik + d_kj (三角不等式)

2. UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
--------------------------------------------------------------------------------
层次聚类方法。

算法:
    1. 初始化 n 个簇，每簇一个分类单元
    2. 计算所有簇对之间的距离
    3. 合并距离最小的两个簇
    4. 新簇到其他簇的距离 = 旧距离的加权平均
    5. 重复直到只剩一个簇

数学公式:
    d_{(ij),k} = (n_i × d_ik + n_j × d_jk) / (n_i + n_j)

其中 n_i, n_j 是簇 i, j 的成员数。

3. Neighbor Joining (NJ)
--------------------------------------------------------------------------------
由Saitou和Nei (1987)提出。

核心思想: 最小化总枝长

算法:
    1. 计算Q矩阵: Q_ij = (n-2) × d_ij - Σ_k d_ik - Σ_k d_jk
    2. 找到最小Q值对应的节点对(i,j)
    3. 计算新节点u与i,j的距离:
       d_u,i = 0.5 × d_ij + 0.5 × (Σ_k d_ik - Σ_k d_jk) / (n-2)
    4. 更新距离矩阵
    5. 重复直到只剩3个节点

4. 最小二乘法
--------------------------------------------------------------------------------
寻找使得 Σ_ij w_ij (D_ij - d_ij)² 最小的树

其中 D_ij 是观察距离，d_ij 是树距离。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .tree import NodeType, PhyloNode, PhyloTree

logger = logging.getLogger(__name__)


def _get_distance(dist: dict, t1: str, t2: str) -> float:
    """
    Safely get distance between two taxa, raising error if missing.

    Parameters:
        dist: Distance dictionary
        t1: First taxon
        t2: Second taxon

    Returns:
        Distance value

    Raises:
        KeyError: If distance is not found in matrix
    """
    if t1 == t2:
        return 0.0
    key = (t1, t2)
    reverse_key = (t2, t1)
    if key in dist:
        return dist[key]
    if reverse_key in dist:
        return dist[reverse_key]
    raise KeyError(f"Missing distance between '{t1}' and '{t2}' in distance matrix")


@dataclass
class DistanceMatrix:
    """
    距离矩阵

    属性:
        taxa: 分类单元名称列表
        distances: {(名称1, 名称2): 距离} 字典
        is_metric: 是否满足距离度量条件
    """

    taxa: list[str]
    distances: dict[tuple[str, str], float]

    def __post_init__(self):
        """验证和规范化"""
        # 确保对称性
        for (t1, t2), d in list(self.distances.items()):
            if t1 != t2:
                self.distances[(t2, t1)] = d

    def get_distance(self, taxon1: str, taxon2: str) -> float:
        """
        获取两个分类单元间的距离

        Parameters:
            taxon1: 第一个分类单元
            taxon2: 第二个分类单元

        Returns:
            距离值
        """
        if taxon1 == taxon2:
            return 0.0

        key = (taxon1, taxon2)
        reverse_key = (taxon2, taxon1)

        if key in self.distances:
            return self.distances[key]
        if reverse_key in self.distances:
            return self.distances[reverse_key]

        raise KeyError(f"No distance found between {taxon1} and {taxon2}")

    def to_matrix(self) -> list[list[float]]:
        """
        转换为二维列表形式

        Returns:
            n×n 距离矩阵
        """
        n = len(self.taxa)
        matrix = [[0.0] * n for _ in range(n)]

        for i, t1 in enumerate(self.taxa):
            for j, t2 in enumerate(self.taxa):
                matrix[i][j] = self.get_distance(t1, t2)

        return matrix

    @classmethod
    def from_array(cls, matrix: np.ndarray, labels: list[str]) -> DistanceMatrix:
        """Create from a numpy distance matrix and label list."""
        matrix = np.asarray(matrix, dtype=float)
        n = len(labels)
        distances = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[(labels[i], labels[j])] = float(matrix[i, j])
        return cls(taxa=list(labels), distances=distances)

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> DistanceMatrix:
        """
        从嵌套字典创建

        Parameters:
            data: {分类单元: {分类单元: 距离}}

        Returns:
            DistanceMatrix对象
        """
        taxa = list(data.keys())
        distances = {}

        for t1, inner in data.items():
            for t2, d in inner.items():
                distances[(t1, t2)] = d

        return cls(taxa=taxa, distances=distances)

    @classmethod
    def from_sequences(cls, sequences: dict[str, str], model: str = "p-distance") -> DistanceMatrix:
        """
        从序列计算距离矩阵

        Parameters:
            sequences: {名称: 序列}
            model: 距离模型

        Returns:
            DistanceMatrix对象
        """
        taxa = list(sequences.keys())
        n = len(taxa)
        distances = {}

        for i in range(n):
            for j in range(i, n):
                t1, t2 = taxa[i], taxa[j]
                seq1, seq2 = sequences[t1], sequences[t2]

                if len(seq1) != len(seq2):
                    raise ValueError(f"Sequence lengths mismatch: {t1}={len(seq1)}, {t2}={len(seq2)}")

                # 计算p距离
                diffs = sum(1 for a, b in zip(seq1, seq2, strict=False) if a != b)
                p_dist = diffs / len(seq1)

                # 简单转换 (可扩展更多模型)
                distance = p_dist

                distances[(t1, t2)] = distance
                if i != j:
                    distances[(t2, t1)] = distance

        return cls(taxa=taxa, distances=distances)


class UPGMA:
    """
    UPGMA (Unweighted Pair Group Method with Arithmetic Mean)

    层次聚类方法，假设分子钟假设（所有分支以相同速率进化）。

    使用示例:
        >>> from scipy.spatial.distance import pdist
        >>> # 假设有距离矩阵
        >>> dist = DistanceMatrix.from_dict({
        ...     'A': {'A': 0, 'B': 5, 'C': 7, 'D': 10},
        ...     'B': {'A': 5, 'B': 0, 'C': 6, 'D': 9},
        ...     'C': {'A': 7, 'B': 6, 'C': 0, 'D': 8},
        ...     'D': {'A': 10, 'B': 9, 'C': 8, 'D': 0},
        ... })
        >>> upgma = UPGMA()
        >>> tree = upgma.build(dist)
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.UPGMA")

    def build(self, distance_matrix: DistanceMatrix) -> PhyloTree:
        """
        使用UPGMA构建系统发育树

        Parameters:
            distance_matrix: 距离矩阵

        Returns:
            PhyloTree对象
        """
        taxa = distance_matrix.taxa.copy()
        n = len(taxa)

        if n == 0:
            return PhyloTree()
        if n == 1:
            root = PhyloNode(name=taxa[0], node_type=NodeType.LEAF)
            return PhyloTree(root=root)

        # 初始化簇
        # 每个簇: {members: Set[str], node: PhyloNode, size: int}
        clusters = {}
        for taxon in taxa:
            clusters[taxon] = {"members": {taxon}, "node": PhyloNode(name=taxon, node_type=NodeType.LEAF), "size": 1}

        # 复制距离矩阵用于操作
        dist = {}
        for key, val in distance_matrix.distances.items():
            dist[key] = val

        # 层次聚类
        active_taxa = set(taxa)

        while len(active_taxa) > 1:
            # 找到最近的簇对
            min_dist = float("inf")
            closest_pair = None

            active_list = list(active_taxa)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    t1, t2 = active_list[i], active_list[j]
                    d = _get_distance(dist, t1, t2)

                    if d < min_dist:
                        min_dist = d
                        closest_pair = (t1, t2)

            if closest_pair is None:
                break

            c1, c2 = closest_pair

            # 计算合并高度 (枝长)
            height = min_dist / 2.0

            # 创建新簇
            cluster1 = clusters[c1]
            cluster2 = clusters[c2]
            new_members = cluster1["members"] | cluster2["members"]
            new_size = cluster1["size"] + cluster2["size"]

            # 创建新内部节点
            new_name = f"cluster_{len(new_members)}"
            new_node = PhyloNode(name=new_name, node_type=NodeType.INTERNAL)

            # 设置枝长 = 合并高度 - 子节点已有高度
            # 标准UPGMA公式: branch_length = height - child_height
            # 使用max(0, ...)确保非负枝长，处理非超度量数据
            h1 = cluster1.get("height", 0.0)
            h2 = cluster2.get("height", 0.0)
            cluster1["node"].branch_length = max(0.0, height - h1)
            cluster2["node"].branch_length = max(0.0, height - h2)

            # 连接子节点
            new_node.add_child(cluster1["node"])
            new_node.add_child(cluster2["node"])

            # 存储新簇
            clusters[new_name] = {"members": new_members, "node": new_node, "size": new_size, "height": height}

            # 从活跃集合移除旧簇
            active_taxa.remove(c1)
            active_taxa.remove(c2)
            active_taxa.add(new_name)

            # 计算新簇到其他簇的距离
            for other in active_taxa:
                if other == new_name:
                    continue

                d1 = _get_distance(dist, c1, other)
                d2 = _get_distance(dist, c2, other)

                # 加权平均
                new_dist = (cluster1["size"] * d1 + cluster2["size"] * d2) / new_size
                dist[(new_name, other)] = new_dist
                dist[(other, new_name)] = new_dist

        # 获取根节点
        if not active_taxa:
            return PhyloTree()

        root_name = next(iter(active_taxa))
        root_node = clusters[root_name]["node"]

        self._logger.info(f"UPGMA tree built with {n} taxa")

        return PhyloTree(root=root_node)


class NeighborJoining:
    """
    Neighbor Joining (NJ) 算法

    由Saitou和Nei (1987)提出，不假设分子钟假设。
    在每步选择最小化总枝长的节点对。

    算法:
        1. 计算Q矩阵: Q_ij = (n-2) × d_ij - Σ_k d_ik - Σ_k d_jk
        2. 找到最小Q值对应的(i,j)
        3. 创建新节点u，计算距离
        4. 更新距离矩阵
        5. 重复直到只剩3个节点
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.NeighborJoining")

    def build(self, distance_matrix: DistanceMatrix) -> PhyloTree:
        """
        使用Neighbor Joining构建系统发育树

        Parameters:
            distance_matrix: 距离矩阵

        Returns:
            PhyloTree对象
        """
        taxa = distance_matrix.taxa.copy()
        n = len(taxa)

        if n == 0:
            return PhyloTree()
        if n == 1:
            root = PhyloNode(name=taxa[0], node_type=NodeType.LEAF)
            return PhyloTree(root=root)
        if n == 2:
            # 简单情况
            root = PhyloNode(name="root", node_type=NodeType.INTERNAL)
            root.add_child(PhyloNode(name=taxa[0], node_type=NodeType.LEAF))
            root.add_child(PhyloNode(name=taxa[1], node_type=NodeType.LEAF))
            d = distance_matrix.get_distance(taxa[0], taxa[1]) / 2
            root.children[0].branch_length = d
            root.children[1].branch_length = d
            return PhyloTree(root=root)

        # 工作数据结构
        active = set(taxa)

        # 创建节点对象
        nodes: dict[str, PhyloNode] = {}
        for taxon in taxa:
            nodes[taxon] = PhyloNode(name=taxon, node_type=NodeType.LEAF)

        # 距离矩阵工作副本
        dist = {}
        for key, val in distance_matrix.distances.items():
            dist[key] = val

        # 内部节点计数器
        internal_count = 0

        while len(active) > 3:
            m = len(active)

            # 计算Q矩阵并找到最小值
            active_list = list(active)
            min_q = float("inf")
            min_pair = None

            # 预计算每行的距离和
            row_sums: dict[str, float] = {}
            for t1 in active:
                total = 0.0
                for t2 in active:
                    if t1 != t2:
                        total += _get_distance(dist, t1, t2)
                row_sums[t1] = total

            # 计算Q
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    t1, t2 = active_list[i], active_list[j]
                    d_ij = _get_distance(dist, t1, t2)

                    q_ij = (m - 2) * d_ij - row_sums[t1] - row_sums[t2]

                    if q_ij < min_q:
                        min_q = q_ij
                        min_pair = (t1, t2)

            if min_pair is None:
                break

            i, j = min_pair

            # 计算到新节点的距离
            d_ij = _get_distance(dist, i, j)
            sum_i = row_sums[i]
            sum_j = row_sums[j]

            d_i_u = 0.5 * d_ij + 0.5 * (sum_i - sum_j) / (m - 2)
            d_j_u = 0.5 * d_ij + 0.5 * (sum_j - sum_i) / (m - 2)

            # 创建新内部节点
            internal_count += 1
            u_name = f"internal_{internal_count}"
            u_node = PhyloNode(name=u_name, node_type=NodeType.INTERNAL)

            # 设置枝长并连接
            if d_i_u < 0:
                self._logger.warning(
                    f"Negative branch length {d_i_u:.6f} for node '{i}' in NJ. "
                    f"This indicates non-metric distances. Setting to 0.0."
                )
                d_i_u = 0.0
            if d_j_u < 0:
                self._logger.warning(
                    f"Negative branch length {d_j_u:.6f} for node '{j}' in NJ. "
                    f"This indicates non-metric distances. Setting to 0.0."
                )
                d_j_u = 0.0
            nodes[i].branch_length = d_i_u
            nodes[j].branch_length = d_j_u
            u_node.add_child(nodes[i])
            u_node.add_child(nodes[j])

            nodes[u_name] = u_node

            # 更新活跃集合
            active.remove(i)
            active.remove(j)
            active.add(u_name)

            # 更新距离矩阵
            for k in active:
                if k == u_name:
                    continue

                d_i_k = _get_distance(dist, i, k)
                d_j_k = _get_distance(dist, j, k)

                d_i_j = _get_distance(dist, i, j)
                new_dist = 0.5 * (d_i_k + d_j_k - d_i_j)
                if new_dist < 0:
                    self._logger.warning(
                        f"Negative distance computed for {u_name}-{k}: {new_dist:.6f} (triangle inequality violated). "
                        f"Setting to 0.0. Consider using metric distances for NJ."
                    )
                    new_dist = 0.0
                dist[(u_name, k)] = new_dist
                dist[(k, u_name)] = new_dist

        # 处理最后3个节点
        final_nodes = list(active)
        if len(final_nodes) == 3:
            # 形成三叉树
            root = PhyloNode(name="root", node_type=NodeType.INTERNAL)

            for node_name in final_nodes:
                root.add_child(nodes[node_name])

            # 调整枝长 (NJ final 3-node formula)
            d_01 = _get_distance(dist, final_nodes[0], final_nodes[1])
            d_02 = _get_distance(dist, final_nodes[0], final_nodes[2])
            d_12 = _get_distance(dist, final_nodes[1], final_nodes[2])

            bl0 = (d_01 + d_02 - d_12) / 2.0
            bl1 = (d_01 + d_12 - d_02) / 2.0
            bl2 = (d_02 + d_12 - d_01) / 2.0

            # Warn if any branch length is negative (indicates non-metric data)
            for idx, bl in enumerate([bl0, bl1, bl2]):
                if bl < 0:
                    self._logger.warning(
                        f"Negative branch length {bl:.6f} for node {final_nodes[idx]} in final 3-node adjustment. "
                        f"This indicates non-metric distance matrix."
                    )

            root.children[0].branch_length = max(0.0, bl0)
            root.children[1].branch_length = max(0.0, bl1)
            root.children[2].branch_length = max(0.0, bl2)

        elif len(final_nodes) == 2:
            root = PhyloNode(name="root", node_type=NodeType.INTERNAL)
            for node_name in final_nodes:
                root.add_child(nodes[node_name])
        else:
            root = nodes[final_nodes[0]]

        self._logger.info(f"Neighbor Joining tree built with {n} taxa")

        return PhyloTree(root=root)


def build_upgma_tree(distance_matrix: DistanceMatrix) -> PhyloTree:
    """
    使用UPGMA构建树的便捷函数

    Parameters:
        distance_matrix: 距离矩阵

    Returns:
        PhyloTree对象
    """
    upgma = UPGMA()
    return upgma.build(distance_matrix)


def build_nj_tree(distance_matrix: DistanceMatrix) -> PhyloTree:
    """
    使用Neighbor Joining构建树的便捷函数

    Parameters:
        distance_matrix: 距离矩阵

    Returns:
        PhyloTree对象
    """
    nj = NeighborJoining()
    return nj.build(distance_matrix)
