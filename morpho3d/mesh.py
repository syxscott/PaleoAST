"""
================================================================================
PaleoAST 3D Morphometrics - Mesh Module
================================================================================

本模块提供3D网格数据结构，用于曲面分析和可视化。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Mesh3D:
    """
    三维网格类

    表示三角网格模型。

    属性:
        vertices: 顶点坐标 (n_vertices, 3)
        faces: 三角面索引 (n_faces, 3)
        normals: 顶点法向量 (n_vertices, 3)
        edges: 边列表
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    _edges: list[tuple[int, int]] | None = field(default=None, repr=False)

    def __post_init__(self):
        """验证网格数据"""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            logger.error(f"Vertices must be (n, 3), got {self.vertices.shape}")
            raise ValueError(f"Vertices must be (n, 3), got {self.vertices.shape}")

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            logger.error(f"Faces must be (n, 3), got {self.faces.shape}")
            raise ValueError(f"Faces must be (n, 3), got {self.faces.shape}")

        # Face indices are checked here rather than left to blow up inside a
        # numpy fancy-index later, where the traceback points at index
        # arithmetic instead of at the mesh that is wrong.
        n_vertices = len(self.vertices)
        if self.faces.size:
            lowest = int(self.faces.min())
            highest = int(self.faces.max())
            if lowest < 0 or highest >= n_vertices:
                raise ValueError(f"Face indices must lie in [0, {n_vertices}), got range [{lowest}, {highest}]")

        logger.info(f"Mesh3D created: {len(self.vertices)} vertices, {len(self.faces)} faces")
        if self.normals is None:
            self.compute_normals()

    def compute_normals(self) -> None:
        """计算顶点法向量"""
        logger.debug(f"Computing vertex normals for {len(self.vertices)} vertices")
        n_vertices = len(self.vertices)
        normals = np.zeros((n_vertices, 3))

        for face in self.faces:
            v0, v1, v2 = self.vertices[face]

            # 计算面法向量
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)

            if norm > 1e-10:
                normal = normal / norm

            # 累加到顶点
            for idx in face:
                normals[idx] += normal

        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self.normals = normals / norms

    @property
    def edges(self) -> list[tuple[int, int]]:
        """获取边列表"""
        if self._edges is None:
            self._compute_edges()
        return self._edges

    def _compute_edges(self) -> None:
        """计算所有边"""
        edges_set = set()
        for face in self.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges_set.add(edge)
        self._edges = list(edges_set)

    def compute_surface_area(self) -> float:
        """计算表面积"""
        logger.debug(f"Computing surface area for {len(self.faces)} faces")
        total_area = 0.0

        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            e1 = v1 - v0
            e2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            total_area += area

        logger.info(f"Surface area computed: {total_area:.4f}")
        return total_area

    def compute_volume(self, require_closed: bool = True) -> float:
        """计算体积（散度定理，要求**闭合**曲面）

        The divergence theorem only gives the enclosed volume for a closed
        surface. For an open mesh the per-face tetrahedra terms no longer
        cancel, so the sum silently returns a number that depends on where
        the origin happens to be -- translation of the whole mesh changes the
        answer. That is a wrong number, not a rough one, and it was previously
        returned without complaint.

        Parameters
        ----------
        require_closed:
            When True (the default) an open mesh raises. Pass False to get
            the raw sum anyway, for callers who know what they are doing.
        """
        n_vertices = len(self.vertices)

        # Every edge of a closed orientable surface is shared by exactly two
        # faces. Counting directed face edges and looking for any vertex-edge
        # that appears once detects the common open case cheaply.
        edge_counts: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                edge_counts[(min(a, b), max(a, b))] = edge_counts.get((min(a, b), max(a, b)), 0) + 1
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

        if boundary_edges and require_closed:
            raise ValueError(
                f"Volume is not defined for an open mesh: found "
                f"{len(boundary_edges)} boundary edge(s) (e.g. {boundary_edges[:3]}) "
                f"that are used by only one face out of {len(self.faces)} faces "
                f"spanning {n_vertices} vertices. Close the surface, or call "
                "compute_volume(require_closed=False) to get the raw "
                "divergence sum and accept that it is origin-dependent."
            )

        logger.debug(f"Computing volume for {len(self.faces)} faces (closed surface)")
        total_volume = 0.0

        for face in self.faces:
            v0, v1, v2 = self.vertices[face]

            # 四面体体积 = (1/6) * dot(v0, cross(v1, v2))
            vol = np.dot(v0, np.cross(v1, v2))
            total_volume += vol

        volume = abs(total_volume) / 6.0
        logger.info(f"Volume computed: {volume:.4f}")
        return volume

    def sample_points(self, n_points: int, seed: int | None = None) -> np.ndarray:
        """
        在曲面上采样点（按面积均匀）

        参数:
            n_points: 采样点数
            seed: 随机种子。给定时结果可复现且不触碰全局随机状态；不给定
                时使用一个局部生成器，仍然**不会**污染进程内其他使用者的
                ``np.random``（旧实现直接调用 ``np.random.choice`` /
                ``np.random.random``，既不可传入种子也不线程安全）。

        返回:
            采样点坐标 (n_points, 3)
        """
        logger.info(f"Sampling {n_points} points on mesh surface ({len(self.faces)} faces)")
        # 计算每个面的面积权重
        areas = []
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            e1 = v1 - v0
            e2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            areas.append(area)

        areas = np.array(areas)
        total_area = areas.sum()
        if not np.isfinite(total_area) or total_area <= 0.0:
            raise ValueError(
                "Cannot sample a mesh whose faces have zero total area; check the geometry or remove degenerate faces."
            )
        weights = areas / total_area

        # 局部随机源：与 phylogenetics/heuristic_search 同一约定
        rng = np.random.default_rng(seed)

        # 采样面
        sampled_face_indices = rng.choice(len(self.faces), size=n_points, p=weights)

        # 在每个选中面上采样点
        points = np.zeros((n_points, 3))

        for i, face_idx in enumerate(sampled_face_indices):
            face = self.faces[face_idx]
            v0, v1, v2 = self.vertices[face]

            # 重心坐标采样
            r1 = np.sqrt(rng.random())
            r2 = rng.random()

            u = 1 - r1
            v = r1 * (1 - r2)
            w = r1 * r2

            points[i] = u * v0 + v * v1 + w * v2

        return points


class SurfaceInterpolator:
    """
    曲面插值器

    在3D网格曲面上进行插值。
    """

    def __init__(self, mesh: Mesh3D):
        """
        初始化插值器

        参数:
            mesh: 3D网格
        """
        self._mesh = mesh
        self._logger = logging.getLogger(f"{__name__}.SurfaceInterpolator")

    def interpolate_values(self, vertex_values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
        """
        在查询点插值顶点值

        参数:
            vertex_values: 顶点值 (n_vertices,)
            query_points: 查询点 (n_query, 3)

        返回:
            插值值 (n_query,)
        """
        from scipy.spatial import KDTree

        self._logger.info(
            f"Interpolating values at {len(query_points)} query points from {len(self._mesh.vertices)} mesh vertices"
        )
        tree = KDTree(self._mesh.vertices)

        # 找最近顶点 (k=4 for IDW interpolation)
        k = min(4, len(self._mesh.vertices))
        distances, indices = tree.query(query_points, k=k)

        if k == 1:
            return vertex_values[indices]

        # 使用距离倒数加权平均
        weights = 1.0 / (distances + 1e-10)

        interpolated = np.sum(weights * vertex_values[indices], axis=1) / np.sum(weights, axis=1)

        return interpolated
