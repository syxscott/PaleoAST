"""
================================================================================
PaleoAST 3D Morphometrics - Sliding Semi-landmarks
================================================================================

本模块实现曲线和曲面的半标志点滑动算法。

数学理论:
================================================================================

1. 问题背景
--------------------------------------------------------------------------------
半标志点(Semi-landmarks)用于描述曲线和曲面上的非界标点。
它们不是真正的固定标志点，而是沿特定方向可以滑动。

2. 滑动目标函数
--------------------------------------------------------------------------------
两种主要的滑动准则:

a) 最小弯曲能量 (Minimum Bending Energy):
    滑动半标志点使TPS弯曲能量最小化。

    E_BE = ||Φ(s) - Φ(target)||²

    其中 Φ(s) 是当前半标志点位置，Φ(target) 是目标位置。

b) 最小Procrustes距离 (Minimum Procrustes Distance):
    滑动使当前构型与参考构型的Procrustes距离最小。

    E_PD = ||X_s - X_ref||²

3. 曲线滑动算法
--------------------------------------------------------------------------------
对于曲线上的半标志点，沿切线方向滑动:

    s_i' = s_i + α · t_i

其中 t_i 是单位切向量，α 是滑动步长。

切向量计算:
    t_i = (p_{i+1} - p_{i-1}) / ||p_{i+1} - p_{i-1}||

4. 曲面滑动算法
--------------------------------------------------------------------------------
对于曲面，滑动发生在切平面内:

    s_i' = s_i + α · d_i

其中 d_i 是投影到切平面的位移向量。

切平面法向量 (三角形网格):
    n_i = normalize(Σ n_f for all faces f containing vertex i)

5. 迭代优化
--------------------------------------------------------------------------------
    for iteration = 1 to max_iter:
        1. 固定所有界标点，执行GPA
        2. 计算当前平均形状
        3. 对每个半标志点:
           - 计算切向量/切平面
           - 计算投影位移 d_proj
           - 更新位置: s_new = s_old + λ · d_proj
        4. 重复直到收敛

6. 切向量计算细节
--------------------------------------------------------------------------------
对于闭合曲线:
    t_i = 0.5 * (unit_vector(p_{i+1} - p_i) + unit_vector(p_i - p_{i-1}))

对于开放曲线:
    t_1 = unit_vector(p_2 - p_1)
    t_n = unit_vector(p_n - p_{n-1})

7. Procrustes切向投影
--------------------------------------------------------------------------------
对于最小Procrustes距离准则:

    d_proj = P_⊥ · (X_ref - X_current)

其中 P_⊥ = I - n·n^T 是垂直于切向量的投影矩阵。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .gpa3d import GPA3D, GPA3DResult
from .tps3d import TPS3D

logger = logging.getLogger(__name__)


@dataclass
class SlidingResult:
    """
    半标志点滑动结果

    属性:
        aligned_configs: 对齐后的完整构型
        mean_config: 平均构型
        sliding_history: 滑动历史
        n_iterations: 迭代次数
        final_bending_energy: 最终弯曲能量
        convergence_error: 收敛误差
    """

    aligned_configs: list[np.ndarray]
    mean_config: np.ndarray
    sliding_history: list[dict]
    n_iterations: int
    final_bending_energy: float
    convergence_error: float


class SemiLandmarkSlider:
    """
    半标志点滑动算法

    支持曲线和曲面半标志点的迭代滑动。

    使用示例:
        >>> slider = SemiLandmarkSlider(
        ...     criterion='bending_energy',
        ...     sliding_factor=0.1
        ... )
        >>> slider.set_landmarks(fixed_indices=[0,1,2,3])
        >>> slider.set_curve(topology='closed')
        >>> result = slider.slide(configs)
    """

    CRITERION_BENDING_ENERGY = "bending_energy"
    CRITERION_PROCRUSTES = "procrustes"

    def __init__(
        self,
        criterion: str = "bending_energy",
        sliding_factor: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ):
        """
        初始化滑动算法

        参数:
            criterion: 滑动准则 ('bending_energy' 或 'procrustes')
            sliding_factor: 滑动因子 (0 < λ ≤ 1)
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            verbose: 是否输出详细信息
        """
        if criterion not in (self.CRITERION_BENDING_ENERGY, self.CRITERION_PROCRUSTES):
            raise ValueError(f"Unknown criterion: {criterion}")

        self._criterion = criterion
        self._sliding_factor = np.clip(sliding_factor, 0.01, 1.0)
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._verbose = verbose

        self._fixed_indices: np.ndarray | None = None
        self._semi_indices: np.ndarray | None = None
        self._curve_topology: str = "open"
        self._surface_mesh: np.ndarray | None = None

        self._gpa = GPA3D(tolerance=tolerance)
        self._logger = logging.getLogger(f"{__name__}.SemiLandmarkSlider")

    def set_landmarks(self, fixed_indices: np.ndarray, semi_indices: np.ndarray) -> SemiLandmarkSlider:
        """
        设置界标和半标志点索引

        参数:
            fixed_indices: 固定界标点索引
            semi_indices: 半标志点索引
        """
        self._fixed_indices = np.asarray(fixed_indices, dtype=np.intp)
        self._semi_indices = np.asarray(semi_indices, dtype=np.intp)

        # 验证索引不重叠
        overlap = np.intersect1d(self._fixed_indices, self._semi_indices)
        if len(overlap) > 0:
            raise ValueError(f"Fixed and semi indices overlap: {overlap}")

        return self

    def set_curve_topology(self, topology: str = "open") -> SemiLandmarkSlider:
        """
        设置曲线拓扑

        参数:
            topology: 'open' 或 'closed'
        """
        if topology not in ("open", "closed"):
            raise ValueError(f"Unknown topology: {topology}")

        self._curve_topology = topology
        self._surface_mesh = None
        return self

    def set_surface_mesh(self, faces: np.ndarray) -> SemiLandmarkSlider:
        """
        设置曲面三角形网格

        参数:
            faces: 三角形面索引 (n_faces, 3)
        """
        faces = np.asarray(faces, dtype=np.intp)
        if faces.shape[1] != 3:
            raise ValueError(f"Faces must be (n, 3), got {faces.shape}")

        self._surface_mesh = faces
        return self

    def slide(self, configs: list[np.ndarray]) -> SlidingResult:
        """
        执行半标志点滑动

        参数:
            configs: 构型列表，每个 (n_landmarks, 3)

        返回:
            SlidingResult对象
        """
        if self._fixed_indices is None or self._semi_indices is None:
            raise ValueError("Must call set_landmarks() first")

        n_samples = len(configs)
        configs[0].shape[0]
        n_semi = len(self._semi_indices)

        self._logger.info(
            f"Starting semi-landmark sliding with {n_samples} configs, "
            f"{len(self._fixed_indices)} fixed landmarks, "
            f"{n_semi} semi-landmarks"
        )

        # 初始化构型
        current_configs = [config.copy() for config in configs]

        # 滑动历史
        history = []

        prev_mean = None
        semi_mean_diff = 0.0
        n_iterations = 0

        for iteration in range(self._max_iterations):
            n_iterations = iteration + 1

            # 步骤1: 使用固定界标执行GPA
            gpa_result = self._gpa_with_fixed_landmarks(current_configs)

            # 步骤2: 计算当前平均形状
            current_mean = gpa_result.mean_config

            # 检查收敛
            if prev_mean is not None:
                # 计算半标志点区域的变化
                semi_mean_diff = np.linalg.norm(current_mean[self._semi_indices] - prev_mean[self._semi_indices])

                if self._verbose:
                    self._logger.debug(f"Iteration {iteration}: semi-diff = {semi_mean_diff:.2e}")

                if semi_mean_diff < self._tolerance:
                    self._logger.info(f"Converged after {n_iterations} iterations")
                    break

            prev_mean = current_mean.copy()

            # 步骤3: 滑动半标志点
            for i in range(n_samples):
                if self._surface_mesh is not None:
                    # 曲面滑动
                    new_semi = self._slide_surface_points(
                        current_configs[i], gpa_result.aligned_configs[i], current_mean
                    )
                else:
                    # 曲线滑动
                    new_semi = self._slide_curve_points(current_configs[i], gpa_result.aligned_configs[i], current_mean)

                # 更新构型
                new_config = current_configs[i].copy()
                new_config[self._semi_indices] = new_semi
                current_configs[i] = new_config

            # 记录历史
            history.append(
                {
                    "iteration": iteration,
                    "mean_shape": current_mean.copy(),
                    "bending_energy": self._compute_bending_energy(current_mean, gpa_result.mean_config),
                }
            )

        # 最终GPA
        final_gpa = self._gpa_with_fixed_landmarks(current_configs)

        # 计算最终弯曲能量
        final_be = self._compute_bending_energy(current_configs, final_gpa.mean_config)

        self._logger.info(f"Sliding complete: {n_iterations} iterations, final bending energy = {final_be:.4f}")

        return SlidingResult(
            aligned_configs=final_gpa.aligned_configs,
            mean_config=final_gpa.mean_config,
            sliding_history=history,
            n_iterations=n_iterations,
            final_bending_energy=final_be,
            convergence_error=semi_mean_diff if prev_mean is not None else 0.0,
        )

    def _gpa_with_fixed_landmarks(self, configs: list[np.ndarray]) -> GPA3DResult:
        """
        对固定界标执行GPA

        参数:
            configs: 构型列表

        返回:
            GPA3DResult
        """
        # 提取固定界标
        fixed_configs = [config[self._fixed_indices] for config in configs]

        # 执行GPA
        gpa = GPA3D(tolerance=self._tolerance, verbose=False)
        return gpa.analyze(fixed_configs)

    def _slide_curve_points(self, original: np.ndarray, aligned: np.ndarray, mean_shape: np.ndarray) -> np.ndarray:
        """
        沿曲线滑动半标志点

        参数:
            original: 原始构型
            aligned: 对齐后的构型
            mean_shape: 平均形状

        返回:
            滑动后的半标志点位置
        """
        n_semi = len(self._semi_indices)
        n_total = original.shape[0]

        # 获取半标志点
        semi_original = original[self._semi_indices].copy()
        semi_aligned = aligned[self._semi_indices]
        semi_mean = mean_shape[self._semi_indices]

        # 重新编号的半标志点
        semi_positions = self._semi_indices - self._fixed_indices.min()

        # 计算切向量
        tangents = self._compute_curve_tangents(semi_mean, positions=semi_positions, n_total=n_total)

        # 计算滑动方向
        if self._criterion == self.CRITERION_BENDING_ENERGY:
            # 最小弯曲能量: 沿切线方向移动到均值
            displacement = semi_mean - semi_aligned
        else:
            # 最小Procrustes距离
            displacement = semi_mean - semi_aligned

        # 投影到切线方向
        for i in range(n_semi):
            t = tangents[i]
            d = displacement[i]

            # 投影: d_proj = (d·t) t
            proj = np.dot(d, t) * t

            # 应用滑动因子
            semi_original[i] += self._sliding_factor * proj

        return semi_original

    def _slide_surface_points(self, original: np.ndarray, aligned: np.ndarray, mean_shape: np.ndarray) -> np.ndarray:
        """
        在曲面上滑动半标志点

        参数:
            original: 原始构型
            aligned: 对齐后的构型
            mean_shape: 平均形状

        返回:
            滑动后的半标志点位置
        """
        if self._surface_mesh is None:
            raise ValueError("Surface mesh not set")

        n_semi = len(self._semi_indices)

        # 获取半标志点
        semi_original = original[self._semi_indices].copy()
        semi_aligned = aligned[self._semi_indices]
        semi_mean = mean_shape[self._semi_indices]

        # 计算切平面法向量
        normals = self._compute_surface_normals(semi_original)

        # 计算位移
        if self._criterion == self.CRITERION_BENDING_ENERGY:
            displacement = semi_mean - semi_aligned
        else:
            displacement = semi_mean - semi_aligned

        # 投影到切平面
        for i in range(n_semi):
            n = normals[i]
            d = displacement[i]

            # 投影到切平面: d_proj = d - (d·n)n
            proj = d - np.dot(d, n) * n

            # 应用滑动因子
            semi_original[i] += self._sliding_factor * proj

        return semi_original

    def _compute_curve_tangents(self, points: np.ndarray, positions: np.ndarray, n_total: int) -> np.ndarray:
        """
        计算曲线的单位切向量

        参数:
            points: 半标志点坐标 (n_semi, 3)
            positions: 半标志点在完整构型中的位置
            n_total: 完整构型的标志点总数

        返回:
            切向量 (n_semi, 3)
        """
        n_semi = len(points)
        tangents = np.zeros((n_semi, 3))

        for i, (pos, point) in enumerate(zip(positions, points, strict=False)):
            # 获取前后邻居 — the previous version computed
            # ``pos - 1 if ... else pos + 1`` and discarded the
            # result (dead code). Use ``pos`` for documentation
            # purposes but the actual neighbours are taken from
            # ``points[i-1]`` / ``points[i+1]`` below.
            if pos > 0:
                _prev_pos = pos - 1
            else:
                _prev_pos = pos + 1
            if pos < n_total - 1:
                _next_pos = pos + 1
            else:
                _next_pos = pos - 1

            # 需要获取完整曲线上的邻居
            # 这里简化为使用半标志点内部的邻居
            if i > 0:
                prev_point = points[i - 1]
            else:
                prev_point = points[i + 1] - (points[i + 1] - points[i])

            if i < n_semi - 1:
                next_point = points[i + 1]
            else:
                next_point = points[i - 1] + (points[i] - points[i - 1])

            # 中心差分
            tangent = next_point - prev_point
            norm = np.linalg.norm(tangent)

            if norm > 1e-10:
                tangents[i] = tangent / norm
            else:
                tangents[i] = np.array([1.0, 0.0, 0.0])

        return tangents

    def _compute_surface_normals(self, vertices: np.ndarray) -> np.ndarray:
        """
        计算曲面顶点的法向量

        参数:
            vertices: 顶点坐标 (n, 3)

        返回:
            法向量 (n, 3)
        """
        n = vertices.shape[0]
        normals = np.zeros((n, 3))

        if self._surface_mesh is None:
            return normals

        # 对每个顶点，计算周围面的法向量平均
        for v_idx in range(n):
            normal = np.zeros(3)
            count = 0

            for face in self._surface_mesh:
                if v_idx in face:
                    # 获取面的其他两个顶点
                    other_indices = [i for i in face if i != v_idx]
                    if len(other_indices) == 2:
                        v1 = vertices[other_indices[0]]
                        v2 = vertices[other_indices[1]]
                        v0 = vertices[v_idx]

                        # 计算两个边向量
                        e1 = v1 - v0
                        e2 = v2 - v0

                        # 叉积得到法向量
                        face_normal = np.cross(e1, e2)
                        norm = np.linalg.norm(face_normal)

                        if norm > 1e-10:
                            normal += face_normal / norm
                            count += 1

            # 归一化
            if count > 0:
                normals[v_idx] = normal / np.linalg.norm(normal)
            else:
                normals[v_idx] = np.array([0.0, 0.0, 1.0])

        return normals

    def _compute_bending_energy(self, configs: list[np.ndarray], mean_shape: np.ndarray) -> float:
        """
        计算TPS弯曲能量

        参数:
            configs: 构型列表
            mean_shape: 平均形状

        返回:
            总弯曲能量
        """
        total_be = 0.0

        for config in configs:
            # 创建TPS
            tps = TPS3D(kernel="cubic")
            tps.fit(mean_shape[self._fixed_indices], config[self._fixed_indices])
            total_be += tps._compute_bending_energy(
                tps._compute_kernel_matrix(mean_shape[self._semi_indices], mean_shape[self._semi_indices])
            )

        return total_be


def slide_curve_semi_landmarks(
    configs: list[np.ndarray],
    fixed_indices: np.ndarray,
    semi_indices: np.ndarray,
    criterion: str = "bending_energy",
    sliding_factor: float = 0.1,
) -> SlidingResult:
    """
    曲线半标志点滑动的便捷函数

    参数:
        configs: 构型列表
        fixed_indices: 固定界标索引
        semi_indices: 半标志点索引
        criterion: 滑动准则
        sliding_factor: 滑动因子

    返回:
        SlidingResult
    """
    slider = SemiLandmarkSlider(criterion=criterion, sliding_factor=sliding_factor)
    slider.set_landmarks(fixed_indices, semi_indices)
    slider.set_curve_topology("open")

    return slider.slide(configs)


def slide_surface_semi_landmarks(
    configs: list[np.ndarray],
    fixed_indices: np.ndarray,
    semi_indices: np.ndarray,
    faces: np.ndarray,
    criterion: str = "bending_energy",
    sliding_factor: float = 0.1,
) -> SlidingResult:
    """
    曲面半标志点滑动的便捷函数

    参数:
        configs: 构型列表
        fixed_indices: 固定界标索引
        semi_indices: 半标志点索引
        faces: 三角形面索引
        criterion: 滑动准则
        sliding_factor: 滑动因子

    返回:
        SlidingResult
    """
    slider = SemiLandmarkSlider(criterion=criterion, sliding_factor=sliding_factor)
    slider.set_landmarks(fixed_indices, semi_indices)
    slider.set_surface_mesh(faces)

    return slider.slide(configs)
