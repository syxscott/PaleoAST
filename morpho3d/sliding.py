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

from .gpa3d import GPA3D, GPA3DResult, RotationMatrix
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
            gpa_result, centroids, _scales, _rots = self._gpa_with_fixed_landmarks(current_configs)

            # 步骤2: 计算当前平均形状
            current_mean = gpa_result.mean_config

            # 检查收敛 (相对判据: 绝对阈值会随坐标尺度变化提前/延后
            # 退出, 破坏滑动的尺度不变性)
            if prev_mean is not None:
                # 计算半标志点区域的变化
                semi_mean_diff = np.linalg.norm(current_mean[self._semi_indices] - prev_mean[self._semi_indices])
                denom = np.linalg.norm(prev_mean[self._semi_indices])
                rel_diff = semi_mean_diff / denom if denom > 0.0 else 0.0

                if self._verbose:
                    self._logger.debug(f"Iteration {iteration}: rel-diff = {rel_diff:.2e}")

                if rel_diff < self._tolerance:
                    self._logger.info(f"Converged after {n_iterations} iterations")
                    break

            prev_mean = current_mean.copy()

            # 步骤3: 滑动半标志点 (在GPA对齐坐标系中计算, 再经各自的
            # 逆相似变换映回原始坐标系)。旧实现在对齐系中算位移、却把
            # 它直接加到原始坐标上 (两套坐标混用), 且把对齐系的共识/
            # 对齐位置写回原始构型, 滑动结果随坐标系尺度随机漂移。
            for i in range(n_samples):
                aligned_config = gpa_result.aligned_configs[i]
                if self._surface_mesh is not None:
                    # 曲面滑动
                    new_semi_aligned = self._slide_surface_points(aligned_config, current_mean)
                else:
                    # 曲线滑动
                    new_semi_aligned = self._slide_curve_points(aligned_config, current_mean)

                rot = gpa_result.rotations[i]
                cs = gpa_result.centroid_sizes[i]
                scaled = gpa_result.centroid_sizes[i] > 1e-10

                # 逆变换: raw = aligned @ R * cs + centroid
                slid_raw = new_semi_aligned @ rot
                if scaled:
                    slid_raw = slid_raw * cs
                slid_raw = slid_raw + centroids[i]

                new_config = current_configs[i].copy()
                new_config[self._semi_indices] = slid_raw
                current_configs[i] = new_config

            # 记录历史 (弯曲能在对齐坐标系内计算: 共识与标本构型同帧,
            # 否则 BE 随原始坐标尺度漂移)
            history.append(
                {
                    "iteration": iteration,
                    "mean_shape": current_mean.copy(),
                    "bending_energy": self._compute_bending_energy(
                        gpa_result.aligned_configs, current_mean
                    ),
                }
            )

        # 最终GPA
        final_gpa, _fc, _fs, _fr = self._gpa_with_fixed_landmarks(current_configs)

        # 计算最终弯曲能量
        final_be = self._compute_bending_energy(
            final_gpa.aligned_configs, final_gpa.mean_config
        )

        self._logger.info(f"Sliding complete: {n_iterations} iterations, final bending energy = {final_be:.4f}")

        return SlidingResult(
            aligned_configs=final_gpa.aligned_configs,
            mean_config=final_gpa.mean_config,
            sliding_history=history,
            n_iterations=n_iterations,
            final_bending_energy=final_be,
            convergence_error=semi_mean_diff if prev_mean is not None else 0.0,
        )

    def _gpa_with_fixed_landmarks(
        self, configs: list[np.ndarray]
    ) -> tuple[GPA3DResult, np.ndarray, list[np.ndarray], np.ndarray]:
        """
        对固定界标执行GPA, 并把每个标本的变换 (平移+缩放+旋转)
        应用到完整构型上。

        旧实现只返回固定界标子集的对齐结果 (n_fixed 个点), 随后用
        全树索引 aligned[self._semi_indices] / mean[self._semi_indices]
        访问 → 正常配置下直接 IndexError 崩溃 (2026-09 复审)。
        另一版实现复用 GPA3D 返回的 rotations/centroid_sizes 做逆变换,
        但那是相对"第一轮原始构型"的变换, 而这里的重缩放/重旋转是相对
        已收敛的固定界标子集 —— 复合变换不再是相似变换, 破坏滑动的
        几何正确性与尺度不变性。现在在本函数内为每个标本计算自洽的
        (centroid, cs, R): aligned_full = (config - centroid) / cs @ R.T,
        逆变换 raw = aligned @ R * cs + centroid 精确成立。

        参数:
            configs: 构型列表

        返回:
            (GPA3DResult, centroids, scales, rotations):
            aligned_configs/mean_config 均为完整构型; centroids 为每个
            标本所用平移中心 (固定界标质心, 原始坐标系); scales/rotations
            为与 aligned_configs 自洽的缩放/旋转, 供滑动后逆映射使用。
        """
        # 提取固定界标并执行GPA
        fixed_configs = [config[self._fixed_indices] for config in configs]
        gpa = GPA3D(tolerance=self._tolerance, verbose=False)
        result = gpa.analyze(fixed_configs)

        # 对收敛后的固定界标子集重新取心/取尺度, 并求对齐到固定界标
        # 均值的旋转 —— 保证与 aligned_full 使用同一变换。
        n_dims = np.asarray(configs[0]).shape[1]
        aligned_full: list[np.ndarray] = []
        centroids = np.zeros((len(configs), n_dims))
        scales = np.ones(len(configs))
        rotations: list[np.ndarray] = []

        mean_fixed = np.mean(np.stack(result.aligned_configs), axis=0)
        mean_fixed_c = mean_fixed - mean_fixed.mean(axis=0)

        for k, config in enumerate(configs):
            fixed_pts = config[self._fixed_indices]
            centroid = fixed_pts.mean(axis=0)
            centered_fixed = fixed_pts - centroid
            cs = float(np.sqrt(np.sum(centered_fixed**2)))
            if cs <= 1e-12:
                cs = 1.0
            scaled_fixed = centered_fixed / cs
            rot = RotationMatrix.from_svd(scaled_fixed, mean_fixed_c)
            centroids[k] = centroid
            scales[k] = cs
            rotations.append(rot)
            aligned_full.append((config - centroid) / cs @ rot.T)

        mean_full = np.mean(np.stack(aligned_full), axis=0)

        gpa_result = GPA3DResult(
            aligned_configs=aligned_full,
            mean_config=mean_full,
            centroid_sizes=scales,
            rotations=rotations,
            n_iterations=result.n_iterations,
            final_spread=result.final_spread,
            procrustes_distances=None,
        )
        return gpa_result, centroids, scales, rotations

    def _slide_curve_points(self, aligned: np.ndarray, mean_shape: np.ndarray) -> np.ndarray:
        """
        沿曲线滑动半标志点 (GPA对齐坐标系内计算)

        参数:
            aligned: 该标本对齐后的完整构型 (n_landmarks, 3)
            mean_shape: 对齐系共识构型

        返回:
            滑动后半标志点的对齐系坐标 (n_semi, 3)

        准则:
            procrustes → Bookstein (1997) minPerp 闭式投影: 内部半标志点
            正交投影到同一标本相邻两点的连线上 (精确极小值, 取代旧版
            沿共识切向的 +/-0.1 网格搜索; 端点不滑动)。
            bending_energy → 保持曲率一致的光滑代理: 位移为与共识的
            拉普拉斯(二阶差分)差, 投影到切向后按 sliding_factor 阻尼。
        """
        semi_aligned = np.array(aligned[self._semi_indices], dtype=float, copy=True)
        n_semi = len(self._semi_indices)

        if self._criterion == self.CRITERION_PROCRUSTES:
            # closed-form minPerp: project interior points onto the chord
            # through their neighbours of the same specimen
            for i in range(1, n_semi - 1):
                a = semi_aligned[i - 1]
                b = semi_aligned[i + 1]
                d = b - a
                dd = float(d @ d)
                if dd <= np.finfo(float).eps:
                    continue
                t = float((semi_aligned[i] - a) @ d) / dd
                semi_aligned[i] = a + t * d
            return semi_aligned

        # Bending-energy proxy (aligned frame only):
        # 使标本半标志点曲线的曲率 (二阶差分) 向共识一致。位移只保留
        # 切向分量, 不把半标志点拉向共识位置, 保留真实法向形状变异。
        semi_mean = mean_shape[self._semi_indices]
        tangents = self._compute_curve_tangents(semi_aligned)
        displacement = np.zeros_like(semi_aligned)
        if n_semi >= 3:
            for i in range(1, n_semi - 1):
                lap_mean = semi_mean[i - 1] - 2.0 * semi_mean[i] + semi_mean[i + 1]
                lap_aligned = (
                    semi_aligned[i - 1] - 2.0 * semi_aligned[i] + semi_aligned[i + 1]
                )
                displacement[i] = lap_mean - lap_aligned

        for i in range(n_semi):
            t = tangents[i]
            proj = float(displacement[i] @ t) * t
            semi_aligned[i] += self._sliding_factor * proj

        return semi_aligned

    def _slide_surface_points(self, aligned: np.ndarray, mean_shape: np.ndarray) -> np.ndarray:
        """
        在曲面上滑动半标志点 (GPA对齐坐标系内计算)

        参数:
            aligned: 该标本对齐后的完整构型
            mean_shape: 对齐系共识构型

        返回:
            滑动后半标志点的对齐系坐标 (n_semi, 3)

        旧实现的两条准则分支完全相同 (弯曲能量判据形同虚设), 且法向量
        取自原始坐标系半标志点、位移却来自对齐系 (坐标系混用)。
        """
        if self._surface_mesh is None:
            raise ValueError("Surface mesh not set")

        n_semi = len(self._semi_indices)

        # 获取对齐系半标志点
        semi_aligned = np.array(aligned[self._semi_indices], dtype=float, copy=True)
        semi_mean = mean_shape[self._semi_indices]

        # 计算切平面法向量 (对齐系)
        normals = self._compute_surface_normals(semi_aligned)

        # 计算位移 (对齐系内)
        if self._criterion == self.CRITERION_BENDING_ENERGY:
            # 曲率一致代理: 与共识的拉普拉斯差, 只保留切平面内分量
            displacement = np.zeros_like(semi_aligned)
            if n_semi >= 3:
                for i in range(1, n_semi - 1):
                    lap_mean = semi_mean[i - 1] - 2.0 * semi_mean[i] + semi_mean[i + 1]
                    lap_aligned = (
                        semi_aligned[i - 1] - 2.0 * semi_aligned[i] + semi_aligned[i + 1]
                    )
                    displacement[i] = lap_mean - lap_aligned
        else:
            # 最小 Procrustes: 向共识的切平面内位移
            displacement = semi_mean - semi_aligned

        # 投影到切平面
        for i in range(n_semi):
            n = normals[i]
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n = n / norm
            d = displacement[i]

            # 投影到切平面: d_proj = d - (d·n)n
            proj = d - float(d @ n) * n

            # 应用滑动因子
            semi_aligned[i] += self._sliding_factor * proj

        return semi_aligned

    def _compute_curve_tangents(self, points: np.ndarray) -> np.ndarray:
        """
        计算曲线的单位切向量 (中心差分, 端点用单侧差分)

        参数:
            points: 半标志点坐标 (n_semi, 3), 沿曲线有序排列

        返回:
            切向量 (n_semi, 3)
        """
        n_semi = len(points)
        tangents = np.zeros((n_semi, 3))

        for i in range(n_semi):
            if n_semi == 1:
                tangents[i] = np.array([1.0, 0.0, 0.0])
                continue
            if i == 0:
                prev_point = 2.0 * points[0] - points[1]  # mirror for forward diff
                next_point = points[1]
            elif i == n_semi - 1:
                prev_point = points[i - 1]
                next_point = 2.0 * points[i] - points[i - 1]  # mirror for backward diff
            else:
                prev_point = points[i - 1]
                next_point = points[i + 1]

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
            # 每标本: 共识(固定点) → 标本(固定点) 的 TPS 弯曲能。
            # 旧实现把固定点数目的权重 w 与半标志点数目的核矩阵
            # K(semi, semi) 相乘, 维数不匹配 (仅当两类点数恰好相等时
            # 才不报错, 结果亦无意义)。
            src_pts = mean_shape[self._fixed_indices]
            tps = TPS3D(kernel="thin_plate")
            tps.fit(src_pts, config[self._fixed_indices])
            K = tps._compute_kernel_matrix(src_pts, src_pts)
            total_be += float(tps._compute_bending_energy(K))

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
