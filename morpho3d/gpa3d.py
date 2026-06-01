"""
================================================================================
PaleoAST 3D Morphometrics - 3D Generalized Procrustes Analysis
================================================================================

本模块实现三维广义普氏分析 (3D GPA) 算法。

数学理论:
================================================================================

1. 问题定义
--------------------------------------------------------------------------------
给定 n 个 3D 样本，每个样本有 k 个标志点:

    X_i ∈ ℝ^(k×3), i = 1, ..., n

目标: 找到最优对齐使得形状差异最小化。

2. GPA变换步骤
--------------------------------------------------------------------------------
每次迭代包含:

a) 平移 (Translation):
    每个构型平移到其质心:

    X'_i = X_i - 1̄_Xi

    其中 1̄ = [1,1,...,1]^T/k，X̄_i = mean(X_i)

b) 缩放 (Isotropic Scaling):
    每个构型缩放到单位质心大小:

    X''_i = X'_i / CS_i

    其中 CS_i = √(trace(X'_i^T X'_i)) 为质心大小

c) 旋转 (Rotation):
    使用SVD找到最优旋转:

    R* = arg min_R ||RX''_target - X''_reference||_F

    解: R* = UV^T，其中 UΣV^T = (X''_target)^T X''_reference 的SVD

3. GPA算法流程
--------------------------------------------------------------------------------
    Initialize: X_i^(0) = X_i (原始数据)
    repeat:
        1. 计算当前均值: X̄ = (1/n) Σ X_i^(t-1)
        2. 对每个样本:
           - 平移: X_i' = X_i^(t-1) - X̄_i
           - 缩放: X_i'' = X_i' / CS_i
           - 旋转: X_i^(t) = R_i X_i''
        3. 更新均值: X̄^(t) = (1/n) Σ X_i^(t)
    until ||X̄^(t) - X̄^(t-1)|| < ε

4. 收敛判定
--------------------------------------------------------------------------------
    ||X̄^(t) - X̄^(t-1)||_F < tol * ||X̄^(t-1)||_F

5. Procrustes距离
--------------------------------------------------------------------------------
两构型间的Procrustes距离:

    d_P(X, Y) = ||X* - Y*||_F

其中 X*, Y* 是对齐后的构型。

6. 全局形状空间
--------------------------------------------------------------------------------
所有对齐后的构型构成一个 n×(k×3) 维的形状空间。
协方差矩阵的特征向量定义为"相对扭曲"。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .quaternion import RotationMatrix

logger = logging.getLogger(__name__)


@dataclass
class GPA3DResult:
    """
    3D GPA分析结果

    属性:
        aligned_configs: 对齐后的构型列表
        mean_config: 平均构型
        centroid_sizes: 每个样本的质心大小
        rotations: 每个样本的旋转矩阵
        n_iterations: 迭代次数
        final_spread: 最终散布度
        procrustes_distances: 样本间Procrustes距离矩阵
    """

    aligned_configs: list[np.ndarray]
    mean_config: np.ndarray
    centroid_sizes: np.ndarray
    rotations: list[np.ndarray]
    n_iterations: int
    final_spread: float
    procrustes_distances: np.ndarray | None = None

    @property
    def aligned_configurations(self) -> np.ndarray:
        """Alias: stacked array of aligned configs (n_samples, n_landmarks, dims)."""
        return np.stack(self.aligned_configs)

    @property
    def n_samples(self) -> int:
        """样本数量"""
        return len(self.aligned_configs)

    @property
    def n_landmarks(self) -> int:
        """标志点数量"""
        return self.aligned_configs[0].shape[0] if self.aligned_configs else 0

    def get_shape_coordinates(self) -> np.ndarray:
        """
        获取展平的形状坐标矩阵

        返回:
            (n_samples, n_landmarks×3) 形状矩阵
        """
        return np.array([config.flatten() for config in self.aligned_configs])

    def compute_covariance(self) -> np.ndarray:
        """
        计算形状空间的协方差矩阵

        返回:
            (n_landmarks×3, n_landmarks×3) 协方差矩阵
        """
        coords = self.get_shape_coordinates()
        coords_centered = coords - coords.mean(axis=0)
        return (coords_centered.T @ coords_centered) / (self.n_samples - 1)


class GPA3D:
    """
    三维广义普氏分析 (3D GPA)

    对3D标志点数据进行迭代对齐，消除平移、缩放和旋转变异。

    使用示例:
        >>> gpa = GPA3D(tolerance=1e-8, max_iterations=100)
        >>> configs = [np.random.randn(20, 3) for _ in range(10)]
        >>> result = gpa.analyze(configs)
        >>> print(f"Converged in {result.n_iterations} iterations")
        >>> print(f"Mean shape centroid size: {result.centroid_sizes.mean():.4f}")
    """

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100, scale: bool = True, verbose: bool = False):
        """
        初始化3D GPA

        参数:
            tolerance: 收敛容忍度
            max_iterations: 最大迭代次数
            scale: 是否进行缩放
            verbose: 是否输出详细信息
        """
        self._tolerance = tolerance
        self._max_iter = max_iterations
        self._scale = scale
        self._verbose = verbose

        self._logger = logging.getLogger(f"{__name__}.GPA3D")

    def analyze(self, configs: list[np.ndarray], reference: np.ndarray | None = None) -> GPA3DResult:
        """
        执行3D GPA分析

        参数:
            configs: 构型列表，每个 (n_landmarks, 3)
            reference: 参考构型 (可选，默认使用当前均值)

        返回:
            GPA3DResult对象

        异常:
            ValueError: 构型形状不一致
        """
        # 验证输入
        if configs is None or len(configs) == 0:
            raise ValueError("No configurations provided")

        n_samples = len(configs)

        # 检查标志点数量一致性
        n_landmarks = configs[0].shape[0]
        for i, config in enumerate(configs):
            if len(config.shape) != 2 or config.shape[0] != n_landmarks or config.shape[1] != 3:
                raise ValueError(f"Config {i} has invalid shape {config.shape}, expected ({n_landmarks}, 3)")

        self._logger.info(f"Starting 3D GPA with {n_samples} configs, {n_landmarks} landmarks")

        # 初始化
        aligned = [config.copy() for config in configs]
        centroid_sizes = np.zeros(n_samples)
        rotations = [np.eye(3) for _ in range(n_samples)]

        # 计算初始质心大小
        for i in range(n_samples):
            centroid_sizes[i] = self._compute_centroid_size(aligned[i])

        # 迭代优化
        prev_mean = None
        n_iterations = 0

        for iteration in range(self._max_iter):
            n_iterations = iteration + 1

            # 步骤1: 计算当前均值
            current_mean = np.mean(aligned, axis=0)

            # 检查收敛
            if prev_mean is not None:
                diff = np.linalg.norm(current_mean - prev_mean)
                diff_norm = diff / (np.linalg.norm(prev_mean) + 1e-10)

                if self._verbose:
                    self._logger.debug(f"Iteration {iteration}: diff = {diff:.2e}, diff_norm = {diff_norm:.2e}")

                if diff_norm < self._tolerance:
                    self._logger.info(f"Converged after {n_iterations} iterations")
                    break

            prev_mean = current_mean.copy()

            # 步骤2: 对齐每个构型到当前均值
            for i in range(n_samples):
                # 平移: 移动到质心
                mean_i = np.mean(aligned[i], axis=0)
                translated = aligned[i] - mean_i

                # 缩放: 单位质心大小
                if self._scale:
                    cs = self._compute_centroid_size(translated)
                    if cs > 1e-10:
                        scaled = translated / cs
                    else:
                        scaled = translated
                    centroid_sizes[i] = cs
                else:
                    scaled = translated

                # 旋转: 对齐到参考
                if reference is None:
                    target = current_mean - np.mean(current_mean, axis=0)
                else:
                    target = reference - np.mean(reference, axis=0)

                if self._scale:
                    cs_target = self._compute_centroid_size(target)
                    if cs_target > 1e-10:
                        target = target / cs_target

                # SVD旋转对齐
                try:
                    R = RotationMatrix.from_svd(scaled, target)
                except (np.linalg.LinAlgError, ValueError):
                    R = np.eye(3)
                rotations[i] = R

                # 应用旋转
                aligned[i] = scaled @ R.T

        # 计算最终散布度
        final_mean = np.mean(aligned, axis=0)
        final_spread = np.sum([np.linalg.norm(config - final_mean) ** 2 for config in aligned]) / n_samples

        # 计算Procrustes距离矩阵
        procrustes_distances = self._compute_distance_matrix(aligned)

        self._logger.info(f"GPA complete: {n_iterations} iterations, final spread = {final_spread:.4f}")

        return GPA3DResult(
            aligned_configs=aligned,
            mean_config=final_mean,
            centroid_sizes=centroid_sizes,
            rotations=rotations,
            n_iterations=n_iterations,
            final_spread=final_spread,
            procrustes_distances=procrustes_distances,
        )

    def _compute_centroid_size(self, config: np.ndarray) -> float:
        """
        计算构型的质心大小

        参数:
            config: 构型 (n_landmarks, 3)

        返回:
            质心大小

        数学公式:
            CS = √(Σ ||x_i - x̄||²)
               = √(trace(X^T X))
        """
        config = np.asarray(config, dtype=np.float64)

        # 移到质心
        centered = config - np.mean(config, axis=0)

        # 计算质心大小
        cs = np.sqrt(np.sum(centered**2))

        return cs

    def _compute_distance_matrix(self, configs: list[np.ndarray]) -> np.ndarray:
        """
        计算Procrustes距离矩阵

        参数:
            configs: 对齐后的构型列表

        返回:
            (n, n) 距离矩阵
        """
        n = len(configs)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Procrustes距离
                d = self._procrustes_distance(configs[i], configs[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    def _procrustes_distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """
        计算两构型间的Procrustes距离

        参数:
            config1: 构型1 (n_landmarks, 3)
            config2: 构型2 (n_landmarks, 3)

        返回:
            Procrustes距离

        数学公式:
            d_P(X, Y) = min_{θ,c} ||s·R(θ)X + c1^T - Y||_F / ||Y||_F
        """
        # 使用SVD找到最优旋转
        R = RotationMatrix.from_svd(config1, config2)

        # 应用旋转
        rotated = config1 @ R.T

        # 计算距离
        diff = rotated - config2
        distance = np.sqrt(np.sum(diff**2))

        return distance

    def partial_gpa(self, configs: list[np.ndarray], fixed_indices: np.ndarray) -> GPA3DResult:
        """
        部分GPA: 固定部分标志点

        用于半标志点滑动的预处理步骤。

        参数:
            configs: 构型列表
            fixed_indices: 固定标志点的索引

        返回:
            GPA3DResult对象
        """
        # 提取固定点
        fixed_configs = [config[fixed_indices] for config in configs]

        # 执行标准GPA
        result = self.analyze(fixed_configs)

        return result


def compute_partial_gpa(configs: list[np.ndarray], fixed_indices: np.ndarray, tolerance: float = 1e-8) -> GPA3DResult:
    """
    计算部分GPA的便捷函数

    参数:
        configs: 构型列表
        fixed_indices: 固定标志点索引
        tolerance: 收敛容忍度

    返回:
        GPA3DResult对象
    """
    gpa = GPA3D(tolerance=tolerance)
    return gpa.partial_gpa(configs, fixed_indices)


def procrustes_distance_3d(config1: np.ndarray, config2: np.ndarray) -> tuple[float, np.ndarray, float]:
    """
    计算两个3D构型的Procrustes分析

    参数:
        config1: 构型1 (n_landmarks, 3)
        config2: 构型2 (n_landmarks, 3)

    返回:
        (distance, rotation_matrix, scale_ratio)

    数学公式:
        d² = ||s·R·X₁ - X₂||²

        其中 s 是缩放比，R 是旋转矩阵
    """
    config1 = np.asarray(config1, dtype=np.float64)
    config2 = np.asarray(config2, dtype=np.float64)

    # 平移到质心
    c1 = config1 - np.mean(config1, axis=0)
    c2 = config2 - np.mean(config2, axis=0)

    # 计算质心大小
    cs1 = np.sqrt(np.sum(c1**2))
    cs2 = np.sqrt(np.sum(c2**2))

    # 归一化
    n1 = c1 / cs1
    n2 = c2 / cs2

    # SVD旋转
    R = RotationMatrix.from_svd(n1, n2)

    # 应用变换
    rotated = cs1 * n1 @ R.T

    # 计算距离
    d = np.sqrt(np.sum((rotated - c2) ** 2))

    # 缩放比
    s = cs2 / cs1 if cs1 > 1e-10 else 1.0

    return d, R, s
