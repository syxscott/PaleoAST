"""
================================================================================
PaleoAST 3D Morphometrics - 3D Thin Plate Spline
================================================================================

本模块实现三维薄板样条 (3D TPS) 插值和变形。

数学理论:
================================================================================

1. 问题定义
--------------------------------------------------------------------------------
给定源点 p_i ∈ ℝ³ 和目标点 q_i ∈ ℝ³ (i = 1, ..., n)，
找到映射 f: ℝ³ → ℝ³ 使得 f(p_i) = q_i。

2. TPS核函数
--------------------------------------------------------------------------------
薄板样条使用径向基函数:

    U(r) = |r|² ln(|r|)    (2D情况)
    U(r) = |r|             (3D情况，距离核)
    U(r) = |r|³            (3D情况，薄板核)

更一般的3D TPS核:

    U(r) = { |r|³,          r ≠ 0
           { 0,             r = 0
    
其中 r = ||x - x_i|| 为欧氏距离。

3. TPS函数形式
--------------------------------------------------------------------------------
3D TPS映射 f(x) ∈ ℝ³ 的每个分量:

    f(x) = a₀ + a₁x + a₂y + a₃z + Σᵢ wᵢ U(||x - pᵢ||)

写成矩阵形式:

    f(x) = [1, x^T, W^T] · [a; β; w]

其中:
- a = [a₀, a₁, a₂, a₃]^T 是仿射变换参数 (4,)
- β = [a₁, a₂, a₃]^T 是线性部分
- w = [w₁, ..., wₙ]^T 是核函数权重 (n,)

4. 弯曲能量
--------------------------------------------------------------------------------
TPS的弯曲能量定义为:

    E = ∫∫∫ [f_xx² + f_yy² + f_zz² + 2(f_xy² + f_yz² + f_zx²)] dx dy dz

对于3D TPS:

    E(w) = w^T K w

其中 K_ij = U(||p_i - p_j||) 是核矩阵。

5. 约束条件
--------------------------------------------------------------------------------
为确保解唯一，需要正交条件:

    Σᵢ wᵢ = 0
    Σᵢ wᵢ p_i = 0

这可以通过增广系统实现:

    [K   P] [w]   [q]
    [P^T 0] [a] = [0]

其中 P_ij = [1, p_i] 扩展矩阵。

6. 形变网格
--------------------------------------------------------------------------------
变形后网格点 x' = f(x) 满足:

    x' = A·[1, x^T]^T + W·U(x)

其中 U(x) = [U(||x-p₁||), ..., U(||x-p_n||)]^T

7. Jacobian矩阵
--------------------------------------------------------------------------------
TPS映射的Jacobian:

    J(x) = ∂f/∂x = A_linear + Σᵢ wᵢ ∂U/∂x

对于 |r|³ 核:
    ∂U/∂x = 3||r|| · ∂||r||/∂x = 3r·r^T / ||r||

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TPS3DResult:
    """
    3D TPS分析结果
    
    属性:
        source_points: 源点 (n, 3)
        target_points: 目标点 (n, 3)
        weights: TPS权重 (n,)
        affine_params: 仿射参数 (4, 3)
        bending_energy: 弯曲能量
        grid_points: 变形网格点 (可选)
        deformations: 网格变形向量 (可选)
    """
    source_points: np.ndarray
    target_points: np.ndarray
    weights: np.ndarray
    affine_params: np.ndarray
    bending_energy: float
    grid_points: Optional[np.ndarray] = None
    deformations: Optional[np.ndarray] = None
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        变换新点
        
        参数:
            points: 待变换点 (m, 3)
        
        返回:
            变换后点 (m, 3)
        """
        points = np.asarray(points, dtype=np.float64)
        n_new = points.shape[0]
        
        # 计算距离矩阵
        distances = self._compute_distances(points, self.source_points)
        
        # 核函数值
        K = self._radial_basis_function(distances)
        
        # 增广矩阵 [1, x]
        P = np.hstack([np.ones((n_new, 1)), points])
        
        # 变换
        transformed = P @ self.affine_params + K @ self.weights
        
        return transformed
    
    def _compute_distances(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """计算点间距离矩阵"""
        n = points.shape[0]
        m = centers.shape[0]
        
        distances = np.zeros((n, m))
        for i in range(n):
            diff = points[i] - centers
            distances[i] = np.sqrt(np.sum(diff**2, axis=1))
        
        return distances
    
    def _radial_basis_function(self, r: np.ndarray) -> np.ndarray:
        """
        径向基函数
        
        使用 |r|³ 核
        """
        U = np.where(r > 1e-10, r**3, 0.0)
        return U


class TPS3D:
    """
    三维薄板样条 (3D TPS) 插值
    
    实现基于径向基函数的3D空间变形。
    
    使用示例:
        >>> tps = TPS3D(kernel='cubic')
        >>> source = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        >>> target = np.array([[0,0,0], [1.1,0,0], [0,1.2,0], [0,0,0.9]])
        >>> tps.fit(source, target)
        >>> # 在新点插值
        >>> new_point = np.array([[0.5, 0.5, 0.5]])
        >>> transformed = tps.transform(new_point)
    """
    
    def __init__(
        self,
        kernel: str = 'cubic',
        regularization: float = 0.0
    ):
        """
        初始化3D TPS
        
        参数:
            kernel: 核函数类型 ('cubic', 'thin_plate', 'multiquadric')
            regularization: 正则化参数
        """
        self._kernel = kernel
        self._regularization = regularization
        self._source: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._affine: Optional[np.ndarray] = None
        self._logger = logging.getLogger(f"{__name__}.TPS3D")
    
    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> 'TPS3D':
        """
        拟合TPS变换
        
        参数:
            source: 源点 (n, 3)
            target: 目标点 (n, 3)
        
        返回:
            self
        
        数学公式:
            [K   P] [w]   [q - P·a]
            [P^T 0] [a] = [   0   ]
            
            其中 K_ij = U(||p_i - p_j||)
                  P   = [1, p_i]
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        
        if source.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: source {source.shape} vs target {target.shape}"
            )
        
        if source.shape[1] != 3:
            raise ValueError(f"Points must be 3D, got shape {source.shape}")
        
        n = source.shape[0]
        
        self._logger.info(f"Fitting 3D TPS with {n} control points")
        
        # 计算核矩阵 K_ij = U(||p_i - p_j||)
        K = self._compute_kernel_matrix(source, source)
        
        # 构建增广矩阵
        # P = [1, x, y, z] (n × 4)
        P = np.hstack([np.ones((n, 1)), source])
        
        # 增广系统矩阵
        # [K, P]  (n × (n+4))
        # [P^T, 0] ((n+4) × (n+4))
        top = np.hstack([K, P])
        bottom = np.hstack([P.T, np.zeros((4, 4))])
        A = np.vstack([top, bottom])
        
        # 右端向量 (目标坐标)
        # 分离仿射和弹性部分
        y = target
        
        # 增广右端
        y_aug = np.vstack([y, np.zeros((4, 3))])
        
        # 添加正则化
        if self._regularization > 0:
            reg = self._regularization * np.eye(n)
            A[:n, :n] += reg
        
        # 求解线性系统
        try:
            params = np.linalg.solve(A, y_aug)
        except np.linalg.LinAlgError:
            # 使用最小二乘
            params, _, _, _ = np.linalg.lstsq(A, y_aug, rcond=None)
        
        # 分离权重和仿射参数
        self._weights = params[:n]  # (n,)
        self._affine = params[n:]   # (4, 3)
        self._source = source
        self._target = target
        
        # 计算弯曲能量
        bending_energy = self._compute_bending_energy(K)
        
        self._logger.info(f"TPS fitted, bending energy = {bending_energy:.4f}")
        
        return self
    
    def _compute_kernel_matrix(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """
        计算核函数矩阵
        
        参数:
            points1: 点集1 (n, 3)
            points2: 点集2 (m, 3)
        
        返回:
            K_ij = U(||p_i - q_j||) (n, m)
        """
        n = points1.shape[0]
        m = points2.shape[0]
        
        K = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                r = np.linalg.norm(points1[i] - points2[j])
                K[i, j] = self._kernel_function(r)
        
        return K
    
    def _kernel_function(self, r: float) -> float:
        """
        径向基函数
        
        参数:
            r: 距离
        
        返回:
            U(r)
        """
        if r < 1e-10:
            return 0.0
        
        if self._kernel == 'cubic':
            # |r|³ 核
            return r**3
        elif self._kernel == 'thin_plate':
            # |r| 核 (真正的3D薄板)
            return r
        elif self._kernel == 'multiquadric':
            # √(r² + c²) 核
            c = 1.0
            return np.sqrt(r**2 + c**2)
        elif self._kernel == 'gaussian':
            # 高斯核 exp(-r²/σ²)
            sigma = 1.0
            return np.exp(-r**2 / (2 * sigma**2))
        else:
            return r**3
    
    def _compute_bending_energy(self, K: np.ndarray) -> float:
        """
        计算弯曲能量
        
        参数:
            K: 核矩阵 (n, n)
        
        返回:
            E = w^T K w
        """
        if self._weights is None:
            return 0.0
        
        E = self._weights @ K @ self._weights
        return float(E)
    
    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        变换新点
        
        参数:
            points: 待变换点 (m, 3)
        
        返回:
            变换后点 (m, 3)
        """
        if self._source is None:
            raise ValueError("TPS not fitted, call fit() first")
        
        points = np.asarray(points, dtype=np.float64)
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # 计算核函数值
        K = self._compute_kernel_matrix(points, self._source)
        
        # 增广矩阵
        P = np.hstack([np.ones((points.shape[0], 1)), points])
        
        # 变换
        transformed = P @ self._affine + K @ self._weights
        
        return transformed
    
    def create_deformation_grid(
        self,
        grid_range: Tuple[float, float, float, float, float, float],
        resolution: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建变形网格
        
        参数:
            grid_range: (xmin, xmax, ymin, ymax, zmin, zmax)
            resolution: (nx, ny, nz) 网格分辨率
        
        返回:
            (grid_points, deformations) 网格点和变形向量
        """
        xmin, xmax, ymin, ymax, zmin, zmax = grid_range
        nx, ny, nz = resolution
        
        # 创建网格
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        
        # 生成网格点
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # 变换网格
        deformed = self.transform(grid_points)
        
        # 计算变形向量
        deformations = deformed - grid_points
        
        return grid_points, deformations
    
    def compute_jacobian(self, points: np.ndarray) -> np.ndarray:
        """
        计算TPS映射的Jacobian矩阵
        
        参数:
            points: 采样点 (m, 3)
        
        返回:
            Jacobian矩阵 (m, 3, 3)
        
        数学公式:
            J(x) = A_linear + Σᵢ wᵢ ∂U/∂x
            
            对于 |r|³ 核:
            ∂U/∂x = 3r·r^T / ||r||  (当 r ≠ 0)
        """
        if self._source is None:
            raise ValueError("TPS not fitted")
        
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        m = points.shape[0]
        jacobians = np.zeros((m, 3, 3))
        
        # 线性部分 A_linear
        A_linear = self._affine[1:4, :].T  # (3, 3)
        
        for i in range(m):
            J = A_linear.copy()
            
            # 核函数导数贡献
            for j in range(self._source.shape[0]):
                r_vec = points[i] - self._source[j]
                r = np.linalg.norm(r_vec)
                
                if r > 1e-10:
                    if self._kernel == 'cubic':
                        # ∂(|r|³)/∂x = 3|r|·r̂·r̂^T = 3r·r^T/|r|
                        dU = 3 * np.outer(r_vec, r_vec) / r
                    else:
                        dU = np.eye(3)  # 简化处理
                    
                    J += self._weights[j] * dU
            
            jacobians[i] = J
        
        return jacobians
    
    def get_affine_matrix(self) -> np.ndarray:
        """
        获取仿射变换矩阵
        
        返回:
            4×3 仿射矩阵
            [a₀]   [1, x, y, z] [a₁]
            [a₁] =          · [a₂]
            [a₂]            [a₃]
                               [a₄]
        """
        if self._affine is None:
            raise ValueError("TPS not fitted")
        return self._affine
    
    def get_weights(self) -> np.ndarray:
        """获取核函数权重"""
        if self._weights is None:
            raise ValueError("TPS not fitted")
        return self._weights


def interpolate_tps3d(
    source: np.ndarray,
    target: np.ndarray,
    query_points: np.ndarray,
    kernel: str = 'cubic'
) -> np.ndarray:
    """
    3D TPS插值的便捷函数
    
    参数:
        source: 源点 (n, 3)
        target: 目标点 (n, 3)
        query_points: 查询点 (m, 3)
        kernel: 核函数类型
    
    返回:
        插值结果 (m, 3)
    """
    tps = TPS3D(kernel=kernel)
    tps.fit(source, target)
    return tps.transform(query_points)
