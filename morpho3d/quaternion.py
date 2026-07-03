"""
================================================================================
PaleoAST 3D Morphometrics - Quaternion and Rotation Module
================================================================================

本模块实现基于四元数的3D旋转数学工具。

数学理论:
================================================================================

1. 四元数定义
--------------------------------------------------------------------------------
四元数 q ∈ ℍ 定义为:

    q = w + xi + yj + zk

其中 w, x, y, z ∈ ℝ，i² = j² = k² = ijk = -1

四元数可以表示3D旋转:

    q = [w, x, y, z] = [cos(θ/2), sin(θ/2)u]

其中 θ 为旋转角，u 为单位旋转轴。

2. 四元数乘法 (Hamilton积)
--------------------------------------------------------------------------------
设 q₁ = w₁ + x₁i + y₁j + z₁k，q₂ = w₂ + x₂i + y₂j + z₂k

q₁q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂)
      + (w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i
      + (w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂)j
      + (w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)k

3. 旋转矩阵转换
--------------------------------------------------------------------------------
给定四元数 q = [w, x, y, z]，对应的旋转矩阵 R ∈ SO(3):

    R = ┌ 1-2(y²+z²)    2(xy-wz)      2(xz+wy)   ┐
        │   2(xy+wz)    1-2(x²+z²)    2(yz-wx)   │
        └   2(xz-wy)      2(yz+wx)    1-2(x²+y²) ┘

4. 旋转矩阵正交性验证
--------------------------------------------------------------------------------
R ∈ SO(3) 满足:
- R^T R = I (正交)
- det(R) = +1 (右手系)
- R⁻¹ = R^T

5. SVD旋转对齐
--------------------------------------------------------------------------------
给定两组3D点集 X, Y，最优旋转 R 使得:

    R* = arg min_R ||RX - Y||_F

解为 R* = UV^T，其中 UΣV^T = YX^T 的SVD分解。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Quaternion:
    """
    四元数类

    表示3D旋转的四元数 q = w + xi + yj + zk

    属性:
        w: 标量部分 (cos(θ/2))
        x: i分量 (sin(θ/2) * u_x)
        y: j分量 (sin(θ/2) * u_y)
        z: k分量 (sin(θ/2) * u_z)

    示例:
        >>> q = Quaternion.from_axis_angle([0, 0, 1], np.pi/2)
        >>> print(q.to_rotation_matrix())
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self):
        """四元数验证与归一化

        Historically the constructor left components untouched so that
        intermediate products could be computed without surprise
        renormalisations. The downside was that ``Quaternion(1, 2, 3, 4)``
        silently kept a non-unit magnitude, which made rotation math
        wrong whenever the caller forgot to call ``.normalized()``.

        We now auto-normalise on construction (after the zero-norm
        guard) so the public API matches user expectations: building
        a quaternion always yields a unit quaternion. Callers that
        need the raw components should use ``Quaternion.from_raw`` or
        simply operate on ``np.ndarray`` directly.
        """
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            logger.error("Quaternion magnitude too small")
            raise ValueError("Quaternion magnitude too small")
        # Auto-normalise to unit quaternion. This is safe because
        # ``__post_init__`` runs before the dataclass freezes the
        # fields, so the assignment below is the final value.
        self.w = self.w / norm
        self.x = self.x / norm
        self.y = self.y / norm
        self.z = self.z / norm
        logger.debug(f"Quaternion created: w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}")

    @classmethod
    def from_raw(cls, w: float, x: float, y: float, z: float) -> Quaternion:
        """Create a quaternion-like value without automatic normalization.

        This is for intermediate linear algebra operations such as SLERP's
        linear fallback. Public rotation constructors still return unit
        quaternions through ``__post_init__``.
        """
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        if norm < 1e-10:
            raise ValueError("Quaternion magnitude too small")
        q = cls.__new__(cls)
        q.w = float(w)
        q.x = float(x)
        q.y = float(y)
        q.z = float(z)
        return q

    def normalized(self) -> Quaternion:
        """返回归一化后的四元数副本"""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:
            raise ValueError("Quaternion magnitude too small")
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    @property
    def components(self) -> np.ndarray:
        """获取四元数分量 [w, x, y, z]"""
        return np.array([self.w, self.x, self.y, self.z])

    @property
    def vector(self) -> np.ndarray:
        """获取向量部分 [x, y, z]"""
        return np.array([self.x, self.y, self.z])

    @property
    def conjugate(self) -> Quaternion:
        """返回四元数共轭 q* = w - xi - yj - zk

        注意:由于 ``__post_init__`` 会自动归一化,所以这里必须
        一次性把归一化后的值带进去。否则 ``Quaternion(w, -x, ...)``
        会再被除以 ``sqrt(w² + x² + y² + z²)`` (注意:这个范数
        跟原四元数的范数相同,因为只有符号变化),得到
        ``(w/N, -x/N, ...)``,看起来是对的。但若 ``(w, x, y, z)``
        不为单位四元数,自动归一化会引入额外缩放,使返回值不再
        严格满足 ``q_conj.w == q.w``。所以我们手动归一化后构造。
        """
        n = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        return Quaternion(self.w / n, -self.x / n, -self.y / n, -self.z / n)

    @property
    def inverse(self) -> Quaternion:
        """返回四元数逆 q⁻¹ = q*/|q|²"""
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq < 1e-10:
            raise ValueError("Quaternion magnitude too small for inversion")
        return Quaternion(self.w / norm_sq, -self.x / norm_sq, -self.y / norm_sq, -self.z / norm_sq)

    @property
    def magnitude(self) -> float:
        """返回四元数模 |q|"""
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @property
    def rotation_angle(self) -> float:
        """返回旋转角 θ ∈ [0, π]

        注意:quaternion 是双重覆盖的,``q`` 和 ``-q`` 表示同一个
        旋转。如果输入来自 ``from_axis_angle(axis, 2π)`` 这类
        调用,数学上等价于零旋转,但 ``q.w = -1`` 会让原始的
        ``2 * arccos(-1) = 2π`` 越界。我们将其规范化到 [0, π],
        也就是取与原四元数表示同一旋转的"非负半空间"角度。
        """
        angle = 2 * np.arccos(np.clip(self.w, -1.0, 1.0))
        if angle > np.pi:
            # 2π - θ ∈ [0, π) 表示同一旋转
            angle = 2 * np.pi - angle
        return angle

    @property
    def rotation_axis(self) -> np.ndarray:
        """返回旋转轴单位向量"""
        sin_half = np.sqrt(1 - self.w**2)
        if sin_half < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return np.array([self.x, self.y, self.z]) / sin_half

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> Quaternion:
        """
        从旋转轴和角度创建四元数

        参数:
            axis: 旋转轴向量 (3,)
            angle: 旋转角 (弧度)

        返回:
            Quaternion对象

        数学公式:
            q = [cos(θ/2), sin(θ/2) * u]
            其中 u = axis/|axis| 为单位轴向量
        """
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)

        if norm < 1e-10:
            logger.error("Axis vector too small for quaternion creation")
            raise ValueError("Axis vector too small")

        axis = axis / norm
        half_angle = angle / 2.0

        w = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half

        logger.debug(f"Created quaternion from axis={axis}, angle={angle:.4f} rad ({np.degrees(angle):.2f} deg)")
        return cls(w, x, y, z)

    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> Quaternion:
        """
        从旋转矩阵创建四元数

        参数:
            R: 3x3旋转矩阵

        返回:
            Quaternion对象

        算法 (Shepperd, 1978):
            tr = trace(R)
            if tr > 0:
                w = √(1+tr)/2
                ...
            else:
                选择最大对角元分支
        """
        R = np.asarray(R, dtype=np.float64)

        if R.shape != (3, 3):
            logger.error(f"Rotation matrix must be 3x3, got {R.shape}")
            raise ValueError("Rotation matrix must be 3x3")

        # 验证正交性
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
            logger.error("Matrix is not orthogonal")
            raise ValueError("Matrix is not orthogonal")

        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
            logger.error(f"Matrix determinant is not +1, got {np.linalg.det(R)}")
            raise ValueError("Matrix determinant is not +1")

        logger.debug("Converting 3x3 rotation matrix to quaternion (Shepperd method)")
        tr = np.trace(R)

        if tr > 0:
            w = np.sqrt(1 + tr) / 2.0
            w4 = 4 * w
            x = (R[2, 1] - R[1, 2]) / w4
            y = (R[0, 2] - R[2, 0]) / w4
            z = (R[1, 0] - R[0, 1]) / w4
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2.0
            x4 = 4 * x
            w = (R[2, 1] - R[1, 2]) / x4
            y = (R[1, 0] + R[0, 1]) / x4
            z = (R[0, 2] + R[2, 0]) / x4
        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2.0
            y4 = 4 * y
            w = (R[0, 2] - R[2, 0]) / y4
            x = (R[1, 0] + R[0, 1]) / y4
            z = (R[2, 1] + R[1, 2]) / y4
        else:
            z = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2.0
            z4 = 4 * z
            w = (R[1, 0] - R[0, 1]) / z4
            x = (R[0, 2] + R[2, 0]) / z4
            y = (R[2, 1] + R[1, 2]) / z4

        return cls(w, x, y, z)

    def __mul__(self, other: Quaternion) -> Quaternion:
        """
        四元数乘法 (Hamilton积)

        参数:
            other: 另一个四元数

        返回:
            乘积四元数

        数学公式:
            q₁q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂)
                  + (w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i
                  + (w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂)j
                  + (w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)k
        """
        if not isinstance(other, Quaternion):
            return NotImplemented

        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        logger.debug(f"Quaternion multiplication: ({self}) * ({other})")
        return Quaternion(w, x, y, z)

    def __rmul__(self, scalar: float) -> Quaternion:
        """标量乘法"""
        return Quaternion.from_raw(self.w * scalar, self.x * scalar, self.y * scalar, self.z * scalar)

    def __add__(self, other: Quaternion) -> Quaternion:
        """四元数加法"""
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion.from_raw(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __neg__(self) -> Quaternion:
        """取负"""
        return Quaternion.from_raw(-self.w, -self.x, -self.y, -self.z)

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        使用四元数旋转3D向量

        参数:
            v: 待旋转向量 (3,)

        返回:
            旋转后向量 (3,),|v'| = |v|

        数学公式:
            v' = q * [0, v] * q⁻¹

        其中 [0, v] 是将向量表示为纯四元数。注意:Quaternion
        的 ``__post_init__`` 会自动归一化输入分量,所以不能
        直接 ``Quaternion(0, v[0], v[1], v[2])`` 那样用,
        否则向量的模长会被压缩为 1。我们改用 v' = R(q)·v
        (旋转矩阵) 或者展开的 q·q_v·q⁻¹ 公式直接算向量部分。
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (3,):
            logger.error(f"Vector must be shape (3,), got {v.shape}")
            raise ValueError("Vector must be shape (3,)")

        logger.debug(f"Rotating vector {v} by quaternion {self}")

        # 展开 q · (0, v) · q⁻¹ 的向量部分,避免构造纯四元数
        # (构造纯四元数会被归一化,丢失 |v|)。
        # 标准推导:
        #   t = 2 * cross(q.xyz, v)
        #   v' = v + q.w * t + cross(q.xyz, t)
        qv = np.array([self.x, self.y, self.z])
        t = 2.0 * np.cross(qv, v)
        rotated = v + self.w * t + np.cross(qv, t)
        logger.debug(f"Rotated vector: {rotated}")
        return rotated

    def to_rotation_matrix(self) -> np.ndarray:
        """
        转换为3x3旋转矩阵

        返回:
            旋转矩阵 R ∈ SO(3)

        数学公式:
            R = ┌ 1-2(y²+z²)    2(xy-wz)      2(xz+wy)   ┐
                │   2(xy+wz)    1-2(x²+z²)    2(yz-wx)   │
                └   2(xz-wy)      2(yz+wx)    1-2(x²+y²) ┘
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float64,
        )

        return R

    def to_zyz_euler(self) -> tuple[float, float, float]:
        """
        转换为ZYZ欧拉角

        返回:
            (α, β, γ) 弧度

        数学公式:
            q = [cos(β/2), 0, 0, sin(β/2)] * [cos(α/2), 0, 0, sin(α/2)]
                * [cos(γ/2), 0, 0, sin(γ/2)]
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # 计算 β
        cos_beta = w**2 + z**2 - x**2 - y**2
        beta = np.arccos(np.clip(cos_beta, -1, 1))

        if abs(np.sin(beta)) < 1e-10:
            # 万向节锁死情况
            alpha = 0.0
            gamma = 2 * np.arctan2(w, z)
        else:
            np.sin(beta)
            alpha = np.arctan2(2 * (w * x + y * z), w**2 - x**2 - y**2 + z**2)
            gamma = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - y**2 - z**2)

        return (alpha, beta, gamma)

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """
        四元数球面线性插值 (SLERP)

        参数:
            other: 目标四元数
            t: 插值参数 [0, 1]

        返回:
            插值四元数

        数学公式:
            q(t) = (sin((1-t)θ) * q₁ + sin(tθ) * q₂) / sin(θ)
        """
        if not isinstance(other, Quaternion):
            return NotImplemented

        logger.debug(f"SLERP interpolation at t={t:.4f}")
        # 计算夹角
        dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

        # 确保走短弧
        if dot < 0:
            other = -other
            dot = -dot

        if dot > 0.9995:
            # 线性近似（只归一化最终结果）
            q = self + t * (other + (-self))
            return q.normalized()

        theta_0 = np.arccos(np.clip(dot, -1, 1))
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion(
            s0 * self.w + s1 * other.w,
            s0 * self.x + s1 * other.x,
            s0 * self.y + s1 * other.y,
            s0 * self.z + s1 * other.z,
        )

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

    def __str__(self) -> str:
        return f"{self.w:.4f} + {self.x:.4f}i + {self.y:.4f}j + {self.z:.4f}k"


class RotationMatrix:
    """
    3D旋转矩阵工具类

    提供旋转矩阵的各种操作和验证。
    """

    @staticmethod
    def from_svd(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        使用SVD计算两组点集之间的最优旋转

        参数:
            X: 参考点集 (N, 3)
            Y: 目标点集 (N, 3)

        返回:
            最优旋转矩阵 R (3, 3)

        数学公式:
            R* = arg min_R ||RX - Y||_F

            解: R* = U V^T

            其中 UΣV^T = YX^T 的SVD分解。

        验证:
            - R^T R = I
            - det(R) = +1
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.shape != Y.shape:
            logger.error(f"Shape mismatch: {X.shape} vs {Y.shape}")
            raise ValueError(f"Shape mismatch: {X.shape} vs {Y.shape}")

        if X.ndim != 2 or X.shape[1] != 3:
            logger.error(f"Points must be shape (N, 3), got {X.shape}")
            raise ValueError("Points must be shape (N, 3)")

        logger.info(f"Computing optimal rotation via SVD for {X.shape[0]} point pairs")
        # 计算协方差矩阵
        H = Y.T @ X

        # SVD分解
        U, _S, Vt = np.linalg.svd(H)

        # 计算R
        R = U @ Vt

        # 处理反射情况 (det(R) = -1)。
        # 关键点:只能翻转 U 的最后一列 *或* Vt 的最后一行,不能同时翻转两者,
        # 否则 det(R) 会被翻两次重新变回 -1。这是经典的 SVD 反射修正陷阱。
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # 验证正交性和行列式
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
            raise ValueError("Result is not orthogonal")

        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
            raise ValueError(f"SVD result determinant is not +1, got {np.linalg.det(R)}")

        logger.info("SVD rotation computation complete, matrix verified orthogonal with det=+1")
        return R

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        从轴角创建旋转矩阵

        参数:
            axis: 旋转轴 (3,)
            angle: 旋转角 (弧度)

        返回:
            旋转矩阵

        数学公式 (Rodrigues公式):
            R = I + sin(θ)K + (1-cos(θ))K²

            其中 K = [0, -k_z, k_y; k_z, 0, -k_x; -k_y, k_x, 0]
                  为轴向量的反对称矩阵
        """
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)

        if norm < 1e-10:
            logger.debug("Zero-length axis, returning identity matrix")
            return np.eye(3)

        axis = axis / norm
        logger.debug(f"Creating rotation matrix from axis={axis}, angle={angle:.4f} rad")

        # 反对称矩阵
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)

        # Rodrigues公式
        I = np.eye(3)
        c = np.cos(angle)
        s = np.sin(angle)
        R = I + s * K + (1 - c) * (K @ K)

        return R

    @staticmethod
    def verify_orthogonal(R: np.ndarray, atol: float = 1e-8) -> bool:
        """
        验证旋转矩阵的正交性

        参数:
            R: 3x3矩阵
            atol: 绝对误差容忍度

        返回:
            是否正交
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            return False

        return np.allclose(R @ R.T, np.eye(3), atol=atol)

    @staticmethod
    def verify_special(R: np.ndarray, atol: float = 1e-8) -> bool:
        """
        验证旋转矩阵是特殊正交群SO(3)成员

        参数:
            R: 3x3矩阵
            atol: 绝对误差容忍度

        返回:
            是否属于SO(3)
        """
        R = np.asarray(R, dtype=np.float64)

        # 检查正交性
        if not RotationMatrix.verify_orthogonal(R, atol):
            return False

        # 检查行列式
        det = np.linalg.det(R)
        return np.isclose(det, 1.0, atol=atol)

    @staticmethod
    def compose(*rotations: np.ndarray) -> np.ndarray:
        """
        组合多个旋转

        参数:
            *rotations: 旋转矩阵序列

        返回:
            组合后的旋转矩阵

        注意: 旋转从右到左应用
        """
        if not rotations:
            return np.eye(3)

        result = np.eye(3)
        for R in rotations:
            result = R @ result

        return result

    @staticmethod
    def inverse(R: np.ndarray) -> np.ndarray:
        """
        计算旋转矩阵的逆

        参数:
            R: 旋转矩阵

        返回:
            逆矩阵 = 转置
        """
        R = np.asarray(R, dtype=np.float64)
        return R.T
