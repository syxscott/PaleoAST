"""
================================================================================
PaleoAST 3D Morphometrics - Module Initialization
================================================================================

三维几何形态测量学引擎，支持：
- 3D广义普氏分析 (Quaternion rotation)
- 曲线/曲面半标志点滑动算法
- 3D薄板样条 (TPS)
- 空间形变网格渲染数据

数学基础:
    - 旋转矩阵: SO(3)群
    - 四元数: q = w + xi + yj + zk
    - TPS核函数: U(r) = |r|²ln|r|

作者: PaleoAST Development Team
版本: 4.0.0
"""

from .gpa3d import GPA3D, GPA3DResult
from .mesh import Mesh3D, SurfaceInterpolator
from .quaternion import Quaternion, RotationMatrix
from .sliding import SemiLandmarkSlider, SlidingResult
from .tps3d import TPS3D, TPS3DResult

__all__ = [
    "GPA3D",
    "TPS3D",
    "GPA3DResult",
    "Mesh3D",
    "Quaternion",
    "RotationMatrix",
    "SemiLandmarkSlider",
    "SlidingResult",
    "SurfaceInterpolator",
    "TPS3DResult",
]
