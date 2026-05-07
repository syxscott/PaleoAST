"""
================================================================================
PaleoAST Phase 4 - 3D Morphometrics, Macroevolution & Validation
================================================================================

本阶段包含：
- 3D几何形态测量学引擎 (Sliding Semi-landmarks, Quaternion rotation)
- 宏观演化动力学模型 (FBD, Cohort Survivorship)
- 企业级自动化验证框架 (Chaos TDD)

作者: PaleoAST Development Team
版本: 4.0.0
"""

from . import morpho3d
from . import macroevolution
from . import tests

__all__ = ['morpho3d', 'macroevolution', 'tests']
