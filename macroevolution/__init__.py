"""
================================================================================
PaleoAST Macroevolution - Module Initialization
================================================================================

宏观演化动力学模块，支持：
- Foote边界交叉法存活分析
- 化石生灭过程(FBD)模拟
- 多样性曲线建模

数学基础:
    - 生灭过程: dN/dt = (λ - μ)N
    - Foote公式: p = (1 - q)/(1 - q^(n+1))
    - MCMC采样

作者: PaleoAST Development Team
版本: 4.0.0
"""

from .cohort import CohortSurvivorshipAnalysis
from .fbd import FossilizedBirthDeathProcess, GillespieSimulator
from .diversity import DiversityDynamics

__all__ = [
    'CohortSurvivorshipAnalysis',
    'FossilizedBirthDeathProcess',
    'GillespieSimulator',
    'DiversityDynamics',
]
