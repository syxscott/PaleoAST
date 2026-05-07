"""
================================================================================
PaleoAST Phase 3 - Phylogenetic Inference Engine
================================================================================

本模块提供从零构建的系统发育树推断引擎，包括：
- 树数据结构 (Tree, Node)
- 最大简约法 (Fitch算法)
- 启发式树搜索 (NNI, TBR)
- 严格一致性树
- 距离法 (UPGMA, NJ)
- 最大似然估计
- Bootstrap分析

作者: PaleoAST Development Team
版本: 3.0.0
"""

from .tree import PhyloTree, PhyloNode
from .fitch import FitchAlgorithm
from .heuristic_search import HeuristicSearch, NNIOperator, TBROperator
from .strict_consensus import StrictConsensusTree
from .distance_methods import UPGMA, NeighborJoining
from .likelihood import MaximumLikelihood, SubstitutionModel
from .bootstrap import BootstrapAnalysis

__all__ = [
    'PhyloTree',
    'PhyloNode',
    'FitchAlgorithm',
    'HeuristicSearch',
    'NNIOperator',
    'TBROperator',
    'StrictConsensusTree',
    'UPGMA',
    'NeighborJoining',
    'MaximumLikelihood',
    'SubstitutionModel',
    'BootstrapAnalysis',
]
