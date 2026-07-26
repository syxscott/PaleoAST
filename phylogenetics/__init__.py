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
- 系统发育独立对比 (PIC, Felsenstein 1985)
- 系统发育信号 (Blomberg's K, Pagel's λ)
- Bootstrap分析

作者: PaleoAST Development Team
版本: 3.0.0
"""

from .distance_methods import UPGMA, NeighborJoining
from .fitch import FitchAlgorithm
from .heuristic_search import HeuristicSearch, NNIOperation, TBROperation
from .pic import (
    PICNodeData,
    compute_pic,
    compute_pic_with_ancestral_states,
    validate_pic_assumptions,
)
from .signal import (
    PhylogeneticSignalResult,
    blomberg_k,
    lambda_interpretation,
    pagel_lambda,
    phylogenetic_signal,
    simulate_brownian_motion,
)
from .strict_consensus import StrictConsensusTree
from .tree import PhyloNode, PhyloTree

__all__ = [
    "UPGMA",
    "FitchAlgorithm",
    "HeuristicSearch",
    "NNIOperation",
    "NeighborJoining",
    "PhyloNode",
    "PhyloTree",
    "StrictConsensusTree",
    "TBROperation",
    # PIC (Felsenstein 1985)
    "PICNodeData",
    "compute_pic",
    "compute_pic_with_ancestral_states",
    "validate_pic_assumptions",
    # Phylogenetic signal (Blomberg et al. 2003, Pagel 1999)
    "PhylogeneticSignalResult",
    "blomberg_k",
    "pagel_lambda",
    "phylogenetic_signal",
    "simulate_brownian_motion",
    "lambda_interpretation",
]
