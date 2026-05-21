"""
================================================================================
PaleoAST Phase 3 - Custom I/O Parsing Engine
================================================================================

本模块提供从零构建的万能数据格式解析引擎，包括：
- 词法分析器 (Lexer)
- 语法分析器 (Parser)
- NEXUS格式解析
- FASTA格式解析
- PHYLIP格式解析
- Newick树格式解析
- 自定义二进制缓存格式 (.pastx)

所有解析器均为纯Python实现，无外部依赖。

作者: PaleoAST Development Team
版本: 3.0.0
"""

from .binary_cache import BinaryCache, BinaryCacheHeader, ChunkType
from .dat_parser import DATParser, PASTData, parse_dat_file
from .lexer import BaseLexer, LexerError, Token
from .newick_parser import NewickParser, NewickTree, TreeNode
from .nexus_lexer import NexusLexer, NexusTokenType
from .tps_parser import TPSFile, TPSParser, TPSSpecimen, parse_tps_file

__all__ = [
    # 基础词法分析
    "BaseLexer",
    "LexerError",
    "Token",
    # NEXUS格式
    "NexusLexer",
    "NexusTokenType",
    # Newick格式
    "NewickParser",
    "NewickTree",
    "TreeNode",
    # TPS格式 (形态测量学)
    "TPSParser",
    "TPSFile",
    "TPSSpecimen",
    "parse_tps_file",
    # PAST .dat格式
    "DATParser",
    "PASTData",
    "parse_dat_file",
    # 二进制缓存
    "BinaryCache",
    "BinaryCacheHeader",
    "ChunkType",
]
