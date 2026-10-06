"""
================================================================================
PaleoAST Parsing Engine
================================================================================

Pure-Python parsers and lexers shipped with PaleoAST. No third-party runtime
deps.

What's actually here
--------------------
- Lexer base class (``BaseLexer``) and ``Token`` / ``LexerError``.
- PAST .dat tabular file parser (``DATParser``).
- TPS (Thin Plate Spline) landmark file parser (``TPSParser``).
- Newick tree parser (``NewickParser``).
- NEXUS lexer (``NexusLexer``) — tokenizer only, no full NEXUS parser.
- Custom binary cache format ``.pastx`` (``BinaryCache``).
- Shared missing-value sentinels (``MISSING_SENTINELS`` etc.) used by every
  parser so ``?``, ``*``, ``NA`` etc. mean the same thing everywhere.

What's NOT here (intentional)
-----------------------------
- A NEXUS parser — only the lexer exists; producing a parsed NEXUS tree
  from arbitrary files is not implemented. Pull in Biopython / ``dendropy``
  if you need full NEXUS support.
- FASTA / PHYLIP parsers — neither file format is supported yet.

作者: PaleoAST Development Team
版本: 3.1.0
"""

from .binary_cache import BinaryCache, BinaryCacheHeader, ChunkType
from .dat_parser import DATParser, PASTData, parse_dat_file
from .lexer import BaseLexer, LexerError, Token
from .newick_parser import NewickParser, NewickTree, TreeNode
from .nexus_lexer import NexusLexer, NexusTokenType
from .sentinels import MISSING_SENTINELS, MISSING_SENTINELS_UPPER, is_missing_token
from .tps_parser import TPSFile, TPSParser, TPSSpecimen, parse_tps_file

__all__ = [
    "MISSING_SENTINELS",
    "MISSING_SENTINELS_UPPER",
    "BaseLexer",
    "BinaryCache",
    "BinaryCacheHeader",
    "ChunkType",
    "DATParser",
    "LexerError",
    "NewickParser",
    "NewickTree",
    "NexusLexer",
    "NexusTokenType",
    "PASTData",
    "TPSFile",
    "TPSParser",
    "TPSSpecimen",
    "Token",
    "TreeNode",
    "is_missing_token",
    "parse_dat_file",
    "parse_tps_file",
]

