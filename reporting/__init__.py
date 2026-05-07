"""
================================================================================
PaleoAST Phase 3 - Reporting Module
================================================================================

自动化LaTeX学术报告生成器。

作者: PaleoAST Development Team
版本: 3.0.0
"""

from .latex_preamble import LatexPreamble, DocumentClass
from .table_generator import TableGenerator
from .figure_handler import FigureHandler
from .matrix_converter import MatrixConverter
from .report_builder import ReportBuilder
from .compiler import LatexCompiler

__all__ = [
    'LatexPreamble',
    'DocumentClass',
    'TableGenerator',
    'FigureHandler',
    'MatrixConverter',
    'ReportBuilder',
    'LatexCompiler',
]
