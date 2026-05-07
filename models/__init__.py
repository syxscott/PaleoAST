# =============================================================================
# FILE: models/__init__.py
# =============================================================================
"""
PaleoAST Models Package

This package contains data model classes for managing paleontological data,
including the core DataMatrix class, column/row metadata, diversity results,
and the thread-safe StateManager.

Author: PaleoAST Development Team
Version: 1.0.0
"""

from .data_matrix import DataMatrix, DataMatrixView
from .column_metadata import ColumnMetadata, ColumnMetadataManager
from .row_metadata import RowMetadata, RowMetadataManager
from .diversity_result import DiversityResult, DiversityIndexResult
from .state_manager import StateManager, get_state_manager

__all__ = [
    'DataMatrix',
    'DataMatrixView',
    'ColumnMetadata',
    'ColumnMetadataManager',
    'RowMetadata',
    'RowMetadataManager',
    'DiversityResult',
    'DiversityIndexResult',
    'StateManager',
    'get_state_manager',
]
