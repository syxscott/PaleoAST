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

from .column_metadata import ColumnMetadata, ColumnMetadataManager
from .data_matrix import DataMatrix, DataMatrixView
from .diversity_result import DiversityIndexResult, DiversityResult
from .row_metadata import RowMetadata, RowMetadataManager
from .state_manager import StateManager, get_state_manager

__all__ = [
    "ColumnMetadata",
    "ColumnMetadataManager",
    "DataMatrix",
    "DataMatrixView",
    "DiversityIndexResult",
    "DiversityResult",
    "RowMetadata",
    "RowMetadataManager",
    "StateManager",
    "get_state_manager",
]
