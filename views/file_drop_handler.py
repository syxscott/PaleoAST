# views/file_drop_handler.py
"""
File Drop Handler for PaleoAST

Provides drag-and-drop functionality for loading various file formats.
Supports:
    - CSV files
    - Excel files (.xlsx, .xls)
    - TPS files (morphometrics)
    - PAST .dat files
    - Newick tree files (.nwk)

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
import os
import threading
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class FileDropHandler(QObject):
    """
    Handler for file drag-and-drop operations.

    Signals:
        file_loaded: Emitted when a file is successfully loaded (data, file_type)
        load_failed: Emitted when file loading fails (error_message)
    """

    file_loaded = pyqtSignal(object, str)  # (data, file_type)
    load_failed = pyqtSignal(str)  # (error_message)

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.tps': 'tps',
        '.dat': 'dat',
        '.nwk': 'newick',
        '.tree': 'newick',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.FileDropHandler")

    def can_handle(self, file_path: str) -> bool:
        """
        Check if a file can be handled by this drop handler.

        Parameters:
            file_path: Path to the file

        Returns:
            True if file type is supported
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def get_file_type(self, file_path: str) -> str:
        """
        Get the type of a file based on its extension.

        Parameters:
            file_path: Path to the file

        Returns:
            File type string (csv, excel, tps, dat, newick)
        """
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, 'unknown')

    def handle_file(self, file_path: str) -> Optional[object]:
        """
        Handle a dropped file and parse it.

        Parameters:
            file_path: Path to the file

        Returns:
            Parsed data object or None on failure
        """
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            self._logger.error(error_msg)
            self.load_failed.emit(error_msg)
            return None

        file_type = self.get_file_type(file_path)
        self._logger.info(f"Handling dropped file: {file_path} (type: {file_type})")

        try:
            if file_type == 'csv':
                data = self._parse_csv(file_path)
            elif file_type == 'excel':
                data = self._parse_excel(file_path)
            elif file_type == 'tps':
                data = self._parse_tps(file_path)
            elif file_type == 'dat':
                data = self._parse_dat(file_path)
            elif file_type == 'newick':
                data = self._parse_newick(file_path)
            else:
                error_msg = f"Unsupported file type: {file_type}"
                self._logger.error(error_msg)
                self.load_failed.emit(error_msg)
                return None

            # Emit success signal so the UI can pick up the loaded data.
            self.file_loaded.emit(data, file_type)
            return data

        except Exception as e:
            error_msg = f"Error parsing {file_type} file: {str(e)}"
            self._logger.error(error_msg)
            self.load_failed.emit(error_msg)
            return None

    def _parse_csv(self, file_path: str) -> dict:
        """Parse a CSV file."""
        import pandas as pd

        df = pd.read_csv(file_path)
        return {
            'type': 'matrix',
            'data': df.values,
            'row_labels': df.index.tolist() if df.index.name else None,
            'col_labels': df.columns.tolist(),
        }

    def _parse_excel(self, file_path: str) -> dict:
        """Parse an Excel file."""
        import pandas as pd

        # Try to read all sheets
        xl = pd.ExcelFile(file_path)
        sheets = {}

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets[sheet_name] = {
                'data': df.values,
                'row_labels': df.index.tolist() if df.index.name else None,
                'col_labels': df.columns.tolist(),
            }

        if len(sheets) == 1:
            # Single sheet, return just that data
            return list(sheets.values())[0]

        return {'type': 'multi_sheet', 'sheets': sheets}

    def _parse_tps(self, file_path: str) -> dict:
        """Parse a TPS file."""
        from parsers.tps_parser import parse_tps_file

        tps_data = parse_tps_file(file_path)

        # Convert to matrix format
        matrix = tps_data.to_matrix()

        return {
            'type': 'tps',
            'data': matrix,
            'specimens': [s.id for s in tps_data.specimens],
            'n_landmarks': tps_data.n_landmarks,
            'n_dimensions': tps_data.n_dimensions,
            'raw_data': tps_data,
        }

    def _parse_dat(self, file_path: str) -> dict:
        """Parse a PAST .dat file."""
        from parsers.dat_parser import parse_dat_file

        dat_data = parse_dat_file(file_path)

        return {
            'type': 'matrix',
            'data': dat_data.data,
            'row_labels': dat_data.row_labels,
            'col_labels': dat_data.col_labels,
            'groups': dat_data.groups,
        }

    def _parse_newick(self, file_path: str) -> dict:
        """Parse a Newick tree file."""
        from parsers.newick_parser import NewickParser

        with open(file_path, 'r', encoding='utf-8') as f:
            newick_string = f.read().strip()

        parser = NewickParser()
        tree = parser.parse(newick_string)

        return {
            'type': 'tree',
            'tree': tree,
            'newick_string': newick_string,
        }


# Module-level lock for thread-safe singleton
_file_drop_handler_lock = threading.Lock()


def get_file_drop_handler(parent=None) -> FileDropHandler:
    """
    Get a singleton file drop handler instance.

    Parameters:
        parent: Parent widget

    Returns:
        FileDropHandler instance
    """
    if not hasattr(FileDropHandler, '_instance'):
        with _file_drop_handler_lock:
            # Double-check locking pattern
            if not hasattr(FileDropHandler, '_instance'):
                FileDropHandler._instance = FileDropHandler(parent)
    return FileDropHandler._instance
