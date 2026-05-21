# parsers/dat_parser.py
"""
PAST .dat File Parser for PaleoAST

Parses .dat format files used by PAST (PAleontological STatistics) software.
PAST is a popular freeware for paleontological data analysis.

PAST .dat format is a simple tab or space-separated format with:
- First line: Optional header with column labels
- Subsequent lines: row label followed by data values
- May include group information in comments

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PASTData:
    """Container for parsed PAST data."""
    data: np.ndarray
    row_labels: Optional[list[str]] = None
    col_labels: Optional[list[str]] = None
    groups: Optional[list[str]] = None
    comments: Optional[list[str]] = None
    file_path: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"PAST Data: {self.data.shape[0]} rows x {self.data.shape[1]} columns",
        ]
        if self.groups is not None:
            unique_groups = set(self.groups)
            lines.append(f"Groups: {len(unique_groups)}")
        return "\n".join(lines)


class DATParser:
    """Parser for PAST .dat format files."""

    # PAST comment patterns (compiled for efficiency)
    _COMMENT_PATTERNS = [
        (r'^#(.+)$', re.compile(r'^#(.+)$')),
        (r'^\[(.+)\]$', re.compile(r'^\[(.+)\]$')),
        (r'^\{(.+)\}$', re.compile(r'^\{(.+)\}$')),
    ]

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.DATParser")

    def parse(self, file_path: str) -> PASTData:
        """
        Parse a PAST .dat file.

        Parameters:
            file_path: Path to the .dat file

        Returns:
            PASTData object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self._logger.info(f"Parsing PAST dat file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PAST dat file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        if len(lines) == 0:
            raise ValueError("Empty file")

        # Parse the file
        data_lines = []
        row_labels = []
        col_labels = None
        groups = []
        comments = []
        has_groups = False

        for line_num, line in enumerate(lines):
            stripped = line.strip()
            original_line = stripped

            # Skip empty lines
            if not stripped:
                continue

            # Handle comments (use pre-compiled patterns)
            is_comment = False
            for original_pattern, compiled_pattern in self._COMMENT_PATTERNS:
                match = compiled_pattern.match(stripped)
                if match:
                    comments.append(match.group(1))
                    is_comment = True
                    break

            if is_comment:
                # Check if this is a group assignment line like "{Group1}"
                if stripped.startswith('{') and stripped.endswith('}'):
                    group_name = stripped[1:-1]
                    groups.append(group_name)
                    has_groups = True
                continue

            # Try to parse as data
            parts = self._split_line(stripped)

            if len(parts) == 0:
                continue

            # First line might be header
            if line_num == 0 and self._looks_like_header(parts):
                col_labels = parts
                continue

            # Check if first element is a label (string) or data
            if self._is_label(parts[0]):
                # First element is row label
                row_labels.append(parts[0])
                data_parts = parts[1:]
            else:
                # No row label
                row_labels.append(f"Row_{line_num + 1}")
                data_parts = parts

            # Parse data values
            try:
                # Handle both comma and space separators within data
                data_values = []
                for val in data_parts:
                    val = val.replace(',', '.').strip()
                    if val:
                        data_values.append(float(val))
                data_lines.append(data_values)
            except ValueError as e:
                self._logger.warning(f"Could not parse line {line_num + 1}: {original_line}")
                continue

        if len(data_lines) == 0:
            raise ValueError("No valid data found in file")

        # Check for consistent row lengths
        if data_lines:
            first_row_len = len(data_lines[0])
            for i, row in enumerate(data_lines):
                if len(row) != first_row_len:
                    self._logger.warning(
                        f"Inconsistent row length at line {i+1}: expected {first_row_len}, got {len(row)}. Padding with NaN."
                    )
                    # Pad short rows with NaN
                    while len(row) < first_row_len:
                        row.append(np.nan)
            # Convert to numpy array (all rows now have same length)
            data = np.array(data_lines, dtype=float)

        # Handle column labels
        if col_labels is None:
            col_labels = [f"Col_{i+1}" for i in range(data.shape[1])]
        elif len(col_labels) != data.shape[1]:
            self._logger.warning(
                f"Column label count ({len(col_labels)}) doesn't match data columns ({data.shape[1]})"
            )
            # Adjust column labels
            if len(col_labels) > data.shape[1]:
                col_labels = col_labels[:data.shape[1]]
            else:
                col_labels.extend([f"Col_{i+1}" for i in range(len(col_labels), data.shape[1])])

        # Handle row labels
        if len(row_labels) != data.shape[0]:
            row_labels = [f"Row_{i+1}" for i in range(data.shape[0])]

        # Handle groups
        if not has_groups or len(groups) == 0:
            groups = None

        self._logger.info(f"Parsed {data.shape[0]} rows x {data.shape[1]} columns")

        return PASTData(
            data=data,
            row_labels=row_labels,
            col_labels=col_labels,
            groups=groups,
            comments=comments if comments else None,
            file_path=file_path
        )

    def _split_line(self, line: str) -> list[str]:
        """Split a line by tabs and spaces."""
        # Replace tabs with spaces
        line = line.replace('\t', ' ')
        # Split by multiple spaces
        parts = re.split(r'\s+', line)
        return [p.strip() for p in parts if p.strip()]

    def _looks_like_header(self, parts: list[str]) -> bool:
        """Check if a line looks like a header (mostly strings)."""
        if len(parts) == 0:
            return False

        # Count numeric vs non-numeric
        numeric_count = 0
        for part in parts:
            try:
                float(part.replace(',', '.'))
                numeric_count += 1
            except ValueError:
                pass

        # If more than half are non-numeric, it's likely a header
        return numeric_count < len(parts) / 2

    def _is_label(self, value: str) -> bool:
        """Check if a value looks like a label (non-numeric)."""
        # Remove quotes if present
        value = value.strip('"\'')
        try:
            float(value.replace(',', '.'))
            return False
        except ValueError:
            return True


def parse_dat_file(file_path: str) -> PASTData:
    """
    Convenience function to parse a PAST .dat file.

    Parameters:
        file_path: Path to the .dat file

    Returns:
        PASTData object
    """
    parser = DATParser()
    return parser.parse(file_path)
