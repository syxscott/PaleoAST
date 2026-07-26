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
version: 1.0.1
"""

import logging
import os
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class DATParseError(Exception):
    """Exception raised when DAT file parsing fails.

    Attributes:
        file_path: Path to the file that failed to parse
        line_number: Line number where the error occurred (1-indexed)
        line_content: The original line content that caused the error
        expected_fields: Expected number of fields
        actual_fields: Actual number of fields
        message: Detailed error message
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line_number: int = 0,
        line_content: str = "",
        expected_fields: int = 0,
        actual_fields: int = 0,
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.line_content = line_content
        self.expected_fields = expected_fields
        self.actual_fields = actual_fields
        full_message = f"DAT Parse Error"
        if file_path:
            full_message += f" in {os.path.basename(file_path)}"
        if line_number > 0:
            full_message += f" at line {line_number}"
        full_message += f": {message}"
        if expected_fields > 0 and actual_fields > 0:
            full_message += f" (expected {expected_fields} fields, got {actual_fields})"
        if line_content:
            full_message += f" (line: {line_content[:50]}{'...' if len(line_content) > 50 else ''})"
        super().__init__(full_message)


@dataclass
class PASTData:
    """Container for parsed PAST data."""

    data: np.ndarray
    row_labels: list[str] | None = None
    col_labels: list[str] | None = None
    groups: list[str] | None = None
    comments: list[str] | None = None
    file_path: str | None = None

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
        (r"^#(.+)$", re.compile(r"^#(.+)$")),
        (r"^\[(.+)\]$", re.compile(r"^\[(.+)\]$")),
        (r"^\{(.+)\}$", re.compile(r"^\{(.+)\}$")),
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
            DATParseError: If file format is invalid (field count mismatch)
            ValueError: If file format is invalid
        """
        self._logger.info(f"Parsing PAST dat file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PAST dat file not found: {file_path}")

        # Detect and handle BOM
        with open(file_path, encoding="utf-8-sig", errors="replace") as f:
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
        expected_field_count = None
        has_found_header_or_data = False

        for line_num, line in enumerate(lines):
            stripped = line.strip()
            original_line = stripped

            # Skip empty lines
            if not stripped:
                continue

            # Handle comments (use pre-compiled patterns)
            # Comment lines starting with #
            if stripped.startswith("#"):
                comments.append(stripped[1:].strip())
                continue

            # Handle PAST-specific comment patterns like [comment] and {group}
            is_comment = False
            for original_pattern, compiled_pattern in self._COMMENT_PATTERNS:
                match = compiled_pattern.match(stripped)
                if match:
                    comments.append(match.group(1))
                    is_comment = True
                    break

            if is_comment:
                # Check if this is a group assignment line like "{Group1}"
                if stripped.startswith("{") and stripped.endswith("}"):
                    group_name = stripped[1:-1]
                    groups.append(group_name)
                    has_groups = True
                continue

            # Try to parse as data
            parts = self._split_line(stripped)

            if len(parts) == 0:
                continue

            # First non-comment, non-empty line might be header
            # Use has_found_header_or_data to track whether we've processed
            # the header line (regardless of line number in file)
            if not has_found_header_or_data and self._looks_like_header(parts):
                # If first column is a label, exclude it from col_labels
                if self._is_label(parts[0]):
                    col_labels = parts[1:]
                else:
                    col_labels = parts
                expected_field_count = len(parts) - 1 if self._is_label(parts[0]) else len(parts)
                has_found_header_or_data = True
                continue

            # Determine field count from header or first data line
            if expected_field_count is None:
                # First data line - determine expected field count
                if self._is_label(parts[0]):
                    expected_field_count = len(parts) - 1
                else:
                    expected_field_count = len(parts)
                has_found_header_or_data = True

            # Check if first element is a label (string) or data
            if self._is_label(parts[0]):
                # First element is row label
                row_labels.append(parts[0])
                data_parts = parts[1:]
            else:
                # No row label
                row_labels.append(f"Row_{line_num + 1}")
                data_parts = parts

            # Strict field count validation
            if len(data_parts) != expected_field_count:
                raise DATParseError(
                    f"Field count mismatch: expected {expected_field_count} data fields, got {len(data_parts)}",
                    file_path=file_path,
                    line_number=line_num + 1,
                    line_content=original_line,
                    expected_fields=expected_field_count,
                    actual_fields=len(data_parts),
                )

            # Parse data values with European thousand separator support
            try:
                # Handle both comma and space separators within data
                # Also handle European format: 1.234,56 -> 1234.56
                data_values = []
                for val in data_parts:
                    parsed_val = self._parse_numeric_value(val)
                    if parsed_val is not None:
                        data_values.append(parsed_val)
                data_lines.append(data_values)
            except ValueError as e:
                raise DATParseError(
                    f"Cannot parse data value: {e}",
                    file_path=file_path,
                    line_number=line_num + 1,
                    line_content=original_line,
                )

        if len(data_lines) == 0:
            raise ValueError("No valid data found in file")

        # Convert to numpy array
        data = np.array(data_lines, dtype=float)

        # Handle column labels
        if col_labels is None:
            col_labels = [f"Col_{i + 1}" for i in range(data.shape[1])]
        elif len(col_labels) != data.shape[1]:
            self._logger.warning(f"Column label count ({len(col_labels)}) doesn't match data columns ({data.shape[1]})")
            # Adjust column labels
            if len(col_labels) > data.shape[1]:
                col_labels = col_labels[: data.shape[1]]
            else:
                col_labels.extend([f"Col_{i + 1}" for i in range(len(col_labels), data.shape[1])])

        # Handle row labels
        if len(row_labels) != data.shape[0]:
            row_labels = [f"Row_{i + 1}" for i in range(data.shape[0])]

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
            file_path=file_path,
        )

    def _split_line(self, line: str) -> list[str]:
        """Split a line by tabs and spaces."""
        # Replace tabs with spaces
        line = line.replace("\t", " ")
        # Split by multiple spaces
        parts = re.split(r"\s+", line)
        return [p.strip() for p in parts if p.strip()]

    def _parse_numeric_value(self, value: str) -> float:
        """Parse a numeric value with European thousand separator support.

        Handles formats:
        - Standard: 1234.56
        - European comma decimal: 1234,56
        - European thousand with comma decimal: 1.234,56 -> 1234.56
        - Trailing comma decimal: 1234, -> 1234.0

        Parameters:
            value: String representation of a number

        Returns:
            Parsed float value

        Raises:
            ValueError: If the value cannot be parsed as a number
        """
        value = value.strip()

        if not value:
            raise ValueError(f"Empty value")

        # Check for NaN indicators
        if value.upper() in ("NAN", "NA", "N/A", "NONE", "NULL", "-"):
            return np.nan

        # Determine the decimal separator
        # Count commas and dots
        comma_count = value.count(",")
        dot_count = value.count(".")

        if comma_count == 0 and dot_count == 0:
            # Simple integer or float without separators
            return float(value)

        if comma_count == 1 and dot_count == 0:
            # Single comma - could be decimal separator (European) or thousand separator
            # European format: 1234,56 -> 1234.56
            # Thousand format: 1,234 -> 1234.0
            parts = value.split(",")
            if len(parts[1]) <= 2 and parts[0].isdigit():
                # Likely decimal separator (European format)
                value = value.replace(",", ".")
            else:
                # Likely thousand separator
                value = value.replace(",", "")
        elif comma_count == 1 and dot_count == 1:
            # Both comma and dot - figure out which is which
            # European format: 1.234,56 -> 1234.56
            # US format: 1,234.56 -> 1234.56
            comma_pos = value.find(",")
            dot_pos = value.find(".")

            if comma_pos > dot_pos:
                # European: dot is thousand, comma is decimal
                value = value.replace(".", "").replace(",", ".")
            else:
                # US: comma is thousand, dot is decimal
                value = value.replace(",", "")
        elif comma_count > 1 and dot_count == 0:
            # Multiple commas - likely European thousand separators
            value = value.replace(",", "")
        elif dot_count > 1 and comma_count == 0:
            # Multiple dots - likely European thousand separators with implicit decimal
            value = value.replace(".", "")

        # Handle trailing comma (European decimal without digits after)
        if value.endswith(","):
            value = value[:-1] + ".0"

        return float(value)

    def _looks_like_header(self, parts: list[str]) -> bool:
        """Check if a line looks like a header (mostly strings)."""
        if len(parts) == 0:
            return False

        # Count numeric vs non-numeric
        numeric_count = 0
        for part in parts:
            try:
                float(part.replace(",", "."))
                numeric_count += 1
            except ValueError:
                pass

        # If more than half are non-numeric, it's likely a header
        return numeric_count < len(parts) / 2

    def _is_label(self, value: str) -> bool:
        """Check if a value looks like a label (non-numeric)."""
        # Remove quotes if present
        value = value.strip("\"'")
        try:
            float(value.replace(",", "."))
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
