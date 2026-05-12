# =============================================================================
# FILE: models/column_metadata.py
# =============================================================================
"""
Column Metadata Module for PaleoAST

This module implements the ColumnMetadata class for storing per-column
properties such as data type, grouping information, and visualization settings.

Data Types:
    - Nominal: Categorical data without order (e.g., species names)
    - Ordinal: Categorical data with meaningful order (e.g., stages)
    - Continuous: Quantitative real-valued measurements
    - Binary: Presence/absence data (0/1)
    - Count: Non-negative integer counts

Author: PaleoAST Development Team
Version: 1.0.0
"""

import threading
from dataclasses import dataclass
from typing import Any

from config.colors import CHART_COLORS, CHART_MARKERS
from config.constants import DataType
from utils.exceptions import DataValidationError
from utils.validators import validate_column_metadata


@dataclass(frozen=True)
class ColumnMetadata:
    """
    Immutable metadata for a single column in the DataMatrix.

    This dataclass stores properties that describe how to interpret
    and visualize a particular column, including its data type,
    grouping assignment, and graphical attributes.

    Attributes:
        column_index: Zero-based index of this column
        name: Human-readable column name (from column labels)
        data_type: Statistical data type (nominal, ordinal, continuous, etc.)
        group: Optional group assignment for ANOVA/grouping analyses
        color: Hex color code for visualization (e.g., "#0077BB")
        marker: Matplotlib marker style for scatter plots
        description: Optional text description of the column
        units: Optional measurement units (e.g., "mm", "Ma")

    Mathematical Classification:
        - Nominal: Discrete categories, no order (categorical)
        - Ordinal: Discrete categories with order (ranked)
        - Continuous: Real-valued measurements (quantitative)
        - Binary: 0/1 presence-absence data
        - Count: Non-negative integers (frequency data)

    Example:
        >>> meta = ColumnMetadata(
        ...     column_index=0,
        ...     name="Specimen_Length",
        ...     data_type=DataType.CONTINUOUS,
        ...     group="Treatment_A",
        ...     color="#0077BB",
        ...     marker="o",
        ...     units="mm"
        ... )
        >>> meta.is_numeric
        True
    """

    column_index: int
    name: str
    data_type: str = DataType.CONTINUOUS
    group: str | None = None
    color: str = "#0077BB"
    marker: str = "o"
    description: str | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        """
        Validate metadata fields after initialization.

        Raises:
            DataValidationError: If any field contains invalid values
        """
        # Validate data type
        valid_types = [DataType.NOMINAL, DataType.ORDINAL, DataType.CONTINUOUS, DataType.BINARY, DataType.COUNT]
        if self.data_type not in valid_types:
            raise DataValidationError(
                f"Invalid data type: '{self.data_type}'",
                details={"provided": self.data_type, "valid_types": valid_types},
            )

        # Validate color format
        if not self._is_valid_hex_color(self.color):
            raise DataValidationError(f"Invalid color format: '{self.color}'", details={"expected": "#RRGGBB or #RGB"})

        # Validate marker
        if self.marker not in CHART_MARKERS:
            raise DataValidationError(f"Invalid marker: '{self.marker}'", details={"valid_markers": CHART_MARKERS})

    @staticmethod
    def _is_valid_hex_color(color: str) -> bool:
        """
        Check if color string is valid hex format.

        Parameters:
            color: Color string to validate

        Returns:
            bool: True if valid hex color
        """
        if not isinstance(color, str):
            return False
        color = color.lstrip("#")
        if len(color) not in (3, 6):
            return False
        try:
            int(color, 16)
            return True
        except ValueError:
            return False

    @property
    def is_numeric(self) -> bool:
        """
        Check if this column contains numeric data.

        Returns:
            bool: True if data type is continuous, binary, or count
        """
        return self.data_type in [DataType.CONTINUOUS, DataType.BINARY, DataType.COUNT]

    @property
    def is_categorical(self) -> bool:
        """
        Check if this column contains categorical data.

        Returns:
            bool: True if data type is nominal or ordinal
        """
        return self.data_type in [DataType.NOMINAL, DataType.ORDINAL]

    @property
    def is_group_column(self) -> bool:
        """
        Check if this column is designated as a group/factor column.

        Returns:
            bool: True if group is set
        """
        return self.group is not None

    def with_updates(self, **kwargs) -> "ColumnMetadata":
        """
        Create a new ColumnMetadata with updated fields.

        This allows creating modified copies without directly
        mutating the frozen dataclass.

        Parameters:
            **kwargs: Field names and new values to update

        Returns:
            ColumnMetadata: New instance with updated fields
        """
        current_values = {
            "column_index": self.column_index,
            "name": self.name,
            "data_type": self.data_type,
            "group": self.group,
            "color": self.color,
            "marker": self.marker,
            "description": self.description,
            "units": self.units,
        }
        current_values.update(kwargs)
        return ColumnMetadata(**current_values)


class ColumnMetadataManager:
    """
    Manager class for handling column metadata in the DataMatrix.

    This class provides thread-safe access to column metadata,
    supporting operations like adding, updating, and querying
    column properties.

    Attributes:
        metadata: Dictionary mapping column index to ColumnMetadata

    Example:
        >>> manager = ColumnMetadataManager(n_columns=5)
        >>> manager.set_data_type(0, DataType.CONTINUOUS)
        >>> manager.set_group(0, "Control")
        >>> manager.set_color(1, "#EE7733")
        >>> manager.get_metadata(0)
        ColumnMetadata(column_index=0, name='Var_1', data_type='continuous', ...)
    """

    def __init__(self, n_columns: int, column_labels: list[str] | None = None) -> None:
        """
        Initialize the ColumnMetadataManager.

        Parameters:
            n_columns: Number of columns in the data matrix
            column_labels: Optional list of column names/labels
        """
        self._n_columns = n_columns
        self._column_labels = column_labels or [f"Var_{i + 1}" for i in range(n_columns)]
        self._metadata: dict[int, ColumnMetadata] = {}
        self._lock = threading.RLock()

        # Initialize default metadata for all columns
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """
        Initialize default metadata for all columns.
        """
        for i in range(self._n_columns):
            if i not in self._metadata:
                self._metadata[i] = ColumnMetadata(
                    column_index=i,
                    name=self._column_labels[i],
                    data_type=DataType.CONTINUOUS,
                    color=CHART_COLORS[i % len(CHART_COLORS)],
                    marker=CHART_MARKERS[i % len(CHART_MARKERS)],
                )

    def get_metadata(self, column_index: int) -> ColumnMetadata | None:
        """
        Get metadata for a specific column.

        Parameters:
            column_index: Index of the column

        Returns:
            Optional[ColumnMetadata]: Metadata if exists, None otherwise
        """
        with self._lock:
            return self._metadata.get(column_index)

    def set_metadata(self, column_index: int, metadata: ColumnMetadata) -> None:
        """
        Set metadata for a column.

        Parameters:
            column_index: Index of the column
            metadata: New metadata to set

        Raises:
            IndexError: If column_index out of range
        """
        with self._lock:
            if column_index < 0 or column_index >= self._n_columns:
                raise IndexError(f"Column index {column_index} out of range [0, {self._n_columns})")
            self._metadata[column_index] = metadata

    def set_data_type(self, column_index: int, data_type: str) -> None:
        """
        Set the data type for a column.

        Parameters:
            column_index: Index of the column
            data_type: New data type (from DataType enum)
        """
        with self._lock:
            meta = self._metadata.get(column_index)
            if meta is None:
                meta = ColumnMetadata(column_index=column_index, name=self._column_labels[column_index])

            new_meta = meta.with_updates(data_type=data_type)
            self._metadata[column_index] = new_meta

    def set_group(self, column_index: int, group: str | None) -> None:
        """
        Set the group assignment for a column.

        Parameters:
            column_index: Index of the column
            group: Group name, or None to remove group assignment
        """
        with self._lock:
            meta = self._metadata.get(column_index)
            if meta is None:
                meta = ColumnMetadata(column_index=column_index, name=self._column_labels[column_index])

            new_meta = meta.with_updates(group=group)
            self._metadata[column_index] = new_meta

    def set_color(self, column_index: int, color: str) -> None:
        """
        Set the visualization color for a column.

        Parameters:
            column_index: Index of the column
            color: Hex color code (e.g., "#0077BB")
        """
        with self._lock:
            meta = self._metadata.get(column_index)
            if meta is None:
                meta = ColumnMetadata(column_index=column_index, name=self._column_labels[column_index])

            new_meta = meta.with_updates(color=color)
            self._metadata[column_index] = new_meta

    def set_marker(self, column_index: int, marker: str) -> None:
        """
        Set the visualization marker for a column.

        Parameters:
            column_index: Index of the column
            marker: Matplotlib marker style
        """
        with self._lock:
            meta = self._metadata.get(column_index)
            if meta is None:
                meta = ColumnMetadata(column_index=column_index, name=self._column_labels[column_index])

            new_meta = meta.with_updates(marker=marker)
            self._metadata[column_index] = new_meta

    def get_columns_by_data_type(self, data_type: str) -> list[int]:
        """
        Get indices of all columns with specified data type.

        Parameters:
            data_type: Data type to filter by

        Returns:
            List[int]: Column indices with matching data type
        """
        with self._lock:
            return [idx for idx, meta in self._metadata.items() if meta.data_type == data_type]

    def get_columns_by_group(self, group: str) -> list[int]:
        """
        Get indices of all columns in specified group.

        Parameters:
            group: Group name to filter by

        Returns:
            List[int]: Column indices in the specified group
        """
        with self._lock:
            return [idx for idx, meta in self._metadata.items() if meta.group == group]

    def get_numeric_columns(self) -> list[int]:
        """
        Get indices of all numeric columns.

        Returns:
            List[int]: Column indices with numeric data types
        """
        with self._lock:
            return [idx for idx, meta in self._metadata.items() if meta.is_numeric]

    def get_categorical_columns(self) -> list[int]:
        """
        Get indices of all categorical columns.

        Returns:
            List[int]: Column indices with categorical data types
        """
        with self._lock:
            return [idx for idx, meta in self._metadata.items() if meta.is_categorical]

    def get_group_columns(self) -> list[int]:
        """
        Get indices of all columns designated as group/factor columns.

        Returns:
            List[int]: Column indices that are group columns
        """
        with self._lock:
            return [idx for idx, meta in self._metadata.items() if meta.is_group_column]

    def get_all_groups(self) -> list[str]:
        """
        Get list of all unique group names.

        Returns:
            List[str]: Sorted list of unique group names
        """
        with self._lock:
            groups = set(meta.group for meta in self._metadata.values() if meta.group is not None)
            return sorted(list(groups))

    def get_column_colors(self) -> dict[int, str]:
        """
        Get color mapping for all columns.

        Returns:
            Dict[int, str]: Column index to color mapping
        """
        with self._lock:
            return {idx: meta.color for idx, meta in self._metadata.items()}

    def get_column_markers(self) -> dict[int, str]:
        """
        Get marker mapping for all columns.

        Returns:
            Dict[int, str]: Column index to marker mapping
        """
        with self._lock:
            return {idx: meta.marker for idx, meta in self._metadata.items()}

    def get_data_types(self) -> dict[int, str]:
        """
        Get data type mapping for all columns.

        Returns:
            Dict[int, str]: Column index to data type mapping
        """
        with self._lock:
            return {idx: meta.data_type for idx, meta in self._metadata.items()}

    def get_groups(self) -> dict[int, str | None]:
        """
        Get group mapping for all columns.

        Returns:
            Dict[int, Optional[str]]: Column index to group mapping
        """
        with self._lock:
            return {idx: meta.group for idx, meta in self._metadata.items()}

    def to_dict(self) -> dict[int, dict[str, Any]]:
        """
        Convert all metadata to dictionary format.

        Returns:
            Dict: Metadata as dictionary
        """
        with self._lock:
            return {
                idx: {
                    "name": meta.name,
                    "data_type": meta.data_type,
                    "group": meta.group,
                    "color": meta.color,
                    "marker": meta.marker,
                    "description": meta.description,
                    "units": meta.units,
                }
                for idx, meta in self._metadata.items()
            }

    def from_dict(self, metadata_dict: dict[int, dict[str, Any]]) -> None:
        """
        Load metadata from dictionary format.

        Parameters:
            metadata_dict: Dictionary containing column metadata
        """
        with self._lock:
            for idx, meta_dict in metadata_dict.items():
                self._metadata[idx] = ColumnMetadata(
                    column_index=idx,
                    name=meta_dict.get("name", self._column_labels[idx]),
                    data_type=meta_dict.get("data_type", DataType.CONTINUOUS),
                    group=meta_dict.get("group"),
                    color=meta_dict.get("color", CHART_COLORS[idx % len(CHART_COLORS)]),
                    marker=meta_dict.get("marker", CHART_MARKERS[idx % len(CHART_MARKERS)]),
                    description=meta_dict.get("description"),
                    units=meta_dict.get("units"),
                )

    def validate(self) -> bool:
        """
        Validate all metadata entries.

        Returns:
            bool: True if all metadata is valid

        Raises:
            DataValidationError: If validation fails
        """
        with self._lock:
            return validate_column_metadata(
                {
                    idx: {"type": meta.data_type, "color": meta.color, "marker": meta.marker}
                    for idx, meta in self._metadata.items()
                },
                n_cols=self._n_columns,
            )

    def __repr__(self) -> str:
        """
        String representation.

        Returns:
            str: Description of the manager
        """
        with self._lock:
            n_groups = len(self.get_all_groups())
            return f"ColumnMetadataManager(n_columns={self._n_columns}, n_groups={n_groups})"
