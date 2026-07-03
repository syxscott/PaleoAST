# =============================================================================
# FILE: models/row_metadata.py
# =============================================================================
"""
Row Metadata Module for PaleoAST

This module implements the RowMetadata class for storing per-row properties
such as sample labels, group assignments, and visualization settings.

Row Metadata Usage:
    - Sample identification (specimen IDs, site names)
    - Group assignments for multivariate analyses
    - Visualization properties (colors, markers, sizes)
    - Additional sample information (age, location, etc.)

Author: PaleoAST Development Team
version: 1.0.1
"""

import threading
from dataclasses import dataclass
from typing import Any

from config.colors import CHART_COLORS, CHART_MARKERS
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class RowMetadata:
    """
    Immutable metadata for a single row (sample) in the DataMatrix.

    This dataclass stores properties that describe a sample, including
    its identifier, group membership, and visualization attributes.

    Attributes:
        row_index: Zero-based index of this row
        label: Sample identifier (specimen ID, site name, etc.)
        group: Optional group assignment for comparative analyses
        color: Hex color code for sample visualization
        marker: Matplotlib marker style
        size: Marker size for scatter plots (relative scale)
        description: Optional text description of the sample
        age: Optional geological age (for stratigraphic samples)
        location: Optional location information

    Example:
        >>> meta = RowMetadata(
        ...     row_index=0,
        ...     label="Specimen_001",
        ...     group="Jurassic",
        ...     color="#EE7733",
        ...     marker="o",
        ...     size=100,
        ...     age=165.0,
        ...     location="Germany"
        ... )
        >>> meta.is_grouped
        True
    """

    row_index: int
    label: str
    group: str | None = None
    color: str = "#0077BB"
    marker: str = "o"
    size: float = 60.0
    description: str | None = None
    age: float | None = None
    location: str | None = None

    def __post_init__(self) -> None:
        """
        Validate metadata fields after initialization.
        """
        # Validate color format
        if not self._is_valid_hex_color(self.color):
            raise DataValidationError(f"Invalid color format: '{self.color}'", details={"expected": "#RRGGBB or #RGB"})

        # Validate marker
        if self.marker not in CHART_MARKERS:
            raise DataValidationError(f"Invalid marker: '{self.marker}'", details={"valid_markers": CHART_MARKERS})

        # Validate size
        if self.size <= 0:
            raise DataValidationError(f"Marker size must be positive: '{self.size}'")

    @staticmethod
    def _is_valid_hex_color(color: str) -> bool:
        """
        Check if color string is valid hex format.
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
    def is_grouped(self) -> bool:
        """
        Check if this row is assigned to a group.
        """
        return self.group is not None

    @property
    def has_age(self) -> bool:
        """
        Check if this row has age information.
        """
        return self.age is not None

    @property
    def has_location(self) -> bool:
        """
        Check if this row has location information.
        """
        return self.location is not None

    def with_updates(self, **kwargs) -> "RowMetadata":
        """
        Create a new RowMetadata with updated fields.
        """
        current_values = {
            "row_index": self.row_index,
            "label": self.label,
            "group": self.group,
            "color": self.color,
            "marker": self.marker,
            "size": self.size,
            "description": self.description,
            "age": self.age,
            "location": self.location,
        }
        current_values.update(kwargs)
        return RowMetadata(**current_values)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.
        """
        return {
            "label": self.label,
            "group": self.group,
            "color": self.color,
            "marker": self.marker,
            "size": self.size,
            "description": self.description,
            "age": self.age,
            "location": self.location,
        }


class RowMetadataManager:
    """
    Manager class for handling row metadata in the DataMatrix.

    This class provides thread-safe access to row metadata,
    supporting operations like adding, updating, and querying
    row properties.

    Example:
        >>> manager = RowMetadataManager(n_rows=10)
        >>> manager.set_group(0, "Treatment")
        >>> manager.set_color(1, "#EE7733")
        >>> manager.get_all_groups()
        ['Treatment']
    """

    def __init__(self, n_rows: int, row_labels: list[str] | None = None) -> None:
        """
        Initialize the RowMetadataManager.

        Parameters:
            n_rows: Number of rows in the data matrix
            row_labels: Optional list of row names/identifiers
        """
        self._n_rows = n_rows
        self._row_labels = row_labels or [f"Sample_{i + 1}" for i in range(n_rows)]
        self._metadata: dict[int, RowMetadata] = {}
        self._lock = threading.RLock()

        # Initialize default metadata for all rows
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """
        Initialize default metadata for all rows.
        """
        for i in range(self._n_rows):
            if i not in self._metadata:
                self._metadata[i] = RowMetadata(
                    row_index=i,
                    label=self._row_labels[i],
                    color=CHART_COLORS[i % len(CHART_COLORS)],
                    marker=CHART_MARKERS[i % len(CHART_MARKERS)],
                )

    def get_metadata(self, row_index: int) -> RowMetadata | None:
        """
        Get metadata for a specific row.
        """
        with self._lock:
            return self._metadata.get(row_index)

    def set_metadata(self, row_index: int, metadata: RowMetadata) -> None:
        """
        Set metadata for a row.
        """
        with self._lock:
            if row_index < 0 or row_index >= self._n_rows:
                raise IndexError(f"Row index {row_index} out of range [0, {self._n_rows})")
            self._metadata[row_index] = metadata

    def set_label(self, row_index: int, label: str) -> None:
        """
        Set the label for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=label)

            new_meta = meta.with_updates(label=label)
            self._metadata[row_index] = new_meta

    def set_group(self, row_index: int, group: str | None) -> None:
        """
        Set the group assignment for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=self._row_labels[row_index])

            new_meta = meta.with_updates(group=group)
            self._metadata[row_index] = new_meta

    def set_color(self, row_index: int, color: str) -> None:
        """
        Set the visualization color for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=self._row_labels[row_index])

            new_meta = meta.with_updates(color=color)
            self._metadata[row_index] = new_meta

    def set_marker(self, row_index: int, marker: str) -> None:
        """
        Set the visualization marker for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=self._row_labels[row_index])

            new_meta = meta.with_updates(marker=marker)
            self._metadata[row_index] = new_meta

    def set_size(self, row_index: int, size: float) -> None:
        """
        Set the marker size for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=self._row_labels[row_index])

            new_meta = meta.with_updates(size=size)
            self._metadata[row_index] = new_meta

    def set_age(self, row_index: int, age: float | None) -> None:
        """
        Set the geological age for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=self._row_labels[row_index])

            new_meta = meta.with_updates(age=age)
            self._metadata[row_index] = new_meta

    def set_location(self, row_index: int, location: str | None) -> None:
        """
        Set the location for a row.
        """
        with self._lock:
            meta = self._metadata.get(row_index)
            if meta is None:
                meta = RowMetadata(row_index=row_index, label=self._row_labels[row_index])

            new_meta = meta.with_updates(location=location)
            self._metadata[row_index] = new_meta

    def get_rows_by_group(self, group: str) -> list[int]:
        """
        Get indices of all rows in specified group.
        """
        with self._lock:
            return [idx for idx, meta in self._metadata.items() if meta.group == group]

    def get_all_groups(self) -> list[str]:
        """
        Get list of all unique group names.
        """
        with self._lock:
            groups = set(meta.group for meta in self._metadata.values() if meta.group is not None)
            return sorted(list(groups))

    def get_group_sizes(self) -> dict[str, int]:
        """
        Get the number of rows in each group.
        """
        with self._lock:
            groups = self.get_all_groups()
            return {group: len(self.get_rows_by_group(group)) for group in groups}

    def get_row_colors(self) -> dict[int, str]:
        """
        Get color mapping for all rows.
        """
        with self._lock:
            return {idx: meta.color for idx, meta in self._metadata.items()}

    def get_row_markers(self) -> dict[int, str]:
        """
        Get marker mapping for all rows.
        """
        with self._lock:
            return {idx: meta.marker for idx, meta in self._metadata.items()}

    def get_row_sizes(self) -> dict[int, float]:
        """
        Get size mapping for all rows.
        """
        with self._lock:
            return {idx: meta.size for idx, meta in self._metadata.items()}

    def get_row_labels(self) -> list[str]:
        """
        Get all row labels.
        """
        with self._lock:
            return [
                self._metadata[i].label if i in self._metadata else self._row_labels[i] for i in range(self._n_rows)
            ]

    def get_groups(self) -> dict[int, str | None]:
        """
        Get group mapping for all rows.
        """
        with self._lock:
            return {idx: meta.group for idx, meta in self._metadata.items()}

    def assign_groups_from_labels(self, prefix_separator: str = "_") -> None:
        """
        Automatically assign groups based on common label prefixes.

        For example, labels "Control_1", "Control_2" would all
        be assigned to group "Control".

        Parameters:
            prefix_separator: Character that separates group from ID
        """
        with self._lock:
            for idx, meta in self._metadata.items():
                label = meta.label
                parts = label.split(prefix_separator)
                if len(parts) > 1:
                    group = parts[0]
                    new_meta = meta.with_updates(group=group)
                    self._metadata[idx] = new_meta

    def to_dict(self) -> dict[int, dict[str, Any]]:
        """
        Convert all metadata to dictionary format.
        """
        with self._lock:
            return {idx: meta.to_dict() for idx, meta in self._metadata.items()}

    def from_dict(self, metadata_dict: dict[int, dict[str, Any]]) -> None:
        """
        Load metadata from dictionary format.
        """
        with self._lock:
            for idx, meta_dict in metadata_dict.items():
                self._metadata[idx] = RowMetadata(
                    row_index=idx,
                    label=meta_dict.get("label", self._row_labels[idx]),
                    group=meta_dict.get("group"),
                    color=meta_dict.get("color", CHART_COLORS[idx % len(CHART_COLORS)]),
                    marker=meta_dict.get("marker", CHART_MARKERS[idx % len(CHART_MARKERS)]),
                    size=meta_dict.get("size", 60.0),
                    description=meta_dict.get("description"),
                    age=meta_dict.get("age"),
                    location=meta_dict.get("location"),
                )

    def from_dict_by_label(
        self,
        metadata_dict: dict[int, dict[str, Any]],
        new_labels: list[str],
    ) -> None:
        """Restore metadata whose *label* still exists in ``new_labels``.

        Mirrors :meth:`ColumnMetadataManager.from_dict_by_label`. Useful
        when loading a new dataset that reuses some of the existing
        row labels so the user does not silently lose group/colour
        assignments.
        """
        if not metadata_dict:
            return
        with self._lock:
            for old_idx, meta_dict in metadata_dict.items():
                if old_idx >= len(self._row_labels):
                    continue
                old_label = self._row_labels[old_idx]
                if old_label not in new_labels:
                    continue
                new_idx = new_labels.index(old_label)
                if new_idx >= self._n_rows:
                    continue
                self._metadata[new_idx] = RowMetadata(
                    row_index=new_idx,
                    label=meta_dict.get("label", old_label),
                    group=meta_dict.get("group"),
                    color=meta_dict.get("color", CHART_COLORS[new_idx % len(CHART_COLORS)]),
                    marker=meta_dict.get("marker", CHART_MARKERS[new_idx % len(CHART_MARKERS)]),
                    size=meta_dict.get("size", 60.0),
                    description=meta_dict.get("description"),
                    age=meta_dict.get("age"),
                    location=meta_dict.get("location"),
                )

    def __repr__(self) -> str:
        with self._lock:
            n_groups = len(self.get_all_groups())
            return f"RowMetadataManager(n_rows={self._n_rows}, n_groups={n_groups})"
