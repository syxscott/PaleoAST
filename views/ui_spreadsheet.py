# =============================================================================
# FILE: views/ui_spreadsheet.py
# =============================================================================
"""
Scientific Spreadsheet Widget for PaleoAST

This module implements a high-performance scientific spreadsheet with:
    - Virtual scrolling for large datasets
    - Custom cell rendering with colors and markers
    - Right-click context menus for column operations
    - Bidirectional sync with StateManager

Design Patterns:
    - Observer Pattern: Spreadsheet observes StateManager for data changes
    - Delegate Pattern: Custom delegates for specialized cell rendering
    - Command Pattern: Undo/redo operations

Mathematical Context:
    The spreadsheet displays data matrix X ∈ ℝ^(n×p) where:
        n = number of samples (rows)
        p = number of variables (columns)

    Cell transformations available:
        - Log transform: x' = log(x + 1)
        - Z-score: x' = (x - μ) / σ
        - Center: x' = x - μ
        - Scale: x' = x / σ

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

from PyQt6.QtCore import QItemSelection, QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMenu,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.design_system import get_palette
from config.i18n import _
from models.data_matrix import DataMatrix
from models.state_manager import get_state_manager
from utils.event_bus import get_event_bus


class MarkerStyle(Enum):
    """Marker styles for row/column annotation."""

    CIRCLE = "circle"
    TRIANGLE_UP = "triangle_up"
    TRIANGLE_DOWN = "triangle_down"
    TRIANGLE_LEFT = "triangle_left"
    TRIANGLE_RIGHT = "triangle_right"
    SQUARE = "square"
    DIAMOND = "diamond"
    STAR = "star"
    CROSS = "cross"
    NONE = "none"


class DataType(Enum):
    """Data type classification."""

    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    COUNT = "count"


class SpreadsheetDelegate(QStyledItemDelegate):
    """
    Custom delegate for scientific spreadsheet cells.

    Renders cells with:
        - Color coding based on group
        - Marker symbols in row/column headers
        - Conditional formatting for values
        - Custom alignment and fonts

    Mathematical Context:
        Cell value rendering follows statistical conventions:
            - Scientific notation for very small/large numbers
            - Appropriate decimal places based on measurement precision
            - Color gradients for heatmap-style visualization
    """

    # Color scheme for groups (colorblind-friendly)
    GROUP_COLORS = [
        "#0077BB",  # Blue
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red
        "#33BBEE",  # Cyan
        "#EE3377",  # Magenta
        "#BBBBBB",  # Gray
        "#000000",  # Black
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._row_metadata: dict[int, dict] = {}
        self._col_metadata: dict[int, dict] = {}
        self._data_cache: np.ndarray | None = None
        self._row_labels: list[str] = []
        self._col_labels: list[str] = []

    def set_metadata(self, row_metadata: dict[int, dict], col_metadata: dict[int, dict]) -> None:
        """Set row and column metadata for rendering."""
        self._row_metadata = row_metadata
        self._col_metadata = col_metadata

    def set_data(self, data: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
        """Set data and labels for rendering."""
        self._data_cache = data
        self._row_labels = row_labels
        self._col_labels = col_labels

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        """
        Paint cell with custom rendering.

        This method implements the mathematical cell rendering pipeline:
            1. Get cell value from data matrix
            2. Apply formatting rules
            3. Draw background color (if group assigned)
            4. Draw marker symbol (if row/column has marker)
            5. Draw text value
        """
        painter.save()

        # Get metadata (delegate only receives data cell indices, not header cells)
        row_idx = index.row()
        col_idx = index.column()

        row_meta = self._row_metadata.get(row_idx, {})
        col_meta = self._col_metadata.get(col_idx, {})

        # Draw background
        bg_color = None

        if row_meta.get("color"):
            bg_color = QColor(row_meta["color"])
            bg_color.setAlpha(80)
        elif col_meta.get("color"):
            bg_color = QColor(col_meta["color"])
            bg_color.setAlpha(80)

        if bg_color:
            painter.fillRect(option.rect, QBrush(bg_color))

        # Draw selection highlight
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QBrush(QColor("#3498DB").lighter(150)))

        # Draw data cell content
        if (
            self._data_cache is not None
            and 0 <= row_idx < self._data_cache.shape[0]
            and 0 <= col_idx < self._data_cache.shape[1]
        ):
            value = self._data_cache[row_idx, col_idx]

            if np.isnan(value):
                text = "NaN"
            elif np.isinf(value):
                text = "∞" if value > 0 else "-∞"
            else:
                if abs(value) > 0 and (abs(value) < 0.001 or abs(value) > 10000):
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.4f}"

            # Draw marker if row has one
            marker_size = 10
            marker_value = row_meta.get("marker", MarkerStyle.NONE.value)
            if marker_value and marker_value != MarkerStyle.NONE.value:
                marker_x = option.rect.left() + 4
                marker_y = option.rect.top() + (option.rect.height() - marker_size) // 2
                self._draw_marker(
                    painter, marker_value, marker_x, marker_y, marker_size, row_meta.get("color", "#E74C3C")
                )
                text_rect = option.rect.adjusted(marker_size + 10, 0, 0, 0)
            else:
                text_rect = option.rect.adjusted(4, 0, 0, 0)

            painter.setPen(QPen(QColor("#2C3E50")))
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        else:
            painter.drawText(
                option.rect.adjusted(4, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, ""
            )

        painter.restore()

    def _draw_marker(self, painter: QPainter, marker_type: str, x: float, y: float, size: float, color: str) -> None:
        """Draw marker symbol at specified position."""
        painter.save()
        painter.setPen(QPen(QColor(color), 1.5))
        painter.setBrush(QBrush(QColor(color)))

        cx = x + size // 2
        cy = y + size // 2
        r = size // 2 - 1

        if marker_type == MarkerStyle.CIRCLE.value:
            painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        elif marker_type == MarkerStyle.SQUARE.value:
            painter.drawRect(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        elif marker_type == MarkerStyle.DIAMOND.value:
            path = QPainterPath()
            path.moveTo(cx, cy - r)
            path.lineTo(cx + r, cy)
            path.lineTo(cx, cy + r)
            path.lineTo(cx - r, cy)
            path.closeSubpath()
            painter.drawPath(path)

        elif marker_type == MarkerStyle.TRIANGLE_UP.value:
            path = QPainterPath()
            path.moveTo(cx, cy - r)
            path.lineTo(cx + r, cy + r)
            path.lineTo(cx - r, cy + r)
            path.closeSubpath()
            painter.drawPath(path)

        elif marker_type == MarkerStyle.TRIANGLE_DOWN.value:
            path = QPainterPath()
            path.moveTo(cx, cy + r)
            path.lineTo(cx + r, cy - r)
            path.lineTo(cx - r, cy - r)
            path.closeSubpath()
            painter.drawPath(path)

        elif marker_type == MarkerStyle.TRIANGLE_LEFT.value:
            path = QPainterPath()
            path.moveTo(cx - r, cy)
            path.lineTo(cx + r, cy + r)
            path.lineTo(cx + r, cy - r)
            path.closeSubpath()
            painter.drawPath(path)

        elif marker_type == MarkerStyle.TRIANGLE_RIGHT.value:
            path = QPainterPath()
            path.moveTo(cx + r, cy)
            path.lineTo(cx - r, cy + r)
            path.lineTo(cx - r, cy - r)
            path.closeSubpath()
            painter.drawPath(path)

        elif marker_type == MarkerStyle.STAR.value:
            # 6-pointed star: alternate between outer and inner radius
            path = QPainterPath()
            for i in range(12):
                angle = i * 30 * 3.14159 / 180
                radius = r if i % 2 == 0 else r * 0.4
                px = cx + radius * np.cos(angle)
                py = cy + radius * np.sin(angle)
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            path.closeSubpath()
            painter.drawPath(path)

        elif marker_type == MarkerStyle.CROSS.value:
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
            painter.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Return size hint for cell."""
        return QSize(100, 25)


class ColumnHeaderMenu(QMenu):
    """
    Context menu for column header operations.

    Mathematical Operations Available:
        - Set as Group Column
        - Set Data Type (Continuous/Ordinal/Nominal)
        - Log Transform: x' = ln(x + 1)
        - Z-score: x' = (x - μ) / σ
        - Center: x' = x - μ
        - Scale: x' = x / σ
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._column_index = -1

    def set_column(self, col_idx: int) -> None:
        """Set which column this menu is for."""
        self._column_index = col_idx
        self._build_menu()

    def _build_menu(self) -> None:
        """Build the context menu."""
        self.clear()

        # Group operations
        group_menu = self.addMenu(_("Set as Group Column"))
        for i in range(8):
            action = group_menu.addAction(_("Group {0}").format(i + 1))
            action.setData(("group", i))

        self.addSeparator()

        # Data type
        type_menu = self.addMenu(_("Set Data Type"))
        for dtype in DataType:
            action = type_menu.addAction(dtype.value.capitalize())
            action.setData(("type", dtype))

        self.addSeparator()

        # Transformations
        transform_menu = self.addMenu(_("Transform Column"))

        log_action = transform_menu.addAction(_("Log Transform: x' = ln(x + 1)"))
        log_action.setData(("transform", "log"))

        zscore_action = transform_menu.addAction(_("Z-score: x' = (x - μ) / σ"))
        zscore_action.setData(("transform", "zscore"))

        center_action = transform_menu.addAction(_("Center: x' = x - μ"))
        center_action.setData(("transform", "center"))

        scale_action = transform_menu.addAction(_("Scale: x' = x / σ"))
        scale_action.setData(("transform", "scale"))

        self.addSeparator()

        # Sort operations
        sort_asc_action = self.addAction(_("Sort Ascending"))
        sort_asc_action.setData(("sort", "asc"))

        sort_desc_action = self.addAction(_("Sort Descending"))
        sort_desc_action.setData(("sort", "desc"))

        self.addSeparator()

        # Delete column
        delete_action = self.addAction(_("Delete Column"))
        delete_action.setData(("delete",))


class RowHeaderMenu(QMenu):
    """
    Context menu for row header operations.

    Operations Available:
        - Set Marker Style
        - Set Row Color
        - Assign to Group
        - Hide/Show Row
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._row_index = -1

    def set_row(self, row_idx: int) -> None:
        """Set which row this menu is for."""
        self._row_index = row_idx
        self._build_menu()

    def _build_menu(self) -> None:
        """Build the context menu."""
        self.clear()

        # Marker style
        marker_menu = self.addMenu(_("Set Marker"))
        for style in MarkerStyle:
            action = marker_menu.addAction(style.value.capitalize())
            action.setData(("marker", style))

        self.addSeparator()

        # Row color
        color_menu = self.addMenu(_("Set Color"))
        colors = [
            ("Red", "#E74C3C"),
            ("Orange", "#F39C12"),
            ("Yellow", "#F1C40F"),
            ("Green", "#27AE60"),
            ("Blue", "#3498DB"),
            ("Purple", "#9B59B6"),
            ("Gray", "#95A5A6"),
        ]
        for name, hex_color in colors:
            action = color_menu.addAction(name)
            action.setData(("color", hex_color))

        self.addSeparator()

        # Assign to group
        group_menu = self.addMenu(_("Assign to Group"))
        for i in range(8):
            action = group_menu.addAction(_("Group {0}").format(i + 1))
            action.setData(("group", i))

        self.addSeparator()

        # Hide/Show
        hide_action = self.addAction(_("Hide Row"))
        hide_action.setData(("hide",))

        show_all_action = self.addAction(_("Show All Rows"))
        show_all_action.setData(("show_all",))


class ScientificSpreadsheet(QWidget):
    """
    High-performance scientific spreadsheet widget.

    Features:
        - Virtual scrolling for large datasets
        - Custom cell rendering with metadata support
        - Column/row header context menus
        - Bidirectional sync with StateManager
        - Copy/paste support
        - Undo/redo operations

    Mathematical Context:
        The spreadsheet displays a data matrix X ∈ ℝ^(n×p):
            - Rows (n): Samples/specimens
            - Columns (p): Measured variables

        Supported transformations:
            1. Log transform: x'_ij = ln(x_ij + 1)
               Used for: Skewed count data

            2. Z-score: x'_ij = (x_ij - μ_j) / σ_j
               Used for: Standardizing different scales

            3. Center: x'_ij = x_ij - μ_j
               Used for: Removing mean

            4. Scale: x'_ij = x_ij / σ_j
               Used for: Normalizing variance

    Signals:
        dataChanged: Emitted when data is modified
        selectionChanged: Emitted when selection changes
        columnTransformed: Emitted when column is transformed
    """

    dataChanged = pyqtSignal()
    selectionChanged = pyqtSignal(list, list)
    columnTransformed = pyqtSignal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.ScientificSpreadsheet")
        self._logger.info("ScientificSpreadsheet initialized")

        self._is_dark_theme = False

        # State manager reference
        self._state = get_state_manager()

        # Subscribe to EventBus
        self._event_bus = get_event_bus()
        self._event_bus.data_changed.connect(self._on_data_changed)
        self._event_bus.metadata_changed.connect(self._on_metadata_changed)

        # Guard flag to prevent re-entrant event loops
        self._loading_from_event = False

        # Data
        self._data: np.ndarray | None = None
        self._row_labels: list[str] = []
        self._col_labels: list[str] = []
        self._row_metadata: dict[int, dict] = {}
        self._col_metadata: dict[int, dict] = {}

        # Undo/redo stacks
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []

        # Selection
        self._selected_rows: list[int] = []
        self._selected_cols: list[int] = []

        # Setup UI
        self._setup_ui()
        self._setup_connections()

        # Connect to state manager
        # Note: StateManager doesn't inherit from QObject, so it has no signals
        # self._state.data_changed.connect(self._on_state_data_changed)

    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create table widget
        self._table = QTableWidget(self)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setShowGrid(True)
        self._table.setCornerButtonEnabled(True)
        self._table.setColumnCount(0)
        self._table.setRowCount(0)

        # Custom delegate
        self._delegate = SpreadsheetDelegate()
        self._table.setItemDelegate(self._delegate)

        # Header settings
        h_header = self._table.horizontalHeader()
        h_header.setStretchLastSection(True)
        h_header.setMinimumSectionSize(100)
        h_header.setDefaultSectionSize(120)
        h_header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        v_header = self._table.verticalHeader()
        v_header.setMinimumSectionSize(32)
        v_header.setDefaultSectionSize(28)
        v_header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        layout.addWidget(self._table)

        # Apply themed stylesheet
        self._apply_stylesheet()

        # Context menus
        self._col_header_menu = ColumnHeaderMenu(self._table)
        self._row_header_menu = RowHeaderMenu(self._table)

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {c.bg_primary};
                alternate-background-color: {c.bg_secondary};
                color: {c.text_primary};
                gridline-color: {c.border_light};
                font-family: 'Segoe UI', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid {c.border_light};
                border-radius: 6px;
            }}
            QTableWidget::item {{
                padding: 6px 8px;
                border-bottom: 1px solid {c.border_light};
            }}
            QTableWidget::item:selected {{
                background-color: {c.selected_overlay};
                color: {c.primary};
            }}
            QHeaderView {{
                background-color: {c.bg_tertiary};
                color: {c.text_primary};
            }}
            QHeaderView::section {{
                background-color: {c.bg_tertiary};
                color: {c.text_primary};
                padding: 8px;
                border: none;
                border-right: 1px solid {c.border_light};
                border-bottom: 2px solid {c.primary};
                font-weight: 600;
            }}
            QHeaderView::section:first {{
                border-left: none;
            }}
            QScrollBar:vertical {{
                background-color: {c.bg_secondary};
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {c.border_medium};
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {c.text_secondary};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: transparent;
            }}
            QScrollBar:horizontal {{
                background-color: {c.bg_secondary};
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {c.border_medium};
                min-width: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {c.text_secondary};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Context menu triggers
        self._table.horizontalHeader().customContextMenuRequested.connect(self._show_col_header_menu)
        self._table.verticalHeader().customContextMenuRequested.connect(self._show_row_header_menu)

        # Menu actions
        self._col_header_menu.triggered.connect(self._on_col_menu_action)
        self._row_header_menu.triggered.connect(self._on_row_menu_action)

        # Selection changes
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        # Data changes
        self._table.itemChanged.connect(self._on_item_changed)

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_data_changed(self, matrix) -> None:
        """Handle data_changed event from EventBus."""
        if self._loading_from_event:
            return
        self._loading_from_event = True
        try:
            if matrix is not None:
                self.load_data(
                    matrix.data,
                    row_labels=matrix.row_labels,
                    col_labels=matrix.col_labels,
                )
            else:
                # ``clear_data`` was previously referenced here but
                # the method did not exist on the spreadsheet. Implement
                # the behaviour here directly so the EventBus signal
                # is honoured when data is cleared.
                self._table.blockSignals(True)
                try:
                    self._table.setRowCount(0)
                    self._table.setColumnCount(0)
                finally:
                    self._table.blockSignals(False)
                self._data = None
                self._row_labels = []
                self._col_labels = []
                self._col_metadata = {}
                self._row_metadata = {}
                self.dataChanged.emit()
        finally:
            self._loading_from_event = False

    def _on_metadata_changed(self, scope: str, index: int, metadata: dict) -> None:
        """Handle metadata_changed event from EventBus."""
        if scope == "column" and index in self._col_metadata:
            self._col_metadata[index].update(metadata)
        elif scope == "row" and index in self._row_metadata:
            self._row_metadata[index].update(metadata)

    def load_data(
        self, data: np.ndarray, row_labels: list[str] | None = None, col_labels: list[str] | None = None
    ) -> None:
        """
        Load data into spreadsheet.

        Mathematical Context:
            Input: data matrix X ∈ ℝ^(n×p)
            where n = samples, p = variables

            Labels are auto-generated if not provided:
                row_labels = [f"Sample_{i+1}" for i in range(n)]
                col_labels = [f"Var_{j+1}" for j in range(p)]
        """
        self._data = np.array(data, dtype=np.float64)
        n_rows, n_cols = self._data.shape
        self._logger.info(f"load_data called: n_rows={n_rows}, n_cols={n_cols}")

        # Warn for very large datasets that may cause performance issues
        LARGE_DATASET_THRESHOLD = 50000  # 50k cells triggers optimization
        total_cells = n_rows * n_cols
        if total_cells > LARGE_DATASET_THRESHOLD:
            self._logger.warning(
                f"Large dataset detected: {n_rows}x{n_cols}={total_cells} cells. "
                "Loading may be slow. Consider using a subset for visualization."
            )

        # Generate labels if not provided
        if row_labels is None:
            self._row_labels = [f"Sample_{i + 1}" for i in range(n_rows)]
        else:
            self._row_labels = list(row_labels)

        if col_labels is None:
            self._col_labels = [f"Var_{j + 1}" for j in range(n_cols)]
        else:
            self._col_labels = list(col_labels)

        # Update table - optimized for large datasets
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(n_rows)
            self._table.setColumnCount(n_cols)
            self._table.setVerticalHeaderLabels(self._row_labels)
            self._table.setHorizontalHeaderLabels(self._col_labels)

            # Use setItem row by row for better performance
            for i in range(n_rows):
                for j in range(n_cols):
                    value = self._data[i, j]
                    if np.isnan(value):
                        item = QTableWidgetItem("")
                    else:
                        item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(i, j, item)
        finally:
            self._table.setUpdatesEnabled(True)
            self._table.blockSignals(False)

        # Update delegate
        self._delegate.set_data(self._data, self._row_labels, self._col_labels)
        self._delegate.set_metadata(self._row_metadata, self._col_metadata)

        # Update state manager
        self._state.set_data_matrix(
            DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels)
        )

        # Resize columns to content - skip for very wide tables (performance)
        MAX_COLS_FOR_RESIZE = 100
        if n_cols <= MAX_COLS_FOR_RESIZE:
            self._table.resizeColumnsToContents()
        else:
            # Just set a reasonable default width for wide tables
            self._table.horizontalHeader().setDefaultSectionSize(80)

        self.dataChanged.emit()

    def get_data(self, copy: bool = True) -> np.ndarray | None:
        """
        Get current data matrix.

        Parameters:
            copy: If True (default), returns a copy. If False, returns internal reference.
                  Use False only if you won't modify the returned array.
        """
        if self._data is None:
            return None
        return self._data.copy() if copy else self._data

    def get_selected_data(self) -> np.ndarray | None:
        """
        Get currently selected data submatrix.

        Mathematical Context:
            Returns X_selected ∈ ℝ^(n_selected × p_selected)
            where rows and columns are from current selection.
        """
        if self._data is None:
            return None

        selected = self._table.selectionModel().selectedIndexes()
        if not selected:
            return self._data.copy()

        rows = list(set(idx.row() for idx in selected))
        cols = list(set(idx.column() for idx in selected))
        rows.sort()
        cols.sort()

        return self._data[np.ix_(rows, cols)].copy()

    def _show_col_header_menu(self, pos: QPoint) -> None:
        """Show context menu for column header."""
        col = self._table.columnAt(pos.x())
        if col >= 0:
            self._col_header_menu.set_column(col)
            self._col_header_menu.exec(QCursor.pos())

    def _show_row_header_menu(self, pos: QPoint) -> None:
        """Show context menu for row header."""
        row = self._table.rowAt(pos.y())
        if row >= 0:
            self._row_header_menu.set_row(row)
            self._row_header_menu.exec(QCursor.pos())

    def _on_col_menu_action(self, action) -> None:
        """Handle column header menu action."""
        data = action.data()
        if not data:
            return

        col = self._col_header_menu._column_index
        action_type, value = data

        if action_type == "group":
            # Set group color
            color = SpreadsheetDelegate.GROUP_COLORS[value]
            if col not in self._col_metadata:
                self._col_metadata[col] = {}
            self._col_metadata[col]["color"] = color
            self._col_metadata[col]["group"] = value
            self._state.set_col_metadata(col, self._col_metadata[col])
            self._table.viewport().update()

        elif action_type == "type":
            # Set data type
            if col not in self._col_metadata:
                self._col_metadata[col] = {}
            self._col_metadata[col]["data_type"] = value
            self._state.set_col_metadata(col, self._col_metadata[col])

        elif action_type == "transform":
            # Apply transformation
            self._apply_column_transform(col, value)

        elif action_type == "sort":
            # Sort column
            ascending = value == "asc"
            self._sort_by_column(col, ascending)

        elif action_type == "delete":
            # Delete column - ask for confirmation
            col_label = self._col_labels[col] if col < len(self._col_labels) else f"Column {col + 1}"
            reply = QMessageBox.question(
                self,
                _("Confirm Delete"),
                _("Delete column '{0}'? This can be undone.").format(col_label),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._delete_column(col)

    def _on_row_menu_action(self, action) -> None:
        """Handle row header menu action."""
        data = action.data()
        if not data:
            return

        row = self._row_header_menu._row_index
        action_type, value = data

        if action_type == "marker":
            # Set marker style
            if row not in self._row_metadata:
                self._row_metadata[row] = {}
            self._row_metadata[row]["marker"] = value.value
            self._state.set_row_metadata(row, self._row_metadata[row])
            self._table.viewport().update()

        elif action_type == "color":
            # Set row color
            if row not in self._row_metadata:
                self._row_metadata[row] = {}
            self._row_metadata[row]["color"] = value
            self._state.set_row_metadata(row, self._row_metadata[row])
            self._table.viewport().update()

        elif action_type == "group":
            # Assign to group
            color = SpreadsheetDelegate.GROUP_COLORS[value]
            if row not in self._row_metadata:
                self._row_metadata[row] = {}
            self._row_metadata[row]["color"] = color
            self._row_metadata[row]["group"] = value
            self._state.set_row_metadata(row, self._row_metadata[row])
            self._table.viewport().update()

        elif action_type == "hide":
            # Hide row
            self._table.hideRow(row)

        elif action_type == "show_all":
            # Show all rows
            for i in range(self._table.rowCount()):
                self._table.showRow(i)

    def _apply_column_transform(self, col: int, transform: str, _record_undo: bool = True) -> None:
        """
        Apply mathematical transformation to column.

        Mathematical Transformations:

        1. Log Transform: x' = ln(x + 1)
           Jacobian: ∂x'/∂x = 1/(x + 1)
           Used for: Count data with many zeros

        2. Z-score: x' = (x - μ) / σ
           where μ = (1/n) Σx_i
                 σ = sqrt((1/n) Σ(x_i - μ)²)
           Jacobian: ∂x'/∂x = 1/σ
           Used for: Standardizing variables with different scales

        3. Center: x' = x - μ
           Jacobian: ∂x'/∂x = 1
           Used for: Removing mean effect

        4. Scale: x' = x / σ
           Jacobian: ∂x'/∂x = 1/σ
           Used for: Normalizing variance
        """
        if self._data is None or col < 0 or col >= self._data.shape[1]:
            return

        self._logger.info(f"Column transform: col={col}, transform='{transform}'")
        # Save for undo (skip when called from redo to avoid double-pushing)
        if _record_undo:
            self._undo_stack.append(("transform", col, transform, self._data[:, col].copy()))
            self._redo_stack.clear()

        col_data = self._data[:, col]

        if transform == "log":
            # Log transform: x' = ln(x + 1)
            # Handle negative values and zeros
            new_data = np.log1p(col_data)  # log1p(x) = ln(x + 1)

        elif transform == "zscore":
            # Z-score: x' = (x - μ) / σ
            mean = np.nanmean(col_data)
            std = np.nanstd(col_data)
            if std > 0:
                new_data = (col_data - mean) / std
            else:
                new_data = col_data - mean

        elif transform == "center":
            # Center: x' = x - μ
            mean = np.nanmean(col_data)
            new_data = col_data - mean

        elif transform == "scale":
            # Scale: x' = x / σ
            std = np.nanstd(col_data)
            if std > 0:
                new_data = col_data / std
            else:
                new_data = col_data

        else:
            return

        # Update data
        self._data[:, col] = new_data

        # Update table (block signals to prevent re-triggering itemChanged)
        self._table.blockSignals(True)
        try:
            for i in range(self._data.shape[0]):
                value = new_data[i]
                if np.isnan(value):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)
        finally:
            self._table.blockSignals(False)

        # Update state
        self._state.set_data_matrix(
            DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels)
        )

        # Emit signal
        self.columnTransformed.emit(col, transform)
        self.dataChanged.emit()

    def _sort_by_column(self, col: int, ascending: bool) -> None:
        """Sort data by column."""
        if self._data is None or col < 0 or col >= self._data.shape[1]:
            return
        self._logger.info(f"Sort by column: col={col}, ascending={ascending}")

        # Get sort indices
        sort_data = self._data[:, col]
        # Handle NaN values: substitute +/-inf so they always end up at
        # the *end* of the sorted output regardless of the direction.
        # For ascending, NaNs become +inf (largest); for descending,
        # they become -inf (smallest) and the final ``[::-1]`` flips
        # them to the back. NaN is therefore never mixed into the
        # non-NaN ordering.
        nan_mask = np.isnan(sort_data)
        sentinel = np.inf if ascending else -np.inf
        sort_indices = np.argsort(np.where(nan_mask, sentinel, sort_data), kind="stable")
        if not ascending:
            sort_indices = sort_indices[::-1]

        # Apply sort
        self._data = self._data[sort_indices, :]
        self._row_labels = [self._row_labels[i] for i in sort_indices]

        # Remap row metadata to match new row order
        new_row_metadata = {}
        for new_idx, old_idx in enumerate(sort_indices):
            if old_idx in self._row_metadata:
                new_row_metadata[new_idx] = self._row_metadata[old_idx]
        self._row_metadata = new_row_metadata

        # Reorder table rows
        self._table.blockSignals(True)
        try:
            for new_row in range(self._data.shape[0]):
                for col_idx in range(self._data.shape[1]):
                    value = self._data[new_row, col_idx]
                    if np.isnan(value):
                        item = QTableWidgetItem("")
                    else:
                        item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(new_row, col_idx, item)
            self._table.setVerticalHeaderLabels(self._row_labels)
        finally:
            self._table.blockSignals(False)

        # Update state
        self._state.set_data_matrix(
            DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels)
        )
        self.dataChanged.emit()

    def _delete_column(self, col: int, _record_undo: bool = True) -> None:
        """Delete column from data (optimized)."""
        if self._data is None or col < 0 or col >= self._data.shape[1]:
            return
        self._logger.info(f"Delete column: col={col}")

        # Save for undo - need to save all column metadata since indices shift on delete
        deleted_metadata = self._col_metadata.get(col, {})
        all_metadata = {k: v for k, v in self._col_metadata.items()}
        if _record_undo:
            self._undo_stack.append(
                ("delete_col", col, self._data[:, col].copy(), self._col_labels[col], deleted_metadata, all_metadata)
            )
            self._redo_stack.clear()

        # Remove column using view when possible (avoids full copy for contiguous memory)
        if col == self._data.shape[1] - 1:
            # Last column: can use simple slicing
            self._data = self._data[:, :-1]
        elif col == 0:
            # First column: can use simple slicing
            self._data = self._data[:, 1:]
        else:
            # Middle column: need concatenate
            self._data = np.delete(self._data, col, axis=1)

        # Remove label
        self._col_labels.pop(col)

        # Update metadata indices (optimized with dict comprehension)
        self._col_metadata = {
            old_idx - 1 if old_idx > col else old_idx: meta
            for old_idx, meta in self._col_metadata.items()
            if old_idx != col
        }

        # Update table
        self._table.removeColumn(col)
        self._table.setHorizontalHeaderLabels(self._col_labels)

        # Update state
        self._state.set_data_matrix(
            DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels)
        )
        self.dataChanged.emit()

    def _on_selection_changed(self, selected: "QItemSelection", deselected: "QItemSelection") -> None:
        """Handle selection change."""
        indexes = self._table.selectionModel().selectedIndexes()

        rows = sorted(list(set(idx.row() for idx in indexes)))
        cols = sorted(list(set(idx.column() for idx in indexes)))

        self._selected_rows = rows
        self._selected_cols = cols

        self.selectionChanged.emit(rows, cols)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle cell data change."""
        if self._data is None:
            return

        row = item.row()
        col = item.column()

        if row >= self._data.shape[0] or col >= self._data.shape[1]:
            return

        try:
            value = float(item.text())
        except ValueError:
            # Revert to old value. Use blockSignals to prevent this
            # setText from re-triggering _on_item_changed and creating
            # a recursion loop.
            if not np.isnan(self._data[row, col]):
                self._table.blockSignals(True)
                try:
                    item.setText(str(self._data[row, col]))
                finally:
                    self._table.blockSignals(False)
            return

        old_value = self._data[row, col]
        if old_value == value:
            return
        self._data[row, col] = value

        # Save for undo
        self._undo_stack.append(("cell", row, col, old_value))
        self._redo_stack.clear()

        # Update state. We pass ``_record_undo=False`` because the
        # spreadsheet already has its own undo entry, and
        # ``_reset_metadata=False`` because a single cell edit must
        # not discard the column / row metadata the user has set.
        self._state.set_data_matrix(
            DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels),
            _record_undo=False,
            _reset_metadata=False,
        )
        self.dataChanged.emit()

    def export_to_clipboard(self) -> None:
        """Export selected data to clipboard."""
        selected_data = self.get_selected_data()
        if selected_data is None:
            return

        # Format as tab-separated text
        rows_str = []
        for i in range(selected_data.shape[0]):
            row = [f"{v:.6g}" if not np.isnan(v) else "" for v in selected_data[i, :]]
            rows_str.append("\t".join(row))

        text = "\n".join(rows_str)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def import_from_clipboard(self) -> None:
        """Import data from clipboard."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()

        if not text:
            return

        try:
            # Parse tab-separated text
            lines = text.strip().split("\n")
            data = []
            max_cols = 0
            for line in lines:
                values = line.split("\t")
                row = []
                for v in values:
                    try:
                        row.append(float(v))
                    except ValueError:
                        # Treat unparseable values as missing data
                        # (NaN) instead of failing the whole import.
                        row.append(np.nan)
                data.append(row)
                max_cols = max(max_cols, len(row))

            if not data:
                QMessageBox.warning(self, _("Import Error"), _("Clipboard contains no tabular data to import."))
                return

            # Pad rows to equal length
            for row in data:
                while len(row) < max_cols:
                    row.append(np.nan)

            # np.array(..., dtype=object) would let ragged rows survive
            # but downstream callers expect a 2D float array. Use
            # ``dtype=float`` after padding so the shape is consistent.
            data = np.array(data, dtype=float)
            self.load_data(data)

        except Exception as e:
            QMessageBox.critical(self, _("Import Error"), _("Failed to import data:\n{0}").format(e))

    @staticmethod
    def _make_display_item(value) -> QTableWidgetItem:
        """Build a QTableWidgetItem that correctly renders NaN as empty."""
        if np.isnan(value):
            item = QTableWidgetItem("")
        else:
            item = QTableWidgetItem(str(value))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def undo(self) -> None:
        """Undo last operation."""
        if not self._undo_stack:
            return

        operation = self._undo_stack.pop()

        # Block signals to prevent itemChanged from corrupting stacks
        self._table.blockSignals(True)
        try:
            if operation[0] == "transform":
                _, col, transform, old_data = operation
                self._redo_stack.append(("transform", col, transform))
                self._data[:, col] = old_data
                for i in range(self._data.shape[0]):
                    self._table.setItem(i, col, self._make_display_item(old_data[i]))

            elif operation[0] == "delete_col":
                _, col, col_data, label, deleted_metadata, all_metadata = operation
                self._redo_stack.append(("delete_col", col, col_data, label, deleted_metadata, all_metadata))
                self._data = np.insert(self._data, col, col_data, axis=1)
                self._col_labels.insert(col, label)
                # Restore all metadata from before delete (preserves metadata for all columns)
                self._col_metadata = {k: v for k, v in all_metadata.items()}
                self._col_metadata[col] = deleted_metadata
                self._table.insertColumn(col)
                self._table.setHorizontalHeaderLabels(self._col_labels)
                for i in range(self._data.shape[0]):
                    self._table.setItem(i, col, self._make_display_item(col_data[i]))

            elif operation[0] == "cell":
                _, row, col, old_value = operation
                self._redo_stack.append(("cell", row, col, self._data[row, col]))
                self._data[row, col] = old_value
                self._table.setItem(row, col, self._make_display_item(old_value))
        finally:
            self._table.blockSignals(False)

        self._state.set_data_matrix(
            DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels)
        )
        self.dataChanged.emit()

    def redo(self) -> None:
        """Redo last undone operation."""
        if not self._redo_stack:
            return

        operation = self._redo_stack.pop()

        if operation[0] == "transform":
            _, col, transform = operation
            # Bypass the push-once guard in _apply_column_transform so
            # redo does not double-add to the undo stack.
            self._apply_column_transform(col, transform, _record_undo=False)

        elif operation[0] == "delete_col":
            # The redo entry already contains the column data and
            # metadata, so re-running the deletion with the guard
            # disabled is enough. We must *not* push anything onto
            # the undo stack (we just popped this from the redo
            # stack and the user expects a single logical action).
            _, col = operation
            self._delete_column(col, _record_undo=False)

        elif operation[0] == "cell":
            _, row, col, new_value = operation
            self._undo_stack.append(("cell", row, col, self._data[row, col]))
            self._data[row, col] = new_value
            self._table.blockSignals(True)
            try:
                self._table.setItem(row, col, self._make_display_item(new_value))
            finally:
                self._table.blockSignals(False)
            self._state.set_data_matrix(
                DataMatrix(data=self._data, row_labels=self._row_labels, col_labels=self._col_labels)
            )
            self.dataChanged.emit()
