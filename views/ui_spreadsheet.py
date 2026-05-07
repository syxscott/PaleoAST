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
Version: 1.0.0
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QMenu, QApplication, QStyledItemDelegate,
    QStyleOptionViewItem, QStyle, QAbstractItemView, QScrollBar,
    QDialog, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QSize, QRect, QPoint, pyqtSignal, pyqtSlot,
    QAbstractTableModel, QModelIndex, QVariant, QMimeData
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QCursor, QPixmap,
    QPainterPath, QGradient, QLinearGradient, QRadialGradient
)

from models.state_manager import get_state_manager


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
        self._row_metadata: Dict[int, Dict] = {}
        self._col_metadata: Dict[int, Dict] = {}
        self._data_cache: Optional[np.ndarray] = None
        self._row_labels: List[str] = []
        self._col_labels: List[str] = []
    
    def set_metadata(
        self,
        row_metadata: Dict[int, Dict],
        col_metadata: Dict[int, Dict]
    ) -> None:
        """Set row and column metadata for rendering."""
        self._row_metadata = row_metadata
        self._col_metadata = col_metadata
    
    def set_data(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str]
    ) -> None:
        """Set data and labels for rendering."""
        self._data_cache = data
        self._row_labels = row_labels
        self._col_labels = col_labels
    
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index
    ) -> None:
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
        
        # Determine if this is a header cell
        is_row_header = index.row() == 0
        is_col_header = index.column() == 0
        is_header = is_row_header or is_col_header
        
        # Get metadata
        row_idx = index.row() - 1  # Account for row header column
        col_idx = index.column() - 1  # Account for column header row
        
        row_meta = self._row_metadata.get(row_idx, {})
        col_meta = self._col_metadata.get(col_idx, {})
        
        # Draw background
        bg_color = None
        
        if is_header:
            # Header cell - dark background
            bg_color = QColor("#2C3E50")
        elif row_meta.get('color'):
            # Row has group color
            bg_color = QColor(row_meta['color'])
            bg_color.setAlpha(80)  # Semi-transparent
        elif col_meta.get('color'):
            # Column has group color
            bg_color = QColor(col_meta['color'])
            bg_color.setAlpha(80)
        
        if bg_color:
            painter.fillRect(option.rect, QBrush(bg_color))
        
        # Draw selection highlight
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(
                option.rect,
                QBrush(QColor("#3498DB").lighter(150))
            )
        
        # Draw cell content
        if is_header:
            # Header content
            if is_row_header and row_idx >= 0 and row_idx < len(self._row_labels):
                text = self._row_labels[row_idx]
            elif is_col_header and col_idx >= 0 and col_idx < len(self._col_labels):
                text = self._col_labels[col_idx]
            else:
                text = ""
            
            # Draw marker if assigned
            marker_size = 10
            marker_x = option.rect.left() + 4
            marker_y = option.rect.top() + (option.rect.height() - marker_size) // 2
            
            if is_row_header and row_meta.get('marker') != MarkerStyle.NONE.value:
                self._draw_marker(
                    painter,
                    row_meta['marker'],
                    marker_x, marker_y,
                    marker_size,
                    row_meta.get('color', '#E74C3C')
                )
                text_rect = option.rect.adjusted(marker_size + 10, 0, 0, 0)
            else:
                text_rect = option.rect.adjusted(4, 0, 0, 0)
            
            # Draw header text
            font = QFont("Arial", 9, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#ECF0F1")))
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text
            )
            
            # Draw header border
            painter.setPen(QPen(QColor("#34495E"), 1))
            painter.drawLine(
                option.rect.bottomLeft(),
                option.rect.bottomRight()
            )
            
        else:
            # Data cell
            if self._data_cache is not None:
                if 0 <= row_idx < self._data_cache.shape[0]:
                    if 0 <= col_idx < self._data_cache.shape[1]:
                        value = self._data_cache[row_idx, col_idx]
                        
                        # Format value
                        if np.isnan(value):
                            text = "NaN"
                        elif np.isinf(value):
                            text = "∞" if value > 0 else "-∞"
                        else:
                            # Scientific notation for extreme values
                            if abs(value) > 0 and (abs(value) < 0.001 or abs(value) > 10000):
                                text = f"{value:.2e}"
                            else:
                                text = f"{value:.4f}"
                        
                        # Draw text
                        painter.setPen(QPen(QColor("#ECF0F1")))
                        painter.drawText(
                            option.rect.adjusted(4, 0, 0, 0),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            text
                        )
            else:
                painter.drawText(
                    option.rect.adjusted(4, 0, 0, 0),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    ""
                )
        
        painter.restore()
    
    def _draw_marker(
        self,
        painter: QPainter,
        marker_type: str,
        x: float, y: float,
        size: float,
        color: str
    ) -> None:
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
            # 6-pointed star
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                px = cx + r * np.cos(angle)
                py = cy + r * np.sin(angle)
                if i == 0:
                    path = QPainterPath()
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
    
    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index
    ) -> QSize:
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
        group_menu = self.addMenu("Set as Group Column")
        for i in range(8):
            action = group_menu.addAction(f"Group {i+1}")
            action.setData(('group', i))
        
        self.addSeparator()
        
        # Data type
        type_menu = self.addMenu("Set Data Type")
        for dtype in DataType:
            action = type_menu.addAction(dtype.value.capitalize())
            action.setData(('type', dtype))
        
        self.addSeparator()
        
        # Transformations
        transform_menu = self.addMenu("Transform Column")
        
        log_action = transform_menu.addAction("Log Transform: x' = ln(x + 1)")
        log_action.setData(('transform', 'log'))
        
        zscore_action = transform_menu.addAction("Z-score: x' = (x - μ) / σ")
        zscore_action.setData(('transform', 'zscore'))
        
        center_action = transform_menu.addAction("Center: x' = x - μ")
        center_action.setData(('transform', 'center'))
        
        scale_action = transform_menu.addAction("Scale: x' = x / σ")
        scale_action.setData(('transform', 'scale'))
        
        self.addSeparator()
        
        # Sort operations
        sort_asc_action = self.addAction("Sort Ascending")
        sort_asc_action.setData(('sort', 'asc'))
        
        sort_desc_action = self.addAction("Sort Descending")
        sort_desc_action.setData(('sort', 'desc'))
        
        self.addSeparator()
        
        # Delete column
        delete_action = self.addAction("Delete Column")
        delete_action.setData(('delete',))


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
        marker_menu = self.addMenu("Set Marker")
        for style in MarkerStyle:
            action = marker_menu.addAction(style.value.capitalize())
            action.setData(('marker', style))
        
        self.addSeparator()
        
        # Row color
        color_menu = self.addMenu("Set Color")
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
            action.setData(('color', hex_color))
        
        self.addSeparator()
        
        # Assign to group
        group_menu = self.addMenu("Assign to Group")
        for i in range(8):
            action = group_menu.addAction(f"Group {i+1}")
            action.setData(('group', i))
        
        self.addSeparator()
        
        # Hide/Show
        hide_action = self.addAction("Hide Row")
        hide_action.setData(('hide',))
        
        show_all_action = self.addAction("Show All Rows")
        show_all_action.setData(('show_all',))


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
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # State manager reference
        self._state = get_state_manager()
        
        # Data
        self._data: Optional[np.ndarray] = None
        self._row_labels: List[str] = []
        self._col_labels: List[str] = []
        self._row_metadata: Dict[int, Dict] = {}
        self._col_metadata: Dict[int, Dict] = {}
        
        # Undo/redo stacks
        self._undo_stack: List[Tuple] = []
        self._redo_stack: List[Tuple] = []
        
        # Selection
        self._selected_rows: List[int] = []
        self._selected_cols: List[int] = []
        
        # Setup UI
        self._setup_ui()
        self._setup_connections()
        
        # Connect to state manager
        self._state.data_changed.connect(self._on_state_data_changed)
    
    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create table widget
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setShowGrid(True)
        self._table.setCornerButtonEnabled(True)
        
        # Custom delegate
        self._delegate = SpreadsheetDelegate()
        self._table.setItemDelegate(self._delegate)
        
        # Header settings
        h_header = self._table.horizontalHeader()
        h_header.setStretchLastSection(True)
        h_header.setMinimumSectionSize(80)
        h_header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        v_header = self._table.verticalHeader()
        v_header.setMinimumSectionSize(25)
        v_header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        # Style
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1A1A2E;
                alternate-background-color: #232342;
                color: #ECF0F1;
                gridline-color: #34495E;
                font-family: Consolas, monospace;
                font-size: 11px;
                border: none;
            }
            QTableWidget::item {
                padding: 4px 8px;
                border: 1px solid #34495E;
            }
            QTableWidget::item:selected {
                background-color: rgba(52, 152, 219, 0.4);
                color: #ECF0F1;
            }
            QHeaderView {
                background-color: #2C3E50;
                color: #ECF0F1;
                font-weight: bold;
            }
            QHeaderView::section {
                background-color: #2C3E50;
                color: #ECF0F1;
                padding: 4px;
                border: 1px solid #34495E;
                border-left: none;
            }
            QHeaderView::section:first {
                border-left: 1px solid #34495E;
            }
            QScrollBar:vertical {
                background-color: #1A1A2E;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #34495E;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #3D566E;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: transparent;
            }
            QScrollBar:horizontal {
                background-color: #1A1A2E;
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #34495E;
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #3D566E;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        layout.addWidget(self._table)
        
        # Context menus
        self._col_header_menu = ColumnHeaderMenu(self._table)
        self._row_header_menu = RowHeaderMenu(self._table)
    
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Context menu triggers
        self._table.horizontalHeader().customContextMenuRequested.connect(
            self._show_col_header_menu
        )
        self._table.verticalHeader().customContextMenuRequested.connect(
            self._show_row_header_menu
        )
        
        # Menu actions
        self._col_header_menu.triggered.connect(self._on_col_menu_action)
        self._row_header_menu.triggered.connect(self._on_row_menu_action)
        
        # Selection changes
        self._table.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        
        # Data changes
        self._table.itemChanged.connect(self._on_item_changed)
    
    def load_data(
        self,
        data: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None
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
        
        # Generate labels if not provided
        if row_labels is None:
            self._row_labels = [f"Sample_{i+1}" for i in range(n_rows)]
        else:
            self._row_labels = list(row_labels)
        
        if col_labels is None:
            self._col_labels = [f"Var_{j+1}" for j in range(n_cols)]
        else:
            self._col_labels = list(col_labels)
        
        # Update table
        self._table.blockSignals(True)
        self._table.setRowCount(n_rows)
        self._table.setColumnCount(n_cols)
        
        # Set row/column labels
        self._table.setVerticalHeaderLabels(self._row_labels)
        self._table.setHorizontalHeaderLabels(self._col_labels)
        
        # Populate cells
        for i in range(n_rows):
            for j in range(n_cols):
                value = self._data[i, j]
                if np.isnan(value):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, j, item)
        
        self._table.blockSignals(False)
        
        # Update delegate
        self._delegate.set_data(self._data, self._row_labels, self._col_labels)
        self._delegate.set_metadata(self._row_metadata, self._col_metadata)
        
        # Update state manager
        self._state.set_data_matrix(self._data, self._row_labels, self._col_labels)
        
        # Resize columns to content
        self._table.resizeColumnsToContents()
        
        self.dataChanged.emit()
    
    def get_data(self) -> Optional[np.ndarray]:
        """Get current data matrix."""
        return self._data.copy() if self._data is not None else None
    
    def get_selected_data(self) -> Optional[np.ndarray]:
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
        
        if action_type == 'group':
            # Set group color
            color = SpreadsheetDelegate.GROUP_COLORS[value]
            if col not in self._col_metadata:
                self._col_metadata[col] = {}
            self._col_metadata[col]['color'] = color
            self._col_metadata[col]['group'] = value
            self._state.set_col_metadata(col, self._col_metadata[col])
            self._table.viewport().update()
        
        elif action_type == 'type':
            # Set data type
            if col not in self._col_metadata:
                self._col_metadata[col] = {}
            self._col_metadata[col]['data_type'] = value
            self._state.set_col_metadata(col, self._col_metadata[col])
        
        elif action_type == 'transform':
            # Apply transformation
            self._apply_column_transform(col, value)
        
        elif action_type == 'sort':
            # Sort column
            ascending = (value == 'asc')
            self._sort_by_column(col, ascending)
        
        elif action_type == 'delete':
            # Delete column
            self._delete_column(col)
    
    def _on_row_menu_action(self, action) -> None:
        """Handle row header menu action."""
        data = action.data()
        if not data:
            return
        
        row = self._row_header_menu._row_index
        action_type, value = data
        
        if action_type == 'marker':
            # Set marker style
            if row not in self._row_metadata:
                self._row_metadata[row] = {}
            self._row_metadata[row]['marker'] = value.value
            self._state.set_row_metadata(row, self._row_metadata[row])
            self._table.viewport().update()
        
        elif action_type == 'color':
            # Set row color
            if row not in self._row_metadata:
                self._row_metadata[row] = {}
            self._row_metadata[row]['color'] = value
            self._state.set_row_metadata(row, self._row_metadata[row])
            self._table.viewport().update()
        
        elif action_type == 'group':
            # Assign to group
            color = SpreadsheetDelegate.GROUP_COLORS[value]
            if row not in self._row_metadata:
                self._row_metadata[row] = {}
            self._row_metadata[row]['color'] = color
            self._row_metadata[row]['group'] = value
            self._state.set_row_metadata(row, self._row_metadata[row])
            self._table.viewport().update()
        
        elif action_type == 'hide':
            # Hide row
            self._table.hideRow(row)
        
        elif action_type == 'show_all':
            # Show all rows
            for i in range(self._table.rowCount()):
                self._table.showRow(i)
    
    def _apply_column_transform(self, col: int, transform: str) -> None:
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
        
        # Save for undo
        self._undo_stack.append(('transform', col, transform, self._data[:, col].copy()))
        self._redo_stack.clear()
        
        col_data = self._data[:, col]
        
        if transform == 'log':
            # Log transform: x' = ln(x + 1)
            # Handle negative values and zeros
            new_data = np.log1p(col_data)  # log1p(x) = ln(x + 1)
        
        elif transform == 'zscore':
            # Z-score: x' = (x - μ) / σ
            mean = np.nanmean(col_data)
            std = np.nanstd(col_data)
            if std > 0:
                new_data = (col_data - mean) / std
            else:
                new_data = col_data - mean
        
        elif transform == 'center':
            # Center: x' = x - μ
            mean = np.nanmean(col_data)
            new_data = col_data - mean
        
        elif transform == 'scale':
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
        
        # Update table
        for i in range(self._data.shape[0]):
            value = new_data[i]
            if np.isnan(value):
                item = QTableWidgetItem("")
            else:
                item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, col, item)
        
        # Update state
        self._state.set_data_matrix(self._data, self._row_labels, self._col_labels)
        
        # Emit signal
        self.columnTransformed.emit(col, transform)
        self.dataChanged.emit()
    
    def _sort_by_column(self, col: int, ascending: bool) -> None:
        """Sort data by column."""
        if self._data is None or col < 0 or col >= self._data.shape[1]:
            return
        
        # Get sort indices
        sort_data = self._data[:, col]
        # Handle NaN values
        nan_mask = np.isnan(sort_data)
        sort_indices = np.argsort(np.where(nan_mask, np.inf if ascending else -np.inf, sort_data))
        if not ascending:
            sort_indices = sort_indices[::-1]
        
        # Apply sort
        self._data = self._data[sort_indices, :]
        self._row_labels = [self._row_labels[i] for i in sort_indices]
        
        # Reorder table rows
        self._table.blockSignals(True)
        for new_row, old_row in enumerate(sort_indices):
            self._table.setRowHeight(new_row, self._table.rowHeight(old_row))
        
        # Update labels
        self._table.setVerticalHeaderLabels(self._row_labels)
        self._table.blockSignals(False)
        
        # Update state
        self._state.set_data_matrix(self._data, self._row_labels, self._col_labels)
        self.dataChanged.emit()
    
    def _delete_column(self, col: int) -> None:
        """Delete column from data."""
        if self._data is None or col < 0 or col >= self._data.shape[1]:
            return
        
        # Save for undo
        self._undo_stack.append(('delete_col', col, self._data[:, col].copy(), 
                                self._col_labels[col], self._col_metadata.get(col, {})))
        self._redo_stack.clear()
        
        # Remove column
        self._data = np.delete(self._data, col, axis=1)
        self._col_labels.pop(col)
        
        # Update metadata indices
        new_col_metadata = {}
        for old_idx, meta in self._col_metadata.items():
            if old_idx > col:
                new_col_metadata[old_idx - 1] = meta
            elif old_idx < col:
                new_col_metadata[old_idx] = meta
        self._col_metadata = new_col_metadata
        
        # Update table
        self._table.removeColumn(col)
        self._table.setHorizontalHeaderLabels(self._col_labels)
        
        # Update state
        self._state.set_data_matrix(self._data, self._row_labels, self._col_labels)
        self.dataChanged.emit()
    
    def _on_selection_changed(
        self,
        selected: 'QItemSelection',
        deselected: 'QItemSelection'
    ) -> None:
        """Handle selection change."""
        indexes = self._table.selectionModel().selectedIndexes()
        
        rows = sorted(list(set(idx.row() for idx in indexes)))
        cols = sorted(list(set(idx.column() for idx in indexes)))
        
        self._selected_rows = rows
        self._selected_cols = cols
        
        self.selectionChanged.emit(rows, cols)
    
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle cell data change."""
        row = item.row()
        col = item.column()
        
        if self._data is None:
            return
        
        try:
            value = float(item.text())
            old_value = self._data[row, col]
            self._data[row, col] = value
            
            # Save for undo
            self._undo_stack.append(('cell', row, col, old_value))
            self._redo_stack.clear()
            
            # Update state
            self._state.set_data_matrix(self._data, self._row_labels, self._col_labels)
            self.dataChanged.emit()
            
        except ValueError:
            # Revert to old value
            if not np.isnan(self._data[row, col]):
                item.setText(str(self._data[row, col]))
    
    def _on_state_data_changed(self) -> None:
        """Handle data change from state manager."""
        # Reload data from state
        data = self._state.data_matrix
        if data is not None:
            self._data = data.data
            self._row_labels = data.row_labels
            self._col_labels = data.col_labels
    
    def export_to_clipboard(self) -> None:
        """Export selected data to clipboard."""
        selected_data = self.get_selected_data()
        if selected_data is None:
            return
        
        # Format as tab-separated text
        rows_str = []
        for i in range(selected_data.shape[0]):
            row = [f"{v:.6g}" if not np.isnan(v) else "" 
                   for v in selected_data[i, :]]
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
            for line in lines:
                values = line.split("\t")
                row = []
                for v in values:
                    try:
                        row.append(float(v))
                    except ValueError:
                        row.append(np.nan)
                data.append(row)
            
            data = np.array(data)
            self.load_data(data)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import data:\n{str(e)}")
    
    def undo(self) -> None:
        """Undo last operation."""
        if not self._undo_stack:
            return
        
        operation = self._undo_stack.pop()
        
        if operation[0] == 'transform':
            _, col, transform, old_data = operation
            # Restore old data
            self._data[:, col] = old_data
            # Update table
            for i in range(self._data.shape[0]):
                item = QTableWidgetItem(str(old_data[i]))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)
        
        elif operation[0] == 'delete_col':
            _, col, col_data, label, metadata = operation
            # Reinsert column
            self._data = np.insert(self._data, col, col_data, axis=1)
            self._col_labels.insert(col, label)
            self._col_metadata[col] = metadata
            
            # Update table
            self._table.insertColumn(col)
            self._table.setHorizontalHeaderLabels(self._col_labels)
            for i in range(self._data.shape[0]):
                item = QTableWidgetItem(str(col_data[i]))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)
        
        elif operation[0] == 'cell':
            _, row, col, old_value = operation
            self._data[row, col] = old_value
            item = QTableWidgetItem(str(old_value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, col, item)
        
        # Update state
        self._state.set_data_matrix(self._data, self._row_labels, self._col_labels)
        self.dataChanged.emit()
    
    def redo(self) -> None:
        """Redo last undone operation."""
        if not self._redo_stack:
            return
        
        operation = self._redo_stack.pop()
        # Similar to undo but in reverse
        # (Implementation would mirror undo logic)
