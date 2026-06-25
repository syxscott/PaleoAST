# =============================================================================
# FILE: views/ui_navigation.py
# =============================================================================
"""
Navigation Tree Widget for PaleoAST

This module implements the left-side navigation tree with hierarchical
organization of analysis functions, following the Observer pattern to
synchronize with the application state.

Design Patterns:
    - Observer Pattern: Navigation tree observes StateManager
    - Composite Pattern: Tree items form hierarchical structure
    - Command Pattern: Each navigation item represents a command/action

Author: PaleoAST Development Team
version: 1.0.1
"""

from collections.abc import Callable
from enum import Enum

from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.design_system import BorderRadius, ColorPalette, Typography, get_palette
from config.i18n import _


class NavigationCategory(Enum):
    """Navigation categories."""

    DATA = "Data Management"
    UNIVARIATE = "Univariate"
    MULTIVARIATE = "Multivariate"
    MORPHOMETRICS = "Morphometrics"
    STRATIGRAPHY = "Stratigraphy"
    ECOLOGY = "Ecology"


class NavigationItem:
    """
    Navigation item data class.

    Represents a single item in the navigation tree with associated
    metadata and configuration.
    """

    def __init__(
        self,
        name: str,
        category: NavigationCategory,
        icon_type: str = "",
        children: list["NavigationItem"] | None = None,
        action_callback: Callable | None = None,
    ) -> None:
        self.name = name
        self.category = category
        self.icon_type = icon_type
        self.children = children or []
        self.action_callback = action_callback
        self.section = category.value

        # State
        self.is_expanded = False
        self.is_selected = False

    def __repr__(self) -> str:
        return f"NavigationItem(name={self.name}, category={self.category.value})"


class NavigationDelegate(QStyledItemDelegate):
    """
    Custom delegate for navigation tree items.

    Renders navigation items with custom icons and styling.
    Extends QStyledItemDelegate to provide custom painting.
    """

    # Icon definitions
    ICON_PATTERNS = {
        "folder": "M4,4H20V20H4V4M6,8V18H18V8",
        "file": "M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20",
        "matrix": "M3,3H11V11H3V3M3,13H11V21H3V13M13,3H21V11H13V3M13,13H21V21H13V13M3,5V9H9V5H3M15,5V9H21V5H15M3,15V19H9V15H3M15,15V19H21V15H15",
        "chart": "M22,21H2V3H4V19H6V10H10V19H12V6H16V19H18V14H22V21",
        "diversity": "M12,5.5A3.5,3.5 0 0,1 15.5,9A3.5,3.5 0 0,1 12,12.5A3.5,3.5 0 0,1 8.5,9A3.5,3.5 0 0,1 12,5.5M5,8C5.56,8 6.08,8.15 6.53,8.42C6.38,9.85 6.8,11.27 7.66,12.38C7.16,13.34 6.16,14 5,14A3,3 0 0,1 2,11A3,3 0 0,1 5,8M19,8A3,3 0 0,1 22,11A3,3 0 0,1 19,14C17.84,14 16.84,13.34 16.34,12.38C17.2,11.27 17.62,9.85 17.47,8.42C17.92,8.15 18.44,8 19,8M5.5,18.25C5.5,16.18 8.41,14.5 12,14.5C15.59,14.5 18.5,16.18 18.5,18.25V20H5.5V18.25M0,20V18.5C0,17.11 1.89,15.94 4.45,15.6C3.86,16.28 3.5,17.22 3.5,18.25V20H0M24,20H20.5V18.25C20.5,17.22 20.14,16.28 19.55,15.6C22.11,15.94 24,17.11 24,18.5V20",
        "morphometrics": "M12,5.5L4.5,20H19.5L12,5.5M12,2L1,21H23L12,2Z",
        "stratigraphy": "M3,3H5V5H3V3M7,3H9V5H7V3M11,3H13V5H11V3M15,3H17V5H15V3M19,3H21V5H19V3M3,7H5V9H3V7M7,7H9V9H7V7M11,7H13V9H11V7M15,7H17V9H15V7M19,7H21V9H19V7M3,11H5V13H3V11M7,11H9V13H7V11M11,11H13V13H11V11M15,11H17V13H15V11M19,11H21V13H19V11M3,15H5V17H3V15M7,15H9V17H7V15M11,15H13V17H11V15M15,15H17V17H15V15M19,15H21V17H19V15M3,19H5V21H3V19M7,19H9V21H7V19M11,19H13V21H11V19M15,19H17V21H15V19M19,19H21V21H19V19",
        "pca": "M7.5,5.6L5,7L6.4,4.5L5,2L7.5,3.4L5,1L8,4M17.5,10.5L20,9L18.6,11.5L20,14L17.5,12.6L20,16L13,10L17.5,10.5M5,16L7.5,14.6L5,18L10,12L5,16M13,11L17.5,10.5L14,9L15.5,7.5L13,9L14.5,6L13,5L11,7L5,14L13,11Z",
        "settings": "M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.21,8.95 2.27,9.22 2.46,9.37L4.57,11C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.21,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.67 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z",
        "import": "M9,3V4H4V6H5V4A2,2 0 0,1 7,2H9V3M7,6V8H9V6H7M9,8V10H7V12H9V14H7V16H9V18H7V20H9A2,2 0 0,0 11,18V20A2,2 0 0,0 13,22H15A2,2 0 0,0 17,20V18A2,2 0 0,0 15,16H13V14H15V12H13V10H15V8H13V6H15V4H13V2H15A2,2 0 0,0 17,4V6H19V4H23V6H19V8H21V6H23V10H21V8H23V14H21V12H23V16H21V14H23V18H21V16H23V18A2,2 0 0,0 21,20H19A2,2 0 0,0 17,18V20H19V22H15V20H17V18H15V16H17V14H15V12H17V10H15V8H17V6H15V4H13V6H11V8H13V10H11V12H13V14H11V16H13V18H11V20H13V22H9V20H11V18H9V16H11V14H9V12H11V10H9V8H11V6H9V4H7V6H5V8H7V10H5V12H7V14H5V16H7V18H5V20H7V22H3V2H9",
        "export": "M23,12L19,8V11H10V13H19V16M1,18V6A2,2 0 0,1 3,4H15A2,2 0 0,1 17,6V9H15V6H3V18H15V15H17V18A2,2 0 0,1 15,20H3A2,2 0 0,1 1,18",
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._expanded_items: set = set()
        self._hovered_index = None

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        """
        Paint navigation item with custom rendering.

        This method overrides the default QStyledItemDelegate paint() to provide:
        - Custom icons for each item type
        - Hover highlighting
        - Selection state rendering
        - Expand/collapse indicator icons
        """
        painter.save()

        # Get item data: try to obtain QTreeWidgetItem via the view if available,
        # otherwise fall back to using the index and model directly.
        widget = getattr(option, "widget", None)
        item = None
        try:
            if widget is not None and hasattr(widget, "itemFromIndex"):
                # QTreeWidget has itemFromIndex
                item = widget.itemFromIndex(index)
        except Exception:
            item = None

        # Determine item state
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # Determine children/expanded state in a model-agnostic way
        try:
            # Prefer model rowCount for child count
            has_children = index.model().rowCount(index) > 0
        except Exception:
            has_children = False

        # is_expanded is a property of the view, not the model
        if widget is not None and hasattr(widget, "isExpanded"):
            try:
                is_expanded = widget.isExpanded(index)
            except Exception:
                is_expanded = False
        else:
            is_expanded = False

        # Background painting
        if is_selected:
            # Selected state - use primary with 15% alpha
            selected_color = QColor(59, 130, 246, 38)
            painter.fillRect(option.rect, QBrush(selected_color))
        elif is_hovered:
            # Hover state - use primary with 8% alpha
            hover_color = QColor(30, 64, 175, 20)
            painter.fillRect(option.rect, QBrush(hover_color))

        # Determine icon type from item data (UserRole)
        icon_type = index.data(Qt.ItemDataRole.UserRole)
        if not icon_type and item is not None:
            try:
                icon_type = item.data(0, Qt.ItemDataRole.UserRole)
            except Exception:
                icon_type = None

        # Icon dimensions
        icon_size = 16
        icon_margin = 8
        icon_x = option.rect.left() + icon_margin
        icon_y = option.rect.top() + (option.rect.height() - icon_size) // 2

        # Draw folder/file icon
        if icon_type == "folder":
            self._draw_folder_icon(painter, icon_x, icon_y, icon_size, is_expanded)
        elif icon_type == "pca":
            self._draw_pca_icon(painter, icon_x, icon_y, icon_size)
        elif icon_type == "diversity":
            self._draw_diversity_icon(painter, icon_x, icon_y, icon_size)
        elif icon_type == "morphometrics":
            self._draw_morpho_icon(painter, icon_x, icon_y, icon_size)
        elif icon_type == "stratigraphy":
            self._draw_strat_icon(painter, icon_x, icon_y, icon_size)
        else:
            self._draw_default_icon(painter, icon_x, icon_y, icon_size)

        # Text rendering
        text_left_margin = icon_margin + icon_size + icon_margin + 4
        text_right_margin = -28 if has_children else -4
        text_rect = option.rect.adjusted(text_left_margin, 0, text_right_margin, 0)

        # Get display text (fall back to item text if present)
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if (text is None or text == "") and item is not None:
            try:
                text = item.text(0)
            except Exception:
                text = ""
        if text is None:
            text = ""

        # Draw text
        c = ColorPalette()
        text_color = c.primary if is_selected else c.text_primary
        if is_hovered and not is_selected:
            text_color = c.primary

        t = Typography()
        font = QFont(t.family_primary, t.body_sm_size)
        # Top-level if no valid parent index
        if not index.parent().isValid():
            font.setBold(True)

        painter.setFont(font)
        painter.setPen(QPen(QColor(text_color)))

        # Text alignment
        alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(text_rect, int(alignment), text)

        # Draw expand/collapse arrow for items with children
        if has_children:
            arrow_x = option.rect.right() - 20
            arrow_y = option.rect.top() + option.rect.height() // 2
            self._draw_expand_arrow(painter, arrow_x, arrow_y, 8, is_expanded, is_selected)

        painter.restore()

    def _draw_folder_icon(self, painter: QPainter, x: float, y: float, size: float, is_expanded: bool) -> None:
        """Draw folder icon with optional open state."""
        c = ColorPalette()
        color = QColor(c.primary)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(QBrush(color.lighter(130)))

        if is_expanded:
            # Open folder
            path = QPainterPath()
            path.moveTo(x, y + size * 0.3)
            path.lineTo(x, y + size * 0.8)
            path.lineTo(x + size * 0.9, y + size * 0.8)
            path.lineTo(x + size * 0.9, y + size * 0.3)
            path.lineTo(x + size * 0.6, y + size * 0.3)
            path.lineTo(x + size * 0.5, y + size * 0.1)
            path.lineTo(x + size * 0.1, y + size * 0.3)
            path.closeSubpath()
            painter.drawPath(path)
        else:
            # Closed folder
            path = QPainterPath()
            path.moveTo(x, y + size * 0.3)
            path.lineTo(x, y + size * 0.9)
            path.lineTo(x + size * 0.9, y + size * 0.9)
            path.lineTo(x + size * 0.9, y + size * 0.3)
            path.lineTo(x + size * 0.6, y + size * 0.3)
            path.lineTo(x + size * 0.5, y + size * 0.1)
            path.lineTo(x + size * 0.1, y + size * 0.3)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color.lighter(130)))
            painter.drawPath(path)

    def _draw_pca_icon(self, painter: QPainter, x: float, y: float, size: float) -> None:
        """Draw PCA-style icon with coordinate axes."""
        center_x = int(x + size // 2)
        center_y = int(y + size // 2)
        axis_length = int(size // 2)

        # X-axis (PC1)
        painter.setPen(QPen(QColor("#E74C3C"), 1.5))
        painter.drawLine(center_x, center_y, center_x + axis_length, center_y)

        # Y-axis (PC2)
        painter.setPen(QPen(QColor("#27AE60"), 1.5))
        painter.drawLine(center_x, center_y, center_x, center_y - axis_length)

        # Point at origin
        painter.setBrush(QBrush(QColor("#F39C12")))
        painter.drawEllipse(center_x - 2, center_y - 2, 4, 4)

    def _draw_diversity_icon(self, painter: QPainter, x: float, y: float, size: float) -> None:
        """Draw biodiversity tree icon."""
        painter.setPen(QPen(QColor("#27AE60"), 1.5))

        # Main trunk
        trunk_x = int(x + size // 2)
        painter.drawLine(trunk_x, int(y + size * 0.9), trunk_x, int(y + size * 0.4))

        # Branches
        painter.drawLine(trunk_x, int(y + size * 0.6), int(x + size * 0.2), int(y + size * 0.2))
        painter.drawLine(trunk_x, int(y + size * 0.6), int(x + size * 0.8), int(y + size * 0.2))
        painter.drawLine(trunk_x, int(y + size * 0.4), trunk_x, int(y + size * 0.1))

        # Leaf nodes
        painter.setBrush(QBrush(QColor("#27AE60")))
        for px, py in [
            (x + size * 0.2, y + size * 0.15),
            (x + size * 0.8, y + size * 0.15),
            (trunk_x, y + size * 0.05),
        ]:
            painter.drawEllipse(int(px), int(py), 4, 4)

    def _draw_morpho_icon(self, painter: QPainter, x: float, y: float, size: float) -> None:
        """Draw morphometrics landmark icon."""
        # Triangle connecting landmarks
        from PyQt6.QtCore import QPointF

        points = [
            QPointF(x + size * 0.2, y + size * 0.7),
            QPointF(x + size * 0.8, y + size * 0.7),
            QPointF(x + size * 0.5, y + size * 0.2),
        ]

        painter.setPen(QPen(QColor("#9B59B6"), 1.5))
        painter.drawPolygon(points)

        # Landmark points
        painter.setBrush(QBrush(QColor("#9B59B6")))
        for pt in points:
            painter.drawEllipse(int(pt.x() - 3), int(pt.y() - 3), 6, 6)

    def _draw_strat_icon(self, painter: QPainter, x: float, y: float, size: float) -> None:
        """Draw stratigraphy layers icon."""
        layer_height = size // 4
        colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB"]

        for i, color in enumerate(colors):
            painter.fillRect(int(x), int(y + i * layer_height), int(size), int(layer_height - 1), QBrush(QColor(color)))

    def _draw_default_icon(self, painter: QPainter, x: float, y: float, size: float) -> None:
        """Draw default document icon."""
        painter.setPen(QPen(QColor("#3498DB"), 1.5))
        painter.setBrush(QBrush(QColor("#E8F4F8")))
        painter.drawRect(int(x), int(y), int(size), int(size))

    def _draw_expand_arrow(
        self, painter: QPainter, x: float, y: float, size: float, is_expanded: bool, is_selected: bool
    ) -> None:
        """Draw expand/collapse arrow."""
        c = ColorPalette()
        color = c.text_disabled if not is_selected else c.bg_primary
        painter.setPen(QPen(QColor(color), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()

        if is_expanded:
            # Down arrow
            path.moveTo(x, y - size // 2)
            path.lineTo(x + size // 2, y + size // 2)
            path.lineTo(x - size // 2, y + size // 2)
        else:
            # Right arrow
            path.moveTo(x - size // 2, y)
            path.lineTo(x + size // 2, y + size // 2)
            path.lineTo(x + size // 2, y - size // 2)

        path.closeSubpath()
        painter.drawPath(path)

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Return size hint for item."""
        return QSize(250, 32)


class NavigationTree(QWidget):
    """
    Navigation Tree Widget.

    Provides hierarchical navigation for all analysis functions.
    Emits signals when items are clicked.

    Signals:
        itemClicked: Emitted when navigation item is clicked (NavigationItem)
        sectionChanged: Emitted when section changes (str)
    """

    itemClicked = pyqtSignal(object)  # NavigationItem
    sectionChanged = pyqtSignal(str)  # section name

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._is_dark_theme = False
        self._setup_ui()
        self._build_navigation_tree()
        self._setup_connections()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        t = Typography()
        r = BorderRadius()
        self._tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {c.bg_primary};
                border: 1px solid {c.border_light};
                border-radius: {r.lg};
                outline: none;
                color: {c.text_primary};
                font-family: {t.family_primary};
                font-size: {t.body_sm_size}px;
            }}
            QTreeWidget::item {{
                padding: 6px 4px;
                min-height: 32px;
                border-radius: {r.md};
            }}
            QTreeWidget::item:hover {{
                background-color: {c.hover_overlay};
            }}
            QTreeWidget::item:selected {{
                background-color: {c.selected_overlay};
                color: {c.primary};
            }}
            QTreeWidget::item:selected:active {{
                background-color: {c.active_overlay};
            }}
            QTreeWidget::branch {{
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {c.bg_secondary};
                width: 10px;
                margin: 0px;
                border-radius: {r.sm};
            }}
            QScrollBar::handle:vertical {{
                background-color: {c.border_medium};
                min-height: 20px;
                border-radius: {r.sm};
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
        """)
        self._filter_input.setStyleSheet(f"""
            QLineEdit {{
                color: {c.text_primary};
                background-color: {c.bg_secondary};
                border: 1px solid {c.border_light};
                border-radius: {r.lg};
                padding: 6px 8px;
                font-size: {t.body_size}px;
            }}
            QLineEdit:focus {{
                border: 1px solid {c.primary};
                background-color: {c.bg_primary};
            }}
        """)

    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(16)
        self._tree.setAnimated(True)
        self._tree.setAlternatingRowColors(False)
        self._tree.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)
        self._tree.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # Set custom delegate
        self._delegate = NavigationDelegate()
        self._tree.setItemDelegate(self._delegate)

        # Search/filter input
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText(_("Filter:") + "...")
        self._filter_input.setClearButtonEnabled(True)

        layout.addWidget(self._tree)
        layout.addWidget(self._filter_input)
        self._filter_input.textChanged.connect(self._filter_tree)

        # Apply stylesheet
        self._apply_stylesheet()

    def _filter_tree(self, text: str) -> None:
        """Filter tree items by text (case-insensitive)."""
        text = text.lower().strip()
        for i in range(self._tree.topLevelItemCount()):
            category = self._tree.topLevelItem(i)
            if not text:
                category.setHidden(False)
                self._show_all_children(category)
                continue
            any_visible = False
            for j in range(category.childCount()):
                child = category.child(j)
                child_match = text in child.text(0).lower()
                # Also check grandchildren
                grandchild_match = False
                for k in range(child.childCount()):
                    grandchild = child.child(k)
                    gc_match = text in grandchild.text(0).lower()
                    grandchild.setHidden(not gc_match)
                    if gc_match:
                        grandchild_match = True
                # Show child if it matches or any grandchild matches
                child_visible = child_match or grandchild_match
                child.setHidden(not child_visible)
                if child_visible:
                    any_visible = True
                    # Auto-expand parent when grandchildren match
                    if grandchild_match and not child_match:
                        child.setExpanded(True)
            category.setHidden(not any_visible and text not in category.text(0).lower())

    def _show_all_children(self, item: QTreeWidgetItem) -> None:
        """Recursively show all children of an item."""
        for i in range(item.childCount()):
            child = item.child(i)
            child.setHidden(False)
            self._show_all_children(child)

    def _build_navigation_tree(self) -> None:
        """
        Build the navigation tree structure.

        Tree structure:
        - Data Management
            - Import Data
            - Export Data
            - Matrix Operations
        - Univariate
            - Descriptive Statistics
            - Histogram
            - Box Plot
        - Multivariate
            - PCA
            - PCoA
            - NMDS
            - Cluster Analysis
            - Group Tests
                - ANOSIM
                - PERMANOVA
        - Morphometrics
            - GPA Alignment
            - TPS Deformation
            - Relative Warps
        - Stratigraphy
            - Unitary Associations
            - Spectral Analysis
            - Confidence Intervals
        - Ecology
            - Alpha Diversity
            - Beta Diversity
            - Rarefaction
        """
        # Root items for each category
        categories = {
            NavigationCategory.DATA: self._create_category_item(
                _("Data Management"), "folder", NavigationCategory.DATA
            ),
            NavigationCategory.UNIVARIATE: self._create_category_item(
                _("Univariate"), "folder", NavigationCategory.UNIVARIATE
            ),
            NavigationCategory.MULTIVARIATE: self._create_category_item(
                _("Multivariate"), "folder", NavigationCategory.MULTIVARIATE
            ),
            NavigationCategory.MORPHOMETRICS: self._create_category_item(
                _("Morphometrics"), "folder", NavigationCategory.MORPHOMETRICS
            ),
            NavigationCategory.STRATIGRAPHY: self._create_category_item(
                _("Stratigraphy"), "folder", NavigationCategory.STRATIGRAPHY
            ),
            NavigationCategory.ECOLOGY: self._create_category_item(_("Ecology"), "folder", NavigationCategory.ECOLOGY),
        }

        # Data Management children
        data_children = [
            NavigationItem(_("Import Data"), NavigationCategory.DATA, "import"),
            NavigationItem(_("Export Data"), NavigationCategory.DATA, "export"),
            NavigationItem(_("Matrix Operations"), NavigationCategory.DATA, "matrix"),
        ]
        for child in data_children:
            categories[NavigationCategory.DATA].children.append(child)

        # Univariate children
        univariate_children = [
            NavigationItem(_("Summary"), NavigationCategory.UNIVARIATE, "chart"),
            NavigationItem(_("Normality"), NavigationCategory.UNIVARIATE, "chart"),
            NavigationItem(_("t-test"), NavigationCategory.UNIVARIATE, "chart"),
            NavigationItem(_("ANOVA"), NavigationCategory.UNIVARIATE, "chart"),
            NavigationItem(_("Kruskal-Wallis"), NavigationCategory.UNIVARIATE, "chart"),
        ]
        for child in univariate_children:
            categories[NavigationCategory.UNIVARIATE].children.append(child)

        # Multivariate children
        multivar_children = [
            NavigationItem("PCA", NavigationCategory.MULTIVARIATE, "pca"),
            NavigationItem("PCoA", NavigationCategory.MULTIVARIATE, "pca"),
            NavigationItem("NMDS", NavigationCategory.MULTIVARIATE, "pca"),
            NavigationItem("LDA", NavigationCategory.MULTIVARIATE, "pca"),
            NavigationItem(_("Clustering"), NavigationCategory.MULTIVARIATE, "chart"),
            NavigationItem(_("Group Tests"), NavigationCategory.MULTIVARIATE, "folder"),
            NavigationItem("SIMPER", NavigationCategory.MULTIVARIATE, "chart"),
        ]
        for child in multivar_children:
            categories[NavigationCategory.MULTIVARIATE].children.append(child)

        # Group Tests children (under Multivariate)
        group_tests_children = [
            NavigationItem("ANOSIM", NavigationCategory.MULTIVARIATE, "chart"),
            NavigationItem("PERMANOVA", NavigationCategory.MULTIVARIATE, "chart"),
        ]
        categories[NavigationCategory.MULTIVARIATE].children[-2].children.extend(group_tests_children)

        # Morphometrics children
        morpho_children = [
            NavigationItem(_("GPA Alignment"), NavigationCategory.MORPHOMETRICS, "morphometrics"),
            NavigationItem(_("TPS Deformation"), NavigationCategory.MORPHOMETRICS, "morphometrics"),
            NavigationItem(_("Relative Warps"), NavigationCategory.MORPHOMETRICS, "morphometrics"),
            NavigationItem("EFA", NavigationCategory.MORPHOMETRICS, "morphometrics"),
            NavigationItem(_("Eigenshape"), NavigationCategory.MORPHOMETRICS, "morphometrics"),
            NavigationItem(_("Allometry"), NavigationCategory.MORPHOMETRICS, "morphometrics"),
            NavigationItem(_("Evolution Rate"), NavigationCategory.MORPHOMETRICS, "morphometrics"),
        ]
        for child in morpho_children:
            categories[NavigationCategory.MORPHOMETRICS].children.append(child)

        # Stratigraphy children
        strat_children = [
            NavigationItem(_("Unitary Associations"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem(_("Spectral Analysis"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem("CONISS", NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem(_("Markov"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem(_("Directional"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem(_("Extinction Intervals"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            # Industrial-grade entries added in v1.0.1 - previously
            # accessible only via the ribbon; now also exposed in the
            # left navigation tree.
            NavigationItem(_("Isotope"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem(
                _("Stratigraphic Correlation"),
                NavigationCategory.STRATIGRAPHY,
                "stratigraphy",
            ),
            NavigationItem(_("Wavelet"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
            NavigationItem(_("CA Axis"), NavigationCategory.STRATIGRAPHY, "stratigraphy"),
        ]
        for child in strat_children:
            categories[NavigationCategory.STRATIGRAPHY].children.append(child)

        # Ecology children
        ecology_children = [
            NavigationItem(_("Diversity"), NavigationCategory.ECOLOGY, "diversity"),
            NavigationItem(_("Rarefaction"), NavigationCategory.ECOLOGY, "diversity"),
            NavigationItem(_("Abundance Models"), NavigationCategory.ECOLOGY, "diversity"),
            NavigationItem("SHE", NavigationCategory.ECOLOGY, "diversity"),
            NavigationItem(_("Beta Diversity"), NavigationCategory.ECOLOGY, "diversity"),
            NavigationItem(_("Null Models"), NavigationCategory.ECOLOGY, "diversity"),
        ]
        for child in ecology_children:
            categories[NavigationCategory.ECOLOGY].children.append(child)

        # Add all categories to tree
        for category in categories.values():
            self._add_item_to_tree(None, category)

    def _create_category_item(
        self, name: str, icon_type: str = "folder", category: NavigationCategory | None = None
    ) -> NavigationItem:
        """Create a navigation item for a category."""
        if category is None:
            # Fallback: determine category from name (for backward compatibility)
            category_map = {
                "Data Management": NavigationCategory.DATA,
                "Univariate": NavigationCategory.UNIVARIATE,
                "Multivariate": NavigationCategory.MULTIVARIATE,
                "Morphometrics": NavigationCategory.MORPHOMETRICS,
                "Stratigraphy": NavigationCategory.STRATIGRAPHY,
                "Ecology": NavigationCategory.ECOLOGY,
            }
            category = category_map.get(name, NavigationCategory.DATA)
        return NavigationItem(name=name, category=category, icon_type=icon_type)

    def _add_item_to_tree(self, parent: QTreeWidgetItem | None, item: NavigationItem) -> QTreeWidgetItem:
        """Add navigation item to tree widget recursively."""
        tree_item = QTreeWidgetItem(parent)
        tree_item.setText(0, item.name)
        tree_item.setData(0, Qt.ItemDataRole.UserRole, item.icon_type)
        tree_item.setFlags(tree_item.flags() | Qt.ItemFlag.ItemIsSelectable)

        # Store NavigationItem reference
        tree_item._nav_item = item

        # Add children recursively
        for child in item.children:
            self._add_item_to_tree(tree_item, child)

        # Add to tree widget
        if parent is None:
            self._tree.addTopLevelItem(tree_item)
        else:
            parent.addChild(tree_item)

        return tree_item

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Handle item click.

        When a navigation item is clicked:
        1. Get the NavigationItem data
        2. Emit itemClicked signal with NavigationItem
        3. Update current section
        """
        # Get navigation item
        nav_item = getattr(item, "_nav_item", None)
        if not nav_item:
            return

        # Update selection
        self._tree.setCurrentItem(item)

        # Emit signals
        self.itemClicked.emit(nav_item)
        self.sectionChanged.emit(nav_item.section)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Handle item double-click.

        Double-click on item with children toggles expansion.
        Double-click on leaf item triggers the same action as single-click.
        """
        nav_item = getattr(item, "_nav_item", None)
        if not nav_item:
            return

        # Toggle expansion for items with children
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
        else:
            # Trigger the same action as single-click
            self._tree.setCurrentItem(item)
            self.itemClicked.emit(nav_item)
            self.sectionChanged.emit(nav_item.section)

    def get_current_item(self) -> NavigationItem | None:
        """Get currently selected navigation item."""
        current = self._tree.currentItem()
        if current:
            return getattr(current, "_nav_item", None)
        return None

    def expand_category(self, category: NavigationCategory) -> None:
        """Expand a category in the tree."""
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            nav_item = getattr(item, "_nav_item", None)
            if nav_item and nav_item.category == category:
                item.setExpanded(True)
                break

    def select_item(self, name: str, category: NavigationCategory) -> bool:
        """
        Select a specific item by name and category.

        Returns True if item found and selected, False otherwise.
        """
        # Find category item
        category_item = None
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            nav_item = getattr(item, "_nav_item", None)
            if nav_item and nav_item.category == category:
                category_item = item
                category_item.setExpanded(True)
                break

        if not category_item:
            return False

        # Find child item (including nested grandchildren)
        for i in range(category_item.childCount()):
            child = category_item.child(i)
            if child.text(0) == name:
                self._tree.setCurrentItem(child)
                return True
            # Search grandchildren
            for j in range(child.childCount()):
                grandchild = child.child(j)
                if grandchild.text(0) == name:
                    child.setExpanded(True)
                    self._tree.setCurrentItem(grandchild)
                    return True

        return False
