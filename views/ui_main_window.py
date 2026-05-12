# =============================================================================
# FILE: views/ui_main_window.py
# =============================================================================
"""
Modern Main Window for PaleoAST

This module implements the main application window with:
    - Left navigation tree (QTreeView)
    - Top ribbon toolbar with vector icons
    - Central workspace area
    - Status bar
    - Complete dark/light theme support

Design Patterns Used:
    - Observer Pattern: MainWindow observes StateManager for data changes
    - MVC Pattern: Coordinates views and controllers
    - Singleton Pattern: Uses StateManager for global state

Signals emitted:
    - dataChanged: Emitted when data matrix changes
    - analysisRequested: Emitted when user requests analysis
    - navigationChanged: Emitted when navigation item selected

Author: PaleoAST Development Team
Version: 1.0.0
"""

import logging
import sys
from enum import Enum

logger = logging.getLogger(__name__)

from PyQt6.QtCore import QPoint, QRect, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config.i18n import _, get_translator
from controllers.data_controller import DataController
from controllers.statistics_controller import StatisticsController
from models.state_manager import get_state_manager
from views.ui_dialogs import (
    ClusteringDialog,
    CONISSDialog,
    DirectionalDialog,
    DiversityDialog,
    EFADialog,
    ImportDialog,
    LDADialog,
    MarkovDialog,
    NMDSOptionsDialog,
    PCADialog,
    PCoADialog,
    RarefactionDialog,
    SimperDialog,
    UnivariateDialog,
)
from views.ui_navigation import NavigationItem, NavigationTree
from views.ui_plot_canvas import InteractivePlotCanvas
from views.ui_spreadsheet import ScientificSpreadsheet


class RibbonStyle(Enum):
    """Ribbon button styles."""

    LARGE_ICON = 1
    SMALL_ICON = 2
    TEXT_ONLY = 3
    ICON_TEXT = 4


class VectorIconEngine:
    """
    Vector Icon Engine using QPainter.

    Generates all application icons programmatically without external files.
    Each icon is drawn using primitive shapes and paths.
    """

    @staticmethod
    def create_icon(icon_type: str, size: int = 32) -> QPixmap:
        """
        Create a vector icon of the specified type.

        Icon Types:
            - 'new_file': New data matrix
            - 'open_file': Open CSV file
            - 'save_file': Save to file
            - 'transpose': Transpose matrix
            - 'pca': Principal Component Analysis
            - 'diversity': Biodiversity analysis
            - 'settings': Application settings
            - 'export': Export plot
            - 'undo': Undo operation
            - 'redo': Redo operation
        """
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Set default pen and brush
        pen = QPen(QColor("#2C3E50"))
        pen.setWidth(max(1, size // 16))
        brush = QBrush(QColor("#3498DB"))
        painter.setPen(pen)
        painter.setBrush(brush)

        margin = size // 8
        inner_size = size - 2 * margin

        if icon_type == "new_file":
            # Document with plus sign
            doc_rect = QRect(margin, margin, inner_size, inner_size)
            painter.drawRect(doc_rect)
            # Plus sign
            painter.setPen(QPen(QColor("#27AE60"), max(2, size // 12)))
            center = doc_rect.center()
            painter.drawLine(center.x() - inner_size // 6, center.y(), center.x() + inner_size // 6, center.y())
            painter.drawLine(center.x(), center.y() - inner_size // 6, center.x(), center.y() + inner_size // 6)

        elif icon_type == "open_file":
            # Folder with document
            folder_path = QPainterPath()
            folder_path.moveTo(margin, inner_size // 3 + margin)
            folder_path.lineTo(margin, inner_size - margin)
            folder_path.lineTo(inner_size - margin, inner_size - margin)
            folder_path.lineTo(inner_size - margin, inner_size // 3 + margin)
            folder_path.lineTo(inner_size // 2, inner_size // 3 + margin)
            folder_path.lineTo(inner_size // 3, margin)
            folder_path.closeSubpath()
            painter.drawPath(folder_path)

        elif icon_type == "save_file":
            # Floppy disk
            painter.drawRect(QRect(margin, margin + inner_size // 6, inner_size, inner_size - inner_size // 6))
            painter.setBrush(QBrush(QColor("#E4E7EB")))
            painter.drawRect(QRect(margin + inner_size // 4, margin, inner_size // 2, inner_size // 4))

        elif icon_type == "transpose":
            # Matrix transpose icon (diagonal arrow)
            painter.drawLine(margin, margin, inner_size + margin, inner_size + margin)
            painter.drawLine(margin, margin, margin, margin + inner_size // 4)
            painter.drawLine(margin, margin, margin + inner_size // 4, margin)
            painter.drawLine(
                inner_size + margin, inner_size + margin, inner_size + margin - inner_size // 4, inner_size + margin
            )
            painter.drawLine(
                inner_size + margin, inner_size + margin, inner_size + margin, inner_size + margin - inner_size // 4
            )

        elif icon_type == "pca":
            # 3D coordinate axes with ellipse (PC1, PC2, PC3)
            center_x = size // 2
            center_y = size // 2
            axis_length = inner_size // 2

            # X-axis (PC1)
            painter.setPen(QPen(QColor("#E74C3C"), max(2, size // 16)))
            painter.drawLine(center_x, center_y, center_x + axis_length, center_y)

            # Y-axis (PC2)
            painter.setPen(QPen(QColor("#27AE60"), max(2, size // 16)))
            painter.drawLine(center_x, center_y, center_x, center_y - axis_length)

            # Z-axis hint (PC3)
            painter.setPen(QPen(QColor("#3498DB"), max(2, size // 16)))
            painter.drawLine(center_x, center_y, center_x - axis_length // 2, center_y + axis_length // 2)

            # Ellipse representing variance
            painter.setPen(QPen(QColor("#F39C12"), max(1, size // 24)))
            ellipse_rect = QRect(
                center_x - axis_length // 3, center_y - axis_length // 3, axis_length * 2 // 3, axis_length * 2 // 3
            )
            painter.drawEllipse(ellipse_rect)

        elif icon_type == "diversity":
            # Biodiversity tree/branch icon
            center_x = size // 2
            base_y = size - margin

            # Main trunk
            painter.setPen(QPen(QColor("#27AE60"), max(2, size // 12)))
            painter.drawLine(center_x, base_y, center_x, margin + inner_size // 4)

            # Branches
            painter.drawLine(center_x, margin + inner_size // 2, margin + inner_size // 4, margin)
            painter.drawLine(center_x, margin + inner_size // 2, center_x, margin)
            painter.drawLine(center_x, margin + inner_size // 2, size - margin - inner_size // 4, margin)

        elif icon_type == "settings":
            # Gear/cog wheel
            painter.save()
            painter.translate(size // 2, size // 2)

            num_teeth = 8
            outer_radius = inner_size // 2
            inner_radius = inner_size // 3
            tooth_depth = inner_size // 8

            path = QPainterPath()
            for i in range(num_teeth * 2):
                angle = i * 3.14159 / num_teeth
                radius = outer_radius if i % 2 == 0 else outer_radius - tooth_depth
                x = radius * qCos(angle)
                y = radius * qSin(angle)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            path.closeSubpath()

            painter.drawPath(path)

            # Center hole
            painter.setBrush(QBrush(QColor("#E4E7EB")))
            painter.drawEllipse(QRect(-inner_radius // 2, -inner_radius // 2, inner_radius, inner_radius))
            painter.restore()

        elif icon_type == "export":
            # Arrow pointing outward from box
            box_rect = QRect(margin, margin + inner_size // 4, inner_size, inner_size * 2 // 3)
            painter.drawRect(box_rect)
            # Arrow
            painter.drawLine(box_rect.center().x(), box_rect.top(), box_rect.center().x(), margin)
            painter.drawLine(box_rect.center().x(), margin, margin, margin + inner_size // 4)
            painter.drawLine(box_rect.center().x(), margin, size - margin, margin + inner_size // 4)

        elif icon_type == "undo":
            # Curved arrow left
            center = pixmap.rect().center()
            painter.drawArc(
                QRect(
                    center.x() - inner_size // 3, center.y() - inner_size // 3, inner_size * 2 // 3, inner_size * 2 // 3
                ),
                180 * 16,
                180 * 16,
            )
            # Arrow head
            painter.drawLine(
                center.x() - inner_size // 3,
                center.y(),
                center.x() - inner_size // 3 - inner_size // 6,
                center.y() + inner_size // 6,
            )
            painter.drawLine(
                center.x() - inner_size // 3,
                center.y(),
                center.x() - inner_size // 3 - inner_size // 6,
                center.y() - inner_size // 6,
            )

        elif icon_type == "redo":
            # Curved arrow right
            center = pixmap.rect().center()
            painter.drawArc(
                QRect(
                    center.x() - inner_size // 3, center.y() - inner_size // 3, inner_size * 2 // 3, inner_size * 2 // 3
                ),
                0 * 16,
                180 * 16,
            )
            painter.drawLine(
                center.x() + inner_size // 3,
                center.y(),
                center.x() + inner_size // 3 + inner_size // 6,
                center.y() + inner_size // 6,
            )
            painter.drawLine(
                center.x() + inner_size // 3,
                center.y(),
                center.x() + inner_size // 3 + inner_size // 6,
                center.y() - inner_size // 6,
            )

        elif icon_type == "morphometrics":
            # Landmark points connected by lines
            points = [
                QPoint(margin + inner_size // 4, margin + inner_size // 4),
                QPoint(size - margin - inner_size // 4, margin + inner_size // 4),
                QPoint(size // 2, size - margin - inner_size // 4),
            ]
            painter.setPen(QPen(QColor("#9B59B6"), max(2, size // 16)))
            painter.drawPolygon(points)
            for pt in points:
                painter.setBrush(QBrush(QColor("#9B59B6")))
                painter.drawEllipse(pt, size // 10, size // 10)

        elif icon_type == "stratigraphy":
            # Layered sedimentary strata
            num_layers = 4
            layer_height = inner_size // num_layers
            colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB"]
            for i, color in enumerate(colors):
                painter.setBrush(QBrush(QColor(color)))
                painter.drawRect(QRect(margin, margin + i * layer_height, inner_size, layer_height - 1))

        elif icon_type == "nmds":
            # Stress plot icon
            painter.setPen(QPen(QColor("#16A085"), max(2, size // 16)))
            points_data = [
                QPoint(margin + inner_size // 5, size - margin - inner_size // 5),
                QPoint(size // 2, margin + inner_size // 3),
                QPoint(size - margin - inner_size // 5, size - margin - inner_size // 2),
            ]
            for pt in points_data:
                painter.setBrush(QBrush(QColor("#16A085")))
                painter.drawEllipse(pt, size // 12, size // 12)
            painter.drawPolyline(points_data)

        elif icon_type == "anosim":
            # Box plots comparison
            box_width = inner_size // 4
            box1_x = margin + inner_size // 6
            box2_x = size - margin - inner_size // 6 - box_width

            painter.setBrush(QBrush(QColor("#3498DB")))
            painter.drawRect(QRect(box1_x, margin + inner_size // 3, box_width, inner_size // 2))
            painter.drawLine(box1_x + box_width // 2, margin, box1_x + box_width // 2, margin + inner_size // 3)
            painter.drawLine(
                box1_x + box_width // 2,
                margin + inner_size // 3 + inner_size // 2,
                box1_x + box_width // 2,
                size - margin,
            )

            painter.setBrush(QBrush(QColor("#E74C3C")))
            painter.drawRect(QRect(box2_x, margin + inner_size // 5, box_width, inner_size // 3))
            painter.drawLine(box2_x + box_width // 2, margin, box2_x + box_width // 2, margin + inner_size // 5)
            painter.drawLine(
                box2_x + box_width // 2,
                margin + inner_size // 5 + inner_size // 3,
                box2_x + box_width // 2,
                size - margin,
            )

        else:
            # Default circle icon
            painter.drawEllipse(QRect(margin, margin, inner_size, inner_size))

        painter.end()
        return pixmap


def qCos(angle: float) -> float:
    """Compute cosine using math module."""
    import math

    return math.cos(angle)


def qSin(angle: float) -> float:
    """Compute sine using math module."""
    import math

    return math.sin(angle)


class RibbonButton(QPushButton):
    """
    Modern Ribbon Button with vector icon support.

    Features:
        - Vector icon rendering
        - Multiple display styles (icon only, text only, icon+text)
        - Hover/pressed state animations
        - Tooltip with keyboard shortcut
    """

    def __init__(
        self,
        icon_type: str = "",
        text: str = "",
        style: RibbonStyle = RibbonStyle.ICON_TEXT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._icon_type = icon_type
        self._style = style
        self._is_dark_theme = True

        # Set button properties
        self.setText(text)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Create icon
        if icon_type:
            icon_pixmap = VectorIconEngine.create_icon(icon_type, 24)
            icon = QIcon(icon_pixmap)
            self.setIcon(icon)

        # Apply stylesheet
        self._apply_stylesheet()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set theme and update stylesheet."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        """Apply modern themed stylesheet to button with smooth transitions."""
        self.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #E4E7EB;
                border-radius: 6px;
                padding: 10px 18px;
                color: #2C3E50;
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 12px;
                font-weight: 500;
                min-width: 70px;
                min-height: 32px;
            }
            QPushButton:hover {
                background-color: #F0F2F5;
                border: 1px solid #3498DB;
                color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #3498DB;
                color: white;
                border: 1px solid #2980B9;
            }
            QPushButton:disabled {
                background-color: #F8F9FA;
                color: #95A5A6;
                border: 1px solid #E4E7EB;
            }
            QPushButton[flat="true"] {
                background-color: transparent;
                border: none;
            }
            QPushButton[flat="true"]:hover {
                background-color: rgba(52, 152, 219, 0.08);
                border: none;
            }
        """)


class RibbonGroup(QWidget):
    """
    Ribbon Group containing related buttons.

    A group has a title and contains a horizontal layout of buttons.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._title = title
        self._buttons: list[RibbonButton] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(4)

        # Button container
        self._button_container = QWidget()
        self._button_layout = QHBoxLayout(self._button_container)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(4)
        self._button_layout.addStretch()
        self._layout.addWidget(self._button_container)

        # Title label
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("""
            QLabel {
                color: #3498DB;
                font-size: 10px;
                font-weight: 600;
                padding: 3px;
            }
        """)
        self._layout.addWidget(self._title_label)

    def addButton(
        self, icon_type: str = "", text: str = "", tooltip: str = "", style: RibbonStyle = RibbonStyle.ICON_TEXT
    ) -> RibbonButton:
        """Add a button to the ribbon group."""
        button = RibbonButton(icon_type, text, style, self)

        if tooltip:
            button.setToolTip(tooltip)

        self._buttons.append(button)
        self._button_layout.insertWidget(self._button_layout.count() - 1, button)

        return button


class RibbonTab(QWidget):
    """
    Ribbon Tab containing multiple ribbon groups.

    A tab represents a category of operations (e.g., 'Home', 'Analysis').
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._title = title
        self._groups: list[RibbonGroup] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(8)
        self._layout.addStretch()

    def addGroup(self, title: str) -> RibbonGroup:
        """Add a group to the ribbon tab."""
        group = RibbonGroup(title, self)
        self._groups.append(group)
        self._layout.insertWidget(self._layout.count() - 1, group)
        return group

    def title(self) -> str:
        """Get tab title."""
        return self._title


class RibbonBar(QWidget):
    """
    Modern Ribbon Bar for application toolbar.

    Features:
        - Multiple tabs with groups
        - Collapsible tabs
        - Quick access toolbar
        - Contextual tabs
    """

    tabChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._tabs: list[RibbonTab] = []
        self._current_tab_index = 0
        self._is_dark_theme = False

        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Tab bar
        self._tab_bar = QWidget()
        self._tab_bar_layout = QHBoxLayout(self._tab_bar)
        self._tab_bar_layout.setContentsMargins(4, 4, 4, 4)
        self._tab_bar_layout.setSpacing(0)
        self._tab_button_group: list[QPushButton] = []
        self._main_layout.addWidget(self._tab_bar)

        # Content area
        self._content_area = QWidget()
        self._content_area.setMinimumHeight(80)
        self._content_area.setMaximumHeight(120)
        self._content_layout = QHBoxLayout(self._content_area)
        self._content_layout.setContentsMargins(8, 4, 8, 4)
        self._content_layout.addStretch()
        self._main_layout.addWidget(self._content_area)

        # Separator line
        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.Shape.HLine)
        self._separator.setStyleSheet("""
            QFrame {
                background-color: #E4E7EB;
                max-height: 1px;
            }
        """)
        self._main_layout.addWidget(self._separator)

        self._apply_stylesheet()

    def addTab(self, title: str) -> RibbonTab:
        """Add a new tab to the ribbon."""
        tab = RibbonTab(title, self)
        self._tabs.append(tab)

        # Create tab button
        tab_button = QPushButton(title)
        tab_button.setCheckable(True)
        tab_button.setChecked(len(self._tabs) - 1 == self._current_tab_index)
        tab_button.clicked.connect(lambda checked=False, idx=len(self._tabs) - 1: self._on_tab_clicked(idx))
        self._tab_button_group.append(tab_button)
        self._tab_bar_layout.addWidget(tab_button)

        # Show first tab content
        if len(self._tabs) == 1:
            self._show_tab(0)

        return tab

    def _on_tab_clicked(self, index: int) -> None:
        """Handle tab button click."""
        for i, btn in enumerate(self._tab_button_group):
            btn.setChecked(i == index)

        self._current_tab_index = index
        self._show_tab(index)
        self.tabChanged.emit(index)

    def _show_tab(self, index: int) -> None:
        """Show the content of specified tab."""
        # Remove current content
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().hide()

        # Add new tab content
        if 0 <= index < len(self._tabs):
            tab = self._tabs[index]
            self._content_layout.insertWidget(0, tab)
            tab.show()

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: none;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                color: #2C3E50;
                padding: 8px 14px;
                font-size: 11px;
                font-weight: 600;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(52, 152, 219, 0.08);
                color: #3498DB;
            }
            QPushButton:checked {
                background-color: rgba(52, 152, 219, 0.12);
                border-bottom: 3px solid #3498DB;
                color: #3498DB;
            }
        """)
        self._separator.setStyleSheet("background-color: #E4E7EB; max-height: 1px;")


class StatusBarWidget(QStatusBar):
    """
    Custom status bar widget with data info and progress.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setContentsMargins(8, 2, 8, 2)
        self.setStyleSheet("""
            QStatusBar {
                background-color: #F8F9FA;
                border-top: 1px solid #E4E7EB;
                color: #2C3E50;
            }
        """)

        # Data info label (left side)
        self._info_label = QLabel(_("No data loaded"))
        self._info_label.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 11px;
            }
        """)
        self.addWidget(self._info_label)

        # Memory indicator (right side)
        self._memory_label = QLabel(_("Memory: 0 MB"))
        self._memory_label.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 10px;
            }
        """)
        self.addPermanentWidget(self._memory_label)

        # Progress bar (right side)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(150)
        self._progress_bar.setMaximumHeight(12)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E4E7EB;
                border-radius: 6px;
                text-align: center;
                background-color: #F8F9FA;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 5px;
            }
        """)
        self.addPermanentWidget(self._progress_bar)

    def setInfo(self, text: str) -> None:
        """Set info text."""
        self._info_label.setText(text)

    def setProgress(self, value: int, maximum: int = 100) -> None:
        """Show and update progress bar."""
        if maximum <= 0:
            # Indeterminate mode: show bouncing progress bar
            self._progress_bar.setVisible(True)
            self._progress_bar.setMaximum(0)
        elif value >= maximum:
            # Complete: hide progress bar
            self._progress_bar.setVisible(False)
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(0)
        else:
            self._progress_bar.setVisible(True)
            self._progress_bar.setMaximum(maximum)
            self._progress_bar.setValue(value)


class WorkspaceArea(QWidget):
    """
    Central workspace area containing spreadsheet and plot views.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Stacked widget for different views
        self._stack = QStackedWidget()
        self._layout.addWidget(self._stack)

        # Placeholder widget
        self._placeholder = QLabel(_("Load data to begin analysis"))
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 14px;
                background-color: #FFFFFF;
            }
        """)
        self._stack.addWidget(self._placeholder)

    def addWidget(self, widget: QWidget, name: str = "") -> int:
        """Add a widget to the workspace."""
        return self._stack.addWidget(widget)

    def setCurrentIndex(self, index: int) -> None:
        """Set current widget index."""
        self._stack.setCurrentIndex(index)

    def currentWidget(self) -> QWidget | None:
        """Get current widget."""
        return self._stack.currentWidget()

    def removeWidget(self, widget: QWidget) -> None:
        """Remove widget from workspace."""
        self._stack.removeWidget(widget)


class MainWindow(QMainWindow):
    """
    Main Application Window for PaleoAST.

    This is the central widget that orchestrates all UI components.
    It follows the MVC pattern and observes the StateManager for changes.

    Signals:
        dataLoaded: Emitted when new data is loaded
        analysisCompleted: Emitted when analysis finishes
        plotRequested: Emitted when plot is requested

    Mathematical Context:
        The main window serves as the orchestrator for all statistical operations.
        When user clicks "Run PCA", the following pipeline executes:

        1. User selects columns in spreadsheet (SpreadsheetView)
        2. Click triggers analysisRequested signal
        3. MainWindow slot receives signal
        4. StatisticsController.run_pca() is called
        5. PCA algorithm: $C = \\frac{1}{n-1} X^T X$
        6. Result is cached in StateManager
        7. InteractivePlotCanvas displays PC1 vs PC2 scores
        8. State change triggers Observer updates
    """

    # Signal definitions
    dataLoaded = pyqtSignal(object)  # DataMatrix
    analysisRequested = pyqtSignal(str, dict)  # analysis_type, parameters
    analysisCompleted = pyqtSignal(str, object)  # analysis_type, result
    plotRequested = pyqtSignal(str, object)  # plot_type, data
    navigationChanged = pyqtSignal(str)  # section_name

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger(f"{__name__}.MainWindow")
        self._logger.info("MainWindow created")

        # Initialize controllers
        self._data_controller = DataController()
        self._statistics_controller = StatisticsController()

        # Is dark theme
        self._is_dark_theme = False

        # Initialize state manager
        self._state = get_state_manager()
        
        # UI state management - register data-dependent elements
        self._data_actions = []
        self._data_buttons = []

        # Create widgets
        self._create_ui()

        # Setup connections
        self._setup_connections()

        # Load settings
        self._load_settings()

        # Status update timer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

    def _create_ui(self) -> None:
        """Create all UI components."""
        # Set window properties
        self.setWindowTitle(_("PaleoAST - Paleontological Advanced Statistical Toolkit"))
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Ribbon bar
        self._ribbon = RibbonBar()
        self._setup_ribbon()
        central_layout.addWidget(self._ribbon)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left navigation
        self._navigation = NavigationTree()
        self._navigation.setMaximumWidth(280)
        splitter.addWidget(self._navigation)

        # Workspace area
        self._workspace = WorkspaceArea()
        splitter.addWidget(self._workspace)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1000])

        central_layout.addWidget(splitter)

        # Status bar
        self._status_bar = StatusBarWidget()
        self.setStatusBar(self._status_bar)

        # Create menu bar
        self._create_menu_bar()

        # Create spreadsheet (initially hidden)
        self._spreadsheet = ScientificSpreadsheet()
        self._spreadsheet_index = self._workspace.addWidget(self._spreadsheet, _("Spreadsheet"))
        
        # Initialize UI state based on data availability
        self._update_ui_state()

    def _setup_ribbon(self) -> None:
        """Setup ribbon tabs and groups."""
        # Home tab
        home_tab = self._ribbon.addTab(_("Home"))

        # File operations group
        file_group = home_tab.addGroup(_("File"))
        self._btn_new = file_group.addButton("new_file", _("New"), _("Create new data matrix (Ctrl+N)"))
        self._btn_open = file_group.addButton("open_file", _("Open"), _("Open CSV file (Ctrl+O)"))
        self._btn_save = file_group.addButton("save_file", _("Save"), _("Save to file (Ctrl+S)"))

        # Edit operations group
        edit_group = home_tab.addGroup(_("Edit"))
        edit_group.addButton("undo", _("Undo"), _("Undo last action (Ctrl+Z)"))
        edit_group.addButton("redo", _("Redo"), _("Redo action (Ctrl+Y)"))
        edit_group.addButton("transpose", _("Transpose"), _("Transpose data matrix"))

        # Data transformations group
        transform_group = home_tab.addGroup(_("Transform"))
        self._btn_log_transform = transform_group.addButton("settings", _("Log"), _("Log transformation (base 10)"))
        self._btn_sqrt_transform = transform_group.addButton("settings", _("Sqrt"), _("Square root transformation"))
        self._btn_hellinger_transform = transform_group.addButton("settings", _("Hellinger"), _("Hellinger transformation"))
        self._btn_zscore_transform = transform_group.addButton("settings", _("Z-Score"), _("Z-score standardization"))
        self._btn_percent_transform = transform_group.addButton("settings", _("% Total"), _("Percentage standardization"))
        self._btn_wisconsin_transform = transform_group.addButton("settings", _("Wisconsin"), _("Wisconsin double standardization"))

        # View group
        view_group = home_tab.addGroup(_("View"))
        view_group.addButton("settings", _("Preferences"), _("Application settings"))

        # Analysis tab
        analysis_tab = self._ribbon.addTab(_("Analysis"))

        # Multivariate group
        multivar_group = analysis_tab.addGroup(_("Multivariate"))
        self._btn_pca = multivar_group.addButton("pca", "PCA", _("Principal Component Analysis"))
        self._btn_pcoa = multivar_group.addButton("pcoa", "PCoA", _("Principal Coordinate Analysis"))
        self._btn_nmds = multivar_group.addButton("nmds", "NMDS", _("Non-metric MDS"))
        self._btn_lda = multivar_group.addButton("chart", "LDA", _("Linear Discriminant Analysis"))

        # Univariate group
        univar_group = analysis_tab.addGroup(_("Univariate"))
        self._btn_univariate = univar_group.addButton("chart", _("Stats"), _("Univariate Statistics"))
        self._btn_simper = univar_group.addButton("chart", "SIMPER", _("SIMPER Analysis"))

        # Diversity group
        diversity_group = analysis_tab.addGroup(_("Diversity"))
        self._btn_diversity = diversity_group.addButton("diversity", _("Diversity"), _("Biodiversity indices"))
        self._btn_abundance = diversity_group.addButton("diversity", _("Models"), _("Abundance Models"))
        self._btn_she = diversity_group.addButton("diversity", "SHE", _("SHE Analysis"))

        # Group tests group
        tests_group = analysis_tab.addGroup(_("Tests"))
        self._btn_anosim = tests_group.addButton("anosim", "ANOSIM", _("Analysis of Similarities"))
        self._btn_clustering = tests_group.addButton("chart", _("Cluster"), _("Hierarchical Clustering"))

        # Morphometrics tab
        morpho_tab = self._ribbon.addTab(_("Morphometrics"))

        morpho_group = morpho_tab.addGroup(_("Landmarks"))
        morpho_group.addButton("morphometrics", "GPA", _("Generalized Procrustes Analysis"))

        efa_group = morpho_tab.addGroup(_("Outline"))
        self._btn_efa = efa_group.addButton("morphometrics", "EFA", _("Elliptic Fourier Analysis"))

        # Stratigraphy tab
        strat_tab = self._ribbon.addTab(_("Stratigraphy"))

        strat_group = strat_tab.addGroup(_("Time Series"))
        self._btn_spectral = strat_group.addButton("stratigraphy", _("Spectral"), _("Spectral Analysis"))
        self._btn_coniss = strat_group.addButton("stratigraphy", "CONISS", _("CONISS Zonation"))

        markov_group = strat_tab.addGroup(_("Facies"))
        self._btn_markov = markov_group.addButton("stratigraphy", _("Markov"), _("Markov Chain Analysis"))
        self._btn_directional = markov_group.addButton("stratigraphy", _("Rose"), _("Directional Statistics"))

    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Navigation signals
        self._navigation.itemClicked.connect(self._on_navigation_clicked)

        # File operation buttons (always enabled)
        self._btn_new.clicked.connect(self._on_new_file)
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_save.clicked.connect(self._on_save_file)

        # Transformation buttons
        self._btn_log_transform.clicked.connect(self._on_transform_log)
        self._btn_sqrt_transform.clicked.connect(self._on_transform_sqrt)
        self._btn_hellinger_transform.clicked.connect(self._on_transform_hellinger)
        self._btn_zscore_transform.clicked.connect(self._on_transform_zscore)
        self._btn_percent_transform.clicked.connect(self._on_transform_percent)
        self._btn_wisconsin_transform.clicked.connect(self._on_transform_wisconsin)
        for btn in [self._btn_log_transform, self._btn_sqrt_transform, self._btn_hellinger_transform,
                    self._btn_zscore_transform, self._btn_percent_transform, self._btn_wisconsin_transform]:
            self._register_data_button(btn)

        # Analysis buttons (require data)
        self._btn_pca.clicked.connect(self._on_run_pca)
        self._register_data_button(self._btn_pca)
        self._btn_pcoa.clicked.connect(self._on_run_pcoa)
        self._register_data_button(self._btn_pcoa)
        self._btn_nmds.clicked.connect(self._on_run_nmds)
        self._register_data_button(self._btn_nmds)
        self._btn_lda.clicked.connect(self._on_run_lda)
        self._register_data_button(self._btn_lda)
        self._btn_univariate.clicked.connect(self._on_run_univariate)
        self._register_data_button(self._btn_univariate)
        self._btn_simper.clicked.connect(self._on_run_simper)
        self._register_data_button(self._btn_simper)
        self._btn_diversity.clicked.connect(self._on_run_diversity)
        self._register_data_button(self._btn_diversity)
        self._btn_abundance.clicked.connect(self._on_run_abundance_models)
        self._register_data_button(self._btn_abundance)
        self._btn_she.clicked.connect(self._on_run_she)
        self._register_data_button(self._btn_she)
        self._btn_anosim.clicked.connect(self._on_run_anosim)
        self._register_data_button(self._btn_anosim)
        self._btn_clustering.clicked.connect(self._on_run_clustering)
        self._register_data_button(self._btn_clustering)
        self._btn_spectral.clicked.connect(self._on_run_spectral)
        self._register_data_button(self._btn_spectral)
        self._btn_coniss.clicked.connect(self._on_run_coniss)
        self._register_data_button(self._btn_coniss)
        self._btn_markov.clicked.connect(self._on_run_markov)
        self._register_data_button(self._btn_markov)
        self._btn_directional.clicked.connect(self._on_run_directional)
        self._register_data_button(self._btn_directional)
        self._btn_efa.clicked.connect(self._on_run_efa)
        self._register_data_button(self._btn_efa)

        # Monitor state changes
        self._last_has_data = False

    def _get_groups(self) -> list[int] | None:
        """Get group labels from row metadata, converted to integer indices.

        Returns:
            list[int]: Group indices for each row, or None if no groups are defined.
        """
        rm = self._state.row_metadata
        if rm is None:
            return None
        groups_dict = rm.get_groups()
        if not groups_dict:
            return None
        # Check if any group is actually set (not None)
        values = [groups_dict[i] for i in sorted(groups_dict.keys())]
        if all(v is None for v in values):
            return None

        # Convert string labels to integer indices
        # Get unique labels in order of first appearance
        unique_labels = []
        label_to_idx = {}
        for v in values:
            if v is None:
                label = "Ungrouped"
            else:
                label = v
            if label not in label_to_idx:
                label_to_idx[label] = len(unique_labels)
                unique_labels.append(label)

        # Return integer indices
        result = []
        for v in values:
            if v is None:
                label = "Ungrouped"
            else:
                label = v
            result.append(label_to_idx[label])

        return result

    def _update_ui_state(self) -> None:
        """Update UI element states based on data availability."""
        has_data = self._state.has_data
        
        # Update all registered data-dependent actions
        for action in self._data_actions:
            action.setEnabled(has_data)
        
        # Update all registered data-dependent buttons
        for button in self._data_buttons:
            button.setEnabled(has_data)
    
    def _register_data_action(self, action: QAction) -> None:
        """Register an action as data-dependent."""
        if action not in self._data_actions:
            self._data_actions.append(action)
            action.setEnabled(self._state.has_data)
    
    def _register_data_button(self, button) -> None:
        """Register a button as data-dependent."""
        if button not in self._data_buttons:
            self._data_buttons.append(button)
            button.setEnabled(self._state.has_data)

    def _create_menu_bar(self) -> None:
        """Create application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu(_("&File"))

        new_action = QAction(_("&New Matrix"), self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_file)
        file_menu.addAction(new_action)

        open_action = QAction(_("&Open..."), self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        save_action = QAction(_("&Save"), self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_file)
        file_menu.addAction(save_action)
        self._register_data_action(save_action)
        self._save_action = save_action

        save_as_action = QAction(_("Save &As..."), self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._on_save_file_as)
        file_menu.addAction(save_as_action)
        self._register_data_action(save_as_action)

        file_menu.addSeparator()

        import_action = QAction(_("&Import Data..."), self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self._on_import_data)
        file_menu.addAction(import_action)

        export_action = QAction(_("&Export..."), self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)
        self._register_data_action(export_action)
        self._export_action = export_action

        file_menu.addSeparator()

        exit_action = QAction(_("E&xit"), self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu(_("&Analysis"))

        pca_action = QAction(_("&PCA..."), self)
        pca_action.setShortcut(QKeySequence("Ctrl+1"))
        pca_action.triggered.connect(self._on_run_pca)
        analysis_menu.addAction(pca_action)
        self._register_data_action(pca_action)
        self._pca_action = pca_action

        pcoa_action = QAction(_("P&CoA..."), self)
        pcoa_action.setShortcut(QKeySequence("Ctrl+2"))
        pcoa_action.triggered.connect(self._on_run_pcoa)
        analysis_menu.addAction(pcoa_action)
        self._register_data_action(pcoa_action)
        self._pcoa_action = pcoa_action

        nmds_action = QAction(_("&NMDS..."), self)
        nmds_action.setShortcut(QKeySequence("Ctrl+3"))
        nmds_action.triggered.connect(self._on_run_nmds)
        analysis_menu.addAction(nmds_action)
        self._register_data_action(nmds_action)
        self._nmds_action = nmds_action

        analysis_menu.addSeparator()

        diversity_action = QAction(_("&Diversity..."), self)
        diversity_action.setShortcut(QKeySequence("Ctrl+D"))
        diversity_action.triggered.connect(self._on_run_diversity)
        analysis_menu.addAction(diversity_action)
        self._register_data_action(diversity_action)

        rarefaction_action = QAction(_("&Rarefaction..."), self)
        rarefaction_action.setShortcut(QKeySequence("Ctrl+R"))
        rarefaction_action.triggered.connect(self._on_run_rarefaction)
        analysis_menu.addAction(rarefaction_action)
        self._register_data_action(rarefaction_action)
        
        analysis_menu.addSeparator()

        anosim_action = QAction(_("&ANOSIM..."), self)
        anosim_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        anosim_action.triggered.connect(self._on_run_anosim)
        analysis_menu.addAction(anosim_action)
        self._register_data_action(anosim_action)

        permanova_action = QAction(_("&PERMANOVA..."), self)
        permanova_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        permanova_action.triggered.connect(self._on_run_permanova)
        analysis_menu.addAction(permanova_action)
        self._register_data_action(permanova_action)

        analysis_menu.addSeparator()

        simper_action = QAction(_("&SIMPER..."), self)
        simper_action.triggered.connect(self._on_run_simper)
        analysis_menu.addAction(simper_action)
        self._register_data_action(simper_action)

        lda_action = QAction(_("&LDA / CVA..."), self)
        lda_action.triggered.connect(self._on_run_lda)
        analysis_menu.addAction(lda_action)
        self._register_data_action(lda_action)

        univariate_action = QAction(_("&Univariate Statistics..."), self)
        univariate_action.triggered.connect(self._on_run_univariate)
        analysis_menu.addAction(univariate_action)
        self._register_data_action(univariate_action)

        clustering_action = QAction(_("&Hierarchical Clustering..."), self)
        clustering_action.triggered.connect(self._on_run_clustering)
        analysis_menu.addAction(clustering_action)
        self._register_data_action(clustering_action)

        analysis_menu.addSeparator()

        abundance_action = QAction(_("&Abundance Models..."), self)
        abundance_action.triggered.connect(self._on_run_abundance_models)
        analysis_menu.addAction(abundance_action)
        self._register_data_action(abundance_action)

        she_action = QAction(_("&SHE Analysis..."), self)
        she_action.triggered.connect(self._on_run_she)
        analysis_menu.addAction(she_action)
        self._register_data_action(she_action)

        analysis_menu.addSeparator()

        coniss_action = QAction(_("&CONISS Zonation..."), self)
        coniss_action.triggered.connect(self._on_run_coniss)
        analysis_menu.addAction(coniss_action)
        self._register_data_action(coniss_action)

        markov_action = QAction(_("&Markov Chain..."), self)
        markov_action.triggered.connect(self._on_run_markov)
        analysis_menu.addAction(markov_action)
        self._register_data_action(markov_action)

        directional_action = QAction(_("&Directional Statistics..."), self)
        directional_action.triggered.connect(self._on_run_directional)
        analysis_menu.addAction(directional_action)
        self._register_data_action(directional_action)

        efa_action = QAction(_("&Elliptic Fourier Analysis..."), self)
        efa_action.triggered.connect(self._on_run_efa)
        analysis_menu.addAction(efa_action)
        self._register_data_action(efa_action)

        spectral_action = QAction(_("&Spectral Analysis..."), self)
        spectral_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        spectral_action.triggered.connect(self._on_run_spectral)
        analysis_menu.addAction(spectral_action)
        self._register_data_action(spectral_action)

        # Language menu
        language_menu = menubar.addMenu(_("&Language"))

        self._lang_action_en = QAction("English", self)
        self._lang_action_en.setCheckable(True)
        self._lang_action_en.setChecked(get_translator().get_language() == "en")
        self._lang_action_en.triggered.connect(lambda: self._switch_language("en"))
        language_menu.addAction(self._lang_action_en)

        self._lang_action_zh = QAction("中文", self)
        self._lang_action_zh.setCheckable(True)
        self._lang_action_zh.setChecked(get_translator().get_language() == "zh")
        self._lang_action_zh.triggered.connect(lambda: self._switch_language("zh"))
        language_menu.addAction(self._lang_action_zh)

        # Help menu
        help_menu = menubar.addMenu(_("&Help"))

        about_action = QAction(_("&About PaleoAST"), self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        doc_action = QAction(_("&Documentation"), self)
        doc_action.setShortcut(QKeySequence("F1"))
        doc_action.triggered.connect(self._show_documentation)
        help_menu.addAction(doc_action)

    def _switch_language(self, lang: str) -> None:
        """Switch application language."""
        from PyQt6.QtCore import QSettings

        get_translator().set_language(lang)
        self._lang_action_en.setChecked(lang == "en")
        self._lang_action_zh.setChecked(lang == "zh")
        settings = QSettings("PaleoAST", "PaleoAST")
        settings.setValue("language", lang)

    def _on_navigation_clicked(self, item: NavigationItem) -> None:
        """
        Handle navigation item click.

        Routes leaf item clicks to the corresponding action handler.
        """
        section = item.section
        name = item.name
        self._logger.info(f"Navigation event: section='{section}', name='{name}'")
        self.navigationChanged.emit(section)

        # Action routing for leaf items (items without children)
        action_map = {
            _("Import Data"): self._on_import_data,
            _("Export Data"): self._on_export,
            "PCA": self._on_run_pca,
            "PCoA": self._on_run_pcoa,
            "NMDS": self._on_run_nmds,
            "LDA": self._on_run_lda,
            "ANOSIM": self._on_run_anosim,
            "PERMANOVA": self._on_run_permanova,
            "SIMPER": self._on_run_simper,
            _("Diversity"): self._on_run_diversity,
            _("Rarefaction"): self._on_run_rarefaction,
            _("Spectral Analysis"): self._on_run_spectral,
            _("Univariate"): self._on_run_univariate,
            _("Clustering"): self._on_run_clustering,
            _("Abundance Models"): self._on_run_abundance_models,
            "SHE": self._on_run_she,
            "CONISS": self._on_run_coniss,
            _("Markov"): self._on_run_markov,
            _("Directional"): self._on_run_directional,
            "EFA": self._on_run_efa,
        }

        handler = action_map.get(name)
        if handler is not None:
            handler()
            return

        # For category clicks or un-mapped items, switch to spreadsheet view
        self._workspace.setCurrentIndex(self._spreadsheet_index)

    def _on_new_file(self) -> None:
        """Create new empty data matrix."""
        # Show dialog to specify dimensions
        dialog = QDialog(self)
        dialog.setWindowTitle(_("New Data Matrix"))
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)

        # Sample count
        samples_layout = QHBoxLayout()
        samples_label = QLabel(_("Number of Samples:"))
        samples_spin = QSpinBox()
        samples_spin.setRange(2, 10000)
        samples_spin.setValue(30)
        samples_layout.addWidget(samples_label)
        samples_layout.addWidget(samples_spin)
        samples_layout.addStretch()
        layout.addLayout(samples_layout)

        # Variable count
        vars_layout = QHBoxLayout()
        vars_label = QLabel(_("Number of Variables:"))
        vars_spin = QSpinBox()
        vars_spin.setRange(2, 1000)
        vars_spin.setValue(8)
        vars_layout.addWidget(vars_label)
        vars_layout.addWidget(vars_spin)
        vars_layout.addStretch()
        layout.addLayout(vars_layout)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton(_("Create"))
        cancel_button = QPushButton(_("Cancel"))
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            n_samples = samples_spin.value()
            n_vars = vars_spin.value()

            # Create random data
            import numpy as np
            from models.data_matrix import DataMatrix

            data = np.random.randn(n_samples, n_vars) * 10 + 50
            row_labels = [f"Sample_{i+1}" for i in range(n_samples)]
            col_labels = [f"Var_{j+1}" for j in range(n_vars)]
            matrix = DataMatrix(data, row_labels=row_labels, col_labels=col_labels)

            # Update state manager
            self._state.set_data_matrix(matrix)

            # Load into spreadsheet
            self._spreadsheet.load_data(data, row_labels=row_labels, col_labels=col_labels)
            self._workspace.setCurrentIndex(self._spreadsheet_index)

            # Update UI state now that we have data
            self._update_ui_state()

            self._status_bar.setInfo(_("New matrix: {0} samples x {1} variables").format(n_samples, n_vars))

    def _on_open_file(self) -> None:
        """Open data file (CSV/TXT/Excel)."""
        filepath, _ext = QFileDialog.getOpenFileName(
            self, _("Open Data File"), "",
            _("Data Files (*.csv *.txt *.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx *.xls);;All Files (*)")
        )

        if filepath:
            try:
                self._logger.info(f"Opening file: '{filepath}'")
                ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
                if ext in ("xlsx", "xls"):
                    matrix = self._data_controller.load_excel(filepath, has_header=True, has_row_labels=True)
                else:
                    matrix = self._data_controller.load_csv(filepath, has_header=True, has_row_labels=True)

                # Update state manager
                self._state.set_data_matrix(matrix)

                self._spreadsheet.load_data(matrix.data, row_labels=matrix.row_labels, col_labels=matrix.col_labels)
                self._workspace.setCurrentIndex(self._spreadsheet_index)

                # Update UI state now that we have data
                self._update_ui_state()

                self._status_bar.setInfo(_("Loaded: {0}").format(filepath.split("/")[-1]))

            except Exception as e:
                self._logger.error(f"Failed to load file '{filepath}': {e}")
                QMessageBox.critical(self, _("Import Error"), _("Failed to load file:\n{0}").format(str(e)))

    def _on_save_file(self) -> bool:
        """Save current data. Returns True if save succeeded, False otherwise."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("No data to save."))
            return False

        filepath, _ext = QFileDialog.getSaveFileName(self, _("Save Data"), "", _("CSV Files (*.csv);;All Files (*)"))

        if filepath:
            try:
                self._data_controller.export_csv(filepath)
                self._status_bar.setInfo(_("Saved: {0}").format(filepath.split("/")[-1]))
                return True
            except Exception as e:
                QMessageBox.critical(self, _("Save Error"), str(e))
                return False
        return False

    def _on_save_file_as(self) -> None:
        """Save data with new name."""
        self._on_save_file()

    def _on_import_data(self) -> None:
        """Show import data dialog with conflict checking."""
        # Check if we need to confirm overwrite
        if self._state.has_data:
            reply = QMessageBox.question(
                self, _("Overwrite Data?"),
                _("You already have data loaded. Do you want to replace it?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        dialog = ImportDialog(self)
        dialog.dataImported.connect(self._on_data_imported)
        dialog.exec()

    def _on_data_imported(self, data, metadata) -> None:
        """Handle imported data with UI state updates."""
        from models.data_matrix import DataMatrix

        row_labels = metadata.get("row_labels")
        col_labels = metadata.get("col_labels")
        matrix = DataMatrix(data, row_labels=row_labels, col_labels=col_labels)

        # Update state manager
        self._state.set_data_matrix(matrix)

        self._spreadsheet.load_data(data, **metadata)
        self._workspace.setCurrentIndex(self._spreadsheet_index)

        # Update UI state now that we have data
        self._update_ui_state()

        # Show success message
        n_samples, n_vars = data.shape
        self._status_bar.setInfo(
            _("Data imported: {0} samples x {1} variables").format(n_samples, n_vars)
        )

    def _apply_transformation(self, transform_func, name: str) -> None:
        """Apply a transformation to the current data."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            data = self._state.data_matrix.data
            transformed = transform_func(data)

            # Update state
            from models.data_matrix import DataMatrix
            matrix = DataMatrix(
                transformed,
                row_labels=self._state.data_matrix.row_labels,
                col_labels=self._state.data_matrix.col_labels
            )
            self._state.set_data_matrix(matrix)

            # Update spreadsheet
            self._spreadsheet.load_data(
                transformed,
                row_labels=self._state.data_matrix.row_labels,
                col_labels=self._state.data_matrix.col_labels
            )

            self._status_bar.setInfo(_("{0} transformation applied").format(name))

        except Exception as e:
            QMessageBox.critical(self, _("Transformation Error"), str(e))

    def _on_transform_log(self) -> None:
        """Apply log10 transformation."""
        from utils.transformations import log_transform
        self._apply_transformation(log_transform, "Log")

    def _on_transform_sqrt(self) -> None:
        """Apply square root transformation."""
        from utils.transformations import sqrt_transform
        self._apply_transformation(sqrt_transform, "Sqrt")

    def _on_transform_hellinger(self) -> None:
        """Apply Hellinger transformation."""
        from utils.transformations import hellinger_transform
        self._apply_transformation(hellinger_transform, "Hellinger")

    def _on_transform_zscore(self) -> None:
        """Apply Z-score standardization."""
        from utils.transformations import zscore_standardize
        self._apply_transformation(zscore_standardize, "Z-Score")

    def _on_transform_percent(self) -> None:
        """Apply percentage standardization."""
        from utils.transformations import percent_standardize
        self._apply_transformation(percent_standardize, "% Total")

    def _on_transform_wisconsin(self) -> None:
        """Apply Wisconsin double standardization."""
        from utils.transformations import wisconsin_double_standardize
        self._apply_transformation(wisconsin_double_standardize, "Wisconsin")

    def _on_export(self) -> None:
        """Export analysis results and data."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        
        filepath, _ext = QFileDialog.getSaveFileName(
            self, _("Export Data"), "",
            _("CSV Files (*.csv);;All Files (*)")
        )
        
        if filepath:
            try:
                self._data_controller.export_csv(filepath)
                QMessageBox.information(
                    self, _("Export Successful"),
                    _("Data successfully exported to {0}").format(filepath.split("/")[-1])
                )
                self._logger.info(f"Data exported to {filepath}")
            except Exception as e:
                self._logger.error(f"Export failed: {e}")
                QMessageBox.critical(self, _("Export Error"), str(e))

    def _on_run_pca(self) -> None:
        """
        Run Principal Component Analysis.

        Mathematical Pipeline:
            Given data matrix X ∈ ℝ^(n×p):

            1. Center the data: Z = X - μ (subtract column means)
               where μ_j = (1/n) Σᵢ x_ij

            2. Compute covariance matrix:
               C = (1/(n-1)) Z^T Z ∈ ℝ^(p×p)

            3. Eigendecomposition:
               C v_j = λ_j v_j
               where λ_1 ≥ λ_2 ≥ ... ≥ λ_p are eigenvalues
                     v_j are corresponding eigenvectors

            4. Project onto principal components:
               PC_scores = Z @ V ∈ ℝ^(n×k)

               where V = [v_1, v_2, ..., v_k] is the loading matrix

            5. Variance explained by PC_j:
               r²_j = λ_j / Σλ_i × 100%
        """
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = PCADialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                self._status_bar.setProgress(0, 0)  # Show indeterminate

                result = self._statistics_controller.run_pca(
                    n_components=params["n_components"], method=params["method"]
                )

                self._status_bar.setProgress(100, 100)

                # Create and display plot
                plot = InteractivePlotCanvas()
                plot.plot_pca_scores(result)

                plot_index = self._workspace.addWidget(plot, _("PCA Plot"))
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(
                    _("PCA: {0} components, PC1+PC2 = {1:.1f}%").format(
                        result.n_components, result.explained_variance[0] + result.explained_variance[1]
                    )
                )

            except Exception as e:
                QMessageBox.critical(self, _("PCA Error"), str(e))

    def _on_run_pcoa(self) -> None:
        """Run Principal Coordinate Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = PCoADialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                result = self._statistics_controller.run_pcoa(
                    metric=params["metric"], n_components=params["n_components"]
                )

                plot = InteractivePlotCanvas()
                plot.plot_pcoa_scores(result)

                plot_index = self._workspace.addWidget(plot, _("PCoA Plot"))
                self._workspace.setCurrentIndex(plot_index)

            except Exception as e:
                QMessageBox.critical(self, _("PCoA Error"), str(e))

    def _on_run_nmds(self) -> None:
        """Run Non-metric MDS."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = NMDSOptionsDialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                result = self._statistics_controller.run_nmds(
                    metric=params["metric"], n_dimensions=params["n_dimensions"],
                    n_restarts=params["n_restarts"], max_iterations=params["max_iterations"],
                    tolerance=params["tolerance"]
                )

                plot = InteractivePlotCanvas()
                plot.plot_nmds(result)

                plot_index = self._workspace.addWidget(plot, _("NMDS Plot"))
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(_("NMDS: stress = {0:.4f}").format(result.stress))

            except Exception as e:
                QMessageBox.critical(self, _("NMDS Error"), str(e))

    def _on_run_diversity(self) -> None:
        """Run diversity analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = DiversityDialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                sample_name = params.get("sample_name", "").strip() or "Sample 1"
                result = self._statistics_controller.analyze_diversity(sample_name=sample_name)

                plot = InteractivePlotCanvas()
                plot.plot_diversity_summary(result)

                plot_index = self._workspace.addWidget(plot, _("Diversity Plot"))
                self._workspace.setCurrentIndex(plot_index)

            except Exception as e:
                QMessageBox.critical(self, _("Diversity Error"), str(e))

    def _on_run_rarefaction(self) -> None:
        """Run rarefaction analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = RarefactionDialog(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                sample_name = params.get("samples", ["Sample 1"])[0] if params.get("samples") else "Sample 1"
                result = self._statistics_controller.analyze_rarefaction(sample_name=sample_name)

                plot = InteractivePlotCanvas()
                plot.plot_rarefaction(result)

                plot_index = self._workspace.addWidget(plot, _("Rarefaction Plot"))
                self._workspace.setCurrentIndex(plot_index)

            except Exception as e:
                QMessageBox.critical(self, _("Rarefaction Error"), str(e))

    def _on_run_spectral(self) -> None:
        """Run spectral analysis (power spectrum and periodogram analysis)."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_spectral(
                data=self._state.data_matrix.data
            )
            plot = InteractivePlotCanvas()
            plot.plot_spectral(result)
            plot_index = self._workspace.addWidget(plot, _("Spectral Analysis"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("Spectral analysis completed"))
        except Exception as e:
            self._logger.error(f"Spectral analysis failed: {e}")
            QMessageBox.critical(self, _("Spectral Analysis Error"), str(e))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_anosim(self) -> None:
        """Run Analysis of Similarity (ANOSIM) test."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_anosim(
                data=self._state.data_matrix.data
            )
            plot = InteractivePlotCanvas()
            plot.plot_anosim_results(result)
            plot_index = self._workspace.addWidget(plot, _("ANOSIM Results"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("ANOSIM analysis completed"))
        except Exception as e:
            self._logger.error(f"ANOSIM analysis failed: {e}")
            QMessageBox.critical(self, _("ANOSIM Error"), str(e))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_permanova(self) -> None:
        """Run Permutational Multivariate Analysis of Variance (PERMANOVA) test."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_permanova(
                data=self._state.data_matrix.data
            )
            plot = InteractivePlotCanvas()
            plot.plot_permanova_results(result)
            plot_index = self._workspace.addWidget(plot, _("PERMANOVA Results"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("PERMANOVA analysis completed"))
        except Exception as e:
            self._logger.error(f"PERMANOVA analysis failed: {e}")
            QMessageBox.critical(self, _("PERMANOVA Error"), str(e))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_simper(self) -> None:
        """Run SIMPER analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = SimperDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_simper(
                    data=self._state.data_matrix.data,
                    groups=self._get_groups(),
                    metric=params.get("metric", "bray_curtis"),
                )
                plot = InteractivePlotCanvas()
                plot.plot_simper_results(result)
                plot_index = self._workspace.addWidget(plot, "SIMPER")
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo("SIMPER analysis completed")
            except Exception as e:
                QMessageBox.critical(self, "SIMPER Error", str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_univariate(self) -> None:
        """Run univariate statistics."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = UnivariateDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                test_type = params.get("test_type", 0)
                data = self._state.data_matrix.data
                col_names = self._state.data_matrix.col_labels

                if test_type == 0:  # Summary
                    result = self._statistics_controller.analyze_univariate_summary(data, col_names)
                    msg = result.summary()
                    QMessageBox.information(self, _("Summary Statistics"), msg)
                elif test_type == 1:  # Normality
                    results = self._statistics_controller.analyze_normality(data, col_names)
                    lines = [f"{col_names[i] if i < len(col_names) else f'Var{i}'}: W={r.shapiro_stat:.4f}, p={r.shapiro_p:.4f} {'*' if r.is_normal_shapiro else 'ns'}"
                             for i, r in enumerate(results)]
                    QMessageBox.information(self, _("Normality Test"), "\n".join(lines))
                elif test_type == 2:  # t-test
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_t_test(data, groups=groups)
                    lines = [f"{col_names[i] if i < len(col_names) else f'Var{i}'}: t={r.statistic:.4f}, p={r.p_value:.4f}"
                             for i, r in enumerate(results)]
                    QMessageBox.information(self, _("t-test Results"), "\n".join(lines))
                elif test_type == 3:  # ANOVA
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_anova(data, groups=groups)
                    lines = [f"{col_names[i] if i < len(col_names) else f'Var{i}'}: F={r.f_statistic:.4f}, p={r.p_value:.4f}"
                             for i, r in enumerate(results)]
                    QMessageBox.information(self, _("ANOVA Results"), "\n".join(lines))
                elif test_type == 4:  # Kruskal-Wallis
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_kruskal_wallis(data, groups=groups)
                    lines = [f"{col_names[i] if i < len(col_names) else f'Var{i}'}: H={r.statistic:.4f}, p={r.p_value:.4f}"
                             for i, r in enumerate(results)]
                    QMessageBox.information(self, _("Kruskal-Wallis Results"), "\n".join(lines))

                self._status_bar.setInfo(_("Univariate analysis completed"))
            except Exception as e:
                QMessageBox.critical(self, _("Univariate Error"), str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_lda(self) -> None:
        """Run Linear Discriminant Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        groups = self._get_groups()
        if groups is None:
            QMessageBox.warning(
                self, _("No Groups"),
                _("LDA requires group assignments. Please set row groups first via the spreadsheet metadata.")
            )
            return

        dialog = LDADialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                self._logger.info(f"Running LDA with {len(set(groups))} groups, n_components={params.get('n_components')}")
                result = self._statistics_controller.analyze_lda(
                    data=self._state.data_matrix.data,
                    groups=groups,
                    n_components=params.get("n_components"),
                )
                plot = InteractivePlotCanvas()
                plot.plot_lda_scores(result)
                plot_index = self._workspace.addWidget(plot, "LDA")
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(_("LDA analysis completed"))
                self._logger.info(f"LDA completed: accuracy={result.accuracy:.4f}, {result.n_classes} classes")
            except Exception as e:
                self._logger.error(f"LDA analysis failed: {e}")
                QMessageBox.critical(self, _("LDA Error"), str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_clustering(self) -> None:
        """Run hierarchical clustering."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = ClusteringDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_clustering(
                    data=self._state.data_matrix.data,
                    n_clusters=params.get("n_clusters", 3),
                    method=params.get("method", "ward"),
                    metric=params.get("metric", "euclidean"),
                )
                plot = InteractivePlotCanvas()
                plot.plot_dendrogram(result, labels=self._state.data_matrix.row_labels)
                plot_index = self._workspace.addWidget(plot, _("Clustering"))
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(
                    _("Clustering: {0} clusters, cophenetic r={1:.3f}").format(
                        result.n_clusters, result.cophenetic_correlation
                    )
                )
            except Exception as e:
                QMessageBox.critical(self, _("Clustering Error"), str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_abundance_models(self) -> None:
        """Fit species-abundance distribution models."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            results = self._statistics_controller.analyze_abundance_models()
            plot = InteractivePlotCanvas()
            plot.plot_abundance_models(results)
            plot_index = self._workspace.addWidget(plot, _("Abundance Models"))
            self._workspace.setCurrentIndex(plot_index)
            msg_lines = [f"{fit.model_name}: R²={fit.r_squared:.4f}, AIC={fit.aic:.2f}" for fit in results.values()]
            self._status_bar.setInfo(_("Abundance models fitted"))
        except Exception as e:
            QMessageBox.critical(self, _("Abundance Models Error"), str(e))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_she(self) -> None:
        """Run SHE analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_she()
            plot = InteractivePlotCanvas()
            plot.plot_she_curve(result)
            plot_index = self._workspace.addWidget(plot, "SHE")
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("SHE analysis completed"))
        except Exception as e:
            QMessageBox.critical(self, "SHE Error", str(e))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_coniss(self) -> None:
        """Run CONISS zonation."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = CONISSDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_coniss(
                    data=self._state.data_matrix.data,
                    n_zones=params.get("n_zones", 4),
                )
                QMessageBox.information(
                    self, "CONISS",
                    result.summary()
                )
                self._status_bar.setInfo(_("CONISS: {0} zones").format(result.n_zones))
            except Exception as e:
                QMessageBox.critical(self, "CONISS Error", str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_markov(self) -> None:
        """Run Markov chain analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = MarkovDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_markov()
                QMessageBox.information(
                    self, _("Markov Chain Analysis"),
                    result.summary()
                )
                self._status_bar.setInfo(_("Markov analysis completed"))
            except Exception as e:
                QMessageBox.critical(self, _("Markov Error"), str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_directional(self) -> None:
        """Run directional statistics."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = DirectionalDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_directional()
                bin_edges, counts = self._statistics_controller.bin_rose_diagram(
                    n_bins=params.get("n_bins", 12)
                )
                plot = InteractivePlotCanvas()
                plot.plot_rose_diagram(bin_edges, counts, result.mean_direction_deg)
                plot_index = self._workspace.addWidget(plot, _("Rose Diagram"))
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(
                    _("Directional: mean={0:.1f}°, Rayleigh p={1:.4f}").format(
                        result.mean_direction_deg, result.rayleigh_p
                    )
                )
            except Exception as e:
                QMessageBox.critical(self, _("Directional Error"), str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_efa(self) -> None:
        """Run Elliptic Fourier Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = EFADialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_efa(
                    contour=self._state.data_matrix.data[:, :2],
                    n_harmonics=params.get("n_harmonics", 10),
                    n_points=params.get("n_points", 200),
                )
                plot = InteractivePlotCanvas()
                plot.plot_efa_contours(result.original, result.reconstructed,
                                       f"EFA ({result.n_harmonics} harmonics)")
                plot_index = self._workspace.addWidget(plot, "EFA")
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(
                    _("EFA: {0} harmonics, {1} points").format(result.n_harmonics, result.n_points)
                )
            except Exception as e:
                QMessageBox.critical(self, "EFA Error", str(e))
            finally:
                self._status_bar.setProgress(100, 100)

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            _("About PaleoAST"),
            """
            <h2>PaleoAST</h2>
            <p>{}</p>
            <p>{}</p>
            <p>Copyright © 2024 PaleoAST Development Team</p>
            <hr>
            <p>{}</p>
            <ul>
                <li>{}</li>
                <li>{}</li>
                <li>{}</li>
                <li>{}</li>
                <li>{}</li>
                <li>{}</li>
                <li>{}</li>
            </ul>
            """.format(
                _("Paleontological Advanced Statistical Toolkit"),
                _("Version 1.0.0"),
                _("A comprehensive tool for paleontological data analysis including:"),
                _("Multivariate Statistics (PCA, PCoA, NMDS, LDA)"),
                _("Group Comparison Tests (ANOSIM, PERMANOVA, SIMPER)"),
                _("Univariate Statistics (ANOVA, t-test, Kruskal-Wallis)"),
                _("Ecology (Diversity, Abundance Models, SHE, Clustering)"),
                _("Stratigraphy (CONISS, Markov, Directional/Rose)"),
                _("Morphometrics (GPA, EFA, Eigenshape)"),
                _("Data Transformations (Hellinger, Box-Cox, KNN Imputation)"),
            ),
        )

    def _show_documentation(self) -> None:
        """Show documentation."""
        QMessageBox.information(
            self, _("Documentation"), _("See ARCHITECTURE_BLUEPRINT.md for detailed documentation.")
        )

    def _update_status(self) -> None:
        """Update status bar information and monitor data state changes."""
        # Check if data state changed
        current_has_data = self._state.has_data
        if current_has_data != self._last_has_data:
            self._last_has_data = current_has_data
            self._update_ui_state()
        
        if self._state.has_data:
            matrix = self._state.data_matrix
            info = _("Data: {0} samples x {1} variables").format(matrix.n_samples, matrix.n_variables)

            if self._state.is_modified:
                info += _(", modified")

            self._status_bar.setInfo(info)
        else:
            self._status_bar.setInfo(_("No data loaded"))

    def _load_settings(self) -> None:
        """Load application settings."""
        settings = QSettings("PaleoAST", "PaleoAST")

        # Restore window geometry
        geometry = settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Restore state
        state = settings.value("window/state")
        if state:
            self.restoreState(state)

    def _save_settings(self) -> None:
        """Save application settings."""
        settings = QSettings("PaleoAST", "PaleoAST")

        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Check for unsaved changes
        if self._state.is_modified:
            reply = QMessageBox.question(
                self,
                _("Unsaved Changes"),
                _("You have unsaved changes. Do you want to save before closing?"),
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Save:
                # Try to save; if user cancels or save fails, don't close
                if not self._on_save_file():
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Discard:
                pass
            else:
                event.ignore()
                return

        # Save settings
        self._save_settings()

        # Stop status timer
        self._status_timer.stop()

        event.accept()


def main() -> None:
    """Main entry point for GUI application."""

    app = QApplication(sys.argv)
    app.setApplicationName("PaleoAST")
    app.setOrganizationName("PaleoAST")
    app.setOrganizationDomain("paleoast.org")

    # Set application style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
