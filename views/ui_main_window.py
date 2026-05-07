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

import sys
import traceback
from typing import Optional, Dict, Any, List
from enum import Enum

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QDockWidget, QTreeView, QLabel, QStatusBar, QMessageBox,
    QMenuBar, QMenu, QToolBar, QPushButton, QStackedWidget,
    QFrame, QScrollArea, QGroupBox, QCheckBox, QSpinBox,
    QComboBox, QProgressBar, QApplication, QFileDialog,
    QDialog, QSizePolicy, QStyle
)
from PyQt6.QtCore import (
    Qt, QSize, QTimer, QPoint, QRect, pyqtSignal, pyqtSlot,
    QThread, QSettings, QObject, QAbstractItemModel,
    QAbstractItemView, QItemSelectionModel
)
from PyQt6.QtGui import (
    QAction, QIcon, QPainter, QPen, QBrush, QColor, QFont,
    QPixmap, QImage, QCursor, QKeySequence, QPalette,
    QLinearGradient, QRadialGradient, QConicalGradient
)

from models.state_manager import get_state_manager
from controllers.data_controller import DataController
from controllers.statistics_controller import StatisticsController
from views.ui_spreadsheet import ScientificSpreadsheet
from views.ui_navigation import NavigationTree, NavigationItem
from views.ui_dialogs import (
    PCADialog, PCoADialog, NMDSOptionsDialog,
    DiversityDialog, RarefactionDialog, ImportDialog
)
from views.ui_plot_canvas import InteractivePlotCanvas


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
        
        if icon_type == 'new_file':
            # Document with plus sign
            doc_rect = QRect(margin, margin, inner_size, inner_size)
            painter.drawRect(doc_rect)
            # Plus sign
            painter.setPen(QPen(QColor("#27AE60"), max(2, size // 12)))
            center = doc_rect.center()
            painter.drawLine(
                center.x() - inner_size // 6, center.y(),
                center.x() + inner_size // 6, center.y()
            )
            painter.drawLine(
                center.x(), center.y() - inner_size // 6,
                center.x(), center.y() + inner_size // 6
            )
            
        elif icon_type == 'open_file':
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
            
        elif icon_type == 'save_file':
            # Floppy disk
            painter.drawRect(QRect(margin, margin + inner_size // 6, 
                                   inner_size, inner_size - inner_size // 6))
            painter.setBrush(QBrush(QColor("#2C3E50")))
            painter.drawRect(QRect(margin + inner_size // 4, margin,
                                   inner_size // 2, inner_size // 4))
            
        elif icon_type == 'transpose':
            # Matrix transpose icon (diagonal arrow)
            painter.drawLine(margin, margin, inner_size + margin, inner_size + margin)
            painter.drawLine(margin, margin, margin, margin + inner_size // 4)
            painter.drawLine(margin, margin, margin + inner_size // 4, margin)
            painter.drawLine(inner_size + margin, inner_size + margin, 
                           inner_size + margin - inner_size // 4, inner_size + margin)
            painter.drawLine(inner_size + margin, inner_size + margin,
                           inner_size + margin, inner_size + margin - inner_size // 4)
            
        elif icon_type == 'pca':
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
            painter.drawLine(center_x, center_y, center_x - axis_length // 2, 
                           center_y + axis_length // 2)
            
            # Ellipse representing variance
            painter.setPen(QPen(QColor("#F39C12"), max(1, size // 24)))
            ellipse_rect = QRect(center_x - axis_length // 3, center_y - axis_length // 3,
                                axis_length * 2 // 3, axis_length * 2 // 3)
            painter.drawEllipse(ellipse_rect)
            
        elif icon_type == 'diversity':
            # Biodiversity tree/branch icon
            center_x = size // 2
            base_y = size - margin
            
            # Main trunk
            painter.setPen(QPen(QColor("#27AE60"), max(2, size // 12)))
            painter.drawLine(center_x, base_y, center_x, margin + inner_size // 4)
            
            # Branches
            painter.drawLine(center_x, margin + inner_size // 2,
                           margin + inner_size // 4, margin)
            painter.drawLine(center_x, margin + inner_size // 2,
                           center_x, margin)
            painter.drawLine(center_x, margin + inner_size // 2,
                           size - margin - inner_size // 4, margin)
            
        elif icon_type == 'settings':
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
            painter.setBrush(QBrush(QColor("#2C3E50")))
            painter.drawEllipse(QRect(-inner_radius // 2, -inner_radius // 2,
                                     inner_radius, inner_radius))
            painter.restore()
            
        elif icon_type == 'export':
            # Arrow pointing outward from box
            box_rect = QRect(margin, margin + inner_size // 4,
                           inner_size, inner_size * 2 // 3)
            painter.drawRect(box_rect)
            # Arrow
            painter.drawLine(box_rect.center().x(), box_rect.top(),
                           box_rect.center().x(), margin)
            painter.drawLine(box_rect.center().x(), margin,
                           margin, margin + inner_size // 4)
            painter.drawLine(box_rect.center().x(), margin,
                           size - margin, margin + inner_size // 4)
            
        elif icon_type == 'undo':
            # Curved arrow left
            center = pixmap.rect().center()
            painter.drawArc(QRect(center.x() - inner_size // 3, center.y() - inner_size // 3,
                                 inner_size * 2 // 3, inner_size * 2 // 3),
                          180 * 16, 180 * 16)
            # Arrow head
            painter.drawLine(center.x() - inner_size // 3, center.y(),
                           center.x() - inner_size // 3 - inner_size // 6, 
                           center.y() + inner_size // 6)
            painter.drawLine(center.x() - inner_size // 3, center.y(),
                           center.x() - inner_size // 3 - inner_size // 6,
                           center.y() - inner_size // 6)
            
        elif icon_type == 'redo':
            # Curved arrow right
            center = pixmap.rect().center()
            painter.drawArc(QRect(center.x() - inner_size // 3, center.y() - inner_size // 3,
                                 inner_size * 2 // 3, inner_size * 2 // 3),
                          0 * 16, 180 * 16)
            painter.drawLine(center.x() + inner_size // 3, center.y(),
                           center.x() + inner_size // 3 + inner_size // 6,
                           center.y() + inner_size // 6)
            painter.drawLine(center.x() + inner_size // 3, center.y(),
                           center.x() + inner_size // 3 + inner_size // 6,
                           center.y() - inner_size // 6)
            
        elif icon_type == 'morphometrics':
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
                
        elif icon_type == 'stratigraphy':
            # Layered sedimentary strata
            num_layers = 4
            layer_height = inner_size // num_layers
            colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB"]
            for i, color in enumerate(colors):
                painter.setBrush(QBrush(QColor(color)))
                painter.drawRect(QRect(margin, margin + i * layer_height,
                                      inner_size, layer_height - 1))
                
        elif icon_type == 'nmds':
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
            
        elif icon_type == 'anosim':
            # Box plots comparison
            box_width = inner_size // 4
            box1_x = margin + inner_size // 6
            box2_x = size - margin - inner_size // 6 - box_width
            
            painter.setBrush(QBrush(QColor("#3498DB")))
            painter.drawRect(QRect(box1_x, margin + inner_size // 3,
                                  box_width, inner_size // 2))
            painter.drawLine(box1_x + box_width // 2, margin,
                            box1_x + box_width // 2, margin + inner_size // 3)
            painter.drawLine(box1_x + box_width // 2, margin + inner_size // 3 + inner_size // 2,
                            box1_x + box_width // 2, size - margin)
            
            painter.setBrush(QBrush(QColor("#E74C3C")))
            painter.drawRect(QRect(box2_x, margin + inner_size // 5,
                                  box_width, inner_size // 3))
            painter.drawLine(box2_x + box_width // 2, margin,
                            box2_x + box_width // 2, margin + inner_size // 5)
            painter.drawLine(box2_x + box_width // 2, margin + inner_size // 5 + inner_size // 3,
                            box2_x + box_width // 2, size - margin)
            
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
        parent: Optional[QWidget] = None
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
        """Apply themed stylesheet to button."""
        if self._is_dark_theme:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #34495E;
                    border: 1px solid #2C3E50;
                    border-radius: 4px;
                    padding: 8px 16px;
                    color: #ECF0F1;
                    font-family: Arial, sans-serif;
                    font-size: 11px;
                    min-width: 70px;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #3D566E;
                    border: 1px solid #3498DB;
                }
                QPushButton:pressed {
                    background-color: #2C3E50;
                    border: 1px solid #2980B9;
                }
                QPushButton:disabled {
                    background-color: #2C3E50;
                    color: #7F8C8D;
                    border: 1px solid #34495E;
                }
                QPushButton:flat {
                    background-color: transparent;
                    border: none;
                }
                QPushButton:flat:hover {
                    background-color: rgba(52, 152, 219, 0.2);
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #ECF0F1;
                    border: 1px solid #BDC3C7;
                    border-radius: 4px;
                    padding: 8px 16px;
                    color: #2C3E50;
                    font-family: Arial, sans-serif;
                    font-size: 11px;
                    min-width: 70px;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #3498DB;
                    color: white;
                    border: 1px solid #2980B9;
                }
                QPushButton:pressed {
                    background-color: #2980B9;
                }
            """)


class RibbonGroup(QWidget):
    """
    Ribbon Group containing related buttons.
    
    A group has a title and contains a horizontal layout of buttons.
    """
    
    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        
        self._title = title
        self._buttons: List[RibbonButton] = []
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        
        # Title label
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("""
            QLabel {
                color: #3498DB;
                font-size: 10px;
                font-weight: bold;
                padding: 2px;
            }
        """)
        self._layout.addWidget(self._title_label)
        
        # Button container
        self._button_container = QWidget()
        self._button_layout = QHBoxLayout(self._button_container)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(4)
        self._button_layout.addStretch()
        self._layout.addWidget(self._button_container)
    
    def addButton(
        self,
        icon_type: str = "",
        text: str = "",
        tooltip: str = "",
        style: RibbonStyle = RibbonStyle.ICON_TEXT
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
    
    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        
        self._title = title
        self._groups: List[RibbonGroup] = []
        
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
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self._tabs: List[RibbonTab] = []
        self._current_tab_index = 0
        self._is_dark_theme = True
        
        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        
        # Tab bar
        self._tab_bar = QWidget()
        self._tab_bar_layout = QHBoxLayout(self._tab_bar)
        self._tab_bar_layout.setContentsMargins(4, 4, 4, 4)
        self._tab_bar_layout.setSpacing(0)
        self._tab_button_group: List[QPushButton] = []
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
                background-color: #2C3E50;
                max-height: 2px;
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
        tab_button.clicked.connect(lambda: self._on_tab_clicked(len(self._tabs) - 1))
        self._tab_button_group.append(tab_button)
        self._tab_bar_layout.insertWidget(self._tab_bar_layout.count() - 1, tab_button)
        
        # Connect to tab change signal
        tab_button.clicked.connect(lambda checked, idx=len(self._tabs) - 1: 
                                  self.tabChanged.emit(idx) if checked else None)
        
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
        if self._is_dark_theme:
            self.setStyleSheet("""
                QWidget {
                    background-color: #2C3E50;
                }
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #ECF0F1;
                    padding: 8px 16px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(52, 152, 219, 0.3);
                }
                QPushButton:checked {
                    background-color: #34495E;
                    border-bottom: 2px solid #3498DB;
                    color: #3498DB;
                }
            """)
            self._separator.setStyleSheet("background-color: #34495E; max-height: 1px;")
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #ECF0F1;
                }
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #2C3E50;
                    padding: 8px 16px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(52, 152, 219, 0.2);
                }
                QPushButton:checked {
                    background-color: white;
                    border-bottom: 2px solid #3498DB;
                    color: #3498DB;
                }
            """)
            self._separator.setStyleSheet("background-color: #BDC3C7; max-height: 1px;")


class StatusBarWidget(QWidget):
    """
    Custom status bar widget with data info and progress.
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 2, 8, 2)
        
        # Data info label
        self._info_label = QLabel("No data loaded")
        self._info_label.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 11px;
            }
        """)
        self._layout.addWidget(self._info_label)
        
        self._layout.addStretch()
        
        # Memory indicator
        self._memory_label = QLabel("Memory: 0 MB")
        self._memory_label.setStyleSheet("""
            QLabel {
                color: #7F8C8D;
                font-size: 10px;
            }
        """)
        self._layout.addWidget(self._memory_label)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(150)
        self._progress_bar.setMaximumHeight(12)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #34495E;
                border-radius: 4px;
                text-align: center;
                background-color: #2C3E50;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 3px;
            }
        """)
        self._layout.addWidget(self._progress_bar)
    
    def setInfo(self, text: str) -> None:
        """Set info text."""
        self._info_label.setText(text)
    
    def setProgress(self, value: int, maximum: int = 100) -> None:
        """Show and update progress bar."""
        if maximum <= 0:
            self._progress_bar.setVisible(False)
        else:
            self._progress_bar.setVisible(True)
            self._progress_bar.setMaximum(maximum)
            self._progress_bar.setValue(value)


class WorkspaceArea(QWidget):
    """
    Central workspace area containing spreadsheet and plot views.
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        
        # Stacked widget for different views
        self._stack = QStackedWidget()
        self._layout.addWidget(self._stack)
        
        # Placeholder widget
        self._placeholder = QLabel("Load data to begin analysis")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("""
            QLabel {
                color: #7F8C8D;
                font-size: 14px;
                background-color: #1A1A2E;
            }
        """)
        self._stack.addWidget(self._placeholder)
    
    def addWidget(self, widget: QWidget, name: str = "") -> int:
        """Add a widget to the workspace."""
        return self._stack.addWidget(widget)
    
    def setCurrentIndex(self, index: int) -> None:
        """Set current widget index."""
        self._stack.setCurrentIndex(index)
    
    def currentWidget(self) -> Optional[QWidget]:
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
        
        # Initialize controllers
        self._data_controller = DataController()
        self._statistics_controller = StatisticsController()
        
        # Initialize state manager
        self._state = get_state_manager()
        
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
        self.setWindowTitle("PaleoAST - Paleontological Advanced Statistical Toolkit")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1A1A2E;
            }
            QWidget {
                font-family: Arial, sans-serif;
            }
            QMenuBar {
                background-color: #2C3E50;
                color: #ECF0F1;
                border-bottom: 1px solid #34495E;
            }
            QMenuBar::item {
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #34495E;
            }
            QMenu {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
            }
            QMenu::item:selected {
                background-color: #3498DB;
            }
            QStatusBar {
                background-color: #1A1A2E;
                color: #95A5A6;
                border-top: 1px solid #2C3E50;
            }
           QDockWidget {
                color: #ECF0F1;
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }
            QDockWidget::title {
                background-color: #2C3E50;
                padding: 4px;
            }
        """)
        
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
        
        # Left navigation dock
        self._nav_dock = QDockWidget("Navigation", self)
        self._nav_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                        Qt.DockWidgetArea.RightDockWidgetArea)
        self._nav_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self._navigation = NavigationTree()
        self._nav_dock.setWidget(self._navigation)
        self._nav_dock.setMaximumWidth(280)
        splitter.addWidget(self._nav_dock)
        
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
        self._spreadsheet_index = self._workspace.addWidget(
            self._spreadsheet, "Spreadsheet"
        )
    
    def _setup_ribbon(self) -> None:
        """Setup ribbon tabs and groups."""
        # Home tab
        home_tab = self._ribbon.addTab("Home")
        
        # File operations group
        file_group = home_tab.addGroup("File")
        file_group.addButton("new_file", "New", "Create new data matrix (Ctrl+N)")
        file_group.addButton("open_file", "Open", "Open CSV file (Ctrl+O)")
        file_group.addButton("save_file", "Save", "Save to file (Ctrl+S)")
        
        # Edit operations group
        edit_group = home_tab.addGroup("Edit")
        edit_group.addButton("undo", "Undo", "Undo last action (Ctrl+Z)")
        edit_group.addButton("redo", "Redo", "Redo action (Ctrl+Y)")
        edit_group.addButton("transpose", "Transpose", "Transpose data matrix")
        
        # View group
        view_group = home_tab.addGroup("View")
        view_group.addButton("settings", "Preferences", "Application settings")
        
        # Analysis tab
        analysis_tab = self._ribbon.addTab("Analysis")
        
        # Multivariate group
        multivar_group = analysis_tab.addGroup("Multivariate")
        multivar_group.addButton("pca", "PCA", "Principal Component Analysis")
        multivar_group.addButton("pcoa", "PCoA", "Principal Coordinate Analysis")
        multivar_group.addButton("nmds", "NMDS", "Non-metric MDS")
        
        # Diversity group
        diversity_group = analysis_tab.addGroup("Diversity")
        diversity_group.addButton("diversity", "Diversity", "Biodiversity indices")
        
        # Group tests group
        tests_group = analysis_tab.addGroup("Tests")
        tests_group.addButton("anosim", "ANOSIM", "Analysis of Similarities")
        
        # Morphometrics tab
        morpho_tab = self._ribbon.addTab("Morphometrics")
        
        morpho_group = morpho_tab.addGroup("Landmarks")
        morpho_group.addButton("morphometrics", "GPA", "Generalized Procrustes Analysis")
        
        # Stratigraphy tab
        strat_tab = self._ribbon.addTab("Stratigraphy")
        
        strat_group = strat_tab.addGroup("Time Series")
        strat_group.addButton("stratigraphy", "Spectral", "Spectral Analysis")
    
    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Navigation signals
        self._navigation.itemClicked.connect(self._on_navigation_clicked)
        
        # Ribbon signals (simplified connection to actions)
        # Each ribbon button connects to appropriate handler
        ribbon_buttons = self._find_ribbon_buttons()
        for button in ribbon_buttons:
            text = button.text().lower()
            if "new" in text:
                button.clicked.connect(self._on_new_file)
            elif "open" in text:
                button.clicked.connect(self._on_open_file)
            elif "save" in text:
                button.clicked.connect(self._on_save_file)
            elif "pca" in text:
                button.clicked.connect(self._on_run_pca)
            elif "diversity" in text:
                button.clicked.connect(self._on_run_diversity)
            elif "spectral" in text or "stratigraphy" in text:
                button.clicked.connect(self._on_run_spectral)
    
    def _find_ribbon_buttons(self) -> List[RibbonButton]:
        """Find all ribbon buttons."""
        buttons = []
        for tab in [self._ribbon._tabs] if hasattr(self._ribbon, '_tabs') else []:
            for group in tab._groups:
                buttons.extend(group._buttons)
        return buttons
    
    def _create_menu_bar(self) -> None:
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Matrix", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_file)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._on_save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        import_action = QAction("&Import Data...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self._on_import_data)
        file_menu.addAction(import_action)
        
        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        pca_action = QAction("&PCA...", self)
        pca_action.setShortcut(QKeySequence("Ctrl+1"))
        pca_action.triggered.connect(self._on_run_pca)
        analysis_menu.addAction(pca_action)
        
        pcoa_action = QAction("P&CoA...", self)
        pcoa_action.setShortcut(QKeySequence("Ctrl+2"))
        pcoa_action.triggered.connect(self._on_run_pcoa)
        analysis_menu.addAction(pcoa_action)
        
        nmds_action = QAction("&NMDS...", self)
        nmds_action.setShortcut(QKeySequence("Ctrl+3"))
        nmds_action.triggered.connect(self._on_run_nmds)
        analysis_menu.addAction(nmds_action)
        
        analysis_menu.addSeparator()
        
        diversity_action = QAction("&Diversity...", self)
        diversity_action.setShortcut(QKeySequence("Ctrl+D"))
        diversity_action.triggered.connect(self._on_run_diversity)
        analysis_menu.addAction(diversity_action)
        
        rarefaction_action = QAction("&Rarefaction...", self)
        rarefaction_action.setShortcut(QKeySequence("Ctrl+R"))
        rarefaction_action.triggered.connect(self._on_run_rarefaction)
        analysis_menu.addAction(rarefaction_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About PaleoAST", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        doc_action = QAction("&Documentation", self)
        doc_action.setShortcut(QKeySequence("F1"))
        doc_action.triggered.connect(self._show_documentation)
        help_menu.addAction(doc_action)
    
    def _on_navigation_clicked(self, item: NavigationItem) -> None:
        """
        Handle navigation item click.
        
        Mathematical Context:
            Navigation items trigger different analysis workflows:
            
            For "Multivariate > PCA":
                User navigates: Home > Multivariate > PCA
                System loads selected columns from spreadsheet
                User configures parameters in dialog
                Pipeline executes: X ∈ ℝ^(n×p) → PCA → PC_scores ∈ ℝ^(n×k)
                
                Eigenvalue decomposition: $S v_j = \\lambda_j v_j$
                where $S = \\frac{1}{n-1} X^T X$ is the covariance matrix
        """
        section = item.section
        self.navigationChanged.emit(section)
        
        # Switch workspace view based on section
        if section in ["Data Management", "Univariate", "Multivariate", 
                       "Morphometrics", "Stratigraphy", "Ecology"]:
            self._workspace.setCurrentIndex(self._spreadsheet_index)
    
    def _on_new_file(self) -> None:
        """Create new empty data matrix."""
        # Show dialog to specify dimensions
        dialog = QDialog(self)
        dialog.setWindowTitle("New Data Matrix")
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        # Sample count
        samples_layout = QHBoxLayout()
        samples_label = QLabel("Number of Samples:")
        samples_spin = QSpinBox()
        samples_spin.setRange(2, 10000)
        samples_spin.setValue(30)
        samples_layout.addWidget(samples_label)
        samples_layout.addWidget(samples_spin)
        samples_layout.addStretch()
        layout.addLayout(samples_layout)
        
        # Variable count
        vars_layout = QHBoxLayout()
        vars_label = QLabel("Number of Variables:")
        vars_spin = QSpinBox()
        vars_spin.setRange(2, 1000)
        vars_spin.setValue(8)
        vars_layout.addWidget(vars_label)
        vars_layout.addWidget(vars_spin)
        vars_layout.addStretch()
        layout.addLayout(vars_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Create")
        cancel_button = QPushButton("Cancel")
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
            data = np.random.randn(n_samples, n_vars) * 10 + 50
            
            # Load into spreadsheet
            self._spreadsheet.load_data(data)
            self._workspace.setCurrentIndex(self._spreadsheet_index)
            
            self._status_bar.setInfo(
                f"New matrix: {n_samples} samples × {n_vars} variables"
            )
    
    def _on_open_file(self) -> None:
        """Open CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        
        if filepath:
            try:
                matrix = self._data_controller.load_csv(
                    filepath,
                    has_header=True,
                    has_row_labels=True
                )
                
                self._spreadsheet.load_data(
                    matrix.data,
                    row_labels=matrix.row_labels,
                    col_labels=matrix.col_labels
                )
                self._workspace.setCurrentIndex(self._spreadsheet_index)
                
                self._status_bar.setInfo(
                    f"Loaded: {filepath.split('/')[-1]}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Failed to load file:\n{str(e)}"
                )
    
    def _on_save_file(self) -> None:
        """Save current data."""
        if not self._state.has_data:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filepath:
            try:
                self._data_controller.export_csv(filepath)
                self._status_bar.setInfo(f"Saved: {filepath.split('/')[-1]}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))
    
    def _on_save_file_as(self) -> None:
        """Save data with new name."""
        self._on_save_file()
    
    def _on_import_data(self) -> None:
        """Show import data dialog."""
        dialog = ImportDialog(self)
        dialog.dataImported.connect(self._on_data_imported)
        dialog.exec()
    
    def _on_data_imported(self, data, metadata) -> None:
        """Handle imported data."""
        self._spreadsheet.load_data(data, **metadata)
        self._workspace.setCurrentIndex(self._spreadsheet_index)
    
    def _on_export(self) -> None:
        """Export analysis results."""
        QMessageBox.information(self, "Export", "Export functionality")
    
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
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        dialog = PCADialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            
            try:
                self._status_bar.setProgress(0, 0)  # Show indeterminate
                
                result = self._statistics_controller.run_pca(
                    n_components=params['n_components'],
                    method=params['method']
                )
                
                self._status_bar.setProgress(100, 100)
                
                # Create and display plot
                plot = InteractivePlotCanvas()
                plot.plot_pca_scores(result)
                
                plot_index = self._workspace.addWidget(plot, "PCA Plot")
                self._workspace.setCurrentIndex(plot_index)
                
                self._status_bar.setInfo(
                    f"PCA: {result.n_components} components, "
                    f"PC1+PC2 = {result.explained_variance[0] + result.explained_variance[1]:.1f}%"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "PCA Error", str(e))
    
    def _on_run_pcoa(self) -> None:
        """Run Principal Coordinate Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        dialog = PCoADialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            
            try:
                result = self._statistics_controller.run_pcoa(
                    metric=params['metric'],
                    n_components=params['n_components']
                )
                
                plot = InteractivePlotCanvas()
                plot.plot_pcoa_scores(result)
                
                plot_index = self._workspace.addWidget(plot, "PCoA Plot")
                self._workspace.setCurrentIndex(plot_index)
                
            except Exception as e:
                QMessageBox.critical(self, "PCoA Error", str(e))
    
    def _on_run_nmds(self) -> None:
        """Run Non-metric MDS."""
        if not self._state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        dialog = NMDSOptionsDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            
            try:
                result = self._statistics_controller.run_nmds(
                    metric=params['metric'],
                    n_dimensions=params['n_dimensions'],
                    n_restarts=params['n_restarts']
                )
                
                plot = InteractivePlotCanvas()
                plot.plot_nmds(result)
                
                plot_index = self._workspace.addWidget(plot, "NMDS Plot")
                self._workspace.setCurrentIndex(plot_index)
                
                self._status_bar.setInfo(
                    f"NMDS: stress = {result.stress:.4f}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "NMDS Error", str(e))
    
    def _on_run_diversity(self) -> None:
        """Run diversity analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        dialog = DiversityDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            
            try:
                result = self._statistics_controller.analyze_diversity(
                    sample_name=params['sample_name']
                )
                
                plot = InteractivePlotCanvas()
                plot.plot_diversity_summary(result)
                
                plot_index = self._workspace.addWidget(plot, "Diversity Plot")
                self._workspace.setCurrentIndex(plot_index)
                
            except Exception as e:
                QMessageBox.critical(self, "Diversity Error", str(e))
    
    def _on_run_rarefaction(self) -> None:
        """Run rarefaction analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        dialog = RarefactionDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            
            try:
                result = self._statistics_controller.analyze_rarefaction(
                    sample_name=params['sample_name']
                )
                
                plot = InteractivePlotCanvas()
                plot.plot_rarefaction(result)
                
                plot_index = self._workspace.addWidget(plot, "Rarefaction Plot")
                self._workspace.setCurrentIndex(plot_index)
                
            except Exception as e:
                QMessageBox.critical(self, "Rarefaction Error", str(e))
    
    def _on_run_spectral(self) -> None:
        """Run spectral analysis."""
        QMessageBox.information(self, "Spectral Analysis", 
                               "Spectral analysis: select time and value columns first.")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PaleoAST",
            """
            <h2>PaleoAST</h2>
            <p>Paleontological Advanced Statistical Toolkit</p>
            <p>Version 1.0.0</p>
            <p>Copyright © 2024 PaleoAST Development Team</p>
            <hr>
            <p>A comprehensive tool for paleontological data analysis including:</p>
            <ul>
                <li>Multivariate Statistics (PCA, PCoA, NMDS)</li>
                <li>Group Comparison Tests (ANOSIM, PERMANOVA)</li>
                <li>Geometric Morphometrics (GPA, TPS)</li>
                <li>Biodiversity Analysis</li>
                <li>Spectral Analysis</li>
            </ul>
            """
        )
    
    def _show_documentation(self) -> None:
        """Show documentation."""
        QMessageBox.information(
            self,
            "Documentation",
            "See ARCHITECTURE_BLUEPRINT.md for detailed documentation."
        )
    
    def _update_status(self) -> None:
        """Update status bar information."""
        if self._state.has_data:
            matrix = self._state.data_matrix
            info = f"Data: {matrix.n_samples} samples × {matrix.n_variables} variables"
            
            if self._state.is_modified:
                info += " (modified)"
            
            self._status_bar.setInfo(info)
        else:
            self._status_bar.setInfo("No data loaded")
    
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
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_file()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
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
    from PyQt6.QtWidgets import QApplication
    
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
