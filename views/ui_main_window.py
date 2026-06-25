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
version: 1.0.1
"""

import logging
import os
import sys
from enum import Enum

logger = logging.getLogger(__name__)

import contextlib

from PyQt6.QtCore import QPoint, QRect, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.design_system import BorderRadius, Typography, get_palette
from config.i18n import _, get_translator
from controllers.data_controller import DataController
from controllers.statistics_controller import StatisticsController
from models.state_manager import get_state_manager
from utils.event_bus import get_event_bus
from views.diagnostic_console import DiagnosticConsole
from views.file_drop_handler import FileDropHandler
from views.ui_allometry_dialogs import AllometryDialog
from views.ui_beta_diversity_dialogs import BetaDiversityDialog
from views.ui_dialogs import (
    BiostratigraphyDialog,
    CCADialog,
    ClusteringDialog,
    CONISSDialog,
    DirectionalDialog,
    DiversityDialog,
    EFADialog,
    ImportDialog,
    IsotopeAnalysisDialog,
    LDADialog,
    MarkovDialog,
    NMDSOptionsDialog,
    PCADialog,
    PCoADialog,
    PaleoEnvironmentDialog,
    RarefactionDialog,
    SimperDialog,
    SpatialRipleyKDialog,
    StratigraphicCorrelationDialog,
    TPSGridDialog,
    UnivariateDialog,
    WaveletDialog,
)
from views.ui_evolution_rate_dialogs import EvolutionRateDialog
from views.ui_extinction_dialogs import ExtinctionIntervalDialog
from views.ui_imputation_dialog import ImputationDialog
from views.ui_navigation import NavigationItem, NavigationTree
from views.ui_null_model_dialogs import NullModelDialog
from views.ui_pcm_dialogs import AncestralStateDialog, PhyloANOVADialog, PhyloSignalDialog, PICDialog
from views.ui_plot_canvas import InteractivePlotCanvas
from views.ui_spreadsheet import ScientificSpreadsheet


def format_user_error(e: Exception, operation: str = "") -> str:
    """
    将技术性异常消息转换为用户友好的中文提示。

    参数:
        e: 捕获的异常
        operation: 操作名称（如 "PCA"、"GPA" 等）

    返回:
        用户友好的错误消息
    """
    error_msg = str(e)
    operation_hint = f"{operation} " if operation else ""

    # 数据类型错误（最常见的中文字符或 "NA" 问题）
    if isinstance(e, (ValueError, TypeError)):
        # 检查是否是非法字符问题（更精确的匹配）
        if any(
            keyword in error_msg.lower()
            for keyword in [
                "could not convert string",
                "invalid literal for float",
                "can't convert",
                "string to float",
                "could not convert",
                "无法转换",
                "invalid choice",
                "not a valid",
            ]
        ):
            return _(
                "{0}失败：数据包含无效字符。\n\n"
                "请检查以下几点：\n"
                "• 选中的数据仅包含数值，不含文字或符号\n"
                "• 不存在缺失值标记（如 NA、NaN、-、空格等）\n"
                "• 如有中文或特殊字符，请先删除或替换"
            ).format(operation_hint)

        # 检查是否是数值计算错误（如 log(负数)、sqrt(负数)）
        if any(
            keyword in error_msg.lower()
            for keyword in ["negative value", "invalid value", "math domain error", "不能求", "数值计算"]
        ):
            return _(
                "{0}失败：数值计算错误。\n\n"
                "请检查以下几点：\n"
                "• 数据中是否存在负数（特别是对数运算前）\n"
                "• 是否存在零值（某些除法运算前）\n"
                "• 数值是否在有效范围内"
            ).format(operation_hint)

        # 检查是否是维度不匹配问题
        if any(keyword in error_msg.lower() for keyword in ["dimension", "shape", "axes"]):
            return _(
                "{0}失败：数据维度不匹配。\n\n"
                "请检查以下几点：\n"
                "• 数据的行数和列数符合分析要求\n"
                "• 不同数据集的样本数量是否一致\n"
                "• Landmark 数据是否为完整的 x,y 坐标对"
            ).format(operation_hint)

        # 检查是否是空数据问题
        if "empty" in error_msg.lower() or "没有数据" in error_msg:
            return _("{0}失败：数据为空。\n\n请确保已选中有效的数据区域。").format(operation_hint)

        # 通用数据类型错误
        return _(
            "{0}失败：数据类型错误。\n\n"
            "错误信息：{1}\n\n"
            "请检查选中的数据是否为数值类型，并确保无缺失值。"
        ).format(operation_hint, error_msg[:100])

    # 验证错误
    if "ValidationError" in type(e).__name__ or "验证" in error_msg:
        return _("{0}失败：数据验证未通过。\n\n{1}").format(operation_hint, error_msg)

    # 收敛错误（迭代算法未收敛）
    if "ConvergenceError" in type(e).__name__ or "收敛" in error_msg:
        return _(
            "{0}警告：算法未收敛。\n\n"
            "这通常是由于数据质量问题或参数设置不当导致。\n"
            "建议：\n"
            "• 检查数据中是否存在异常值\n"
            "• 尝试增加迭代次数\n"
            "• 尝试使用不同的初始化参数"
        ).format(operation_hint)

    # 矩阵计算错误
    if "singular" in error_msg.lower() or "matrix" in error_msg.lower():
        return _(
            "{0}失败：矩阵计算错误。\n\n"
            "这通常是由于数据中存在线性相关（多重共线性）导致。\n"
            "建议：\n"
            "• 检查并移除高度相关的变量\n"
            "• 标准化数据后再试\n"
            "• 减少变量数量"
        ).format(operation_hint)

    # 默认：显示原始错误消息的前100个字符
    return _("{0}时发生错误：\n\n{1}\n\n如果问题持续存在，请检查数据格式是否正确。").format(
        operation_hint, error_msg[:200]
    )


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

        elif icon_type == "pcoa":
            # 2D scatter with convex hull representing a metric space
            center = (size // 2, size // 2)
            points = [
                QPoint(center[0] - inner_size // 3, center[1] - inner_size // 4),
                QPoint(center[0] + inner_size // 4, center[1] - inner_size // 3),
                QPoint(center[0] + inner_size // 3, center[1] + inner_size // 4),
                QPoint(center[0] - inner_size // 4, center[1] + inner_size // 3),
                QPoint(center[0] - inner_size // 3, center[1] - inner_size // 4),
            ]
            painter.setPen(QPen(QColor("#8E44AD"), max(2, size // 16)))
            painter.setBrush(QBrush(QColor("#8E44AD")))
            for pt in points:
                painter.drawEllipse(pt, size // 18, size // 18)
            painter.drawPolyline(points[:-1])

        elif icon_type == "chart":
            # Generic bar chart (used by LDA, CCA, stats, etc.)
            bar_count = 4
            bar_width = inner_size // (bar_count * 2)
            bar_colors = ["#3498DB", "#E74C3C", "#27AE60", "#F39C12"]
            for i in range(bar_count):
                height_factor = 0.4 + 0.6 * (1 - abs(i - 1.5) / 2)
                bar_height = int(inner_size * height_factor)
                x = margin + i * (inner_size // bar_count) + bar_width // 2
                y = margin + inner_size - bar_height
                painter.setBrush(QBrush(QColor(bar_colors[i])))
                painter.drawRect(QRect(x, y, bar_width, bar_height))

        elif icon_type == "imputation":
            # Grid of cells with one cell highlighted to suggest "filling in"
            painter.setPen(QPen(QColor("#7F8C8D"), max(1, size // 32)))
            cell_size = inner_size // 3
            grid_origin_x = margin + (inner_size - 3 * cell_size) // 2
            grid_origin_y = margin + (inner_size - 3 * cell_size) // 2
            for r in range(3):
                for c in range(3):
                    rect = QRect(
                        grid_origin_x + c * cell_size,
                        grid_origin_y + r * cell_size,
                        cell_size,
                        cell_size,
                    )
                    if (r, c) == (1, 1):
                        painter.setBrush(QBrush(QColor("#27AE60")))
                    else:
                        painter.setBrush(QBrush(QColor("#ECF0F1")))
                    painter.drawRect(rect)

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
        self._is_dark_theme = getattr(parent, "_is_dark_theme", False) if parent else False

        # Set button properties
        self.setText(text)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

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
        c = get_palette(self._is_dark_theme)
        t = Typography()
        r = BorderRadius()
        ss = (
            "QPushButton {"
            "background-color: " + c.bg_secondary + "; "
            "color: " + c.text_primary + "; "
            "border: 1px solid " + c.border_light + "; "
            "border-radius: " + r.md + "; "
            "padding: 4px 10px; "
            "font-family: " + t.family_primary + "; "
            "font-size: " + str(t.body_sm_size) + "px; "
            "font-weight: " + str(t.medium) + "; "
            "min-width: 60px; "
            "min-height: 24px; "
            "} "
            "QPushButton:hover { "
            "background-color: " + c.bg_tertiary + "; "
            "border: 1px solid " + c.primary + "; "
            "color: " + c.primary + "; "
            "} "
            "QPushButton:focus { "
            "border: 2px solid " + c.primary + "; "
            "} "
            "QPushButton:pressed { "
            "background-color: " + c.primary + "; "
            "color: white; "
            "border: 1px solid " + c.primary_dark + "; "
            "} "
            "QPushButton:disabled { "
            "background-color: " + c.bg_tertiary + "; "
            "color: " + c.text_disabled + "; "
            "border: 1px solid " + c.border_light + "; "
            "} "
            'QPushButton[flat="true"] { '
            "background-color: transparent; "
            "border: none; "
            "} "
            'QPushButton[flat="true"]:hover { '
            "background-color: " + c.hover_overlay + "; "
            "border: none; "
            "}"
        )
        self.setStyleSheet(ss)


class RibbonGroup(QWidget):
    """
    Ribbon Group containing related buttons.

    A group has a title and contains a horizontal layout of buttons.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._title = title
        self._buttons: list[RibbonButton] = []
        self._is_dark_theme = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(2)

        # Button container
        self._button_container = QWidget()
        self._button_layout = QHBoxLayout(self._button_container)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(4)
        self._button_layout.addStretch()
        self._layout.addWidget(self._button_container)

        # Title label
        t = Typography()
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet(
            "QLabel { "
            "color: " + get_palette().primary + "; "
            "font-size: " + str(t.caption_size) + "px; "
            "font-weight: " + str(t.semibold) + "; "
            "padding: 1px; "
            "}"
        )
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

    def addComboBox(self, items: list[str], tooltip: str = "", on_change=None) -> QComboBox:
        """Add a combo box (dropdown) to the ribbon group."""
        combo = QComboBox(self)
        combo.addItems(items)
        combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if tooltip:
            combo.setToolTip(tooltip)

        c = get_palette(self._is_dark_theme)
        t = Typography()
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {c.bg_secondary};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 3px;
                padding: 2px 6px;
                font-size: {t.body_size}px;
                min-width: 90px;
                max-height: 24px;
            }}
            QComboBox:hover {{
                border-color: {c.primary};
            }}
            QComboBox::dropDown {{
                border: none;
                width: 18px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid {c.text_secondary};
                margin-right: 2px;
            }}
            QComboBox QAbstractItemView {{
                background: {c.bg_primary};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                selection-background-color: {c.primary};
                min-width: 100px;
            }}
        """)

        if on_change:
            combo.currentIndexChanged.connect(on_change)

        self._button_layout.insertWidget(self._button_layout.count() - 1, combo)
        return combo

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and propagate to all buttons."""
        self._is_dark_theme = is_dark
        c = get_palette(is_dark)
        t = Typography()
        self._title_label.setStyleSheet(
            "QLabel { "
            "color: " + c.primary + "; "
            "font-size: " + str(t.caption_size) + "px; "
            "font-weight: " + str(t.semibold) + "; "
            "padding: 1px; "
            "}"
        )
        for button in self._buttons:
            button.setDarkTheme(is_dark)


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
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(4)
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

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and propagate to all groups.

        ``RibbonTab`` is a plain container widget - it does not own a
        stylesheet of its own (visual appearance is delegated to the
        child ``RibbonGroup`` instances and the global app stylesheet).
        Earlier versions called a non-existent ``self._apply_stylesheet``
        here, which crashed the whole theme-switch chain with
        ``AttributeError`` the first time the user toggled the theme.
        """
        self._is_dark_theme = is_dark
        for group in self._groups:
            group.setDarkTheme(is_dark)


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

        # Prevent ribbon from expanding vertically
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Tab bar
        self._tab_bar = QWidget()
        self._tab_bar_layout = QHBoxLayout(self._tab_bar)
        self._tab_bar_layout.setContentsMargins(4, 2, 4, 0)
        self._tab_bar_layout.setSpacing(2)
        self._tab_button_group: list[QPushButton] = []
        self._main_layout.addWidget(self._tab_bar)

        # Content area
        self._content_area = QWidget()
        self._content_area.setMinimumHeight(68)
        self._content_area.setMaximumHeight(84)
        self._content_layout = QHBoxLayout(self._content_area)
        self._content_layout.setContentsMargins(8, 4, 8, 4)
        self._content_layout.addStretch()
        self._main_layout.addWidget(self._content_area)

        # Separator line
        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.Shape.HLine)
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
        # ``len()`` is evaluated once at lambda definition, capturing the
        # current tab count; this is the intended behaviour, not a
        # default-argument pitfall. The B008 warning is a false positive.
        tab_button.clicked.connect(lambda checked=False, idx=len(self._tabs) - 1: self._on_tab_clicked(idx))  # noqa: B008
        self._tab_button_group.append(tab_button)
        self._tab_bar_layout.addWidget(tab_button)

        # Show first tab content; hide all others
        if len(self._tabs) == 1:
            self._show_tab(0)
        else:
            tab.hide()

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

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and propagate to all tabs."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()
        for tab in self._tabs:
            tab.setDarkTheme(is_dark)

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        t = Typography()
        r = BorderRadius()
        ss = (
            "QWidget { background-color: " + c.bg_primary + "; border: none; } "
            "QPushButton { "
            "background-color: transparent; "
            "border: none; "
            "color: " + c.text_primary + "; "
            "padding: 4px 10px; "
            "font-size: " + str(t.body_sm_size) + "px; "
            "font-weight: " + str(t.semibold) + "; "
            "border-radius: " + r.md + "; "
            "} "
            "QPushButton:hover { "
            "background-color: " + c.hover_overlay + "; "
            "color: " + c.primary + "; "
            "} "
            "QPushButton:checked { "
            "background-color: " + c.active_overlay + "; "
            "border-bottom: 2px solid " + c.primary + "; "
            "color: " + c.primary + "; "
            "}"
        )
        self.setStyleSheet(ss)
        self._separator.setStyleSheet("background-color: " + c.border_light + "; max-height: 1px;")


class StatusBarWidget(QStatusBar):
    """
    Custom status bar widget with data info and progress.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_dark_theme = False
        self._t = Typography()
        self._r = BorderRadius()
        self.setContentsMargins(8, 2, 8, 2)

        # Data info label (left side)
        self._info_label = QLabel(_("No data loaded"))
        self.addWidget(self._info_label)

        # Memory indicator (right side)
        self._memory_label = QLabel(_("Memory: 0 MB"))
        self.addPermanentWidget(self._memory_label)

        # Progress bar (right side)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(150)
        self._progress_bar.setMaximumHeight(12)
        self._progress_bar.setVisible(False)
        self.addPermanentWidget(self._progress_bar)

        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet."""
        c = get_palette(self._is_dark_theme)
        t = self._t
        r = self._r
        self.setStyleSheet(
            "QStatusBar { "
            "background-color: " + c.bg_secondary + "; "
            "border-top: 1px solid " + c.border_light + "; "
            "color: " + c.text_secondary + "; "
            "}"
        )
        self._info_label.setStyleSheet(
            "QLabel { color: " + c.text_secondary + "; font-size: " + str(t.body_sm_size) + "px; }"
        )
        self._memory_label.setStyleSheet(
            "QLabel { color: " + c.text_disabled + "; font-size: " + str(t.caption_size) + "px; }"
        )
        self._progress_bar.setStyleSheet(
            "QProgressBar { "
            "border: 1px solid " + c.border_light + "; "
            "border-radius: " + r.md + "; "
            "text-align: center; "
            "background-color: " + c.bg_secondary + "; "
            "} "
            "QProgressBar::chunk { "
            "background-color: " + c.primary + "; "
            "border-radius: " + r.sm + "; "
            "}"
        )

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

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

    # Signal emitted when current widget changes
    currentChanged = pyqtSignal(object)  # widget

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_dark_theme = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Stacked widget for different views
        self._stack = QStackedWidget()
        self._stack.currentChanged.connect(self._on_current_changed)
        self._layout.addWidget(self._stack)

        # Placeholder widget
        t = Typography()
        self._placeholder = QLabel(_("Load data to begin analysis"))
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "QLabel { "
            "color: " + get_palette().text_disabled + "; "
            "font-size: " + str(t.body_lg_size) + "px; "
            "background-color: " + get_palette().bg_primary + "; "
            "}"
        )
        self._stack.addWidget(self._placeholder)

    def _on_current_changed(self, index: int) -> None:
        """Handle current widget change."""
        widget = self._stack.widget(index)
        self.currentChanged.emit(widget)

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and propagate to current widget.

        The workspace can host arbitrary widgets including plain
        ``QWidget`` figure-host containers (created by
        :meth:`MainWindow._embed_figure_in_workspace`) and
        ``QTreeWidget`` UAZ-hierarchy hosts. Only widgets that
        actually expose ``setDarkTheme`` are notified; for figure
        hosts, the matplotlib Figure is repainted via
        :meth:`MainWindow._apply_dark_theme_to_figure` (delegated by
        re-emitting the ``currentChanged`` signal so the parent
        MainWindow can react). This avoids the historical
        ``AttributeError: 'QWidget' object has no attribute
        'setDarkTheme'`` crash when toggling the theme after a
        publication-quality figure had been embedded.
        """
        self._is_dark_theme = is_dark
        c = get_palette(is_dark)
        t = Typography()
        self._placeholder.setStyleSheet(
            "QLabel { "
            "color: " + c.text_disabled + "; "
            "font-size: " + str(t.body_lg_size) + "px; "
            "background-color: " + c.bg_primary + "; "
            "}"
        )
        # Propagate the theme to every widget in the stack that has a
        # ``setDarkTheme`` method, not just the currently visible one.
        # Iterating the whole stack means previously-embedded plots
        # also re-theme correctly when the user switches modes.
        for i in range(self._stack.count()):
            w = self._stack.widget(i)
            if w is None or w is self._placeholder:
                continue
            setter = getattr(w, "setDarkTheme", None)
            if callable(setter):
                try:
                    setter(is_dark)
                except Exception:
                    # Best-effort: never let a single widget's failure
                    # break the global theme switch.
                    pass

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

        # Subscribe to EventBus for data-driven updates
        self._event_bus = get_event_bus()
        self._event_bus.data_changed.connect(self._on_data_changed)
        self._event_bus.undo_stack_changed.connect(self._on_undo_stack_changed)

        # Create widgets
        self._create_ui()

        # Setup connections
        self._setup_connections()

        # Load settings
        self._load_settings()

        # Status update timer (retained only for memory monitoring)
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

        # Setup drag and drop
        self._setup_drag_drop()

    def _setup_drag_drop(self) -> None:
        """Setup drag and drop for file loading."""
        self.setAcceptDrops(True)
        self._drop_handler = FileDropHandler(self)
        self._drop_handler.file_loaded.connect(self._on_file_dropped)
        self._drop_handler.load_failed.connect(self._on_file_drop_failed)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self._drop_handler.can_handle(file_path):
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle file drop event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self._drop_handler.can_handle(file_path):
                    event.acceptProposedAction()
                    self._status_bar.setInfo(_("Loading file..."))
                    self._drop_handler.handle_file(file_path)
                    return
        super().dropEvent(event)

    def _on_file_dropped(self, data: dict, file_type: str) -> None:
        """Handle successful file load via drag and drop."""
        try:
            from models.data_matrix import DataMatrix

            if data.get("type") == "tree":
                QMessageBox.information(
                    self,
                    _("File Loaded"),
                    _("Tree file loaded: {0}\nNote: Tree visualization coming soon.").format(file_type),
                )
                return

            matrix_data = data.get("data")
            if matrix_data is None:
                raise ValueError("No data in parsed file")

            row_labels = data.get("row_labels")
            col_labels = data.get("col_labels")

            new_matrix = DataMatrix(matrix_data, row_labels=row_labels, col_labels=col_labels)

            self._state.set_data_matrix(new_matrix)
            self._spreadsheet.load_data(matrix_data, row_labels=row_labels, col_labels=col_labels)

            self._status_bar.setInfo(
                _("Loaded: {0} rows x {1} columns").format(new_matrix.n_samples, new_matrix.n_variables)
            )

            QMessageBox.information(
                self,
                _("File Loaded"),
                _("Successfully loaded {0}\n{1} rows x {2} columns").format(
                    file_type, new_matrix.n_samples, new_matrix.n_variables
                ),
            )

        except Exception as e:
            QMessageBox.critical(self, _("Load Error"), format_user_error(e, "文件加载"))

    def _on_file_drop_failed(self, error_msg: str) -> None:
        """Handle file load failure."""
        self._status_bar.setInfo(_("Load failed"))
        QMessageBox.critical(self, _("Load Error"), error_msg)

    def _create_ui(self) -> None:
        """Create all UI components."""
        # Set window properties
        self.setWindowTitle(_("PaleoAST - Paleontological Advanced Statistical Toolkit"))
        self.setMinimumSize(1024, 700)
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
        self._navigation.setMinimumWidth(200)
        self._navigation.setMaximumWidth(350)
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

        # Diagnostic console (dockable)
        self._diagnostic_console = DiagnosticConsole(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._diagnostic_console)

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
        self._btn_undo = edit_group.addButton("undo", _("Undo"), _("Undo last action (Ctrl+Z)"))
        self._btn_redo = edit_group.addButton("redo", _("Redo"), _("Redo action (Ctrl+Y)"))
        self._btn_transpose = edit_group.addButton("transpose", _("Transpose"), _("Transpose data matrix"))
        self._btn_imputation = edit_group.addButton("imputation", _("NaN"), _("Missing Value Imputation"))

        # Data transformations group
        transform_group = home_tab.addGroup(_("Transform"))
        self._btn_log_transform = transform_group.addButton("settings", _("Log"), _("Log transformation (base 10)"))
        self._btn_sqrt_transform = transform_group.addButton("settings", _("Sqrt"), _("Square root transformation"))
        self._btn_hellinger_transform = transform_group.addButton(
            "settings", _("Hellinger"), _("Hellinger transformation")
        )
        self._btn_zscore_transform = transform_group.addButton("settings", _("Z-Score"), _("Z-score standardization"))
        self._btn_percent_transform = transform_group.addButton(
            "settings", _("% Total"), _("Percentage standardization")
        )
        self._btn_wisconsin_transform = transform_group.addButton(
            "settings", _("Wisconsin"), _("Wisconsin double standardization")
        )

        # View group
        view_group = home_tab.addGroup(_("View"))
        self._btn_preferences = view_group.addButton("settings", _("Preferences"), _("Application settings"))

        # Analysis tab
        analysis_tab = self._ribbon.addTab(_("Analysis"))

        # Multivariate group
        multivar_group = analysis_tab.addGroup(_("Multivariate"))
        self._btn_pca = multivar_group.addButton("pca", "PCA", _("Principal Component Analysis"))
        self._btn_pcoa = multivar_group.addButton("pcoa", "PCoA", _("Principal Coordinate Analysis"))
        self._btn_nmds = multivar_group.addButton("nmds", "NMDS", _("Non-metric MDS"))
        self._btn_lda = multivar_group.addButton("chart", "LDA", _("Linear Discriminant Analysis"))
        self._btn_cca = multivar_group.addButton("chart", "CCA", _("Canonical Correspondence Analysis"))

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
        self._btn_gpa = morpho_group.addButton("morphometrics", "GPA", _("Generalized Procrustes Analysis"))

        efa_group = morpho_tab.addGroup(_("Outline"))
        self._btn_efa = efa_group.addButton("morphometrics", "EFA", _("Elliptic Fourier Analysis"))

        tps_group = morpho_tab.addGroup(_("Deformation"))
        self._btn_tps_grid = tps_group.addButton("morphometrics", _("Grid"), _("TPS Deformation Grid"))

        # Spatial tab
        spatial_tab = self._ribbon.addTab(_("Spatial"))
        spatial_group = spatial_tab.addGroup(_("Point Pattern"))
        self._btn_ripley_k = spatial_group.addButton("chart", _("Ripley K"), _("Ripley's K Spatial Analysis"))

        # Stratigraphy tab
        strat_tab = self._ribbon.addTab(_("Stratigraphy"))

        strat_group = strat_tab.addGroup(_("Time Series"))
        self._btn_spectral = strat_group.addButton("stratigraphy", _("Spectral"), _("Spectral Analysis"))
        self._btn_coniss = strat_group.addButton("stratigraphy", "CONISS", _("CONISS Zonation"))
        self._btn_wavelet = strat_group.addButton("stratigraphy", _("Wavelet"), _("Wavelet CWT Analysis"))
        self._btn_isotope = strat_group.addButton("stratigraphy", _("Isotope"), _("Isotope Time Series"))
        self._btn_strat_corr = strat_group.addButton("stratigraphy", _("Correlation"), _("Stratigraphic Correlation"))

        bio_group = strat_tab.addGroup(_("Biostratigraphy"))
        self._btn_biostrat = bio_group.addButton("stratigraphy", _("Biozone"), _("UA/RASC Biostratigraphy"))

        paleo_group = strat_tab.addGroup(_("Paleo-Environment"))
        self._btn_paleo_env = paleo_group.addButton(
            "stratigraphy", _("CA Axis"), _("Paleo-Env. CA Reconstruction")
        )

        markov_group = strat_tab.addGroup(_("Facies"))
        self._btn_markov = markov_group.addButton("stratigraphy", _("Markov"), _("Markov Chain Analysis"))
        self._btn_directional = markov_group.addButton("stratigraphy", _("Rose"), _("Directional Statistics"))

    def _setup_connections(self) -> None:
        """Setup signal-slot connections."""
        # Navigation signals
        self._navigation.itemClicked.connect(self._on_navigation_clicked)
        # self.navigationChanged is a public signal exposed for plugins
        # and external observers; make sure the default internal
        # observer is connected so an early emit does not get lost.
        try:
            self.navigationChanged.connect(self._on_navigation_changed_external)
        except TypeError:
            # Slot may already be connected or signal is unavailable.
            pass

        # File operation buttons (always enabled)
        self._btn_new.clicked.connect(self._on_new_file)
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_save.clicked.connect(self._on_save_file)

        # Edit operation buttons
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_redo.clicked.connect(self._on_redo)
        self._btn_transpose.clicked.connect(self._on_transpose)
        self._btn_imputation.clicked.connect(self._on_run_imputation)
        self._register_data_button(self._btn_undo)
        self._register_data_button(self._btn_redo)
        self._register_data_button(self._btn_transpose)
        self._register_data_button(self._btn_imputation)

        # View buttons
        self._btn_preferences.clicked.connect(self._on_preferences)

        # Transformation buttons
        self._btn_log_transform.clicked.connect(self._on_transform_log)
        self._btn_sqrt_transform.clicked.connect(self._on_transform_sqrt)
        self._btn_hellinger_transform.clicked.connect(self._on_transform_hellinger)
        self._btn_zscore_transform.clicked.connect(self._on_transform_zscore)
        self._btn_percent_transform.clicked.connect(self._on_transform_percent)
        self._btn_wisconsin_transform.clicked.connect(self._on_transform_wisconsin)
        for btn in [
            self._btn_log_transform,
            self._btn_sqrt_transform,
            self._btn_hellinger_transform,
            self._btn_zscore_transform,
            self._btn_percent_transform,
            self._btn_wisconsin_transform,
        ]:
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
        self._btn_cca.clicked.connect(self._on_run_cca)
        self._register_data_button(self._btn_cca)
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
        self._btn_isotope.clicked.connect(self._on_run_isotope)
        self._register_data_button(self._btn_isotope)
        self._btn_strat_corr.clicked.connect(self._on_run_stratigraphic)
        self._register_data_button(self._btn_strat_corr)
        self._btn_markov.clicked.connect(self._on_run_markov)
        self._register_data_button(self._btn_markov)
        self._btn_directional.clicked.connect(self._on_run_directional)
        self._register_data_button(self._btn_directional)
        self._btn_wavelet.clicked.connect(self._on_run_wavelet)
        self._register_data_button(self._btn_wavelet)
        self._btn_biostrat.clicked.connect(self._on_run_biostrat)
        self._register_data_button(self._btn_biostrat)
        self._btn_paleo_env.clicked.connect(self._on_run_paleo_env)
        self._register_data_button(self._btn_paleo_env)
        self._btn_efa.clicked.connect(self._on_run_efa)
        self._register_data_button(self._btn_efa)
        self._btn_tps_grid.clicked.connect(self._on_run_tps_grid)
        self._register_data_button(self._btn_tps_grid)
        self._btn_gpa.clicked.connect(self._on_run_gpa)
        self._register_data_button(self._btn_gpa)
        self._btn_ripley_k.clicked.connect(self._on_run_ripley_k)
        self._register_data_button(self._btn_ripley_k)

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

    def _on_data_changed(self, matrix) -> None:
        """Handle data_changed event from EventBus."""
        self._update_ui_state()
        self._update_data_display()

    def _on_undo_stack_changed(self) -> None:
        """Handle undo_stack_changed event from EventBus."""
        self._update_ui_state()

    def _update_data_display(self) -> None:
        """Update data-dependent display elements."""
        if self._state.has_data:
            matrix = self._state.data_matrix
            info = _("Data: {0} samples x {1} variables").format(matrix.n_samples, matrix.n_variables)
            if self._state.is_modified:
                info += _(", modified")
            self._status_bar.setInfo(info)
        else:
            self._status_bar.setInfo(_("No data loaded"))

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

        isotope_action = QAction(_("&Isotope Time Series..."), self)
        isotope_action.triggered.connect(self._on_run_isotope)
        analysis_menu.addAction(isotope_action)
        self._register_data_action(isotope_action)

        strat_action = QAction(_("&Stratigraphic Correlation..."), self)
        strat_action.triggered.connect(self._on_run_stratigraphic)
        analysis_menu.addAction(strat_action)
        self._register_data_action(strat_action)

        # Phylogenetic Comparative Methods submenu
        analysis_menu.addSeparator()
        pcm_submenu = QMenu(_("Phylogenetic Comparative Methods"), self)
        analysis_menu.addMenu(pcm_submenu)

        pic_action = QAction(_("&Independent Contrasts (PIC)..."), self)
        pic_action.triggered.connect(self._on_run_pic)
        pcm_submenu.addAction(pic_action)

        asr_action = QAction(_("&Ancestral State Reconstruction..."), self)
        asr_action.triggered.connect(self._on_run_ancestral_states)
        pcm_submenu.addAction(asr_action)

        signal_action = QAction(_("&Phylogenetic Signal (Blomberg's K)..."), self)
        signal_action.triggered.connect(self._on_run_phylogenetic_signal)
        pcm_submenu.addAction(signal_action)

        pcanova_action = QAction(_("Phylogenetic &ANOVA..."), self)
        pcanova_action.triggered.connect(self._on_run_phylo_anova)
        pcm_submenu.addAction(pcanova_action)

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
        """Switch application language with restart prompt.

        The menu-bar check state is only flipped after the user
        confirms an immediate restart. If the user defers ("Later"),
        the actual translator language remains unchanged and the menu
        check-marks are rolled back to reflect the real current state.
        """
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import QMessageBox

        current_lang = get_translator().get_language()
        if lang == current_lang:
            return

        # Ask user whether to restart now.
        # NOTE: do NOT update QSettings / check-marks yet. We commit
        # the new language only if the user explicitly accepts the
        # restart, otherwise the menu would be lying about the
        # in-session language.
        msg = QMessageBox(self)
        msg.setWindowTitle(_("Language Changed"))
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(
            _("The language has been changed to {0}.").format(
                "中文" if lang == "zh" else "English"
            )
        )
        msg.setInformativeText(_("Restart now to apply the change?"))
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.button(QMessageBox.StandardButton.Yes).setText(_("Restart Now"))
        msg.button(QMessageBox.StandardButton.No).setText(_("Later"))

        if msg.exec() != QMessageBox.StandardButton.Yes:
            # User deferred. Roll back the menu check-marks so they
            # reflect the actual current language.
            self._lang_action_en.setChecked(current_lang == "en")
            self._lang_action_zh.setChecked(current_lang == "zh")
            return

        # User accepted the restart: persist the choice *before* the
        # process is replaced (QSettings flush is best-effort but
        # explicit saves are safer).
        settings = QSettings("PaleoAST", "PaleoAST")
        settings.setValue("language", lang)
        settings.sync()
        self._lang_action_en.setChecked(lang == "en")
        self._lang_action_zh.setChecked(lang == "zh")
        self._restart_application()

    def _restart_application(self) -> None:
        """Restart the application process.

        Uses :func:`os.execv` to replace the current process with a
        fresh interpreter invocation. ``QApplication.quit`` is called
        first so that Qt's internal cleanup runs before the exec swap;
        otherwise the new process can occasionally fail to bind to
        the same display on some platforms.
        """
        import os
        import sys

        from PyQt6.QtWidgets import QApplication

        # Save any pending state (settings, undo/redo) before the swap.
        with contextlib.suppress(Exception):
            QApplication.instance().aboutToQuit.emit()

        # Stop the event loop explicitly. ``os.execv`` does not run
        # atexit handlers, so we are responsible for flushing state.
        with contextlib.suppress(Exception):
            self._save_settings()

        QApplication.quit()

        # Replace the current process with a fresh interpreter.
        # ``os.execv`` is preferred over ``os.execl`` because it accepts
        # the full argument list as a single sequence.
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except OSError:
            # If exec fails (e.g. the binary is gone) fall back to a
            # plain exit and let the user restart manually.
            sys.exit(0)

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
            _("Matrix Operations"): self._on_matrix_operations,
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
            _("Summary"): lambda: self._on_run_univariate_by_index(0),
            _("Normality"): lambda: self._on_run_univariate_by_index(1),
            _("t-test"): lambda: self._on_run_univariate_by_index(2),
            _("ANOVA"): lambda: self._on_run_univariate_by_index(3),
            _("Kruskal-Wallis"): lambda: self._on_run_univariate_by_index(4),
            _("Clustering"): self._on_run_clustering,
            _("Abundance Models"): self._on_run_abundance_models,
            "SHE": self._on_run_she,
            "CONISS": self._on_run_coniss,
            _("Markov"): self._on_run_markov,
            _("Directional"): self._on_run_directional,
            "EFA": self._on_run_efa,
            _("GPA Alignment"): self._on_run_gpa,
            _("TPS Deformation"): self._on_run_tps_grid,
            _("Relative Warps"): self._on_run_efa,  # Uses EFA as backend
            _("Unitary Associations"): self._on_run_biostrat,
            _("Biozone"): self._on_run_biostrat,
            _("Allometry"): self._on_run_allometry,
            _("Evolution Rate"): self._on_run_evolution_rate,
            _("Extinction Intervals"): self._on_run_extinction_intervals,
            _("Beta Diversity"): self._on_run_beta_diversity,
            _("Null Models"): self._on_run_null_models,
            # Stratigraphy & paleo-environment entries also reachable
            # via the navigation tree (mirrors the ribbon buttons).
            _("Isotope"): self._on_run_isotope,
            _("Stratigraphic Correlation"): self._on_run_stratigraphic,
            _("Wavelet"): self._on_run_wavelet,
            _("CA Axis"): self._on_run_paleo_env,
        }

        handler = action_map.get(name)
        if handler is not None:
            handler()
            return

        # For category clicks or un-mapped items, switch to spreadsheet view
        self._workspace.setCurrentIndex(self._spreadsheet_index)

    def _on_navigation_changed_external(self, section: str) -> None:
        """Default internal observer for ``navigationChanged``.

        Plugins may emit ``navigationChanged`` programmatically. To keep
        the in-app status bar consistent we surface the section name
        there as well.
        """
        if not section:
            return
        self._status_bar.setInfo(_("Section: {0}").format(section))

    def _on_matrix_operations(self) -> None:
        """Switch to spreadsheet view for matrix operations."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        self._workspace.setCurrentIndex(self._spreadsheet_index)
        self._status_bar.setInfo(_("Matrix Operations: edit data in spreadsheet"))

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
            row_labels = [f"Sample_{i + 1}" for i in range(n_samples)]
            col_labels = [f"Var_{j + 1}" for j in range(n_vars)]
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
            self,
            _("Open Data File"),
            "",
            _(
                "Data Files (*.csv *.txt *.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx *.xls);;All Files (*)"
            ),
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

                self._status_bar.setInfo(_("Loaded: {0}").format(os.path.basename(filepath)))

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
                self._status_bar.setInfo(_("Saved: {0}").format(os.path.basename(filepath)))
                return True
            except Exception as e:
                QMessageBox.critical(self, _("Save Error"), str(e))
                return False
        return False

    def _on_save_file_as(self) -> None:
        """Save data with new name."""
        self._on_save_file()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and propagate to all child widgets."""
        self._is_dark_theme = is_dark
        from config.design_system import get_stylesheet

        self.setStyleSheet(get_stylesheet(is_dark))
        self._ribbon.setDarkTheme(is_dark)
        self._status_bar.setDarkTheme(is_dark)
        self._workspace.setDarkTheme(is_dark)
        self._navigation.setDarkTheme(is_dark)
        # Re-theme any embedded matplotlib Figure widgets that we
        # added via :meth:`_embed_figure_in_workspace`. These do not
        # implement ``setDarkTheme`` themselves; the helper applies
        # the palette directly to the Figure's axes / spines / labels
        # and triggers a canvas redraw so the embedded plot matches
        # the rest of the UI after the theme toggle.
        self._retheme_embedded_figures(is_dark)

    def _retheme_embedded_figures(self, is_dark: bool) -> None:
        """Re-paint every embedded Matplotlib figure in the workspace
        to match the current theme."""
        try:
            stack = self._workspace._stack
            for i in range(stack.count()):
                widget = stack.widget(i)
                if widget is None:
                    continue
                canvas = widget.property("figure_canvas")
                # ``figure_canvas`` is set by ``_embed_figure_in_workspace``
                # to a ``FigureCanvasQTAgg``; ``_show_uaz_tree`` does not
                # set this property, so only true figure hosts respond.
                figure = getattr(canvas, "figure", None)
                if figure is None:
                    continue
                if is_dark:
                    self._apply_dark_theme_to_figure(figure)
                else:
                    self._apply_light_theme_to_figure(figure)
                try:
                    canvas.draw_idle()
                except Exception:
                    self._logger.debug(
                        "canvas.draw_idle() failed during re-theme", exc_info=True
                    )
        except Exception:
            self._logger.debug("_retheme_embedded_figures failed", exc_info=True)

    def _apply_light_theme_to_figure(self, figure: object) -> None:
        """Restore the default (light) Matplotlib palette on a figure.

        Mirrors :meth:`_apply_dark_theme_to_figure` so the user can
        toggle the global theme back to light after dark-theming an
        embedded plot. Without this restoration the dark colours would
        stay baked into the figure and only the surrounding chrome
        would flip back to white.
        """
        try:
            from config.design_system import get_palette

            palette = get_palette(False)
            bg = palette.bg_primary
            fg = palette.text_primary
            border = palette.border_medium
            try:
                figure.patch.set_facecolor(bg)  # type: ignore[attr-defined]
            except Exception:
                pass

            suptitle = getattr(figure, "_suptitle", None)
            if suptitle is not None:
                try:
                    suptitle.set_color(fg)
                except Exception:
                    pass

            for ax in figure.get_axes():  # type: ignore[attr-defined]
                try:
                    ax.set_facecolor(bg)
                except Exception:
                    pass
                for spine in ax.spines.values():
                    try:
                        spine.set_color(border)
                    except Exception:
                        pass
                try:
                    ax.tick_params(colors=fg, which="both")
                except Exception:
                    pass
                for text_attr in ("title", "_left_title", "_right_title"):
                    text_obj = getattr(ax, text_attr, None)
                    if text_obj is not None:
                        try:
                            text_obj.set_color(fg)
                        except Exception:
                            pass
                for axis_attr in ("xaxis", "yaxis"):
                    axis_obj = getattr(ax, axis_attr, None)
                    if axis_obj is None:
                        continue
                    label = getattr(axis_obj, "label", None)
                    if label is not None:
                        try:
                            label.set_color(fg)
                        except Exception:
                            pass
                legend = ax.get_legend()
                if legend is not None:
                    try:
                        for text in legend.get_texts():
                            text.set_color(fg)
                        frame = legend.get_frame()
                        if frame is not None:
                            frame.set_facecolor(bg)
                            frame.set_edgecolor(border)
                    except Exception:
                        pass
        except Exception:  # pragma: no cover - best-effort
            self._logger.debug("_apply_light_theme_to_figure failed", exc_info=True)

    def _on_undo(self) -> None:
        """Undo last state change and refresh spreadsheet."""
        if not self._state.has_data:
            return
        self._state.undo()
        matrix = self._state.data_matrix
        if matrix is not None:
            self._spreadsheet.load_data(
                matrix.data,
                row_labels=matrix.row_labels,
                col_labels=matrix.col_labels,
            )
            self._status_bar.setInfo(_("Undo"))

    def _on_redo(self) -> None:
        """Redo last undone change and refresh spreadsheet."""
        if not self._state.has_data:
            return
        self._state.redo()
        matrix = self._state.data_matrix
        if matrix is not None:
            self._spreadsheet.load_data(
                matrix.data,
                row_labels=matrix.row_labels,
                col_labels=matrix.col_labels,
            )
            self._status_bar.setInfo(_("Redo"))

    def _on_transpose(self) -> None:
        """Transpose the data matrix."""
        if not self._state.has_data:
            return
        try:
            matrix = self._state.data_matrix
            transposed_data = matrix.data.T
            from models.data_matrix import DataMatrix

            new_matrix = DataMatrix(
                transposed_data,
                row_labels=matrix.col_labels,
                col_labels=matrix.row_labels,
            )
            self._state.set_data_matrix(new_matrix)
            self._spreadsheet.load_data(
                transposed_data,
                row_labels=matrix.col_labels,
                col_labels=matrix.row_labels,
            )
            self._status_bar.setInfo(
                _("Transposed: {0} samples x {1} variables").format(new_matrix.n_samples, new_matrix.n_variables)
            )
        except Exception as e:
            QMessageBox.critical(self, _("Transpose Error"), str(e))

    def _on_run_imputation(self) -> None:
        """Open missing value imputation dialog."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            import numpy as np

            from config.imputation import ImputationMethod, impute
            from models.data_matrix import DataMatrix

            data = self._state.data_matrix.data
            nan_mask = np.isnan(data)
            total_nan = int(np.sum(nan_mask))

            if total_nan == 0:
                QMessageBox.information(
                    self, _("No Missing Values"), _("The current dataset contains no missing values.")
                )
                return

            # Analyze missing values
            rows_with_nan = int(np.any(nan_mask, axis=1).sum())
            cols_with_nan = int(np.any(nan_mask, axis=0).sum())
            nan_by_row = np.sum(nan_mask, axis=1)
            nan_by_col = np.sum(nan_mask, axis=0)

            # Show dialog
            dialog = ImputationDialog(
                self,
                nan_count=total_nan,
                rows_with_nan=rows_with_nan,
                cols_with_nan=cols_with_nan,
                nan_by_row=nan_by_row,
                nan_by_col=nan_by_col,
                n_rows=data.shape[0],
                n_cols=data.shape[1],
                nan_proportion=total_nan / data.size,
            )
            dialog.setDarkTheme(self._is_dark_theme)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                params = dialog.get_parameters()
                method_map = {
                    "mean": ImputationMethod.MEAN,
                    "median": ImputationMethod.MEDIAN,
                    "knn": ImputationMethod.KNN,
                    "remove_rows": ImputationMethod.REMOVE_ROWS,
                    "remove_columns": ImputationMethod.REMOVE_COLUMNS,
                }
                method = method_map.get(params.get("method", "mean"), ImputationMethod.MEAN)
                k = params.get("k", 5)

                # Apply imputation
                result = impute(data, method, k=k)

                # Update state
                row_labels = self._state.data_matrix.row_labels
                col_labels = self._state.data_matrix.col_labels

                new_matrix = DataMatrix(
                    result.data,
                    row_labels=row_labels,
                    col_labels=col_labels,
                )
                self._state.set_data_matrix(new_matrix)
                self._spreadsheet.load_data(
                    result.data,
                    row_labels=row_labels,
                    col_labels=col_labels,
                )

                self._status_bar.setInfo(result.summary)
                QMessageBox.information(self, _("Imputation Complete"), result.summary)

        except Exception as e:
            QMessageBox.critical(self, _("Imputation Error"), format_user_error(e, "缺失值处理"))

    def _on_preferences(self) -> None:
        """Show application preferences."""
        QMessageBox.information(
            self,
            _("Preferences"),
            _("Application settings can be configured via the Settings menu."),
        )

    def _on_import_data(self) -> None:
        """Show import data dialog with conflict checking."""
        # Check if we need to confirm overwrite
        if self._state.has_data:
            reply = QMessageBox.question(
                self,
                _("Overwrite Data?"),
                _("You already have data loaded. Do you want to replace it?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        dialog = ImportDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
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

        # ``ScientificSpreadsheet.load_data`` only accepts the explicit
        # ``row_labels`` / ``col_labels`` parameters, not arbitrary
        # kwargs, so pass them positionally here. The previous
        # implementation used ``**metadata`` which worked only by
        # accident and would have broken the moment a new key was
        # added to ``metadata``.
        self._spreadsheet.load_data(
            data,
            row_labels=row_labels,
            col_labels=col_labels,
        )
        self._workspace.setCurrentIndex(self._spreadsheet_index)

        # Update UI state now that we have data
        self._update_ui_state()

        # Show success message
        n_samples, n_vars = data.shape
        self._status_bar.setInfo(_("Data imported: {0} samples x {1} variables").format(n_samples, n_vars))

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
                col_labels=self._state.data_matrix.col_labels,
            )
            self._state.set_data_matrix(matrix)

            # Update spreadsheet
            self._spreadsheet.load_data(
                transformed,
                row_labels=self._state.data_matrix.row_labels,
                col_labels=self._state.data_matrix.col_labels,
            )

            self._status_bar.setInfo(_("{0} transformation applied").format(name))

        except Exception as e:
            QMessageBox.critical(self, _("Transformation Error"), format_user_error(e, name))

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

        filepath, _ext = QFileDialog.getSaveFileName(self, _("Export Data"), "", _("CSV Files (*.csv);;All Files (*)"))

        if filepath:
            try:
                self._data_controller.export_csv(filepath)
                QMessageBox.information(
                    self,
                    _("Export Successful"),
                    _("Data successfully exported to {0}").format(os.path.basename(filepath)),
                )
                self._logger.info(f"Data exported to {filepath}")
            except Exception as e:
                self._logger.error(f"Export failed: {e}")
                QMessageBox.critical(self, _("Export Error"), str(e))

    def _cleanup_plot_widgets(self) -> None:
        """Remove all plot canvases from the workspace to prevent memory leaks.

        Handles both the legacy :class:`InteractivePlotCanvas` instances
        and any "cleanable" workspace host widget. The latter is
        identified by the dynamic ``workspace_cleanable`` Qt property
        (set on the container by :meth:`_embed_figure_in_workspace` and
        :meth:`_show_uaz_tree`). The spreadsheet is always preserved.
        """
        stack = self._workspace._stack
        for i in range(stack.count() - 1, -1, -1):
            widget = stack.widget(i)
            if widget is self._spreadsheet:
                continue
            is_interactive_plot = isinstance(widget, InteractivePlotCanvas)
            # Read the new property name with backward-compat fallback.
            is_cleanable = False
            if widget is not None:
                marker = widget.property("workspace_cleanable")
                if marker is not None and bool(marker):
                    is_cleanable = True
                # Backward-compatibility: legacy property name.
                elif widget.property("figure_canvas") is not None:
                    is_cleanable = True
            if is_interactive_plot or is_cleanable:
                stack.removeWidget(widget)
                widget.deleteLater()

    def _add_plot_to_workspace(self, plot: InteractivePlotCanvas, name: str) -> int:
        """Remove old plots and add a new plot to the workspace."""
        self._cleanup_plot_widgets()
        return self._workspace.addWidget(plot, name)

    def _embed_figure_in_workspace(
        self,
        figure: object,
        name: str,
        dark_theme: bool = False,
    ) -> int | None:
        """Embed a pre-built matplotlib Figure into the workspace.

        The :class:`matplotlib.figure.Figure` object returned by the
        analysis plotters can be hosted inside the workspace as a
        stand-alone ``QWidget`` wrapping a ``FigureCanvasQTAgg``. This
        keeps the new industrial-grade plot routines (stratigraphic
        correlation, paleo-environmental CA, etc.) fully integrated
        without forcing them to depend on the ``InteractivePlotCanvas``
        template.

        Parameters:
            figure: Matplotlib Figure returned by the plotter.
            name: Workspace tab name.
            dark_theme: Whether to apply the dark-theme background.

        Returns:
            The workspace index of the newly added widget, or ``None``
            if ``figure`` is ``None``.
        """
        if figure is None:
            return None

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from config.design_system import get_palette

        container = QWidget()
        container.setObjectName("FigureHostWidget")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        canvas = FigureCanvasQTAgg(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if dark_theme:
            self._apply_dark_theme_to_figure(figure)
        layout.addWidget(canvas)
        # Tagged so :meth:`_cleanup_plot_widgets` can garbage-collect
        # this container on the next analysis run.
        container.setProperty("workspace_cleanable", True)
        # Keep a reference to the canvas as a *separate* dynamic
        # property so external callers can still introspect it if needed
        # (e.g. to call ``draw_idle`` from elsewhere).
        container.setProperty("figure_canvas", canvas)

        self._cleanup_plot_widgets()
        idx = self._workspace.addWidget(container, name)
        self._workspace.setCurrentIndex(idx)
        try:
            canvas.draw_idle()
        except Exception:  # pragma: no cover - best-effort UI update
            self._logger.debug("canvas.draw_idle() failed", exc_info=True)
        return idx

    def _apply_dark_theme_to_figure(self, figure: object) -> None:
        """Apply a uniform dark-theme palette to a Matplotlib figure.

        Setting only ``figure.patch.set_facecolor`` leaves the axes
        background, tick colour, spine colour and text colour at the
        Matplotlib default (white / black), which looks broken under
        dark mode. This helper iterates every axes child and brings
        them in line with the design system palette.

        Robust against:
          - Colorbar axes (whose ``xaxis.label`` / ``yaxis.label`` may
            be empty ``Text`` objects without a meaningful color).
          - Figures with a ``suptitle`` that also needs re-colouring.
          - Texts/legend frames that are absent.
        """
        try:
            from config.design_system import get_palette

            palette = get_palette(True)
            bg = palette.bg_primary
            fg = palette.text_primary
            border = palette.border_medium
            figure.patch.set_facecolor(bg)  # type: ignore[attr-defined]

            # Re-colour the figure-level suptitle, if any. Matplotlib
            # stores it on the private ``_suptitle`` attribute when
            # ``Figure.suptitle()`` has been called.
            suptitle = getattr(figure, "_suptitle", None)
            if suptitle is not None:
                try:
                    suptitle.set_color(fg)
                except Exception:
                    pass

            for ax in figure.get_axes():  # type: ignore[attr-defined]
                try:
                    ax.set_facecolor(bg)
                except Exception:
                    pass
                for spine in ax.spines.values():
                    try:
                        spine.set_color(border)
                    except Exception:
                        pass
                try:
                    ax.tick_params(colors=fg, which="both")
                except Exception:
                    pass
                for text_attr in ("title", "_left_title", "_right_title"):
                    text_obj = getattr(ax, text_attr, None)
                    if text_obj is not None:
                        try:
                            text_obj.set_color(fg)
                        except Exception:
                            pass
                for axis_attr in ("xaxis", "yaxis"):
                    axis_obj = getattr(ax, axis_attr, None)
                    if axis_obj is None:
                        continue
                    label = getattr(axis_obj, "label", None)
                    if label is not None:
                        try:
                            label.set_color(fg)
                        except Exception:
                            pass
                legend = ax.get_legend()
                if legend is not None:
                    try:
                        for text in legend.get_texts():
                            text.set_color(fg)
                        frame = legend.get_frame()
                        if frame is not None:
                            frame.set_facecolor(bg)
                            frame.set_edgecolor(border)
                    except Exception:
                        pass
        except Exception:  # pragma: no cover - best-effort UI hint
            self._logger.debug("_apply_dark_theme_to_figure failed", exc_info=True)

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
        dialog.setDarkTheme(self._is_dark_theme)

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

                plot_index = self._add_plot_to_workspace(plot, _("PCA Plot"))
                self._workspace.setCurrentIndex(plot_index)

                ev = result.explained_variance
                cum2 = ev[0] + ev[1] if len(ev) >= 2 else ev[0] if len(ev) == 1 else 0.0
                self._status_bar.setInfo(
                    _("PCA: {0} components, PC1+PC2 = {1:.1f}%").format(
                        result.n_components, cum2
                    )
                )

            except Exception as e:
                QMessageBox.critical(self, _("PCA Error"), format_user_error(e, "PCA"))

    def _on_run_pcoa(self) -> None:
        """Run Principal Coordinate Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = PCoADialog(self)
        dialog.setDarkTheme(self._is_dark_theme)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                result = self._statistics_controller.run_pcoa(
                    metric=params["metric"], n_components=params["n_components"]
                )

                plot = InteractivePlotCanvas()
                plot.plot_pcoa_scores(result)

                plot_index = self._add_plot_to_workspace(plot, _("PCoA Plot"))
                self._workspace.setCurrentIndex(plot_index)

            except Exception as e:
                QMessageBox.critical(self, _("PCoA Error"), format_user_error(e, "PCoA"))

    def _on_run_nmds(self) -> None:
        """Run Non-metric MDS."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = NMDSOptionsDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                result = self._statistics_controller.run_nmds(
                    metric=params["metric"],
                    n_dimensions=params["n_dimensions"],
                    n_restarts=params["n_restarts"],
                    max_iterations=params["max_iterations"],
                    tolerance=params["tolerance"],
                )

                plot = InteractivePlotCanvas()
                plot.plot_nmds(result)

                plot_index = self._add_plot_to_workspace(plot, _("NMDS Plot"))
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(_("NMDS: stress = {0:.4f}").format(result.stress))

            except Exception as e:
                QMessageBox.critical(self, _("NMDS Error"), format_user_error(e, "NMDS"))

    def _on_run_diversity(self) -> None:
        """Run diversity analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = DiversityDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                sample_name = params.get("sample_name", "").strip() or "Sample 1"
                # Resolve the sample-name to a row index. Previously the
                # controller always used ``data[0]`` and only the
                # ``sample_name`` field was used as a label, which gave
                # misleading results when the user typed any other name.
                matrix = self._state.data_matrix
                sample_index = self._resolve_sample_index(sample_name, matrix)
                if sample_index is None:
                    QMessageBox.warning(
                        self,
                        _("Sample Not Found"),
                        _("No sample named '{0}' is loaded.").format(sample_name),
                    )
                    return
                result = self._statistics_controller.analyze_diversity(
                    abundances=matrix.data[sample_index],
                    sample_name=sample_name,
                )

                plot = InteractivePlotCanvas()
                plot.plot_diversity_summary(result)

                plot_index = self._add_plot_to_workspace(plot, _("Diversity Plot"))
                self._workspace.setCurrentIndex(plot_index)

            except Exception as e:
                QMessageBox.critical(self, _("Diversity Error"), format_user_error(e, "多样性分析"))

    def _on_run_rarefaction(self) -> None:
        """Run rarefaction analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        sample_names = self._state.data_matrix.row_labels if self._state.data_matrix else []
        dialog = RarefactionDialog(self, sample_names=sample_names)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()

            try:
                selected_samples = params.get("samples", [])
                if not selected_samples:
                    QMessageBox.information(
                        self,
                        _("No Selection"),
                        _("Please select at least one sample to rarefy."),
                    )
                    return
                max_n = params.get("max_n", 100)
                step = params.get("step", 5)
                n_points = max(10, max_n // step) if step > 0 else 50
                # Resolve the first selected sample to its row so the
                # analysis actually reflects the user's choice.
                matrix = self._state.data_matrix
                sample_index = self._resolve_sample_index(selected_samples[0], matrix)
                if sample_index is None:
                    QMessageBox.warning(
                        self,
                        _("Sample Not Found"),
                        _("No sample named '{0}' is loaded.").format(selected_samples[0]),
                    )
                    return
                result = self._statistics_controller.analyze_rarefaction(
                    abundances=matrix.data[sample_index],
                    sample_name=selected_samples[0],
                    n_points=n_points,
                )

                plot = InteractivePlotCanvas()
                plot.plot_rarefaction(result)

                plot_index = self._add_plot_to_workspace(plot, _("Rarefaction Plot"))
                self._workspace.setCurrentIndex(plot_index)

            except Exception as e:
                QMessageBox.critical(self, _("Rarefaction Error"), format_user_error(e, "稀疏化分析"))

    @staticmethod
    def _resolve_sample_index(name: str, matrix) -> int | None:
        """Resolve a sample identifier (label, 1-based, or 0-based index).

        The Diversity / Rarefaction dialogs let the user type an
        arbitrary sample label, but the controller historically only
        used ``data[0]``. This helper makes the dispatch explicit:

            1. If ``name`` is a row label, return its index.
            2. If ``name`` parses as an integer, return that index
               (1-based indices like "1", "2" are accepted for
               ergonomic reasons).
            3. Otherwise fall back to row 0 to keep the analysis
               runnable rather than silently failing.

        Returns ``None`` only when the matrix is empty.
        """
        if matrix is None or matrix.n_samples == 0:
            return None
        labels = list(matrix.row_labels)
        if name in labels:
            return labels.index(name)
        try:
            idx = int(name)
            if 1 <= idx <= matrix.n_samples:
                return idx - 1
            if 0 <= idx < matrix.n_samples:
                return idx
        except (ValueError, TypeError):
            pass
        return 0

    def _on_run_spectral(self) -> None:
        """Run spectral analysis (power spectrum and periodogram analysis)."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_spectral(data=self._state.data_matrix.data)
            plot = InteractivePlotCanvas()
            plot.plot_spectral(result)
            plot_index = self._add_plot_to_workspace(plot, _("Spectral Analysis"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("Spectral analysis completed"))
        except Exception as e:
            self._logger.error(f"Spectral analysis failed: {e}")
            QMessageBox.critical(self, _("Spectral Analysis Error"), format_user_error(e, "频谱分析"))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_anosim(self) -> None:
        """Run Analysis of Similarity (ANOSIM) test."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        groups = self._get_groups()
        if groups is None:
            QMessageBox.warning(
                self,
                _("No Groups"),
                _(
                    "Please define groups in the spreadsheet before running ANOSIM.\n"
                    "Use the Group column to assign samples to groups."
                ),
            )
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_anosim(
                data=self._state.data_matrix.data,
                groups=groups,
            )
            plot = InteractivePlotCanvas()
            plot.plot_anosim_results(result)
            plot_index = self._add_plot_to_workspace(plot, _("ANOSIM Results"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("ANOSIM analysis completed"))
        except Exception as e:
            self._logger.error(f"ANOSIM analysis failed: {e}")
            QMessageBox.critical(self, _("ANOSIM Error"), format_user_error(e, "ANOSIM"))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_permanova(self) -> None:
        """Run Permutational Multivariate Analysis of Variance (PERMANOVA) test."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        groups = self._get_groups()
        if groups is None:
            QMessageBox.warning(
                self,
                _("No Groups"),
                _(
                    "Please define groups in the spreadsheet before running PERMANOVA.\n"
                    "Use the Group column to assign samples to groups."
                ),
            )
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_permanova(
                data=self._state.data_matrix.data,
                groups=groups,
            )
            plot = InteractivePlotCanvas()
            plot.plot_permanova_results(result)
            plot_index = self._add_plot_to_workspace(plot, _("PERMANOVA Results"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("PERMANOVA analysis completed"))
        except Exception as e:
            self._logger.error(f"PERMANOVA analysis failed: {e}")
            QMessageBox.critical(self, _("PERMANOVA Error"), format_user_error(e, "PERMANOVA"))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_simper(self) -> None:
        """Run SIMPER analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        groups = self._get_groups()
        if groups is None:
            QMessageBox.warning(
                self,
                _("No Groups"),
                _(
                    "Please define groups in the spreadsheet before running SIMPER.\n"
                    "Use the Group column to assign samples to groups."
                ),
            )
            return

        dialog = SimperDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_simper(
                    data=self._state.data_matrix.data,
                    groups=groups,
                    metric=params.get("metric", "bray_curtis"),
                )
                plot = InteractivePlotCanvas()
                plot.plot_simper_results(result)
                plot_index = self._add_plot_to_workspace(plot, "SIMPER")
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo("SIMPER analysis completed")
            except Exception as e:
                QMessageBox.critical(self, _("SIMPER Error"), format_user_error(e, "SIMPER"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_univariate_selection_changed(self, index: int) -> None:
        """Handle univariate dropdown selection."""
        dialog = UnivariateDialog(self)
        dialog.set_pre_selected_test(index)
        dialog.setDarkTheme(self._is_dark_theme)
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
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: W={r.shapiro_stat:.4f}, p={r.shapiro_p:.4f} {'*' if r.is_normal_shapiro else 'ns'}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("Normality Test"), "\n".join(lines))
                elif test_type == 2:  # t-test
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_t_test(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: t={r.statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("t-test Results"), "\n".join(lines))
                elif test_type == 3:  # ANOVA
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_anova(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: F={r.f_statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("ANOVA Results"), "\n".join(lines))
                elif test_type == 4:  # Kruskal-Wallis
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_kruskal_wallis(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: H={r.statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("Kruskal-Wallis Results"), "\n".join(lines))

                self._status_bar.setInfo(_("Univariate analysis completed"))
            except Exception as e:
                QMessageBox.critical(self, _("Univariate Error"), format_user_error(e, "单变量统计"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_univariate(self) -> None:
        """Run univariate statistics."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        dialog = UnivariateDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
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
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: W={r.shapiro_stat:.4f}, p={r.shapiro_p:.4f} {'*' if r.is_normal_shapiro else 'ns'}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("Normality Test"), "\n".join(lines))
                elif test_type == 2:  # t-test
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_t_test(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: t={r.statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("t-test Results"), "\n".join(lines))
                elif test_type == 3:  # ANOVA
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_anova(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: F={r.f_statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("ANOVA Results"), "\n".join(lines))
                elif test_type == 4:  # Kruskal-Wallis
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_kruskal_wallis(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: H={r.statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("Kruskal-Wallis Results"), "\n".join(lines))

                self._status_bar.setInfo(_("Univariate analysis completed"))
            except Exception as e:
                QMessageBox.critical(self, _("Univariate Error"), format_user_error(e, "单变量统计"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_univariate_by_index(self, index: int) -> None:
        """Run univariate analysis by test index (0=Summary, 1=Normality, 2=t-test, 3=ANOVA, 4=Kruskal-Wallis)."""
        dialog = UnivariateDialog(self)
        dialog.set_pre_selected_test(index)
        dialog.setDarkTheme(self._is_dark_theme)
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
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: W={r.shapiro_stat:.4f}, p={r.shapiro_p:.4f} {'*' if r.is_normal_shapiro else 'ns'}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("Normality Test"), "\n".join(lines))
                elif test_type == 2:  # t-test
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_t_test(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: t={r.statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("t-test Results"), "\n".join(lines))
                elif test_type == 3:  # ANOVA
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_anova(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: F={r.f_statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("ANOVA Results"), "\n".join(lines))
                elif test_type == 4:  # Kruskal-Wallis
                    groups = self._get_groups()
                    results = self._statistics_controller.analyze_kruskal_wallis(data, groups=groups)
                    lines = [
                        f"{col_names[i] if i < len(col_names) else f'Var{i}'}: H={r.statistic:.4f}, p={r.p_value:.4f}"
                        for i, r in enumerate(results)
                    ]
                    QMessageBox.information(self, _("Kruskal-Wallis Results"), "\n".join(lines))

                self._status_bar.setInfo(_("Univariate analysis completed"))
            except Exception as e:
                QMessageBox.critical(self, _("Univariate Error"), format_user_error(e, "单变量统计"))
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
                self,
                _("No Groups"),
                _("LDA requires group assignments. Please set row groups first via the spreadsheet metadata."),
            )
            return

        dialog = LDADialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                self._logger.info(
                    f"Running LDA with {len(set(groups))} groups, n_components={params.get('n_components')}"
                )
                result = self._statistics_controller.analyze_lda(
                    data=self._state.data_matrix.data,
                    groups=groups,
                    n_components=params.get("n_components"),
                )
                plot = InteractivePlotCanvas()
                plot.plot_lda_scores(result)
                plot_index = self._add_plot_to_workspace(plot, "LDA")
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(_("LDA analysis completed"))
                self._logger.info(f"LDA completed: accuracy={result.accuracy:.4f}, {result.n_classes} classes")
            except Exception as e:
                self._logger.error(f"LDA analysis failed: {e}")
                QMessageBox.critical(self, _("LDA Error"), format_user_error(e, "LDA"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_cca(self) -> None:
        """Run Canonical Correspondence Analysis (CCA) or Redundancy Analysis (RDA)."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        data = self._state.data_matrix.data
        col_labels = self._state.data_matrix.col_labels

        # Get environmental columns (user selects from dialog)
        dialog = CCADialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        # Provide column choices: first half as species, second half as env
        mid = max(1, data.shape[1] // 2)
        species_cols = col_labels[:mid] if mid < len(col_labels) else col_labels
        env_cols = col_labels[mid : mid + min(mid, len(col_labels) - mid)] if mid < len(col_labels) else []
        dialog.set_column_names(species_cols, env_cols)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                # Get selected env column indices
                selected_env = params.get("env_columns", [])
                env_indices = [i for i, c in enumerate(col_labels) if c in selected_env]

                if not env_indices:
                    QMessageBox.warning(
                        self, _("No Selection"), _("Please select at least one environmental variable.")
                    )
                    return

                # Split data into species (Y) and environmental (X) matrices
                Y = data[:, :mid] if mid < data.shape[1] else data
                X = data[:, env_indices]

                self._logger.info(
                    f"Running CCA/RDA: Y.shape={Y.shape}, X.shape={X.shape}, method={params.get('method')}"
                )

                result = self._statistics_controller.run_cca(
                    Y=Y,
                    X=X,
                    n_components=params.get("n_components"),
                    method=params.get("method"),
                )

                plot = InteractivePlotCanvas()
                plot.plot_cca_triplot(result)

                plot_index = self._add_plot_to_workspace(
                    plot, _("{0} Triplot").format(params.get("method", "cca").upper())
                )
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(
                    _("{0}: {1:.1f}% constrained variance").format(
                        params.get("method", "cca").upper(), result.constrained_variance
                    )
                )
                self._logger.info(f"CCA/RDA completed: constrained_variance={result.constrained_variance:.2f}%")

            except Exception as e:
                self._logger.error(f"CCA/RDA analysis failed: {e}")
                QMessageBox.critical(
                    self,
                    _("{0} Error").format(params.get("method", "CCA").upper()),
                    format_user_error(e, params.get("method", "CCA").upper()),
                )
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_clustering(self) -> None:
        """Run hierarchical clustering."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = ClusteringDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
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
                plot_index = self._add_plot_to_workspace(plot, _("Clustering"))
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(
                    _("Clustering: {0} clusters, cophenetic r={1:.3f}").format(
                        result.n_clusters, result.cophenetic_corr
                    )
                )
            except Exception as e:
                QMessageBox.critical(self, _("Clustering Error"), format_user_error(e, "聚类分析"))
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
            plot_index = self._add_plot_to_workspace(plot, _("Abundance Models"))
            self._workspace.setCurrentIndex(plot_index)
            [f"{fit.model_name}: R²={fit.r_squared:.4f}, AIC={fit.aic:.2f}" for fit in results.values()]
            self._status_bar.setInfo(_("Abundance models fitted"))
        except Exception as e:
            QMessageBox.critical(self, _("Abundance Models Error"), format_user_error(e, "丰度模型"))
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
            plot_index = self._add_plot_to_workspace(plot, "SHE")
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("SHE analysis completed"))
        except Exception as e:
            QMessageBox.critical(self, _("SHE Error"), format_user_error(e, "SHE分析"))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_coniss(self) -> None:
        """Run CONISS zonation."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = CONISSDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_coniss(
                    data=self._state.data_matrix.data,
                    n_zones=params.get("n_zones", 4),
                )
                QMessageBox.information(self, "CONISS", result.summary())
                self._status_bar.setInfo(_("CONISS: {0} zones").format(result.n_zones))
            except Exception as e:
                QMessageBox.critical(self, _("CONISS Error"), format_user_error(e, "CONISS"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_markov(self) -> None:
        """Run Markov chain analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = MarkovDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_markov()
                QMessageBox.information(self, _("Markov Chain Analysis"), result.summary())
                self._status_bar.setInfo(_("Markov analysis completed"))
            except Exception as e:
                QMessageBox.critical(self, _("Markov Error"), format_user_error(e, "马尔可夫链"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_directional(self) -> None:
        """Run directional statistics."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = DirectionalDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                result = self._statistics_controller.analyze_directional()
                bin_edges, counts = self._statistics_controller.bin_rose_diagram(n_bins=params.get("n_bins", 12))
                plot = InteractivePlotCanvas()
                plot.plot_rose_diagram(bin_edges, counts, result.mean_direction_deg)
                plot_index = self._add_plot_to_workspace(plot, _("Rose Diagram"))
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(
                    _("Directional: mean={0:.1f}°, Rayleigh p={1:.4f}").format(
                        result.mean_direction_deg, result.rayleigh_p
                    )
                )
            except Exception as e:
                QMessageBox.critical(self, _("Directional Error"), format_user_error(e, "方向统计"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_efa(self) -> None:
        """Run Elliptic Fourier Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = EFADialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
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
                plot.plot_efa_contours(result.original, result.reconstructed, f"EFA ({result.n_harmonics} harmonics)")
                plot_index = self._add_plot_to_workspace(plot, "EFA")
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(
                    _("EFA: {0} harmonics, {1} points").format(result.n_harmonics, result.n_points)
                )
            except Exception as e:
                QMessageBox.critical(self, _("EFA Error"), format_user_error(e, "EFA"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_isotope(self) -> None:
        """Run Isotope Time Series Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = IsotopeAnalysisDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            self._status_bar.setProgress(0, 0)
            try:
                import numpy as np

                from stratigraphy.isotope_analysis import IsotopeAnalyzer, IsotopeData

                # Create IsotopeData from loaded data
                data = self._state.data_matrix.data
                if data.shape[1] < 3:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _("Need at least 3 columns: depth, age, and isotope values"),
                    )
                    return

                # Build IsotopeData from loaded data
                # Column 0: depth, Column 1: age, Column 2+: isotope values
                iso_kwargs = {
                    "depth": data[:, 0],
                    "age": data[:, 1],
                }

                # Map additional columns to isotope types (only if column has valid data)
                isotope_names = ["d13C", "d18O", "sr", "nd"]
                for i, name in enumerate(isotope_names):
                    col_idx = i + 2
                    if col_idx < data.shape[1]:
                        col_data = data[:, col_idx]
                        # Only include if column has at least some non-NaN values
                        if not np.all(np.isnan(col_data)):
                            iso_kwargs[name] = col_data

                # Check we have at least one isotope
                if len(iso_kwargs) <= 2:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _("Need at least one isotope column with valid data"),
                    )
                    return

                iso_data = IsotopeData(**iso_kwargs)

                analyzer = IsotopeAnalyzer()
                result = analyzer.analyze(
                    iso_data,
                    detect_excursions=params.get("detect_excursions", True),
                    excursion_threshold=params.get("excursion_threshold", 2.0),
                    excursion_min_duration=params.get("excursion_min_duration", 2),
                    compute_correlations=params.get("compute_correlations", True),
                )

                # Show summary
                self._status_bar.setInfo(_("Isotope: {0} excursions detected").format(len(result.excursions)))
                QMessageBox.information(
                    self,
                    _("Analysis Complete"),
                    result.summary(),
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    _("Isotope Error"),
                    format_user_error(e, "同位素分析"),
                )
            finally:
                # ``setProgress(100, 100)`` is intentionally inside the
                # ``finally`` so that an early ``return`` (e.g. the
                # ``Insufficient Data`` warning above) does not leave
                # the progress bar stuck at 0.
                self._status_bar.setProgress(100, 100)

    def _on_run_stratigraphic(self) -> None:
        """Run Stratigraphic Correlation Analysis.

        Builds the multi-section correlation analysis and, when
        ``render_plot=True`` in the dialog, also produces a
        publication-quality warping-path diagram via
        :meth:`StratigraphyPlotter.plot_stratigraphic_correlation`.
        """
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = StratigraphicCorrelationDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        col_labels = []
        try:
            col_labels = list(self._state.data_matrix.col_labels or [])
        except AttributeError:
            col_labels = []
        if col_labels:
            dialog.set_column_labels(col_labels)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                import numpy as np

                from stratigraphy.correlation import (
                    StratigraphicCorrelationAnalyzer,
                    StratigraphicSection,
                )

                # Create sections from loaded data
                data = self._state.data_matrix.data
                n_rows = len(data)

                if data.ndim != 2 or data.shape[1] < 1:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _("Need at least 1 numeric column to build a section."),
                    )
                    return
                if n_rows < 2:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _("Need at least 2 stratigraphic samples per section."),
                    )
                    return

                # Honour the user-selected height column index. When the
                # user did not pick anything, fall back to column 0.
                height_col = int(params.get("height_column", 0)) % data.shape[1]

                # Section construction strategy
                # ---------------------------------------------------------
                # Three layouts are supported:
                #   (1) shape[1] >= 4: classical (height, thickness) pairs,
                #       laid out columnwise as h1, t1, h2, t2, ...
                #   (2) shape[1] in (2, 3): one height column + the rest
                #       interpreted as per-section *proxy signals* (lithology
                #       index, abundance, isotope value...). Each proxy
                #       column becomes one StratigraphicSection that shares
                #       the height axis but carries its own signal in
                #       ``heights`` so the DTW compares the signals, not
                #       the identical height array.
                #   (3) shape[1] == 1: degenerate case - we replicate the
                #       column twice so the analyser still produces a
                #       trivial 1.0 self-similarity, instead of refusing.
                # ---------------------------------------------------------
                sections: list[StratigraphicSection] = []
                base_heights = np.asarray(data[:, height_col], dtype=np.float64)
                # Pre-compute per-row thicknesses from the height column;
                # all sections share the same vertical sampling grid.
                if base_heights.size > 1:
                    base_t = np.diff(base_heights)
                    base_t = np.append(base_t, base_t[-1] if base_t.size > 0 else 1.0)
                else:
                    base_t = np.array([1.0], dtype=np.float64)

                if data.shape[1] >= 4:
                    section_count = data.shape[1] // 2
                    for i in range(section_count):
                        h_col = i * 2
                        t_col = i * 2 + 1
                        if t_col >= data.shape[1]:
                            continue
                        h = np.asarray(data[:, h_col], dtype=np.float64)
                        t = np.asarray(data[:, t_col], dtype=np.float64)
                        t_diff = np.diff(t)
                        t_diff = np.append(
                            t_diff, t_diff[-1] if t_diff.size > 0 else 1.0
                        )
                        sec_name = (
                            col_labels[h_col]
                            if 0 <= h_col < len(col_labels)
                            else _("Section {0}").format(i + 1)
                        )
                        sections.append(
                            StratigraphicSection(
                                name=str(sec_name),
                                heights=h,
                                thicknesses=t_diff,
                                lithologies=["layer"] * len(h),
                            )
                        )
                elif data.shape[1] >= 2:
                    # Layout (2): build one section per non-height column,
                    # putting the column's values into ``heights`` so the
                    # downstream DTW actually compares signals rather than
                    # comparing the identical height axis to itself.
                    proxy_cols = [c for c in range(data.shape[1]) if c != height_col]
                    for pc in proxy_cols:
                        signal = np.asarray(data[:, pc], dtype=np.float64)
                        sec_name = (
                            col_labels[pc]
                            if 0 <= pc < len(col_labels)
                            else _("Section {0}").format(pc + 1)
                        )
                        sections.append(
                            StratigraphicSection(
                                name=str(sec_name),
                                heights=signal,  # DTW compares these
                                thicknesses=base_t,
                                lithologies=["layer"] * n_rows,
                            )
                        )
                else:
                    # Layout (3): degenerate single column. Replicate to
                    # guarantee >= 2 sections so the analyser does not
                    # refuse the data. The user gets a sim=1.0 trivial
                    # answer and at least sees the column rendered.
                    h = np.asarray(data[:, 0], dtype=np.float64)
                    only_name = (
                        col_labels[0]
                        if 0 < len(col_labels)
                        else _("Section 1")
                    )
                    sections.append(
                        StratigraphicSection(
                            name=str(only_name),
                            heights=h,
                            thicknesses=base_t,
                            lithologies=["layer"] * n_rows,
                        )
                    )
                    sections.append(
                        StratigraphicSection(
                            name=_("{0} (copy)").format(only_name),
                            heights=h.copy(),
                            thicknesses=base_t.copy(),
                            lithologies=["layer"] * n_rows,
                        )
                    )

                analyzer = StratigraphicCorrelationAnalyzer()
                # Forward the user-selected ``max_pairs`` to the analyser
                # so the ``best_matches`` list is long enough for the
                # plotter to honour the same setting. ``-1`` means
                # "all pairs", which we translate into a generous cap
                # of N*(N-1)/2.
                requested_max_pairs = int(params.get("max_pairs", 3))
                n_secs = len(sections)
                total_pairs = max(1, n_secs * (n_secs - 1) // 2)
                if requested_max_pairs == -1:
                    analyser_max_matches = total_pairs
                else:
                    analyser_max_matches = max(1, min(requested_max_pairs, total_pairs))
                result = analyzer.analyze(
                    sections,
                    method=params.get("correlation_method", "dtw"),
                    max_matches=analyser_max_matches,
                )

                # Optionally render the publication-quality warping-path plot
                rendered_figure = None
                if params.get("render_plot", True):
                    try:
                        from visualization.stratigraphy_plot import StratigraphyPlotter

                        plotter = StratigraphyPlotter()
                        rendered_figure = plotter.plot_stratigraphic_correlation(
                            correlation_result=result,
                            title=_("Stratigraphic Correlation (DTW warping paths)"),
                            cmap_name=params.get("cmap_name", "viridis"),
                            max_pairs=int(params.get("max_pairs", 3)),
                        )
                    except Exception as plot_exc:  # pragma: no cover
                        self._logger.warning(
                            "Failed to render stratigraphic correlation plot: %s",
                            plot_exc,
                        )

                if rendered_figure is not None:
                    self._embed_figure_in_workspace(
                        rendered_figure,
                        _("Stratigraphic Correlation"),
                        dark_theme=self._is_dark_theme,
                    )
                else:
                    self._status_bar.setInfo(_("Stratigraphic Correlation: complete"))
                    QMessageBox.information(
                        self, _("Analysis Complete"), result.summary()
                    )

            except Exception as e:
                self._logger.critical(f"Stratigraphic correlation failed: {e}")
                QMessageBox.critical(
                    self,
                    _("Correlation Error"),
                    format_user_error(e, "地层相关性"),
                )
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_paleo_env(self) -> None:
        """Run Paleo-Environmental Reconstruction via Correspondence Analysis.

        Wraps :class:`ecology.paleoenv.PaleoEnvironmentReconstructor` and
        exposes its first-axis reconstruction as both a numeric result
        and an optional height-vs-axis plot.
        """
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = PaleoEnvironmentDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        col_labels = []
        try:
            col_labels = list(self._state.data_matrix.col_labels or [])
        except AttributeError:
            col_labels = []
        if col_labels:
            dialog.set_column_labels(col_labels)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                import numpy as np

                from ecology.paleoenv import PaleoEnvironmentReconstructor

                data = self._state.data_matrix.data
                if data.ndim != 2 or data.shape[1] < 2:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _(
                            "Need at least 2 columns (1 height + at least 1 taxon) "
                            "for paleo-environmental reconstruction."
                        ),
                    )
                    return
                if data.shape[0] < 2:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _(
                            "Need at least 2 stratigraphic samples to perform "
                            "correspondence analysis."
                        ),
                    )
                    return

                height_col = int(params.get("height_column", 0)) % data.shape[1]
                taxon_indices = [
                    int(c)
                    for c in params.get("taxon_columns", [])
                    if 0 <= int(c) < data.shape[1] and int(c) != height_col
                ]
                if len(taxon_indices) < 1:
                    QMessageBox.warning(
                        self,
                        _("No Taxa Selected"),
                        _(
                            "Please select at least one taxon column in the dialog."
                        ),
                    )
                    return

                heights = np.asarray(data[:, height_col], dtype=np.float64)
                abundance = np.asarray(data[:, taxon_indices], dtype=np.float64)

                # Build column labels for the chosen taxa (best-effort).
                # Built *before* validation so error messages can quote
                # the human-readable taxon names rather than raw indices.
                taxon_labels: list[str] = []
                for idx in taxon_indices:
                    if 0 <= idx < len(col_labels):
                        taxon_labels.append(
                            "{0}: {1}".format(idx, col_labels[idx])
                        )
                    else:
                        taxon_labels.append("col_{0}".format(idx))

                # --------------------------------------------------------
                # Pre-validate to give actionable error messages instead
                # of letting ``PaleoEnvironmentReconstructor.reconstruct``
                # raise a generic ``DataValidationError`` that the user
                # has to decode.
                # --------------------------------------------------------
                validation_issues: list[str] = []
                if not np.all(np.isfinite(heights)):
                    bad_rows = np.where(~np.isfinite(heights))[0].tolist()
                    validation_issues.append(
                        _(
                            "Height column contains non-finite values at rows: {0}"
                        ).format(bad_rows[:10])
                    )
                else:
                    h_diff = np.diff(heights)
                    if not (np.all(h_diff > 0) or np.all(h_diff < 0)):
                        validation_issues.append(
                            _(
                                "Heights must be strictly monotonic. Please sort "
                                "the rows by stratigraphic height first."
                            )
                        )
                if not np.all(np.isfinite(abundance)):
                    validation_issues.append(
                        _(
                            "Abundance matrix contains NaN/Inf. Please clean or "
                            "impute the data before running CA."
                        )
                    )
                if np.any(abundance < 0):
                    neg_cnt = int(np.sum(abundance < 0))
                    validation_issues.append(
                        _(
                            "Abundance matrix contains {0} negative cell(s); "
                            "CA requires non-negative input."
                        ).format(neg_cnt)
                    )
                if abundance.size > 0:
                    row_sums = abundance.sum(axis=1)
                    zero_rows = np.where(row_sums <= 0.0)[0]
                    if zero_rows.size > 0:
                        validation_issues.append(
                            _(
                                "Rows with zero total abundance at indices {0}; "
                                "drop these rows or pick more taxa."
                            ).format(zero_rows[:10].tolist())
                        )
                    col_sums = abundance.sum(axis=0)
                    zero_cols = np.where(col_sums <= 0.0)[0]
                    if zero_cols.size > 0:
                        bad_taxa = [taxon_labels[int(c)] for c in zero_cols.tolist()]
                        validation_issues.append(
                            _(
                                "Taxa with zero total abundance: {0}; "
                                "remove them from the selection."
                            ).format(bad_taxa[:10])
                        )

                if validation_issues:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        "\n\n".join(validation_issues),
                    )
                    return

                reconstructor = PaleoEnvironmentReconstructor()
                result = reconstructor.reconstruct(
                    abundance_matrix=abundance,
                    heights=heights,
                    taxon_names=taxon_labels,
                    calibrate_direction=bool(params.get("calibrate_direction", True)),
                )

                # Optionally render a height-vs-CA-axis plot
                rendered_figure = None
                if params.get("render_plot", True):
                    try:
                        from matplotlib.figure import Figure

                        # NOTE: do NOT hard-code a white facecolor here.
                        # Leaving the Figure's default lets
                        # ``_apply_dark_theme_to_figure`` (called from
                        # ``_embed_figure_in_workspace`` when
                        # ``dark_theme=True``) re-paint background and
                        # foreground consistently. A hard-coded white
                        # would override the dark palette and produce a
                        # white-on-dark "blank-card" look.
                        fig = Figure(figsize=(8, 5))
                        ax = fig.add_subplot(111)
                        ax.plot(
                            result.heights,
                            result.row_species_axis,
                            "o-",
                            color="#1E40AF",
                            linewidth=2.0,
                            markersize=6,
                            label=_("CA axis 1 (reconstructed paleo-env. proxy)"),
                        )
                        ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)
                        ax.set_xlabel(_("Stratigraphic height"))
                        ax.set_ylabel(_("CA axis 1 score"))
                        was_flipped_txt = _("yes") if result.was_flipped else _("no")
                        title = _(
                            "Paleo-Environmental CA Reconstruction\n"
                            "Inertia explained = {0:.2%} | "
                            "r(axis, height) = {1:+.3f} | "
                            "Auto-flip = {2}"
                        ).format(
                            result.explained_inertia,
                            result.pearson_corr_axis_vs_height,
                            was_flipped_txt,
                        )
                        ax.set_title(title, fontsize=11)
                        ax.grid(True, linestyle=":", alpha=0.4)
                        ax.legend(loc="best", fontsize=9)
                        fig.tight_layout()
                        rendered_figure = fig
                    except Exception as plot_exc:  # pragma: no cover
                        self._logger.warning(
                            "Failed to render paleo-env plot: %s", plot_exc
                        )

                if rendered_figure is not None:
                    self._embed_figure_in_workspace(
                        rendered_figure,
                        _("Paleo-Environmental CA Axis"),
                        dark_theme=self._is_dark_theme,
                    )

                # Always show the textual summary
                self._status_bar.setInfo(
                    _("Paleo-Env. CA: inertia={0:.2%}, r={1:+.3f}").format(
                        result.explained_inertia,
                        result.pearson_corr_axis_vs_height,
                    )
                )
                QMessageBox.information(
                    self,
                    _("Paleo-Environment Complete"),
                    result.summary(),
                )

            except Exception as e:
                self._logger.critical(f"Paleo-environmental reconstruction failed: {e}")
                QMessageBox.critical(
                    self,
                    _("Paleo-Environment Error"),
                    format_user_error(e, "古环境重建"),
                )
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_gpa(self) -> None:
        """Run Generalized Procrustes Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            self._status_bar.setProgress(0, 0)
            result = self._statistics_controller.analyze_gpa(
                data=self._state.data_matrix.data,
            )

            plot = InteractivePlotCanvas()
            # Plot GPA-aligned landmarks
            if hasattr(result, "aligned_configurations"):
                coords = result.aligned_configurations
                mean_shape = coords.mean(axis=0) if hasattr(coords, "mean") else coords
                plot.plot_efa_contours(
                    coords[0] if len(coords.shape) > 2 else coords,
                    mean_shape,
                    title=_("GPA Aligned Landmarks"),
                )

            plot_index = self._add_plot_to_workspace(plot, _("GPA Alignment"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("GPA analysis completed"))
        except Exception as e:
            self._logger.error(f"GPA analysis failed: {e}")
            QMessageBox.critical(self, _("GPA Error"), format_user_error(e, "GPA"))
        finally:
            self._status_bar.setProgress(100, 100)

    def _on_run_tps_grid(self) -> None:
        """Run TPS Deformation Grid visualization."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = TPSGridDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                # Get GPA result from cache for TPS visualization
                tps_result = self._state.get_cached_result("gpa_result")

                if tps_result is None:
                    QMessageBox.information(
                        self,
                        _("No TPS Result"),
                        _("Please run GPA (Generalized Procrustes Analysis) first to compute TPS deformation."),
                    )
                    return

                plot = InteractivePlotCanvas()
                plot.plot_tps_deformation_grid(
                    tps_result,
                    grid_shape=(params.get("grid_rows", 15), params.get("grid_cols", 15)),
                    show_vectors=params.get("show_vectors", True),
                )

                plot_index = self._add_plot_to_workspace(plot, _("TPS Deformation Grid"))
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(_("TPS Deformation Grid displayed"))
                self._logger.info("TPS deformation grid displayed")

            except Exception as e:
                self._logger.error(f"TPS grid visualization failed: {e}")
                QMessageBox.critical(self, _("TPS Grid Error"), format_user_error(e, "TPS网格"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_ripley_k(self) -> None:
        """Run Ripley's K spatial point pattern analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = SpatialRipleyKDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                result = self._statistics_controller.analyze_spatial_ripley_k(
                    coords=None,  # Will use first 2 columns
                    r_max=params.get("r_max"),
                    n_r_values=params.get("n_r_values"),
                    n_simulations=params.get("n_simulations"),
                )

                plot = InteractivePlotCanvas()
                plot.plot_ripley_k(result, show_points=params.get("show_points", True))

                plot_index = self._add_plot_to_workspace(plot, _("Ripley's K"))
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(result.interpretation)
                self._logger.info(f"RipleyK completed: {result.interpretation[:50]}")

            except Exception as e:
                self._logger.error(f"Ripley K analysis failed: {e}")
                QMessageBox.critical(self, _("Ripley K Error"), format_user_error(e, "Ripley K"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_wavelet(self) -> None:
        """Run Wavelet CWT analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = WaveletDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                import numpy as np

                data = self._state.data_matrix.data
                # Use first column as time, second as values
                time = data[:, 0]
                values = data[:, 1] if data.shape[1] > 1 else data[:, 0]

                # Import here to avoid circular imports
                from stratigraphy.spectral_analysis import SpectralAnalyzer

                analyzer = SpectralAnalyzer()
                scales = np.arange(params.get("min_scale", 2), params.get("max_scale", 50))
                result = analyzer.wavelet_transform(
                    time,
                    values,
                    wavelet=params.get("wavelet", "morlet"),
                    scales=scales,
                )

                plot = InteractivePlotCanvas()
                plot.plot_wavelet_scalogram(result)

                plot_index = self._add_plot_to_workspace(plot, _("Wavelet CWT"))
                self._workspace.setCurrentIndex(plot_index)

                self._status_bar.setInfo(
                    _("{0} wavelet: peak freq = {1:.4f}").format(result.wavelet, result.peak_frequency)
                )
                self._logger.info(f"Wavelet CWT completed: {result.summary()}")

            except Exception as e:
                self._logger.error(f"Wavelet analysis failed: {e}")
                QMessageBox.critical(self, _("Wavelet Error"), format_user_error(e, "小波分析"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_biostrat(self) -> None:
        """Run UA/RASC Biostratigraphy analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = BiostratigraphyDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)

                import numpy as np

                data = self._state.data_matrix.data
                col_labels = self._state.data_matrix.col_labels

                # FAD/LAD data: first half of columns are FADs, second half are LADs
                mid = data.shape[1] // 2
                if mid < 2:
                    QMessageBox.warning(self, _("Insufficient Data"), _("Need at least 4 columns for FAD/LAD data."))
                    return

                fad_matrix = data[:, :mid]
                lad_matrix = data[:, mid : mid * 2]

                # Get event names from column labels. We need exactly
                # ``mid`` labels for the FAD half; if the col_labels are
                # missing/short we synthesize the rest defensively.
                col_labels_list = list(col_labels or [])
                if len(col_labels_list) >= mid:
                    event_names = col_labels_list[:mid]
                else:
                    event_names = col_labels_list + [
                        f"Event_{i + 1}" for i in range(len(col_labels_list), mid)
                    ]

                method = params.get("method", "ua")

                if method == "ua":
                    from stratigraphy.biostratigraphy import UAAnalyzer

                    analyzer = UAAnalyzer()
                    result = analyzer.analyze(
                        fad_matrix,
                        lad_matrix,
                        event_names=event_names,
                        min_section_occurrence=params.get("min_section_occurrence", 2),
                        uaz_similarity_threshold=params.get("uaz_similarity_threshold", 0.8),
                        enable_cyclic_check=bool(
                            params.get("enable_cyclic_check", True)
                        ),
                    )
                    # Surface detected cyclic contradictions prominently.
                    cyclic = result.cyclic_contradictions or []
                    if cyclic:
                        lines = [_("Detected {0} cyclic FAD contradiction(s):").format(len(cyclic))]
                        for entry in cyclic[:5]:
                            lines.append(
                                "  - {0} ↔ {1}  (a→b in {2}, b→a in {3})".format(
                                    entry.get("event_a", "?"),
                                    entry.get("event_b", "?"),
                                    entry.get("n_sections_a_before_b", 0),
                                    entry.get("n_sections_b_before_a", 0),
                                )
                            )
                        if len(cyclic) > 5:
                            lines.append("  ... ({0} more)".format(len(cyclic) - 5))
                        QMessageBox.warning(self, _("Cyclic Contradictions Detected"), "\n".join(lines))
                else:
                    from stratigraphy.biostratigraphy import RASCAnalyzer

                    analyzer = RASCAnalyzer()
                    # RASC needs a distance matrix - compute from FAD/LAD
                    dist = np.abs(fad_matrix.mean(axis=0)[:, np.newaxis] - lad_matrix.mean(axis=0)[np.newaxis, :])
                    result = analyzer.analyze(
                        dist, event_names=event_names, n_iterations=params.get("rasc_iterations", 100)
                    )

                # Show result summary
                QMessageBox.information(self, _("Biostratigraphy Complete"), result.summary())

                # For UA runs, also show the UAZ hierarchy in a tree view.
                if method == "ua" and getattr(result, "uaz_groups", None):
                    self._show_uaz_tree(result)

                self._status_bar.setInfo(_("{0}: {1} events").format(method.upper(), len(result.events)))
                self._logger.info(f"Biostratigraphy completed: {result.summary()}")

            except Exception as e:
                self._logger.error(f"Biostratigraphy analysis failed: {e}")
                QMessageBox.critical(self, _("Biostratigraphy Error"), format_user_error(e, "生物地层学"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _show_uaz_tree(self, result: object) -> None:
        """Embed the Unitary Association Zone (UAZ) hierarchy tree in the
        workspace.

        Each top-level node corresponds to a UAZ group; its children are
        the underlying maximal cliques (Unitary Associations). Leaf-level
        tooltip carries the event-union for that UAZ. This complements the
        textual ``result.summary()`` with an interactive, navigable
        representation of the merging hierarchy.
        """
        if not getattr(result, "uaz_groups", None):
            return

        tree = QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels([_("Zone / UAZ"), _("Events"), _("Similarity")])
        tree.setMinimumSize(720, 480)
        tree.setAlternatingRowColors(True)

        # Build a fast index from zone-index → zone name so we don't
        # walk ``result.zones`` for every UAZ child node.
        zone_index_to_name: dict[int, str] = {}
        for idx, zone in enumerate(getattr(result, "zones", []) or []):
            zone_index_to_name[idx] = zone.name

        for uaz in result.uaz_groups:
            uaz_name = str(uaz.get("uaz_name", "UAZ ?"))
            event_union = uaz.get("event_union", [])
            mean_sim_raw = uaz.get("mean_similarity", 0.0)
            try:
                mean_sim = float(mean_sim_raw)
            except (TypeError, ValueError):
                mean_sim = 0.0

            top = QTreeWidgetItem(tree)
            top.setText(0, uaz_name)
            top.setText(1, ", ".join(str(e) for e in event_union))
            # Format ``inf`` and NaN gracefully instead of printing
            # the literal "inf" string in the UI.
            if not (mean_sim == mean_sim) or mean_sim in (float("inf"), float("-inf")):
                top.setText(2, "—")
            else:
                top.setText(2, "{0:.3f}".format(mean_sim))
            top.setToolTip(1, "\n".join(str(e) for e in event_union))

            for zone_idx in uaz.get("zone_indices", []) or []:
                leaf = QTreeWidgetItem(top)
                leaf.setText(0, zone_index_to_name.get(int(zone_idx), "Zone ?"))
                if 0 <= int(zone_idx) < len(result.zones):
                    zone_obj = result.zones[int(zone_idx)]
                    zone_events = zone_obj.events
                    leaf.setText(1, ", ".join(zone_events))
                    leaf.setToolTip(1, "\n".join(zone_events))
                    # Show the per-zone dissimilarity-to-predecessor that
                    # we now propagate from ``_merge_to_uaz`` (Fix #4).
                    diss = getattr(zone_obj, "dissimilarity_to_predecessor", None)
                    if diss is None:
                        leaf.setText(2, "—")
                    else:
                        try:
                            d_val = float(diss)
                            if d_val != d_val or d_val in (
                                float("inf"),
                                float("-inf"),
                            ):
                                leaf.setText(2, "—")
                            else:
                                leaf.setText(2, "{0:.3f}".format(d_val))
                        except (TypeError, ValueError):
                            leaf.setText(2, "—")
                else:
                    leaf.setText(1, "—")
                    leaf.setText(2, "—")

        for col in range(tree.columnCount()):
            tree.resizeColumnToContents(col)
        tree.expandAll()

        # Embed into the workspace using a tagged container so
        # :meth:`_cleanup_plot_widgets` can garbage-collect it.
        container = QWidget()
        container.setObjectName("UAZTreeHostWidget")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tree)
        # Tag for the generic cleanup hook. We do NOT reuse
        # ``figure_canvas`` here because the embedded widget is a
        # QTreeWidget, not a Matplotlib canvas, and overloading the
        # property name would mislead future maintainers.
        container.setProperty("workspace_cleanable", True)

        self._cleanup_plot_widgets()
        idx = self._workspace.addWidget(container, _("UAZ Hierarchy"))
        self._workspace.setCurrentIndex(idx)

    def _on_run_pic(self) -> None:
        """Run Phylogenetic Independent Contrasts (PIC) analysis.

        Note: PIC requires a tree and trait values. The dialog itself
        accepts the Newick string and trait dict, so we do not block
        on ``has_data`` here (the data matrix is not the required
        input). However, the dialog still benefits from the dark
        theme and a consistent UX.
        """
        dialog = PICDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_ancestral_states(self) -> None:
        """Run Ancestral State Reconstruction (ASR) analysis."""
        dialog = AncestralStateDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_phylogenetic_signal(self) -> None:
        """Run Blomberg's K phylogenetic signal analysis."""
        dialog = PhyloSignalDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_phylo_anova(self) -> None:
        """Run Phylogenetic ANOVA analysis."""
        dialog = PhyloANOVADialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

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
                _("Version 1.0.1"),
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
        """Update status bar information (memory monitoring)."""
        # Update data display via event-driven approach
        self._update_data_display()

        # Update memory usage label
        try:
            import psutil

            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self._status_bar._memory_label.setText(_("Memory: {0:.1f} MB").format(mem_mb))
        except Exception:
            pass

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

    # =========================================================================
    # New Analysis Handlers
    # =========================================================================

    def _on_run_allometry(self) -> None:
        """Run Allometry analysis."""
        dialog = AllometryDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_evolution_rate(self) -> None:
        """Run Evolution Rate analysis."""
        dialog = EvolutionRateDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_extinction_intervals(self) -> None:
        """Run Extinction Confidence Intervals analysis."""
        dialog = ExtinctionIntervalDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_beta_diversity(self) -> None:
        """Run Beta Diversity analysis."""
        dialog = BetaDiversityDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()

    def _on_run_null_models(self) -> None:
        """Run Null Model analysis."""
        dialog = NullModelDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.exec()


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
