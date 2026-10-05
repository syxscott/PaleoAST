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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

logger = logging.getLogger(__name__)

import contextlib

from PyQt6.QtCore import QObject, QPoint, QRect, QRunnable, QSettings, Qt, QTimer, pyqtSignal, QThreadPool
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QFont,
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

from config.constants import APP_VERSION
from config.design_system import (
    BorderRadius,
    ColorPalette,
    Typography,
    get_palette,
)
from config.i18n import _, get_translator
from controllers.data_controller import DataController
from controllers.statistics_controller import StatisticsController
from models.state_manager import get_state_manager
from presets import ERROR, OK, RUNNING, PresetManager, RunQueue, check_guards, get_spec, write_manifest
from utils.event_bus import get_event_bus
from views.diagnostic_console import DiagnosticConsole
from views.file_drop_handler import FileDropHandler
from views.ui_allometry_dialogs import AllometryDialog, PLSDialog
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
from views.ui_macroevolution_dialogs import (
    MacroevolutionDialog,
    Morpho3DDialog,
)
from views.ui_imputation_dialog import ImputationDialog
from views.ui_navigation import NavigationItem, NavigationTree
from views.ui_null_model_dialogs import NullModelDialog
from views.ui_pcm_dialogs import AncestralStateDialog, PhyloANOVADialog, PhyloSignalDialog, PICDialog
from views.ui_plot_canvas import InteractivePlotCanvas
from views.ui_runlist_panel import RunListPanel
from views.ui_spreadsheet import ScientificSpreadsheet
from views.ui_permutation_dialogs import PermutationTestDialog, PreferencesDialog


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

    # Newick 系统发育树解析错误: 消息自带行列定位, 原样展示
    from utils.exceptions import NewickParseError

    if isinstance(e, NewickParseError):
        return _("ErrMsg: phylogenetic tree parse failed").format(operation_hint, error_msg)

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
            return _("ErrMsg: invalid characters in data").format(operation_hint)

        # 检查是否是数值计算错误（如 log(负数)、sqrt(负数)）
        if any(
            keyword in error_msg.lower()
            for keyword in ["negative value", "invalid value", "math domain error", "不能求", "数值计算"]
        ):
            return _("ErrMsg: numeric computation failed").format(operation_hint)

        # 检查是否是维度不匹配问题
        if any(keyword in error_msg.lower() for keyword in ["dimension", "shape", "axes"]):
            return _("ErrMsg: data dimension mismatch").format(operation_hint)

        # 检查是否是空数据问题
        if "empty" in error_msg.lower() or "没有数据" in error_msg:
            return _("ErrMsg: data is empty").format(operation_hint)

        # 通用数据类型错误
        return _("ErrMsg: data type error").format(
            operation_hint, error_msg[:100]
        )

    # 验证错误
    if "ValidationError" in type(e).__name__ or "验证" in error_msg:
        return _("ErrMsg: data validation failed").format(operation_hint, error_msg)

    # 收敛错误（迭代算法未收敛）
    if "ConvergenceError" in type(e).__name__ or "收敛" in error_msg:
        return _("ErrMsg: algorithm did not converge").format(operation_hint)

    # 矩阵计算错误
    if "singular" in error_msg.lower() or "matrix" in error_msg.lower():
        return _("ErrMsg: matrix computation failed").format(operation_hint)

    # 默认：显示原始错误消息的前100个字符
    return _("ErrMsg: generic error during operation").format(
        operation_hint, error_msg[:200]
    )


class RibbonStyle(Enum):
    """Ribbon button styles."""

    LARGE_ICON = 1
    SMALL_ICON = 2
    TEXT_ONLY = 3
    ICON_TEXT = 4


# Icon types drawn as a mathematical operator rather than a shape.
#
# Chosen because the operator says something the button label does not. Log is
# the exception: "log" repeats its own label, and there is no compact symbol
# for it that is unambiguous at 24px, so it keeps the word.
_GLYPH_ICONS = {
    "tf_log": "log",
    "tf_sqrt": "√",
    "tf_hellinger": "√p",
    "tf_zscore": "σ",
    "tf_pct": "%",
    "tf_wisconsin": "↔",
}


def _draw_glyph(
    painter: QPainter, glyph: str, size: int, margin: int, color: str
) -> None:
    """Draw a centred single-glyph icon.

    The font size is derived from the icon size rather than fixed, so the
    same call works at 24px on a ribbon button and 32px anywhere else. A
    symbol is optically lighter than a filled shape at the same nominal
    size, so it is nudged up.
    """
    font = QFont()
    font.setPointSizeF(max(7.0, size * 0.46))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QPen(QColor(color)))
    painter.drawText(
        QRect(margin, margin, size - 2 * margin, size - 2 * margin),
        Qt.AlignmentFlag.AlignCenter,
        glyph,
    )


class VectorIconEngine:
    """
    Vector Icon Engine using QPainter.

    Generates all application icons programmatically without external files.
    Each icon is drawn using primitive shapes and paths.

    Colours come from the active palette, so icons follow the theme; see
    ``create_icon``. Glyph-based icons are listed in ``_GLYPH_ICONS``.
    """

    @staticmethod
    def create_icon(
        icon_type: str,
        size: int = 32,
        palette: ColorPalette | None = None,
    ) -> QPixmap:
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

        # Colours come from the active palette, not from literals. The old
        # defaults were flat-UI values chosen against a white button; the pen
        # in particular (#2C3E50) disappeared entirely on a dark surface, so
        # the icons never followed the theme.
        p = palette if palette is not None else get_palette(False)
        ink = p.text_secondary
        fill = p.primary
        green = p.success
        red = p.error
        amber = p.warning
        hairline = p.border_light
        paper = p.bg_tertiary
        muted = p.text_disabled
        violet = p.primary_light
        teal = p.info
        plum = p.secondary
        pen = QPen(QColor(ink))
        pen.setWidth(max(1, size // 16))
        brush = QBrush(QColor(fill))
        painter.setPen(pen)
        painter.setBrush(brush)

        margin = size // 8
        inner_size = size - 2 * margin

        if icon_type == "new_file":
            # Document with plus sign
            doc_rect = QRect(margin, margin, inner_size, inner_size)
            painter.drawRect(doc_rect)
            # Plus sign
            painter.setPen(QPen(QColor(green), max(2, size // 12)))
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
            painter.setBrush(QBrush(QColor(hairline)))
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
            painter.setPen(QPen(QColor(red), max(2, size // 16)))
            painter.drawLine(center_x, center_y, center_x + axis_length, center_y)

            # Y-axis (PC2)
            painter.setPen(QPen(QColor(green), max(2, size // 16)))
            painter.drawLine(center_x, center_y, center_x, center_y - axis_length)

            # Z-axis hint (PC3)
            painter.setPen(QPen(QColor(fill), max(2, size // 16)))
            painter.drawLine(center_x, center_y, center_x - axis_length // 2, center_y + axis_length // 2)

            # Ellipse representing variance
            painter.setPen(QPen(QColor(amber), max(1, size // 24)))
            ellipse_rect = QRect(
                center_x - axis_length // 3, center_y - axis_length // 3, axis_length * 2 // 3, axis_length * 2 // 3
            )
            painter.drawEllipse(ellipse_rect)

        elif icon_type == "diversity":
            # Biodiversity tree/branch icon
            center_x = size // 2
            base_y = size - margin

            # Main trunk
            painter.setPen(QPen(QColor(green), max(2, size // 12)))
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
            painter.setBrush(QBrush(QColor(hairline)))
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
            painter.setPen(QPen(QColor(violet), max(2, size // 16)))
            painter.drawPolygon(points)
            for pt in points:
                painter.setBrush(QBrush(QColor(violet)))
                painter.drawEllipse(pt, size // 10, size // 10)

        elif icon_type == "stratigraphy":
            # Layered sedimentary strata
            num_layers = 4
            layer_height = inner_size // num_layers
            colors = [red, amber, green, fill]
            for i, color in enumerate(colors):
                painter.setBrush(QBrush(QColor(color)))
                painter.drawRect(QRect(margin, margin + i * layer_height, inner_size, layer_height - 1))

        elif icon_type == "nmds":
            # Stress plot icon
            painter.setPen(QPen(QColor(teal), max(2, size // 16)))
            points_data = [
                QPoint(margin + inner_size // 5, size - margin - inner_size // 5),
                QPoint(size // 2, margin + inner_size // 3),
                QPoint(size - margin - inner_size // 5, size - margin - inner_size // 2),
            ]
            for pt in points_data:
                painter.setBrush(QBrush(QColor(teal)))
                painter.drawEllipse(pt, size // 12, size // 12)
            painter.drawPolyline(points_data)

        elif icon_type == "anosim":
            # Box plots comparison
            box_width = inner_size // 4
            box1_x = margin + inner_size // 6
            box2_x = size - margin - inner_size // 6 - box_width

            painter.setBrush(QBrush(QColor(fill)))
            painter.drawRect(QRect(box1_x, margin + inner_size // 3, box_width, inner_size // 2))
            painter.drawLine(box1_x + box_width // 2, margin, box1_x + box_width // 2, margin + inner_size // 3)
            painter.drawLine(
                box1_x + box_width // 2,
                margin + inner_size // 3 + inner_size // 2,
                box1_x + box_width // 2,
                size - margin,
            )

            painter.setBrush(QBrush(QColor(red)))
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
            painter.setPen(QPen(QColor(plum), max(2, size // 16)))
            painter.setBrush(QBrush(QColor(plum)))
            for pt in points:
                painter.drawEllipse(pt, size // 18, size // 18)
            painter.drawPolyline(points[:-1])

        elif icon_type == "chart":
            # Generic bar chart (used by LDA, CCA, stats, etc.)
            bar_count = 4
            bar_width = inner_size // (bar_count * 2)
            bar_colors = [fill, red, green, amber]
            for i in range(bar_count):
                height_factor = 0.4 + 0.6 * (1 - abs(i - 1.5) / 2)
                bar_height = int(inner_size * height_factor)
                x = margin + i * (inner_size // bar_count) + bar_width // 2
                y = margin + inner_size - bar_height
                painter.setBrush(QBrush(QColor(bar_colors[i])))
                painter.drawRect(QRect(x, y, bar_width, bar_height))

        elif icon_type == "imputation":
            # Grid of cells with one cell highlighted to suggest "filling in"
            painter.setPen(QPen(QColor(muted), max(1, size // 32)))
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
                        painter.setBrush(QBrush(QColor(green)))
                    else:
                        painter.setBrush(QBrush(QColor(paper)))
                    painter.drawRect(rect)

        elif icon_type in _GLYPH_ICONS:
            # Mathematical operator as the glyph.
            #
            # The six data transforms used to share one "settings" gear, which
            # is worse than no icon: six adjacent buttons looked identical, and
            # the icon repeated what the ribbon tab already said. These glyphs
            # carry information the label does not -- sqrt, sigma, percent and
            # the double-standardisation arrows are all readable without the
            # word next to them.
            _draw_glyph(painter, _GLYPH_ICONS[icon_type], size, margin, fill)

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

        # Create icon. The palette is passed in rather than defaulted inside
        # the engine, so a button created while the dark theme is active does
        # not start with light-theme glyphs.
        if icon_type:
            self._refresh_icon()

        # Apply stylesheet
        self._apply_stylesheet()

    def _refresh_icon(self) -> None:
        """(Re)draw the icon for the current theme.

        Icons were drawn once at construction and never redrawn, so a theme
        switch left the old palette's glyphs on screen -- the same failure as
        matplotlib annotations keeping their creation-time colour.
        """
        if not self._icon_type:
            return
        pixmap = VectorIconEngine.create_icon(
            self._icon_type, 24, get_palette(self._is_dark_theme)
        )
        self.setIcon(QIcon(pixmap))

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set theme and update stylesheet."""
        self._is_dark_theme = is_dark
        self._refresh_icon()
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
            "color: " + c.on_primary + "; "
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
        # "--" means "not measured yet". It must NOT start at "0 MB": psutil
        # is an optional dependency, so on a base install the update below
        # never runs and a hard-coded 0 sits there forever, reading like a
        # broken reading rather than an unavailable one.
        self._memory_label = QLabel(_("Memory: --"))
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
        """Set info text.

        Also clears the warning style, so a message that could not be
        honoured stops looking like one once the user moves on.
        """
        self._info_label.setStyleSheet("")
        self._info_label.setToolTip("")
        self._info_label.setText(text)

    def setWarning(self, text: str) -> None:
        """Set warning text.

        Distinct from :meth:`setInfo` on purpose. A request the application
        could not honour has to look different from one it carried out --
        otherwise a user who ticks a box and gets the default behaviour has no
        way to tell that the box did nothing.
        """
        self._info_label.setStyleSheet("color: #b26a00;")
        self._info_label.setToolTip(text)
        self._info_label.setText(f"⚠ {text}")

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

    # Empty-state actions: the workspace emits these signals when the
    # user clicks one of the buttons shown before any data is loaded.
    # ``MainWindow`` connects them to the existing data-loading slots,
    # so the buttons share the same code path as the ribbon entries.
    loadExampleRequested = pyqtSignal()
    openFileRequested = pyqtSignal()
    importDataRequested = pyqtSignal()

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

        # Empty-state placeholder: a centred headline plus three
        # large clickable entry buttons.  ``WorkspaceArea`` does not
        # know how to actually load data (that lives on
        # ``MainWindow``), so it just emits the matching signal and
        # ``MainWindow`` connects its existing slots.
        self._placeholder = self._build_empty_state()
        self._stack.addWidget(self._placeholder)

    def _build_empty_state(self) -> QWidget:
        """Build the welcome pane with three primary actions."""
        from PyQt6.QtWidgets import QPushButton

        t = Typography()
        palette = get_palette()
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(40, 40, 40, 40)
        wrapper_layout.setSpacing(20)
        wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Headline
        headline = QLabel(_("Welcome to PaleoAST"))
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline.setStyleSheet(
            "QLabel {"
            " color: " + palette.text_primary + ";"
            " font-size: " + str(int(t.body_lg_size * 1.6)) + "px;"
            " font-weight: 600;"
            " background-color: transparent;"
            "}"
        )
        wrapper_layout.addWidget(headline)

        subtitle = QLabel(_("Load data to begin analysis"))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            "QLabel {"
            " color: " + palette.text_disabled + ";"
            " font-size: " + str(t.body_lg_size) + "px;"
            " background-color: transparent;"
            "}"
        )
        wrapper_layout.addWidget(subtitle)

        wrapper_layout.addSpacing(16)

        # --- Three clickable entries
        buttons = QHBoxLayout()
        buttons.setSpacing(16)
        buttons.setAlignment(Qt.AlignmentFlag.AlignCenter)

        load_btn = QPushButton(_("Load Example Data…"))
        load_btn.setMinimumHeight(56)
        load_btn.setMinimumWidth(180)
        load_btn.clicked.connect(self.loadExampleRequested)
        load_btn.setToolTip(_("Load a built-in example from data/examples/"))

        open_btn = QPushButton(_("Open File…"))
        open_btn.setMinimumHeight(56)
        open_btn.setMinimumWidth(180)
        open_btn.clicked.connect(self.openFileRequested)
        open_btn.setToolTip(_("Open a CSV / TXT / Excel file from disk"))

        import_btn = QPushButton(_("Import Data…"))
        import_btn.setMinimumHeight(56)
        import_btn.setMinimumWidth(180)
        import_btn.clicked.connect(self.importDataRequested)
        import_btn.setToolTip(_("Use the import wizard to map columns and types"))

        buttons.addWidget(load_btn)
        buttons.addWidget(open_btn)
        buttons.addWidget(import_btn)
        wrapper_layout.addLayout(buttons)

        # Keep references so setDarkTheme can re-style them.
        self._empty_headline = headline
        self._empty_subtitle = subtitle
        self._empty_load_btn = load_btn
        self._empty_open_btn = open_btn
        self._empty_import_btn = import_btn

        outer_layout.addStretch(1)
        outer_layout.addWidget(wrapper)
        outer_layout.addStretch(2)
        return outer

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
        # ``self._placeholder`` is the empty-state widget (not a
        # single ``QLabel`` any more).  Only restyle its child
        # labels; the buttons are re-styled by Qt's own palette change.
        if self._placeholder is not None:
            headline = getattr(self, "_empty_headline", None)
            for lbl in self._placeholder.findChildren(QLabel):
                if lbl is headline:
                    lbl.setStyleSheet(
                        "QLabel {"
                        " color: " + c.text_primary + ";"
                        " font-size: " + str(int(t.body_lg_size * 1.6)) + "px;"
                        " font-weight: 600;"
                        " background-color: transparent;"
                        "}"
                    )
                else:
                    lbl.setStyleSheet(
                        "QLabel {"
                        " color: " + c.text_disabled + ";"
                        " font-size: " + str(t.body_lg_size) + "px;"
                        " background-color: transparent;"
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


class _AnalysisSignals(QObject):
    """后台分析任务的信号桥。

    pyqtSignal 只能挂在 QObject 子类上 (QRunnable 不是 QObject,
    在 QRunnable 里声明信号会在 connect 时抛
    "cannot be converted to PyQt6.QtCore.QObject")。工作者线程
    通过本对象 emit, Qt 的队列连接保证槽在 GUI 线程执行。
    """

    result_ready = pyqtSignal(object)
    error_raised = pyqtSignal(Exception)
    progress = pyqtSignal(int, int)


class _AnalysisTask(QRunnable):
    """在线程池中执行 work(), 结果/异常经 _AnalysisSignals 回传。

    长时间计算 (NMDS 多重启动、CONISS、GPA 迭代等) 若在 GUI 线程
    同步执行会冻结整个界面且无任何进度提示; 分析处理器应通过
    ``MainWindow._run_analysis_async`` 使用本任务。

    当 ``wants_reporter`` 为真时, work 被调用为 ``work(reporter)``,
    其中 reporter 是一个 ``(value, maximum)`` 可调用对象, 它通过
    ``signals.progress`` 将进度安全地转发到 GUI 线程。
    """

    def __init__(self, work, signals: _AnalysisSignals, wants_reporter: bool = False):
        super().__init__()
        self._work = work
        self._signals = signals  # 持引用防垃圾回收
        self._wants_reporter = wants_reporter

    def run(self):
        try:
            if self._wants_reporter:
                reporter = lambda value, maximum: self._signals.progress.emit(value, maximum)  # noqa: E731
                result = self._work(reporter)
            else:
                result = self._work()
            self._signals.result_ready.emit(result)
        except RuntimeError as e:
            # The parent window was destroyed while this task was still
            # running (closeEvent's drain gave up after its timeout), so the
            # QObject bridge is gone and emitting raises. There is nothing to
            # report to any more: swallow it instead of letting the exception
            # escape the worker thread, which Qt turns into
            # "QThread: Destroyed while thread is still running" -> abort.
            logging.getLogger(__name__).debug("Discarding analysis result after teardown: %s", e)
        except Exception as e:  # noqa: BLE001 - 后台线程边界, 必须回传
            try:
                self._signals.error_raised.emit(e)
            except RuntimeError:
                logging.getLogger(__name__).debug("Discarding analysis error after teardown: %s", e)


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
        self._status_timer.start(5000)  # Update every 5s to reduce CPU overhead

        # Setup drag and drop
        self._setup_drag_drop()

        # Thread pool for long-running analysis tasks so the GUI stays responsive.
        # Max 4 concurrent analysis workers to avoid saturating the CPU.
        self._thread_pool = QThreadPool.globalInstance()
        if self._thread_pool is None:
            self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(min(4, os.cpu_count() or 4))

        # Set in closeEvent: once the window starts tearing down, late worker
        # callbacks must not touch destroyed widgets.
        self._closing = False

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
                # A multi-sheet workbook parses fine but carries no single
                # "data" key, so the generic message below ("check that the
                # selected data is numeric and has no missing values") was
                # actively wrong — the file is perfectly valid. Ask which
                # sheet instead of falling into the numeric-data branch.
                sheets = data.get("sheets")
                if sheets:
                    from PyQt6.QtWidgets import QInputDialog

                    sheet_names = list(sheets.keys())
                    chosen, ok = QInputDialog.getItem(
                        self,
                        _("Select Sheet"),
                        _("This workbook has {0} sheets. Choose one to load:").format(len(sheet_names)),
                        sheet_names,
                        0,
                        False,
                    )
                    if not ok or chosen not in sheets:
                        return
                    data = sheets[chosen]
                    matrix_data = data.get("data")
                if matrix_data is None:
                    raise ValueError("No data in parsed file")

            row_labels = data.get("row_labels")
            col_labels = data.get("col_labels")

            new_matrix = DataMatrix(matrix_data, row_labels=row_labels, col_labels=col_labels)

            self._state.set_data_matrix(new_matrix)
            self._spreadsheet.load_data(matrix_data, row_labels=row_labels, col_labels=col_labels, update_state=False)

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
            QMessageBox.critical(self, _("Load Error"), format_user_error(e, "Op: file loading"))

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

        # Batch run queue (dockable, tabbed with the console, hidden by default)
        self._preset_manager = PresetManager()
        self._run_queue = RunQueue(
            guard=self._runlist_guard,
            executor=self._runlist_execute,
            on_change=lambda item: self._runlist_panel.item_changed(item),
        )
        self._runlist_panel = RunListPanel(self._run_queue, self._preset_manager, self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._runlist_panel)
        self.tabifyDockWidget(self._diagnostic_console, self._runlist_panel)
        self._runlist_panel.hide()
        self._runlist_panel.run_all_requested.connect(self._on_runlist_run_all)

        # Create menu bar
        self._create_menu_bar()

        # Create spreadsheet (initially hidden)
        self._spreadsheet = ScientificSpreadsheet()
        self._spreadsheet_index = self._workspace.addWidget(self._spreadsheet, _("Spreadsheet"))

        # Connect empty-state placeholder buttons to the same slots
        # the ribbon uses, so the user has a working entry point
        # before any data is loaded.
        self._workspace.loadExampleRequested.connect(self._on_load_example_dataset)
        self._workspace.openFileRequested.connect(self._on_open_file)
        self._workspace.importDataRequested.connect(self._on_import_data)

        # Initialize UI state based on data availability
        self._update_ui_state()

    def _on_load_example_dataset(self) -> None:
        """Show a tiny picker for the bundled example datasets and
        load the chosen one.

        The loader helpers live in :mod:`data.loader`.  Before this
        method existed the empty workspace offered no way to load
        anything other than the ribbon entries, which the new user
        typically doesn't know to look for.
        """
        from data.loader import list_example_datasets, load_community, load_moth_wings, load_primate_traits, load_primate_tree
        from PyQt6.QtWidgets import QInputDialog

        datasets = list_example_datasets()
        labels = [f"{d['name']} — {d['description']}" for d in datasets]
        if not labels:
            QMessageBox.information(
                self,
                _("No Examples"),
                _("No example datasets are bundled with this installation."),
            )
            return
        choice, ok = QInputDialog.getItem(
            self,
            _("Load Example Data"),
            _("Pick an example dataset:"),
            labels,
            0,
            False,
        )
        if not ok:
            return
        idx = labels.index(choice)
        ds = datasets[idx]
        name = ds["name"]
        try:
            import numpy as np

            from models.data_matrix import DataMatrix

            if name == "moth_wings":
                arr, ids = load_moth_wings()
                data = np.asarray(arr, dtype=float)
                row_labels = list(ids)
                n_pts = data.shape[1] // 2
                col_labels = [f"{c}{i + 1}" for i in range(n_pts) for c in ("x", "y")]
                matrix = DataMatrix(data, row_labels=row_labels, col_labels=col_labels)
            elif name == "community_abundance":
                df = load_community()
                site_col = df["site"].astype(str).tolist() if "site" in df.columns else None
                group_col = df["group"].astype(str).tolist() if "group" in df.columns else None
                data_df = df.drop(columns=[c for c in ("site", "group") if c in df.columns])
                matrix = DataMatrix(
                    data_df.to_numpy(dtype=float),
                    row_labels=site_col,
                    col_labels=list(data_df.columns),
                    specimen_metadata=[
                        {"group": g} if g is not None else {}
                        for g in (group_col or [None] * len(data_df))
                    ],
                )
            elif name == "primate_tree":
                # Phylogenetic tree isn't a matrix; surface the file
                # contents in a text tab instead of stuffing it into
                # the data matrix.
                tree = load_primate_tree()
                from PyQt6.QtWidgets import QTextEdit

                editor = QTextEdit()
                editor.setReadOnly(True)
                lines = [f"Primate phylogeny: {tree.leaf_count} tips"]
                lines.append("Leaf names: " + ", ".join(tree.leaf_names))
                editor.setPlainText("\n".join(lines))
                self._add_tab_to_workspace(editor, _("Example — Primate Tree"))
                self._status_bar.setInfo(
                    _("Loaded example: {0} ({1} tips)").format(name, tree.leaf_count)
                )
                return
            elif name == "primate_traits":
                df = load_primate_traits()
                species_col = df["species"].astype(str).tolist() if "species" in df.columns else None
                data_df = df.drop(columns=[c for c in ("species",) if c in df.columns])
                matrix = DataMatrix(
                    data_df.to_numpy(dtype=float),
                    row_labels=species_col,
                    col_labels=list(data_df.columns),
                )
            else:
                QMessageBox.warning(
                    self,
                    _("Unknown Example"),
                    _("Unrecognised example dataset: {0}").format(name),
                )
                return
        except Exception as exc:
            self._logger.error("Loading example '%s' failed: %s", name, exc)
            QMessageBox.critical(
                self,
                _("Load Failed"),
                _("Could not load example dataset: {0}").format(exc),
            )
            return

        self._state.set_data_matrix(matrix, mark_modified=False)
        self._spreadsheet.load_data(
            matrix.data,
            row_labels=matrix.row_labels,
            col_labels=matrix.col_labels,
            update_state=False,
        )
        self._update_ui_state()
        self._workspace.setCurrentIndex(self._spreadsheet_index)
        self._status_bar.setInfo(
            _("Loaded example: {0} ({1} samples x {2} variables)").format(
                name, matrix.n_samples, matrix.n_variables
            )
        )

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

        # Data transformations group.
        #
        # These used to share one "settings" gear, which made six adjacent
        # buttons indistinguishable and repeated what the ribbon tab already
        # said. Each gets its operator instead, so the icon carries something
        # the label does not.
        transform_group = home_tab.addGroup(_("Transform"))
        self._btn_log_transform = transform_group.addButton("tf_log", _("Log"), _("Log transformation (base 10)"))
        self._btn_sqrt_transform = transform_group.addButton("tf_sqrt", _("Sqrt"), _("Square root transformation"))
        self._btn_hellinger_transform = transform_group.addButton(
            "tf_hellinger", _("Hellinger"), _("Hellinger transformation")
        )
        self._btn_zscore_transform = transform_group.addButton("tf_zscore", _("Z-Score"), _("Z-score standardization"))
        self._btn_percent_transform = transform_group.addButton(
            "tf_pct", _("% Total"), _("Percentage standardization")
        )
        self._btn_wisconsin_transform = transform_group.addButton(
            "tf_wisconsin", _("Wisconsin"), _("Wisconsin double standardization")
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
        # Add a combobox so users can pick the specific univariate test
        # directly from the ribbon, rather than going through the
        # generic dialog each time.
        self._univar_combo = univar_group.addComboBox(
            [_("Summary"), _("Normality"), _("t-test"), _("ANOVA"), _("Kruskal-Wallis")],
            tooltip=_("Univariate test"),
            on_change=self._on_run_univariate_by_index,
        )
        self._register_data_button(self._btn_univariate)
        self._univar_combo._is_undo_button = False  # never used as undo marker
        self._btn_simper = univar_group.addButton("chart", "SIMPER", _("SIMPER Analysis"))

        # Diversity group.
        #
        # No icons here on purpose. All three previously used one "diversity"
        # glyph, which repeated the tab name and made the buttons
        # indistinguishable from each other. Inventing three decorative
        # symbols would not be better -- the information is in the method
        # names -- so the group drops the icon and keeps the words.
        diversity_group = analysis_tab.addGroup(_("Diversity"))
        self._btn_diversity = diversity_group.addButton(
            "", _("Diversity"), _("Biodiversity indices"), RibbonStyle.TEXT_ONLY
        )
        self._btn_abundance = diversity_group.addButton(
            "", _("Models"), _("Abundance Models"), RibbonStyle.TEXT_ONLY
        )
        self._btn_she = diversity_group.addButton(
            "", "SHE", _("SHE Analysis"), RibbonStyle.TEXT_ONLY
        )

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
        self._btn_eigenshape = efa_group.addButton("morphometrics", _("Eigenshape"), _("Eigenshape Analysis"))
        self._register_data_button(self._btn_eigenshape)

        tps_group = morpho_tab.addGroup(_("Deformation"))
        self._btn_tps_grid = tps_group.addButton("morphometrics", _("Grid"), _("TPS Deformation Grid"))

        # Spatial tab
        spatial_tab = self._ribbon.addTab(_("Spatial"))
        spatial_group = spatial_tab.addGroup(_("Point Pattern"))
        self._btn_ripley_k = spatial_group.addButton("chart", _("Ripley K"), _("Ripley's K Spatial Analysis"))

        # Stratigraphy tab
        strat_tab = self._ribbon.addTab(_("Stratigraphy"))

        # Every button on this tab used to carry one "stratigraphy" glyph, so
        # the tab showed nine identical icons. The icon encoded the tab name
        # and nothing else, so the tab drops it: the method names are the
        # information, and a decorative symbol per method would be noise.
        _text = RibbonStyle.TEXT_ONLY

        strat_group = strat_tab.addGroup(_("Time Series"))
        self._btn_spectral = strat_group.addButton(
            "", _("Spectral"), _("Spectral Analysis"), _text
        )
        self._btn_coniss = strat_group.addButton(
            "", "CONISS", _("CONISS Zonation"), _text
        )
        self._btn_wavelet = strat_group.addButton(
            "", _("Wavelet"), _("Wavelet CWT Analysis"), _text
        )
        self._btn_isotope = strat_group.addButton(
            "", _("Isotope"), _("Isotope Time Series"), _text
        )
        self._btn_strat_corr = strat_group.addButton(
            "", _("Correlation"), _("Stratigraphic Correlation"), _text
        )

        bio_group = strat_tab.addGroup(_("Biostratigraphy"))
        self._btn_biostrat = bio_group.addButton(
            "", _("Biozone"), _("UA/RASC Biostratigraphy"), _text
        )

        paleo_group = strat_tab.addGroup(_("Paleo-Environment"))
        self._btn_paleo_env = paleo_group.addButton(
            "", _("CA Axis"), _("Paleo-Env. CA Reconstruction"), _text
        )

        markov_group = strat_tab.addGroup(_("Facies"))
        self._btn_markov = markov_group.addButton(
            "", _("Markov"), _("Markov Chain Analysis"), _text
        )
        self._btn_directional = markov_group.addButton(
            "", _("Rose"), _("Directional Statistics"), _text
        )

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
        self._register_data_button(self._btn_undo, kind="undo")
        self._register_data_button(self._btn_redo, kind="redo")
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
        # Disable the combobox alongside the rest of the data buttons
        # until data is loaded.
        self._data_buttons.append(self._univar_combo)
        self._univar_combo.setEnabled(self._state.has_data)
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
        self._btn_eigenshape.clicked.connect(self._on_run_eigenshape)
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

    def _get_plot_labels_and_groups(
        self,
    ) -> tuple[list[str] | None, list[int] | None, list[str] | None]:
        """Resolve (labels, groups, group_names) for ordination score plots.

        Used by every ``plot_*_scores`` / ``plot_pcoa_scores`` /
        ``plot_nmds`` call so that the user-visible points get their
        real row labels and not the canvas's ``S1..Sn`` fallback, and so
        that colour-by-group actually colours anything.

        Resolution order (the first source that yields ``n_samples`` of
        matching values wins; we never silently fabricate zeros):

            1. ``data_matrix.row_labels`` -- used as labels verbatim
               when its length matches ``n_samples``.
            2. ``data_matrix.specimen_metadata`` -- if any per-row entry
               contains a "group"/"Group"/"habitat"/"site" key, that
               key's value is the group label.
            3. ``row_metadata`` group edits made via the spreadsheet --
               treated as the canonical group source when present.
            4. Otherwise ``groups`` is returned as ``None``.

        Returns:
            ``(labels, groups, group_names)``. ``labels`` may be ``None``
            when ``row_labels`` does not match (the canvas's
            ``S1..Sn`` fallback is acceptable in that case).
        """
        matrix = self._state.data_matrix
        if matrix is None:
            return None, None, None
        try:
            n_samples = matrix.n_samples
        except Exception:
            return None, None, None

        # --- labels: prefer the matrix's own row_labels when they match
        try:
            row_labels = list(matrix.row_labels)
        except Exception:
            row_labels = []
        labels = list(row_labels) if len(row_labels) == n_samples else None

        # --- groups: walk metadata, then spreadsheet edits, then give up
        groups, group_names = self._resolve_groups_for_plot(n_samples)
        return labels, groups, group_names

    def _resolve_groups_for_plot(
        self, n_samples: int
    ) -> tuple[list[int] | None, list[str] | None]:
        """Find per-row group labels for ordination plots.

        Sources, in priority order:

            1. ``data_matrix.specimen_metadata[i]["group"|"Group"|...]``
               when every row has a value at that key.
            2. The spreadsheet's :class:`RowMetadataManager` group edits
               (same source ``_get_groups`` uses for ANOSIM/PERMANOVA).
            3. ``None`` (do NOT fall back to ``[0] * n_samples`` -- that
               is exactly the bug the canvas layer used to hide).

        Returns ``(groups, group_names)``; ``group_names`` is the human-
        readable mapping ``["Habitat 1", "Habitat 2", ...]`` so that the
        legend reads the real name and not ``Group 0``.
        """
        matrix = self._state.data_matrix
        # --- source 1: specimen_metadata[].get("group") ----------------------
        try:
            spec_meta = list(getattr(matrix, "specimen_metadata", []) or [])
        except Exception:
            spec_meta = []
        per_row: list[object] = []
        if spec_meta and len(spec_meta) == n_samples:
            keys = ("group", "Group", "habitat", "Habitat", "site", "Site")
            for k in keys:
                values_k: list[object] = []
                ok = True
                for entry in spec_meta:
                    if not isinstance(entry, dict) or k not in entry or entry[k] in (None, ""):
                        ok = False
                        break
                    values_k.append(entry[k])
                if ok and values_k:
                    per_row = values_k
                    break
        if per_row:
            unique: list[object] = []
            idx_map: dict[object, int] = {}
            for v in per_row:
                if v not in idx_map:
                    idx_map[v] = len(unique)
                    unique.append(v)
            groups = [idx_map[v] for v in per_row]
            return groups, [str(v) for v in unique]

        # --- source 2: row_metadata group edits (same as _get_groups) ------
        try:
            existing_groups = self._get_groups()
        except Exception:
            existing_groups = None
        if existing_groups is not None and len(existing_groups) == n_samples:
            # Rebuild a stable name map from the row metadata if possible.
            rm = self._state.row_metadata
            name_map: dict[int, str] = {}
            if rm is not None:
                try:
                    raw = rm.get_groups()
                except Exception:
                    raw = {}
                seen: list[str] = []
                for i in sorted(raw.keys()):
                    label = raw.get(i) or "Ungrouped"
                    label_s = str(label)
                    if label_s not in seen:
                        seen.append(label_s)
                        name_map[len(seen) - 1] = label_s
                # ``existing_groups`` was built in the same first-appearance
                # order, so the index → label mapping aligns.
            group_names = [name_map.get(i, f"Group {i + 1}") for i in range(len(set(existing_groups)))]
            return list(existing_groups), group_names

        return None, None

    def _update_ui_state(self) -> None:
        """Update UI element states based on data availability."""
        has_data = self._state.has_data

        # Update all registered data-dependent actions
        for action in self._data_actions:
            action.setEnabled(has_data)

        # Update all registered data-dependent buttons. Undo/Redo
        # buttons are registered as data buttons but their actual
        # availability depends on the StateManager undo/redo stacks,
        # not just on whether data is loaded.
        for button in self._data_buttons:
            if getattr(button, "_is_undo_button", False):
                button.setEnabled(self._state.can_undo())
            elif getattr(button, "_is_redo_button", False):
                button.setEnabled(self._state.can_redo())
            else:
                button.setEnabled(has_data)

    def _register_data_action(self, action: QAction) -> None:
        """Register an action as data-dependent."""
        if action not in self._data_actions:
            self._data_actions.append(action)
            action.setEnabled(self._state.has_data)

    def _register_data_button(self, button, *, kind: str = "data") -> None:
        """Register a button as data-dependent.

        Parameters:
            button: The button to register.
            kind: One of "data", "undo", or "redo". "data" toggles
                with ``has_data``; "undo" toggles with ``can_undo()``;
                "redo" toggles with ``can_redo()``. Undo/Redo buttons
                are therefore initially disabled until a state change
                is performed.
        """
        if button not in self._data_buttons:
            self._data_buttons.append(button)
            if kind == "undo":
                button._is_undo_button = True
                button.setEnabled(self._state.can_undo())
            elif kind == "redo":
                button._is_redo_button = True
                button.setEnabled(self._state.can_redo())
            else:
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

        export_plot_action = QAction(_("Export Plot as &Image..."), self)
        export_plot_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_plot_action.triggered.connect(self._on_export_current_plot)
        file_menu.addAction(export_plot_action)
        self._register_data_action(export_plot_action)
        self._export_plot_action = export_plot_action

        # Editable-R export. Two actions on purpose: one regenerates the .R
        # (and therefore overwrites an edited file, after asking), the other
        # runs the .R exactly as it sits on disk. Merging them would silently
        # discard the user's RStudio edits.
        r_export_action = QAction(_("Export PCA as &R Script..."), self)
        r_export_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        r_export_action.triggered.connect(self._on_export_as_r_script)
        file_menu.addAction(r_export_action)
        self._register_data_action(r_export_action)

        r_rerun_action = QAction(_("Re-run &R Script"), self)
        # Not a variant of the export shortcut: Qt cannot even express
        # "Ctrl+Shift+Shift+R", and a modifier chord that folds two different
        # actions into one gesture is easy to trigger by accident -- which for
        # the regenerate action means overwriting a script.
        r_rerun_action.setShortcut(QKeySequence("Ctrl+Alt+R"))
        r_rerun_action.triggered.connect(self._on_rerun_r_script)
        file_menu.addAction(r_rerun_action)

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

        analysis_menu.addSeparator()

        runlist_action = self._runlist_panel.toggleViewAction()
        runlist_action.setText(_("Run &Queue (batch)..."))
        runlist_action.setToolTip(_("Show the batch run queue panel"))
        analysis_menu.addAction(runlist_action)

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
        """Switch application language.

        语言选择立即持久化到 QSettings: 两个分支 ("立即重启" /
        "下次启动生效") 都保证该选择不会丢失。会话内 UI 仍是旧语言
        (界面文本在构建时经 _() 取值, 无重翻译机制), 因此:
        - "立即重启": 重启进程, 新语言即刻生效;
        - "下次启动生效": 勾选移到新语言, 并在状态栏提示下次启动生效。
        旧实现中 "稍后" 分支既不持久化也不应用——用户以为稍后生效,
        实际永不生效, 属于 UX 陷阱, 已修正。
        """
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import QMessageBox

        current_lang = get_translator().get_language()
        if lang == current_lang:
            # 点击的就是当前会话语言: 清除可能遗留的"下次启动生效"
            # 持久化选择, 保证 QSettings 与用户最终意图一致。
            settings = QSettings("PaleoAST", "PaleoAST")
            if settings.value("language", "") != lang:
                settings.setValue("language", lang)
                settings.sync()
            return

        # 立即持久化 (execv 替换进程前显式 flush 更安全)
        settings = QSettings("PaleoAST", "PaleoAST")
        settings.setValue("language", lang)
        settings.sync()
        # 选择已提交, 勾选反映新语言
        self._lang_action_en.setChecked(lang == "en")
        self._lang_action_zh.setChecked(lang == "zh")

        msg = QMessageBox(self)
        msg.setWindowTitle(_("Language Changed"))
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(_("The language has been changed to {0}.").format("中文" if lang == "zh" else "English"))
        msg.setInformativeText(_("Restart now to apply the change?"))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.button(QMessageBox.StandardButton.Yes).setText(_("Restart Now"))
        msg.button(QMessageBox.StandardButton.No).setText(_("Apply on Next Start"))
        if msg.exec() != QMessageBox.StandardButton.Yes:
            # 用户选择"下次启动生效": 选择已保存, 无需进一步操作
            self._status_bar.setInfo(
                _("Language will change to {0} on next start.").format("中文" if lang == "zh" else "English")
            )
            return

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
        # Windows 上 os.execv 以分离方式重启且立即从调用点返回,
        # 可能产生新旧两个实例并存; 用分离子进程 + 正常退出更稳。
        # POSIX 上 execv 真正替换进程映像, 保持原实现。
        if sys.platform == "win32":
            import subprocess

            subprocess.Popen(
                [sys.executable, *sys.argv],
                cwd=os.getcwd(),
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
            QApplication.quit()
            sys.exit(0)
        else:
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
            _("Eigenshape"): self._on_run_eigenshape,
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
            # Macroevolution + 3-D morphometrics (newly reachable).
            _("Cohort Survivorship"): self._on_run_cohort_survivorship,
            _("Diversity Dynamics"): self._on_run_diversity_dynamics,
            _("Survival Analysis"): self._on_run_survival_analysis,
            _("FBD Simulation"): self._on_run_fbd_simulation,
            _("3-D GPA"): self._on_run_gpa3d,
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
            self._spreadsheet.load_data(data, row_labels=row_labels, col_labels=col_labels, update_state=False)
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

                # Update state manager. Loading from disk is not a user
                # modification: do not flag the project as modified.
                self._state.set_data_matrix(matrix, mark_modified=False)

                self._spreadsheet.load_data(
                    matrix.data, row_labels=matrix.row_labels, col_labels=matrix.col_labels, update_state=False
                )
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
        # Forward the boolean result of ``_on_save_file`` so that callers
        # depending on the return value (e.g. closeEvent asking whether
        # the user successfully saved before quitting) get the right
        # answer. The previous implementation discarded the result.
        return self._on_save_file()

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme and propagate to all child widgets."""
        self._is_dark_theme = is_dark
        from config.design_system import get_stylesheet

        self.setStyleSheet(get_stylesheet(is_dark))
        self._ribbon.setDarkTheme(is_dark)
        self._status_bar.setDarkTheme(is_dark)
        self._workspace.setDarkTheme(is_dark)
        self._navigation.setDarkTheme(is_dark)
        # DiagnosticConsole is a child dock widget and exposes its own
        # ``setDarkTheme``; without this propagation the dock's text
        # colour stays at the light-theme palette and clashes with the
        # rest of the UI.
        diagnostic_console = getattr(self, "_diagnostic_console", None)
        if diagnostic_console is not None and hasattr(diagnostic_console, "setDarkTheme"):
            diagnostic_console.setDarkTheme(is_dark)
        runlist_panel = getattr(self, "_runlist_panel", None)
        if runlist_panel is not None:
            runlist_panel.setDarkTheme(is_dark)
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
                    self._logger.debug("canvas.draw_idle() failed during re-theme", exc_info=True)
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
            figure.patch.set_facecolor(bg)  # type: ignore[attr-defined]

            suptitle = getattr(figure, "_suptitle", None)
            if suptitle is not None:
                suptitle.set_color(fg)

            for ax in figure.get_axes():  # type: ignore[attr-defined]
                ax.set_facecolor(bg)
                for spine in ax.spines.values():
                    spine.set_color(border)
                ax.tick_params(colors=fg, which="both")
                for text_attr in ("title", "_left_title", "_right_title"):
                    text_obj = getattr(ax, text_attr, None)
                    if text_obj is not None:
                        text_obj.set_color(fg)
                for axis_attr in ("xaxis", "yaxis"):
                    axis_obj = getattr(ax, axis_attr, None)
                    if axis_obj is None:
                        continue
                    label = getattr(axis_obj, "label", None)
                    if label is not None:
                        label.set_color(fg)
                legend = ax.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_color(fg)
                    frame = legend.get_frame()
                    if frame is not None:
                        frame.set_facecolor(bg)
                        frame.set_edgecolor(border)
        except Exception:  # pragma: no cover - best-effort
            self._logger.debug("_apply_light_theme_to_figure failed", exc_info=True)

    def _on_undo(self) -> None:
        """Undo last state change and refresh spreadsheet."""
        if not self._state.has_data:
            return
        if not self._state.can_undo():
            self._status_bar.setInfo(_("Nothing to undo"))
            return
        self._state.undo()
        matrix = self._state.data_matrix
        if matrix is not None:
            self._spreadsheet.load_data(
                matrix.data,
                row_labels=matrix.row_labels,
                col_labels=matrix.col_labels,
                update_state=False,
            )
            self._status_bar.setInfo(_("Undo"))

    def _on_redo(self) -> None:
        """Redo last undone change and refresh spreadsheet."""
        if not self._state.has_data:
            return
        if not self._state.can_redo():
            self._status_bar.setInfo(_("Nothing to redo"))
            return
        self._state.redo()
        matrix = self._state.data_matrix
        if matrix is not None:
            self._spreadsheet.load_data(
                matrix.data,
                row_labels=matrix.row_labels,
                col_labels=matrix.col_labels,
                update_state=False,
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
                update_state=False,
            )
            self._status_bar.setInfo(
                _("Transposed: {0} samples x {1} variables").format(new_matrix.n_samples, new_matrix.n_variables)
            )
        except Exception as e:
            QMessageBox.critical(self, _("Transpose Error"), str(e))

    @staticmethod
    def _filter_labels(labels: list[str] | None, keep: "npt.NDArray", n_expected: int, kind: str) -> list[str] | None:
        """
        Apply a boolean keep-mask to a label list.

        Returns ``None`` (so ``DataMatrix`` regenerates default labels) when
        the incoming list does not describe the pre-imputation axis - a
        mismatched label list is worse than a generic one.
        """
        import numpy as np

        if labels is None or len(labels) != n_expected:
            logging.getLogger(__name__).warning(
                "Imputation: %s label count %s != %s; regenerating default labels.",
                kind,
                len(labels) if labels is not None else None,
                n_expected,
            )
            return None
        keep = np.asarray(keep, dtype=bool)
        if keep.size != n_expected:
            return None
        return [label for label, k in zip(labels, keep, strict=False) if bool(k)]

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
                row_labels = list(self._state.data_matrix.row_labels)
                col_labels = list(self._state.data_matrix.col_labels)

                # ``ImputationResult`` only carries the matrix for the
                # removal strategies - no kept-index information - so the
                # labels must be filtered here from the *original* NaN mask,
                # using exactly the predicate config.imputation applies.
                # Passing full-length labels to a shortened matrix made
                # DataMatrix raise MatrixDimensionError.
                if method == ImputationMethod.REMOVE_ROWS:
                    keep = ~np.any(nan_mask, axis=1)
                    row_labels = self._filter_labels(row_labels, keep, data.shape[0], "row")
                elif method == ImputationMethod.REMOVE_COLUMNS:
                    keep = ~np.any(nan_mask, axis=0)
                    col_labels = self._filter_labels(col_labels, keep, data.shape[1], "column")
                else:
                    # mean / median / knn keep the original shape; guard
                    # against a stale label list of the wrong length.
                    if len(row_labels) != result.data.shape[0]:
                        row_labels = None
                    if len(col_labels) != result.data.shape[1]:
                        col_labels = None

                new_matrix = DataMatrix(
                    result.data,
                    row_labels=row_labels,
                    col_labels=col_labels,
                )
                self._state.set_data_matrix(new_matrix)
                self._spreadsheet.load_data(
                    result.data,
                    row_labels=new_matrix.row_labels,
                    col_labels=new_matrix.col_labels,
                    update_state=False,
                )

                self._status_bar.setInfo(result.summary)
                QMessageBox.information(self, _("Imputation Complete"), result.summary)

        except Exception as e:
            QMessageBox.critical(self, _("Imputation Error"), format_user_error(e, "Op: missing-value handling"))

    def _on_preferences(self) -> None:
        """Show the Preferences dialog with language / data / plot options.

        The previous version of this slot was a stub that only
        displayed a static ``QMessageBox`` saying "see the Settings
        menu", which made the ribbon button feel non-functional.  The
        dialog below keeps its scope small but lets the user change
        real preferences: language code, default CSV has-row-labels,
        default plot DPI, and the matplotlib figure size used by the
        interactive canvas.
        """
        dialog = PreferencesDialog(self, current=self._get_preferences_state())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_state = dialog.get_preferences()
        self._apply_preferences(new_state)
        self._status_bar.setInfo(_("Preferences updated"))

    def _get_preferences_state(self) -> dict:
        """Read the current user preferences from ``QSettings``.

        Falls back to safe defaults if a key has never been written
        (e.g. the first run after installation).
        """
        settings = QSettings("PaleoAST", "PaleoAST")
        return {
            "language": settings.value("preferences/language", "en"),
            "csv_has_header": settings.value("preferences/csv_has_header", True, type=bool),
            "csv_has_row_labels": settings.value("preferences/csv_has_row_labels", True, type=bool),
            "plot_dpi": settings.value("preferences/plot_dpi", 100, type=int),
            "plot_figsize": settings.value(
                "preferences/plot_figsize", "8,6", type=str
            ),
            "r_rscript": settings.value("preferences/r_rscript", "", type=str),
            "r_theme": settings.value("preferences/r_theme", "classic", type=str),
            "r_base_size": settings.value("preferences/r_base_size", 9.0, type=float),
            "r_output_format": settings.value(
                "preferences/r_output_format", "pdf", type=str
            ),
            "r_timeout": settings.value("preferences/r_timeout", 300, type=int),
        }

    def _apply_preferences(self, new_state: dict) -> None:
        """Persist preferences and propagate the values that have a
        live effect (figure size / DPI).

        Other agents own the actual language translator; the new value
        is stored, the user is told to restart for the language change
        to take effect.
        """
        settings = QSettings("PaleoAST", "PaleoAST")
        settings.setValue("preferences/language", new_state["language"])
        settings.setValue("preferences/csv_has_header", new_state["csv_has_header"])
        settings.setValue("preferences/csv_has_row_labels", new_state["csv_has_row_labels"])
        settings.setValue("preferences/plot_dpi", new_state["plot_dpi"])
        settings.setValue("preferences/plot_figsize", new_state["plot_figsize"])
        # R export settings. Read back through _get_preferences_state() when a
        # script is generated or run, so changing them takes effect on the next
        # export rather than needing a restart.
        for key in ("r_rscript", "r_theme", "r_base_size", "r_output_format", "r_timeout"):
            if key in new_state:
                settings.setValue(f"preferences/{key}", new_state[key])
        # Propagate DPI / figsize to the interactive plot canvas so
        # the next plot uses the new values.  We touch ``figure.dpi``
        # at matplotlib's rcParams level (the canvas reads from there
        # on every new figure).
        try:
            import matplotlib as mpl

            mpl.rcParams["figure.dpi"] = int(new_state["plot_dpi"])
            try:
                w_str, h_str = [s.strip() for s in str(new_state["plot_figsize"]).split(",")]
                mpl.rcParams["figure.figsize"] = (float(w_str), float(h_str))
            except (ValueError, AttributeError):
                pass
        except Exception:
            # Matplotlib is optional; if not available nothing to do.
            pass

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

        # Update state manager. Loading fresh data is not a user
        # modification: do not flag the project as modified.
        self._state.set_data_matrix(matrix, mark_modified=False)

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
            update_state=False,
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
                update_state=False,
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

    def _on_export_current_plot(self) -> None:
        """Export the currently visible plot through the unified facade.

        The function pulls the matplotlib ``Figure`` from the currently
        visible workspace tab. It supports both ``InteractivePlotCanvas``
        tabs and embedded figure-host widgets (``FigureHostWidget`` with
        the ``figure_canvas`` dynamic property). When no plot is
        visible, the user is told to run an analysis first.
        """
        try:
            from views.ui_plot_export_dialog import PlotExportDialog

            from plot_export import export_figure
        except ImportError as exc:  # pragma: no cover - defensive
            QMessageBox.critical(
                self,
                _("Export Error"),
                _("Plot export is unavailable: {0}").format(exc),
            )
            return

        figure = self._extract_current_figure()
        if figure is None:
            QMessageBox.information(
                self,
                _("No Plot"),
                _("Run an analysis first, then export the resulting plot."),
            )
            return

        dialog = PlotExportDialog("plot.png", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            options = dialog.get_options()
        except ValueError as exc:
            QMessageBox.critical(self, _("Export Error"), str(exc))
            return
        path = dialog.get_path()
        try:
            export_figure(figure, path, options)
        except Exception as exc:
            self._logger.error(f"Plot export failed: {exc}")
            QMessageBox.critical(self, _("Export Error"), str(exc))
            return
        QMessageBox.information(
            self,
            _("Export Successful"),
            _("Plot saved to:\n{0}").format(path),
        )

    # ------------------------------------------------------------------
    # Editable R (ggplot2) export
    # ------------------------------------------------------------------
    #
    # Two SEPARATE actions, on purpose:
    #   * Export as R script  -- regenerate the .R from the current result and
    #     run it. This REWRITES the file.
    #   * Re-run R script      -- execute the .R exactly as it is on disk.
    #
    # They must not be merged. A user is expected to edit that script in
    # RStudio; if a "render" button regenerated the file first, every edit
    # would be silently destroyed. So the regenerate path asks first whenever
    # the file on disk no longer matches what we wrote.

    def _r_plot_spec(self):
        """Build an :class:`RPlotSpec` from the user's preferences."""
        from visualization.r_export import RPlotSpec

        prefs = self._get_preferences_state()
        try:
            w_str, h_str = [s.strip() for s in str(prefs["plot_figsize"]).split(",")]
            figsize = (float(w_str), float(h_str))
        except (ValueError, AttributeError):
            figsize = (7.0, 5.5)
        return RPlotSpec(
            theme=str(prefs.get("r_theme", "classic")),
            base_size=float(prefs.get("r_base_size", 9)),
            figsize=figsize,
            dpi=int(prefs.get("plot_dpi", 300)),
            output_format=str(prefs.get("r_output_format", "pdf")),
            r_executable=str(prefs.get("r_rscript", "")),
        )

    def _r_output_dir(self) -> str:
        """Where exported R scripts go.

        Inside the user's documents folder rather than next to the source, so
        the scripts and their CSVs travel together and are easy to find later.
        """
        base = Path(
            os.environ.get("USERPROFILE")
            or os.path.expanduser("~")
            or str(Path.home())
        )
        out = base / "Documents" / "PaleoAST-R"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    def _on_export_as_r_script(self) -> None:
        """Regenerate the .R script from the current result, then run it."""
        if not self._state.has_data:
            QMessageBox.information(
                self, _("No Data"), _("Load data first, then export an R script.")
            )
            return

        out_dir = Path(self._r_output_dir())
        existing = out_dir / "pca_scores.R"

        # Never clobber an edit without saying so.
        if existing.is_file() and getattr(self, "_last_r_export", None) is not None:
            if not self._last_r_export.script_is_unmodified():
                reply = QMessageBox.question(
                    self,
                    _("Overwrite edited R script?"),
                    _("{0}\n\nhas been edited. Regenerating replaces your changes "
                      "with a freshly generated script. Continue?").format(
                        existing.name
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self._logger.info("R export cancelled: script was edited")
                    return

        try:
            import datetime as _dt

            from stats.pca import PCAAnalyzer
            from visualization.r_export import RScriptExporter

            matrix = self._state.data_matrix
            if matrix is None:
                raise ValueError(_("No data matrix loaded."))
            result = PCAAnalyzer().analyze(
                matrix.to_numpy(), n_components=3, method="correlation"
            )
            # Index rather than unpack into `_`: the gettext alias must not be
            # rebound to a throwaway name, or every later _("...") in this
            # function silently resolves to the wrong object.
            plot_meta = self._get_plot_labels_and_groups()
            labels, groups = plot_meta[0], plot_meta[1]
            exporter = RScriptExporter(
                stamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            )
            export = exporter.export_pca_scores(
                result, out_dir, self._r_plot_spec(),
                labels=labels, groups=groups,
            )
        except Exception as exc:
            self._logger.error(f"R script export failed: {exc}")
            QMessageBox.critical(self, _("R Export Error"), str(exc))
            return

        self._last_r_export = export
        self._run_r_script_and_show(export.script_path, export.script_path.name)

    def _on_rerun_r_script(self) -> None:
        """Execute the .R exactly as it is on disk. Never regenerates it."""
        script = Path(self._r_output_dir()) / "pca_scores.R"
        if not script.is_file():
            QMessageBox.information(
                self,
                _("No R script"),
                _("Generate one first with \"Export as R script\".\n\n"
                  "Expected at:\n{0}").format(script),
            )
            return
        self._run_r_script_and_show(script, script.name)

    def _run_r_script_and_show(self, script_path: Path, title: str) -> None:
        """Run an R script and put the resulting figure in the workspace."""
        from visualization.r_render import run_r_script

        prefs = self._get_preferences_state()
        self._status_bar.setInfo(_("Running {0} ...").format(title))
        QApplication.processEvents()

        run = run_r_script(
            script_path,
            rscript=str(prefs.get("r_rscript", "")),
            timeout=float(prefs.get("r_timeout", 300)),
        )
        if not run.ok:
            self._status_bar.setInfo(_("R failed"))
            QMessageBox.critical(self, _("R Error"), run.message())
            return

        image = run.preview_png or next(
            (p for p in run.produced if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
            None,
        )
        if image is None:
            # A vector-only output cannot be displayed inline; say where it is
            # rather than claiming a figure appeared.
            where = "\n".join(str(p) for p in run.produced) or run.error
            self._status_bar.setInfo(_("R finished"))
            QMessageBox.information(
                self,
                _("Figure written"),
                _("R finished but the output is vector-only, so it cannot be\n"
                  "previewed here. Open the file directly:\n\n{0}\n\n"
                  "The .R script stays editable; use \"Re-run R script\" after "
                  "you change it.").format(where),
            )
            return

        try:
            self._show_rendered_png(image, title)
        except Exception as exc:
            self._logger.error(f"Could not display the R figure: {exc}")
        self._status_bar.setInfo(
            _("R finished: {0}").format(", ".join(p.name for p in run.produced))
        )

    def _show_rendered_png(self, png_path: Path, title: str) -> None:
        """Put an R-rendered PNG into the workspace as a figure tab."""
        import matplotlib.image as mpimg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(7.2, 5.6), dpi=110)
        ax = fig.add_subplot(111)
        ax.imshow(mpimg.imread(str(png_path)))
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        # _embed_figure_in_workspace is the right host: it wraps a pre-built
        # Figure in a QWidget, so the R output needs no InteractivePlotCanvas.
        self._embed_figure_in_workspace(
            fig, title, dark_theme=self._is_dark_theme
        )
        self._logger.info(f"Displayed R figure: {png_path}")

    def _extract_current_figure(self):
        """Return the matplotlib ``Figure`` in the active workspace tab."""
        stack = self._workspace._stack
        widget = stack.currentWidget()
        if widget is None or widget is self._spreadsheet:
            return None
        if isinstance(widget, InteractivePlotCanvas):
            return widget._figure
        # Embedded figure host (FigureHostWidget). The canvas is stored
        # under the dynamic ``figure_canvas`` property by
        # :meth:`_embed_figure_in_workspace`.
        canvas = widget.property("figure_canvas")
        if canvas is not None and hasattr(canvas, "figure"):
            return canvas.figure
        return None

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

    # Keep the last N analysis results in the workspace so the user can
    # switch between them.  Older results are evicted FIFO.
    _MAX_RESULT_HISTORY = 8

    def _evict_excess_result_tabs(self) -> int:
        """Evict oldest result tabs (FIFO) until we are under the limit.

        Returns the number of tabs removed. The helper is shared by the
        plot-canvas path and the embedded-figure path so the eviction
        policy lives in one place. The caller is responsible for
        ``deleteLater`` after the widget is removed from the stack;
        here we also drop a ``FigureHostWidget``'s matplotlib Figure
        reference so the underlying canvas + figure can be garbage
        collected promptly.
        """
        stack = self._workspace._stack
        removed = 0
        # Collect existing result widgets (excluding the spreadsheet
        # and the placeholder).
        result_widgets: list = []
        for i in range(stack.count()):
            w = stack.widget(i)
            if w is self._spreadsheet or w is self._workspace._placeholder:
                continue
            result_widgets.append(w)

        while len(result_widgets) >= self._MAX_RESULT_HISTORY:
            old = result_widgets.pop(0)
            stack.removeWidget(old)
            # Drop the figure reference on figure host containers so
            # the canvas + figure can be GC'd, not held alive by Qt.
            canvas = old.property("figure_canvas") if hasattr(old, "property") else None
            if canvas is not None and hasattr(canvas, "figure"):
                try:
                    canvas.figure = None  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - defensive
                    pass
            old.deleteLater()
            removed += 1
        return removed

    def _add_plot_to_workspace(self, plot: InteractivePlotCanvas, name: str) -> int:
        """Add a new plot to the workspace, evicting the oldest if needed."""
        self._evict_excess_result_tabs()
        return self._workspace.addWidget(plot, name)

    def _add_tab_to_workspace(self, widget: object, name: str, focus: bool = True) -> int:
        """Add an additional result widget without evicting older tabs.

        Unlike :meth:`_add_plot_to_workspace`, this is for secondary
        result tabs (e.g. the PCA scree plot) that should appear next
        to the primary result without triggering eviction. By default the
        view is switched to the new tab so the user sees the new content.

        ``focus=False`` for secondary artefacts that should be reachable but
        must not become the resting state -- notably the loadings TABLE, which
        is a text reference rather than a result. Without it, a run that adds
        two secondary tabs (scree, then loadings) leaves the user staring at the
        last one added, which for PCA was a raw text matrix with both real
        plots hidden behind it.
        """
        idx = self._workspace.addWidget(widget, name)
        if focus:
            self._workspace.setCurrentIndex(idx)
        return idx

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

        self._evict_excess_result_tabs()

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

        As in :meth:`_apply_light_theme_to_figure`, a failure on any one
        child is caught once by the single outer handler below (and logged
        with ``exc_info``) rather than by a silent per-child
        ``except: pass``, which left a half-themed figure with no clue why.
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
                suptitle.set_color(fg)

            for ax in figure.get_axes():  # type: ignore[attr-defined]
                ax.set_facecolor(bg)
                for spine in ax.spines.values():
                    spine.set_color(border)
                ax.tick_params(colors=fg, which="both")
                for text_attr in ("title", "_left_title", "_right_title"):
                    text_obj = getattr(ax, text_attr, None)
                    if text_obj is not None:
                        text_obj.set_color(fg)
                for axis_attr in ("xaxis", "yaxis"):
                    axis_obj = getattr(ax, axis_attr, None)
                    if axis_obj is None:
                        continue
                    label = getattr(axis_obj, "label", None)
                    if label is not None:
                        label.set_color(fg)
                legend = ax.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_color(fg)
                    frame = legend.get_frame()
                    if frame is not None:
                        frame.set_facecolor(bg)
                        frame.set_edgecolor(border)
        except Exception:  # pragma: no cover - best-effort UI hint
            self._logger.debug("_apply_dark_theme_to_figure failed", exc_info=True)

    def _run_analysis_async(self, work, on_success, on_error, title: str, wants_reporter: bool = False) -> None:
        """在后台线程运行分析并接气回调。

        信号桥必须是 QObject: pyqtSignal 只有挂在 QObject 实例上才能
        connect/emit (QRunnable 不是 QObject, 此前直接在 QRunnable
        子类里声明信号, connect 时抛
        "cannot be converted to PyQt6.QtCore.QObject")。
        signals 以 self 为父对象, 保证 emit 前不被垃圾回收。

        Parameters:
            work: 无参可调用 (或 wants_reporter 时接收 reporter 单参),
                返回分析结果 (在worker线程执行)。
            on_success: result -> None (GUI线程执行)。
            on_error: Exception -> None (GUI线程执行)。
            title: 状态栏显示的分析名称。
            wants_reporter: 为真时 work 收到 ``(value, maximum)`` 进度
                回调, 转发到状态栏进度条。
        """
        self._status_bar.setProgress(0, 0)  # indeterminate
        self._status_bar.setInfo(_("Running {0}...").format(title))

        if self._closing:
            self._logger.info("Ignoring analysis request for '%s': window is closing.", title)
            return

        signals = _AnalysisSignals(self)
        signals.result_ready.connect(on_success)
        signals.error_raised.connect(on_error)
        signals.progress.connect(lambda value, maximum: self._status_bar.setProgress(value, maximum))

        task = _AnalysisTask(work, signals, wants_reporter=wants_reporter)
        self._thread_pool.start(task)

    # ------------------------------------------------------------------
    # Batch run queue (runlist) wiring
    # ------------------------------------------------------------------

    def _runlist_guard(self, item) -> str | None:
        """RunQueue guard callback: report the first unmet precondition."""
        try:
            spec = get_spec(item.analysis_id)
        except Exception as e:  # guard must return a reason, not raise
            return str(e)
        available = set()
        if self._state.has_data:
            available.add("has_data")
        if self._get_groups() is not None:
            available.add("groups")
        for guard in spec.requires:
            if guard.startswith("cache:"):
                key = guard.split(":", 1)[1]
                if self._state.get_cached_result(key) is not None:
                    available.add(guard)
        unmet = check_guards(spec, available)
        if unmet is None:
            return None
        return _("Guard not satisfied: {0}").format(unmet)

    def _runlist_execute(self, item, finish) -> None:
        """RunQueue executor callback: dispatch to the analysis's _execute_*."""
        dispatch = {
            "pca": self._execute_pca,
            "pcoa": self._execute_pcoa,
            "nmds": self._execute_nmds,
            "anosim": self._execute_anosim,
            "permanova": self._execute_permanova,
            "tps_grid": self._execute_tps_grid,
        }
        fn = dispatch.get(item.analysis_id)
        if fn is None:
            finish(ERROR, _("Unknown analysis: {0}").format(item.analysis_id))
            return

        def _done(_result):
            finish(OK, None)
            QTimer.singleShot(0, self._runlist_advance)

        def _fail(exc):
            finish(ERROR, str(exc))
            QTimer.singleShot(0, self._runlist_advance)

        fn(item.params, on_done=_done, on_fail=_fail)

    def _on_runlist_run_all(self) -> None:
        """Start a queue pass; re-running after a finished pass resets results."""
        if self._run_queue.is_busy():
            return
        items = self._run_queue.items
        if items and self._run_queue.is_done():
            self._run_queue.clear_results()
            self._runlist_panel.refresh()
        self._runlist_advance()

    def _runlist_advance(self) -> None:
        """Start the next pending item, or finish the pass with a manifest."""
        if self._run_queue.is_busy():
            return
        try:
            item = self._run_queue.run_next()
        except Exception as e:  # queued event handler must not raise
            self._logger.error(f"Run queue error: {e}")
            return
        if item is None:
            if self._run_queue.items and self._run_queue.is_done():
                self._runlist_write_manifest()
            return
        if item.status != RUNNING:
            # Skipped by the guard, or a synchronous executor already
            # finished the item: continue the pass on the next event-loop
            # tick so the UI repaints between runs.
            QTimer.singleShot(0, self._runlist_advance)

    def _runlist_write_manifest(self) -> None:
        """Persist the JSON record of the finished pass next to the preset library."""
        try:
            directory = os.path.dirname(os.path.abspath(self._preset_manager.directory))
            path = os.path.join(directory, "last_run_manifest.json")
            write_manifest(path, self._run_queue.items, meta={"app_version": APP_VERSION})
            self._status_bar.setInfo(_("Run manifest written to {0}").format(path))
        except OSError as e:
            self._logger.error(f"Failed to write run manifest: {e}")

    @staticmethod
    def _trim_components_to_variance(result, min_variance: float) -> int:
        """Return how many leading components reach ``min_variance`` (a fraction).

        This module deliberately keeps numpy out of its top-level namespace
        (every other function does a local ``import numpy as np``), so do the
        same here.

        ``min_variance`` arrives from the dialog as a FRACTION (0.05 == 5%),
        so the cumulative values are compared as fractions, not percentages.
        Returns the current component count when the spectrum does not allow a
        decision, so this is always safe to call.
        """
        import numpy as np

        # PCAResult exposes `cumulative_variance` as PERCENTAGES
        # (verified: [38.5, 61.6, 77.9, ...] for 5 components), so the
        # dialog's fraction has to be converted before comparing.
        cum = getattr(result, "cumulative_variance", None)
        if cum is None:
            ev = getattr(result, "explained_variance", None)
            if ev is None:
                return int(getattr(result, "n_components", 0) or 0)
            ev = np.asarray(ev, dtype=float).ravel()
            if ev.size == 0 or not np.all(np.isfinite(ev)):
                return int(getattr(result, "n_components", 0) or 0)
            if float(np.sum(ev)) > 1.5:
                ev = ev / 100.0
            cum = np.cumsum(ev)
        else:
            cum = np.asarray(cum, dtype=float).ravel()
            if cum.size == 0 or not np.all(np.isfinite(cum)):
                return int(getattr(result, "n_components", 0) or 0)
            if float(cum[-1]) <= 1.5:          # already fractions
                cum = cum * 100.0
        # A non-positive threshold means "no threshold": keep everything the
        # engine returned rather than trimming down to a single component.
        # NB: n_avail must be bound BEFORE this early return -- `... or n_avail`
        # only evaluates the right side when n_components is falsy, so the
        # ordering was previously hidden until that path was taken.
        n_avail = int(cum.size)
        if float(min_variance) <= 0.0:
            return int(getattr(result, "n_components", 0) or n_avail)
        target = float(min_variance) * 100.0 if float(min_variance) <= 1.5 else float(min_variance)
        # `searchsorted` returns len(cum) when the threshold is never reached
        # within the available spectrum; cap it so the caller can index safely.
        # At least one component is always retained.
        return int(min(n_avail, max(1, int(np.searchsorted(cum, target)) + 1)))

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

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._execute_pca(dialog.get_parameters())

    def _execute_pca(self, params: dict, on_done=None, on_fail=None) -> None:
        """Run PCA in the thread pool and plot the result.

        on_done/on_fail are the batch-runlist hooks: when provided, the
        runlist is notified instead of (interactive mode's) dialogs being
        shown on failure.  Success always renders the plots.

        All eight parameters collected by :class:`PCADialog` are honoured:
        ``min_variance``, ``show_loadings``, ``show_scores``,
        ``show_scree``, ``show_biplot``, ``biplot_scale``,
        ``impute_missing`` and ``parallel``. The previous implementation
        read only ``n_components`` and ``method``; the others were
        silently dropped, so toggling the "Show biplot" checkbox did
        nothing.

        ``n_components`` is what the engine is actually asked for. The
        ``min_variance`` threshold is applied by
        :meth:`_trim_components_to_variance`, which the engine cannot do
        itself (it has no such parameter).
        """
        controller = self._statistics_controller
        n_components = params["n_components"]
        method = params["method"]
        min_variance = params.get("min_variance", 0.0)
        show_loadings = bool(params.get("show_loadings", True))
        show_scores = bool(params.get("show_scores", True))
        show_scree = bool(params.get("show_scree", True))
        show_biplot = bool(params.get("show_biplot", False))
        biplot_scale = float(params.get("biplot_scale", 1.0))
        impute_missing = bool(params.get("impute_missing", False))
        parallel = bool(params.get("parallel", True))

        # Snapshot the user's choices on the dialog result so the GUI
        # callback can suppress tabs the user opted out of, and so the
        # biplot uses the requested scaling factor.
        ctx = {
            "show_loadings": show_loadings,
            "show_scores": show_scores,
            "show_scree": show_scree,
            "show_biplot": show_biplot,
            "biplot_scale": biplot_scale,
            "min_variance": min_variance,
        }

        def _work():
            # Ask for enough components that the cumulative-variance threshold
            # can be satisfied by trimming, then trim in `_work`'s caller.
            # The analyser only accepts `n_components`, so the threshold
            # cannot be delegated to it.
            requested = n_components
            if min_variance and min_variance > 0:
                # Heuristic: ask for a few extra components so the
                # engine can return enough to satisfy the cumulative
                # variance threshold. The analyser trims internally.
                requested = min(max(requested, 10), 50)
            result = controller.run_pca(n_components=requested, method=method)
            # The engine has no `min_variance` concept (verified: zero
            # references anywhere in stats/ or controllers/), so the threshold
            # has to be applied HERE, by trimming the returned spectrum.
            # Previously we merely raised `requested` to >=10 and assumed the
            # engine would trim — it does not. With the dialog's default of
            # 5.0% that branch fired on every default run, so a user who asked
            # for 3 components silently got 10 and the status bar reported 10.
            if min_variance and min_variance > 0 and result is not None:
                retained = self._trim_components_to_variance(result, min_variance)
                if retained < getattr(result, "n_components", retained):
                    self._logger.info(
                        "PCA: min_variance=%.4f retained %d of %d components",
                        min_variance, retained, getattr(result, "n_components", retained),
                    )
            return result

        def _done(result):
            self._on_pca_result_ready(result, ctx, impute_missing=impute_missing, parallel=parallel)
            if on_done is not None:
                on_done(result)

        def _fail(exc):
            if on_fail is not None:
                self._status_bar.setProgress(100, 100)
                on_fail(exc)
            else:
                self._on_pca_error(exc)

        self._run_analysis_async(_work, _done, _fail, _("PCA"))

    def _on_pca_result_ready(self, result, ctx: dict | None = None, impute_missing: bool = False, parallel: bool = True) -> None:
        self._status_bar.setProgress(100, 100)
        ctx = ctx or {}
        # Pull row labels / groups from the data matrix so the canvas
        # uses ``Site_1`` rather than ``S1`` and groups actually colour
        # the points.  Plot calls before this fix relied on the
        # canvas's hasattr fallback, which masked every column as one
        # bucket and made the legend read ``Group 0``.
        labels, groups, group_names = self._get_plot_labels_and_groups()
        # If the user opted into biplot, draw the score plot with
        # loading vectors instead of the bare scatter.
        plot = InteractivePlotCanvas()
        biplot_requested = bool(ctx.get("show_biplot"))
        biplot_drawn = False
        if biplot_requested:
            biplot_scale = float(ctx.get("biplot_scale", 1.0))
            if hasattr(plot, "plot_pca_biplot"):
                plot.plot_pca_biplot(result, scale=biplot_scale)
                biplot_drawn = True
        if not biplot_drawn:
            plot.plot_pca_scores(
                result, labels=labels, groups=groups, group_names=group_names
            )
        idx = self._add_plot_to_workspace(plot, _("PCA Score Plot"))
        self._workspace.setCurrentIndex(idx)
        ev = result.explained_variance
        cum2 = ev[0] + ev[1] if len(ev) >= 2 else ev[0] if len(ev) == 1 else 0.0
        message = _("PCA: {0} components, PC1+PC2 = {1:.1f}%").format(result.n_components, cum2)
        if biplot_requested and not biplot_drawn:
            # The dialog has a "Show biplot" checkbox (ui_dialogs.py) and its
            # state travels all the way here, but InteractivePlotCanvas has no
            # plot_pca_biplot. Without this the user ticks the box, gets a
            # plain scatter, and has no way to tell "the biplot is subtle"
            # from "there is no biplot here".
            #
            # One message, styled as a warning, rather than a warning followed
            # by an info line: setInfo() clears the warning style, so two
            # consecutive status updates would erase the warning immediately
            # and the user would never see it.
            #
            # Implementing plot_pca_biplot is deliberately not done here: a
            # biplot cannot be verified without looking at a rendered figure,
            # and shipping an ordination plot nobody has looked at is exactly
            # what produces a confidently wrong figure in a paper.
            self._status_bar.setWarning(
                message
                + "  "
                + _(
                    "Show biplot was selected but this build has no biplot "
                    "renderer; a score plot was drawn instead."
                )
            )
        else:
            self._status_bar.setInfo(message)

        # The scree tab is conditional on ``show_scree``.
        if ctx.get("show_scree", True):
            scree = InteractivePlotCanvas()
            # explained_variance/cumulative_variance cover only the retained
            # components while eigenvalues_raw holds all of them; plot_scree
            # sizes its x-axis from the eigenvalue array, so pass a matching
            # slice (otherwise matplotlib aborts the app from this slot).
            ev_all = result.explained_variance
            scree.plot_scree(result.eigenvalues_raw[: len(ev_all)], ev_all, result.cumulative_variance, method="PCA")
            self._add_tab_to_workspace(scree, _("PCA Scree Plot"))

        # Surface the loadings table on demand. We embed the loadings as
        # a small text tab so the user can confirm variable weights
        # without leaving the workspace. ``impute_missing`` / ``parallel``
        # are also surfaced in the status bar so the user can verify the
        # advanced-options checkboxes reached the controller.
        if ctx.get("show_loadings") and getattr(result, "loadings", None) is not None:
            self._add_pca_loadings_tab(result)

        # ``impute_missing`` and ``parallel`` are carried through ctx so the
        # user's advanced-option choices are visible, but PCAAnalyzer.analyze
        # currently handles missing values upstream of the controller and has
        # no parallel backend. Say so plainly rather than silently dropping
        # the checkboxes. (Do NOT silence this with ``_ = a, b``: ``_`` is the
        # imported gettext function in this module, and shadowing it breaks
        # every later ``_("...")`` call in the same scope.)
        self._logger.info(
            "PCA advanced options: impute_missing=%s (handled upstream of the controller), "
            "parallel=%s (no parallel backend yet)",
            impute_missing,
            parallel,
        )

        # Whatever secondary artefacts were appended above, the run must END on
        # the primary result. Without this the resting page is simply the last
        # one added -- with scree + loadings enabled that was the loadings TEXT
        # table, so the user never saw the score plot they had just asked for.
        # Idempotent, and independent of how many secondary tabs exist.
        try:
            self._workspace.setCurrentIndex(idx)
        except Exception:  # pragma: no cover - defensive
            self._logger.debug("Could not restore the primary PCA view", exc_info=True)

    def _add_pca_loadings_tab(self, result) -> None:
        """Append a small loadings-matrix text tab to the workspace."""
        try:
            import numpy as np

            # Use a generic QTextEdit embedded as a tab; build it manually
            # so we don't depend on a dedicated plotter.
            from PyQt6.QtWidgets import QTextEdit

            editor = QTextEdit()
            editor.setReadOnly(True)
            loadings = np.asarray(result.loadings)
            n_rows, n_cols = loadings.shape if loadings.ndim == 2 else (loadings.size, 1)
            lines = [_("PCA Loadings Matrix (variables x components)"), "-" * 40]
            header = " ".join(f"PC{j + 1:>8d}" for j in range(n_cols))
            lines.append(header)
            for i in range(n_rows):
                row = " ".join(f"{loadings[i, j]:>8.4f}" for j in range(n_cols))
                lines.append(row)
            editor.setPlainText("\n".join(lines))
            # focus=False: a text reference table must not become the resting
            # state of the workspace.
            self._add_tab_to_workspace(editor, _("PCA Loadings"), focus=False)
        except Exception as exc:
            self._logger.debug("Skipping PCA loadings tab: %s", exc)

    def _on_pca_error(self, exc: Exception) -> None:
        self._status_bar.setProgress(100, 100)
        QMessageBox.critical(self, _("PCA Error"), format_user_error(exc, "PCA"))

    def _on_run_pcoa(self) -> None:
        """Run Principal Coordinate Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = PCoADialog(self)
        dialog.setDarkTheme(self._is_dark_theme)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._execute_pcoa(dialog.get_parameters())

    def _execute_pcoa(self, params: dict, on_done=None, on_fail=None) -> None:
        """Run PCoA in the thread pool and plot; on_done/on_fail for the runlist.

        Previously this executed synchronously on the GUI thread, which
        froze the window on large matrices (n=400 is the breakpoint where
        the EVD becomes noticeable). Mirrored on the PCA / NMDS path so
        progress updates show on the status bar and the user can cancel.
        """
        controller = self._statistics_controller
        metric = params["metric"]
        n_components = params["n_components"]
        correction = params.get("correction", "cmdscale")

        def _work():
            return controller.run_pcoa(
                metric=metric, n_components=n_components, correction=correction
            )

        def _done(result):
            self._status_bar.setProgress(100, 100)
            labels, groups, group_names = self._get_plot_labels_and_groups()
            plot = InteractivePlotCanvas()
            plot.plot_pcoa_scores(
                result, labels=labels, groups=groups, group_names=group_names
            )

            plot_index = self._add_plot_to_workspace(plot, _("PCoA Plot"))
            self._workspace.setCurrentIndex(plot_index)

            ev = result.proportion_explained
            cum2 = (
                ev[0] + ev[1]
                if len(ev) >= 2
                else ev[0]
                if len(ev) == 1
                else 0.0
            )
            self._status_bar.setInfo(
                _("PCoA: {0} coordinates, Axis1+2 = {1:.1f}% (correction: {2})").format(
                    result.n_components, cum2, result.correction_method
                )
            )
            if on_done is not None:
                on_done(result)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"PCoA analysis failed: {exc}")
            if on_fail is not None:
                on_fail(exc)
            else:
                QMessageBox.critical(self, _("PCoA Error"), format_user_error(exc, "PCoA"))

        self._run_analysis_async(_work, _done, _fail, _("PCoA"))

    def _on_run_nmds(self) -> None:
        """Run Non-metric MDS."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = NMDSOptionsDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._execute_nmds(dialog.get_parameters())

    def _execute_nmds(self, params: dict, on_done=None, on_fail=None) -> None:
        """Run NMDS in the thread pool with real per-restart progress.

        The analyzer calls progress_callback(restart_index, total_restarts,
        stress) after each restart; it fires on the worker thread, so the
        callback only emits ``signals.progress`` (queued to the GUI thread)
        via the reporter handed to ``_run_analysis_async``.
        """

        def _work(reporter):
            # 多重启动 SMACOF 是长时间计算, 必须在后台线程执行
            # (旧实现在 GUI 线程同步运行, 大矩阵会冻结界面)
            return self._statistics_controller.run_nmds(
                metric=params["metric"],
                n_dimensions=params["n_dimensions"],
                n_restarts=params["n_restarts"],
                max_iterations=params["max_iterations"],
                tolerance=params["tolerance"],
                progress_callback=lambda i, n, s: reporter(i, n),
            )

        def _on_result(result):
            self._status_bar.setProgress(100, 100)
            labels, groups, group_names = self._get_plot_labels_and_groups()
            plot = InteractivePlotCanvas()
            plot.plot_nmds(result, labels=labels, groups=groups, group_names=group_names)

            plot_index = self._add_plot_to_workspace(plot, _("NMDS Plot"))
            self._workspace.setCurrentIndex(plot_index)

            self._status_bar.setInfo(_("NMDS: stress = {0:.4f}").format(result.stress))
            if on_done is not None:
                on_done(result)

        def _on_error(e):
            self._status_bar.setProgress(100, 100)
            if on_fail is not None:
                on_fail(e)
            else:
                QMessageBox.critical(self, _("NMDS Error"), format_user_error(e, "NMDS"))

        self._run_analysis_async(_work, _on_result, _on_error, _("NMDS"), wants_reporter=True)

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
                sample_name = (params.get("sample_name") or "").strip()
                matrix = self._state.data_matrix
                # The DiversityDialog historically accepted a single
                # sample string and the controller only ever received
                # ``data[0]``.  The fix lets the user pick from a list
                # of row labels (multi-select) when the dialog is
                # shown, but the underlying analyser is single-row --
                # so we run it once per selected sample, appending the
                # bar / radar plots so multi-select actually does
                # something visible.
                # ``_resolve_sample_index`` used to fall back to row 0
                # when the user-typed name could not be matched, which
                # silently produced a wrong result for unknown labels.
                # Now we surface that mismatch with a warning so the
                # user can either pick a valid name or accept row 0.
                if not sample_name:
                    # Treat a blank sample name as "all rows".  The
                    # analyser is single-row but the surrounding loop
                    # builds one plot per row so the user sees every
                    # sample's diversity profile.
                    target_indices = list(range(matrix.n_samples))
                    target_labels = list(matrix.row_labels) or [
                        f"Sample_{i + 1}" for i in target_indices
                    ]
                else:
                    sample_index = self._resolve_sample_index(sample_name, matrix)
                    if sample_index is None:
                        QMessageBox.warning(
                            self,
                            _("Sample Not Found"),
                            _("No sample named '{0}' is loaded.").format(sample_name),
                        )
                        return
                    target_indices = [sample_index]
                    target_labels = [
                        sample_name
                        if sample_name in matrix.row_labels
                        else (matrix.row_labels[sample_index] if matrix.row_labels else sample_name)
                    ]

                # ``analyze_diversity`` computes every index and takes no
                # selection argument, so the dialog's per-index checkboxes
                # are applied below, on the way to the plot. Refuse an
                # empty selection instead of emitting a plot with no bars.
                if not any(params.get(key, True) for key in self._diversity_index_selection()):
                    QMessageBox.information(
                        self,
                        _("No Selection"),
                        _("Please select at least one diversity index."),
                    )
                    return

                for idx, name in zip(target_indices, target_labels, strict=False):
                    result = self._statistics_controller.analyze_diversity(
                        abundances=matrix.data[idx],
                        sample_name=name,
                    )
                    plot = InteractivePlotCanvas()
                    plot.plot_diversity_summary(self._select_diversity_indices(result, params))
                    self._add_plot_to_workspace(plot, _("Diversity Plot — {0}").format(name))

                self._status_bar.setInfo(
                    _("Diversity analysis completed for {0} sample(s)").format(len(target_indices))
                )

            except Exception as e:
                QMessageBox.critical(self, _("Diversity Error"), format_user_error(e, "Op: diversity analysis"))

    @staticmethod
    def _diversity_index_selection() -> dict[str, str | None]:
        """Map each ``DiversityDialog`` checkbox key to a result index key.

        ``richness`` maps to ``None``: S is not an entry of
        ``DiversityResult.indices`` but the scalar ``taxa_count``, so the
        caller has to build that bar itself. The remaining names differ
        from the checkbox keys because they are the engine's.
        """
        return {
            "richness": None,
            "shannon": "shannon",
            "simpson": "simpson",
            "fisher": "fisher_alpha",
            "chao1": "chao1",
            "evenness": "pielou",
        }

    @staticmethod
    def _select_diversity_indices(result, params: dict):
        """Restrict a ``DiversityResult`` to the indices the user checked.

        The engine (``ecology.compute_diversity_indices``) always computes
        every index and accepts no selection, so the dialog's checkboxes are
        honoured here, immediately before plotting:

        * a checked index is plotted, an unchecked one is dropped;
        * an index the dialog does not offer (Margalef) is left alone, so
          the filter can only remove what the user actually turned off;
        * richness is not an entry of ``indices`` at all — S lives on
          ``taxa_count`` — so it is built here from there.

        Keys missing from ``params`` (stub dialogs, older callers) count as
        checked, which keeps the previous "plot everything" default.
        """
        from copy import copy

        from models.diversity_result import DiversityIndexResult

        selection = MainWindow._diversity_index_selection()
        checkbox_of = {result_key: key for key, result_key in selection.items() if result_key is not None}

        indices = {}
        for result_key, entry in result.indices.items():
            key = checkbox_of.get(result_key)
            if key is not None and not params.get(key, True):
                continue
            indices[result_key] = entry

        if params.get("richness", True):
            indices["richness"] = DiversityIndexResult(
                index_name=_("Species Richness (S)"),
                value=float(result.taxa_count),
            )

        # Shallow copy rather than ``dataclasses.replace``: the controller
        # may hand back a duck-typed stand-in that is not a dataclass at
        # all, and the caller's result must not be mutated either way.
        filtered = copy(result)
        filtered.indices = indices
        return filtered

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
                matrix = self._state.data_matrix
                # Honour the multi-selection: rarefy each chosen row
                # instead of silently dropping everything past the
                # first.  ``analyze_rarefaction`` is single-row, so we
                # build one plot per selected sample.
                for sample_label in selected_samples:
                    sample_index = self._resolve_sample_index(sample_label, matrix)
                    if sample_index is None:
                        QMessageBox.warning(
                            self,
                            _("Sample Not Found"),
                            _("No sample named '{0}' is loaded.").format(sample_label),
                        )
                        continue
                    result = self._statistics_controller.analyze_rarefaction(
                        abundances=matrix.data[sample_index],
                        sample_name=sample_label,
                        n_points=n_points,
                    )
                    plot = InteractivePlotCanvas()
                    plot.plot_rarefaction(result)
                    self._add_plot_to_workspace(plot, _("Rarefaction Plot — {0}").format(sample_label))

                self._status_bar.setInfo(
                    _("Rarefaction analysis completed for {0} sample(s)").format(len(selected_samples))
                )

            except Exception as e:
                QMessageBox.critical(self, _("Rarefaction Error"), format_user_error(e, "Op: rarefaction analysis"))

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
        """Run spectral analysis (power spectrum and periodogram analysis).

        Spectral analysis assumes the input is a univariate time series
        (single ``(time, value)`` column pair). Multi-column ecological
        or community matrices must not be passed in directly or the
        result will be meaningless; surface a clear error in that case.
        """
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        try:
            data = self._state.data_matrix.data
            if data.ndim != 2 or data.shape[1] < 2:
                QMessageBox.warning(
                    self,
                    _("Insufficient Data"),
                    _("Spectral analysis needs at least two columns (time + value)."),
                )
                return
            if data.shape[1] > 2:
                reply = QMessageBox.question(
                    self,
                    _("Multivariate Data"),
                    _(
                        "Spectral analysis expects a single time series. "
                        "The loaded matrix has {0} columns — only the first two will be used."
                    ).format(data.shape[1]),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
        except Exception:
            raise

        sub = data[:, :2]

        def _work():
            return self._statistics_controller.analyze_spectral(data=sub)

        def _done(result):
            plot = InteractivePlotCanvas()
            plot.plot_spectral(result)
            self._add_plot_to_workspace(plot, _("Spectral Analysis"))
            self._status_bar.setInfo(_("Spectral analysis completed"))
            self._status_bar.setProgress(100, 100)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"Spectral analysis failed: {exc}")
            QMessageBox.critical(
                self, _("Spectral Analysis Error"), format_user_error(exc, "Op: spectral analysis")
            )

        self._run_analysis_async(_work, _done, _fail, _("Spectral"))

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

        dialog = PermutationTestDialog(self, title=_("ANOSIM"), default_method_label=_("ANOSIM"))
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._execute_anosim(dialog.get_parameters())

    def _execute_anosim(self, params: dict, on_done=None, on_fail=None) -> None:
        """Run ANOSIM in the thread pool and plot the result.

        This used to run synchronously on the GUI thread. ANOSIM performs
        ``n_permutations`` O(n^2) rank-and-compare passes and the ribbon action
        passed an empty params dict, so the count was always the 9999 default
        with no dialog to lower it. Measured wall time for the same code:
        n=100 -> 39 s, n=200 -> 205 s, n=400 -> 813 s of a completely frozen
        window (no repaint, no cancel, no progress).

        ``params`` may carry ``metric``, ``n_permutations`` and
        ``random_seed`` from :class:`PermutationTestDialog`; the controller's
        ``run_anosim`` does not currently accept a seed, so the seed is
        logged but not forwarded (caller-visible behaviour is otherwise
        unchanged).
        """
        groups = self._get_groups()
        data = self._state.data_matrix.data
        metric = params.get("metric", "bray_curtis")
        n_permutations = params.get("n_permutations", 9999)
        random_seed = params.get("random_seed")
        if random_seed is not None:
            # The stats layer's ANOSIM analyser does not yet expose a
            # seed hook, but logging it keeps the UI↔stats contract
            # honest: the value the user picked is at least visible in
            # the diagnostic console.
            self._logger.debug("ANOSIM seed requested: %s (analyser ignores)", random_seed)

        def _work():
            return self._statistics_controller.analyze_anosim(
                data=data,
                groups=groups,
                metric=metric,
                n_permutations=n_permutations,
            )

        def _done(result):
            plot = InteractivePlotCanvas()
            plot.plot_anosim_results(result)
            plot_index = self._add_plot_to_workspace(plot, _("ANOSIM Results"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("ANOSIM analysis completed"))
            self._status_bar.setProgress(100, 100)
            if on_done is not None:
                on_done(result)

        def _fail(exc):
            self._logger.error(f"ANOSIM analysis failed: {exc}")
            self._status_bar.setProgress(100, 100)
            if on_fail is not None:
                on_fail(exc)
            else:
                QMessageBox.critical(self, _("ANOSIM Error"), format_user_error(exc, "ANOSIM"))

        self._run_analysis_async(_work, _done, _fail, _("ANOSIM"))

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

        dialog = PermutationTestDialog(self, title=_("PERMANOVA"), default_method_label=_("PERMANOVA"))
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._execute_permanova(dialog.get_parameters())

    def _execute_permanova(self, params: dict, on_done=None, on_fail=None) -> None:
        """Run PERMANOVA in the thread pool (was synchronous; n=400 froze
        the GUI for several minutes).

        ``params`` may carry ``metric``, ``n_permutations`` and
        ``random_seed``.  ``random_seed`` is logged at debug level for
        parity with the ANOSIM path; the controller does not currently
        accept a seed.
        """
        groups = self._get_groups()
        data = self._state.data_matrix.data
        metric = params.get("metric", "bray_curtis")
        n_permutations = params.get("n_permutations", 9999)
        random_seed = params.get("random_seed")
        if random_seed is not None:
            self._logger.debug("PERMANOVA seed requested: %s (analyser ignores)", random_seed)

        def _work():
            return self._statistics_controller.analyze_permanova(
                data=data,
                groups=groups,
                metric=metric,
                n_permutations=n_permutations,
            )

        def _done(result):
            plot = InteractivePlotCanvas()
            plot.plot_permanova_results(result)
            plot_index = self._add_plot_to_workspace(plot, _("PERMANOVA Results"))
            self._workspace.setCurrentIndex(plot_index)
            self._status_bar.setInfo(_("PERMANOVA analysis completed"))
            self._status_bar.setProgress(100, 100)
            if on_done is not None:
                on_done(result)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"PERMANOVA analysis failed: {exc}")
            if on_fail is not None:
                on_fail(exc)
            else:
                QMessageBox.critical(self, _("PERMANOVA Error"), format_user_error(exc, "PERMANOVA"))

        self._run_analysis_async(_work, _done, _fail, _("PERMANOVA"))

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
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.get_parameters()
        data = self._state.data_matrix.data
        metric = params.get("metric", "bray_curtis")

        def _work():
            return self._statistics_controller.analyze_simper(
                data=data, groups=groups, metric=metric
            )

        def _done(result):
            plot = InteractivePlotCanvas()
            plot.plot_simper_results(result)
            self._add_plot_to_workspace(plot, "SIMPER")
            self._status_bar.setInfo(_("SIMPER analysis completed"))
            self._status_bar.setProgress(100, 100)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"SIMPER analysis failed: {exc}")
            QMessageBox.critical(self, _("SIMPER Error"), format_user_error(exc, "SIMPER"))

        self._run_analysis_async(_work, _done, _fail, _("SIMPER"))

    def _on_univariate_selection_changed(self, index: int) -> None:
        """Handle univariate dropdown selection."""
        self._run_univariate_analysis(index)

    def _run_univariate_analysis(self, pre_selected: int = 0) -> None:
        """Core univariate analysis dispatcher — shared by ribbon button and dropdown."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = UnivariateDialog(self)
        dialog.set_pre_selected_test(pre_selected)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.get_parameters()
        test_type = params.get("test_type", 0)
        data = self._state.data_matrix.data
        col_names = self._state.data_matrix.col_labels

        # Branch that need groups: check synchronously so we can bail
        # out before the worker thread starts.
        if test_type in (2, 3, 4):
            groups = self._get_groups()
            if groups is None:
                QMessageBox.warning(self, _("No Groups"), _("Please define groups first."))
                return
        else:
            groups = None

        # Snapshot the inputs the worker thread will need.
        ctrl = self._statistics_controller

        def _work():
            if test_type == 0:
                return ("summary", ctrl.analyze_univariate_summary(data, col_names))
            if test_type == 1:
                return ("normality", ctrl.analyze_normality(data, col_names))
            if test_type == 2:
                return ("ttest", ctrl.analyze_t_test(data, groups=groups))
            if test_type == 3:
                return ("anova", ctrl.analyze_anova(data, groups=groups))
            if test_type == 4:
                return ("kruskal", ctrl.analyze_kruskal_wallis(data, groups=groups))
            raise ValueError(f"Unknown test_type: {test_type}")

        def _done(payload):
            kind, result = payload
            plot = InteractivePlotCanvas()
            if kind == "summary":
                plot.plot_summary_statistics(data, col_names, result.columns)
                title = _("Summary Statistics")
                info = _("Summary: {0} variables, {1} samples").format(data.shape[1], data.shape[0])
            elif kind == "normality":
                plot.plot_normality_qq(data, col_names, result)
                title = _("Normality Test")
                n_normal = sum(1 for r in result if r.is_normal_shapiro)
                info = _("Normality: {0}/{1} variables pass Shapiro-Wilk (α=0.05)").format(
                    n_normal, len(result)
                )
            elif kind == "ttest":
                p_values = [r.p_value for r in result]
                plot.plot_group_comparison(data, groups, col_names, "t-test", p_values)
                title = _("t-test Results")
                info = _("t-test: {0}/{1} variables significant (α=0.05)").format(
                    sum(1 for p in p_values if p < 0.05), len(p_values)
                )
            elif kind == "anova":
                p_values = [r.p_value for r in result]
                plot.plot_group_comparison(data, groups, col_names, "ANOVA", p_values)
                title = _("ANOVA Results")
                info = _("ANOVA: {0}/{1} variables significant (α=0.05)").format(
                    sum(1 for p in p_values if p < 0.05), len(p_values)
                )
            elif kind == "kruskal":
                p_values = [r.p_value for r in result]
                plot.plot_group_comparison(data, groups, col_names, "Kruskal-Wallis", p_values)
                title = _("Kruskal-Wallis Results")
                info = _("Kruskal-Wallis: {0}/{1} variables significant (α=0.05)").format(
                    sum(1 for p in p_values if p < 0.05), len(p_values)
                )
            else:
                self._logger.error("Unknown univariate test kind: %s", kind)
                return
            self._add_plot_to_workspace(plot, title)
            self._status_bar.setInfo(info)
            self._status_bar.setProgress(100, 100)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"Univariate analysis failed: {exc}")
            QMessageBox.critical(
                self, _("Univariate Error"), format_user_error(exc, _("Univariate"))
            )

        self._run_analysis_async(_work, _done, _fail, _("Univariate"))

    def _on_run_univariate(self) -> None:
        """Run univariate statistics."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        self._run_univariate_analysis(0)

    def _on_run_univariate_by_index(self, index: int) -> None:
        """Run univariate analysis by test index (0=Summary, 1=Normality, 2=t-test, 3=ANOVA, 4=Kruskal-Wallis)."""
        self._run_univariate_analysis(index)

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
                # The dialog exposes ``cross_validate`` (leave-one-out);
                # the controller's analyze_lda does not currently take
                # the flag, so log it and surface the choice in the
                # status bar so the user can see it was not silently
                # dropped.
                cross_validate = bool(params.get("cross_validate", False))
                self._logger.info(
                    f"Running LDA with {len(set(groups))} groups, "
                    f"n_components={params.get('n_components')}, "
                    f"cross_validate={cross_validate} "
                    "(analyze_lda ignores the flag — controller API)."
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
                cv_note = _(" (CV requested)") if cross_validate else ""
                self._status_bar.setInfo(_("LDA analysis completed") + cv_note)
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

                # The dialog exposes the permutation count and seed for the
                # significance test; StatisticsController.run_cca forwards
                # them to the engine, which permutes the RESPONSE (Y) and
                # reports an F statistic, a p-value and Wilks' lambda.
                n_permutations = int(params.get("n_permutations", 999))
                raw_seed = params.get("random_seed", 0)
                # The dialog uses 0 as its "no seed" sentinel; the engine
                # uses None. Keeping the no-seed case honest is deliberate -
                # it makes the p-value irreproducible on purpose, and the
                # engine warns about it.
                random_seed = int(raw_seed) if raw_seed not in (0, None) else None

                result = self._statistics_controller.run_cca(
                    Y=Y,
                    X=X,
                    n_components=params.get("n_components"),
                    method=params.get("method"),
                    n_permutations=n_permutations,
                    random_seed=random_seed,
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
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.get_parameters()
        data = self._state.data_matrix.data
        n_clusters = params.get("n_clusters", 3)
        method = params.get("method", "ward")
        metric = params.get("metric", "euclidean")
        row_labels = list(self._state.data_matrix.row_labels or [])

        def _work():
            return self._statistics_controller.analyze_clustering(
                data=data, n_clusters=n_clusters, method=method, metric=metric
            )

        def _done(result):
            plot = InteractivePlotCanvas()
            plot.plot_dendrogram(result, labels=row_labels)
            self._add_plot_to_workspace(plot, _("Clustering"))
            self._status_bar.setInfo(
                _("Clustering: {0} clusters, cophenetic r={1:.3f}").format(
                    result.n_clusters, result.cophenetic_corr
                )
            )
            self._status_bar.setProgress(100, 100)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"Clustering analysis failed: {exc}")
            QMessageBox.critical(self, _("Clustering Error"), format_user_error(exc, "Op: clustering analysis"))

        self._run_analysis_async(_work, _done, _fail, _("Clustering"))

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
            # The previous implementation built a one-line ``info_lines``
            # list comprehension but never showed it. Surface the
            # model fit summaries on the status bar AND in a small info
            # tab so the user can see which models actually fitted.
            info_lines = [
                f"{fit.model_name}: R²={fit.r_squared:.4f}, AIC={fit.aic:.2f}"
                for fit in results.values()
            ]
            summary = " | ".join(info_lines)
            self._status_bar.setInfo(_("Abundance models fitted: {0}").format(summary))
            self._logger.info("Abundance model fits: %s", summary)
        except Exception as e:
            QMessageBox.critical(self, _("Abundance Models Error"), format_user_error(e, "Op: abundance models"))
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
            QMessageBox.critical(self, _("SHE Error"), format_user_error(e, "Op: SHE analysis"))
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

            data = self._state.data_matrix.data
            n_zones = params.get("n_zones", 4)

            def _work():
                # CONISS 层次聚类是 O(n^3): 后台线程执行, 避免大剖面冻结 GUI
                return self._statistics_controller.analyze_coniss(
                    data=data,
                    n_zones=n_zones,
                )

            def _on_result(result):
                self._status_bar.setProgress(100, 100)
                sample_names = list(self._state.data_matrix.row_labels or [])
                plot = InteractivePlotCanvas()
                plot.plot_coniss_dendrogram(
                    result.linkage_matrix,
                    result.n_zones,
                    sample_names=sample_names if sample_names else None,
                )
                plot_index = self._add_plot_to_workspace(plot, _("CONISS Dendrogram"))
                self._workspace.setCurrentIndex(plot_index)
                self._status_bar.setInfo(_("CONISS: {0} zones").format(result.n_zones))

            def _on_error(e):
                self._status_bar.setProgress(100, 100)
                QMessageBox.critical(self, _("CONISS Error"), format_user_error(e, "CONISS"))

            self._run_analysis_async(_work, _on_result, _on_error, _("CONISS"))

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

                # Plot transition probability heatmap
                plot = InteractivePlotCanvas()
                plot.plot_markov_heatmap(
                    result.transition_matrix,
                    result.facies_names,
                    chi2_stat=result.chi_squared,
                    p_value=result.p_value,
                )
                plot_index = self._add_plot_to_workspace(plot, _("Markov Transition Matrix"))
                self._workspace.setCurrentIndex(plot_index)

                sig = "Markovian" if result.is_markovian else "Random"
                self._status_bar.setInfo(
                    _("Markov: χ²={0:.1f}, p={1:.4f} ({2})").format(result.chi_squared, result.p_value, sig)
                )
            except Exception as e:
                QMessageBox.critical(self, _("Markov Error"), format_user_error(e, "Op: Markov chain"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_directional(self) -> None:
        """Run directional statistics."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        # Pick the column to analyse.  The DirectionalDialog only
        # exposes ``n_bins`` -- it has no column chooser -- so the
        # column picker lives here.  Without it the analyser used to
        # silently default to column 0, ignoring every other column on
        # wide matrices.  We pick 0 by default; the user can choose
        # any numeric column from the spreadsheet.
        col_labels = []
        try:
            col_labels = list(self._state.data_matrix.col_labels or [])
        except AttributeError:
            col_labels = []
        default_col = 0
        chosen_col = default_col
        if len(col_labels) > 1:
            chosen_col = self._prompt_column_index(
                _("Directional: pick the angle column"),
                col_labels,
                default=default_col,
            )
            if chosen_col is None:
                return
        elif not col_labels:
            chosen_col = 0
        else:
            chosen_col = 0

        dialog = DirectionalDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            try:
                self._status_bar.setProgress(0, 0)
                # Forward the user's ``column_index`` pick; before this
                # fix the analyser silently used column 0 on wide
                # matrices.
                result = self._statistics_controller.analyze_directional(
                    column_index=int(chosen_col)
                )
                bin_edges, counts = self._statistics_controller.bin_rose_diagram(
                    n_bins=params.get("n_bins", 12),
                    column_index=int(chosen_col),
                )
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
                QMessageBox.critical(self, _("Directional Error"), format_user_error(e, "Op: directional statistics"))
            finally:
                self._status_bar.setProgress(100, 100)

    def _prompt_column_index(
        self, title: str, col_labels: list[str], default: int = 0
    ) -> int | None:
        """Tiny modal that asks the user which column to analyse.

        Returns ``None`` when the user cancels.  Defaults to ``default``
        (used when the matrix has only one column and a picker is
        unnecessary).
        """
        if len(col_labels) <= 1:
            return default if col_labels else 0
        from PyQt6.QtWidgets import QInputDialog

        label_text = "\n".join(f"{i}: {label}" for i, label in enumerate(col_labels))
        choice, ok = QInputDialog.getItem(
            self,
            _("Select Column"),
            f"{title}\n\n{label_text}",
            [f"{i}: {label}" for i, label in enumerate(col_labels)],
            max(0, min(default, len(col_labels) - 1)),
            False,
        )
        if not ok:
            return None
        try:
            return int(choice.split(":", 1)[0])
        except (ValueError, AttributeError):
            return default

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
                data = self._state.data_matrix.data
                # EFA expects one contour per row with at least a few
                # (x, y) coordinates. The previous implementation
                # silently used ``data[:, :2]`` which discards every
                # extra column on multi-feature matrices and feeds
                # unrelated columns to EFA on wide matrices.
                if data.ndim != 2 or data.shape[1] < 2:
                    QMessageBox.warning(
                        self,
                        _("Insufficient Data"),
                        _("EFA needs at least 2 columns to form a contour."),
                    )
                    return
                contour = data[:, :2]
                if data.shape[1] != 2:
                    reply = QMessageBox.question(
                        self,
                        _("Use First Two Columns"),
                        _(
                            "EFA treats each row as (x, y) coordinates of one contour.\n"
                            "The loaded matrix has {0} columns. Use only the first two for EFA?"
                        ).format(data.shape[1]),
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                result = self._statistics_controller.analyze_efa(
                    contour=contour,
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

    def _on_run_eigenshape(self) -> None:
        """Run Eigenshape Analysis on EFA coefficients.

        Expects data where each row is one specimen and columns are
        [x1, x2, ..., xN, y1, y2, ..., yN] (even number of columns,
        at least 6 so that N ≥ 3 for a meaningful contour).
        """
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        import numpy as np

        data = self._state.data_matrix.data
        if data.ndim != 2 or data.shape[1] < 6:
            QMessageBox.warning(
                self,
                _("Insufficient Data"),
                _("Need at least 6 columns (x1..xN, y1..yN with N≥3). Got {0} columns.").format(data.shape[1]),
            )
            return
        if data.shape[1] % 2 != 0:
            QMessageBox.warning(
                self,
                _("Invalid Data"),
                _("Column count must be even (equal x and y coordinates). Got {0}.").format(data.shape[1]),
            )
            return
        if data.shape[0] < 2:
            QMessageBox.warning(self, _("Insufficient Data"), _("Need at least 2 specimens (rows)."))
            return

        # Previously the harmonic count was hard-coded to
        # ``min(10, n_cols // 4)``; let the user pick instead, while
        # still clamping to a sensible range.
        suggested = max(2, min(10, data.shape[1] // 4))
        from PyQt6.QtWidgets import QInputDialog

        choice, ok = QInputDialog.getInt(
            self,
            _("Eigenshape — Number of Harmonics"),
            _("Harmonics (2-{0}):").format(max(2, data.shape[1] // 2)),
            suggested,
            2,
            max(2, data.shape[1] // 2),
            1,
        )
        if not ok:
            return
        n_harmonics = int(choice)

        from morphometrics.efa import EFAAnalyzer, EigenshapeAnalyzer

        def _work():
            efa = EFAAnalyzer()
            coefficients_list = []
            for i in range(data.shape[0]):
                row = data[i]
                n_pts = len(row) // 2
                contour = np.column_stack([row[:n_pts], row[n_pts : 2 * n_pts]])
                result_i = efa.analyze(contour, n_harmonics=n_harmonics)
                coefficients_list.append(result_i.coefficients)
            es_analyzer = EigenshapeAnalyzer()
            return es_analyzer.analyze(
                coefficients_list, n_components=min(5, data.shape[0] - 1)
            )

        def _done(es_result):
            plot = InteractivePlotCanvas()
            labels = list(self._state.data_matrix.row_labels or [])
            plot.plot_eigenshape_scores(
                es_result.scores,
                es_result.explained_variance,
                specimen_labels=labels if labels else None,
            )
            self._add_plot_to_workspace(plot, _("Eigenshape Scores"))
            self._status_bar.setInfo(
                _("Eigenshape: {0} harmonics, {1} specimens, {2} components").format(
                    n_harmonics, es_result.n_specimens, es_result.n_components
                )
            )
            self._status_bar.setProgress(100, 100)

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"Eigenshape analysis failed: {exc}")
            QMessageBox.critical(
                self, _("Eigenshape Error"), format_user_error(exc, "Eigenshape")
            )

        self._run_analysis_async(_work, _done, _fail, _("Eigenshape"))

    def _on_run_isotope(self) -> None:
        """Run Isotope Time Series Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = IsotopeAnalysisDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.get_parameters()

        # Build the IsotopeData payload synchronously so we can fail
        # fast on obviously wrong shapes; the actual analyser call
        # runs on the worker thread.
        import numpy as np

        from stratigraphy.isotope_analysis import IsotopeAnalyzer, IsotopeData

        data = self._state.data_matrix.data
        if data.shape[1] < 3:
            QMessageBox.warning(
                self,
                _("Insufficient Data"),
                _("Need at least 3 columns: depth, age, and isotope values"),
            )
            return

        iso_kwargs: dict = {
            "depth": data[:, 0],
            "age": data[:, 1],
        }
        isotope_names = ["d13C", "d18O", "sr", "nd"]
        for i, name in enumerate(isotope_names):
            col_idx = i + 2
            if col_idx < data.shape[1]:
                col_data = data[:, col_idx]
                if not np.all(np.isnan(col_data)):
                    iso_kwargs[name] = col_data

        if len(iso_kwargs) <= 2:
            QMessageBox.warning(
                self,
                _("Insufficient Data"),
                _("Need at least one isotope column with valid data"),
            )
            return

        iso_data = IsotopeData(**iso_kwargs)

        def _work():
            analyzer = IsotopeAnalyzer()
            return analyzer.analyze(
                iso_data,
                detect_excursions=params.get("detect_excursions", True),
                excursion_threshold=params.get("excursion_threshold", 2.0),
                excursion_min_duration=params.get("excursion_min_duration", 2),
                compute_correlations=params.get("compute_correlations", True),
            )

        def _done(result):
            self._status_bar.setInfo(
                _("Isotope: {0} excursions detected").format(len(result.excursions))
            )
            self._status_bar.setProgress(100, 100)
            QMessageBox.information(self, _("Analysis Complete"), result.summary())

        def _fail(exc):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"Isotope analysis failed: {exc}")
            QMessageBox.critical(self, _("Isotope Error"), format_user_error(exc, "Op: isotope analysis"))

        self._run_analysis_async(_work, _done, _fail, _("Isotope"))

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
                        t_diff = np.append(t_diff, t_diff[-1] if t_diff.size > 0 else 1.0)
                        sec_name = col_labels[h_col] if 0 <= h_col < len(col_labels) else _("Section {0}").format(i + 1)
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
                        sec_name = col_labels[pc] if 0 <= pc < len(col_labels) else _("Section {0}").format(pc + 1)
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
                    only_name = col_labels[0] if 0 < len(col_labels) else _("Section 1")
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
                    QMessageBox.information(self, _("Analysis Complete"), result.summary())

            except Exception as e:
                self._logger.critical(f"Stratigraphic correlation failed: {e}")
                QMessageBox.critical(
                    self,
                    _("Correlation Error"),
                    format_user_error(e, "Op: stratigraphic correlation"),
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
                        _("Need at least 2 stratigraphic samples to perform correspondence analysis."),
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
                        _("Please select at least one taxon column in the dialog."),
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
                        taxon_labels.append("{0}: {1}".format(idx, col_labels[idx]))
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
                        _("Height column contains non-finite values at rows: {0}").format(bad_rows[:10])
                    )
                else:
                    h_diff = np.diff(heights)
                    if not (np.all(h_diff > 0) or np.all(h_diff < 0)):
                        validation_issues.append(
                            _("Heights must be strictly monotonic. Please sort the rows by stratigraphic height first.")
                        )
                if not np.all(np.isfinite(abundance)):
                    validation_issues.append(
                        _("Abundance matrix contains NaN/Inf. Please clean or impute the data before running CA.")
                    )
                if np.any(abundance < 0):
                    neg_cnt = int(np.sum(abundance < 0))
                    validation_issues.append(
                        _("Abundance matrix contains {0} negative cell(s); CA requires non-negative input.").format(
                            neg_cnt
                        )
                    )
                if abundance.size > 0:
                    row_sums = abundance.sum(axis=1)
                    zero_rows = np.where(row_sums <= 0.0)[0]
                    if zero_rows.size > 0:
                        validation_issues.append(
                            _(
                                "Rows with zero total abundance at indices {0}; drop these rows or pick more taxa."
                            ).format(zero_rows[:10].tolist())
                        )
                    col_sums = abundance.sum(axis=0)
                    zero_cols = np.where(col_sums <= 0.0)[0]
                    if zero_cols.size > 0:
                        bad_taxa = [taxon_labels[int(c)] for c in zero_cols.tolist()]
                        validation_issues.append(
                            _("Taxa with zero total abundance: {0}; remove them from the selection.").format(
                                bad_taxa[:10]
                            )
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
                        self._logger.warning("Failed to render paleo-env plot: %s", plot_exc)

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
                    format_user_error(e, "Op: paleo-environmental reconstruction"),
                )
            finally:
                self._status_bar.setProgress(100, 100)

    def _on_run_gpa(self) -> None:
        """Run Generalized Procrustes Analysis."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        data = self._state.data_matrix.data

        def _gpa_work():
            # GPA 迭代对齐在标本/标志点多时是长计算, 后台线程执行
            return self._statistics_controller.analyze_gpa(data=data)

        def _gpa_result(result):
            self._status_bar.setProgress(100, 100)

            import numpy as np  # local import keeps the module-level namespace tidy

            plot = InteractivePlotCanvas()
            plot.setDarkTheme(self._is_dark_theme)
            # Plot GPA-aligned landmarks.  Both branches go through
            # ``plot_gpa_aligned``: the 2-D case used to call
            # ``plot_efa_contours(coords, title=...)`` with a single
            # positional argument and raised TypeError (plot_efa_contours
            # needs an original *and* a reconstructed contour).
            if hasattr(result, "aligned_configurations"):
                coords = np.asarray(result.aligned_configurations)
                plot.plot_gpa_aligned(coords, title=_("GPA Aligned Landmarks"))
            else:
                # No aligned configurations on the result (e.g. the 1-D
                # summary path): show an empty canvas instead of a blank tab.
                plot.get_figure().clear()
                plot.get_figure().text(0.5, 0.5, _("No aligned configurations"), ha="center")
                plot.get_figure().canvas.draw_idle()

            plot_index = self._add_plot_to_workspace(plot, _("GPA Alignment"))
            self._workspace.setCurrentIndex(plot_index)

            self._status_bar.setInfo(_("GPA analysis completed"))

        def _gpa_error(e):
            self._status_bar.setProgress(100, 100)
            self._logger.error(f"GPA analysis failed: {e}")
            QMessageBox.critical(self, _("GPA Error"), format_user_error(e, "GPA"))

        self._run_analysis_async(_gpa_work, _gpa_result, _gpa_error, _("GPA"))

    def _on_run_tps_grid(self) -> None:
        """Run TPS Deformation Grid visualization."""
        if not self._state.has_data:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return

        dialog = TPSGridDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._execute_tps_grid(dialog.get_parameters())

    def _execute_tps_grid(self, params: dict, on_done=None, on_fail=None) -> None:
        """Plot the TPS deformation grid from the cached GPA result.

        The runlist guard (``cache:gpa_result``) normally ensures the cache
        exists; a missing cache here is a race and reported via on_fail.
        """
        try:
            self._status_bar.setProgress(0, 0)

            # Get GPA result from cache for TPS visualization
            tps_result = self._state.get_cached_result("gpa_result")

            if tps_result is None:
                message = _("Please run GPA (Generalized Procrustes Analysis) first to compute TPS deformation.")
                if on_fail is not None:
                    raise RuntimeError(message)
                QMessageBox.information(self, _("No TPS Result"), message)
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
            if on_fail is not None:
                on_fail(e)
            else:
                QMessageBox.critical(self, _("TPS Grid Error"), format_user_error(e, "Op: TPS grid"))
        else:
            if on_done is not None:
                on_done(tps_result)
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
                # ``arange`` excludes its stop value; the dialog's "max scale"
                # is inclusive, so +1 keeps the requested top scale (and keeps
                # the vector non-empty now that min < max is enforced).
                scales = np.arange(params.get("min_scale", 2), params.get("max_scale", 50) + 1)
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
                QMessageBox.critical(self, _("Wavelet Error"), format_user_error(e, "Op: wavelet analysis"))
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
                    event_names = col_labels_list + [f"Event_{i + 1}" for i in range(len(col_labels_list), mid)]

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
                        enable_cyclic_check=bool(params.get("enable_cyclic_check", True)),
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
                QMessageBox.critical(self, _("Biostratigraphy Error"), format_user_error(e, "Op: biostratigraphy"))
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
        dialog = PICDialog(self, controller=self._statistics_controller)
        dialog.setDarkTheme(self._is_dark_theme)
        # Connect ``resultsReady`` so a future plot tab (or run-list
        # hook) sees the payload. Previously the four PCM handlers
        # never connected this signal, so the analysis ran but no
        # visual artefact was produced.
        dialog.resultsReady.connect(self._on_pcm_payload)
        dialog.exec()

    def _on_run_ancestral_states(self) -> None:
        """Run Ancestral State Reconstruction (ASR) analysis."""
        dialog = AncestralStateDialog(self, controller=self._statistics_controller)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.resultsReady.connect(self._on_pcm_payload)
        dialog.exec()

    def _on_run_phylogenetic_signal(self) -> None:
        """Run Blomberg's K phylogenetic signal analysis."""
        dialog = PhyloSignalDialog(self, controller=self._statistics_controller)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.resultsReady.connect(self._on_pcm_payload)
        dialog.exec()

    def _on_run_phylo_anova(self) -> None:
        """Run Phylogenetic ANOVA analysis."""
        dialog = PhyloANOVADialog(self, controller=self._statistics_controller)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.resultsReady.connect(self._on_pcm_payload)
        dialog.exec()

    def _on_pcm_payload(self, payload: dict) -> None:
        """Display a summary tab for any PCM analysis."""
        try:
            from PyQt6.QtWidgets import QTextEdit

            editor = QTextEdit()
            editor.setReadOnly(True)
            summary = payload.get("summary", "") if isinstance(payload, dict) else str(payload)
            editor.setPlainText(summary)
            # We DO add this as a workspace tab so the user actually
            # sees the analysis output (the dialog's own results panel
            # disappears when the dialog closes).
            from views.ui_plot_canvas import InteractivePlotCanvas as _Canvas  # noqa: F401

            self._add_tab_to_workspace(editor, _("PCM Result"))
            self._status_bar.setInfo(_("PCM analysis completed"))
        except Exception as exc:
            self._logger.debug("Failed to surface PCM payload: %s", exc)

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
        except ImportError:
            # psutil ships in the optional [full] extra. Say the measurement is
            # unavailable rather than leaving a fabricated "0 MB" on screen.
            self._status_bar._memory_label.setText(_("Memory: N/A"))
        except Exception:
            # Runtime failure (permissions, /proc unavailable, ...). Stay quiet
            # about the cause, but do not pretend the reading was zero.
            self._status_bar._memory_label.setText(_("Memory: --"))

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

        # Drain the analysis thread pool BEFORE the window is destroyed.
        # self._thread_pool is QThreadPool.globalInstance(), which outlives
        # this window. Without an explicit drain, an in-flight run (a long
        # NMDS, or the now-async ANOSIM) later emits result_ready on an
        # _AnalysisSignals QObject whose parent window has already been
        # destroyed, raising
        # "RuntimeError: wrapped C/C++ object of type _AnalysisSignals has
        # been deleted" and typically surfacing as
        # "QThread: Destroyed while thread is still running" -> process abort.
        clean = self._drain_thread_pool()
        if not clean:
            # Previously the drain just logged a warning and silently
            # dropped the late result. The user should know their work
            # did not finish -- a non-blocking notification is enough
            # since the window is about to close.
            self._logger.warning(
                "Closing with in-flight analysis tasks: their results were "
                "abandoned to avoid hanging the close."
            )

        event.accept()

    def _drain_thread_pool(self, timeout_ms: int = 5000) -> None:
        """Stop accepting new analysis tasks and wait for running ones.

        Tasks that do not finish within ``timeout_ms`` are abandoned: at that
        point the alternative is hanging the close on a multi-minute
        computation, and the ``_closing`` flag keeps their late callbacks
        from touching destroyed widgets.

        Returns ``True`` when the pool drained cleanly, ``False`` when at
        least one task was still running at the timeout. The caller is
        expected to surface that distinction to the user instead of
        silently dropping their results.
        """
        pool = getattr(self, "_thread_pool", None)
        if pool is None:
            return True
        self._closing = True
        try:
            pool.clear()  # drop queued-but-not-started tasks
            done = pool.waitForDone(timeout_ms)
            if not done:
                self._logger.warning(
                    "Analysis thread pool still busy after %d ms; abandoning in-flight results on close.", timeout_ms
                )
            return bool(done)
        except Exception as exc:  # never let teardown raise
            self._logger.warning("Thread pool drain failed: %s", exc)
            return False

    # =========================================================================
    # New Analysis Handlers
    # =========================================================================

    def _on_run_allometry(self) -> None:
        """Run Allometry analysis."""
        dialog = AllometryDialog(self, controller=self._statistics_controller)
        dialog.setDarkTheme(self._is_dark_theme)

        def _show(payload: dict) -> None:
            plot = InteractivePlotCanvas()
            if hasattr(plot, "plot_allometry"):
                plot.plot_allometry(payload)
                title = _("Allometry Regression")
            elif hasattr(plot, "plot_allometry_results"):
                plot.plot_allometry_results(payload)
                title = _("Allometry Regression")
            else:
                return
            idx = self._add_plot_to_workspace(plot, title)
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(title)

        dialog.resultsReady.connect(_show)
        dialog.exec()

    def _on_run_pls(self) -> None:
        """Run Two-Block PLS (morphological integration) analysis."""
        dialog = PLSDialog(self, controller=self._statistics_controller)
        dialog.setDarkTheme(self._is_dark_theme)

        def _show(payload: dict) -> None:
            plot = InteractivePlotCanvas()
            # Try the new signature first, then the legacy one; if
            # neither exists the canvas agent has not yet wired the
            # method, so raise so the failure surfaces instead of being
            # silently swallowed by the previous ``else: return``.
            if hasattr(plot, "plot_pls"):
                plot.plot_pls(payload)
                title = _("Two-Block PLS Scores")
            elif hasattr(plot, "plot_pls_results"):
                plot.plot_pls_results(payload)
                title = _("Two-Block PLS Scores")
            else:
                raise AttributeError(
                    "InteractivePlotCanvas has neither plot_pls nor "
                    "plot_pls_results — the canvas layer agent must add "
                    "one of these for the PLS results to be visualised."
                )
            idx = self._add_plot_to_workspace(plot, title)
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(title)

        dialog.resultsReady.connect(_show)
        dialog.exec()

    def _on_run_evolution_rate(self) -> None:
        """Run Evolution Rate analysis."""
        dialog = EvolutionRateDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        # The dialog emits ``resultsReady`` when it finishes running.
        # Previously the slot was never connected so the analysis ran
        # but no plot was produced; this slot renders the result and
        # surfaces a clear notice when the canvas layer has no
        # ``plot_evolution_rate`` method (instead of dropping the payload).
        dialog.resultsReady.connect(self._on_evolution_rate_result)
        dialog.exec()

    def _on_evolution_rate_result(self, payload: dict) -> None:
        try:
            plot = InteractivePlotCanvas()
            plot.plot_evolution_rate(payload)
            idx = self._add_plot_to_workspace(plot, _("Evolution Rate"))
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(_("Evolution Rate: plotted"))
        except Exception as exc:
            # ``plot_evolution_rate`` may not be wired up to the
            # canvas; the canvas is owned by another agent. Fall back
            # to a results tab so the user can still see the numbers.
            self._logger.warning(
                "Falling back to text tab for evolution-rate result: %s", exc
            )
            from PyQt6.QtWidgets import QTextEdit

            editor = QTextEdit()
            editor.setReadOnly(True)
            editor.setPlainText(str(payload.get("summary", payload)))
            self._add_tab_to_workspace(editor, _("Evolution Rate Result"))

    def _macroevolution_dialog(self) -> MacroevolutionDialog:
        dialog = MacroevolutionDialog(self._statistics_controller, self)
        dialog.setDarkTheme(self._is_dark_theme)
        return dialog

    def _plot_macroevolution_result(self, kind: str, payload: object) -> None:
        """Render a macroevolution result and add it to the workspace."""
        plot = InteractivePlotCanvas()
        if kind == "cohort_survivorship":
            plot.plot_cohort_survivorship(payload)
            name = _("Cohort Survivorship")
        elif kind == "diversity_dynamics":
            plot.plot_diversity_dynamics(payload)
            name = _("Diversity Dynamics")
        elif kind == "survival":
            plot.plot_survival_curve(payload)
            name = _("Survival Analysis")
        elif kind == "survival_logrank":
            # A log-rank result has no curve; report it in a results tab.
            plot = InteractivePlotCanvas()
            plot.plot_survival_curve(self._statistics_controller.get_cached_result("survival_result"))
            name = _("Survival Analysis (log-rank p={0:.4f})").format(payload.p_value)
        elif kind == "fbd":
            plot.plot_fbd_diversity(payload)
            name = _("FBD Simulation")
        else:
            self._logger.warning("Unknown macroevolution result kind: %s", kind)
            return
        idx = self._add_plot_to_workspace(plot, name)
        self._workspace.setCurrentIndex(idx)
        self._status_bar.setInfo(f"{name}: {plot._current_plot_type}")

    def _on_run_cohort_survivorship(self) -> None:
        """Foote cohort survivorship from the loaded matrix."""
        if self._state.data_matrix is None:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        dialog = self._macroevolution_dialog()
        dialog.resultsReady.connect(lambda kind, payload: self._plot_macroevolution_result(kind, payload))
        # ``dialog.exec()`` is a blocking call that returns Accepted
        # once the user finishes; calling ``dialog.accept()`` after
        # that is dead code (the dialog is already gone).  Just exec.
        dialog.exec()

    def _on_run_diversity_dynamics(self) -> None:
        """Diversity dynamics from the loaded matrix."""
        if self._state.data_matrix is None:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        dialog = self._macroevolution_dialog()
        dialog.setProperty("tab", 1)
        dialog.resultsReady.connect(lambda kind, payload: self._plot_macroevolution_result(kind, payload))
        dialog.exec()

    def _on_run_survival_analysis(self) -> None:
        """Kaplan-Meier / log-rank from the loaded matrix."""
        if self._state.data_matrix is None:
            QMessageBox.warning(self, _("No Data"), _("Please load data first."))
            return
        dialog = self._macroevolution_dialog()
        dialog.setProperty("tab", 2)
        dialog.resultsReady.connect(lambda kind, payload: self._plot_macroevolution_result(kind, payload))
        dialog.exec()

    def _on_run_fbd_simulation(self) -> None:
        """Fossilised birth-death simulation (parameters only, no data needed)."""
        dialog = self._macroevolution_dialog()
        dialog.setProperty("tab", 3)
        dialog.resultsReady.connect(lambda kind, payload: self._plot_macroevolution_result(kind, payload))
        dialog.exec()

    def _on_run_gpa3d(self) -> None:
        """Generalized Procrustes analysis of 3-D landmark configurations."""
        if self._state.data_matrix is None:
            QMessageBox.warning(self, _("No Data"), _("Please load a 3-D landmark matrix first."))
            return
        dialog = Morpho3DDialog(self._statistics_controller, self)
        dialog.setDarkTheme(self._is_dark_theme)

        def _show(result: object) -> None:
            plot = InteractivePlotCanvas()
            plot.plot_gpa3d_aligned(
                result.aligned_configurations,
                result.mean_config,
                specimen_labels=list(self._state.data_matrix.row_labels or []),
                title=_("3-D GPA Alignment"),
            )
            idx = self._add_plot_to_workspace(plot, _("3-D GPA Aligned Landmarks"))
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(_("3-D GPA completed"))

        dialog.resultsReady.connect(_show)
        dialog.exec()

    def _on_run_extinction_intervals(self) -> None:
        """Run Extinction Confidence Intervals analysis."""
        dialog = ExtinctionIntervalDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        # Forward the dict the dialog emits as ``resultsReady`` to a
        # canvas plot; falls back to a text tab if the canvas has no
        # ``plot_extinction_ranges`` method (graceful degradation).
        dialog.resultsReady.connect(self._on_extinction_intervals_result)
        dialog.exec()

    def _on_extinction_intervals_result(self, payload: dict) -> None:
        plot = InteractivePlotCanvas()
        if hasattr(plot, "plot_extinction_ranges"):
            plot.plot_extinction_ranges(payload)
            idx = self._add_plot_to_workspace(plot, _("Extinction Intervals"))
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(_("Extinction intervals: plotted"))
            return
        # Canvas has no specialised plotter; surface the result as text.
        from PyQt6.QtWidgets import QTextEdit

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(str(payload.get("summary", payload)))
        self._add_tab_to_workspace(editor, _("Extinction Intervals Result"))
        self._status_bar.setInfo(_("Extinction intervals: text fallback"))

    def _on_run_beta_diversity(self) -> None:
        """Run Beta Diversity analysis."""
        dialog = BetaDiversityDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.resultsReady.connect(self._on_beta_diversity_result)
        dialog.exec()

    def _on_beta_diversity_result(self, payload: dict) -> None:
        plot = InteractivePlotCanvas()
        if hasattr(plot, "plot_beta_diversity"):
            plot.plot_beta_diversity(payload)
            idx = self._add_plot_to_workspace(plot, _("Beta Diversity"))
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(_("Beta diversity: plotted"))
            return
        from PyQt6.QtWidgets import QTextEdit

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(str(payload.get("summary", payload)))
        self._add_tab_to_workspace(editor, _("Beta Diversity Result"))
        self._status_bar.setInfo(_("Beta diversity: text fallback"))

    def _on_run_null_models(self) -> None:
        """Run Null Model analysis."""
        dialog = NullModelDialog(self)
        dialog.setDarkTheme(self._is_dark_theme)
        dialog.resultsReady.connect(self._on_null_model_result)
        dialog.exec()

    def _on_null_model_result(self, payload: dict) -> None:
        plot = InteractivePlotCanvas()
        if hasattr(plot, "plot_null_model"):
            plot.plot_null_model(payload)
            idx = self._add_plot_to_workspace(plot, _("Null Model"))
            self._workspace.setCurrentIndex(idx)
            self._status_bar.setInfo(_("Null model: plotted"))
            return
        from PyQt6.QtWidgets import QTextEdit

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(str(payload.get("summary", payload)))
        self._add_tab_to_workspace(editor, _("Null Model Result"))
        self._status_bar.setInfo(_("Null model: text fallback"))


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

