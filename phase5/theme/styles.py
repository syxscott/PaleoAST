"""
================================================================================
PaleoAST Phase 5 - Ultimate QSS Style Engine
================================================================================

本模块包含极其庞大的现代化QSS样式字符串池，
覆盖PyQt6/PySide6所有组件的深度美化。

作者: PaleoAST Development Team
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class StyleMode(Enum):
    """样式模式"""
    DARK = "dark"
    LIGHT = "light"


@dataclass
class ColorPalette:
    """调色板"""
    # 主色
    primary: str = "#3498DB"
    primary_hover: str = "#5DADE2"
    primary_pressed: str = "#2980B9"
    
    # 强调色
    accent: str = "#2ECC71"
    accent_hover: str = "#58D68D"
    
    # 警告/错误
    warning: str = "#F39C12"
    error: str = "#E74C3C"
    success: str = "#27AE60"
    info: str = "#3498DB"
    
    # 背景色 (深色模式)
    bg_dark_1: str = "#1A1A2E"
    bg_dark_2: str = "#16213E"
    bg_dark_3: str = "#0F3460"
    bg_dark_4: str = "#1F4068"
    
    # 背景色 (浅色模式)
    bg_light_1: str = "#FFFFFF"
    bg_light_2: str = "#F8F9FA"
    bg_light_3: str = "#E9ECEF"
    
    # 文字色
    text_dark: str = "#ECF0F1"
    text_dark_secondary: str = "#95A5A6"
    text_light: str = "#2C3E50"
    text_light_secondary: str = "#7F8C8D"
    
    # 边框色
    border_dark: str = "#34495E"
    border_light: str = "#BDC3C7"
    
    # 表格色
    table_header: str = "#34495E"
    table_row_alt: str = "#2C3E50"
    table_grid: str = "#3D566E"


class PaleoASTStyles:
    """
    巨型QSS样式字符串池
    
    包含针对所有PyQt6组件的深度美化样式。
    
    使用示例:
        >>> styles = PaleoASTStyles()
        >>> qss = styles.get_complete_stylesheet()
        >>> app.setStyleSheet(qss)
    """
    
    def __init__(self, mode: StyleMode = StyleMode.DARK):
        """
        初始化样式引擎
        
        参数:
            mode: 样式模式
        """
        self._mode = mode
        self._colors = ColorPalette()
    
    @property
    def mode(self) -> StyleMode:
        """获取当前模式"""
        return self._mode
    
    def set_mode(self, mode: StyleMode) -> 'PaleoASTStyles':
        """
        设置模式
        
        参数:
            mode: 新模式
        
        返回:
            self
        """
        self._mode = mode
        return self
    
    def get_qmainwindow_styles(self) -> str:
        """
        获取QMainWindow样式
        
        返回:
            QSS字符串
        """
        if self._mode == StyleMode.DARK:
            return """
            QMainWindow {
                background-color: #1A1A2E;
                color: #ECF0F1;
            }
            
            QMainWindow::separator {
                background-color: #34495E;
                width: 1px;
                height: 1px;
            }
            
            QMainWindow::separator:hover {
                background-color: #3498DB;
            }
            """
        else:
            return """
            QMainWindow {
                background-color: #F8F9FA;
                color: #2C3E50;
            }
            
            QMainWindow::separator {
                background-color: #BDC3C7;
                width: 1px;
                height: 1px;
            }
            
            QMainWindow::separator:hover {
                background-color: #3498DB;
            }
            """
    
    def get_qwidget_styles(self) -> str:
        """获取QWidget样式"""
        if self._mode == StyleMode.DARK:
            return """
            QWidget {
                background-color: #1A1A2E;
                color: #ECF0F1;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 10pt;
            }
            
            QWidget:disabled {
                color: #7F8C8D;
            }
            """
        else:
            return """
            QWidget {
                background-color: #FFFFFF;
                color: #2C3E50;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 10pt;
            }
            
            QWidget:disabled {
                color: #BDC3C7;
            }
            """
    
    def get_qpushbutton_styles(self) -> str:
        """获取QPushButton样式"""
        if self._mode == StyleMode.DARK:
            return """
            QPushButton {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #34495E,
                    stop: 1 #2C3E50
                );
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 6px;
                padding: 8px 20px;
                min-height: 28px;
                font-size: 10pt;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3D566E,
                    stop: 1 #34495E
                );
                border: 1px solid #3498DB;
            }
            
            QPushButton:pressed {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2C3E50,
                    stop: 1 #1A1A2E
                );
                border: 1px solid #2980B9;
                padding-top: 9px;
                padding-bottom: 7px;
            }
            
            QPushButton:disabled {
                background-color: #2C3E50;
                color: #7F8C8D;
                border: 1px solid #34495E;
            }
            
            QPushButton:focus {
                outline: none;
                border: 1px solid #3498DB;
            }
            
            /* Primary Button */
            QPushButton[class="primary"] {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3498DB,
                    stop: 1 #2980B9
                );
                border: 1px solid #3498DB;
            }
            
            QPushButton[class="primary"]:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5DADE2,
                    stop: 1 #3498DB
                );
            }
            
            QPushButton[class="primary"]:pressed {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2980B9,
                    stop: 1 #1A5276
                );
            }
            
            /* Success Button */
            QPushButton[class="success"] {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2ECC71,
                    stop: 1 #27AE60
                );
                border: 1px solid #2ECC71;
                color: #FFFFFF;
            }
            
            QPushButton[class="success"]:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #58D68D,
                    stop: 1 #2ECC71
                );
            }
            
            /* Danger Button */
            QPushButton[class="danger"] {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #E74C3C,
                    stop: 1 #C0392B
                );
                border: 1px solid #E74C3C;
                color: #FFFFFF;
            }
            
            QPushButton[class="danger"]:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #EC7063,
                    stop: 1 #E74C3C
                );
            }
            
            /* Icon Button */
            QPushButton[class="icon"] {
                background-color: transparent;
                border: none;
                padding: 4px;
                min-width: 28px;
                min-height: 28px;
            }
            
            QPushButton[class="icon"]:hover {
                background-color: rgba(52, 152, 219, 0.2);
                border-radius: 4px;
            }
            
            QPushButton[class="icon"]:pressed {
                background-color: rgba(52, 152, 219, 0.3);
            }
            """
        else:
            return """
            QPushButton {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FFFFFF,
                    stop: 1 #ECF0F1
                );
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                padding: 8px 20px;
                min-height: 28px;
                font-size: 10pt;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #F8F9FA,
                    stop: 1 #E9ECEF
                );
                border: 1px solid #3498DB;
            }
            
            QPushButton:pressed {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #E9ECEF,
                    stop: 1 #DEE2E6
                );
            }
            
            QPushButton:disabled {
                background-color: #F8F9FA;
                color: #ADB5BD;
                border: 1px solid #DEE2E6;
            }
            """
    
    def get_qlineedit_styles(self) -> str:
        """获取QLineEdit样式"""
        if self._mode == StyleMode.DARK:
            return """
            QLineEdit {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 8px 12px;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
            }
            
            QLineEdit:hover {
                border: 1px solid #3498DB;
            }
            
            QLineEdit:focus {
                border: 2px solid #3498DB;
                padding: 7px 11px;
            }
            
            QLineEdit:disabled {
                background-color: #34495E;
                color: #7F8C8D;
                border: 1px solid #2C3E50;
            }
            
            QLineEdit:read-only {
                background-color: #34495E;
                color: #95A5A6;
            }
            
            /* Placeholder */
            QLineEdit[text=""] {
                color: #7F8C8D;
            }
            """
        else:
            return """
            QLineEdit {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                padding: 8px 12px;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
            }
            
            QLineEdit:hover {
                border: 1px solid #3498DB;
            }
            
            QLineEdit:focus {
                border: 2px solid #3498DB;
                padding: 7px 11px;
            }
            
            QLineEdit:disabled {
                background-color: #F8F9FA;
                color: #ADB5BD;
                border: 1px solid #DEE2E6;
            }
            """
    
    def get_qtextedit_styles(self) -> str:
        """获取QTextEdit样式"""
        if self._mode == StyleMode.DARK:
            return """
            QTextEdit, QPlainTextEdit {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 8px;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
            }
            
            QTextEdit:hover, QPlainTextEdit:hover {
                border: 1px solid #3498DB;
            }
            
            QTextEdit:focus, QPlainTextEdit:focus {
                border: 2px solid #3498DB;
            }
            
            QTextEdit:disabled, QPlainTextEdit:disabled {
                background-color: #34495E;
                color: #7F8C8D;
            }
            
            /* Scrollbar integration */
            QTextEdit QScrollBar:vertical, QPlainTextEdit QScrollBar:vertical {
                background-color: #1A1A2E;
            }
            """
        else:
            return """
            QTextEdit, QPlainTextEdit {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                padding: 8px;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
            }
            
            QTextEdit:hover, QPlainTextEdit:hover {
                border: 1px solid #3498DB;
            }
            
            QTextEdit:focus, QPlainTextEdit:focus {
                border: 2px solid #3498DB;
            }
            """
    
    def get_qcombobox_styles(self) -> str:
        """获取QComboBox样式"""
        if self._mode == StyleMode.DARK:
            return """
            QComboBox {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 8px 12px;
                min-height: 28px;
            }
            
            QComboBox:hover {
                border: 1px solid #3498DB;
            }
            
            QComboBox:focus {
                border: 2px solid #3498DB;
            }
            
            QComboBox:disabled {
                background-color: #34495E;
                color: #7F8C8D;
            }
            
            /* Dropdown arrow */
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #95A5A6;
                margin-right: 8px;
            }
            
            QComboBox::down-arrow:hover {
                border-top-color: #3498DB;
            }
            
            /* Dropdown menu */
            QComboBox QAbstractItemView {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
                padding: 4px;
                outline: 0;
            }
            
            QComboBox QAbstractItemView::item {
                min-height: 32px;
                padding: 6px 12px;
                border-radius: 2px;
            }
            
            QComboBox QAbstractItemView::item:hover {
                background-color: #34495E;
            }
            
            QComboBox QAbstractItemView::item:selected {
                background-color: #3498DB;
            }
            """
        else:
            return """
            QComboBox {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                padding: 8px 12px;
                min-height: 28px;
            }
            
            QComboBox:hover {
                border: 1px solid #3498DB;
            }
            
            QComboBox:focus {
                border: 2px solid #3498DB;
            }
            
            QComboBox:disabled {
                background-color: #F8F9FA;
                color: #ADB5BD;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #7F8C8D;
                margin-right: 8px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
                padding: 4px;
            }
            
            QComboBox QAbstractItemView::item:hover {
                background-color: #E9ECEF;
            }
            
            QComboBox QAbstractItemView::item:selected {
                background-color: #3498DB;
            }
            """
    
    def get_qspinbox_styles(self) -> str:
        """获取QSpinBox样式"""
        if self._mode == StyleMode.DARK:
            return """
            QSpinBox, QDoubleSpinBox {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                padding: 8px 12px;
                min-height: 28px;
            }
            
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #3498DB;
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3498DB;
                padding: 7px 11px;
            }
            
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #34495E;
                color: #7F8C8D;
            }
            
            /* Up/Down buttons */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                border: none;
                background-color: transparent;
                width: 16px;
                subcontrol-position: right;
                subcontrol-origin: padding;
            }
            
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: rgba(52, 152, 219, 0.2);
            }
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                border: none;
                background-color: transparent;
                width: 16px;
                subcontrol-position: left;
                subcontrol-origin: padding;
            }
            
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: rgba(52, 152, 219, 0.2);
            }
            
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #95A5A6;
                width: 8px;
                height: 5px;
            }
            
            QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
                border-bottom-color: #3498DB;
            }
            
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #95A5A6;
                width: 8px;
                height: 5px;
            }
            
            QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
                border-top-color: #3498DB;
            }
            """
        else:
            return """
            QSpinBox, QDoubleSpinBox {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                padding: 8px 12px;
                min-height: 28px;
            }
            
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #3498DB;
            }
            
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3498DB;
            }
            
            QSpinBox::up-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #7F8C8D;
            }
            
            QSpinBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #7F8C8D;
            }
            """
    
    def get_qslider_styles(self) -> str:
        """获取QSlider样式"""
        if self._mode == StyleMode.DARK:
            return """
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background-color: #34495E;
                border-radius: 3px;
                margin: 8px 0;
            }
            
            QSlider::groove:vertical {
                border: none;
                width: 6px;
                background-color: #34495E;
                border-radius: 3px;
                margin: 0 8px;
            }
            
            QSlider::handle:horizontal {
                background-color: #3498DB;
                border: 2px solid #2980B9;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:vertical {
                background-color: #3498DB;
                border: 2px solid #2980B9;
                width: 16px;
                height: 16px;
                margin: 0 -5px;
                border-radius: 8px;
            }
            
            QSlider::handle:hover {
                background-color: #5DADE2;
                border-color: #3498DB;
            }
            
            QSlider::handle:pressed {
                background-color: #2980B9;
            }
            
            QSlider::sub-page:horizontal {
                background-color: #3498DB;
                border-radius: 3px;
            }
            
            QSlider::add-page:horizontal {
                background-color: #34495E;
                border-radius: 3px;
            }
            
            QSlider::sub-page:vertical {
                background-color: #3498DB;
                border-radius: 3px;
            }
            
            QSlider::add-page:vertical {
                background-color: #34495E;
                border-radius: 3px;
            }
            """
        else:
            return """
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background-color: #DEE2E6;
                border-radius: 3px;
                margin: 8px 0;
            }
            
            QSlider::handle:horizontal {
                background-color: #3498DB;
                border: 2px solid #FFFFFF;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:hover {
                background-color: #5DADE2;
            }
            
            QSlider::sub-page:horizontal {
                background-color: #3498DB;
                border-radius: 3px;
            }
            """
    
    def get_qscrollbar_styles(self) -> str:
        """获取QScrollBar样式"""
        if self._mode == StyleMode.DARK:
            return """
            QScrollBar:vertical {
                background-color: transparent;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            
            QScrollBar:horizontal {
                background-color: transparent;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #34495E;
                min-height: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            
            QScrollBar::handle:horizontal {
                background-color: #34495E;
                min-width: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #3D566E;
                width: 14px;
                margin: 0px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background-color: #3D566E;
                height: 14px;
                margin: 0px;
            }
            
            QScrollBar::handle:vertical:pressed {
                background-color: #3498DB;
            }
            
            QScrollBar::handle:horizontal:pressed {
                background-color: #3498DB;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            
            /* Corner */
            QScrollArea > QScrollBar::corner {
                background-color: #1A1A2E;
            }
            """
        else:
            return """
            QScrollBar:vertical {
                background-color: #F8F9FA;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            
            QScrollBar:horizontal {
                background-color: #F8F9FA;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #BDC3C7;
                min-height: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            
            QScrollBar::handle:horizontal {
                background-color: #BDC3C7;
                min-width: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            
            QScrollBar::handle:hover {
                background-color: #95A5A6;
            }
            
            QScrollBar::handle:pressed {
                background-color: #3498DB;
            }
            """
    
    def get_qtreeview_styles(self) -> str:
        """获取QTreeView样式"""
        if self._mode == StyleMode.DARK:
            return """
            QTreeView, QListView, QTableView {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 4px;
                alternate-background-color: #34495E;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
                show-decoration-selected: 1;
            }
            
            QTreeView:hover, QListView:hover, QTableView:hover {
                border: 1px solid #3498DB;
            }
            
            QTreeView:focus, QListView:focus, QTableView:focus {
                border: 2px solid #3498DB;
            }
            
            /* Header */
            QHeaderView::section {
                background-color: #34495E;
                color: #ECF0F1;
                padding: 8px 12px;
                border: none;
                border-bottom: 2px solid #3498DB;
                font-weight: 600;
            }
            
            QHeaderView::section:hover {
                background-color: #3D566E;
            }
            
            QHeaderView::section:pressed {
                background-color: #2C3E50;
            }
            
            /* Corner */
            QHeaderView::corner {
                background-color: #34495E;
                border: none;
            }
            
            /* Items */
            QTreeView::item, QListView::item, QTableWidget::item {
                padding: 6px 8px;
                border-radius: 2px;
            }
            
            QTreeView::item:hover, QListView::item:hover {
                background-color: rgba(52, 152, 219, 0.1);
            }
            
            QTreeView::item:selected, QListView::item:selected {
                background-color: #3498DB;
            }
            
            /* Branch indicators */
            QTreeView::branch {
                background-color: transparent;
            }
            
            QTreeView::branch:has-siblings:!adjoins-item {
                border-image: url(none) 0;
            }
            
            /* Indicators */
            QTreeView::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
            }
            
            QTreeView::indicator:unchecked {
                border: 2px solid #95A5A6;
                background-color: transparent;
            }
            
            QTreeView::indicator:unchecked:hover {
                border-color: #3498DB;
            }
            
            QTreeView::indicator:checked {
                image: none;
                border: none;
                background-color: #3498DB;
            }
            
            QTreeView::indicator:checked:hover {
                background-color: #5DADE2;
            }
            
            QTreeView::indicator:indeterminate:hover {
                border-color: #3498DB;
                background-color: rgba(52, 152, 219, 0.2);
            }
            
            /* ScrollBar integration */
            QTreeView QScrollBar:vertical, QListView QScrollBar:vertical, 
            QTableView QScrollBar:vertical {
                background-color: #1A1A2E;
            }
            
            QTreeView QScrollBar:horizontal, QListView QScrollBar:horizontal,
            QTableView QScrollBar:horizontal {
                background-color: #1A1A2E;
            }
            """
        else:
            return """
            QTreeView, QListView, QTableView {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                alternate-background-color: #F8F9FA;
                selection-background-color: #3498DB;
                selection-color: #FFFFFF;
                show-decoration-selected: 1;
            }
            
            QTreeView:hover, QListView:hover, QTableView:hover {
                border: 1px solid #3498DB;
            }
            
            QHeaderView::section {
                background-color: #ECF0F1;
                color: #2C3E50;
                padding: 8px 12px;
                border: none;
                border-bottom: 2px solid #3498DB;
                font-weight: 600;
            }
            
            QHeaderView::section:hover {
                background-color: #DEE2E6;
            }
            
            QTreeView::item:selected, QListView::item:selected {
                background-color: #3498DB;
            }
            """
    
    def get_qgroupbox_styles(self) -> str:
        """获取QGroupBox样式"""
        if self._mode == StyleMode.DARK:
            return """
            QGroupBox {
                background-color: transparent;
                color: #3498DB;
                border: 1px solid #34495E;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: 600;
                font-size: 11pt;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 8px;
                background-color: #1A1A2E;
            }
            
            QGroupBox::title:hover {
                color: #5DADE2;
            }
            
            QGroupBox:disabled {
                color: #7F8C8D;
            }
            
            QGroupBox::flat {
                border: none;
                margin-top: 0px;
                padding-top: 0px;
            }
            
            QGroupBox::flat::title {
                background-color: transparent;
            }
            """
        else:
            return """
            QGroupBox {
                background-color: transparent;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: 600;
                font-size: 11pt;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 8px;
                background-color: #FFFFFF;
            }
            """
    
    def get_qtabwidget_styles(self) -> str:
        """获取QTabWidget样式"""
        if self._mode == StyleMode.DARK:
            return """
            QTabWidget::pane {
                background-color: #2C3E50;
                border: 1px solid #34495E;
                border-radius: 4px;
                top: -1px;
            }
            
            QTabWidget::pane:selected {
                border: 1px solid #3498DB;
                border-bottom-color: #2C3E50;
            }
            
            QTabBar::tab {
                background-color: #34495E;
                color: #95A5A6;
                border: 1px solid #34495E;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 10px 20px;
                min-width: 100px;
                font-weight: 500;
            }
            
            QTabBar::tab:hover {
                background-color: #3D566E;
                color: #ECF0F1;
            }
            
            QTabBar::tab:selected {
                background-color: #2C3E50;
                color: #3498DB;
                border: 1px solid #3498DB;
                border-bottom: 1px solid #2C3E50;
            }
            
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            
            /* Close button */
            QTabBar::close-button {
                image: none;
                width: 12px;
                height: 12px;
                border-radius: 6px;
                background-color: transparent;
            }
            
            QTabBar::close-button:hover {
                background-color: rgba(231, 76, 60, 0.8);
            }
            
            QTabBar::scroller {
                width: 40px;
            }
            
            QTabBar QToolButton {
                background-color: #34495E;
                border: none;
                border-radius: 4px;
            }
            
            QTabBar QToolButton:hover {
                background-color: #3D566E;
            }
            """
        else:
            return """
            QTabWidget::pane {
                background-color: #FFFFFF;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                top: -1px;
            }
            
            QTabBar::tab {
                background-color: #F8F9FA;
                color: #7F8C8D;
                border: 1px solid #BDC3C7;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 10px 20px;
                min-width: 100px;
            }
            
            QTabBar::tab:hover {
                background-color: #E9ECEF;
                color: #2C3E50;
            }
            
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #3498DB;
                border: 1px solid #3498DB;
                border-bottom: 1px solid #FFFFFF;
            }
            """
    
    def get_qmenu_styles(self) -> str:
        """获取QMenu样式"""
        if self._mode == StyleMode.DARK:
            return """
            QMenu {
                background-color: #2C3E50;
                color: #ECF0F1;
                border: 1px solid #34495E;
                border-radius: 6px;
                padding: 6px;
            }
            
            QMenu::item {
                padding: 8px 32px 8px 16px;
                border-radius: 4px;
                min-height: 24px;
            }
            
            QMenu::item:selected {
                background-color: #3498DB;
            }
            
            QMenu::item:disabled {
                color: #7F8C8D;
            }
            
            QMenu::separator {
                height: 1px;
                background-color: #34495E;
                margin: 6px 8px;
            }
            
            QMenu::indicator {
                width: 18px;
                height: 18px;
                margin-left: 4px;
            }
            
            QMenu::indicator:non-exclusive:unchecked {
                border: 2px solid #95A5A6;
                border-radius: 3px;
                background-color: transparent;
            }
            
            QMenu::indicator:non-exclusive:checked {
                image: none;
                border: none;
                background-color: #3498DB;
                border-radius: 3px;
            }
            
            QMenu::indicator:exclusive:unchecked {
                border: 2px solid #95A5A6;
                border-radius: 8px;
                background-color: transparent;
            }
            
            QMenu::indicator:exclusive:checked {
                image: none;
                border: none;
                background-color: #3498DB;
                border-radius: 8px;
            }
            
            QMenu::right-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #95A5A6;
            }
            
            /* Submenu */
            QMenu::sub-menu {
                background-color: #2C3E50;
            }
            
            /* Title */
            QMenu::title {
                background-color: #34495E;
                padding: 8px 16px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            """
        else:
            return """
            QMenu {
                background-color: #FFFFFF;
                color: #2C3E50;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                padding: 6px;
            }
            
            QMenu::item {
                padding: 8px 32px 8px 16px;
                border-radius: 4px;
            }
            
            QMenu::item:selected {
                background-color: #3498DB;
                color: #FFFFFF;
            }
            
            QMenu::separator {
                height: 1px;
                background-color: #E9ECEF;
                margin: 6px 8px;
            }
            
            QMenu::indicator:checked {
                background-color: #3498DB;
            }
            """
    
    def get_qmenubar_styles(self) -> str:
        """获取QMenuBar样式"""
        if self._mode == StyleMode.DARK:
            return """
            QMenuBar {
                background-color: #1A1A2E;
                color: #ECF0F1;
                border-bottom: 1px solid #34495E;
                padding: 4px;
            }
            
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background-color: #34495E;
            }
            
            QMenuBar::item:pressed {
                background-color: #3498DB;
            }
            """
        else:
            return """
            QMenuBar {
                background-color: #F8F9FA;
                color: #2C3E50;
                border-bottom: 1px solid #DEE2E6;
                padding: 4px;
            }
            
            QMenuBar::item:selected {
                background-color: #3498DB;
                color: #FFFFFF;
            }
            """
    
    def get_qtoolbar_styles(self) -> str:
        """获取QToolBar样式"""
        if self._mode == StyleMode.DARK:
            return """
            QToolBar {
                background-color: #1A1A2E;
                border: none;
                padding: 4px;
                spacing: 4px;
            }
            
            QToolBar::separator {
                width: 1px;
                background-color: #34495E;
                margin: 4px 8px;
            }
            
            QToolBar::handle {
                background-color: #34495E;
                border-radius: 4px;
                width: 8px;
                margin: 4px 2px;
            }
            
            QToolBar::handle:hover {
                background-color: #3498DB;
            }
            
            QToolButton {
                background-color: transparent;
                color: #ECF0F1;
                border: none;
                border-radius: 4px;
                padding: 6px;
                min-width: 32px;
                min-height: 32px;
            }
            
            QToolButton:hover {
                background-color: rgba(52, 152, 219, 0.2);
            }
            
            QToolButton:pressed {
                background-color: rgba(52, 152, 219, 0.3);
            }
            
            QToolButton:checked {
                background-color: #3498DB;
            }
            
            QToolButton:disabled {
                color: #7F8C8D;
            }
            """
        else:
            return """
            QToolBar {
                background-color: #F8F9FA;
                border: none;
                padding: 4px;
                spacing: 4px;
            }
            
            QToolBar::separator {
                background-color: #DEE2E6;
                margin: 4px 8px;
            }
            
            QToolButton:hover {
                background-color: #E9ECEF;
            }
            
            QToolButton:checked {
                background-color: #3498DB;
                color: #FFFFFF;
            }
            """
    
    def get_qtoolbutton_styles(self) -> str:
        """获取QToolButton样式"""
        return self.get_qtoolbar_styles()
    
    def get_qdialog_styles(self) -> str:
        """获取QDialog样式"""
        if self._mode == StyleMode.DARK:
            return """
            QDialog {
                background-color: #1A1A2E;
                color: #ECF0F1;
            }
            
            QDialog QLabel {
                color: #ECF0F1;
            }
            
            QDialog QPushButton {
                min-width: 80px;
            }
            """
        else:
            return """
            QDialog {
                background-color: #FFFFFF;
                color: #2C3E50;
            }
            """
    
    def get_qmessagebox_styles(self) -> str:
        """获取QMessageBox样式"""
        if self._mode == StyleMode.DARK:
            return """
            QMessageBox {
                background-color: #1A1A2E;
            }
            
            QMessageBox QLabel {
                color: #ECF0F1;
                padding: 8px;
            }
            
            QMessageBox QPushButton {
                min-width: 80px;
            }
            """
        else:
            return """
            QMessageBox {
                background-color: #FFFFFF;
            }
            """
    
    def get_qcheckbox_styles(self) -> str:
        """获取QCheckBox样式"""
        if self._mode == StyleMode.DARK:
            return """
            QCheckBox {
                color: #ECF0F1;
                spacing: 8px;
                padding: 4px;
            }
            
            QCheckBox:hover {
                color: #FFFFFF;
            }
            
            QCheckBox:disabled {
                color: #7F8C8D;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #95A5A6;
                border-radius: 4px;
                background-color: transparent;
            }
            
            QCheckBox::indicator:hover {
                border-color: #3498DB;
            }
            
            QCheckBox::indicator:checked {
                image: none;
                border: none;
                background-color: #3498DB;
                border-radius: 4px;
            }
            
            QCheckBox::indicator:checked:hover {
                background-color: #5DADE2;
            }
            
            QCheckBox::indicator:indeterminate {
                border: 2px solid #3498DB;
                background-color: rgba(52, 152, 219, 0.3);
            }
            
            QCheckBox::indicator:indeterminate:hover {
                background-color: rgba(52, 152, 219, 0.5);
            }
            """
        else:
            return """
            QCheckBox {
                color: #2C3E50;
                spacing: 8px;
                padding: 4px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #BDC3C7;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            
            QCheckBox::indicator:checked {
                background-color: #3498DB;
                border-color: #3498DB;
            }
            """
    
    def get_qradiobutton_styles(self) -> str:
        """获取QRadioButton样式"""
        if self._mode == StyleMode.DARK:
            return """
            QRadioButton {
                color: #ECF0F1;
                spacing: 8px;
                padding: 4px;
            }
            
            QRadioButton:hover {
                color: #FFFFFF;
            }
            
            QRadioButton:disabled {
                color: #7F8C8D;
            }
            
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #95A5A6;
                border-radius: 10px;
                background-color: transparent;
            }
            
            QRadioButton::indicator:hover {
                border-color: #3498DB;
            }
            
            QRadioButton::indicator:checked {
                border: 2px solid #3498DB;
                background-color: radial-gradient(
                    circle, #3498DB 6px, transparent 8px
                );
            }
            
            QRadioButton::indicator:checked:hover {
                border-color: #5DADE2;
            }
            """
        else:
            return """
            QRadioButton {
                color: #2C3E50;
                spacing: 8px;
                padding: 4px;
            }
            
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #BDC3C7;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
            
            QRadioButton::indicator:checked {
                border-color: #3498DB;
                background-color: radial-gradient(
                    circle, #3498DB 6px, transparent 8px
                );
            }
            """
    
    def get_qlabel_styles(self) -> str:
        """获取QLabel样式"""
        if self._mode == StyleMode.DARK:
            return """
            QLabel {
                color: #ECF0F1;
                background-color: transparent;
                padding: 2px;
            }
            
            QLabel:disabled {
                color: #7F8C8D;
            }
            
            QLabel[class="title"] {
                font-size: 14pt;
                font-weight: bold;
                color: #3498DB;
            }
            
            QLabel[class="subtitle"] {
                font-size: 11pt;
                color: #95A5A6;
            }
            
            QLabel[class="error"] {
                color: #E74C3C;
            }
            
            QLabel[class="success"] {
                color: #2ECC71;
            }
            
            QLabel[class="warning"] {
                color: #F39C12;
            }
            """
        else:
            return """
            QLabel {
                color: #2C3E50;
                background-color: transparent;
                padding: 2px;
            }
            
            QLabel:disabled {
                color: #ADB5BD;
            }
            """
    
    def get_qprogressbar_styles(self) -> str:
        """获取QProgressBar样式"""
        if self._mode == StyleMode.DARK:
            return """
            QProgressBar {
                background-color: #34495E;
                border: none;
                border-radius: 6px;
                height: 12px;
                text-align: center;
                color: #ECF0F1;
                font-size: 9pt;
                font-weight: 500;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498DB,
                    stop: 0.5 #2ECC71,
                    stop: 1 #27AE60
                );
                border-radius: 6px;
                margin: 2px;
            }
            
            QProgressBar:indeterminate {
                background-color: #34495E;
            }
            
            QProgressBar:indeterminate::chunk {
                background-color: #3498DB;
                border-radius: 6px;
            }
            """
        else:
            return """
            QProgressBar {
                background-color: #E9ECEF;
                border: none;
                border-radius: 6px;
                height: 12px;
                text-align: center;
                color: #2C3E50;
                font-size: 9pt;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498DB,
                    stop: 1 #2ECC71
                );
                border-radius: 6px;
            }
            """
    
    def get_qdockwidget_styles(self) -> str:
        """获取QDockWidget样式"""
        if self._mode == StyleMode.DARK:
            return """
            QDockWidget {
                color: #ECF0F1;
                titlebar-close-icon: url(none);
                titlebar-normal-icon: url(none);
            }
            
            QDockWidget::title {
                background-color: #2C3E50;
                text-align: left;
                padding: 8px;
                border: 1px solid #34495E;
                border-radius: 4px 4px 0 0;
            }
            
            QDockWidget::title:hover {
                background-color: #34495E;
            }
            
            QDockWidget::close-button {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
                subcontrol-position: right;
            }
            
            QDockWidget::close-button:hover {
                background-color: rgba(231, 76, 60, 0.8);
            }
            
            QDockWidget::float-button {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
                subcontrol-position: right;
            }
            
            QDockWidget::float-button:hover {
                background-color: rgba(52, 152, 219, 0.8);
            }
            
            QDockWidget::content {
                background-color: #2C3E50;
                border: 1px solid #34495E;
                border-top: none;
                border-radius: 0 0 4px 4px;
            }
            """
        else:
            return """
            QDockWidget::title {
                background-color: #F8F9FA;
                text-align: left;
                padding: 8px;
                border: 1px solid #BDC3C7;
            }
            
            QDockWidget::content {
                background-color: #FFFFFF;
                border: 1px solid #BDC3C7;
                border-top: none;
            }
            """
    
    def get_qsplitter_styles(self) -> str:
        """获取QSplitter样式"""
        if self._mode == StyleMode.DARK:
            return """
            QSplitter {
                background-color: #1A1A2E;
            }
            
            QSplitter::handle {
                background-color: #34495E;
            }
            
            QSplitter::handle:horizontal {
                width: 2px;
            }
            
            QSplitter::handle:vertical {
                height: 2px;
            }
            
            QSplitter::handle:hover {
                background-color: #3498DB;
            }
            
            QSplitter::handle:pressed {
                background-color: #2980B9;
            }
            """
        else:
            return """
            QSplitter::handle {
                background-color: #DEE2E6;
            }
            
            QSplitter::handle:hover {
                background-color: #3498DB;
            }
            """
    
    def get_qframe_styles(self) -> str:
        """获取QFrame样式"""
        if self._mode == StyleMode.DARK:
            return """
            QFrame[frameShape="4"], QFrame[frameShape="5"] {
                border: none;
                background-color: #34495E;
            }
            
            QFrame[frameShape="4"] {
                max-height: 1px;
            }
            
            QFrame[frameShape="5"] {
                max-width: 1px;
            }
            """
        else:
            return """
            QFrame[frameShape="4"] {
                background-color: #DEE2E6;
                max-height: 1px;
            }
            
            QFrame[frameShape="5"] {
                background-color: #DEE2E6;
                max-width: 1px;
            }
            """
    
    def get_complete_stylesheet(self) -> str:
        """
        获取完整的QSS样式表
        
        返回:
            完整的QSS字符串
        """
        styles = []
        
        # 基础组件
        styles.append(self.get_qwidget_styles())
        styles.append(self.get_qmainwindow_styles())
        
        # 按钮类
        styles.append(self.get_qpushbutton_styles())
        styles.append(self.get_qtoolbutton_styles())
        
        # 输入类
        styles.append(self.get_qlineedit_styles())
        styles.append(self.get_qtextedit_styles())
        styles.append(self.get_qcombobox_styles())
        styles.append(self.get_qspinbox_styles())
        styles.append(self.get_qslider_styles())
        
        # 选择类
        styles.append(self.get_qcheckbox_styles())
        styles.append(self.get_qradiobutton_styles())
        
        # 显示类
        styles.append(self.get_qlabel_styles())
        styles.append(self.get_qprogressbar_styles())
        
        # 容器类
        styles.append(self.get_qgroupbox_styles())
        styles.append(self.get_qtabwidget_styles())
        styles.append(self.get_qtreeview_styles())
        styles.append(self.get_qdockwidget_styles())
        styles.append(self.get_qsplitter_styles())
        
        # 菜单类
        styles.append(self.get_qmenu_styles())
        styles.append(self.get_qmenubar_styles())
        styles.append(self.get_qtoolbar_styles())
        
        # 对话框
        styles.append(self.get_qdialog_styles())
        styles.append(self.get_qmessagebox_styles())
        
        # 滚动条
        styles.append(self.get_qscrollbar_styles())
        
        # 框架
        styles.append(self.get_qframe_styles())
        
        return "\n\n".join(styles)
    
    def get_component_style(self, component: str) -> str:
        """
        获取特定组件样式
        
        参数:
            component: 组件名称
        
        返回:
            QSS字符串
        """
        component_map = {
            'mainwindow': self.get_qmainwindow_styles,
            'widget': self.get_qwidget_styles,
            'pushbutton': self.get_qpushbutton_styles,
            'toolbutton': self.get_qtoolbutton_styles,
            'lineedit': self.get_qlineedit_styles,
            'textedit': self.get_qtextedit_styles,
            'combobox': self.get_qcombobox_styles,
            'spinbox': self.get_qspinbox_styles,
            'slider': self.get_qslider_styles,
            'checkbox': self.get_qcheckbox_styles,
            'radiobutton': self.get_qradiobutton_styles,
            'label': self.get_qlabel_styles,
            'progressbar': self.get_qprogressbar_styles,
            'groupbox': self.get_qgroupbox_styles,
            'tabwidget': self.get_qtabwidget_styles,
            'treeview': self.get_qtreeview_styles,
            'listview': self.get_qtreeview_styles,
            'tableview': self.get_qtreeview_styles,
            'dockwidget': self.get_qdockwidget_styles,
            'splitter': self.get_qsplitter_styles,
            'menu': self.get_qmenu_styles,
            'menubar': self.get_qmenubar_styles,
            'toolbar': self.get_qtoolbar_styles,
            'dialog': self.get_qdialog_styles,
            'messagebox': self.get_qmessagebox_styles,
            'scrollbar': self.get_qscrollbar_styles,
            'frame': self.get_qframe_styles,
        }
        
        getter = component_map.get(component.lower())
        if getter:
            return getter()
        return ""


# 全局样式实例
_default_styles = PaleoASTStyles()


def get_dark_stylesheet() -> str:
    """获取深色模式样式表"""
    return PaleoASTStyles(StyleMode.DARK).get_complete_stylesheet()


def get_light_stylesheet() -> str:
    """获取浅色模式样式表"""
    return PaleoASTStyles(StyleMode.LIGHT).get_complete_stylesheet()


def get_default_stylesheet() -> str:
    """获取默认样式表"""
    return _default_styles.get_complete_stylesheet()
