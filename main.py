"""
================================================================================
PaleoAST - Paleontological Advanced Statistical Toolkit
================================================================================

专业古生物学高级统计分析平台

主入口文件 - 商业级启动流程
- 全局异常拦截
- 闪屏动画
- 主题样式注入
- 模块懒加载

版本: 5.0.0
作者: PaleoAST Development Team
许可证: MIT
"""

import sys
import os
import logging
import time
import traceback
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum

# 防止在导入前就崩溃
try:
    import numpy as np
    import scipy as sp
except ImportError:
    print("ERROR: Required scientific computing packages not found!")
    print("Please install: numpy scipy")
    sys.exit(1)


# =============================================================================
# 配置日志系统
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    设置日志系统
    
    返回:
        配置好的根日志记录器
    """
    # 创建日志目录
    log_dir = Path.home() / ".paleoast" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件
    log_file = log_dir / f"paleoast_{time.strftime('%Y%m%d')}.log"
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


# =============================================================================
# 全局异常处理
# =============================================================================

class ExceptionHandler:
    """
    全局异常处理器
    
    捕获所有未处理异常，防止程序闪退。
    """
    
    def __init__(self):
        """初始化"""
        self._original_hook = None
        self._logger = logging.getLogger("PaleoAST.ExceptionHandler")
    
    def install(self) -> None:
        """安装异常处理器"""
        self._original_hook = sys.excepthook
        sys.excepthook = self._handle_exception
        self._logger.info("Global exception handler installed")
    
    def _handle_exception(
        self,
        exc_type: type,
        exc_value: BaseException,
        exc_tb: Any
    ) -> None:
        """处理异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许Ctrl+C
            self._original_hook(exc_type, exc_value, exc_tb)
            return
        
        # 记录错误
        self._logger.critical(
            f"Unhandled exception: {exc_type.__name__}: {exc_value}",
            exc_info=(exc_type, exc_value, exc_tb)
        )
        
        # 尝试显示错误对话框
        self._show_error_dialog(exc_type, exc_value, exc_tb)
    
    def _show_error_dialog(
        self,
        exc_type: type,
        exc_value: BaseException,
        exc_tb: Any
    ) -> None:
        """显示错误对话框"""
        try:
            # 导入Qt
            from PyQt6.QtWidgets import (
                QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                QLabel, QPushButton, QTextEdit, QGroupBox, QMessageBox
            )
            from PyQt6.QtCore import Qt
            from PyQt6.QtGui import QFont
            
            app = QApplication.instance()
            if not app:
                return
            
            # 创建对话框
            dialog = QDialog()
            from config.i18n import _
            dialog.setWindowTitle(_("PaleoAST - Error Detected"))
            dialog.setMinimumSize(700, 500)
            dialog.setModal(True)
            
            layout = QVBoxLayout(dialog)
            
            # 标题
            title = QLabel("⚠️  " + _("Application Error"))
            title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            title.setStyleSheet("color: #E74C3C; padding: 10px;")
            
            # 信息
            info = QLabel(
                _("PaleoAST encountered an error. The error has been logged.\n"
                  "You can continue, but unexpected behavior may occur.")
            )
            info.setWordWrap(True)
            
            # 错误详情
            tb_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            
            group = QGroupBox(_("Technical Details"))
            group_layout = QVBoxLayout(group)
            
            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setFont(QFont("Consolas", 8))
            error_text.setText(
                f"Type: {exc_type.__name__}\n"
                f"Message: {exc_value}\n\n"
                f"Traceback:\n{tb_text}"
            )
            error_text.setStyleSheet("""
                QTextEdit {
                    background-color: #2C3E50;
                    color: #ECF0F1;
                    border: 1px solid #34495E;
                    border-radius: 4px;
                    padding: 8px;
                }
            """)
            group_layout.addWidget(error_text)
            
            # 按钮
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            
            export_btn = QPushButton(_("Export Log"))
            export_btn.clicked.connect(lambda: self._export_log(tb_text))
            
            continue_btn = QPushButton(_("Continue"))
            continue_btn.setDefault(True)
            continue_btn.clicked.connect(dialog.accept)
            
            quit_btn = QPushButton(_("Quit"))
            quit_btn.setStyleSheet("background-color: #E74C3C; color: white;")
            quit_btn.clicked.connect(lambda: sys.exit(1))
            
            btn_layout.addWidget(export_btn)
            btn_layout.addWidget(continue_btn)
            btn_layout.addWidget(quit_btn)
            
            # 组装
            layout.addWidget(title)
            layout.addWidget(info)
            layout.addWidget(group, stretch=1)
            layout.addLayout(btn_layout)
            
            dialog.exec()
            
        except Exception as e:
            # 降级到控制台
            print("=" * 80)
            print("CRITICAL ERROR:")
            print(f"Type: {exc_type.__name__}")
            print(f"Message: {exc_value}")
            print("Traceback:")
            print(''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            print("=" * 80)
    
    def _export_log(self, tb_text: str) -> None:
        """导出日志"""
        try:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"paleoast_error_{timestamp}.log"
            
            from config.i18n import _
            path, _ = QFileDialog.getSaveFileName(
                None,
                _("Export Error Log"),
                str(Path.home() / "Desktop" / filename),
                _("Log Files (*.log);;Text Files (*.txt)")
            )

            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(tb_text)
                QMessageBox.information(None, _("Exported"), _("Log saved to:\n{0}").format(path))
        except Exception as e:
            # 忽略导出失败，继续执行
            pass


# =============================================================================
# 主题样式
# =============================================================================

def get_light_theme_stylesheet() -> str:
    """获取现代清爽亮色主题样式表 (Modern Flat Light Theme)"""
    from config.design_system import get_modern_stylesheet
    return get_modern_stylesheet()

def get_dark_theme_stylesheet() -> str:
    """
    获取深色主题样式表
    
    返回:
        QSS样式字符串
    """
    return """
    /* ====================================================================
       PaleoAST Professional Dark Theme
       ==================================================================== */
    
    /* 全局字体 */
    * {
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 10pt;
    }
    
    /* 主窗口 */
    QMainWindow {
        background-color: #1A1A2E;
        color: #ECF0F1;
    }
    
    /* 中央部件 */
    QWidget {
        background-color: #1A1A2E;
        color: #ECF0F1;
    }
    
    /* 按钮 */
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
    }
    
    QPushButton:disabled {
        background-color: #2C3E50;
        color: #7F8C8D;
    }
    
    /* 主按钮 */
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
    
    /* 输入框 */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #2C3E50;
        color: #ECF0F1;
        border: 1px solid #34495E;
        border-radius: 4px;
        padding: 8px 12px;
        selection-background-color: #3498DB;
    }
    
    QLineEdit:hover, QTextEdit:hover {
        border: 1px solid #3498DB;
    }
    
    QLineEdit:focus, QTextEdit:focus {
        border: 2px solid #3498DB;
    }
    
    /* 组合框 */
    QComboBox {
        background-color: #2C3E50;
        color: #ECF0F1;
        border: 1px solid #34495E;
        border-radius: 4px;
        padding: 8px 12px;
    }
    
    QComboBox:hover {
        border: 1px solid #3498DB;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2C3E50;
        color: #ECF0F1;
        border: 1px solid #34495E;
        selection-background-color: #3498DB;
    }
    
    /* 滚动条 */
    QScrollBar:vertical {
        background-color: transparent;
        width: 10px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background-color: #34495E;
        min-height: 30px;
        border-radius: 5px;
        margin: 2px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #3D566E;
        width: 14px;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    
    QScrollBar:horizontal {
        background-color: transparent;
        height: 10px;
        margin: 0;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #34495E;
        min-width: 30px;
        border-radius: 5px;
        margin: 2px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #3D566E;
        height: 14px;
    }
    
    /* 表格 */
    QTableWidget, QTreeWidget, QListWidget {
        background-color: #2C3E50;
        color: #ECF0F1;
        border: 1px solid #34495E;
        border-radius: 4px;
        gridline-color: #34495E;
        alternate-background-color: #34495E;
    }
    
    QHeaderView::section {
        background-color: #34495E;
        color: #ECF0F1;
        padding: 8px;
        border: none;
        border-bottom: 2px solid #3498DB;
        font-weight: 600;
    }
    
    /* 菜单 */
    QMenuBar {
        background-color: #1A1A2E;
        color: #ECF0F1;
        border-bottom: 1px solid #34495E;
    }
    
    QMenuBar::item:selected {
        background-color: #34495E;
    }
    
    QMenu {
        background-color: #2C3E50;
        color: #ECF0F1;
        border: 1px solid #34495E;
        border-radius: 6px;
        padding: 4px;
    }
    
    QMenu::item:selected {
        background-color: #3498DB;
    }
    
    QMenu::separator {
        height: 1px;
        background-color: #34495E;
    }
    
    /* 标签页 */
    QTabWidget::pane {
        background-color: #2C3E50;
        border: 1px solid #34495E;
        border-radius: 4px;
    }
    
    QTabBar::tab {
        background-color: #34495E;
        color: #95A5A6;
        border: 1px solid #34495E;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 10px 20px;
    }
    
    QTabBar::tab:hover {
        background-color: #3D566E;
    }
    
    QTabBar::tab:selected {
        background-color: #2C3E50;
        color: #3498DB;
        border: 1px solid #3498DB;
    }
    
    /* 工具栏 */
    QToolBar {
        background-color: #1A1A2E;
        border: none;
        spacing: 4px;
    }
    
    QToolButton {
        background-color: transparent;
        color: #ECF0F1;
        border: none;
        border-radius: 4px;
        padding: 6px;
    }
    
    QToolButton:hover {
        background-color: rgba(52, 152, 219, 0.2);
    }
    
    /* 分组框 */
    QGroupBox {
        background-color: transparent;
        color: #3498DB;
        border: 1px solid #34495E;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: 600;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 8px;
        background-color: #1A1A2E;
    }
    
    /* 进度条 */
    QProgressBar {
        background-color: #34495E;
        border: none;
        border-radius: 6px;
        height: 12px;
        text-align: center;
        color: #ECF0F1;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #3498DB,
            stop: 0.5 #2ECC71,
            stop: 1 #27AE60
        );
        border-radius: 6px;
    }
    
    /* 标签 */
    QLabel {
        color: #ECF0F1;
        background-color: transparent;
    }
    
    /* 停靠窗口 */
    QDockWidget {
        color: #ECF0F1;
        border: 1px solid #34495E;
        titlebar-close-icon: url(none);
        titlebar-normal-icon: url(none);
    }
    
    QDockWidget::title {
        background-color: #2C3E50;
        padding: 8px;
    }
    
    /* 分裂器 */
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
    
    /* 复选框 */
    QCheckBox {
        color: #ECF0F1;
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #95A5A6;
        border-radius: 3px;
        background-color: transparent;
    }
    
    QCheckBox::indicator:checked {
        background-color: #3498DB;
        border-color: #3498DB;
    }
    
    /* 单选框 */
    QRadioButton {
        color: #ECF0F1;
        spacing: 8px;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #95A5A6;
        border-radius: 9px;
        background-color: transparent;
    }
    
    QRadioButton::indicator:checked {
        border-color: #3498DB;
        background-color: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 #3498DB, stop:0.75 #3498DB, stop:0.76 transparent);
    }
    
    /* 滑块 */
    QSlider::groove:horizontal {
        height: 6px;
        background-color: #34495E;
        border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background-color: #3498DB;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    
    QSlider::sub-page:horizontal {
        background-color: #3498DB;
        border-radius: 3px;
    }
    
    /* 状态栏 */
    QStatusBar {
        background-color: #1A1A2E;
        color: #95A5A6;
        border-top: 1px solid #34495E;
    }
    
    QStatusBar::item {
        border: none;
    }
    
    /* 对话框 */
    QDialog {
        background-color: #1A1A2E;
    }
    
    /* 消息框 */
    QMessageBox {
        background-color: #1A1A2E;
    }
    
    QMessageBox QLabel {
        color: #ECF0F1;
    }
    """


# =============================================================================
# 闪屏窗口
# =============================================================================

class SplashScreen:
    """
    启动闪屏窗口
    
    显示加载进度和状态信息。
    """
    
    def __init__(self):
        """初始化"""
        self._logger = logging.getLogger("PaleoAST.Splash")
        self._widget: Optional[Any] = None
        self._label: Optional[Any] = None
        self._progress: Optional[Any] = None
        self._status: Optional[Any] = None
    
    def show(self) -> None:
        """显示闪屏"""
        try:
            from PyQt6.QtWidgets import (
                QApplication, QWidget, QVBoxLayout, QLabel,
                QProgressBar, QGraphicsOpacityEffect
            )
            from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
            from PyQt6.QtGui import QFont, QPainter, QLinearGradient, QColor
            
            app = QApplication.instance()
            if not app:
                return
            
            # 创建窗口
            self._widget = QWidget(
                None,
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
            )
            self._widget.setFixedSize(500, 350)
            self._widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            
            # 居中
            screen = app.primaryScreen()
            if screen:
                geo = screen.geometry()
                self._widget.move(
                    (geo.width() - 500) // 2,
                    (geo.height() - 350) // 2
                )
            
            # 布局
            layout = QVBoxLayout(self._widget)
            layout.setContentsMargins(40, 60, 40, 40)
            layout.setSpacing(15)
            
            # 标题
            title = QLabel("PaleoAST")
            title.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("color: #FFFFFF; background: transparent;")
            
            # 副标题
            from config.i18n import _
            subtitle = QLabel(_("Paleontological Advanced Statistical Toolkit"))
            subtitle.setFont(QFont("Segoe UI", 10))
            subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle.setStyleSheet("color: #3498DB; background: transparent;")
            
            # 进度条
            self._progress = QProgressBar()
            self._progress.setFixedHeight(8)
            self._progress.setTextVisible(False)
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setStyleSheet("""
                QProgressBar {
                    background-color: rgba(52, 152, 219, 0.2);
                    border: none;
                    border-radius: 4px;
                }
                QProgressBar::chunk {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #3498DB,
                        stop: 1 #2ECC71
                    );
                    border-radius: 4px;
                }
            """)
            
            # 状态
            self._status = QLabel(_("Initializing..."))
            self._status.setFont(QFont("Consolas", 9))
            self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._status.setStyleSheet("color: #95A5A6; background: transparent;")
            self._status.setWordWrap(True)
            
            # 版本
            version = QLabel(_("Version 5.0.0"))
            version.setFont(QFont("Segoe UI", 8))
            version.setAlignment(Qt.AlignmentFlag.AlignRight)
            version.setStyleSheet("color: #7F8C8D; background: transparent;")
            
            # 添加到布局
            layout.addStretch(1)
            layout.addWidget(title)
            layout.addWidget(subtitle)
            layout.addSpacing(30)
            layout.addWidget(self._progress)
            layout.addSpacing(10)
            layout.addWidget(self._status, stretch=1)
            layout.addWidget(version)
            
            # 淡入动画
            effect = QGraphicsOpacityEffect(self._widget)
            self._widget.setGraphicsEffect(effect)
            
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(800)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            
            self._widget.show()
            anim.start()
            
            app.processEvents()
            
        except Exception as e:
            self._logger.error(f"Failed to show splash: {e}")
    
    def update(self, value: int, status: str = "") -> None:
        """
        更新进度
        
        参数:
            value: 进度值 0-100
            status: 状态消息
        """
        try:
            from PyQt6.QtWidgets import QApplication
            
            if self._progress:
                self._progress.setValue(value)
            
            if self._status and status:
                self._status.setText(status)
            
            app = QApplication.instance()
            if app:
                app.processEvents()
                
        except Exception as e:
            self._logger.error(f"Failed to update splash: {e}")
    
    def close(self) -> None:
        """关闭闪屏"""
        try:
            from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect
            from PyQt6.QtCore import QPropertyAnimation, Qt as QtCore
            
            if not self._widget:
                return
            
            # 淡出动画
            effect = QGraphicsOpacityEffect(self._widget)
            self._widget.setGraphicsEffect(effect)
            
            anim = QPropertyAnimation(effect, b"opacity", self._widget)
            anim.setDuration(500)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            from PyQt6.QtCore import QEasingCurve
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(self._widget.close)
            anim.start()
            
            app = QApplication.instance()
            if app:
                app.processEvents()
                
        except Exception as e:
            self._logger.error(f"Failed to close splash: {e}")
            if self._widget:
                self._widget.close()


# =============================================================================
# 主应用类
# =============================================================================

class PaleoASTApplication:
    """
    PaleoAST应用程序主类
    
    管理应用程序的完整生命周期。
    """
    
    VERSION = "5.0.0"
    APP_NAME = "PaleoAST"
    
    def __init__(self):
        """初始化"""
        self._logger = logging.getLogger("PaleoAST")
        self._app: Optional[Any] = None
        self._splash: Optional[SplashScreen] = None
        self._main_window: Optional[Any] = None
        self._exception_handler: Optional[ExceptionHandler] = None
    
    def run(self) -> int:
        """
        运行应用程序
        
        返回:
            退出代码
        """
        try:
            # 1. 初始化Qt
            self._init_qt()

            # 1.5 初始化国际化
            self._init_i18n()

            # 2. 显示闪屏
            self._splash = SplashScreen()
            self._splash.show()
            
            # 3. 安装异常处理器
            self._exception_handler = ExceptionHandler()
            self._exception_handler.install()
            
            # 4. 预热NumPy/SciPy
            from config.i18n import _
            self._splash.update(20, _("Warming up NumPy/SciPy..."))
            self._warmup_scientific()

            # 5. 加载配置
            self._splash.update(40, _("Loading configuration..."))
            self._load_config()

            # 6. 加载模块
            self._splash.update(60, _("Loading modules..."))
            self._load_modules()

            # 7. 创建主窗口
            self._splash.update(80, _("Creating main window..."))
            self._create_main_window()

            # 8. 应用主题
            self._splash.update(90, _("Applying theme..."))
            self._apply_theme()

            # 9. 完成
            self._splash.update(100, _("Ready!"))
            time.sleep(0.5)
            
            # 10. 关闭闪屏并显示主窗口
            self._splash.close()
            self._show_main_window()
            
            self._logger.info(f"{self.APP_NAME} v{self.VERSION} started successfully")
            
            # 进入事件循环
            return self._app.exec()
            
        except Exception as e:
            self._logger.critical(f"Failed to start application: {e}")
            traceback.print_exc()
            return 1
    
    def _init_qt(self) -> None:
        """初始化Qt"""
        self._logger.info("Initializing Qt...")
        
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        
        self._app = QApplication(sys.argv)
        self._app.setApplicationName(self.APP_NAME)
        self._app.setApplicationVersion(self.VERSION)
        self._app.setOrganizationName("PaleoAST")
        self._app.setQuitOnLastWindowClosed(True)

    def _init_i18n(self) -> None:
        """初始化国际化系统"""
        from config.i18n import register_translations, get_translator, _reset_translator
        # Reset singleton so it picks up QObject support now that QApplication exists
        _reset_translator()
        register_translations()

        translator = get_translator()

        # 加载保存的语言偏好
        from PyQt6.QtCore import QSettings
        settings = QSettings("PaleoAST", "PaleoAST")
        saved_lang = settings.value("language", "en")
        if saved_lang in ("en", "zh"):
            translator.set_language(saved_lang)

        self._logger.info(f"i18n initialized, language: {translator.get_language()}")

    def _warmup_scientific(self) -> None:
        """预热科学计算库"""
        self._logger.info("Warming up scientific libraries...")
        
        import numpy as np
        import scipy as sp
        
        # 执行简单计算确保库已加载
        _ = np.dot(np.random.rand(100, 100), np.random.rand(100, 100))
        _ = sp.linalg.svd(np.random.rand(50, 50))
        
        self._logger.info("Scientific libraries warmed up")
    
    def _load_config(self) -> None:
        """加载配置"""
        self._logger.info("Loading configuration...")
        
        try:
            import config
            self._logger.info("Configuration loaded")
        except ImportError:
            self._logger.warning("Config module not found, using defaults")
    
    def _load_modules(self) -> None:
        """加载模块"""
        self._logger.info("Loading modules...")
        
        from config.i18n import _
        modules = [
            ("models", _("Data models")),
            ("statistics", _("Statistics engine")),
            ("morphometrics", _("Morphometrics")),
            ("ecology", _("Ecology engine")),
            ("stratigraphy", _("Stratigraphy")),
            ("visualization", _("Visualization")),
            ("controllers", _("Controllers")),
            ("views", _("Views")),
        ]
        
        loaded = []
        for module_name, description in modules:
            try:
                __import__(module_name)
                loaded.append(module_name)
                self._splash.update(
                    60 + len(loaded) * 2,
                    _("Loaded {0}...").format(description)
                )
                self._logger.info(f"Module loaded: {module_name}")
            except ImportError as e:
                self._logger.warning(f"Module not loaded: {module_name} - {e}")
    
    def _create_main_window(self) -> None:
        """创建主窗口"""
        self._logger.info("Creating main window...")
        
        try:
            from PyQt6.QtWidgets import (
                QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QLabel, QMenuBar, QMenu, QStatusBar, QToolBar,
                QDockWidget, QTextEdit, QTabWidget, QSplitter
            )
            from PyQt6.QtCore import Qt
            
            # 尝试导入自定义主窗口
            try:
                from views.ui_main_window import MainWindow
                self._main_window = MainWindow()
            except ImportError:
                # 使用默认主窗口
                self._main_window = QMainWindow()
                self._main_window.setWindowTitle(f"{self.APP_NAME} v{self.VERSION}")
                self._main_window.resize(1200, 800)
                
                # 中心部件
                central = QWidget()
                layout = QVBoxLayout(central)
                
                # 欢迎标签
                welcome = QLabel(
                    f"<h1>Welcome to {self.APP_NAME}</h1>"
                    f"<p>Version {self.VERSION}</p>"
                    f"<p>Professional Paleontology Analysis Platform</p>"
                )
                welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
                welcome.setStyleSheet("""
                    QLabel {
                        color: #3498DB;
                        padding: 50px;
                    }
                """)
                layout.addWidget(welcome)
                
                self._main_window.setCentralWidget(central)
                
                # 菜单栏
                self._create_menu_bar()
                
                # 状态栏
                status = QStatusBar()
                status.showMessage("Ready")
                self._main_window.setStatusBar(status)
            
            self._logger.info("Main window created")
            
        except Exception as e:
            self._logger.error(f"Failed to create main window: {e}")
            raise
    
    def _create_menu_bar(self) -> None:
        """创建菜单栏"""
        try:
            from PyQt6.QtWidgets import QMenuBar, QMenu
            
            menubar = self._main_window.menuBar()
            
            # 文件菜单
            file_menu = menubar.addMenu("&File")
            
            # 编辑菜单
            edit_menu = menubar.addMenu("&Edit")
            
            # 数据菜单
            data_menu = menubar.addMenu("&Data")
            
            # 统计菜单
            stats_menu = menubar.addMenu("&Statistics")
            
            # 形态测量菜单
            morph_menu = menubar.addMenu("&Morphometrics")
            
            # 可视化菜单
            view_menu = menubar.addMenu("&Visualization")
            
            # 帮助菜单
            help_menu = menubar.addMenu("&Help")
            
        except Exception as e:
            self._logger.warning(f"Failed to create menu bar: {e}")
    
    def _apply_theme(self) -> None:
        """应用主题"""
        self._logger.info("Applying theme...")
        
        stylesheet = get_light_theme_stylesheet()
        self._app.setStyleSheet(stylesheet)
        
        self._logger.info("Theme applied")
    
    def _show_main_window(self) -> None:
        """显示主窗口"""
        if self._main_window:
            self._main_window.show()
            self._main_window.activateWindow()
            self._main_window.raise_()


# =============================================================================
# 程序入口
# =============================================================================

def main() -> int:
    """
    主入口函数
    
    返回:
        退出代码
    """
    # 设置日志
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info(f"PaleoAST v{PaleoASTApplication.VERSION} Starting...")
    logger.info("=" * 60)
    
    # 创建并运行应用
    app = PaleoASTApplication()
    exit_code = app.run()
    
    logger.info(f"PaleoAST exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
