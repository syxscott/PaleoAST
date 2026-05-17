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

版本: 1.0.0
作者: PaleoAST Development Team
许可证: MIT
"""

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# 防止在导入前就崩溃
try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
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
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
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

    def _handle_exception(self, exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        """处理异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许Ctrl+C
            self._original_hook(exc_type, exc_value, exc_tb)
            return

        # 记录错误
        self._logger.critical(
            f"Unhandled exception: {exc_type.__name__}: {exc_value}", exc_info=(exc_type, exc_value, exc_tb)
        )

        # 尝试显示错误对话框
        self._show_error_dialog(exc_type, exc_value, exc_tb)

    def _show_error_dialog(self, exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        """显示错误对话框"""
        try:
            # 导入Qt
            from PyQt6.QtGui import QFont
            from PyQt6.QtWidgets import (
                QApplication,
                QDialog,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QPushButton,
                QTextEdit,
                QVBoxLayout,
            )

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
                _(
                    "PaleoAST encountered an error. The error has been logged.\n"
                    "You can continue, but unexpected behavior may occur."
                )
            )
            info.setWordWrap(True)

            # 错误详情
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

            group = QGroupBox(_("Technical Details"))
            group_layout = QVBoxLayout(group)

            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setFont(QFont("Consolas", 8))
            error_text.setText(f"Type: {exc_type.__name__}\nMessage: {exc_value}\n\nTraceback:\n{tb_text}")
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

        except Exception:
            # 降级到控制台
            print("=" * 80)
            print("CRITICAL ERROR:")
            print(f"Type: {exc_type.__name__}")
            print(f"Message: {exc_value}")
            print("Traceback:")
            print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
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
                _("Log Files (*.log);;Text Files (*.txt)"),
            )

            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(tb_text)
                QMessageBox.information(None, _("Exported"), _("Log saved to:\n{0}").format(path))
        except Exception:
            # 忽略导出失败，继续执行
            pass


# =============================================================================
# 主题样式
# =============================================================================


def get_light_theme_stylesheet() -> str:
    """获取现代清爽亮色主题样式表 (Modern Flat Light Theme)"""
    from config.design_system import get_modern_stylesheet

    return get_modern_stylesheet()



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
        self._widget: Any | None = None
        self._label: Any | None = None
        self._progress: Any | None = None
        self._status: Any | None = None
        self._fade_anim: Any | None = None  # Keep animation alive

    def show(self) -> None:
        """显示闪屏"""
        try:
            from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
            from PyQt6.QtGui import QFont
            from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

            app = QApplication.instance()
            if not app:
                return

            # 创建窗口
            self._widget = QWidget(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            self._widget.setFixedSize(500, 350)
            # Use solid dark background first, then set opacity for fade-in
            self._widget.setStyleSheet("""
                QWidget {
                    background-color: #1A1A2E;
                    border-radius: 12px;
                }
            """)
            # Start fully transparent for fade-in effect
            self._widget.setWindowOpacity(0.0)

            # 居中
            screen = app.primaryScreen()
            if screen:
                geo = screen.geometry()
                self._widget.move((geo.width() - 500) // 2, (geo.height() - 350) // 2)

            # 布局
            layout = QVBoxLayout(self._widget)
            layout.setContentsMargins(40, 60, 40, 40)
            layout.setSpacing(15)

            # 标题
            title = QLabel("PaleoAST")
            title.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("color: #FFFFFF;")

            # 副标题
            from config.i18n import _

            subtitle = QLabel(_("Paleontological Advanced Statistical Toolkit"))
            subtitle.setFont(QFont("Segoe UI", 10))
            subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle.setStyleSheet("color: #3498DB;")

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
            self._status.setStyleSheet("color: #95A5A6;")
            self._status.setWordWrap(True)

            # 版本
            version = QLabel(_("Version {0}").format(PaleoASTApplication.VERSION))
            version.setFont(QFont("Segoe UI", 8))
            version.setAlignment(Qt.AlignmentFlag.AlignRight)
            version.setStyleSheet("color: #7F8C8D;")

            # 添加到布局
            layout.addStretch(1)
            layout.addWidget(title)
            layout.addWidget(subtitle)
            layout.addSpacing(30)
            layout.addWidget(self._progress)
            layout.addSpacing(10)
            layout.addWidget(self._status, stretch=1)
            layout.addWidget(version)

            # 淡入动画 - keep reference to prevent GC
            self._fade_anim = QPropertyAnimation(self._widget, b"windowOpacity")
            self._fade_anim.setDuration(800)
            self._fade_anim.setStartValue(0.0)
            self._fade_anim.setEndValue(1.0)
            self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

            self._widget.show()
            self._fade_anim.start()

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
            from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
            from PyQt6.QtWidgets import QApplication

            if not self._widget:
                return

            # 淡出动画 - keep reference to prevent GC
            self._fade_anim = QPropertyAnimation(self._widget, b"windowOpacity")
            self._fade_anim.setDuration(500)
            self._fade_anim.setStartValue(1.0)
            self._fade_anim.setEndValue(0.0)
            self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._fade_anim.finished.connect(self._widget.close)
            self._fade_anim.start()

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

    VERSION = "1.0.0"
    APP_NAME = "PaleoAST"

    def __init__(self):
        """初始化"""
        self._logger = logging.getLogger("PaleoAST")
        self._app: Any | None = None
        self._splash: SplashScreen | None = None
        self._main_window: Any | None = None
        self._exception_handler: ExceptionHandler | None = None

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

        from PyQt6.QtGui import QIcon
        from PyQt6.QtWidgets import QApplication

        self._app = QApplication(sys.argv)
        self._app.setApplicationName(self.APP_NAME)
        self._app.setApplicationVersion(self.VERSION)
        self._app.setOrganizationName("PaleoAST")
        self._app.setQuitOnLastWindowClosed(True)

        # Set application icon
        logo_path = Path(__file__).parent / "logo.png"
        if logo_path.exists():
            self._app.setWindowIcon(QIcon(str(logo_path)))
            self._logger.info(f"Application icon loaded: {logo_path}")

    def _init_i18n(self) -> None:
        """初始化国际化系统"""
        from config.i18n import _reset_translator, get_translator, register_translations

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
                self._splash.update(60 + len(loaded) * 2, _("Loaded {0}...").format(description))
                self._logger.info(f"Module loaded: {module_name}")
            except ImportError as e:
                self._logger.warning(f"Module not loaded: {module_name} - {e}")

    def _create_main_window(self) -> None:
        """创建主窗口"""
        self._logger.info("Creating main window...")

        try:
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import (
                QLabel,
                QMainWindow,
                QStatusBar,
                QVBoxLayout,
                QWidget,
            )

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

                # 状态栏
                status = QStatusBar()
                status.showMessage("Ready")
                self._main_window.setStatusBar(status)

            self._logger.info("Main window created")

        except Exception as e:
            self._logger.error(f"Failed to create main window: {e}")
            raise

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
