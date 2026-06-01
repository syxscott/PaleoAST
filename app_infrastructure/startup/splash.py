"""
================================================================================
PaleoAST - Commercial Splash Screen
================================================================================

本模块实现商业级无边框启动闪屏，包含动画效果。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from PyQt6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class SplashScreenStyle(Enum):
    """闪屏样式"""

    DARK = "dark"
    LIGHT = "light"
    GRADIENT = "gradient"


class AnimatedLabel(QLabel):
    """动画标签基类"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._opacity = 1.0
        self._animation_group: QParallelAnimationGroup | None = None

    def setOpacity(self, opacity: float) -> None:
        """设置透明度"""
        self._opacity = max(0.0, min(1.0, opacity))
        self.setWindowOpacity(self._opacity)
        self.update()

    def getOpacity(self) -> float:
        """获取透明度"""
        return self._opacity

    opacity = pyqtProperty(float, getOpacity, setOpacity)


class CircularProgressIndicator(QWidget):
    """
    圆形进度指示器

    显示一个旋转的加载动画。
    """

    def __init__(self, parent: QWidget | None = None, size: int = 60):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._rotation = 0
        self._color = QColor("#3498DB")
        self._line_width = 4
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._is_running = False

    def start(self) -> None:
        """启动旋转"""
        if not self._is_running:
            self._is_running = True
            self._timer.start(30)

    def stop(self) -> None:
        """停止旋转"""
        self._is_running = False
        self._timer.stop()

    def _rotate(self) -> None:
        """旋转动画"""
        self._rotation = (self._rotation + 10) % 360
        self.update()

    def setColor(self, color: QColor) -> None:
        """设置颜色"""
        self._color = color
        self.update()

    def paintEvent(self, event) -> None:
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # 绘制背景圆
        painter.setBrush(QColor(40, 40, 40))
        painter.drawEllipse(self.rect())

        # 绘制进度弧
        pen = QPen(self._color, self._line_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        rect = self.rect().adjusted(8, 8, -8, -8)
        start_angle = self._rotation * 16
        span_angle = 90 * 16

        painter.drawArc(rect, start_angle, span_angle)


class SplashScreen(QWidget):
    """
    商业级启动闪屏

    特点:
    - 无边框窗口
    - 透明背景带渐变
    - 动画效果 (淡入淡出、旋转)
    - 实时进度条
    - 详细启动提示

    使用示例:
        >>> splash = SplashScreen()
        >>> splash.show()
        >>> splash.update_progress(50, "Loading modules...")
        >>> splash.finish()
    """

    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度, 消息
    finished = pyqtSignal()

    def __init__(self, style: SplashScreenStyle = SplashScreenStyle.GRADIENT, duration_ms: int = 3000):
        """
        初始化闪屏

        参数:
            style: 样式
            duration_ms: 显示最小时间
        """
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        self._style = style
        self._duration_ms = duration_ms
        self._start_time = time.time()
        self._is_closing = False

        # UI组件
        self._title_label: QLabel | None = None
        self._subtitle_label: QLabel | None = None
        self._progress_bar: QProgressBar | None = None
        self._status_label: QLabel | None = None
        self._version_label: QLabel | None = None
        self._progress_indicator: CircularProgressIndicator | None = None

        # 动画
        self._fade_animation: QPropertyAnimation | None = None
        self._title_animation: QPropertyAnimation | None = None

        # 日志
        self._logger = logging.getLogger(f"{__name__}.SplashScreen")

        # 初始化UI
        self._init_ui()
        self._init_animations()

        self._logger.info("Splash screen initialized")

    def _init_ui(self) -> None:
        """初始化UI"""
        # 设置窗口属性
        self.setFixedSize(600, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 居中显示
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 60, 40, 40)
        main_layout.setSpacing(15)

        # 设置字体
        title_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
        subtitle_font = QFont("Segoe UI", 12)
        status_font = QFont("Consolas", 9)
        version_font = QFont("Segoe UI", 9)

        # 标题
        self._title_label = QLabel("PaleoAST", self)
        self._title_label.setFont(title_font)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color: #FFFFFF; background: transparent;")

        # 副标题
        self._subtitle_label = QLabel("Paleontological Advanced Statistical Toolkit", self)
        self._subtitle_label.setFont(subtitle_font)
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle_label.setStyleSheet("color: #3498DB; background: transparent;")

        # 进度指示器
        self._progress_indicator = CircularProgressIndicator(self, size=50)
        self._progress_indicator.start()

        # 进度条
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(52, 152, 219, 0.2);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498DB,
                    stop: 1 #2ECC71
                );
                border-radius: 3px;
            }
        """)

        # 状态标签
        self._status_label = QLabel("Initializing...", self)
        self._status_label.setFont(status_font)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("""
            color: #95A5A6;
            background: transparent;
        """)
        self._status_label.setWordWrap(True)

        # 版本标签
        self._version_label = QLabel("Version 1.0.1", self)
        self._version_label.setFont(version_font)
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self._version_label.setStyleSheet("color: #7F8C8D; background: transparent;")

        # 添加到布局
        main_layout.addStretch(1)
        main_layout.addWidget(self._title_label)
        main_layout.addWidget(self._subtitle_label)
        main_layout.addSpacing(20)
        main_layout.addWidget(self._progress_indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._progress_bar)
        main_layout.addSpacing(10)
        main_layout.addWidget(self._status_label, stretch=1)
        main_layout.addWidget(self._version_label)

    def _init_animations(self) -> None:
        """初始化动画"""
        # 淡入动画
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self._fade_animation.setDuration(800)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def showEvent(self, event) -> None:
        """显示事件"""
        super().showEvent(event)
        # 开始淡入动画
        self._fade_animation.start()

    def paintEvent(self, event) -> None:
        """绘制背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._style == SplashScreenStyle.DARK:
            # 深色背景
            painter.fillRect(self.rect(), QColor(44, 62, 80))
        elif self._style == SplashScreenStyle.LIGHT:
            # 浅色背景
            painter.fillRect(self.rect(), QColor(236, 240, 241))
        else:
            # 渐变背景
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(44, 62, 80))
            gradient.setColorAt(0.5, QColor(52, 73, 94))
            gradient.setColorAt(1, QColor(44, 62, 80))
            painter.fillRect(self.rect(), gradient)

        # 绘制圆角矩形边框
        painter.setPen(QColor(52, 152, 219, 100))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        from PyQt6.QtCore import QRectF

        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        painter.drawRoundedRect(rect, 15, 15)

    def update_progress(self, value: int, message: str = "", details: str = "") -> None:
        """
        更新进度

        参数:
            value: 进度值 0-100
            message: 状态消息
            details: 详细信息
        """
        if self._is_closing:
            return

        # 更新进度条
        self._progress_bar.setValue(value)

        # 更新状态消息
        if message:
            full_message = message
            if details:
                full_message = f"{message}\n{details}"
            self._status_label.setText(full_message)

        # 发送信号
        self.progress_updated.emit(value, message)

        # 处理事件
        QApplication.processEvents()

    def append_log(self, message: str) -> None:
        """
        追加日志消息

        参数:
            message: 日志消息
        """
        current = self._status_label.text()
        self._status_label.setText(f"{current}\n{message}" if current else message)
        QApplication.processEvents()

    def set_phase(self, phase: str) -> None:
        """
        设置阶段标题

        参数:
            phase: 阶段名称
        """
        self._subtitle_label.setText(phase)

    def finish(self, main_window: QWidget = None) -> None:
        """
        完成闪屏

        参数:
            main_window: 主窗口 (用于关闭闪屏)
        """
        if self._is_closing:
            return

        self._is_closing = True
        self._progress_indicator.stop()

        # 确保最小显示时间
        elapsed = (time.time() - self._start_time) * 1000
        remaining = max(0, self._duration_ms - elapsed)

        if remaining > 0:
            QTimer.singleShot(int(remaining), lambda: self._do_close(main_window))
        else:
            self._do_close(main_window)

    def _do_close(self, main_window: QWidget = None) -> None:
        """执行关闭"""
        # 淡出动画
        fade_out = QPropertyAnimation(self, b"windowOpacity")
        fade_out.setDuration(500)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        fade_out.finished.connect(self._on_fade_out_finished)
        fade_out.start()

    def _on_fade_out_finished(self) -> None:
        """淡出完成"""
        self.close()
        self.finished.emit()
        self._logger.info("Splash screen closed")


class SplashScreenManager:
    """
    闪屏管理器

    管理多个启动阶段和进度。

    使用示例:
        >>> manager = SplashScreenManager(splash)
        >>> manager.add_phase("Loading modules...", 5)
        >>> manager.add_step("Import numpy")
        >>> manager.add_step("Import scipy")
        >>> manager.complete()
    """

    def __init__(self, splash: SplashScreen):
        """
        初始化管理器

        参数:
            splash: 闪屏实例
        """
        self._splash = splash
        self._phases: list[dict] = []
        self._current_phase: int = -1
        self._current_step: int = 0
        self._total_steps: int = 0
        self._logger = logging.getLogger(f"{__name__}.SplashManager")

    def add_phase(self, name: str, weight: int = 1, steps: list[str] | None = None) -> int:
        """
        添加阶段

        参数:
            name: 阶段名称
            weight: 权重 (占总进度的比例)
            steps: 该阶段的步骤列表

        返回:
            阶段索引
        """
        phase_idx = len(self._phases)
        self._phases.append(
            {"name": name, "weight": weight, "steps": steps or [], "start_progress": 0, "current_step": 0}
        )
        self._total_steps += len(steps) if steps else 0

        self._logger.debug(f"Added phase: {name} (weight: {weight})")
        return phase_idx

    def start_phase(self, phase_idx: int) -> None:
        """
        开始阶段

        参数:
            phase_idx: 阶段索引
        """
        if phase_idx >= len(self._phases):
            self._logger.warning(f"Invalid phase index: {phase_idx}")
            return

        self._current_phase = phase_idx
        phase = self._phases[phase_idx]

        # 计算起始进度
        total_weight = sum(p["weight"] for p in self._phases)
        completed_weight = sum(self._phases[i]["weight"] for i in range(phase_idx))

        phase_start = int(completed_weight / total_weight * 100)
        phase["start_progress"] = phase_start

        # 更新闪屏
        self._splash.set_phase(phase["name"])
        self._splash.update_progress(phase_start, f"Starting: {phase['name']}")

        self._logger.debug(f"Started phase: {phase['name']}")

    def complete_step(self, message: str = "") -> None:
        """
        完成当前步骤

        参数:
            message: 步骤消息
        """
        if self._current_phase < 0:
            return

        phase = self._phases[self._current_phase]
        n_steps = len(phase["steps"])

        if n_steps == 0:
            return

        phase["current_step"] += 1
        self._current_step += 1

        # 计算进度
        total_weight = sum(p["weight"] for p in self._phases)
        current_weight = sum(
            self._phases[i]["weight"] * (self._phases[i]["current_step"] / max(1, len(self._phases[i]["steps"])))
            for i in range(self._current_phase + 1)
        )

        progress = int(current_weight / total_weight * 100)

        # 更新闪屏
        if message:
            self._splash.update_progress(progress, message)
        else:
            step_msg = phase["steps"][phase["current_step"] - 1] if phase["current_step"] <= n_steps else ""
            self._splash.update_progress(progress, step_msg)

    def complete(self) -> None:
        """完成所有阶段"""
        self._splash.update_progress(100, "Ready!")
        self._logger.info("All phases completed")


def create_professional_splash(style: SplashScreenStyle = SplashScreenStyle.GRADIENT) -> SplashScreen:
    """
    创建专业闪屏

    参数:
        style: 样式

    返回:
        配置好的SplashScreen
    """
    splash = SplashScreen(style=style, duration_ms=2000)
    splash.set_phase("Professional Paleontology Analysis Platform")
    return splash


if __name__ == "__main__":
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    splash = create_professional_splash()
    splash.show()

    # 模拟加载过程
    for i in range(0, 101, 10):
        splash.update_progress(i, f"Loading... {i}%")
        time.sleep(0.2)

    time.sleep(1)
    splash.finish()

    sys.exit(app.exec())
