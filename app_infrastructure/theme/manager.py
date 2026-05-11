"""
================================================================================
PaleoAST Phase 5 - Theme Manager
================================================================================

主题管理器，支持动态切换深色/浅色模式。

作者: PaleoAST Development Team
"""

import logging
from typing import Optional, Callable
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from .styles import PaleoASTStyles, StyleMode, get_dark_stylesheet, get_light_stylesheet

logger = logging.getLogger(__name__)


class ThemeManager(QObject):
    """
    主题管理器
    
    管理应用的主题切换，支持动态切换深色/浅色模式。
    
    使用示例:
        >>> app = QApplication([])
        >>> theme_manager = ThemeManager(app)
        >>> theme_manager.set_theme(ThemeManager.DARK)
        >>> theme_manager.toggle_theme()
    """
    
    # 主题常量
    DARK = StyleMode.DARK
    LIGHT = StyleMode.LIGHT
    
    # 信号
    theme_changed = pyqtSignal(str)  # 主题改变的信号
    
    def __init__(self, app: QApplication):
        """
        初始化主题管理器
        
        参数:
            app: QApplication实例
        """
        super().__init__()
        
        self._app = app
        self._current_theme = StyleMode.DARK
        self._styles_engine = PaleoASTStyles(self._current_theme)
        self._change_callbacks: list = []
        
        logger.info("ThemeManager initialized with dark theme")
    
    @property
    def current_theme(self) -> StyleMode:
        """获取当前主题"""
        return self._current_theme
    
    @property
    def is_dark(self) -> bool:
        """是否为深色模式"""
        return self._current_theme == StyleMode.DARK
    
    def set_theme(self, theme: StyleMode) -> None:
        """
        设置主题
        
        参数:
            theme: 主题模式
        """
        if theme not in (StyleMode.DARK, StyleMode.LIGHT):
            logger.warning(f"Invalid theme: {theme}")
            return
        
        if self._current_theme == theme:
            logger.debug(f"Theme already set to {theme.value}")
            return
        
        self._current_theme = theme
        self._styles_engine.set_mode(theme)
        
        # 应用样式
        stylesheet = self._styles_engine.get_complete_stylesheet()
        self._app.setStyleSheet(stylesheet)
        
        # 通知回调
        self.theme_changed.emit(theme.value)
        
        for callback in self._change_callbacks:
            try:
                callback(theme)
            except Exception as e:
                logger.error(f"Error in theme change callback: {e}")
        
        logger.info(f"Theme changed to {theme.value}")
    
    def toggle_theme(self) -> None:
        """切换主题"""
        new_theme = StyleMode.LIGHT if self._current_theme == StyleMode.DARK else StyleMode.DARK
        self.set_theme(new_theme)
    
    def register_change_callback(self, callback: Callable[[StyleMode], None]) -> None:
        """
        注册主题改变回调
        
        参数:
            callback: 回调函数，签名为 callback(theme: StyleMode)
        """
        self._change_callbacks.append(callback)
    
    def unregister_change_callback(self, callback: Callable[[StyleMode], None]) -> None:
        """
        注销主题改变回调
        
        参数:
            callback: 回调函数
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def apply_theme(self) -> None:
        """应用当前主题到应用"""
        stylesheet = self._styles_engine.get_complete_stylesheet()
        self._app.setStyleSheet(stylesheet)
        logger.debug(f"Applied {self._current_theme.value} theme")
    
    def get_stylesheet(self, theme: Optional[StyleMode] = None) -> str:
        """
        获取指定主题的样式表
        
        参数:
            theme: 主题模式，None表示当前主题
        
        返回:
            QSS字符串
        """
        if theme is None:
            theme = self._current_theme
        
        styles = PaleoASTStyles(theme)
        return styles.get_complete_stylesheet()
