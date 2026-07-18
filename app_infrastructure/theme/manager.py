"""
================================================================================
PaleoAST - Theme Manager
================================================================================

主题管理器，支持动态切换深色/浅色模式。

作者: PaleoAST Development Team
"""

import logging
import weakref
from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from .styles import PaleoASTStyles, StyleMode

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
        # Use a list of weak references so callbacks don't keep
        # their owners alive indefinitely. The previous plain list
        # was never cleaned up by ``unregister_change_callback`` for
        # transient subscribers (e.g. dialogs), causing unbounded
        # growth over the application's lifetime.
        self._change_callbacks: list[weakref.ref] = []

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

        # Iterate over a snapshot so callbacks that unregister
        # themselves during dispatch don't trip the iteration.
        dead: list[weakref.ref] = []
        for cb_ref in list(self._change_callbacks):
            callback = cb_ref()
            if callback is None:
                dead.append(cb_ref)
                continue
            try:
                callback(theme)
            except Exception as e:
                logger.error(f"Error in theme change callback: {e}")
        # Reap dead refs
        for d in dead:
            try:
                self._change_callbacks.remove(d)
            except ValueError:
                pass

        logger.info(f"Theme changed to {theme.value}")

    def toggle_theme(self) -> None:
        """切换主题"""
        new_theme = StyleMode.LIGHT if self._current_theme == StyleMode.DARK else StyleMode.DARK
        self.set_theme(new_theme)

    def register_change_callback(
        self,
        callback: Callable[[StyleMode], None],
        *,
        owner: object | None = None,
    ) -> None:
        """
        注册主题改变回调

        The callback is held via a weak reference so that simply
        forgetting to call ``unregister_change_callback`` does not
        leak memory. For bound methods, pass the bound object as
        ``owner`` so the weak reference can resolve it; otherwise
        the callback is held by strong reference (legacy behaviour).

        参数:
            callback: 回调函数，签名为 callback(theme: StyleMode)
            owner: Optional owning object. When provided, the
                callback is dropped automatically once ``owner`` is
                garbage-collected.
        """
        if owner is not None:
            try:
                ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") else weakref.ref(callback)
            except TypeError:
                ref = weakref.ref(callback)
        else:
            # No owner: hold a strong reference so the callback
            # remains callable. Callers that omit ``owner`` must
            # pair every register with an unregister, or accept
            # the documented strong-reference semantics.
            self._change_callbacks.append(_StrongRef(callback))
            return
        self._change_callbacks.append(ref)

    def unregister_change_callback(self, callback: Callable[[StyleMode], None]) -> None:
        """
        注销主题改变回调

        参数:
            callback: 回调函数
        """
        for ref in list(self._change_callbacks):
            target = ref() if not isinstance(ref, _StrongRef) else ref.callback
            if target is callback:
                try:
                    self._change_callbacks.remove(ref)
                except ValueError:
                    pass
                return

    def apply_theme(self) -> None:
        """应用当前主题到应用"""
        stylesheet = self._styles_engine.get_complete_stylesheet()
        self._app.setStyleSheet(stylesheet)
        logger.debug(f"Applied {self._current_theme.value} theme")

    def get_stylesheet(self, theme: StyleMode | None = None) -> str:
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


class _StrongRef:
    """Strong-reference wrapper so legacy ``register_change_callback``
    callers that omit ``owner`` retain their callback lifetime."""

    __slots__ = ("callback",)

    def __init__(self, callback: Callable[[StyleMode], None]) -> None:
        self.callback = callback

    def __call__(self) -> Callable[[StyleMode], None] | None:
        return self.callback
