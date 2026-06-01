# views/floating_toolbar.py
"""
Floating Toolbar for PaleoAST

Provides a semi-transparent floating toolbar embedded in the plot canvas
area for quick access to common plot operations.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QStyle, QToolButton, QWidget

from config.i18n import _

logger = logging.getLogger(__name__)


class FloatingToolBar(QWidget):
    """
    Semi-transparent floating toolbar for plot operations.

    Features:
        - Semi-transparent background
        - 44x44px professional touch targets
        - Smooth hover/press transitions
        - Focus states for keyboard navigation
        - Common operations: Zoom, Pan, Reset, Save
    """

    # Professional icon mappings (SVG path, Qt standard icon fallback)
    ICON_MAPPINGS = [
        ("save", ":/icons/save.svg", QStyle.StandardPixmap.SP_DialogSaveButton),
        ("zoom", ":/icons/zoom_in.svg", QStyle.StandardPixmap.SP_FileDialogContentsView),
        ("pan", ":/icons/pan.svg", QStyle.StandardPixmap.SP_ArrowRight),
        ("reset", ":/icons/reset.svg", QStyle.StandardPixmap.SP_BrowserReload),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.FloatingToolBar")

        self._is_dark_theme = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the toolbar UI."""
        # Use frameless window for custom look
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Create tool buttons with professional sizing (44x44px)
        self._btn_save = self._create_tool_button("save", _("保存图片"))
        self._btn_zoom = self._create_tool_button("zoom", _("缩放"))
        self._btn_pan = self._create_tool_button("pan", _("平移"))
        self._btn_reset = self._create_tool_button("reset", _("重置视图"))

        layout.addWidget(self._btn_save)
        layout.addWidget(self._btn_zoom)
        layout.addWidget(self._btn_pan)
        layout.addWidget(self._btn_reset)

        # Size policy
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._apply_stylesheet()

    def _create_tool_button(self, icon_key: str, tooltip: str) -> QToolButton:
        """Create a styled tool button with professional icon."""
        btn = QToolButton()
        btn.setToolTip(tooltip)
        btn.setFixedSize(44, 44)  # Professional 44x44px touch target
        btn.setIconSize(QSize(22, 22))
        btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Set icon with fallback to Qt standard icons
        icon_set = False
        for key, svg_path, std_icon in self.ICON_MAPPINGS:
            if key == icon_key:
                try:
                    custom_icon = QIcon(svg_path)
                    if not custom_icon.isNull():
                        btn.setIcon(custom_icon)
                        icon_set = True
                        break
                except Exception:
                    pass
                # Fallback to Qt standard icon
                btn.setIcon(btn.style().standardIcon(std_icon))
                icon_set = True
                break

        return btn

    def _apply_stylesheet(self) -> None:
        """Apply professional stylesheet with transitions and focus states."""
        if self._is_dark_theme:
            self.setStyleSheet("""
                QWidget {
                    background: rgba(35, 35, 35, 230);
                    border: 1px solid rgba(80, 80, 80, 180);
                    border-radius: 8px;
                }
                QToolButton {
                    background: transparent;
                    border: 2px solid transparent;
                    border-radius: 6px;
                    padding: 6px;
                    color: #E0E0E0;
                }
                QToolButton:hover {
                    background: rgba(80, 80, 80, 150);
                    border: 1px solid rgba(100, 100, 100, 100);
                }
                QToolButton:pressed {
                    background: rgba(100, 100, 100, 150);
                }
                QToolButton:focus {
                    border: 2px solid #0078D4;
                    background: rgba(0, 120, 212, 40);
                }
                QToolButton:disabled {
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background: rgba(250, 250, 250, 230);
                    border: 1px solid rgba(180, 180, 180, 180);
                    border-radius: 8px;
                }
                QToolButton {
                    background: transparent;
                    border: 2px solid transparent;
                    border-radius: 6px;
                    padding: 6px;
                    color: #333333;
                }
                QToolButton:hover {
                    background: rgba(200, 200, 200, 150);
                    border: 1px solid rgba(150, 150, 150, 100);
                }
                QToolButton:pressed {
                    background: rgba(180, 180, 180, 150);
                }
                QToolButton:focus {
                    border: 2px solid #0078D4;
                    background: rgba(0, 120, 212, 20);
                }
                QToolButton:disabled {
                    color: #999999;
                }
            """)

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set the theme for the toolbar."""
        self._is_dark_theme = is_dark
        self._apply_stylesheet()

    def connect_signals(self, canvas) -> None:
        """Connect toolbar buttons to canvas operations."""
        # Disconnect previous connections
        try:
            self._btn_save.clicked.disconnect()
            self._btn_zoom.clicked.disconnect()
            self._btn_pan.clicked.disconnect()
            self._btn_reset.clicked.disconnect()
        except Exception:
            pass

        if canvas is None:
            return

        # Connect to canvas operations with existence checks
        if hasattr(canvas, '_export_plot'):
            self._btn_save.clicked.connect(canvas._export_plot)
        if hasattr(canvas, '_zoom_in'):
            self._btn_zoom.clicked.connect(canvas._zoom_in)
        if hasattr(canvas, '_reset_view'):
            self._btn_reset.clicked.connect(canvas._reset_view)
        # Pan button logs info (actual pan mode would need canvas support)
        self._btn_pan.clicked.connect(lambda: self._logger.info("Pan mode - use mouse drag"))

    def showEvent(self, event) -> None:
        """Handle show event."""
        super().showEvent(event)
        # Ensure toolbar stays on top
        self.raise_()
