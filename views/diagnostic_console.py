# views/diagnostic_console.py
"""
Diagnostic Console for PaleoAST

Provides a real-time logging console widget that displays
computation status and logs from the application.

Author: PaleoAST Development Team
version: 1.0.1
"""

import logging
from datetime import datetime

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QDockWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.design_system import get_palette
from config.i18n import _


class ConsoleTextEdit(QTextEdit):
    """
    Custom text edit widget for console output.

    Features:
        - Auto-scroll to bottom on new output
        - Different colors for log levels
        - Theme-aware colors
        - Copy support
        - Clear support
    """

    # Theme-aware colors (will be updated based on theme)
    DARK_COLORS = {
        "DEBUG": "#888888",  # Gray
        "INFO": "#E0E0E0",  # Light gray (visible on dark)
        "WARNING": "#FFA500",  # Orange
        "ERROR": "#FF5252",  # Light red
        "CRITICAL": "#FF1744",  # Bright red
    }

    LIGHT_COLORS = {
        "DEBUG": "#888888",  # Gray
        "INFO": "#000000",  # Black
        "WARNING": "#FFA500",  # Orange
        "ERROR": "#FF0000",  # Red
        "CRITICAL": "#8B0000",  # Dark Red
    }

    MAX_LINES = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self._line_count = 0
        self._is_dark_theme = False

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme for colors."""
        self._is_dark_theme = is_dark

    def append_log(self, level: str, message: str, timestamp: bool = True) -> None:
        """Append a log message with color coding."""
        colors = self.DARK_COLORS if self._is_dark_theme else self.LIGHT_COLORS
        color = colors.get(level, "#000000")

        text = ""
        if timestamp:
            text += f"<span style='color: #888888;'>[{datetime.now().strftime('%H:%M:%S')}]</span> "

        text += f"<span style='color: {color};'>{message}</span>"

        self.append(text)
        self._line_count += 1

        # Truncate if exceeds max lines
        if self._line_count > self.MAX_LINES:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
            self.setTextCursor(cursor)
            self._line_count -= 1
        # NOTE: ``QTextCursor.SelectionType.LineUnderCursor`` is the
        # canonical enum value in PyQt6. Older PyQt5 code paths used
        # ``QTextCursor.LineUnderCursor`` (deprecated) which still
        # works on some platforms but fails on others — keeping the
        # enum-qualified form above is intentional.

        # Auto-scroll to bottom
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.ensureCursorVisible()


class DiagnosticConsole(QDockWidget):
    """
    Dockable diagnostic console widget.

    Features:
        - Real-time log display
        - Clear button
        - Pause/Resume functionality
        - Auto-hide on startup option
        - Theme support (dark/light)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_("诊断控制台"), parent)
        self._logger = logging.getLogger(f"{__name__}.DiagnosticConsole")

        self._is_paused = False
        self._is_dark_theme = False
        self._message_buffer: list[tuple] = []

        self._setup_ui()
        self._setup_logging()

    def _setup_ui(self) -> None:
        """Setup the console UI."""
        # Console widget
        console_widget = QWidget()
        layout = QVBoxLayout(console_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Text display
        self._console = ConsoleTextEdit()
        layout.addWidget(self._console)

        # Button bar
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)

        self._btn_clear = QPushButton(_("清空"))
        self._btn_clear.clicked.connect(self.clear)
        button_layout.addWidget(self._btn_clear)

        self._btn_pause = QPushButton(_("暂停"))
        self._btn_pause.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self._btn_pause)

        # Create a horizontal layout for buttons
        from PyQt6.QtWidgets import QHBoxLayout

        btn_bar = QHBoxLayout()
        btn_bar.addWidget(self._btn_clear)
        btn_bar.addWidget(self._btn_pause)
        btn_bar.addStretch()

        layout.addLayout(btn_bar)

        self.setWidget(console_widget)

        # Set initial visibility
        self.setVisible(False)

        # Apply theme
        self._apply_stylesheet()

    def _apply_stylesheet(self) -> None:
        """Apply themed stylesheet with professional transitions and focus states."""
        c = get_palette(self._is_dark_theme)

        if self._is_dark_theme:
            self.setStyleSheet(f"""
                QDockWidget {{
                    titleBarCloseButtonVisible: true;
                }}
                QTextEdit {{
                    background: {c.bg_primary};
                    color: {c.text_primary};
                    border: 1px solid {c.border_light};
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 12px;
                }}
                QPushButton {{
                    background: {c.bg_tertiary};
                    color: {c.text_primary};
                    border: 1px solid {c.border_light};
                    padding: 6px 14px;
                    border-radius: 4px;
                    min-width: 60px;
                }}
                QPushButton:hover {{
                    background: {c.bg_hover};
                    border-color: {c.primary};
                }}
                QPushButton:pressed {{
                    background: {c.primary};
                    color: white;
                }}
                QPushButton:focus {{
                    border: 2px solid {c.primary};
                    outline: none;
                }}
                QPushButton:disabled {{
                    color: #666666;
                    background: {c.bg_tertiary};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QDockWidget {{
                    titleBarCloseButtonVisible: true;
                }}
                QTextEdit {{
                    background: {c.bg_primary};
                    color: {c.text_primary};
                    border: 1px solid {c.border_light};
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 12px;
                }}
                QPushButton {{
                    background: {c.bg_tertiary};
                    color: {c.text_primary};
                    border: 1px solid {c.border_light};
                    padding: 6px 14px;
                    border-radius: 4px;
                    min-width: 60px;
                }}
                QPushButton:hover {{
                    background: {c.bg_hover};
                    border-color: {c.primary};
                }}
                QPushButton:pressed {{
                    background: {c.primary};
                    color: white;
                }}
                QPushButton:focus {{
                    border: 2px solid {c.primary};
                    outline: none;
                }}
                QPushButton:disabled {{
                    color: #999999;
                    background: {c.bg_tertiary};
                }}
            """)

    def setDarkTheme(self, is_dark: bool) -> None:
        """Set dark/light theme."""
        self._is_dark_theme = is_dark
        self._console.setDarkTheme(is_dark)
        self._apply_stylesheet()

    def _setup_logging(self) -> None:
        """Setup logging handler."""
        self._handler = ConsoleLogHandler(self)
        self._handler.setLevel(logging.INFO)

        # Add handler to root logger
        logging.getLogger().addHandler(self._handler)

    def append_message(self, level: str, message: str) -> None:
        """Append a log message to the console."""
        if self._is_paused:
            self._message_buffer.append((level, message))
            return

        self._console.append_log(level, message)

    def clear(self) -> None:
        """Clear the console."""
        self._console.clear()
        self._console._line_count = 0
        self._message_buffer.clear()

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self._is_paused = not self._is_paused

        if self._is_paused:
            self._btn_pause.setText(_("继续"))
            self._logger.debug("Console paused")
        else:
            self._btn_pause.setText(_("暂停"))
            self._logger.debug("Console resumed")
            # Flush buffer
            for level, msg in self._message_buffer:
                self._console.append_log(level, msg)
            self._message_buffer.clear()

    def showEvent(self, event) -> None:
        """Handle show event."""
        super().showEvent(event)
        # Set focus to console
        self._console.setFocus()

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.append_message("INFO", message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.append_message("WARNING", message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.append_message("ERROR", message)


class ConsoleLogHandler(QObject, logging.Handler):
    """
    Custom logging handler that sends logs to the diagnostic console.

    This handler is thread-safe: worker threads emit a Qt signal
    rather than touching the QTextEdit directly. Qt delivers the
    signal on the GUI thread (the queue connection is the default
    for cross-thread emits), so ``append_message`` runs only on the
    thread that owns ``DiagnosticConsole``. The previous
    implementation used ``QMutex`` to serialise concurrent calls
    into the QTextEdit, but ``QTextEdit`` is not reentrant — a
    mutex does not make it safe to call ``append`` from a worker
    thread, it only makes it *seem* safe until a race window is
    hit.
    """

    # Signal emitted from worker threads; the slot lives on the GUI
    # thread, so QTextEdit mutations happen there.
    _message_signal = pyqtSignal(str, str)

    def __init__(self, console: DiagnosticConsole) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self._console = console
        # Connect the signal with the default (auto) connection.
        # ``_console`` lives on the GUI thread, and because this
        # QObject also lives there, Qt picks ``Qt.DirectConnection``
        # automatically — which is correct for same-thread signal
        # delivery. When the signal is emitted from a worker
        # thread, Qt posts the slot to the receiver's thread.
        self._message_signal.connect(self._console.append_message)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the console (worker thread safe)."""
        try:
            msg = self.format(record)
            level = record.levelname
            # Cross-thread safe: Qt queues the slot invocation on the
            # GUI thread when called from a worker.
            self._message_signal.emit(level, msg)
        except Exception:
            self.handleError(record)


class StatusBarLogHandler(logging.Handler):
    """
    Lightweight logging handler that updates the status bar.

    This is used for showing brief operation status in the status bar
    without flooding the diagnostic console.
    """

    def __init__(self, status_bar) -> None:
        super().__init__()
        self._status_bar = status_bar

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the status bar."""
        try:
            # Guard against the status bar having been destroyed but this
            # handler still being referenced by the logging framework.
            if self._status_bar is None:
                return
            msg = self.format(record)
            # Only show info-level messages
            if record.levelno == logging.INFO:
                self._status_bar.setInfo(msg)
        except Exception:
            # Never let a logging handler raise; that would break logging globally.
            pass
