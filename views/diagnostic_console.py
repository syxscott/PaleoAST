# views/diagnostic_console.py
"""
Diagnostic Console for PaleoAST

Provides a real-time logging console widget that displays
computation status and logs from the application.

Author: PaleoAST Development Team
version: 1.1.0
"""

import logging
import sys
import weakref
from datetime import datetime

from PyQt6 import sip
from PyQt6.QtCore import QObject, pyqtSignal
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

    def append(self, text: str) -> None:
        """Append, but never let a dead QTextEdit take the process with it.

        The handler's signal lands on ``DiagnosticConsole.append_message``,
        which forwards here. The dock can outlive this text widget, and during
        interpreter teardown Qt destroys children first -- so the RECEIVER
        has to tolerate being dead, not just the sender.

        A Qt slot raising is not a normal Python exception path: PyQt6 can
        abort the process (SIGABRT, exit 134) instead of propagating, which is
        how a CI job died with ``Fatal Python error: Aborted`` inside
        logging's own handler chain. Guarding the sender alone left this
        reachable; the abort landed on ``super().append()`` here.
        """
        try:
            if sip.isdeleted(self):
                return
            super().append(text)
        except RuntimeError:
            # C++ half is gone; nothing to render into.
            self._line_count = 0

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

    def closeEvent(self, event) -> None:
        """Detach the logging handler before the console is destroyed.

        The handler is a QObject attached to the ROOT logger, so it outlives
        this widget unless it is removed explicitly. At interpreter shutdown
        ``logging.shutdown()`` walks the root logger's handlers and calls
        ``getattr(h, "flushOnClose", True)`` on each; by then Qt has already
        destroyed the C++ side, and that attribute lookup raises
        ``RuntimeError: wrapped C/C++ object ... has been deleted``.
        ``logging.shutdown`` only swallows ``OSError``/``ValueError``, so the
        RuntimeError escaped and printed a traceback on every clean exit.

        Removing the handler here keeps a live reference out of the root
        logger's list, so shutdown never touches it.
        """
        handler = getattr(self, "_handler", None)
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - Qt may already be gone
                pass
            self._handler = None
        super().closeEvent(event)

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

    # Do not flush this handler during ``logging.shutdown()``. That call ends
    # up doing ``getattr(h, "flushOnClose", True)`` on every handler in the
    # module-global ``logging._handlerList`` -- a weakref list that
    # ``Handler.close()`` does NOT remove from, so this entry is still visited
    # at atexit. By then Qt has destroyed the C++ side of this QObject, and the
    # instance-dict lookup routes through sip and raises
    # ``RuntimeError: wrapped C/C++ object ... has been deleted``.
    # ``logging.shutdown`` only guards OSError/ValueError, so the traceback
    # printed on every otherwise-clean exit.
    #
    # Declaring this as a CLASS attribute fixes it at the root: ``getattr``
    # resolves it on the type and never touches the dead instance. Nothing is
    # buffered here anyway -- ``emit`` hands each record straight to the Qt
    # signal -- so there is nothing to flush.
    flushOnClose = False

    # Signal emitted from worker threads; the slot lives on the GUI
    # thread, so QTextEdit mutations happen there.
    _message_signal = pyqtSignal(str, str)

    def __init__(self, console: DiagnosticConsole) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self)
        # WEAK reference, deliberately. A strong ref here keeps the Python
        # wrapper alive after Qt has destroyed the C++ half, so every "is it
        # still alive?" check sees a live object and lets the emit through --
        # which is precisely the crash this class of guard keeps missing.
        # With a weakref the dead-receiver state becomes unrepresentable:
        # once the console is gone, the ref returns None and emit() no-ops.
        self._console_ref = weakref.ref(console)
        # Set once the underlying QObject is gone; see flush()/close().
        self._closed = False
        # Connect the signal with the default (auto) connection.
        # The console lives on the GUI thread, and because this QObject also
        # lives there, Qt picks ``Qt.DirectConnection`` automatically -- which
        # is correct for same-thread signal delivery. When the signal is
        # emitted from a worker thread, Qt posts the slot to the receiver's
        # thread.
        self._message_signal.connect(console.append_message)

        # Qt does NOT deliver closeEvent when it destroys an object during
        # teardown, so the dock's own closeEvent cannot be relied on to detach
        # this handler. destroyed() is the earliest reliable notice.
        try:
            console.destroyed.connect(self._on_console_destroyed)
        except (RuntimeError, TypeError):  # pragma: no cover - already gone
            self._closed = True

    @property
    def _console(self):
        """The console if it is still alive, else None."""
        try:
            return self._console_ref()
        except TypeError:  # pragma: no cover - ref cleared
            return None

    def _on_console_destroyed(self, *_args) -> None:
        """The console's C++ object is gone: stop emitting and detach."""
        self.close()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the console (worker thread safe).

        Two different death modes have to be covered, and neither is a Python
        exception -- PyQt6 aborts the PROCESS from C++ when a signal reaches a
        receiver whose QObject is gone, so nothing here can be caught after the
        fact. This crashed a CI job twice:

        1. Mid-session: the console was destroyed (dock closed) while the
           handler was still on the ROOT logger, so the next unrelated log
           record emitted into the void.
        2. Interpreter teardown, which is what still aborted after case 1 was
           fixed. ``sip.isdeleted()`` is the wrong tool here: during
           finalisation the Python wrapper is still alive and the C++ half is
           already going away, so it reports False and the guard lets the
           emit through. ``sys.is_finalizing()`` is the signal that actually
           matches, so that is checked first.
        """
        if getattr(self, "_closed", False):
            return
        # Nothing is worth logging to once the interpreter is shutting down,
        # and touching Qt from an atexit/exit path is exactly what aborts.
        if sys.is_finalizing():
            self._closed = True
            return
        console = self._console
        if console is None:
            # The console is gone, so the "emit into the void" state is
            # unrepresentable rather than merely detectable. Detach so later
            # records cost nothing.
            self.close()
            return
        try:
            # Checked BEFORE emitting because the emit is the uncatchable part:
            # PyQt6 aborts the process instead of raising when a signal
            # reaches a receiver whose C++ object is gone.
            if sip.isdeleted(console):
                self.close()
                return
        except (RuntimeError, TypeError):
            self.close()
            return

        try:
            msg = self.format(record)
            level = record.levelname
            # Cross-thread safe: Qt queues the slot invocation on the
            # GUI thread when called from a worker.
            self._message_signal.emit(level, msg)
        except RuntimeError:
            # The C++ half of this QObject is gone (QApplication already
            # destroyed, or the console was closed). Nothing to log to, and
            # handleError() would re-enter the same dead object.
            self._closed = True
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Flush defensively — the Qt object may already be destroyed.

        This handler is attached to the ROOT logger, so ``logging.shutdown()``
        calls ``flush()`` during interpreter teardown, which happens *after*
        Qt has torn down the C++ objects. Touching the signal then raises
        ``RuntimeError: wrapped C/C++ object ... has been deleted`` out of
        logging's own shutdown path, printing a traceback on every clean exit.
        """
        if getattr(self, "_closed", False):
            return
        try:
            super().flush()
        except RuntimeError:
            self._closed = True

    def close(self) -> None:
        """Detach from the logger and mark the Qt side dead.

        Unregistering here means a later ``logging.shutdown()`` skips this
        handler entirely, and ``_closed`` short-circuits the flush that would
        otherwise touch the deleted QObject.
        """
        self._closed = True
        try:
            logging.getLogger().removeHandler(self)
        except Exception:  # pragma: no cover - interpreter already tearing down
            pass
        try:
            logging.Handler.close(self)
        except Exception:  # pragma: no cover
            pass


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
