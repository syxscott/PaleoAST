# =============================================================================
# FILE: views/script_console.py
# =============================================================================
"""
The script console window.

WHY THIS EXISTS
~~~~~~~~~~~~~~~
PAST3's single largest advantage over a menu-driven tool is
its scripting layer. With fifty samples and twenty variables, no amount
of clicking gets through the grid, and batch work -- a distance matrix
per pair of sections, a growth curve per taxon, a permutation test per
comparison -- is the normal case rather than the exception.

There is no separate language here. The console runs Python against
:mod:`utils.script_session`, which supplies the part a bare REPL cannot
know:

    >>> analyses()                      # every runnable analysis, by name
    >>> run("mantel", n_permutations=999)
    >>> data                            # the current spreadsheet

``data`` is fetched on each access, so a script run after the user
edits a cell sees the edit. ``run()`` formats through each analysis's
own ``summary()``, which is the text the menus already show.

EXECUTION IS SYNCHRONOUS
~~~~~~~~~~~~~~~~~~~~~~~~
A long script freezes the window, exactly as a long analysis from a
menu does. That is stated in the dialog rather than worked around with
a worker thread: a thread would have to marshal every result back
across the GUI boundary, and for a console whose whole job is to hand
back arbitrary objects, that marshal is where the bugs would be. A
half-second script is the expected case; a half-hour one is a script
that belongs in a file.

Author: PaleoAST Development Team
version: 1.1.0
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config.design_system import get_palette
from config.i18n import _
from utils.script_session import ScriptSession

logger = logging.getLogger(__name__)


class ScriptConsoleDialog(QDialog):
    """A Python console wired to the running application.

    Parameters
    ----------
    parent:
        Parent widget.
    controller:
        The StatisticsController, exposed to scripts as ``controller``.
    data_provider:
        Zero-argument callable returning the current data, exposed as
        ``data``.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        controller: Any = None,
        data_provider: Any = None,
    ) -> None:
        """Build the console."""
        super().__init__(parent)
        self._logger = logging.getLogger(f"{__name__}.ScriptConsoleDialog")
        self.setWindowTitle(_("Script Console"))
        self.resize(900, 640)
        self.setObjectName("ScriptConsoleDialog")

        self._session = ScriptSession(data_provider=data_provider, controller=controller)
        self._build_ui()
        self._apply_stylesheet()
        self._write_banner()

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out the editor, the output pane and the buttons."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._hint = QLabel(_('Python. Try analyses() to list what can be run, or run("mantel", n_permutations=999).'))
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self._input = QPlainTextEdit()
        self._input.setPlaceholderText(_("Type Python here. The value of a final expression is echoed."))
        self._input.setTabChangesFocus(True)
        self._input.setFont(QFont("Consolas", 10))
        splitter.addWidget(self._input)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 10))
        splitter.addWidget(self._output)

        splitter.setSizes([260, 320])
        layout.addWidget(splitter, stretch=1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self._reset_button = QPushButton(_("Reset namespace"))
        self._reset_button.setToolTip(_("Forget every variable defined in this console"))
        self._reset_button.clicked.connect(self._on_reset)
        buttons.addWidget(self._reset_button)

        self._clear_button = QPushButton(_("Clear"))
        self._clear_button.clicked.connect(self._on_clear)
        buttons.addWidget(self._clear_button)

        self._run_button = QPushButton(_("Run (Ctrl+Enter)"))
        self._run_button.setDefault(True)
        self._run_button.clicked.connect(self._on_run)
        buttons.addWidget(self._run_button)

        layout.addLayout(buttons)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._on_run)

    def _apply_stylesheet(self) -> None:
        """Theme the panes from the shared palette.

        Deliberately minimal. The application has a full design system
        for the analysis dialogs; a console needs a monospace surface,
        not a redesign, and hard-coding more than that here would make
        the two diverge the first time the palette changes.
        """
        c = get_palette()
        self._output.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background-color: {c.bg_secondary};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 4px;
            }}
            """
        )
        self._input.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background-color: {c.bg_primary};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 4px;
            }}
            """
        )

    def _write_banner(self) -> None:
        """Show what the console can do and how many analyses it holds."""
        names = self._session.analyses()
        categories = self._session.categories()
        lines = [
            _("PaleoAST script console."),
            _("{0} analyses available in {1} categories.").format(len(names), len(categories)),
            _("Names stay defined between runs. Ctrl+Enter runs the block."),
            "",
        ]
        for category in categories:
            subset = self._session.analyses(category=category)
            lines.append(f"{category}: {', '.join(subset)}")
        self._append("\n".join(lines))

    # -- running ----------------------------------------------------------

    def _append(self, text: str) -> None:
        """Add text to the output pane and scroll to the end."""
        self._output.appendPlainText(text)
        bar = self._output.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_run(self) -> None:
        """Execute the editor contents.

        The block is cleared afterwards so the next one starts fresh,
        which is the usual REPL habit and stops a re-run from repeating
        side effects nobody noticed.
        """
        source = self._input.toPlainText()
        if not source.strip():
            return
        self._input.clear()
        self._append(f">>> {source}")
        try:
            result = self._session.execute(source)
        except Exception:  # a console must not be the thing that crashes
            self._logger.exception("ScriptSession.execute raised unexpectedly")
            self._append(_("The console failed to run this block."))
            self._status.setText(_("Error"))
            return
        text = result.text
        if text:
            self._append(text)
        self._status.setText(_("Ready") if result.ok else _("Finished with an error"))

    def _on_clear(self) -> None:
        """Empty the output pane. Variables are untouched."""
        self._output.clear()
        self._status.setText("")

    def _on_reset(self) -> None:
        """Forget every user-defined name, then re-show the banner."""
        self._session.reset()
        self._output.clear()
        self._write_banner()
        self._status.setText(_("Namespace reset"))


__all__ = ["ScriptConsoleDialog"]
