# =============================================================================
# FILE: views/ui_runlist_panel.py
# =============================================================================
"""
Run-queue (batch) dock panel.

A thin Qt view over :class:`presets.runner.RunQueue` and the preset library
(:class:`presets.manager.PresetManager`): rows show queued analyses, their
parameter summaries and live status; buttons add/remove runs and start a
pass.  All sequencing logic lives in :class:`RunQueue` (Qt-free, unit
tested); this panel only renders items and forwards button clicks.

Borrowed from the surface-morphometrics-gui job workflow: a runlist is
built from saved preset files, executed one item at a time, and each item
keeps its own status/error text for the post-run manifest.

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.constants import APP_VERSION
from config.i18n import _
from presets.manager import PresetManager
from presets.registry import ANALYSIS_REGISTRY, PresetError
from presets.runner import ERROR, OK, PENDING, RUNNING, SKIPPED, RunQueue, RunQueueItem
from presets.schema import Preset

logger = logging.getLogger(__name__)


def _status_label(status: str) -> str:
    return {
        PENDING: _("Pending"),
        RUNNING: _("Running..."),
        OK: _("OK"),
        ERROR: _("Error"),
        SKIPPED: _("Skipped"),
    }.get(status, status)


def _params_summary(item: RunQueueItem) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(item.params.items()))


class AddRunDialog(QDialog):
    """Pick an analysis and (optionally) a saved preset to queue one run."""

    def __init__(self, manager: PresetManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle(_("Add Run to Queue"))
        self.setModal(True)
        layout = QVBoxLayout(self)

        form = QHBoxLayout()
        form.addWidget(QLabel(_("Analysis:")))
        self._analysis_combo = QComboBox()
        for analysis_id in sorted(ANALYSIS_REGISTRY):
            self._analysis_combo.addItem(ANALYSIS_REGISTRY[analysis_id].label, analysis_id)
        form.addWidget(self._analysis_combo, stretch=1)
        layout.addLayout(form)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(_("Preset:")))
        self._preset_combo = QComboBox()
        preset_row.addWidget(self._preset_combo, stretch=1)
        layout.addLayout(preset_row)

        self._analysis_combo.currentIndexChanged.connect(self._reload_presets)
        self._reload_presets()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reload_presets(self) -> None:
        analysis_id = self._analysis_combo.currentData()
        self._preset_combo.clear()
        self._preset_combo.addItem(_("Default parameters"), None)
        for path in self._manager.iter_files():
            try:
                preset = self._manager.load(path)
            except (OSError, ValueError, PresetError) as exc:
                logger.warning("Skipping invalid preset %s: %s", path, exc)
                continue
            if preset.analysis_id == analysis_id:
                self._preset_combo.addItem(preset.name, preset)

    def selected(self) -> tuple[str, Preset | None]:
        """Return ``(analysis_id, preset or None)``."""
        return self._analysis_combo.currentData(), self._preset_combo.currentData()


class RunListPanel(QDockWidget):
    """Dockable batch-run table backed by a shared :class:`RunQueue`."""

    run_all_requested = pyqtSignal()

    def __init__(
        self,
        queue: RunQueue,
        manager: PresetManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(_("Run Queue"), parent)
        self.setObjectName("RunListPanel")
        self._logger = logging.getLogger(f"{__name__}.RunListPanel")
        self._queue = queue
        self._manager = manager
        self._is_dark = False
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)

        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            [_("Analysis"), _("Preset"), _("Parameters"), _("Status")]
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        buttons = QHBoxLayout()
        self._add_button = QPushButton(_("Add..."))
        self._add_button.clicked.connect(self._on_add)
        self._remove_button = QPushButton(_("Remove"))
        self._remove_button.clicked.connect(self._on_remove)
        self._save_preset_button = QPushButton(_("Save Preset..."))
        self._save_preset_button.setToolTip(_("Save the selected run's parameters as a preset"))
        self._save_preset_button.clicked.connect(self._on_save_preset)
        self._run_all_button = QPushButton(_("Run All"))
        self._run_all_button.clicked.connect(self.run_all_requested)
        self._reset_button = QPushButton(_("Reset Results"))
        self._reset_button.clicked.connect(self._on_reset)
        for button in (
            self._add_button,
            self._remove_button,
            self._save_preset_button,
            self._run_all_button,
            self._reset_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.setWidget(body)

    # ------------------------------------------------------------------
    def setDarkTheme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self.refresh()

    def _row_for(self, item: RunQueueItem) -> int:
        items = self._queue.items
        for row, candidate in enumerate(items):
            if candidate is item:
                return row
        return -1

    def refresh(self) -> None:
        items = self._queue.items
        self._table.setRowCount(len(items))
        for row, item in enumerate(items):
            status = _status_label(item.status)
            if item.error:
                status = f"{status}: {item.error}"
            cells = (
                ANALYSIS_REGISTRY[item.analysis_id].label if item.analysis_id in ANALYSIS_REGISTRY
                else item.analysis_id,
                item.preset_name or "-",
                _params_summary(item),
                status,
            )
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(str(text)))

    def item_changed(self, item: RunQueueItem) -> None:
        """RunQueue on_change hook: repaint one row (and the busy state)."""
        row = self._row_for(item)
        if row < 0:
            self.refresh()
            return
        status = _status_label(item.status)
        if item.error:
            status = f"{status}: {item.error}"
        self._table.setItem(row, 3, QTableWidgetItem(status))
        self._run_all_button.setEnabled(not self._queue.is_busy())

    # ------------------------------------------------------------------
    def _on_add(self) -> None:
        dialog = AddRunDialog(self._manager, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        analysis_id, preset = dialog.selected()
        if not analysis_id:
            return
        spec = ANALYSIS_REGISTRY[analysis_id]
        if preset is not None:
            params, preset_name = dict(preset.params), preset.name
            for warning in preset.warnings:
                self._logger.warning(f"Preset '{preset.name}': {warning}")
        else:
            params, preset_name = spec.defaults(), ""
        self._queue.add(RunQueueItem(analysis_id=analysis_id, params=params, preset_name=preset_name))
        self.refresh()

    def _on_save_preset(self) -> None:
        """Save the selected queued run's parameters as a named preset."""
        row = self._table.currentRow()
        items = self._queue.items
        if row < 0 or row >= len(items):
            QMessageBox.information(
                self, _("Save Preset"), _("Select a queued run first to save its parameters.")
            )
            return
        item = items[row]
        name, ok = QInputDialog.getText(self, _("Save Preset"), _("Preset name:"))
        if not ok or not name.strip():
            return
        preset = Preset(
            name=name.strip(),
            analysis_id=item.analysis_id,
            params=dict(item.params),
            created_at=datetime.now().isoformat(timespec="seconds"),
            app_version=APP_VERSION,
        )
        try:
            path = self._manager.save(preset)
        except (OSError, ValueError) as exc:  # PresetError subclasses ValueError
            self._logger.error(f"Preset save failed: {exc}")
            QMessageBox.warning(self, _("Save Preset Failed"), str(exc))
            return
        self._logger.info(f"Preset saved: {path}")

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        items = self._queue.items
        if row < len(items) and items[row].status == RUNNING:
            self._logger.info("Cannot remove a running item")
            return
        self._queue.remove(row)
        self.refresh()

    def _on_reset(self) -> None:
        if self._queue.is_busy():
            return
        self._queue.clear_results()
        self.refresh()
