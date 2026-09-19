"""Qt smoke tests for views/ui_runlist_panel.py (offscreen, no dialogs shown)."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from presets.manager import PresetManager
from presets.registry import ANALYSIS_REGISTRY
from presets.runner import OK, PENDING, RUNNING, RunQueue, RunQueueItem
from views.ui_runlist_panel import RunListPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(["paleoast-runlist-tests"])
    yield app


def _make_panel(tmp_path, executor=None):
    queue = RunQueue(executor=executor or (lambda item, finish: finish(OK, None)))
    panel = RunListPanel(queue, PresetManager(str(tmp_path)))
    return queue, panel


class TestRunListPanel:
    def test_refresh_renders_rows(self, qapp, tmp_path):
        queue, panel = _make_panel(tmp_path)
        queue.add(RunQueueItem(analysis_id="pca", params={"n_components": 3}, preset_name="My PCA"))
        queue.add(RunQueueItem(analysis_id="anosim", params={}))
        panel.refresh()
        table = panel._table
        assert table.rowCount() == 2
        assert table.item(0, 0).text() == ANALYSIS_REGISTRY["pca"].label
        assert table.item(0, 1).text() == "My PCA"
        assert "n_components=3" in table.item(0, 2).text()
        assert table.item(0, 3).text() == "Pending"
        # no preset name -> placeholder dash
        assert table.item(1, 1).text() == "-"

    def test_item_changed_updates_status_cell(self, qapp, tmp_path):
        queue, panel = _make_panel(tmp_path)
        item = RunQueueItem(analysis_id="pcoa", params={})
        queue.add(item)
        panel.refresh()
        item.status = OK
        panel.item_changed(item)
        assert panel._table.item(0, 3).text() == "OK"

    def test_item_changed_shows_error_text(self, qapp, tmp_path):
        queue, panel = _make_panel(tmp_path)
        item = RunQueueItem(analysis_id="nmds", params={})
        queue.add(item)
        panel.refresh()
        item.status = RUNNING
        panel.item_changed(item)
        assert panel._table.item(0, 3).text() == "Running..."
        item.status = OK
        item.error = "boom"
        panel.item_changed(item)
        assert panel._table.item(0, 3).text() == "OK: boom"

    def test_remove_deletes_row(self, qapp, tmp_path):
        queue, panel = _make_panel(tmp_path)
        queue.add(RunQueueItem(analysis_id="pca", params={}))
        queue.add(RunQueueItem(analysis_id="pca", params={}))
        panel.refresh()
        panel._table.setCurrentCell(0, 0)
        panel._on_remove()
        assert len(queue.items) == 1
        assert panel._table.rowCount() == 1

    def test_remove_refuses_running_item(self, qapp, tmp_path):
        queue, panel = _make_panel(tmp_path)
        item = RunQueueItem(analysis_id="pca", params={})
        queue.add(item)
        panel.refresh()
        item.status = RUNNING
        panel._table.setCurrentCell(0, 0)
        panel._on_remove()
        assert len(queue.items) == 1  # untouched

    def test_reset_returns_items_to_pending(self, qapp, tmp_path):
        queue, panel = _make_panel(tmp_path)
        queue.add(RunQueueItem(analysis_id="pca", params={}))
        queue.run_next()
        assert queue.items[0].status == OK
        panel._on_reset()
        assert queue.items[0].status == PENDING
        assert panel._table.item(0, 3).text() == "Pending"

    def test_status_hook_keeps_panel_in_sync(self, qapp, tmp_path):
        # The queue's on_change lambda in MainWindow wiring points at
        # item_changed; emulate the same hookup here.
        queue = RunQueue(
            executor=lambda item, finish: finish(OK, None),
            on_change=lambda item: panel.item_changed(item),
        )
        panel = RunListPanel(queue, PresetManager(str(tmp_path)))
        queue.add(RunQueueItem(analysis_id="pca", params={}))
        panel.refresh()
        queue.run_next()
        assert panel._table.item(0, 3).text() == "OK"
