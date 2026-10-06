# =============================================================================
# FILE: tests/test_dat_group_labels_survive_loading.py
# =============================================================================
"""
A PAST .dat file's {Group} lines must reach the analysis pipeline.

THE BUG
-------
``parsers/dat_parser.py`` reads the ``{Group}`` blocks and puts one label per
data row into ``DATResult.groups``; ``file_drop_handler._parse_dat`` forwards
it; and then ``_on_file_dropped`` built ``DataMatrix`` from the row and column
labels only. The grouping was dropped on the floor.

The failure is silent and it is a *result* failure, not a cosmetic one. The
file loads, the row count in the status bar is correct, and every
grouping-aware method -- ANOSIM, SIMPER, PERMANOVA, and the colour-by-group
on every ordination plot -- then runs as if the whole file were a single
group. Nothing on screen says the information was thrown away.

WHY THE FIX IS NARROW
---------------------
The grouping travels in ``specimen_metadata``, because that is the source
``_resolve_groups_for_plot`` already reads, and it only accepts a value when
*every* row has one. So a partial grouping must be refused rather than padded:
inventing a group for the unlabelled rows would produce a confidently wrong
analysis instead of an obvious absence.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the loader lives on the Qt main window")

from models.data_matrix import DataMatrix
from parsers.dat_parser import parse_dat_file
from views.ui_main_window import MainWindow

# The PAST layout the parser accepts: a {Group} line, the header row, then that
# group's rows; a later {Group} line switches group WITHOUT repeating the
# header. Verified against parse_dat_file rather than assumed -- repeating the
# header makes the parser try to parse "Length" as a float.
DAT_WITH_GROUPS = """{Habitat A}
Name\tLength\tWidth
Specimen1\t10.5\t5.2
Specimen2\t12.3\t6.1
{Habitat B}
Specimen3\t11.0\t5.9
Specimen4\t12.8\t6.4
"""

# The realistic PAST layout: data rows appear before the first {Group} line, so
# those rows carry no label. The parser reports None for them, and the loader
# must refuse the whole grouping rather than invent one.
DAT_WITH_UNLABELLED_HEAD = """Name\tLength\tWidth
Specimen1\t10.5\t5.2
Specimen2\t12.3\t6.1
{Habitat A}
Specimen3\t11.0\t5.9
Specimen4\t12.8\t6.4
"""


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _metadata(groups, n_rows):
    win = MainWindow.__new__(MainWindow)
    win._logger = _SilentLog()
    return MainWindow._specimen_metadata_from_groups(win, groups, np.zeros((n_rows, 2)))


class _SilentLog:
    def warning(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def test_parser_still_reads_the_group_lines(tmp_path):
    """The parser was never the problem; keep that pinned separately."""
    p = tmp_path / "grouped.dat"
    p.write_text(DAT_WITH_GROUPS, encoding="utf-8")
    result = parse_dat_file(str(p))
    assert result.groups == [
        "Habitat A",
        "Habitat A",
        "Habitat B",
        "Habitat B",
    ], result.groups


def test_parser_reports_none_for_rows_before_the_first_group(tmp_path):
    """The layout PAST actually writes, and the source of the partial case."""
    p = tmp_path / "unlabelled_head.dat"
    p.write_text(DAT_WITH_UNLABELLED_HEAD, encoding="utf-8")
    result = parse_dat_file(str(p))
    assert result.groups == [None, None, "Habitat A", "Habitat A"], result.groups
    assert _metadata(result.groups, 4) is None, "an unlabelled head must refuse the grouping, not invent a group"


def test_complete_grouping_becomes_specimen_metadata():
    meta = _metadata(["A", "A", "B", "B"], 4)
    assert meta == [{"group": "A"}, {"group": "A"}, {"group": "B"}, {"group": "B"}]


def test_no_groups_yields_none():
    assert _metadata(None, 4) is None
    assert _metadata([], 4) is None


def test_length_mismatch_is_refused_not_misaligned():
    """Padding or truncating would attach a group to the wrong specimen."""
    assert _metadata(["A", "B"], 4) is None
    assert _metadata(["A"] * 6, 4) is None


def test_partially_labelled_grouping_is_refused():
    """A .dat may have rows before the first {Group} line.

    Filling those in would invent a group the file never declared, and the
    analysis would report a difference that is an artefact of the guess.
    """
    assert _metadata(["A", None, "B", "B"], 4) is None
    assert _metadata(["A", "", "B", "B"], 4) is None


def test_dropped_dat_file_ends_up_grouped(qapp, tmp_path, monkeypatch):
    """End to end: the file the parser reads is the matrix the app gets."""
    p = tmp_path / "grouped.dat"
    p.write_text(DAT_WITH_GROUPS, encoding="utf-8")

    from views.file_drop_handler import FileDropHandler

    handler = FileDropHandler.__new__(FileDropHandler)
    handler._logger = _SilentLog()
    parsed = FileDropHandler._parse_dat(handler, str(p))
    assert parsed["groups"] == [
        "Habitat A",
        "Habitat A",
        "Habitat B",
        "Habitat B",
    ]

    meta = _metadata(parsed["groups"], len(parsed["data"]))
    assert meta is not None, "the groups were dropped between parse and matrix"

    matrix = DataMatrix(
        parsed["data"],
        row_labels=parsed["row_labels"],
        col_labels=parsed["col_labels"],
        specimen_metadata=meta,
    )
    # And the source _resolve_groups_for_plot actually reads can see them.
    assert matrix.specimen_metadata[0]["group"] == "Habitat A"
    assert matrix.specimen_metadata[3]["group"] == "Habitat B"
