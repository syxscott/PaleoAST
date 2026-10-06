"""Regression tests for shared missing-value sentinels.

Covers defects 1 (DAT/TPS missing-value sentinels) and 2 (improved
field-count error message).

A single shared sentinel set (parsers.sentinels) must drive every parser so
``?``, ``*``, ``NA``, ``N/A``, ``NaN``, ``None``, ``NULL`` and ``-`` all
mean "missing" in .dat, .tps and NEXUS files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from parsers import sentinels
from parsers.dat_parser import DATParseError, DATParser, parse_dat_file
from parsers.nexus_lexer import NexusLexer, NexusTokenType
from parsers.tps_parser import TPSParser


class TestSentinelsSharedModule:
    """The shared ``parsers.sentinels`` module is the single source of truth."""

    def test_canonical_set_contains_required_sentinels(self):
        assert "?" in sentinels.MISSING_SENTINELS
        assert "*" in sentinels.MISSING_SENTINELS
        assert "-" in sentinels.MISSING_SENTINELS
        assert "NA" in sentinels.MISSING_SENTINELS
        assert "N/A" in sentinels.MISSING_SENTINELS
        assert "NAN" in sentinels.MISSING_SENTINELS
        assert "NULL" in sentinels.MISSING_SENTINELS
        assert "NONE" in sentinels.MISSING_SENTINELS

    def test_empty_string_is_not_a_sentinel(self):
        # An empty cell is a layout error (field-count mismatch), not a
        # value error; turning it into NaN silently hides ragged files.
        assert sentinels.is_missing_token("") is False
        assert sentinels.is_missing_token("   ") is False

    def test_is_missing_token_is_case_insensitive_and_strips(self):
        assert sentinels.is_missing_token("?") is True
        assert sentinels.is_missing_token("  *  ") is True
        assert sentinels.is_missing_token("nan") is True
        assert sentinels.is_missing_token("NaN") is True
        assert sentinels.is_missing_token("n/a") is True
        assert sentinels.is_missing_token("NULL") is True
        assert sentinels.is_missing_token("none") is True
        # Non-sentinels
        assert sentinels.is_missing_token("0") is False
        assert sentinels.is_missing_token("42") is False
        assert sentinels.is_missing_token("hello") is False

    def test_upper_set_is_uppercased(self):
        for s in sentinels.MISSING_SENTINELS_UPPER:
            assert s == s.upper()


class TestDATParserMissingSentinels:
    """DAT parser must accept the full shared sentinel set as NaN."""

    @pytest.mark.parametrize(
        "token",
        ["?", "*", "-", "NA", "N/A", "NaN", "None", "NULL", "nan"],
    )
    def test_token_becomes_nan(self, token):
        content = (
            f"Name\tLength\tWidth\n"
            f"Specimen1\t{token}\t5.2\n"
            f"Specimen2\t12.3\t{token}\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            data = parse_dat_file(path)
            assert np.isnan(data.data[0, 0])
            assert np.isnan(data.data[1, 1])
            assert data.data[0, 1] == 5.2
            assert data.data[1, 0] == 12.3
        finally:
            Path(path).unlink()

    def test_all_cells_missing_row(self):
        content = (
            "Name\tLength\tWidth\n"
            "Specimen1\t?\t*\n"
            "Specimen2\tNA\t-\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            data = parse_dat_file(path)
            assert np.all(np.isnan(data.data))
        finally:
            Path(path).unlink()


class TestDATParserFieldCountError:
    """Empty cells must surface as a structured error, never silent zero-fill."""

    def test_short_row_raises_with_line_number(self):
        content = (
            "Name\tLength\tWidth\n"
            "Specimen1\t10.5\t5.2\n"
            "Specimen2\t12.3\n"  # one value missing
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            with pytest.raises(DATParseError) as exc_info:
                DATParser().parse(path)
            msg = str(exc_info.value)
            assert "line 3" in msg
            assert "expected 2" in msg
            assert "got 1" in msg
            # The exception must carry structured fields for tools
            assert exc_info.value.line_number == 3
            assert exc_info.value.expected_fields == 2
            assert exc_info.value.actual_fields == 1
        finally:
            Path(path).unlink()

    def test_error_message_mentions_sentinels(self):
        """Users must be told about ? and * as the right way to mark missing."""
        content = (
            "Name\tLength\tWidth\n"
            "Specimen1\t10.5\t5.2\n"
            "Specimen2\t12.3\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            with pytest.raises(DATParseError) as exc_info:
                DATParser().parse(path)
            msg = str(exc_info.value)
            # The hint must list at least the two main sentinels the user
            # is most likely to know.
            assert "?" in msg
            assert "*" in msg
        finally:
            Path(path).unlink()

    def test_missing_cell_never_becomes_zero(self):
        """Regression for the red-line: missing values must NEVER become 0.0."""
        content = (
            "Name\tLength\tWidth\n"
            "Specimen1\t10.5\t5.2\n"
            "Specimen2\t12.3\n"  # one value missing
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            with pytest.raises(DATParseError):
                DATParser().parse(path)
        finally:
            Path(path).unlink()


class TestTPSParserMissingSentinels:
    """TPS coordinates with ``?`` or ``*`` must parse as NaN, not crash."""

    def _write(self, content: str) -> str:
        # delete=False, so the file has to outlive this block for the parser
        # to open it -- but the handle itself is closed by the `with` rather
        # than by its reference count dropping, which is what happened
        # before and left the write racing the read on Windows.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tps", delete=False
        ) as handle:
            handle.write(content)
            return handle.name

    def test_question_mark_landmark_becomes_nan(self):
        content = (
            "LM=2\n"
            "ID=spec_a\n"
            "0.0 0.0\n"
            "? ?\n"  # missing landmark
            "ID=spec_b\n"
            "1.0 1.0\n"
            "2.0 2.0\n"
        )
        path = self._write(content)
        try:
            tps = TPSParser().parse(path)
            assert len(tps.specimens) == 2
            assert tps.specimens[0].id == "spec_a"
            assert np.isnan(tps.specimens[0].landmarks[1, 0])
            assert np.isnan(tps.specimens[0].landmarks[1, 1])
            assert tps.specimens[0].landmarks[0, 0] == 0.0
            # No NaN in the second specimen
            assert not np.any(np.isnan(tps.specimens[1].landmarks))
        finally:
            Path(path).unlink()

    def test_asterisk_landmark_becomes_nan(self):
        content = (
            "LM=2\n"
            "ID=spec_a\n"
            "* *\n"  # both coords missing
            "1.0 1.0\n"
        )
        path = self._write(content)
        try:
            tps = TPSParser().parse(path)
            assert np.isnan(tps.specimens[0].landmarks[0, 0])
            assert np.isnan(tps.specimens[0].landmarks[0, 1])
            assert tps.specimens[0].landmarks[1, 0] == 1.0
        finally:
            Path(path).unlink()

    def test_na_landmark_becomes_nan(self):
        content = (
            "LM=2\n"
            "ID=spec_a\n"
            "NA NA\n"
            "1.0 1.0\n"
        )
        path = self._write(content)
        try:
            tps = TPSParser().parse(path)
            assert np.isnan(tps.specimens[0].landmarks[0, 0])
            assert np.isnan(tps.specimens[0].landmarks[0, 1])
        finally:
            Path(path).unlink()


class TestNexusLexerMissingSentinels:
    """NEXUS lexer must surface ``?`` and ``-`` as MISSING tokens, not
    crash and not as random symbols."""

    def setup_method(self):
        self.lexer = NexusLexer()

    def test_question_mark_is_missing_token(self):
        tokens = self.lexer.tokenize("?")
        miss = [t for t in tokens if t.type == NexusTokenType.MISSING]
        assert len(miss) == 1
        assert miss[0].value == "?"

    def test_minus_is_missing_token(self):
        tokens = self.lexer.tokenize("-")
        miss = [t for t in tokens if t.type == NexusTokenType.MISSING]
        assert len(miss) == 1
        assert miss[0].value == "-"

    def test_asterisk_is_missing_token(self):
        tokens = self.lexer.tokenize("*")
        miss = [t for t in tokens if t.type == NexusTokenType.MISSING]
        assert len(miss) == 1
        assert miss[0].value == "*"

    def test_na_is_not_a_missing_token(self):
        """NEXUS distinguishes the literal string 'NA' from the missing
        symbol. Only the shared glyph sentinels are tokens; spelling-based
        sentinels like ``NA`` are handled downstream by the matrix parser."""
        tokens = self.lexer.tokenize("NA")
        # NA is a plain identifier; the lexer should NOT promote it to MISSING
        assert all(t.type != NexusTokenType.MISSING for t in tokens)
