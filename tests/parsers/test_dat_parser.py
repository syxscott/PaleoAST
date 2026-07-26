"""Tests for DAT parser - covering BOM, field count mismatch, European thousand separators, and NaN handling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from parsers.dat_parser import DATParseError, DATParser, parse_dat_file


class TestDATParserBOM:
    """Tests for BOM handling in DAT files."""

    def test_parse_file_with_utf8_bom(self):
        """Test that files with UTF-8 BOM are parsed correctly."""
        content = """﻿Name	Length	Width
Specimen1	10.5	5.2
Specimen2	12.3	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape[0] == 2
            assert result.data.shape[1] == 2
        finally:
            Path(filepath).unlink()


class TestDATParserEuropeanFormat:
    """Tests for European thousand separator handling."""

    def test_european_decimal_comma(self):
        """Test parsing European decimal comma format (1234,56)."""
        content = """Name	Length	Width
Specimen1	10,5	5,2
Specimen2	12,3	6,1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape == (2, 2)
            np.testing.assert_almost_equal(result.data[0, 0], 10.5)
            np.testing.assert_almost_equal(result.data[0, 1], 5.2)
        finally:
            Path(filepath).unlink()

    def test_european_thousand_separator_dot(self):
        """Test parsing European thousand separator with dot (1.234,56 -> 1234.56)."""
        content = """Name	Length	Width
Specimen1	1.234,56	5.2
Specimen2	2.345,67	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            np.testing.assert_almost_equal(result.data[0, 0], 1234.56)
            np.testing.assert_almost_equal(result.data[1, 0], 2345.67)
        finally:
            Path(filepath).unlink()

    def test_us_thousand_separator_comma(self):
        """Test parsing US thousand separator (1,234.56 -> 1234.56)."""
        content = """Name	Length	Width
Specimen1	1,234.56	5.2
Specimen2	2,345.67	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            np.testing.assert_almost_equal(result.data[0, 0], 1234.56)
            np.testing.assert_almost_equal(result.data[1, 0], 2345.67)
        finally:
            Path(filepath).unlink()


class TestDATParserFieldCountMismatch:
    """Tests for strict field count validation - should raise DATParseError."""

    def test_missing_field_raises_error(self):
        """Test that missing fields in a row raise DATParseError with line number."""
        content = """Name	Length	Width
Specimen1	10.5	5.2
Specimen2	12.3
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = DATParser()
            with pytest.raises(DATParseError) as exc_info:
                parser.parse(filepath)
            assert "line 3" in str(exc_info.value)
            assert "Field count mismatch" in str(exc_info.value)
            assert exc_info.value.expected_fields == 2
            assert exc_info.value.actual_fields == 1
        finally:
            Path(filepath).unlink()

    def test_extra_field_raises_error(self):
        """Test that extra fields in a row raise DATParseError."""
        content = """Name	Length	Width
Specimen1	10.5	5.2
Specimen2	12.3	6.1	7.9
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = DATParser()
            with pytest.raises(DATParseError) as exc_info:
                parser.parse(filepath)
            assert "Field count mismatch" in str(exc_info.value)
            assert exc_info.value.expected_fields == 2
            assert exc_info.value.actual_fields == 3
        finally:
            Path(filepath).unlink()

    def test_header_mismatch_raises_error(self):
        """Test that field count mismatch between header and data raises error."""
        content = """Name	Length
Specimen1	10.5	5.2
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = DATParser()
            with pytest.raises(DATParseError) as exc_info:
                parser.parse(filepath)
            assert "Field count mismatch" in str(exc_info.value)
        finally:
            Path(filepath).unlink()


class TestDATParserNaNHandling:
    """Tests for NaN value detection and reporting."""

    def test_nan_values_parsed_correctly(self):
        """Test that NaN values are correctly parsed as np.nan."""
        content = """Name	Length	Width
Specimen1	10.5	NaN
Specimen2	NaN	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert np.isnan(result.data[0, 1])
            assert np.isnan(result.data[1, 0])
        finally:
            Path(filepath).unlink()

    def test_na_nan_null_handled(self):
        """Test that various NA/NaN/Null representations are handled."""
        content = """Name	Length	Width
Specimen1	NA	5.2
Specimen2	12.3	N/A
Specimen3	-	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert np.isnan(result.data[0, 0])
            assert np.isnan(result.data[1, 1])
            assert np.isnan(result.data[2, 0])
        finally:
            Path(filepath).unlink()


class TestDATParserCommentLines:
    """Tests for comment line handling."""

    def test_hash_comment_lines_skipped(self):
        """Test that # comment lines are skipped."""
        content = """# This is a comment
Name	Length	Width
Specimen1	10.5	5.2
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape == (1, 2)
            assert result.comments is not None
            assert "This is a comment" in result.comments[0]
        finally:
            Path(filepath).unlink()

    def test_bracket_comment_lines_skipped(self):
        """Test that [comment] lines are skipped."""
        content = """[This is a comment]
Name	Length	Width
Specimen1	10.5	5.2
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape == (1, 2)
        finally:
            Path(filepath).unlink()

    def test_curly_brace_group_lines(self):
        """Test that {GroupName} lines are handled as group assignments."""
        content = """{Group1}
Name	Length	Width
Specimen1	10.5	5.2
Specimen2	12.3	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.groups is not None
            assert "Group1" in result.groups
        finally:
            Path(filepath).unlink()


class TestDATParserValidFiles:
    """Tests for valid DAT file parsing."""

    def test_parse_simple_dat_file(self):
        """Test parsing a simple DAT file."""
        content = """Name	Length	Width
Specimen1	10.5	5.2
Specimen2	12.3	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape == (2, 2)
            assert result.row_labels == ["Specimen1", "Specimen2"]
            assert result.col_labels == ["Name", "Length", "Width"]
        finally:
            Path(filepath).unlink()

    def test_parse_dat_file_without_header(self):
        """Test parsing DAT file without header row."""
        content = """Specimen1	10.5	5.2
Specimen2	12.3	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape == (2, 3)
            assert result.row_labels == ["Specimen1", "Specimen2"]
        finally:
            Path(filepath).unlink()

    def test_parse_dat_file_with_only_data(self):
        """Test parsing DAT file with no labels."""
        content = """10.5	5.2
12.3	6.1
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".dat", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_dat_file(filepath)
            assert result.data.shape == (2, 2)
        finally:
            Path(filepath).unlink()
