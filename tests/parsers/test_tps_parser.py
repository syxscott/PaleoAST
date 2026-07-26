"""Tests for TPS parser - covering BOM, parse errors with line numbers, and comment handling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from parsers.tps_parser import TPSParseError, TPSParser, parse_tps_file


class TestTPSParserBOM:
    """Tests for BOM handling in TPS files."""

    def test_parse_file_with_utf8_bom(self):
        """Test that files with UTF-8 BOM are parsed correctly."""
        content = "﻿LM=3\nID=Specimen1\n10.0 20.0\n30.0 40.0\n50.0 60.0\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert result.n_landmarks == 3
            assert len(result.specimens) == 1
            assert result.specimens[0].id == "Specimen1"
            np.testing.assert_array_almost_equal(result.specimens[0].landmarks, [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        finally:
            Path(filepath).unlink()

    def test_parse_file_with_utf16_bom(self):
        """Test that files with UTF-16 BOM are parsed correctly."""
        content = "LM=3\nID=Specimen1\n10.0 20.0\n30.0 40.0\n50.0 60.0\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-16", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert result.n_landmarks == 3
            assert len(result.specimens) == 1
        finally:
            Path(filepath).unlink()


class TestTPSParserCommentLines:
    """Tests for comment line handling in TPS files."""

    def test_parse_file_with_exclamation_comments(self):
        """Test that ! comment lines are parsed correctly."""
        content = """! This is a comment
LM=3
ID=Specimen1
! Another comment
10.0 20.0
30.0 40.0
50.0 60.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert result.n_landmarks == 3
            assert len(result.comments) == 2
            assert "This is a comment" in result.comments[0]
        finally:
            Path(filepath).unlink()

    def test_parse_file_with_empty_lines(self):
        """Test that empty lines are skipped without error."""
        content = """LM=3
ID=Specimen1

10.0 20.0

30.0 40.0
50.0 60.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert result.n_landmarks == 3
            assert len(result.specimens) == 1
        finally:
            Path(filepath).unlink()


class TestTPSParserStrictMode:
    """Tests for strict mode - parse errors should raise TPSParseError."""

    def test_invalid_coordinate_dimension_raises_error(self):
        """Test that coordinate dimension mismatch raises TPSParseError with line number."""
        content = """LM=3
DIM=2
ID=Specimen1
10.0 20.0
30.0 40.0 50.0
60.0 70.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = TPSParser(strict_mode=True)
            with pytest.raises(TPSParseError) as exc_info:
                parser.parse(filepath)
            assert "line 5" in str(exc_info.value)
            assert "Invalid coordinate dimension" in str(exc_info.value)
        finally:
            Path(filepath).unlink()

    def test_invalid_landmark_count_raises_error(self):
        """Test that missing landmarks raise TPSParseError."""
        content = """LM=5
DIM=2
ID=Specimen1
10.0 20.0
30.0 40.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = TPSParser(strict_mode=True)
            # Should raise error about missing landmarks
            with pytest.raises(TPSParseError):
                parser.parse(filepath)
        finally:
            Path(filepath).unlink()

    def test_non_numeric_coordinate_raises_error(self):
        """Test that non-numeric coordinates raise TPSParseError with line number."""
        content = """LM=3
DIM=2
ID=Specimen1
10.0 20.0
abc def
60.0 70.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = TPSParser(strict_mode=True)
            with pytest.raises(TPSParseError) as exc_info:
                parser.parse(filepath)
            assert "line 5" in str(exc_info.value)
            assert "Cannot parse coordinate line" in str(exc_info.value)
        finally:
            Path(filepath).unlink()

    def test_invalid_dim_value_raises_error(self):
        """Test that invalid DIM value raises TPSParseError."""
        content = """LM=3
DIM=5
ID=Specimen1
10.0 20.0
30.0 40.0
50.0 60.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = TPSParser(strict_mode=True)
            with pytest.raises(TPSParseError) as exc_info:
                parser.parse(filepath)
            assert "DIM" in str(exc_info.value)
            assert "must be 2 or 3" in str(exc_info.value)
        finally:
            Path(filepath).unlink()


class TestTPSParserLegacyMode:
    """Tests for legacy (non-strict) mode - errors are collected but parsing continues."""

    def test_non_strict_mode_collects_errors(self):
        """Test that non-strict mode collects errors and continues parsing."""
        content = """LM=3
DIM=2
ID=Specimen1
10.0 20.0
abc def
60.0 70.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            parser = TPSParser(strict_mode=False)
            result = parser.parse(filepath)
            # Should still parse valid specimens
            assert len(result.specimens) >= 1
            # Error should be tracked
            assert parser._parse_errors.has_errors()
        finally:
            Path(filepath).unlink()


class TestTPSParserValidFiles:
    """Tests for valid TPS file parsing."""

    def test_parse_simple_2d_tps_file(self):
        """Test parsing a simple 2D TPS file."""
        content = """LM=3
DIM=2
ID=Specimen1
10.0 20.0
30.0 40.0
50.0 60.0
ID=Specimen2
15.0 25.0
35.0 45.0
55.0 65.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert result.n_landmarks == 3
            assert result.n_dimensions == 2
            assert len(result.specimens) == 2
            assert result.specimens[0].id == "Specimen1"
            assert result.specimens[1].id == "Specimen2"
        finally:
            Path(filepath).unlink()

    def test_parse_tps_file_with_scale(self):
        """Test parsing TPS file with scale factor."""
        content = """LM=2
DIM=2
SCALE=1.5
ID=Specimen1
10.0 20.0
30.0 40.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert result.specimens[0].scale == 1.5
        finally:
            Path(filepath).unlink()

    def test_parse_tps_file_without_id(self):
        """Test parsing TPS file without explicit ID (auto-generated IDs)."""
        content = """LM=2
DIM=2
10.0 20.0
30.0 40.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            assert len(result.specimens) == 1
            assert result.specimens[0].id.startswith("Specimen_")
        finally:
            Path(filepath).unlink()

    def test_to_matrix(self):
        """Test converting landmarks to 2D matrix."""
        content = """LM=2
DIM=2
ID=Specimen1
10.0 20.0
30.0 40.0
ID=Specimen2
15.0 25.0
35.0 45.0
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tps", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        try:
            result = parse_tps_file(filepath)
            matrix = result.to_matrix()
            assert matrix.shape == (2, 4)
            np.testing.assert_array_almost_equal(matrix[0], [10.0, 20.0, 30.0, 40.0])
        finally:
            Path(filepath).unlink()
