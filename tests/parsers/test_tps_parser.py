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


class TestTPSCurveSections:
    """CO=/POINTS= curve accumulation and get_curves() wiring (W2)."""

    def _write(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tps", delete=False
        ) as f:
            f.write(content)
        return f.name

    def test_multiple_curves_accumulate(self):
        content = """LM=3
DIM=2
ID=S1
0.0 0.0
1.0 0.0
2.0 0.0
CO=
curveA
POINTS=0 0
POINTS=0.5 0.1
POINTS=1 0
CO=
curveB
POINTS=1 0
POINTS=1.5 0.2
POINTS=2 0
"""
        path = self._write(content)
        try:
            result = parse_tps_file(path)
            assert len(result.specimens) == 1
            cp = result.specimens[0].curve_points
            assert [c["name"] for c in cp["curves"]] == ["curveA", "curveB"]
            assert len(cp["curves"][0]["points"]) == 3
            assert len(cp["curves"][1]["points"]) == 3
            # legacy aliases point at the LAST curve
            assert cp["points"] is cp["curves"][1]["points"]
        finally:
            Path(path).unlink()

    def test_curves_attach_after_auto_finalize(self):
        """CO section follows the LM block; the specimen was already
        finalized by landmark count — curves must not be dropped."""
        content = """LM=3
DIM=2
ID=S1
0.0 0.0
1.0 0.0
2.0 0.0
CO=
curveA
POINTS=0 0
POINTS=0.5 0.1
POINTS=1 0
ID=S2
3.0 3.0
4.0 4.0
5.0 5.0
"""
        path = self._write(content)
        try:
            result = parse_tps_file(path)
            assert len(result.specimens) == 2
            s1, s2 = result.specimens
            assert s1.curve_points is not None
            assert len(s1.curve_points["curves"]) == 1
            assert len(s1.curve_points["curves"][0]["points"]) == 3
            assert s2.curve_points is None  # ID reset; no CO for S2
        finally:
            Path(path).unlink()

    def test_get_curves_blocks(self):
        from parsers.tps_parser import TPSFile, TPSSpecimen

        spec = TPSSpecimen(
            id="S1",
            landmarks=np.zeros((9, 2)),
            curve_points={
                "curves": [
                    {"order": [], "points": [[0, 0]] * 4},
                    {"order": [], "points": [[0, 0]] * 5},
                ]
            },
        )
        f = TPSFile(specimens=[spec], n_landmarks=9, n_dimensions=2, comments=[])
        assert f.get_curves(n_fixed=0) == [[0, 1, 2, 3], [4, 5, 6, 7, 8]]

        spec9 = TPSSpecimen(
            id="S1",
            landmarks=np.zeros((10, 2)),
            curve_points=spec.curve_points,
        )
        f10 = TPSFile(specimens=[spec9], n_landmarks=10, n_dimensions=2, comments=[])
        assert f10.get_curves(n_fixed=1) == [[1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_get_curves_overflow_raises(self):
        from parsers.tps_parser import TPSFile, TPSSpecimen

        spec = TPSSpecimen(
            id="S1",
            landmarks=np.zeros((4, 2)),
            curve_points={"curves": [{"order": [], "points": [[0, 0]] * 5}]},
        )
        f = TPSFile(specimens=[spec], n_landmarks=4, n_dimensions=2, comments=[])
        with pytest.raises(ValueError, match="overflow|declares"):
            f.get_curves(n_fixed=0)

    def test_get_curves_inconsistent_lengths_raise(self):
        from parsers.tps_parser import TPSFile, TPSSpecimen

        specs = [
            TPSSpecimen(
                id="S1",
                landmarks=np.zeros((6, 2)),
                curve_points={"curves": [{"order": [], "points": [[0, 0]] * 3}]},
            ),
            TPSSpecimen(
                id="S2",
                landmarks=np.zeros((6, 2)),
                curve_points={"curves": [{"order": [], "points": [[0, 0]] * 4}]},
            ),
        ]
        f = TPSFile(specimens=specs, n_landmarks=6, n_dimensions=2, comments=[])
        with pytest.raises(ValueError, match="disagree"):
            f.get_curves(n_fixed=0)

    def test_get_curves_no_data_raises(self):
        from parsers.tps_parser import TPSFile, TPSSpecimen

        spec = TPSSpecimen(id="S1", landmarks=np.zeros((3, 2)), curve_points=None)
        f = TPSFile(specimens=[spec], n_landmarks=3, n_dimensions=2, comments=[])
        with pytest.raises(ValueError, match="No CO="):
            f.get_curves(n_fixed=0)
