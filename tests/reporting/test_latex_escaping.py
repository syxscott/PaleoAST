"""Regression tests for LaTeX escaping in the reporting layer.

Three of the four LaTeX producers in this package escaped their caller-supplied
text and one did not, which is the worst possible ratio: the gap is invisible
until a document fails to compile, and it fails on somebody else's machine.

The strings that reach a report are taxon names, and they carry the characters
LaTeX treats specially all the time: ``Globigerinoides_ruber`` has an
underscore, ``Smith & Jones 2021`` an ampersand.
"""

from __future__ import annotations

import numpy as np
import pytest

from reporting.figure_handler import _escape_latex
from reporting.matrix_converter import MatrixConverter
from reporting.report_builder import ReportBuilder
from reporting.table_generator import TableGenerator

MATRIX = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

#: Strings that must never reach a LaTeX document unescaped.
HOSTILE = [
    "Globigerinoides_ruber",
    "Smith & Jones 2021",
    "50% complete",
    "sample#3",
    "$O$ isotope",
    "A~B",
    "x^2",
    "}\n\\input{secret.tex}{",
]


def _raw_latex_characters(text: str) -> list[str]:
    """LaTeX-significant characters still present in ``text`` unescaped."""
    # Strip the known-good replacements, then look for the bare characters.
    stripped = text
    for good in (
        r"\textbackslash{}",
        r"\&",
        r"\%",
        r"\$",
        r"\#",
        r"\_",
        r"\{",
        r"\}",
        r"\textasciitilde{}",
        r"\textasciicircum{}",
    ):
        stripped = stripped.replace(good, "")
    return [c for c in "&%#_${}~^" if c in stripped]


class TestMatrixConverterEscapesLabels:
    @pytest.mark.parametrize("label", HOSTILE)
    def test_row_labels_are_escaped(self, label):
        out = MatrixConverter.to_latex(MATRIX, [label, "second"], ["a", "b", "c"])
        assert _escape_latex(label) in out, f"{label!r} was not escaped"
        assert not _raw_latex_characters(label) or _escape_latex(label) in out

    @pytest.mark.parametrize("label", HOSTILE)
    def test_col_labels_are_escaped(self, label):
        out = MatrixConverter.to_latex(MATRIX, None, [label, "b", "c"])
        assert _escape_latex(label) in out, f"{label!r} was not escaped"

    def test_real_taxon_name_produces_compilable_output(self):
        out = MatrixConverter.to_latex(MATRIX, ["Globigerinoides_ruber", "Turborotalia_ampla"], ["v1", "v2", "v3"])
        assert r"Globigerinoides\_ruber" in out
        assert "Globigerinoides_ruber &" not in out

    def test_injection_attempt_cannot_close_the_cell(self):
        out = MatrixConverter.to_latex(MATRIX, ["}\n\\input{secret.tex}{", "b"], None)
        assert r"\}" in out
        assert "}\n\\input" not in out

    def test_numbers_are_untouched_by_escaping(self):
        out = MatrixConverter.to_latex(MATRIX)
        assert "1.0000" in out and "6.0000" in out


class TestMatrixConverterValidatesShapes:
    """A wrong-length label list used to emit malformed LaTeX silently."""

    @pytest.mark.parametrize("labels", [["a", "b"], ["a", "b", "c", "d"], []])
    def test_col_labels_must_match_column_count(self, labels):
        with pytest.raises(ValueError, match="col_labels"):
            MatrixConverter.to_latex(MATRIX, None, labels)

    @pytest.mark.parametrize("labels", [["only-one"], [], ["a", "b", "c"]])
    def test_row_labels_must_match_row_count(self, labels):
        with pytest.raises(ValueError, match="row_labels"):
            MatrixConverter.to_latex(MATRIX, labels, ["a", "b", "c"])

    def test_non_2d_matrix_is_refused(self):
        with pytest.raises(ValueError, match="2-D"):
            MatrixConverter.to_latex(np.array([1.0, 2.0, 3.0]))

    def test_matching_labels_still_work(self):
        out = MatrixConverter.to_latex(MATRIX, ["r0", "r1"], ["a", "b", "c"])
        assert "r0" in out and "c" in out

    def test_every_row_has_the_declared_number_of_cells(self):
        import re

        out = MatrixConverter.to_latex(MATRIX, ["r0", "r1"], ["a", "b", "c"])
        m = re.search(r"\\begin\{tabular\}\{\|(.+)\|\}", out)
        declared = len(m.group(1).split("|"))
        body = [ln for ln in out.splitlines() if "&" in ln and "hline" not in ln and "begin" not in ln]
        assert body, "no body rows emitted"
        assert {len(ln.split("&")) for ln in body} == {declared}


class TestReportBuilderEscapesEverything:
    @staticmethod
    def _tex(tmp_path, builder) -> str:
        out = tmp_path / "report.tex"
        builder.generate(str(out))
        return out.read_text(encoding="utf-8")

    def test_keywords_are_escaped(self, tmp_path):
        builder = ReportBuilder()
        builder.set_abstract("Summary", keywords=["N=5 & 95% CI", "Globigerinoides_ruber"])
        latex = self._tex(tmp_path, builder)
        assert r"\&" in latex
        assert r"\_" in latex

    def test_section_label_is_escaped_like_its_title(self, tmp_path):
        builder = ReportBuilder()
        builder.add_section("Body_mass & 95% CI", "text", label="sec_results_2021")
        latex = self._tex(tmp_path, builder)
        assert r"\label{sec\_results\_2021}" in latex
        assert r"Body\_mass \& 95\% CI" in latex

    def test_title_escaping_still_holds(self, tmp_path):
        builder = ReportBuilder()
        builder.set_title("Results (n=5)")
        latex = self._tex(tmp_path, builder)
        assert "Results (n=5)" in latex


class TestSiblingProducersStillEscape:
    """The fix must not have disturbed the ones that already worked."""

    def test_table_generator_escapes(self):
        out = TableGenerator.from_matrix([["Globigerinoides_ruber", 1.0]], headers=["taxon", "v"])
        assert r"\_" in out

    def test_figure_handler_escapes(self):
        from reporting.figure_handler import FigureHandler

        out = FigureHandler.include_figure("f.png", caption="Body_mass & 95%")
        assert r"\_" in out and r"\&" in out
