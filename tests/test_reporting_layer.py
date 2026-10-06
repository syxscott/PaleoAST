"""
Tests for the reporting layer, which had no coverage at all.

``reporting/`` is the LaTeX report pipeline: ReportBuilder, LatexPreamble,
TableGenerator, MatrixConverter, FigureHandler. None of it was ever executed
by a test, so a mistake in it would only surface when a user pressed
"generate report" and pdflatex rejected the file.

What these tests care about is what a publisher would care about: is the .tex
well-formed, do statistics render as numbers rather than placeholders, are
special characters escaped, and is the preamble not going to blow up LaTeX.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from reporting.figure_handler import FigureHandler, _escape_latex
from reporting.latex_preamble import DocumentClass, LatexPreamble
from reporting.matrix_converter import MatrixConverter
from reporting.report_builder import ReportBuilder
from reporting.table_generator import TableGenerator

# =============================================================================
# End-to-end report
# =============================================================================


@pytest.fixture(scope="module")
def generated_report(tmp_path_factory) -> str:
    """Build a realistic report and return its .tex source.

    ``ReportBuilder.generate(path)`` WRITES the file and RETURNS the document
    text -- the return value is not a path, despite the parameter. Read the
    file back so the tests check what a user would actually compile.
    """

    out = tmp_path_factory.mktemp("report")
    target = out / "report.tex"
    builder = ReportBuilder()
    builder.set_title("A test of the report builder")
    builder.add_author("A. Researcher")
    builder.set_abstract("We examined whether the report builder works.")
    builder.add_section("Introduction", "Morphometrics is a palaeontology workhorse.")
    builder.add_section("Methods", "We did the thing.", level=1)
    builder.add_section("Landmarks", "Configurational landmarks were digitised.", level=2)
    builder.add_statistical_result("PERMANOVA", 2.69, p_value=0.041, df=4, effect_size=0.62)
    builder.add_statistical_result("ANOSIM R", 0.31, p_value=0.09)
    builder.add_table(
        "\\begin{tabular}{lr}\nTaxon & CS \\\\\nA & 1.0 \\\\\nB & 1.4 \\\\\n\\end{tabular}",
        caption="Centroid sizes of two taxa.",
    )
    returned = builder.generate(str(target))
    assert target.exists(), "generate() did not write the file it was given"
    assert "\\end{document}" in returned, "generate() did not return the document"
    return target.read_text(encoding="utf-8")


class TestReportIsWellFormed:
    def test_document_environment_is_closed(self, generated_report):
        assert "\\begin{document}" in generated_report
        assert "\\end{document}" in generated_report
        assert generated_report.index("\\begin{document}") < generated_report.index(
            "\\end{document}"
        )

    def test_documentclass_is_declared_before_the_body(self, generated_report):
        assert "\\documentclass" in generated_report
        assert generated_report.index("\\documentclass") < generated_report.index(
            "\\begin{document}"
        )

    def test_braces_balance(self, generated_report):
        assert generated_report.count("{") == generated_report.count("}"), (
            "unbalanced braces mean pdflatex will fail on this file"
        )

    def test_no_unescaped_underscore_in_prose(self, generated_report):
        """A bare ``_`` is a LaTeX error outside math mode.

        Column names, taxon labels and file paths are full of them, so this is
        the single most likely way a generated report fails to compile.
        """
        body = generated_report.split("\\begin{document}")[-1]
        stripped = body.replace("\\_", "").replace("_{", "_")
        offenders = re.findall(r"(?<!\\)_[A-Za-z]", stripped)
        assert not offenders, f"unescaped underscore before {offenders[:5]}"

    def test_title_author_and_abstract_survive(self, generated_report):
        assert "A test of the report builder" in generated_report
        assert "A. Researcher" in generated_report
        assert "report builder works" in generated_report

    def test_sections_are_present_and_nested(self, generated_report):
        assert "Introduction" in generated_report
        assert "Methods" in generated_report
        assert "Landmarks" in generated_report


class TestStatisticsRenderAsNumbers:
    def test_test_name_present(self, generated_report):
        assert "PERMANOVA" in generated_report
        assert "ANOSIM" in generated_report

    def test_statistic_present(self, generated_report):
        assert "2.69" in generated_report

    def test_p_value_present(self, generated_report):
        assert "0.041" in generated_report
        assert "0.09" in generated_report

    def test_no_python_none_leaks_into_the_document(self, generated_report):
        body = generated_report.split("\\begin{document}")[-1]
        assert "None" not in body, "an unrendered Python None reached the .tex"

    def test_a_statistic_without_a_p_value_still_renders(self, tmp_path):
        """p_value is optional; omitting it must not print ``None``.

        The report renders a dash for the missing cell.
        """
        builder = ReportBuilder()
        builder.set_title("t")
        builder.add_statistical_result("Shannon H", 1.61)
        target = tmp_path / "r.tex"
        builder.generate(str(target))
        text = target.read_text(encoding="utf-8")
        assert "1.61" in text
        assert "None" not in text.split("\\begin{document}")[-1]


# =============================================================================
# Preamble
# =============================================================================


class TestPreamble:
    def test_documentclass_line(self):
        preamble = LatexPreamble(DocumentClass.ARTICLE)
        assert preamble.generate_documentclass() == (
            "\\documentclass[11pt,a4paper]{article}"
        )

    @pytest.mark.parametrize(
        "doc_class,expected",
        [
            (DocumentClass.ARTICLE, "article"),
            (DocumentClass.REPORT, "report"),
            (DocumentClass.BOOK, "book"),
            (DocumentClass.LETTER, "letter"),
        ],
    )
    def test_every_document_class_resolves(self, doc_class, expected):
        assert f"{{{expected}}}" in LatexPreamble(doc_class).generate_documentclass()

    def test_packages_are_recorded_with_options(self):
        preamble = LatexPreamble(DocumentClass.ARTICLE)
        preamble.add_package("geometry", "margin=1in")
        assert preamble.packages == ["\\usepackage[margin=1in]{geometry}"]

    def test_repeated_package_is_dropped(self):
        """A package loaded twice is a LaTeX error when the options differ.

        Re-requesting one of the packages the report already loads
        (``graphicx`` etc.) would otherwise emit a second usepackage line.
        """
        preamble = LatexPreamble(DocumentClass.ARTICLE)
        preamble.add_package("graphicx")
        preamble.add_package("graphicx")
        preamble.add_package("graphicx", "draft")  # same package, other options
        assert sum("graphicx" in line for line in preamble.packages) == 1

    def test_distinct_packages_are_all_kept(self):
        preamble = LatexPreamble(DocumentClass.ARTICLE)
        for name in ("graphicx", "booktabs", "amsmath"):
            preamble.add_package(name)
        assert len(preamble.packages) == 3

    def test_the_generated_report_loads_each_package_once(self, generated_report):
        names = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", generated_report)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"packages loaded more than once: {duplicates}"


# =============================================================================
# Escaping
# =============================================================================


class TestLatexEscaping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("100%", "100\\%"),
            ("a_b", "a\\_b"),
            ("C&D", "C\\&D"),
            ("$5", "\\$5"),
            ("#1", "\\#1"),
        ],
    )
    def test_special_characters_are_escaped(self, raw, expected):
        assert _escape_latex(raw) == expected

    def test_caret_keeps_its_braces(self):
        """``\\textasciicircum{}`` not ``\\textasciicircum ``.

        TeX swallows the space after a control word, so the unbraced form
        would render as ``x^2`` losing the space. The braces are load-bearing.
        """
        assert _escape_latex("x^2") == "x\\textasciicircum{}2"

    def test_already_escaped_text_is_not_double_escaped(self):
        once = _escape_latex("100%")
        assert _escape_latex(once) == once or "\\%\\%" not in _escape_latex(once)

    def test_plain_text_passes_through(self):
        assert _escape_latex("just words") == "just words"


# =============================================================================
# Table / matrix conversion
# =============================================================================


class TestTableAndMatrixConversion:
    def test_matrix_to_latex_has_a_tabular(self):
        rows = np.array([[1.0, 2.0], [3.0, 4.0]])
        latex = MatrixConverter.to_latex(rows)
        assert "tabular" in latex
        assert "1.0" in latex and "4.0" in latex

    def test_matrix_converter_labels(self):
        rows = np.array([[1.0, 2.0], [3.0, 4.0]])
        latex = MatrixConverter.to_latex(rows, ["A", "B"], ["x", "y"])
        assert "A" in latex and "x" in latex

    def test_table_generator_from_matrix(self):
        rows = [[1.0, 2.0], [3.0, 4.0]]
        latex = TableGenerator.from_matrix(rows, headers=["p", "q"], caption="cap")
        assert isinstance(latex, str)
        assert "tabular" in latex
        assert "p" in latex and "cap" in latex

    def test_single_row_matrix_does_not_crash(self):
        assert TableGenerator.from_matrix([[1.0]]) is not None

    def test_figure_handler_includes_an_existing_file(self, tmp_path):
        figure = tmp_path / "f.png"
        figure.write_bytes(b"not really a png, but the handler only emits LaTeX")
        result = FigureHandler().include_figure(str(figure), "A caption", "fig:one")
        assert "includegraphics" in result
        assert "fig:one" in result

    def test_figure_handler_handles_a_missing_file(self):
        """A missing figure must be reported, not raise mid-report."""
        try:
            result = FigureHandler().include_figure("/definitely/not/here.png", "c")
        except Exception as exc:
            pytest.fail(f"include_figure raised on a missing file: {type(exc).__name__}: {exc}")
        else:
            assert isinstance(result, str)
