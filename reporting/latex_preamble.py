"""LaTeX preamble and document class definitions for PaleoAST report generation."""
from enum import Enum, auto
from typing import List, Optional


class DocumentClass(Enum):
    """LaTeX document classes."""
    ARTICLE = auto()
    REPORT = auto()
    BOOK = auto()
    LETTER = auto()
    BEAMER = auto()


class LatexPreamble:
    """LaTeX preamble manager for document generation."""

    def __init__(
        self,
        document_class: DocumentClass = DocumentClass.ARTICLE,
        font_size: int = 11,
        paper_size: str = "a4paper"
    ):
        self._doc_class = document_class
        self._font_size = font_size
        self._paper_size = paper_size
        self._packages: List[str] = []
        self._extra_preamble: List[str] = []

    @property
    def packages(self) -> List[str]:
        return self._packages.copy()

    def add_package(self, name: str, options: Optional[str] = None):
        if options:
            self._packages.append(f"\\usepackage[{options}]{{{name}}}")
        else:
            self._packages.append(f"\\usepackage{{{name}}}")

    def add_preamble_line(self, line: str):
        self._extra_preamble.append(line)

    def generate_documentclass(self) -> str:
        class_map = {
            DocumentClass.ARTICLE: "article",
            DocumentClass.REPORT: "report",
            DocumentClass.BOOK: "book",
            DocumentClass.LETTER: "letter",
            DocumentClass.BEAMER: "beamer",
        }
        cls = class_map.get(self._doc_class, "article")
        return f"\\documentclass[{self._font_size}pt,{self._paper_size}]{{{cls}}}"
